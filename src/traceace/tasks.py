"""Task registry, tier-guard, and the ``run()`` entrypoint.

Every unit of work in traceace is a *task*: a function decorated with :func:`task`
declaring the runtime tier it needs and the highest tier it should be allowed to
run on. :func:`run` is the single entrypoint (the notebook's ``run()`` calls it).
It enforces the compute discipline the brief demands (§4):

* **Tier guard.** Below the required tier -> refuse and name the runtime to switch
  to. *Above* ``max_tier`` -> refuse by default (an idle/oversized GPU is the #1
  waste vector), unless ``allow_waste=True``.
* **Header + summary.** Prints task/tier/subsample/cache-status/ETA before, and
  wall/units/metric/output after — so the operator can judge a running GPU cell.
* **Manifest.** Writes ``runs/<task>/<ts>.json`` with git SHA, config seed, kwargs,
  wall time, tier, units and returned metrics — the audit trail EXPERIMENTS.md is
  built from.
* **Budget ledger.** Appends to ``runs/budget.jsonl`` via :mod:`traceace.budget`.
* **shutdown_after.** Optionally stops billing when an overnight job finishes.

Tasks are plain functions returning a ``dict`` of metrics/paths (or ``None``).
Control kwargs (``force``, ``subsample``, ``allow_waste``, ``shutdown_after``) are
interpreted by the runner; everything else is forwarded to the task.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from . import budget
from .config import get_config
from .logging_utils import get_logger, setup_logging
from .paths import ensure_dirs, logs_dir, runs_dir
from .runtime import Tier, detect_accelerator, tier_at_least, tier_at_most

log = get_logger("tasks")

TaskFn = Callable[..., Any]

# Modules that define tasks; imported once so their @task decorators register.
_TASK_MODULES = [
    "traceace.data",
    "traceace.eda",
    "traceace.cv",
    "traceace.robust_cv",
    "traceace.models.baseline",
    "traceace.models.gbdt",
    "traceace.models.sparse_text",
    "traceace.models.bge_attention",
    "traceace.models.hierarchical_transformer",
    "traceace.models.move_classifier",
    "traceace.models.transcript_encoder",
    "traceace.features.structural",
    "traceace.features.linguistic",
    "traceace.features.temporal",
    "traceace.features.lo_alignment",
    "traceace.features.feedback",
    "traceace.features.trajectory",
    "traceace.features.embeddings",
    "traceace.features.window_embeddings",
    "traceace.features.content",
    "traceace.annotate",
    "traceace.calibration",
    "traceace.ensemble",
    "traceace.evaluate",
    "traceace.objective_eval",
    "traceace.objective_repeated",
    "traceace.interpret",
    "traceace.experiments",
    "traceace.unseen_lo",
    "traceace.packaging.build_submission",
    "traceace.packaging.verify",
    "traceace.selftest",
    "traceace.maintenance",
]


@dataclass
class TaskSpec:
    name: str
    fn: TaskFn
    requires: Tier
    max_tier: Tier
    description: str


_REGISTRY: dict[str, TaskSpec] = {}
_LOADED = False


def task(
    name: str,
    requires: Tier = "cpu",
    max_tier: Tier = "h100",
    description: str = "",
) -> Callable[[TaskFn], TaskFn]:
    """Register ``fn`` as a task.

    Parameters
    ----------
    name:
        Dotted task name, e.g. ``features.structural``.
    requires:
        Minimum tier needed. ``run`` refuses below this.
    max_tier:
        Highest tier this task should run on. ``run`` refuses *above* this unless
        ``allow_waste=True``. A CPU task declares ``max_tier="cpu"`` so it can never
        silently burn units on an attached A100.
    """

    def deco(fn: TaskFn) -> TaskFn:
        if name in _REGISTRY:
            raise ValueError(f"duplicate task registration: {name}")
        _REGISTRY[name] = TaskSpec(
            name=name,
            fn=fn,
            requires=requires,
            max_tier=max_tier,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
        )
        return fn

    return deco


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    for mod in _TASK_MODULES:
        try:
            import_module(mod)
        except ModuleNotFoundError as exc:
            # A task module may legitimately not exist yet during the build; log and
            # continue so the registry still works for what does exist.
            log.debug("task module not importable yet: %s (%s)", mod, exc)
    _LOADED = True


def list_tasks() -> list[TaskSpec]:
    _ensure_loaded()
    return [spec for _, spec in sorted(_REGISTRY.items())]


def get_task(name: str) -> TaskSpec:
    _ensure_loaded()
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"unknown task {name!r}. Known: {known}")
    return _REGISTRY[name]


# --- git / manifest helpers --------------------------------------------------
def git_sha(short: bool = True) -> str:
    cfg = get_config()
    args = ["git", "-C", str(cfg.repo_dir), "rev-parse"]
    args += ["--short", "HEAD"] if short else ["HEAD"]
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _prior_wall_seconds(name: str) -> float | None:
    """Median wall time of prior runs of ``name`` (for ETA), or None."""
    d = runs_dir() / name.replace(".", "_")
    if not d.is_dir():
        return None
    walls: list[float] = []
    for p in sorted(d.glob("*.json")):
        try:
            walls.append(float(json.loads(p.read_text()).get("wall_seconds", 0.0)))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            continue
    walls = [w for w in walls if w > 0]
    if not walls:
        return None
    walls.sort()
    return walls[len(walls) // 2]


def _write_manifest(name: str, payload: dict[str, Any]) -> Path:
    d = runs_dir() / name.replace(".", "_")
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = d / f"{ts}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown (no prior run)"
    if seconds < 90:
        return f"~{seconds:.0f}s"
    return f"~{seconds / 60:.1f} min"


def _maybe_shutdown() -> None:
    """Sync results to Drive, then stop the Colab runtime so an idle GPU stops billing.

    The sync is NOT optional. ``unassign()`` destroys the runtime's local SSD, which is
    where every artifact lives — without the sync, an overnight ``shutdown_after=True``
    training run would delete its own results the moment it finished. If the sync fails,
    the runtime is deliberately left RUNNING (billing and all): a few units of idle GPU
    are recoverable, vanished results are not.
    """
    try:
        from .maintenance import sync_artifacts  # noqa: PLC0415

        log.warning("shutdown_after=True -> syncing artifacts to Drive before shutdown.")
        sync_artifacts()
    except Exception:
        log.exception(
            "ARTIFACT SYNC FAILED — leaving the runtime attached so results are not lost. "
            "Run maintenance.sync_artifacts manually, then disconnect."
        )
        return
    try:
        log.warning("sync complete -> terminating runtime to stop billing.")
        from google.colab import runtime as colab_runtime  # noqa: PLC0415

        colab_runtime.unassign()
    except Exception as exc:  # not on Colab, or API changed
        log.info("shutdown requested but no Colab runtime to unassign (%s)", exc)


def run(
    name: str,
    *,
    force: bool = False,
    subsample: int | None = None,
    allow_waste: bool = False,
    shutdown_after: bool = False,
    **kwargs: Any,
) -> Any:
    """Run task ``name`` with the tier-guard, manifest and budget ledger.

    Extra ``kwargs`` are forwarded to the task function. ``force`` and ``subsample``
    are also forwarded (tasks accept them) *and* recorded in the manifest.
    """
    _ensure_loaded()
    cfg = get_config()
    setup_logging(logs_dir())
    ensure_dirs()

    spec = get_task(name)
    acc = detect_accelerator()
    tier = acc.tier

    # --- tier guard ---------------------------------------------------------
    if not tier_at_least(tier, spec.requires):
        raise RuntimeError(
            f"[tier-guard] '{name}' needs at least '{spec.requires}' but you are on "
            f"'{tier}' ({acc.name}). Switch to a {spec.requires.upper()} runtime."
        )
    if not tier_at_most(tier, spec.max_tier) and not allow_waste:
        rate = cfg.unit_rates.get(tier, 0.0)
        raise RuntimeError(
            f"[tier-guard] '{name}' is a {spec.max_tier.upper()} task but you are on "
            f"'{tier}' ({acc.name}, ~{rate:g} units/hr). That wastes units. "
            f"Switch to a {spec.max_tier.upper()} runtime, or pass allow_waste=True."
        )

    # --- header -------------------------------------------------------------
    eta = _fmt_eta(_prior_wall_seconds(name))
    rate = cfg.unit_rates.get(tier, 0.0)
    print("=" * 64)
    print(f"▶ TASK   {name}")
    print(f"  tier   {tier} ({acc.name}) · ~{rate:g} units/hr")
    print(f"  args   subsample={subsample} force={force} {kwargs or ''}".rstrip())
    print(f"  eta    {eta}")
    print("=" * 64, flush=True)

    # --- run + time ---------------------------------------------------------
    fwd = dict(kwargs)
    fwd["force"] = force
    fwd["subsample"] = subsample
    fwd = _filter_kwargs(spec.fn, fwd)

    t0 = time.monotonic()
    error: str | None = None
    result: Any = None
    try:
        result = spec.fn(**fwd)
    except Exception as exc:  # record failure in manifest, then re-raise
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        wall = time.monotonic() - t0
        entry = budget.record(name, tier, wall, git_sha=git_sha(), subsample=subsample)
        metrics = result if isinstance(result, dict) else {}
        manifest = {
            "task": name,
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "tier": tier,
            "accelerator": acc.name,
            "wall_seconds": round(wall, 3),
            "units": entry.units,
            "cumulative_units": entry.cumulative_units,
            "seed": cfg.seed,
            "subsample": subsample,
            "kwargs": {k: v for k, v in kwargs.items() if _jsonable(v)},
            "metrics": {k: v for k, v in metrics.items() if _jsonable(v)},
            "error": error,
        }
        mpath = _write_manifest(name, manifest)

        # --- summary --------------------------------------------------------
        headline = _pick_headline(metrics)
        print("-" * 64)
        status = "FAILED" if error else "done"
        print(
            f"◀ {name} {status} · {wall:.1f}s · {entry.units:.3f} units "
            f"(cum {entry.cumulative_units:.2f})"
        )
        if headline:
            print(f"  {headline}")
        outp = metrics.get("output_path") or metrics.get("output")
        if outp:
            print(f"  output {outp}")
        print(f"  manifest {mpath}")
        print("-" * 64, flush=True)

    if shutdown_after:
        _maybe_shutdown()
    return result


def _num(metrics: dict[str, Any], key: str) -> float | None:
    """Read a metric as a float, tolerating richer shapes.

    Repeated-seed tasks report a *distribution* rather than a scalar, so a metric may be a
    dict like ``{"mean": ..., "sd": ...}``. Formatting that with ``:.5f`` raised a
    TypeError and killed the run's summary **after the task had already succeeded** —
    losing the console report of a multi-minute job. Never let presentation break a result.
    """
    v = metrics.get(key)
    if isinstance(v, int | float) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, dict):
        m = v.get("mean")
        if isinstance(m, int | float):
            return float(m)
    return None


def _pick_headline(metrics: dict[str, Any]) -> str:
    """Prefer log loss + delta vs baseline in the summary line."""
    parts = []
    for key in ("logloss", "log_loss", "cv_logloss"):
        v = _num(metrics, key)
        if v is not None:
            parts.append(f"logloss={v:.5f}")
            break
    for key in ("delta_vs_lo_only", "delta_logloss"):
        v = _num(metrics, key)
        if v is not None:
            sd = metrics.get(key, {})
            sd_s = (
                f" ± {sd['sd']:.5f}"
                if isinstance(sd, dict) and isinstance(sd.get("sd"), int | float)
                else ""
            )
            parts.append(f"Δ_vs_lo_only={v:+.5f}{sd_s}")
            break
    v = _num(metrics, "auc")
    if v is not None:
        parts.append(f"auc={v:.4f}")
    return " · ".join(parts)


def _jsonable(v: Any) -> bool:
    try:
        json.dumps(v, default=str)
        return True
    except (TypeError, ValueError):
        return False


def _filter_kwargs(fn: TaskFn, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs the task doesn't accept (unless it takes **kwargs)."""
    import inspect

    sig = inspect.signature(fn)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    accepted = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}
