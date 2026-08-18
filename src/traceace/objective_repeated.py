"""Paired A/B comparison of two configurations across repeated objective-fold assignments.

**Why a whole module for this.** A single objective-fold assignment is far noisier than a
session-fold one. Each fold holds only ~80 of 398 objectives and objective difficulty is
lumpy, so the paired AUROC delta between two configurations has an SD around **0.037** across
folds — larger than every effect we are trying to measure. The first honest measurement of
``lo_text_difficulty`` came out at **+0.0105 ± 0.0374, positive in 3/5 folds**: a mean that
looks decisive attached to an interval that comfortably includes zero.

This project has already learned the lesson once on session folds, where the noise floor is
~5e-4 and single-seed readings below ~1e-3 were driving block decisions (see
``features/assemble.py``). The same discipline has to extend to objective folds, and the
constant is two orders of magnitude larger.

**What this does.** Rebuilds the objective fold table under R different assignments of the
same objectives, retrains both configurations on each, and reports the delta **paired within
assignment**. Pairing is what makes this affordable: the assignment-to-assignment variance is
enormous and identical for both arms, so it cancels.

The reported statistic is the mean paired delta, its standard error, and the sign count. A
configuration is promotable when the mean exceeds the standard error by a comfortable margin
*and* the sign is consistent — not when the mean alone looks good.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import load_oof
from .logging_utils import get_logger
from .objective_eval import projected_lb, within_fold_auc
from .paths import runs_dir
from .progress import pbar
from .robust_cv import load_robust_folds
from .tasks import task

log = get_logger("objective_repeated")

DEFAULT_FOLD_SEEDS = [11, 23, 37, 41, 59]


def _fold_aucs(experiment: str, fold_seed: int, subsample: int | None) -> dict[int, float]:
    """Within-fold AUC for one experiment under one fold assignment."""
    oof = load_oof(experiment, subsample=subsample)
    folds = load_robust_folds("objective", subsample=subsample, fold_seed=fold_seed)
    merged = oof.drop(columns=["fold"], errors="ignore").merge(
        folds[["response_id", "fold"]], on="response_id", how="inner", validate="one_to_one"
    )
    _, _, per_fold, _ = within_fold_auc(merged)
    return per_fold


@task(
    "evaluate.objective_repeated",
    requires="cpu",
    max_tier="cpu",
    description="paired A/B across repeated objective-fold assignments, with a real error bar",
)
def compare(
    force: bool = False,
    subsample: int | None = None,
    baseline_kwargs: dict[str, Any] | None = None,
    candidate_kwargs: dict[str, Any] | None = None,
    label: str = "lo_text_difficulty",
    fold_seeds: list[int] | None = None,
    task_name: str = "model.gbdt",
) -> dict[str, Any]:
    """Compare two configurations of ``task_name`` across repeated objective-fold assignments.

    Defaults compare transcript-only against transcript + ``lo_text_difficulty``, which is the
    open question this module was written to settle.

    Both arms are always retrained on *the same* fold assignment before being compared, so a
    stale OOF from a different assignment can never enter the comparison.
    """
    from .tasks import run as run_task

    seeds = list(fold_seeds or DEFAULT_FOLD_SEEDS)
    base_kw = dict(
        baseline_kwargs
        or {
            "split_mode": "objective",
            "include_lo_prior": False,
            "include_lo_text_difficulty": False,
        }
    )
    cand_kw = dict(
        candidate_kwargs
        or {
            "split_mode": "objective",
            "include_lo_prior": False,
            "include_lo_text_difficulty": True,
        }
    )

    records: list[dict[str, Any]] = []
    for fold_seed in pbar(seeds, desc="objective_repeated: fold assignments", unit="seed"):
        run_task("cv.robust_build", kind="objective", fold_seed=fold_seed, force=force)

        arm_aucs: dict[str, dict[int, float]] = {}
        for arm, kwargs in (("baseline", base_kw), ("candidate", cand_kw)):
            experiment = f"rep.{label}.{arm}.s{fold_seed}"
            run_task(
                task_name,
                experiment=experiment,
                fold_seed=fold_seed,
                subsample=subsample,
                force=True,  # a stale OOF here silently compares different fold assignments
                **kwargs,
            )
            arm_aucs[arm] = _fold_aucs(experiment, fold_seed, subsample)

        shared = sorted(set(arm_aucs["baseline"]) & set(arm_aucs["candidate"]))
        if not shared:
            raise RuntimeError(f"fold seed {fold_seed} produced no comparable folds")
        for fold in shared:
            records.append(
                {
                    "fold_seed": fold_seed,
                    "fold": fold,
                    "baseline_auc": arm_aucs["baseline"][fold],
                    "candidate_auc": arm_aucs["candidate"][fold],
                    "delta": arm_aucs["candidate"][fold] - arm_aucs["baseline"][fold],
                }
            )

    frame = pd.DataFrame(records)

    # Aggregate to one number per fold ASSIGNMENT first. Folds inside an assignment are not
    # independent (they partition the same objectives), so treating all 25 as independent
    # would understate the error bar by roughly sqrt(5).
    per_seed = frame.groupby("fold_seed")["delta"].mean()
    deltas = per_seed.to_numpy(dtype=float)
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1)) if len(deltas) > 1 else float("nan")
    stderr = sd / np.sqrt(len(deltas)) if len(deltas) > 1 else float("nan")
    positive = int((deltas > 0).sum())

    baseline_auc = float(frame.groupby("fold_seed")["baseline_auc"].mean().mean())
    candidate_auc = float(frame.groupby("fold_seed")["candidate_auc"].mean().mean())

    # Promotion needs the effect to clear its own uncertainty AND to be directionally
    # consistent. Either alone has already misled this project once.
    promotable = bool(
        np.isfinite(stderr) and mean > 2 * stderr and positive >= max(4, len(deltas) - 1)
    )

    result: dict[str, Any] = {
        "label": label,
        "task": task_name,
        "fold_seeds": seeds,
        "n_assignments": len(deltas),
        "baseline_auc": round(baseline_auc, 5),
        "candidate_auc": round(candidate_auc, 5),
        "paired_delta_auc": round(mean, 5),
        "paired_delta_sd": round(sd, 5) if np.isfinite(sd) else None,
        "paired_delta_stderr": round(stderr, 5) if np.isfinite(stderr) else None,
        "positive_assignments": f"{positive}/{len(deltas)}",
        "per_assignment_delta": {int(k): round(float(v), 5) for k, v in per_seed.items()},
        "projected_lb_baseline": round(projected_lb(baseline_auc), 5),
        "projected_lb_candidate": round(projected_lb(candidate_auc), 5),
        "promotable": promotable,
        "verdict": (
            f"PROMOTE — {mean:+.5f} AUROC, {positive}/{len(deltas)} assignments positive, "
            f"{mean / stderr:.1f}x its standard error"
            if promotable
            else f"HOLD — {mean:+.5f} ± {stderr:.5f} AUROC over {len(deltas)} assignments, "
            f"{positive}/{len(deltas)} positive. Not distinguishable from noise."
        ),
    }

    out_dir = runs_dir() / "objective_repeated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}.json").write_text(
        json.dumps({**result, "per_fold": records}, indent=2, default=str)
    )
    result["output_path"] = str(out_dir / f"{label}.json")
    log.info("%s", result["verdict"])
    return result


@task(
    "evaluate.objective_noise_floor",
    requires="cpu",
    max_tier="cpu",
    description="measure the objective-fold noise floor so no decision is made below it",
)
def noise_floor(
    force: bool = False,
    subsample: int | None = None,
    fold_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Retrain the *same* configuration twice per assignment and measure the spread.

    The delta between two identical configurations is zero by construction, so whatever
    spread this reports is pure noise — the floor beneath which no A/B result means anything.
    Only the model seed differs between the arms.
    """
    return compare(
        force=force,
        subsample=subsample,
        label="noise_floor",
        fold_seeds=fold_seeds,
        baseline_kwargs={
            "split_mode": "objective",
            "include_lo_prior": False,
            "include_lo_text_difficulty": False,
            "seed": 1234,
        },
        candidate_kwargs={
            "split_mode": "objective",
            "include_lo_prior": False,
            "include_lo_text_difficulty": False,
            "seed": 4321,
        },
    )
