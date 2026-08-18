"""Score any OOF prediction set the way the leaderboard scores it.

**Why this module exists.** Session-grouped CV ranks our models by a quantity the
leaderboard does not measure. The per-objective difficulty lookup supplies ~80% of our
session-CV AUROC and scores 0.500 on unseen objectives, which is the regime the test set is
in (docs/ENDGAME.md §2). Every model we shelved — ``model.bge_attention``, the hierarchical
ModernBERT pilot — was shelved against an objective-difficulty baseline that does not exist
on the leaderboard.

So this module reports two numbers for any experiment:

* **within-objective-fold AUROC** — AUC computed separately inside each held-out-objective
  fold and averaged. This is the discrimination that survives when the objective is new.
* **projected leaderboard log loss** — via ``LL ≈ C − k·(AUC−0.5)²`` from ``conf/base.yaml``.
  Log loss on this task is a monotone function of AUROC with a constant C shared across the
  whole public table, so AUROC converts directly into a leaderboard position.

⚠️ **Two traps this module is built to avoid.**

1. **Never pool across objective folds.** Objective folds have different base rates, so
   pooling predictions and scoring AUC once gives a *constant* predictor AUC 0.457 rather
   than 0.500. :func:`within_fold_auc` scores inside each fold; :func:`pooled_auc_artifact`
   quantifies the illusion so it stays visible rather than being rediscovered.
2. **Rescoring is not retraining.** An OOF frame produced under session folds saw its
   validation objectives during training. Re-partitioning those predictions by objective is
   a cheap *upper bound* — a triage screen, not evidence. Runs whose training manifest does
   not say ``split_mode="objective"`` are reported as ``optimistic``, and promotion requires
   an honest retrain (``model.gbdt(split_mode="objective")`` or
   ``model.transcript_encoder(split_mode="objective")``).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import get_config
from .evaluate import experiment_dir, experiment_name, load_oof, logloss
from .io import LABEL_COL
from .logging_utils import get_logger
from .paths import runs_dir
from .robust_cv import load_robust_folds
from .tasks import task

log = get_logger("objective_eval")

# Minimum rows and minimum members of each class for a fold's AUC to mean anything.
MIN_FOLD_ROWS = 200


def lb_constants() -> tuple[float, float]:
    """``(k, C)`` for the leaderboard projection. Never hardcoded — see conf/base.yaml."""
    cfg = get_config()
    k = cfg.get("leaderboard", "k", default=None)
    c = cfg.get("leaderboard", "c", default=None)
    if k is None or c is None:
        raise RuntimeError(
            "conf/base.yaml is missing leaderboard.k / leaderboard.c; refusing to guess "
            "the projection constants"
        )
    return float(k), float(c)


def projected_lb(auc_value: float) -> float:
    """Projected leaderboard log loss for a calibrated model at this AUROC.

    An estimate derived from leaderboard feedback (docs/ENDGAME.md §1), used to put
    candidates on one comparable scale. Not held-out evidence, and never quoted as such.
    """
    k, c = lb_constants()
    return float(c - k * (auc_value - 0.5) ** 2)


def auroc_needed_for(target_logloss: float) -> float:
    """Inverse of :func:`projected_lb` — the AUROC a target leaderboard score requires."""
    k, c = lb_constants()
    if target_logloss >= c:
        return 0.5
    return float(np.sqrt((c - target_logloss) / k) + 0.5)


def within_fold_auc(
    frame: pd.DataFrame, fold_col: str = "fold", pred_col: str = "pred"
) -> tuple[float, float, dict[int, float], list[int]]:
    """AUC inside each fold, then averaged. Returns ``(mean, sd, per_fold, skipped)``.

    Folds too small or single-class to admit an AUC are skipped rather than silently
    contributing 0.5, and are reported by name so a degenerate split cannot pass unnoticed.
    """
    from sklearn.metrics import roc_auc_score

    per_fold: dict[int, float] = {}
    skipped: list[int] = []
    for fold, part in frame.groupby(fold_col):
        y = part[LABEL_COL].to_numpy(dtype=float)
        if len(part) < MIN_FOLD_ROWS or len(np.unique(y)) < 2:
            skipped.append(int(fold))
            continue
        per_fold[int(fold)] = float(roc_auc_score(y, part[pred_col].to_numpy(dtype=float)))
    if not per_fold:
        raise RuntimeError("no fold was large enough or class-balanced enough to score")
    values = np.array(list(per_fold.values()), dtype=float)
    return float(values.mean()), float(values.std()), per_fold, skipped


def pooled_auc_artifact(frame: pd.DataFrame, fold_col: str = "fold") -> float:
    """AUC of a per-fold *constant* predictor, pooled across folds.

    Should be 0.500 and is not: objective folds differ in base rate, so pooling manufactures
    ranking signal out of nothing. Reported on every run so the trap stays visible.
    """
    from sklearn.metrics import roc_auc_score

    constants = frame.groupby(fold_col)[LABEL_COL].transform("mean").to_numpy(dtype=float)
    y = frame[LABEL_COL].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, constants))


def _training_split_mode(experiment: str, subsample: int | None, cv_seed: int | None) -> str:
    """Read how an experiment was actually trained, so honesty is not left to the operator."""
    manifest = experiment_dir(experiment, subsample, cv_seed) / "training_manifest.json"
    if not manifest.is_file():
        return "unknown"
    try:
        return str(json.loads(manifest.read_text()).get("split_mode", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "unknown"


def score_experiment(
    experiment: str,
    subsample: int | None = None,
    cv_seed: int | None = None,
) -> dict[str, Any]:
    """Score one OOF experiment under objective-purged folds."""
    oof = load_oof(experiment, subsample=subsample, cv_seed=cv_seed)
    folds = load_robust_folds("objective", subsample=subsample)

    merged = oof.merge(
        folds[["response_id", "fold"]].rename(columns={"fold": "obj_fold"}),
        on="response_id",
        how="inner",
        validate="one_to_one",
    )
    coverage = len(merged) / max(len(oof), 1)
    if coverage < 0.99:
        raise RuntimeError(
            f"{experiment!r}: only {len(merged)}/{len(oof)} OOF rows carry an objective-fold "
            "assignment. The OOF and the fold table describe different cohorts — refusing to "
            "report a score computed on an unknown subset."
        )

    mean_auc, sd_auc, per_fold, skipped = within_fold_auc(merged, fold_col="obj_fold")
    split_mode = _training_split_mode(experiment, subsample, cv_seed)
    honest = split_mode == "objective"

    result: dict[str, Any] = {
        "experiment": experiment_name(experiment, subsample, cv_seed),
        "n": int(len(merged)),
        "within_objective_fold_auc": round(mean_auc, 5),
        "within_objective_fold_auc_sd": round(sd_auc, 5),
        "per_fold_auc": {k: round(v, 5) for k, v in sorted(per_fold.items())},
        "skipped_folds": skipped,
        "session_cv_logloss": round(
            logloss(merged[LABEL_COL].to_numpy(), merged["pred"].to_numpy()), 5
        ),
        "projected_lb_logloss": round(projected_lb(mean_auc), 5),
        "trained_split_mode": split_mode,
        "evidence": "honest" if honest else "optimistic",
        "constant_predictor_pooled_auc": round(pooled_auc_artifact(merged, "obj_fold"), 4),
    }
    if not honest:
        result["caveat"] = (
            f"trained with split_mode={split_mode!r}, so validation objectives were seen "
            "during training. This is a triage upper bound, not promotion evidence — "
            "retrain with split_mode='objective' before acting on it."
        )
    return result


@task(
    "evaluate.by_objective_fold",
    requires="cpu",
    max_tier="cpu",
    description="score OOF sets on unseen objectives (within-fold AUROC) + projected leaderboard",
)
def by_objective_fold(
    force: bool = False,
    subsample: int | None = None,
    experiments: list[str] | None = None,
    cv_seed: int | None = None,
) -> dict[str, Any]:
    """Rank every available experiment by the only metric that tracks the leaderboard.

    With ``experiments=None`` this sweeps every OOF set on disk, which is the intended first
    action of a session: it re-reads the whole experiment history through the correct lens.
    """
    from .evaluate import list_oof

    if experiments is None:
        names = [
            n
            for n in list_oof()
            if "__sub" not in n and "__cv" not in n and not n.startswith("abl.")
        ]
    else:
        names = list(experiments)

    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    for name in names:
        try:
            rows.append(score_experiment(name, subsample=subsample, cv_seed=cv_seed))
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            failures[name] = str(exc)
            log.warning("by_objective_fold: skipping %s (%s)", name, exc)

    if not rows:
        raise RuntimeError(
            "no experiment could be scored. Run cv.robust_build(kind='objective') and at "
            f"least one model task first. Failures: {failures}"
        )

    rows.sort(key=lambda r: -float(r["within_objective_fold_auc"]))
    table = pd.DataFrame(rows)

    k, c = lb_constants()
    targets = {
        "top_15_gate_0.6043": round(auroc_needed_for(0.6043), 4),
        "top_10_0.6017": round(auroc_needed_for(0.6017), 4),
        "rank_1_0.5961": round(auroc_needed_for(0.5961), 4),
    }

    out_dir = runs_dir() / "objective_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": rows,
        "failures": failures,
        "projection": {"k": k, "c": c},
        "auroc_targets": targets,
        "note": (
            "within_objective_fold_auc is the metric that decides. 'optimistic' rows were "
            "trained on session folds and saw their validation objectives."
        ),
    }
    (out_dir / "by_objective_fold.json").write_text(json.dumps(payload, indent=2, default=str))

    log.info("AUROC required — top-15 %.4f · top-10 %.4f · #1 %.4f", *targets.values())
    for row in rows:
        log.info(
            "%-42s obj-AUC %.4f ± %.4f  -> proj LB %.4f  [%s]",
            row["experiment"],
            row["within_objective_fold_auc"],
            row["within_objective_fold_auc_sd"],
            row["projected_lb_logloss"],
            row["evidence"],
        )

    # The headline must be the best *honestly* trained model. An optimistic row can top the
    # table at AUC 0.72 while being worth 0.60 on the leaderboard — that is the exact
    # confusion this module exists to end, so it must not be reproduced in its own summary.
    honest_rows = [r for r in rows if r["evidence"] == "honest"]
    summary: dict[str, Any] = {
        "n_scored": len(rows),
        "n_honest": len(honest_rows),
        "n_failed": len(failures),
        "auroc_targets": targets,
        "best_optimistic_experiment": rows[0]["experiment"],
        "best_optimistic_auc": rows[0]["within_objective_fold_auc"],
    }
    if honest_rows:
        best = honest_rows[0]
        summary.update(
            {
                "best_experiment": best["experiment"],
                "best_within_objective_fold_auc": best["within_objective_fold_auc"],
                "best_projected_lb": best["projected_lb_logloss"],
                "clears_top_15_gate": bool(
                    best["within_objective_fold_auc"] >= targets["top_15_gate_0.6043"]
                ),
            }
        )
    else:
        summary["best_experiment"] = None
        summary["warning"] = (
            "no experiment was trained objective-disjoint, so nothing here is promotion "
            "evidence. Retrain with split_mode='objective'."
        )
        log.warning("%s", summary["warning"])
    summary["table"] = table[
        [
            "experiment",
            "within_objective_fold_auc",
            "within_objective_fold_auc_sd",
            "projected_lb_logloss",
            "evidence",
        ]
    ].to_dict("records")
    summary["output_path"] = str(out_dir / "by_objective_fold.json")
    return summary
