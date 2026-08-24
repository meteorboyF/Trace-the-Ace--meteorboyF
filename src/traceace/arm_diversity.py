"""Measure how *decorrelated* our ensemble arms actually are — the ensemble's whole premise.

**Why this module exists.** Submission #2 added a second encoder arm on the strength of a
+0.0066 objective-fold CV gain and the leaderboard did not move (0.6091 → 0.6101, inside
paired noise). That is the signature of arms which are not independent: blending two
near-identical predictors cannot reduce variance, whatever CV says about their average.

For two calibrated arms of equal skill whose *logits* correlate at ρ, averaging shrinks the
noise variance by ``(1+ρ)/2``, so the effective signal-to-noise improves by

```
                    gain_factor = sqrt( 2 / (1+ρ) )
```

ρ = 0.95 buys 1.3%. ρ = 0.70 buys 8.5%. ρ = 0.40 buys 19%. **The correlation, not the CV
delta, decides whether an arm is worth a slot in the blend.** This module measures ρ so the
next arm is chosen on evidence rather than on a delta that has now failed to transfer three
times running (docs/ENDGAME.md §2a).

Correlations are computed **inside each objective fold and then averaged**, for the same
reason AUC is (``objective_eval`` §trap 1): folds differ in base rate, so pooling inflates
agreement between any two predictors that both track the fold mean.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import load_oof
from .io import LABEL_COL
from .logging_utils import get_logger
from .objective_eval import (
    MIN_FOLD_ROWS,
    _training_split_mode,
    projected_lb,
    within_fold_auc,
)
from .robust_cv import load_robust_folds
from .tasks import task

log = get_logger("arm_diversity")

EPS = 1e-6


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def gain_factor(rho: float) -> float:
    """Signal-to-noise multiplier from averaging two equal-skill arms correlated at ``rho``.

    The number that should gate every "add another arm" decision. Returns 1.0 for perfectly
    correlated arms — i.e. no dividend at all, which is what the leaderboard reported.
    """
    return float(np.sqrt(2.0 / (1.0 + max(-0.999, min(0.999, rho)))))


def _aligned_frame(experiments: list[str], subsample: int | None) -> pd.DataFrame:
    """Inner-join every arm's OOF on ``response_id`` and attach the objective fold.

    Refuses a partial join: comparing arms scored on different row subsets would make the
    correlations describe an unknown cohort.
    """
    folds = load_robust_folds("objective", subsample=subsample)
    merged: pd.DataFrame | None = None
    sizes: dict[str, int] = {}

    for name in experiments:
        oof = load_oof(name, subsample=subsample)[["response_id", LABEL_COL, "pred"]]
        sizes[name] = len(oof)
        part = oof.rename(columns={"pred": name})
        if merged is None:
            merged = part
        else:
            merged = merged.merge(
                part.drop(columns=[LABEL_COL]),
                on="response_id",
                how="inner",
                validate="one_to_one",
            )
    if merged is None:
        raise RuntimeError("no experiments given")

    smallest = min(sizes.values())
    if len(merged) < 0.99 * smallest:
        raise RuntimeError(
            f"arms overlap on only {len(merged)} rows against a smallest OOF of {smallest}: "
            f"{sizes}. Refusing to correlate predictions on an unknown shared subset."
        )

    out = merged.merge(
        folds[["response_id", "fold"]].rename(columns={"fold": "obj_fold"}),
        on="response_id",
        how="inner",
        validate="one_to_one",
    )
    if len(out) < 0.99 * len(merged):
        raise RuntimeError("objective-fold table does not cover the joined OOF rows")
    return out


def _within_fold_corr(frame: pd.DataFrame, left: str, right: str) -> float:
    """Pearson correlation of logits, computed per objective fold and averaged."""
    values: list[float] = []
    for _, part in frame.groupby("obj_fold"):
        if len(part) < MIN_FOLD_ROWS:
            continue
        a, b = _logit(part[left].to_numpy()), _logit(part[right].to_numpy())
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        values.append(float(np.corrcoef(a, b)[0, 1]))
    if not values:
        return float("nan")
    return float(np.mean(values))


def _blend_auc(frame: pd.DataFrame, arms: list[str]) -> float:
    """Within-fold AUROC of the equal-weight logit-space average of ``arms``."""
    blended = frame.copy()
    blended["pred"] = 1.0 / (
        1.0 + np.exp(-np.mean([_logit(blended[a].to_numpy()) for a in arms], axis=0))
    )
    mean_auc, _, _, _ = within_fold_auc(blended, fold_col="obj_fold")
    return mean_auc


@task(
    "evaluate.arm_diversity",
    requires="cpu",
    max_tier="cpu",
    description="pairwise logit correlation between arms + the dividend blending them buys",
)
def arm_diversity(
    experiments: list[str] | str,
    subsample: int | None = None,
) -> dict[str, Any]:
    """Report each arm's solo skill, every pairwise correlation, and the blend's actual lift.

    The headline is ``verdict``: whether the measured all-arm blend beats the best solo arm by
    more than the dividend its correlations can support. A blend that does not clear its own
    correlation ceiling is fitting noise, and the leaderboard will say so.
    """
    if isinstance(experiments, str):
        experiments = [e.strip() for e in experiments.split(",") if e.strip()]
    if len(experiments) < 2:
        raise RuntimeError("need at least two experiments to measure diversity")

    frame = _aligned_frame(experiments, subsample)

    arms: dict[str, Any] = {}
    for name in experiments:
        scored = frame.rename(columns={name: "pred"})
        mean_auc, sd_auc, per_fold, _ = within_fold_auc(scored, fold_col="obj_fold")
        split_mode = _training_split_mode(name, subsample, None)
        arms[name] = {
            "within_objective_fold_auc": round(mean_auc, 5),
            "sd_across_folds": round(sd_auc, 5),
            "per_fold_auc": {k: round(v, 5) for k, v in sorted(per_fold.items())},
            "evidence": "honest" if split_mode == "objective" else "optimistic",
        }

    pairs: list[dict[str, Any]] = []
    for left, right in combinations(experiments, 2):
        rho = _within_fold_corr(frame, left, right)
        pair_auc = _blend_auc(frame, [left, right])
        solo_best = max(
            arms[left]["within_objective_fold_auc"], arms[right]["within_objective_fold_auc"]
        )
        pairs.append(
            {
                "pair": f"{left} + {right}",
                "logit_corr": round(rho, 4),
                "snr_gain_factor": round(gain_factor(rho), 4),
                "pair_blend_auc": round(pair_auc, 5),
                "lift_over_best_solo": round(pair_auc - solo_best, 5),
            }
        )
    pairs.sort(key=lambda row: row["logit_corr"])

    all_auc = _blend_auc(frame, experiments)
    best_solo_name = max(experiments, key=lambda n: arms[n]["within_objective_fold_auc"])
    best_solo = arms[best_solo_name]["within_objective_fold_auc"]
    lift = all_auc - best_solo

    tightest = max((p["logit_corr"] for p in pairs if np.isfinite(p["logit_corr"])), default=1.0)
    verdict = (
        "arms are near-duplicates — blending cannot pay; find a genuinely different arm"
        if tightest > 0.90
        else "arms carry distinct signal — the blend is worth its slot"
        if lift > 0.002
        else "arms differ but the blend does not beat the best solo; ship the solo arm"
    )

    result = {
        "n_rows": int(len(frame)),
        "arms": arms,
        "pairs": pairs,
        "best_solo": {"experiment": best_solo_name, "auc": round(best_solo, 5)},
        "all_arm_blend_auc": round(all_auc, 5),
        "lift_over_best_solo": round(lift, 5),
        "projected_lb_best_solo": round(projected_lb(best_solo), 5),
        "projected_lb_all_arms": round(projected_lb(all_auc), 5),
        "tightest_pair_corr": round(tightest, 4),
        "verdict": verdict,
    }
    log.info(
        "arm_diversity: best solo %s=%.5f · all-arm blend %.5f (lift %+.5f) · tightest rho %.3f",
        best_solo_name,
        best_solo,
        all_auc,
        lift,
        tightest,
    )
    return result
