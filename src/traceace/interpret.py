"""Research artifacts — the write-up's raw material.

This module treats interpretability as a first-class output (§11), not a postscript.
It produces, into ``artifacts/figures/`` (PNG + PDF, colourblind-safe) and JSON under
``runs/interpret/``:

* **Cross-fold feature importance with dispersion** — mean ± std across folds, so a claim
  rests on stability rather than one lucky fit.
* **Per-slice performance** — transcript length, turn count, student talk ratio, and
  learning objective. Where does the model work, and where does it fail?
* **Reliability diagrams** pre/post calibration.
* **Key-moments attribution over transcript position** — where in a session does the
  LO-relevant signal sit? Built directly on the ``lo_*`` position features, so it
  describes what the model actually reads.
* **Tutoring-move taxonomy vs outcome** — move distribution related to correctness.
* **Ablation** — each feature block's marginal contribution, so we can say *what*
  mattered rather than merely *that* something worked.

Every figure is sized for direct reuse in the 4-page write-up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import (
    expected_calibration_error,
    load_oof,
    logloss,
    quantile_bins,
    reliability_curve,
    slice_report,
)
from .io import LABEL_COL
from .logging_utils import get_logger
from .paths import figures_dir, models_dir, runs_dir
from .progress import pbar
from .tasks import task
from .viz import PALETTE, save_fig, setup_mpl

log = get_logger("interpret")


def _out_dir() -> Path:
    d = runs_dir() / "interpret"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- figures -----------------------------------------------------------------
def plot_importance(imp: pd.DataFrame, top_n: int = 20, stem: str = "importance") -> list[Path]:
    """Horizontal bar chart of cross-fold gain importance with std error bars."""
    setup_mpl()
    import matplotlib.pyplot as plt

    d = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, max(3.0, 0.28 * len(d))))
    ax.barh(
        d["feature"],
        d["gain_mean"],
        xerr=d.get("gain_std"),
        color=PALETTE[0],
        error_kw={"ecolor": "#444", "elinewidth": 0.8},
    )
    ax.set_xlabel("LightGBM gain (mean ± s.d. across folds)")
    ax.set_title("Which features carry the signal?")
    ax.tick_params(axis="y", labelsize=7)
    return save_fig(fig, figures_dir() / stem)


def plot_reliability(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], stem: str = "reliability"
) -> list[Path]:
    """Reliability diagram; one line per variant (e.g. raw vs calibrated)."""
    setup_mpl()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=1, label="perfect calibration")
    for i, (name, (mp, fp)) in enumerate(curves.items()):
        ax.plot(mp, fp, marker="o", ms=4, color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency correct")
    ax.set_title("Are the probabilities honest?")
    ax.legend(fontsize=8)
    return save_fig(fig, figures_dir() / stem)


def plot_key_moments(
    positions: np.ndarray, weights: np.ndarray | None = None, stem: str = "key_moments"
) -> list[Path]:
    """Distribution of LO-relevant window positions across the session."""
    setup_mpl()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.hist(positions, bins=25, weights=weights, color=PALETTE[2], edgecolor="white")
    ax.set_xlabel("position in session  (0 = start, 1 = end)")
    ax.set_ylabel("number of responses")
    ax.set_title("Where in a lesson does the assessed topic get discussed?")
    return save_fig(fig, figures_dir() / stem)


def plot_slice(
    rows: list[dict[str, Any]], title: str, xlabel: str, stem: str, baseline: float | None = None
) -> list[Path]:
    setup_mpl()
    import matplotlib.pyplot as plt

    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    labels = [r["slice_value"] for r in rows]
    vals = [r["logloss"] for r in rows]
    ax.bar(range(len(rows)), vals, color=PALETTE[1])
    if baseline is not None:
        ax.axhline(baseline, ls="--", color=PALETTE[3], lw=1.2, label="baseline.lo_only")
        ax.legend(fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("log loss (lower is better)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return save_fig(fig, figures_dir() / stem)


# --- tasks -------------------------------------------------------------------
@task(
    "interpret.report",
    requires="cpu",
    max_tier="cpu",
    description="feature importance, slices, reliability, key moments — research artifacts",
)
def report(
    experiment: str = "model.gbdt",
    force: bool = False,
    subsample: int | None = None,
) -> dict[str, Any]:
    from .evaluate import baseline_logloss

    oof = load_oof(experiment)
    y = oof[LABEL_COL].to_numpy(dtype=float)
    p = oof["pred"].to_numpy(dtype=float)
    figures: list[str] = []
    res: dict[str, Any] = {"experiment": experiment}

    base_ll = baseline_logloss()
    res["logloss"] = logloss(y, p)
    res["ece"] = expected_calibration_error(y, p)
    if base_ll is not None:
        res["baseline_lo_only_logloss"] = base_ll
        res["delta_vs_lo_only"] = res["logloss"] - base_ll

    # 1. cross-fold feature importance
    imp_path = models_dir() / experiment.replace(".", "_") / "importance.parquet"
    if imp_path.is_file():
        imp = pd.read_parquet(imp_path)
        figures += [
            str(p_)
            for p_ in plot_importance(imp, stem=f"importance_{experiment.replace('.', '_')}")
        ]
        res["top_features"] = imp.head(20).to_dict("records")
        # per-block aggregate importance
        from .features.assemble import block_of

        imp["block"] = imp["feature"].map(block_of)
        res["importance_by_block"] = (
            imp.groupby("block")["gain_mean"].sum().sort_values(ascending=False).to_dict()
        )

    # 2. reliability (raw, plus calibrated if present)
    curves = {}
    mp_, fp_, _ = reliability_curve(y, p)
    curves["raw"] = (mp_, fp_)
    try:
        cal = load_oof(f"{experiment}.calibrated")
        cmp_, cfp_, _ = reliability_curve(cal[LABEL_COL].to_numpy(), cal["pred"].to_numpy())
        curves["calibrated"] = (cmp_, cfp_)
        res["calibrated_logloss"] = logloss(cal[LABEL_COL].to_numpy(), cal["pred"].to_numpy())
    except FileNotFoundError:
        pass
    figures += [
        str(p_)
        for p_ in plot_reliability(curves, stem=f"reliability_{experiment.replace('.', '_')}")
    ]

    # 3. slices
    slices: dict[str, pd.Series] = {}
    for col, name in [
        ("struct_n_utterances", "turn_count"),
        ("struct_student_talk_ratio", "student_talk_ratio"),
    ]:
        if col in oof.columns:
            slices[name] = quantile_bins(oof[col])
    if "learning_objective_id" in oof.columns:
        top_los = oof["learning_objective_id"].value_counts().head(10).index
        slices["learning_objective"] = (
            oof["learning_objective_id"]
            .where(oof["learning_objective_id"].isin(top_los), other="(other)")
            .astype(str)
        )
    if slices:
        sr = slice_report(oof, slices)
        res["slices"] = sr
        for name, rows in sr.items():
            figures += [
                str(p_)
                for p_ in plot_slice(
                    rows[:12],
                    f"Where does the model work? ({name})",
                    name,
                    f"slice_{name}_{experiment.replace('.', '_')}",
                    baseline=base_ll,
                )
            ]

    # 4. key moments (needs the lo_alignment block)
    try:
        from .features.assemble import load_block

        lo = load_block("lo_alignment", subsample=subsample)
        if "lo_best_pos" in lo.columns:
            figures += [
                str(p_)
                for p_ in plot_key_moments(
                    lo["lo_best_pos"].dropna().to_numpy(), stem="key_moments"
                )
            ]
            res["key_moments"] = {
                "best_pos_mean": float(lo["lo_best_pos"].mean()),
                "best_pos_median": float(lo["lo_best_pos"].median()),
                "best_pos_q25": float(lo["lo_best_pos"].quantile(0.25)),
                "best_pos_q75": float(lo["lo_best_pos"].quantile(0.75)),
            }
    except (FileNotFoundError, KeyError) as exc:
        log.debug("key-moments figure skipped: %s", exc)

    res["figures"] = figures
    path = _out_dir() / f"{experiment.replace('.', '_')}.json"
    path.write_text(json.dumps(res, indent=2, default=str))
    res["output_path"] = str(path)
    log.info("interpret.report[%s]: %d figures", experiment, len(figures))
    return res


@task(
    "interpret.ablation",
    requires="cpu",
    max_tier="cpu",
    description="marginal contribution of each feature block (leave-one-block-out)",
)
def ablation(
    force: bool = False,
    subsample: int | None = None,
    blocks: list[str] | None = None,
    num_boost_round: int = 400,
) -> dict[str, Any]:
    """Train with all blocks, then leave each block out in turn, and report the delta.

    A block whose removal barely changes log loss did not matter — that is a publishable
    negative result and is logged as such.
    """
    from .features.assemble import DEFAULT_BLOCKS
    from .models.gbdt import train as gbdt_train

    blocks = list(blocks or DEFAULT_BLOCKS)
    results: dict[str, float] = {}

    full = gbdt_train(
        blocks=blocks,
        subsample=subsample,
        num_boost_round=num_boost_round,
        experiment="ablation.full",
    )
    results["all_blocks"] = float(full["logloss"])

    for b in pbar(blocks, desc="interpret.ablation", unit="block"):
        remaining = [x for x in blocks if x != b]
        if not remaining:
            continue
        r = gbdt_train(
            blocks=remaining,
            subsample=subsample,
            num_boost_round=num_boost_round,
            experiment=f"ablation.drop_{b}",
        )
        results[f"drop_{b}"] = float(r["logloss"])

    marginal = {
        b: results[f"drop_{b}"] - results["all_blocks"] for b in blocks if f"drop_{b}" in results
    }
    res = {
        "logloss_by_config": results,
        # positive = removing the block HURT, i.e. the block was contributing
        "marginal_contribution": dict(sorted(marginal.items(), key=lambda kv: -kv[1])),
        "blocks": blocks,
    }
    path = _out_dir() / "ablation.json"
    path.write_text(json.dumps(res, indent=2, default=str))
    res["output_path"] = str(path)
    log.info("ablation: %s", res["marginal_contribution"])
    return res
