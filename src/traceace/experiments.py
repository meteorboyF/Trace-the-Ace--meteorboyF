"""Repeated-seed experiment drivers: headline scores and paired block ablation.

Every number produced here carries an error bar. See :mod:`traceace.repeated` for why
that is not optional at this sample size.
"""

from __future__ import annotations

import contextlib
import io
import json
from typing import Any

from .logging_utils import get_logger
from .paths import runs_dir
from .progress import pbar
from .repeated import DEFAULT_SEEDS, RepeatedResult, format_table, paired_delta, summarize
from .tasks import task

log = get_logger("experiments")


def _quiet(fn, **kw):
    """Run a task function without its console chatter (we drive our own bars)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(**kw)


def _ensure_folds_and_baseline(cv_seed: int, subsample: int | None) -> float:
    """Build folds + the lo_only baseline for one seed; return the baseline log loss."""
    from .cv import build as cv_build
    from .models.baseline import lo_only

    _quiet(cv_build, cv_seed=cv_seed, subsample=subsample, force=True)
    b = _quiet(lo_only, cv_seed=cv_seed, subsample=subsample)
    return float(b["logloss"])


@task(
    "evaluate.repeated",
    requires="cpu",
    max_tier="cpu",
    description="headline score across repeated fold assignments, with mean ± SD",
)
def repeated_score(
    force: bool = False,
    subsample: int | None = None,
    blocks: list[str] | None = None,
    seeds: tuple[int, ...] | list[int] = DEFAULT_SEEDS,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
) -> dict[str, Any]:
    """Train the model under several fold assignments and report the spread.

    A single CV score is one draw; this reports the distribution it was drawn from.
    """
    from .models.gbdt import train

    seeds = list(seeds)
    model_ll: list[float] = []
    base_ll: list[float] = []
    deltas: list[float] = []
    aucs: list[float] = []

    for s in pbar(seeds, desc="evaluate.repeated seeds", unit="seed"):
        b = _ensure_folds_and_baseline(s, subsample)
        r = _quiet(
            train,
            experiment="rep.model",
            blocks=blocks,
            cv_seed=s,
            subsample=subsample,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        model_ll.append(float(r["logloss"]))
        aucs.append(float(r["auc"]))
        base_ll.append(b)
        deltas.append(float(r["logloss"]) - b)

    res_model = summarize("model.gbdt logloss", model_ll, seeds)
    res_base = summarize("baseline.lo_only logloss", base_ll, seeds)
    res_delta = summarize("delta_vs_lo_only", deltas, seeds)
    res_auc = summarize("auc", aucs, seeds)

    out = {
        "seeds": seeds,
        "blocks": blocks,
        "model": res_model.to_dict(),
        "baseline": res_base.to_dict(),
        "delta_vs_lo_only": res_delta.to_dict(),
        "auc": res_auc.to_dict(),
        # headline numbers, pre-formatted for docs
        "headline": (
            f"{res_model.mean:.5f} ± {res_model.sd:.5f} "
            f"(delta {res_delta.mean:+.5f} ± {res_delta.sd:.5f}, "
            f"{'excludes' if res_delta.significant else 'INCLUDES'} zero)"
        ),
        "logloss": res_model.mean,  # so the task runner headlines something sensible
    }
    d = runs_dir() / "repeated"
    d.mkdir(parents=True, exist_ok=True)
    (d / "score.json").write_text(json.dumps(out, indent=2, default=str))
    out["output_path"] = str(d / "score.json")

    log.info("evaluate.repeated: %s", out["headline"])
    print("\n" + format_table([res_model, res_base, res_delta, res_auc], value_label="mean"))
    return out


@task(
    "interpret.ablation_repeated",
    requires="cpu",
    max_tier="cpu",
    description="PAIRED leave-one-block-out deltas across seeds, with mean ± SD",
)
def ablation_repeated(
    force: bool = False,
    subsample: int | None = None,
    blocks: list[str] | None = None,
    seeds: tuple[int, ...] | list[int] = DEFAULT_SEEDS,
    num_boost_round: int = 1200,
    early_stopping_rounds: int = 80,
) -> dict[str, Any]:
    """Leave-one-block-out, paired within each fold assignment.

    For every seed we fit the full stack and each leave-one-out variant on the **same**
    folds, so shared fold-assignment noise cancels in the difference. This is what makes a
    5e-4 effect measurable at all.
    """
    from .features.assemble import ALL_BLOCKS
    from .models.gbdt import train

    blocks = list(blocks or ALL_BLOCKS)
    seeds = list(seeds)

    full_by_seed: dict[int, float] = {}
    drop_by_seed: dict[str, dict[int, float]] = {b: {} for b in blocks}

    total = len(seeds) * (1 + len(blocks))
    with pbar(total=total, desc="ablation (paired, repeated)", unit="fit") as bar:
        for s in seeds:
            _ensure_folds_and_baseline(s, subsample)
            r = _quiet(
                train,
                experiment="abl.full",
                blocks=blocks,
                cv_seed=s,
                subsample=subsample,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )
            full_by_seed[s] = float(r["logloss"])
            bar.update(1)

            for b in blocks:
                remaining = [x for x in blocks if x != b]
                if not remaining:
                    bar.update(1)
                    continue
                rr = _quiet(
                    train,
                    experiment=f"abl.drop_{b}",
                    blocks=remaining,
                    cv_seed=s,
                    subsample=subsample,
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                )
                drop_by_seed[b][s] = float(rr["logloss"])
                bar.update(1)

    # positive delta => removing the block RAISED log loss => the block contributed
    results: list[RepeatedResult] = [
        paired_delta(b, full_by_seed, drop_by_seed[b]) for b in blocks if drop_by_seed[b]
    ]
    full_summary = summarize("all_blocks logloss", list(full_by_seed.values()), seeds)

    out: dict[str, Any] = {
        "seeds": seeds,
        "blocks": blocks,
        "all_blocks": full_summary.to_dict(),
        "marginal_contribution": {r.name: r.to_dict() for r in results},
        "ranked": [r.name for r in sorted(results, key=lambda r: -r.mean)],
        "distinguishable_from_zero": [r.name for r in results if r.significant],
    }
    d = runs_dir() / "interpret"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ablation_repeated.json").write_text(json.dumps(out, indent=2, default=str))
    out["output_path"] = str(d / "ablation_repeated.json")

    print(f"\nPAIRED leave-one-block-out across {len(seeds)} fold assignments")
    print("positive = removing the block made log loss WORSE (block contributes)\n")
    print(format_table(results))
    print(f"\nall_blocks logloss: {full_summary.mean:.5f} ± {full_summary.sd:.5f}")
    return out


@task(
    "evaluate.semantic_repeated",
    requires="cpu",
    max_tier="cpu",
    description="paired deployable-vs-BGE comparison with fold-safe content PCA",
)
def semantic_repeated(
    force: bool = False,
    subsample: int | None = None,
    seeds: tuple[int, ...] | list[int] = DEFAULT_SEEDS,
    num_boost_round: int = 1200,
    early_stopping_rounds: int = 80,
    promotion_threshold: float = 0.001,
) -> dict[str, Any]:
    """Compare the deployable stack against both BGE feature hypotheses.

    The embedding-alignment cache is selected explicitly rather than through the
    production lexical alias. Content PCA is fit by :func:`models.gbdt.train` inside
    every outer training fold. Positive paired deltas mean the BGE variant improved
    log loss relative to the deployable baseline.
    """
    from .features.assemble import DEFAULT_BLOCKS
    from .models.gbdt import train

    seeds = list(seeds)
    configs = {
        "deployable": list(DEFAULT_BLOCKS),
        "bge_alignment": [*DEFAULT_BLOCKS, "lo_alignment_embedding"],
        "bge_content": [*DEFAULT_BLOCKS, "content"],
    }
    scores: dict[str, dict[int, float]] = {name: {} for name in configs}

    with pbar(total=len(seeds) * len(configs), desc="semantic comparison", unit="fit") as bar:
        for s in seeds:
            _ensure_folds_and_baseline(s, subsample)
            for name, blocks in configs.items():
                result = _quiet(
                    train,
                    experiment=f"semantic.{name}",
                    blocks=blocks,
                    cv_seed=s,
                    subsample=subsample,
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                )
                scores[name][s] = float(result["logloss"])
                bar.update(1)

    score_summaries = {
        name: summarize(f"{name} logloss", [values[s] for s in seeds], seeds)
        for name, values in scores.items()
    }
    improvements = {
        name: summarize(
            f"{name} improvement",
            [scores["deployable"][s] - scores[name][s] for s in seeds],
            seeds,
        )
        for name in ("bge_alignment", "bge_content")
    }
    promoted = {
        name: bool(
            result.mean >= promotion_threshold
            and result.ci95[0] > 0
            and result.n_same_sign == result.n
        )
        for name, result in improvements.items()
    }
    eligible = [name for name, ok in promoted.items() if ok]
    winner = max(eligible, key=lambda name: improvements[name].mean) if eligible else None

    out: dict[str, Any] = {
        "seeds": seeds,
        "configs": configs,
        "scores": {name: result.to_dict() for name, result in score_summaries.items()},
        "improvement_vs_deployable": {
            name: result.to_dict() for name, result in improvements.items()
        },
        "promotion_threshold": promotion_threshold,
        "promoted": promoted,
        "winner": winner,
        "logloss": score_summaries["deployable"].mean,
    }
    d = runs_dir() / "repeated"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "semantic_comparison.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    out["output_path"] = str(path)

    print("\nPAIRED BGE improvement over deployable baseline (positive is better)\n")
    print(format_table(list(improvements.values()), value_label="gain"))
    print(
        f"\npromotion gate: gain >= {promotion_threshold:.4f}, CI above zero, "
        f"and same sign in every seed\npromoted: {promoted}"
    )
    return out
