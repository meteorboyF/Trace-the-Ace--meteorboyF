# Robust transformer plan

The frozen-embedding route is retired as a deployment candidate. Full grouped CV rewarded
learning-objective difficulty (`~0.72` AUROC), while the leaderboard behaved like the
transcript-only regime (`~0.60`). More compute is useful only after validation reproduces that
shift.

## Promotion splits

`cv.robust_build` creates two additional five-fold tables:

- `kind="objective"`: whole objectives are held out and every validation session is purged
  from training. This prevents both topic and identical-transcript leakage.
- `kind="domain"`: sessions are clustered from label-free structural, linguistic, and timing
  features, then whole transcript-style clusters are held out.

Ordinary session CV remains a diagnostic. A transformer is not promoted because it wins only
there.

## Model

`model.hierarchical_transformer` fine-tunes `answerdotai/ModernBERT-base` (Apache-2.0):

- role-tagged transcript chunks, retaining the final eight 512-token chunks by default;
- shared encoder for transcript chunks and learning-objective text;
- transcript-only attention/head plus an objective-conditioned attention/head;
- 50% objective dropout and an auxiliary transcript-only loss;
- bf16 on A100, gradient checkpointing, accumulation, clipping, and early stopping.

Objective IDs are never model inputs. The initial prediction is an equal blend of the two
heads; this is fixed before leaderboard feedback.

## Compute ladder

1. CPU: build normal features, then both robust fold tables.
2. A100 smoke: 500 sessions, objective fold 0, one epoch.
3. A100 one-fold run: full data, objective fold 0. Stop unless it improves the transcript-only
   baseline on the same held-out rows.
4. Repeat fold 0 using `split_mode="domain"`. Stop if the gain disappears.
5. Only then run all five folds. Package inference only after both robust regimes pass.

Example one-fold command:

```python
run(
    "model.hierarchical_transformer",
    split_mode="objective",
    fold=0,
    epochs=4,
)
run("maintenance.sync_artifacts", allow_waste=True)
```

The task is research-only today. `submission.build` continues to accept only the verified
LightGBM family; this prevents an unevaluated transformer from entering a submission.
