# First Colab GPU run — operator checklist

This run is an evidence-gathering run. The current verified submission remains the safe
fallback. Semantic/content features are **not deployable** until `main.py` gains an offline
encoder implementation and train/serve parity tests; packaging now refuses them explicitly.

## 1. Put the private inputs on Drive

Create this private Drive layout:

```text
MyDrive/
└── trace-the-ace/
    └── data/
        └── raw.zip
```

`raw.zip` must contain the canonical inputs below, either directly or under one enclosing
folder that `data.ingest` can normalize:

```text
train_features.csv
train_labels.csv
submission_format.csv
submission_format_smoke.csv
train_transcripts.zip
```

Do not put any competition data in GitHub. Confirm the Drive file finishes uploading before
starting a paid runtime.

## 2. Before attaching a GPU

1. Push the reviewed code and notebook to `main`.
2. Open `notebooks/Trace_the_Ace_Runner.ipynb` from GitHub in Colab.
3. Start on **CPU + High RAM**.
4. Run setup, staging/restore, feature, model, and self-test cells through the CPU pipeline.
5. Confirm every cell is green. A task exception now stops the cell; do not continue after one.

## 3. L4 smoke gate

Switch to an **L4**, leave `RUN_FULL_GPU = False`, and run only the semantic GPU cell:

```python
run("features.window_embeddings", subsample=500)
run("features.embeddings", subsample=500)
```

Proceed only if both complete, cache paths are printed, no OOM occurs, and the budget report
shows the expected L4 rate.

## 4. Full semantic extraction

Set:

```python
RUN_FULL_GPU = True
```

Run the semantic GPU cell. It performs the full window extraction, synchronizes
`data/features/` and `data/interim/` to Drive, and disconnects the paid runtime.

Do not interrupt the synchronization step. The expected Drive outputs are:

```text
MyDrive/trace-the-ace/cache/features.tar
MyDrive/trace-the-ace/cache/interim.tar
MyDrive/trace-the-ace/artifacts/rollup.tar
MyDrive/trace-the-ace/runs/runs.tar
```

## 5. CPU evaluation after reconnect

Reconnect to **CPU + High RAM**, run setup and:

```python
run("maintenance.restore_artifacts")
run("features.lo_alignment", backend="embedding", force=True)
run("features.content")
run("interpret.ablation_repeated")
```

Interpretation gates:

- Do not use single-seed scores.
- Require a paired repeated-seed improvement whose interval excludes zero.
- Treat the current global-PCA content result as exploratory until PCA is fitted per CV fold.
- Do not call `submission.build` on an experiment containing `cont_`, `emb_`, or `move_`;
  the build now refuses these research-only feature families.

## 6. What to report back

Copy only aggregate task output—never transcript or learning-objective text:

- `features.window_embeddings`: wall time, window count, dimensions, cache paths.
- `features.embeddings`: wall time, session count, dimensions.
- `features.lo_alignment`: response count and feature count.
- `interpret.ablation_repeated`: mean ± SD and 95% interval for each changed block.
- `budget.report`: units spent and remaining.
- Any traceback, with paths/IDs redacted if necessary.

At that point decide between:

1. Submit the verified CPU + cross-fitted-shrinkage ZIP.
2. Spend development time implementing offline encoder inference only if semantic features
   show a real repeated-seed gain large enough to justify the added runtime risk.
