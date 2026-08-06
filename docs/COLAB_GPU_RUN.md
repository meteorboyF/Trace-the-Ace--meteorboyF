# L4 transformer smoke — operator checklist

This is an engineering smoke, not a submission build and not evidence of leaderboard quality.
The verified transcript-only ZIP remains untouched. Do not upload private competition data to
GitHub or any API.

## 1. CPU preparation

Open the runner from GitHub on **CPU + High RAM**, run the setup cell, and confirm the printed
commit matches the latest reviewed SHA. Then run:

```python
run("maintenance.restore_artifacts")
run("data.ingest")
run("data.consolidate")
run("cv.build")
run("cv.robust_build", kind="objective")
run("cv.robust_build", kind="domain")
run("maintenance.sync_artifacts")
```

The objective folds must report five validation cohorts of roughly 7K responses each. The
domain folds require the cached structural, linguistic, and temporal feature blocks. If those
are missing, build those three blocks on CPU and rerun the domain task.

## 2. L4 settings

Start a fresh **NVIDIA L4** runtime and edit only these setup flags:

```python
RUN_FULL_GPU = False
RUN_EXPERIMENTAL_GPU = False
RUN_ATTENTION_GPU = False
RUN_TRANSFORMER_GPU = True
RUN_TRANSFORMER_ALL_FOLDS = False
```

Run setup and confirm `accel: NVIDIA L4`. Restore artifacts and stage raw data; CPU-tagged
staging calls on an attached L4 require `allow_waste=True`:

```python
run("maintenance.restore_artifacts", allow_waste=True)
run("data.ingest", allow_waste=True)
run("data.consolidate", allow_waste=True)
```

## 3. Run only the transformer GPU cell

Run the cell containing `model.hierarchical_transformer`. The notebook invokes:

```python
run(
    "model.hierarchical_transformer",
    subsample=500,
    split_mode="objective",
    fold=0,
    epochs=1,
    batch_size=1,
    accumulation_steps=16,
    max_chunks=4,
)
```

This configuration samples four chunks uniformly across each lesson and is deliberately
conservative for 24 GB VRAM. It downloads the official ModernBERT checkpoint, performs one
training epoch, evaluates the transcript/conditional/blended heads, saves a half-precision
checkpoint without casting integer buffers, syncs artifacts, and disconnects.

Do not set `RUN_TRANSFORMER_ALL_FOLDS=True`. Do not run BGE, vLLM, submission, or A100 timing
cells in this session.

## 4. Report back

Return only aggregate output:

- exact synced commit;
- train and validation row counts;
- blended log loss/AUROC and `gain_vs_train_rate`;
- all three head metrics;
- `peak_gpu_gb`, wall time, and units;
- artifact-sync confirmation or the complete traceback.

The smoke passes engineering only if the checkpoint and OOF artifacts sync successfully and
peak memory leaves reasonable L4 headroom. A full-data fold requires a separate explicit
decision after reviewing these results.
