"""The ``main.py`` written into submission.zip.

Held as a template with exactly two substitutions (``__SEED__``, ``__EPS__``) so the
generated file stays readable and reviewable — ``submission.verify`` statically scans it.

Hard requirements encoded here (§2, §12):

* Progress bars **hard-disabled** before any import that could create one, because the
  container caps logging at 500 lines x 500 chars.
* **Never prints anything derived from the test data** — no excerpts, counts, sums,
  means, or token totals. Only static status strings. This is a disqualification-risk
  rule.
* Each test sample is processed **independently**: no fitting, no pseudo-labeling, no
  cross-sample information sharing.
* Writes ``submission.csv`` beside ``main.py``, matching ``submission_format.csv``
  row set and ordering exactly.

Feature extraction is imported from ``inference_lib.py`` (copied into the zip alongside
this file), which is the *same* module the training pipeline uses — no train/serve skew.
"""

from __future__ import annotations

MAIN_TEMPLATE = '''"""Trace the Ace — competition inference entrypoint.

Runs inside the DrivenData container: Python 3.12, no network, 1x A100, read-only data/.
Reads data/test_features.csv + data/test_transcripts/*.csv, writes submission.csv here.

Logging policy: STATIC STRINGS ONLY. Nothing derived from the test data is ever printed
(no excerpts, counts, sums, means, or token totals) — competition rules make this a
disqualification risk.
"""

from __future__ import annotations

import os

# Disable ALL progress output BEFORE importing anything that might emit it.
os.environ["TRACEACE_PROGRESS"] = "0"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import inference_lib as ilib

DATA = HERE / "data"
ASSETS = HERE / "assets"
SEED = __SEED__
EPS = __EPS__

np.random.seed(SEED)


def log(msg: str) -> None:
    """Print a STATIC status string. Never pass test-derived values here."""
    print(msg, flush=True)


def main() -> int:
    log("start")

    import joblib

    bundle = joblib.load(ASSETS / "model.joblib")
    feature_cols = list(bundle["feature_cols"])
    boosters = bundle["boosters"]
    calibrator = bundle.get("calibrator")
    vectorizer = bundle.get("lo_vectorizer")
    fallback = float(bundle.get("fallback_prob", 0.5))
    lo_prior = dict(bundle.get("lo_prior", {}))
    lo_prior_by_booster = list(bundle.get("lo_prior_by_booster", []))
    log("assets loaded")

    sub_fmt = pd.read_csv(DATA / "submission_format.csv", dtype={"response_id": str})
    features = pd.read_csv(DATA / "test_features.csv", dtype=str)
    required = {
        "response_id",
        "session_id",
        "learning_objective_id",
        "learning_objective",
    }
    if not required.issubset(features.columns):
        raise RuntimeError("test feature schema is invalid")
    if features[list(required)].isna().any().any():
        raise RuntimeError("test features contain missing required metadata")
    if (
        features["response_id"].duplicated().any()
        or sub_fmt["response_id"].duplicated().any()
        or set(features["response_id"]) != set(sub_fmt["response_id"])
    ):
        raise RuntimeError("input response ids are invalid or inconsistent")
    log("inputs read")

    # ---- per-sample feature extraction --------------------------------------
    # INDEPENDENCE: every feature below is a function of (this row's learning objective,
    # this session's own transcript, training-fitted parameters). Rows are grouped by
    # session only to avoid re-reading the same transcript file — never to derive a value
    # from another row. Organizer-prohibited cross-row aggregates are not shipped.
    rows = []
    tdir = DATA / "test_transcripts"
    for sid, grp in features.groupby("session_id", sort=False):
        tdf = None
        try:
            tdf = ilib.normalize_frame(pd.read_csv(tdir / f"{sid}.csv", dtype=str))
        except Exception:
            tdf = None
        transcript_ok = tdf is not None and not tdf.empty

        spans = []
        wm = None
        session_feats = {}
        session_fb = {}
        if transcript_ok:
            try:
                if vectorizer is None:
                    raise RuntimeError
                spans = ilib.windows(tdf)
                wm = vectorizer.transform(ilib.window_texts(tdf, spans))
                session_feats = ilib.all_session_features(tdf)
                session_fb = ilib.feedback_features(tdf, prefix="fbs_")
            except Exception:
                # Do not expose the session id/path through a chained traceback.
                raise RuntimeError("session feature extraction failed") from None

        for _, r in grp.iterrows():
            feats = dict(session_feats)
            feats.update(session_fb)
            lo_text = str(r.get("learning_objective") or "")

            if transcript_ok:
                try:
                    feats.update(
                        ilib.lo_alignment_features(tdf, lo_text, vectorizer, wm, spans)
                    )
                    # the top-k LO-relevant windows, merged and in chronological order
                    keep = ilib.topk_spans(lo_text, vectorizer, wm, spans)
                    sub = ilib.frame_from_spans(tdf, keep)
                    feats.update(ilib.feedback_features(sub, prefix="fb_"))
                    feats.update(ilib.trajectory_features(sub))
                    feats.update(
                        ilib.lo_position_features(
                            keep,
                            [keep],  # SAFE: no other test rows consulted
                            len(tdf),
                            tdf["t_seconds"].to_numpy(dtype=float),
                        )
                    )
                except Exception:
                    raise RuntimeError("response feature extraction failed") from None

            # Learning-objective difficulty prior, from TRAINING data only.
            # Must match the training-time feature name exactly (see models/gbdt.py).
            lo_id = r.get("learning_objective_id")
            feats["lo_prior_enc"] = float(lo_prior.get(str(lo_id), fallback))
            feats["response_id"] = r["response_id"]
            feats["_lo_id"] = lo_id
            feats["_use_fallback"] = not transcript_ok
            rows.append(feats)
    log("features extracted")

    X = pd.DataFrame(rows)

    # Audit trail for submission.verify: the NAMES of the features we produced.
    # Names only — never values, counts or aggregates of the test data.
    try:
        import json as _json

        (HERE / "_produced_features.json").write_text(
            _json.dumps(sorted(c for c in X.columns if not c.startswith("_")))
        )
    except Exception:
        pass

    lo_ids = X.pop("_lo_id").to_numpy() if "_lo_id" in X.columns else None
    use_fallback = X.pop("_use_fallback").to_numpy(dtype=bool)
    ids = X.pop("response_id").to_numpy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols].astype("float64")

    # Column order MUST match what each booster was trained on. LightGBM reads a
    # DataFrame positionally, so a permutation is silent and catastrophic — it cost us a
    # leaderboard submission at AUC 0.4933. Fail loudly instead.
    for b in boosters:
        bn = list(b.feature_name())
        if bn != feature_cols:
            raise RuntimeError("feature order does not match the trained booster")

    # ---- predict: average the per-fold boosters -----------------------------
    preds = np.zeros(len(X), dtype=float)
    needs_lo_prior = "lo_prior_enc" in feature_cols
    if needs_lo_prior and len(lo_prior_by_booster) != len(boosters):
        raise RuntimeError("fold-specific LO-prior artifacts do not match the boosters")
    if needs_lo_prior and lo_ids is None:
        raise RuntimeError("learning-objective ids are missing")
    for i, b in enumerate(boosters):
        X_fold = X
        if needs_lo_prior:
            X_fold = X.copy()
            X_fold["lo_prior_enc"] = ilib.lo_prior_values(
                lo_ids, lo_prior_by_booster[i]
            )
        preds = preds + np.asarray(b.predict(X_fold), dtype=float)
    preds = preds / max(len(boosters), 1)
    log("model predicted")

    if calibrator is not None:
        # plain-number calibrator: pure arithmetic, no sklearn object to unpickle
        preds = ilib.apply_calibration(calibrator, preds, eps=EPS)

    # Rows whose transcript was missing/unreadable/empty use the LO prior exactly. Do not
    # rely on LightGBM returning NaN: it normally emits a plausible finite prediction from
    # an all-NaN feature row.
    bad = ~np.isfinite(preds) | use_fallback
    if bad.any():
        if lo_ids is not None and lo_prior:
            preds[bad] = [float(lo_prior.get(str(v), fallback)) for v in lo_ids[bad]]
        else:
            preds[bad] = fallback
    preds = np.clip(preds, EPS, 1.0 - EPS)

    out = pd.DataFrame({"response_id": ids, "probability": preds})

    # ---- align EXACTLY to the submission format (row set and ordering) ------
    out = sub_fmt[["response_id"]].merge(out, on="response_id", how="left")
    out["probability"] = out["probability"].fillna(fallback).clip(EPS, 1.0 - EPS)
    out.to_csv(HERE / "submission.csv", index=False)

    log("wrote submission.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def render_main(seed: int, clip_eps: float) -> str:
    """Substitute the two runtime constants into the template."""
    return MAIN_TEMPLATE.replace("__SEED__", str(int(seed))).replace(
        "__EPS__", repr(float(clip_eps))
    )
