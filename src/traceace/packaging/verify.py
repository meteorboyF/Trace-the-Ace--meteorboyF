"""Submission verification — loud, paranoid, and worth more than a clever model.

Only **three full submissions per week** are allowed (~15 real attempts remain), and a
rules violation risks disqualification rather than a bad score. So ``submission.verify``
fails hard on every failure mode in §12:

* ``main.py`` not at the zip root (a wrapping folder is the classic fatal mistake)
* row set or ordering mismatch against ``submission_format.csv``
* probabilities outside [0,1], NaN, or missing ``response_id``
* **any print/log that could emit test data** — a static AST scan of ``main.py`` and the
  modules it ships, rejecting prints of non-literal values in data-handling code
* progress bars not disabled
* imports that could touch the network
* projected runtime above 4.5 h against the 6 h cap
* zip above 55 GB, or total log lines above 400

Each check returns a structured result; the task raises unless ``strict=False``.
"""

from __future__ import annotations

import ast
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..io import load_submission_format
from ..logging_utils import get_logger
from ..paths import submission_dir
from ..tasks import task

log = get_logger("submission.verify")

# Modules that imply network access inside a no-network container.
NETWORK_MODULES = {
    "requests",
    "urllib",
    "urllib3",
    "httpx",
    "http",
    "aiohttp",
    "socket",
    "ftplib",
    "telnetlib",
    "smtplib",
    "boto3",
    "botocore",
    "gcsfs",
    "s3fs",
    "paramiko",
    "huggingface_hub",
    "datasets",
    "wandb",
    "mlflow",
    "gdown",
}
# Functions whose output could leak test data if given a non-literal argument.
EMITTERS = {"print", "pprint"}
LOGGER_METHODS = {"info", "warning", "error", "debug", "exception", "critical"}

# Checks that MUST execute on every run. Verified by name at the end of ``verify`` — see
# ADR-018, where nine output checks silently stopped running and the report still looked
# green. Add a check here when it guards something we cannot afford to ship broken.
REQUIRED_CHECKS = (
    "zip_exists",
    "zip_size_under_55gb",
    "main_py_at_zip_root",
    "no_test_data_in_logs",
    "no_network_imports",
    "no_cross_row_features",
    "sklearn_version_matches_runtime",
    "calibrator_is_version_proof",
    "feature_order_matches_booster",
    "progress_bars_disabled",
    "submission_csv_exists",
    "columns_exact",
    "row_count_matches",
    "no_missing_response_id",
    "row_set_matches",
    "row_ORDER_matches",
    "no_nan_probabilities",
    "probabilities_in_unit_interval",
    "probabilities_clipped_off_0_and_1",
)
# Additionally required when ``check_predictions`` is on (the default).
PREDICTION_CHECKS = (
    "beats_coin_flip",
    "prediction_sanity",
    "prediction_mean_near_base_rate",
    "feature_coverage",
    "feature_value_parity",
    "oof_replay_exact",
)
# Additionally required when the bundle ships the neural transcript encoder.
ENCODER_CHECKS = (
    "encoder_assets_complete",
    "encoder_spec_valid",
)

MAX_ZIP_GB = 55.0
MAX_PROJECTED_HOURS = 4.5
MAX_LOG_LINES = 400
# Artifact-integrity threshold, not a model-selection threshold. The known feature-order
# failure scored 0.4738; a correctly packaged transcript-only candidate scores 0.7466 while
# deliberately omitting the objective prior. Candidate quality is judged by held-out OOF and
# leaderboard evidence, whereas this check only needs to reject broken inference pipelines.
MIN_TRAIN_SANITY_AUC = 0.70

# scikit-learn version shipped by the competition container, read from a smoke-test log
# on 2026-07-27. Building against a different version makes joblib emit
# InconsistentVersionWarning when unpickling fitted estimators — "may lead to breaking code
# or invalid results". Update this when the organizers change the image.
RUNTIME_SKLEARN = "1.8.0"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class VerifyResult:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(Check(name, passed, detail))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_checks": len(self.checks),
            "n_failures": len(self.failures),
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }


# --- static scan -------------------------------------------------------------
def _is_static_literal(node: ast.AST) -> bool:
    """True if the expression is a compile-time constant string/number.

    ``print("start")`` is fine. ``print(f"n={n}")`` or ``print(len(df))`` is not — an
    f-string or any call/name could carry test-derived values.
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.JoinedStr):  # f-string: only safe if it has no interpolation
        return all(isinstance(v, ast.Constant) for v in node.values)
    if isinstance(node, ast.BinOp):  # "a" + "b"
        return _is_static_literal(node.left) and _is_static_literal(node.right)
    return False


def _sanctioned_wrapper_spans(tree: ast.AST) -> list[tuple[int, int, set[str]]]:
    """Line spans of the sanctioned static-logging wrapper ``def log(msg: str)``.

    ``main.py`` funnels all output through one tiny wrapper whose body is
    ``print(msg)``. Inside that definition ``msg`` is necessarily a Name, so a naive
    scan flags it forever. We allow printing the wrapper's *own parameters* inside the
    wrapper, and keep enforcing that every **call site** passes a literal — which is
    where a real leak would actually originate.
    """
    spans: list[tuple[int, int, set[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {"log", "_log"}:
            params = {a.arg for a in node.args.args}
            spans.append((node.lineno, node.end_lineno or node.lineno, params))
    return spans


def scan_source(source: str, filename: str) -> tuple[list[str], list[str]]:
    """Return (data_leak_findings, network_findings) for one source file."""
    leaks: list[str] = []
    net: list[str] = []
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [f"{filename}: unparseable ({exc})"], []

    wrapper_spans = _sanctioned_wrapper_spans(tree)

    def _allowed_in_wrapper(call: ast.Call) -> bool:
        for start, end, params in wrapper_spans:
            if start <= call.lineno <= end:
                # only bare parameter names of the wrapper are allowed
                return all(isinstance(a, ast.Name) and a.id in params for a in call.args)
        return False

    for node in ast.walk(tree):
        # imports that could reach the network
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in NETWORK_MODULES:
                    net.append(f"{filename}:{node.lineno} imports {a.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in NETWORK_MODULES:
                net.append(f"{filename}:{node.lineno} imports from {node.module}")

        # emitting calls
        if isinstance(node, ast.Call):
            fname = None
            if isinstance(node.func, ast.Name):
                fname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fname = node.func.attr

            is_emitter = fname in EMITTERS or fname in LOGGER_METHODS
            # our own single-argument static logger wrapper is allowed
            if fname == "log" and isinstance(node.func, ast.Name):
                is_emitter = True

            if is_emitter and not _allowed_in_wrapper(node):
                for arg in node.args:
                    if not _is_static_literal(arg):
                        leaks.append(
                            f"{filename}:{node.lineno} {fname}() emits a non-literal value"
                        )
                        break
    return leaks, net


def check_progress_disabled(source: str) -> bool:
    """main.py must set the progress kill-switch before importing anything bar-capable."""
    return ("TRACEACE_PROGRESS" in source and '"0"' in source) or "TQDM_DISABLE" in source


# --- zip / output checks -----------------------------------------------------
def verify_zip(zip_path: Path, result: VerifyResult) -> dict[str, str]:
    """Structural checks on the archive; returns {name: source} for shipped .py files."""
    sources: dict[str, str] = {}
    if not zip_path.is_file():
        result.add("zip_exists", False, f"{zip_path} not found")
        return sources
    result.add("zip_exists", True, str(zip_path))

    size_gb = zip_path.stat().st_size / 1e9
    result.add("zip_size_under_55gb", size_gb <= MAX_ZIP_GB, f"{size_gb:.3f} GB")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # main.py must be at the ROOT — not nested in a folder
        result.add(
            "main_py_at_zip_root",
            "main.py" in names,
            f"top-level entries: {sorted({n.split('/')[0] for n in names})}",
        )
        for n in names:
            if n.endswith(".py"):
                sources[n] = zf.read(n).decode("utf-8", errors="replace")
    return sources


def verify_predictions(
    sub_csv: Path,
    result: VerifyResult,
    smoke: bool = False,
    format_csv: Path | None = None,
) -> None:
    """Row set, ordering, and value-range checks against the official format.

    ``format_csv`` overrides the official format with the one a specific run was actually
    given. That matters for smoke output: the smoke harness feeds ``main.py`` a 100-row
    subsample, so checking it against the full 10,508-row format reports three guaranteed
    failures (count, set, ordering) that mean nothing. Waiving the checks would be worse —
    ordering is the bug that cost submission #1 — so instead we check ordering *exactly*,
    against the format that run was handed. Same rigour, correct denominator.
    """
    if not sub_csv.is_file():
        result.add("submission_csv_exists", False, f"{sub_csv} not found")
        return
    result.add("submission_csv_exists", True, str(sub_csv))

    got = pd.read_csv(sub_csv, dtype={"response_id": str})
    if format_csv is not None and format_csv.is_file():
        want = pd.read_csv(format_csv, dtype={"response_id": str})
    else:
        want = load_submission_format(smoke=smoke)

    result.add(
        "columns_exact",
        list(got.columns) == ["response_id", "probability"],
        f"got {list(got.columns)}",
    )
    result.add("row_count_matches", len(got) == len(want), f"{len(got)} vs {len(want)}")
    result.add("no_missing_response_id", not got["response_id"].isna().any())
    result.add("row_set_matches", set(got["response_id"]) == set(want["response_id"]))
    result.add(
        "row_ORDER_matches",
        got["response_id"].tolist() == want["response_id"].tolist(),
        "ordering must match submission_format.csv exactly",
    )

    p = pd.to_numeric(got["probability"], errors="coerce").to_numpy()
    result.add("no_nan_probabilities", bool(np.isfinite(p).all()))
    if np.isfinite(p).all():
        result.add(
            "probabilities_in_unit_interval",
            bool((p >= 0).all() and (p <= 1).all()),
            f"min={p.min():.6f} max={p.max():.6f}",
        )
        # exact 0/1 is legal but log-loss suicidal; warn via a check
        result.add(
            "probabilities_clipped_off_0_and_1",
            bool((p > 0).all() and (p < 1).all()),
            "exact 0 or 1 produces infinite log loss if wrong",
        )


def verify_feature_coverage(
    workdir: Path, result: VerifyResult, max_missing_frac: float = 0.0
) -> None:
    """Assert ``main.py`` actually PRODUCES every feature the model expects.

    **This check exists because we nearly shipped without it.** The model bundle listed 185
    features while ``main.py`` computed only 110 — the feedback, trajectory and LO-position
    blocks were never wired into the inference path. The missing 40% arrived as NaN, which
    LightGBM silently accepts, so every format check passed and the submission looked
    healthy. It would have burned one of three weekly attempts on a model missing its
    strongest block.

    Format validity does not imply feature validity.
    """
    import joblib

    bundle_path = workdir / "assets" / "model.joblib"
    debug_path = workdir / "_produced_features.json"
    if not bundle_path.is_file():
        result.add("feature_coverage", False, f"{bundle_path} not found")
        return
    if not debug_path.is_file():
        result.add(
            "feature_coverage",
            False,
            "main.py did not emit _produced_features.json (run submission.smoke first)",
        )
        return

    expected = set(joblib.load(bundle_path)["feature_cols"])
    produced = set(json.loads(debug_path.read_text()))
    missing = expected - produced
    frac = len(missing) / max(len(expected), 1)
    result.add(
        "feature_coverage",
        frac <= max_missing_frac,
        (
            f"{len(missing)}/{len(expected)} expected features NOT produced by main.py "
            f"({frac:.1%}); e.g. {sorted(missing)[:5]}"
            if missing
            else f"all {len(expected)} expected features produced"
        ),
    )


def verify_feature_value_parity(workdir: Path, result: VerifyResult) -> None:
    """Compare shipped feature VALUES with the cached training design matrix.

    Coverage and column order cannot detect a same-named feature whose arithmetic drifted
    between training and inference. This check runs the exact archived ``inference_lib``
    over the private training cohort already staged by ``verify_prediction_sanity`` and
    compares every deployed non-prior value. Only aggregate mismatch diagnostics leave
    this function; no transcript or feature values are logged.
    """
    import importlib.util

    import joblib

    from ..features.assemble import DEFAULT_BLOCKS, build_matrix

    bundle_path = workdir / "assets" / "model.joblib"
    feature_path = workdir / "data" / "test_features.csv"
    lib_path = workdir / "inference_lib.py"
    if not bundle_path.is_file() or not feature_path.is_file() or not lib_path.is_file():
        result.add("feature_value_parity", False, "sanity inputs or archived library missing")
        return

    spec = importlib.util.spec_from_file_location("_traceace_archived_inference", lib_path)
    if spec is None or spec.loader is None:
        result.add("feature_value_parity", False, "could not import archived inference library")
        return
    ilib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ilib)

    bundle = joblib.load(bundle_path)
    vectorizer = bundle.get("lo_vectorizer")
    deployed = [c for c in bundle["feature_cols"] if c != "lo_prior_enc"]
    sample = pd.read_csv(feature_path, dtype=str)
    expected, _ = build_matrix(
        sample[["response_id", "session_id"]].copy(), blocks=list(DEFAULT_BLOCKS)
    )
    expected = expected.set_index("response_id")

    rows: list[dict[str, Any]] = []
    tdir = workdir / "data" / "test_transcripts"
    for _, group in sample.groupby("session_id", sort=False):
        sid = str(group.iloc[0]["session_id"])
        try:
            transcript = ilib.normalize_frame(pd.read_csv(tdir / f"{sid}.csv", dtype=str))
            spans = ilib.windows(transcript)
            matrix = vectorizer.transform(ilib.window_texts(transcript, spans))
            session = ilib.all_session_features(transcript)
            session.update(ilib.feedback_features(transcript, prefix="fbs_"))
        except Exception as exc:
            result.add(
                "feature_value_parity",
                False,
                f"archived feature extraction failed ({type(exc).__name__})",
            )
            return
        for _, record in group.iterrows():
            values = dict(session)
            lo_text = str(record.get("learning_objective") or "")
            values.update(
                ilib.lo_alignment_features(transcript, lo_text, vectorizer, matrix, spans)
            )
            keep = ilib.topk_spans(lo_text, vectorizer, matrix, spans)
            window = ilib.frame_from_spans(transcript, keep)
            values.update(ilib.feedback_features(window, prefix="fb_"))
            values.update(ilib.trajectory_features(window))
            values.update(
                ilib.lo_position_features(
                    keep,
                    [keep],
                    len(transcript),
                    transcript["t_seconds"].to_numpy(dtype=float),
                )
            )
            values["response_id"] = str(record["response_id"])
            rows.append(values)

    actual = pd.DataFrame(rows).set_index("response_id")
    missing = sorted(set(deployed) - set(actual.columns))
    if missing:
        result.add(
            "feature_value_parity",
            False,
            f"archived inference omitted {len(missing)} deployed feature columns",
        )
        return
    ids = sample["response_id"].astype(str).tolist()
    actual_values = actual.loc[ids, deployed].to_numpy(dtype=float)
    expected_values = expected.loc[ids, deployed].to_numpy(dtype=float)
    equal = np.isclose(actual_values, expected_values, rtol=1e-9, atol=1e-9, equal_nan=True)
    finite_delta = np.abs(actual_values - expected_values)
    finite_delta = finite_delta[np.isfinite(finite_delta)]
    max_delta = float(finite_delta.max()) if finite_delta.size else 0.0
    bad_cells = int((~equal).sum())
    bad_cols = int((~equal).any(axis=0).sum())
    result.add(
        "feature_value_parity",
        bad_cells == 0,
        f"{bad_cells} mismatched cells; {len(deployed)} columns checked "
        f"({bad_cols} contain mismatches); "
        f"max_abs_delta={max_delta:.3g}",
    )
    if bad_cells:
        result.add("oof_replay_exact", False, "feature parity failed; replay is invalid")
        return

    from ..cv import load_folds
    from ..evaluate import load_oof

    experiment = str(bundle.get("experiment", "model.gbdt"))
    try:
        folds = load_folds().set_index("response_id")
        oof = load_oof(experiment).set_index("response_id")
        boosters = list(bundle["boosters"])
        prior_specs = list(bundle.get("lo_prior_by_booster", []))
        full_cols = list(bundle["feature_cols"])
        replay = pd.Series(index=ids, dtype=float)
        sample_by_id = sample.set_index("response_id")
        for fold, booster in enumerate(boosters):
            fold_ids = [rid for rid in ids if int(folds.loc[rid, "fold"]) == fold]
            if not fold_ids:
                continue
            matrix = actual.loc[fold_ids, deployed].copy()
            if "lo_prior_enc" in full_cols:
                lo_ids = sample_by_id.loc[fold_ids, "learning_objective_id"].to_numpy()
                matrix["lo_prior_enc"] = ilib.lo_prior_values(lo_ids, prior_specs[fold])
            replay.loc[fold_ids] = np.asarray(booster.predict(matrix[full_cols]), dtype=float)
        expected_oof = oof.loc[ids, "pred"].to_numpy(dtype=float)
        replayed = replay.loc[ids].to_numpy(dtype=float)
        replay_delta = np.abs(replayed - expected_oof)
        replay_max = float(np.nanmax(replay_delta))
        replay_bad = int((~np.isclose(replayed, expected_oof, rtol=1e-12, atol=1e-12)).sum())
        result.add(
            "oof_replay_exact",
            replay_bad == 0,
            f"{replay_bad}/{len(ids)} held-out predictions differ; max_abs_delta={replay_max:.3g}",
        )
    except Exception:
        result.add("oof_replay_exact", False, "could not replay packaged held-out predictions")


def verify_encoder_assets(workdir: Path, result: VerifyResult) -> bool:
    """Structural checks on the vendored encoder. Returns whether an encoder is shipped.

    Adds NO checks when there is no encoder directory — the expected-checks set is widened
    only for bundles that actually ship one, so ``all_expected_checks_ran`` stays exact in
    both configurations.
    """
    encoder_dir = workdir / "assets" / "encoder"
    if not encoder_dir.is_dir():
        return False

    problems: list[str] = []
    if not (workdir / "encoder_lib.py").is_file():
        problems.append("encoder_lib.py missing from zip root")
    tokenizer_dir = encoder_dir / "tokenizer"
    if not tokenizer_dir.is_dir() or not any(tokenizer_dir.iterdir()):
        problems.append("tokenizer/ missing or empty")
    else:
        # The container runs whatever transformers its image froze; a tokenizer_class name
        # from a NEWER transformers (e.g. v5's "TokenizersBackend") makes AutoTokenizer
        # raise before loading anything — the exact failure of submission job id-6296.
        # Only the universal class name is shippable.
        tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
        if tokenizer_config_path.is_file():
            declared = json.loads(tokenizer_config_path.read_text()).get("tokenizer_class")
            if declared not in (None, "PreTrainedTokenizerFast"):
                problems.append(
                    f"tokenizer_class {declared!r} is version-specific and will not load in "
                    "the container; rebuild (submission.build rewrites it)"
                )
        if not (tokenizer_dir / "tokenizer.json").is_file():
            problems.append("tokenizer/tokenizer.json missing (needed for the raw-load fallback)")
    config_dir = encoder_dir / "config"
    if not (config_dir / "config.json").is_file():
        problems.append("config/config.json missing")

    spec: dict[str, Any] | None = None
    try:
        from .encoder_lib import load_encoder_spec

        spec = load_encoder_spec(encoder_dir)
        result.add(
            "encoder_spec_valid",
            True,
            f"model={spec['model_name']} max_tokens={spec['max_tokens']} "
            f"topk={spec['topk_windows']} blend_weight={spec['blend_weight']}",
        )
    except Exception as exc:
        result.add("encoder_spec_valid", False, str(exc))

    if spec is not None:
        checkpoints = [encoder_dir / f"fold{k}.pt" for k in range(int(spec["n_folds"]))]
        missing = [p.name for p in checkpoints if not p.is_file()]
        if missing:
            problems.append(f"fold checkpoints missing: {missing}")
        empty = [p.name for p in checkpoints if p.is_file() and p.stat().st_size < 1_000_000]
        if empty:
            # A ModernBERT-class state dict is hundreds of MB; a tiny file is a truncated
            # copy, and torch.load would fail only at container time.
            problems.append(f"suspiciously small checkpoints (<1MB): {empty}")

    result.add(
        "encoder_assets_complete",
        not problems,
        "; ".join(problems) or "encoder_lib + tokenizer + config + all fold checkpoints present",
    )
    return True


def verify_sklearn_version(workdir: Path, result: VerifyResult) -> None:
    """Assert the bundle was built with the container's scikit-learn version.

    Our first container smoke test warned that a LogisticRegression and a TfidfVectorizer
    pickled under 1.9.0 were being unpickled under 1.8.0, "which might lead to breaking code
    or invalid results". The run still exited 0 — silent-degradation territory. The
    calibrator no longer crosses the pickle boundary at all (plain numbers), but the TF-IDF
    vectorizer still does, so the versions must match.
    """
    import joblib

    bundle_path = workdir / "assets" / "model.joblib"
    manifest_path = workdir / "assets" / "MANIFEST.json"
    if not manifest_path.is_file():
        return
    built_with = json.loads(manifest_path.read_text()).get("sklearn_build_version")
    if built_with is None:
        result.add("sklearn_version_matches_runtime", False, "manifest missing build version")
        return
    result.add(
        "sklearn_version_matches_runtime",
        str(built_with) == RUNTIME_SKLEARN,
        f"built with {built_with}, container has {RUNTIME_SKLEARN}",
    )

    # and confirm no fitted estimator is pickled where a plain-number form would do
    if bundle_path.is_file():
        cal = joblib.load(bundle_path).get("calibrator")
        pickled_cal = isinstance(cal, dict) and "model" in cal
        result.add(
            "calibrator_is_version_proof",
            not pickled_cal,
            "calibrator ships a pickled sklearn estimator"
            if pickled_cal
            else "calibrator is plain numbers (version-proof)",
        )


def verify_feature_order(workdir: Path, result: VerifyResult) -> None:
    """Assert the bundle's feature order equals what the boosters were trained on.

    **The check that was missing when it mattered.** The bundle's ``feature_cols`` was read
    from ``importance.parquet``, which is sorted by gain — a permutation of the training
    order in 179 of 181 positions. ``main.py`` reordered the columns to match, LightGBM read
    them positionally, and every feature was scrambled. Predictions remained confident and
    correctly formatted, so all 20 checks passed; the leaderboard returned AUC 0.4933.
    """
    import joblib

    bundle_path = workdir / "assets" / "model.joblib"
    if not bundle_path.is_file():
        return
    bundle = joblib.load(bundle_path)
    cols = list(bundle["feature_cols"])
    boosters = bundle.get("boosters") or []
    if not boosters:
        result.add("feature_order_matches_booster", False, "no boosters in bundle")
        return
    mismatches = []
    for i, b in enumerate(boosters):
        bn = list(b.feature_name())
        if bn != cols:
            n_diff = sum(1 for a, c in zip(bn, cols) if a != c)
            mismatches.append(f"booster{i}: {n_diff}/{len(bn)} positions differ")
    result.add(
        "feature_order_matches_booster",
        not mismatches,
        "; ".join(mismatches) if mismatches else f"all {len(boosters)} boosters agree",
    )


def _latest_smoke_csv(sdir: Path) -> Path | None:
    """Locate the submission.csv produced by the last ``submission.smoke`` run.

    Prefers the path recorded in ``smoke_report.json`` (authoritative, and survives a
    change to the smoke workdir layout), falling back to the conventional location.
    """
    report = sdir / "smoke_report.json"
    if report.is_file():
        try:
            recorded = json.loads(report.read_text()).get("submission_csv")
            if recorded and Path(recorded).is_file():
                return Path(recorded)
        except (OSError, ValueError):
            pass
    candidate = sdir / "_smoke" / "submission.csv"
    return candidate if candidate.is_file() else None


def verify_prediction_sanity(
    result: VerifyResult,
    zip_path: Path | None = None,
    n_sessions: int = 300,
    max_logloss: float = 0.60,
) -> Path | None:
    """Run the PACKAGED main.py on TRAINING data and check the predictions are sane.

    Every other check inspects structure. This one asks the only question that actually
    matters: **does the shipped artifact predict correctly?** A scrambled, degraded or
    misaligned pipeline fails here even when every structural check passes — which is
    exactly what happened when a feature-order permutation shipped at AUC 0.4933.

    **AUC is the primary integrity gate.** It separates a working artifact from a broken one
    (0.8285 versus 0.4738 when the columns were permuted) and is insensitive to calibration.
    The threshold must not encode the expected strength of one particular experiment: a
    transcript-only model intentionally omits objective-difficulty signal. Candidate quality
    is judged from held-out OOF; this check catches packaging corruption. ``beats_coin_flip``
    reports the unambiguous 0.693 log-loss line separately.

    Deliberately uses training data: it is the only data whose labels we hold, and a model
    that cannot rank data it was trained on is broken regardless of the test distribution.
    """
    import shutil
    import subprocess
    import sys
    import tempfile
    import zipfile

    from ..evaluate import auc as _auc
    from ..evaluate import logloss as _logloss
    from ..io import LABEL_COL, load_train_features, load_train_labels
    from ..paths import transcripts_dir

    sdir = submission_dir()
    zpath = zip_path or (sdir / "submission.zip")
    if not zpath.is_file():
        result.add("prediction_sanity", False, f"{zpath.name} missing")
        return None

    work = Path(tempfile.mkdtemp(prefix="_sanity_", dir=sdir))
    (work / "data" / "test_transcripts").mkdir(parents=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(work)

    feats = load_train_features()
    labels = load_train_labels()
    sess = feats["session_id"].drop_duplicates().head(n_sessions)
    sample = feats[feats["session_id"].isin(sess)].copy()
    for sid in sample["session_id"].unique():
        src = transcripts_dir() / f"{sid}.csv"
        if src.is_file():
            shutil.copyfile(src, work / "data" / "test_transcripts" / f"{sid}.csv")
    sample.to_csv(work / "data" / "test_features.csv", index=False)
    pd.DataFrame({"response_id": sample["response_id"], "probability": 0.5}).to_csv(
        work / "data" / "submission_format.csv", index=False
    )

    # An encoder bundle runs real transformer inference; on a CPU-only verify host that can
    # exceed the base 30-minute budget without anything being wrong.
    has_encoder = (work / "assets" / "encoder").is_dir()
    proc = subprocess.run(
        [sys.executable, "main.py"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=7200 if has_encoder else 1800,
    )
    if proc.returncode != 0:
        result.add("prediction_sanity", False, f"main.py exited {proc.returncode}")
        return work

    got = pd.read_csv(work / "submission.csv", dtype={"response_id": str})
    truth = labels.set_index("response_id")[LABEL_COL]
    got["y"] = got["response_id"].map(truth)
    got = got.dropna(subset=["y"])
    if len(got) < 50:
        result.add("prediction_sanity", False, "too few scorable rows")
        return work

    y = got["y"].to_numpy()
    p = got["probability"].to_numpy()
    ll, au = _logloss(y, p), _auc(y, p)

    # This is an artifact-integrity check, not a model promotion gate. Exact OOF replay below
    # proves which trained model was shipped; AUC and log loss catch grossly broken inference
    # before replay can be attempted.

    # The coin-flip line. Predicting a constant 0.5 on ANY binary labels scores ln(2)=0.693.
    # A model scoring WORSE than that is confidently wrong, not merely weak — which is a
    # qualitatively different diagnosis and the one we missed for two container smoke runs
    # (0.8543 and 0.8330) while a scrambled feature matrix was shipping. Report it always.
    COIN_FLIP = 0.6931
    result.add(
        "beats_coin_flip",
        ll < COIN_FLIP,
        f"logloss={ll:.5f} vs coin-flip {COIN_FLIP:.4f} — "
        + (
            "worse than a constant 0.5 prediction means CONFIDENTLY WRONG, "
            "not weak; suspect scrambled/misaligned features"
            if ll >= COIN_FLIP
            else "below the coin-flip line, so the model carries real signal"
        ),
    )
    # the model saw these rows in training, so it should fit them well
    ok = ll <= max_logloss and au >= MIN_TRAIN_SANITY_AUC
    result.add(
        "prediction_sanity",
        ok,
        f"on TRAINING data: auc={au:.4f} (integrity gate, need "
        f">={MIN_TRAIN_SANITY_AUC:.2f}) · logloss={ll:.5f} "
        f"(integrity bound <={max_logloss}) · mean_pred={p.mean():.4f}",
    )
    # a mean far from the training base rate is itself a red flag
    base = float(labels[LABEL_COL].mean())
    result.add(
        "prediction_mean_near_base_rate",
        abs(float(p.mean()) - base) < 0.10,
        f"mean_pred={p.mean():.4f} vs training base rate {base:.4f}",
    )
    return work


def _extract_for_structural_checks(zip_path: Path, work: Path) -> Path | None:
    """Extract the exact archive under review into a fresh directory."""
    import shutil

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(work)
    except (FileNotFoundError, zipfile.BadZipFile):
        return None
    return work


def verify_no_cross_row_features(workdir: Path, result: VerifyResult) -> None:
    """Assert the shipped model uses no feature derived from OTHER test rows.

    Competition rules require each test sample to be processed independently and preclude
    "using information gathered across multiple test samples as feature inputs". Pending a
    forum ruling we ship without them; this makes the guarantee mechanical, not remembered.
    """
    import joblib

    from ..features.assemble import CROSS_ROW_FEATURES

    bundle_path = workdir / "assets" / "model.joblib"
    if not bundle_path.is_file():
        return
    used = set(joblib.load(bundle_path)["feature_cols"]) & set(CROSS_ROW_FEATURES)
    result.add(
        "no_cross_row_features",
        not used,
        f"model uses cross-row features {sorted(used)}" if used else "none used (rules-safe)",
    )


def project_runtime(
    n_test: int, seconds_for_sample: float, n_sample: int, result: VerifyResult
) -> float:
    """Scale a measured sample runtime to the full test set and check the 4.5h budget."""
    if n_sample <= 0:
        result.add("runtime_projection", False, "no sample timing available")
        return float("nan")
    hours = (seconds_for_sample / n_sample) * n_test / 3600.0
    result.add(
        "projected_runtime_under_4.5h",
        hours <= MAX_PROJECTED_HOURS,
        f"projected {hours:.2f} h for {n_test} rows (cap 6 h, budget 4.5 h)",
    )
    return hours


@task(
    "submission.verify",
    requires="cpu",
    max_tier="cpu",
    description="hard checks on submission.zip: format, leakage, prints, network, runtime",
)
def verify(
    zip_name: str = "submission.zip",
    submission_csv: str | None = None,
    force: bool = False,
    subsample: int | None = None,
    smoke: bool = False,
    strict: bool = True,
    n_log_lines: int | None = None,
    check_predictions: bool = True,
) -> dict[str, Any]:
    sdir = submission_dir()
    result = VerifyResult()
    zip_path = sdir / zip_name

    sources = verify_zip(zip_path, result)

    # --- static scans over every python file we ship ------------------------
    all_leaks: list[str] = []
    all_net: list[str] = []
    for name, src in sources.items():
        leaks, net = scan_source(src, name)
        all_leaks += leaks
        all_net += net
    result.add(
        "no_test_data_in_logs",
        not all_leaks,
        "; ".join(all_leaks[:6]) or "no non-literal emissions",
    )
    result.add(
        "no_network_imports", not all_net, "; ".join(all_net[:6]) or "no network-capable imports"
    )

    # Bundle checks must inspect the EXACT archive named above, never a persistent
    # _smoke directory left by an older build.
    bundle_dir = _extract_for_structural_checks(zip_path, sdir / "_verify")
    encoder_shipped = False
    if bundle_dir is not None:
        verify_no_cross_row_features(bundle_dir, result)
        verify_sklearn_version(bundle_dir, result)
        verify_feature_order(bundle_dir, result)
        encoder_shipped = verify_encoder_assets(bundle_dir, result)

    main_src = sources.get("main.py", "")
    result.add(
        "progress_bars_disabled",
        check_progress_disabled(main_src),
        "main.py must set TRACEACE_PROGRESS=0 / TQDM_DISABLE=1 before imports",
    )

    if n_log_lines is not None:
        result.add("log_lines_under_400", n_log_lines <= MAX_LOG_LINES, f"{n_log_lines} lines")

    # --- output checks ------------------------------------------------------
    # Resolve the CSV the same way regardless of how verify was invoked. The default used
    # to be ``_staging/submission.csv`` — but ``_staging`` is the *zip build* directory and
    # never holds a CSV, while ``submission.smoke`` writes to ``_smoke/``. So a standalone
    # ``submission.verify`` silently skipped all NINE output checks (row set, ordering,
    # probability range, NaN) and only surfaced one cosmetic failure whose remedy text told
    # you to run the task that had, in fact, already produced the file. Row-ordering is
    # precisely the failure that cost submission #1; a check that cannot fire is worse than
    # no check, because it reads as coverage.
    csv_path = Path(submission_csv) if submission_csv else _latest_smoke_csv(sdir)
    if csv_path is not None and csv_path.is_file():
        # Check against the format that run was actually handed, when we can find it.
        run_fmt = csv_path.parent / "data" / "submission_format.csv"
        verify_predictions(
            csv_path, result, smoke=smoke, format_csv=run_fmt if run_fmt.is_file() else None
        )
    else:
        result.add(
            "submission_csv_exists",
            False,
            f"no smoke output found under {sdir / '_smoke'} — run submission.smoke first",
        )

    # The end-to-end sanity check: does THIS shipped artifact actually predict?
    if check_predictions:
        runtime_dir = verify_prediction_sanity(result, zip_path=zip_path)
        if runtime_dir is not None:
            verify_feature_coverage(runtime_dir, result)
            verify_feature_value_parity(runtime_dir, result)
            import shutil

            shutil.rmtree(runtime_dir, ignore_errors=True)

    # Did every check we rely on actually RUN? "0 failures" is not the same as "all checks
    # ran" — for weeks the nine output checks silently never executed because the CSV path
    # resolved to a directory that never contains one (ADR-018), and the verifier still
    # reported a tidy green list. This is the guard against that whole failure mode: a
    # check that goes missing is now itself a failure, by name.
    ran = {c.name for c in result.checks}
    expected = set(REQUIRED_CHECKS)
    if check_predictions:
        expected |= set(PREDICTION_CHECKS)
    if encoder_shipped:
        expected |= set(ENCODER_CHECKS)
    missing = sorted(expected - ran)
    result.add(
        "all_expected_checks_ran",
        not missing,
        f"{len(ran)} checks ran" if not missing else f"NEVER RAN: {', '.join(missing)}",
    )

    payload = result.to_dict()
    out = sdir / "verify_report.json"
    out.write_text(json.dumps(payload, indent=2))
    payload["output_path"] = str(out)

    for c in result.checks:
        log.log(
            20 if c.passed else 40,
            "%s %s %s",
            "PASS" if c.passed else "FAIL",
            c.name,
            f"({c.detail})" if c.detail else "",
        )

    if strict and not result.ok:
        raise AssertionError(
            f"submission.verify FAILED {len(result.failures)} check(s): "
            + "; ".join(f"{c.name}: {c.detail}" for c in result.failures)
        )
    return payload
