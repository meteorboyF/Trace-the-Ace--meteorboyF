"""The Colab runner is an executable pipeline, not unaudited documentation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import traceace


def _notebook_code() -> str:
    path = Path(__file__).resolve().parents[1] / "notebooks" / "Trace_the_Ace_Runner.ipynb"
    notebook = json.loads(path.read_text())
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebook_task_manifest_matches_registry_and_calls(synth_repo):
    """A registered task cannot silently become unreachable from the Colab wrapper."""
    tree = ast.parse(_notebook_code())
    manifest: set[str] | None = None
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "PIPELINE_TASKS"
                for target in node.targets
            ):
                value = ast.literal_eval(node.value)
                manifest = set(value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            calls.add(node.args[0].value)

    assert manifest is not None
    registry = {spec.name for spec in traceace.tasks.list_tasks()}
    assert manifest == registry
    assert calls == manifest


def test_notebook_run_wrapper_reraises_task_failures():
    """A failed feature/calibration cell must stop packaging instead of returning None."""
    tree = ast.parse(_notebook_code())
    wrapper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    handlers = [node for node in ast.walk(wrapper) if isinstance(node, ast.ExceptHandler)]
    assert handlers
    assert any(
        isinstance(statement, ast.Raise) for handler in handlers for statement in handler.body
    )
