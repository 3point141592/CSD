from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_matched_scaling_study.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_matched_scaling_protocol_is_complete_and_paired() -> None:
    source = _source()
    assert "DIMENSIONS = (2, 4, 8, 16, 32, 64)" in source
    assert 'CONDITIONS = ("CurrentAttention", "EquivariantAttention")' in source
    assert '"paired_task_and_start_generation": True' in source
    assert '"paired_effective_batch_seed": True' in source
    for text in (
        '"training_geometries": 80',
        '"starts_per_task": 32',
        '"optimizer_steps": 1500',
        '"batch_size": 960',
        '"candidate_count": 8',
    ):
        assert text in source


def test_both_architectures_use_the_same_reported_metrics() -> None:
    source = _source()
    assert "**_evaluate(condition, model" in source
    assert '"proposal_recall"' not in source  # A single metric definition prevents comparison drift.
    assert "PRIMARY_METRICS" in source
    assert '"iid": (packed_test, h_test)' in source


def test_parameter_match_is_predeclared_and_checked() -> None:
    tree = ast.parse(_source())
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {"RELATIONAL_HIDDEN", "RELATIONAL_TOKEN", "MAX_PARAMETER_MISMATCH"}
    }
    assert assignments["RELATIONAL_HIDDEN"] == {
        2: 81,
        4: 92,
        8: 127,
        16: 234,
        32: 543,
        64: 1340,
    }
    assert assignments["RELATIONAL_TOKEN"] == 32
    assert assignments["MAX_PARAMETER_MISMATCH"] == 50
    assert "if mismatch > MAX_PARAMETER_MISMATCH" in _source()


def test_outputs_are_isolated_from_canonical_results() -> None:
    source = _source()
    assert (
        'DEFAULT_OUTPUT = PAPER / "assets" / "source_data" / "matched_scaling_study"'
        in source
    )
    assert '"original_results_modified": False' in source
