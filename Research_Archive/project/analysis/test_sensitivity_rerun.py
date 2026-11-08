"""Unit tests of the sensitivity evaluator using synthetic test cases."""

import ast
import dataclasses
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import run_sensitivity_rerun as r


@pytest.fixture(scope="module")
def fixture():
    torch.set_num_threads(1)
    return r.make_test(r.driver.matched_config(32), 9040001)


def test_exact_grid_and_no_training():
    p = r.protocol()
    assert len(p["seeds"]) == 30 and len(set(p["seeds"])) == 30
    assert p["dimensions"] == [32, 64]
    assert 2 * 30 * 4 * 4 * 3 * 3 == 8640
    assert p["refinement"]["steps"] == [0, 3, 9, 18]
    assert p["execution"]["training_steps"] == 0
    assert p["cap_audit"]["methods"] == [
        "amplitude_top3_repeat",
        "amplitude_unique8",
        "random_top3_repeat",
    ]
    source = Path(r.__file__).read_text()
    tree = ast.parse(source)
    forbidden = {
        "backward",
        "step",
        "train_group_scorer",
        "train_symmetry_policy",
        "save",
    }
    calls = {
        getattr(n.func, "attr", getattr(n.func, "id", ""))
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert not (forbidden & calls)


def test_test_only_generation_matches_historical_loader():
    config = dataclasses.replace(r.driver.matched_config(32), training_geometries=1)
    benchmark = r.driver._benchmark(config)
    _, _, _, _, tasks, starts = r.driver._datasets(
        r.driver._family(benchmark), benchmark, 9040001
    )
    old, h_old = r.driver._aligned(tasks, starts, torch.device("cpu"))
    new, h_new = r.make_test(config, 9040001)
    assert r.tensor_digest(old, h_old) == r.tensor_digest(new, h_new)


def test_cached_refinement_exactly_matches_fresh_steps(fixture):
    tasks, h = fixture
    proposal, _ = r.multiplicity_proposal(tasks, h)
    # Include poorly placed points so gradient clipping/far fallback are exercised.
    initial = proposal.candidate + 3.0
    saved = r.endpoints(tasks, initial, r.BUDGETS)
    for step in r.BUDGETS:
        assert torch.equal(
            saved[step],
            tasks.refine(initial, steps=step, rate=0.28, max_gradient_norm=4.0),
        )


def test_threshold_scoring_does_not_change_task_or_points(fixture):
    tasks, h = fixture
    before = r.tensor_digest(tasks, h)
    proposal, _ = r.multiplicity_proposal(tasks, h)
    points = proposal.candidate.clone()
    low, *_ = r.score_points(tasks, points, 0.60, 3.0)
    high, *_ = r.score_points(tasks, points, 0.65, 2.0)
    assert high["mode_set_recall"] <= low["mode_set_recall"]
    assert r.tensor_digest(tasks, h) == before
    assert torch.equal(points, proposal.candidate)
    assert tasks.region_threshold == 2.5
    rates = [
        high[k]
        for k in [
            "candidate_useful_rate",
            "candidate_decoy_rate",
            "candidate_anchor_rate",
            "candidate_background_rate",
            "candidate_useful_below_threshold_rate",
        ]
    ]
    assert abs(sum(rates) - 1) < 1e-6


def test_multiplicity_uses_no_hidden_roles(fixture):
    tasks, h = fixture
    first, selected = r.multiplicity_proposal(tasks, h)
    changed = dataclasses.replace(
        tasks,
        useful_centers=torch.zeros_like(tasks.useful_centers),
        goals=torch.zeros_like(tasks.goals),
        component_amplitudes=torch.flip(tasks.component_amplitudes, [1]),
    )
    second, selected2 = r.multiplicity_proposal(changed, h)
    assert torch.equal(first.candidate, second.candidate)
    assert torch.equal(selected, selected2)
    assert torch.unique(selected[0]).numel() == 3
    assert torch.equal(selected[0, :3], selected[0, 3:6])


def test_cap_zero_short_and_long_movements():
    h = torch.zeros(1, 2)
    raw = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [8.0, 0.0],
                [0.0, -9.0],
                [3.0, 4.0],
                [6.0, 0.0],
                [10.0, 10.0],
                [-4.0, 0.0],
            ]
        ]
    )
    proposal = r.LandscapeProposalBatch(
        raw,
        torch.nn.functional.normalize(raw, dim=-1),
        torch.linalg.vector_norm(raw, dim=-1),
        raw.clone(),
        None,
    )
    cap = r.cap_proposal(proposal, h, 6.0)
    assert torch.isfinite(cap.candidate).all()
    assert float(cap.alpha.max()) <= 6.0 + 1e-6
    short = proposal.alpha <= 6
    assert torch.equal(cap.candidate[short], proposal.candidate[short])
    assert torch.allclose(cap.direction[~short], proposal.direction[~short], atol=1e-6)


def test_cap_pair_reuses_same_random_selection(fixture):
    tasks, h = fixture
    groups = r.fair.group_references(tasks, h, r.build_equivariant_components(tasks, h))
    original, selected = r.fair.selector_proposal(
        "random_top3_repeat",
        h,
        groups,
        scorer=None,
        mean=None,
        scale=None,
        random_seed=9040001 + 231001,
    )
    cap = r.cap_proposal(original, h, 6.0)
    assert selected.shape == (30, 8)
    assert torch.allclose(cap.direction, original.direction, atol=2e-6)
    assert (
        float(torch.linalg.vector_norm(cap.candidate - h[:, None, :], dim=-1).max())
        <= 6.0 + 1e-5
    )


@pytest.mark.parametrize(
    "dimension,method",
    [
        (32, "Direct16"),
        (32, "DirectWide"),
        (32, "Relational"),
        (64, "Direct16"),
        (64, "DirectWide"),
        (64, "Relational"),
    ],
)
def test_saved_models_load_frozen(dimension, method):
    model = r.load_model(r.driver.matched_config(dimension), 7, method)
    assert not model.training
    assert all(
        not p.requires_grad and p.dtype == torch.float32 for p in model.parameters()
    )


def test_unit_completion_tampering_is_rejected(tmp_path):
    r.csv_write(tmp_path / "unit.csv", [{"value": 1}])
    r.complete_unit(tmp_path, "test", ["unit.csv"])
    assert r.unit_done(tmp_path, "test")
    (tmp_path / "unit.csv").write_text("modified", encoding="utf-8")
    with pytest.raises(RuntimeError):
        r.unit_done(tmp_path, "test")
