from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).with_name("run_fair_selector_verification.py")
SPEC = importlib.util.spec_from_file_location("fair_selector", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fair = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fair
SPEC.loader.exec_module(fair)


def test_frozen_protocol() -> None:
    protocol = fair.Protocol()
    assert protocol.dimension == 32
    assert protocol.candidates == 8
    assert protocol.refinement_steps == 9
    assert protocol.refinement_rate == 0.28
    assert protocol.disclosed_useful_mode_count == 3
    assert protocol.repeated_slot_allocation == (0, 0, 0, 1, 1, 1, 2, 2)


def test_top3_allocation_repeats_three_groups() -> None:
    assert len(fair.ALLOCATION) == 8
    assert set(fair.ALLOCATION) == {0, 1, 2}
    assert [fair.ALLOCATION.count(index) for index in range(3)] == [3, 3, 2]


def test_selector_scores_cannot_consume_useful_labels() -> None:
    # A deliberately inverted label tensor cannot affect amplitude scores because
    # the scoring API accepts only observable grouped descriptors and geometry.
    feature = torch.zeros(1, 4, 6)
    feature[0, :, 4] = torch.tensor([0.1, 0.4, 0.3, 0.2])
    groups = fair.GroupedReferences(
        feature=feature,
        relative=torch.ones(1, 4, 2),
        mask=torch.ones(1, 4, dtype=torch.bool),
        selectable=torch.ones(1, 4, dtype=torch.bool),
        useful=torch.tensor([[True, False, True, False]]),
        source_count=torch.ones(1, 4, dtype=torch.int64),
        anchor_group=torch.tensor([0]),
    )
    score = fair.scores_for(
        "amplitude_top3_repeat",
        groups,
        scorer=None,
        mean=None,
        scale=None,
        random_seed=1,
    )
    assert torch.equal(
        torch.argsort(score, descending=True), torch.tensor([[1, 2, 3, 0]])
    )


def test_exact_repeat_proposals_have_three_distinct_endpoints() -> None:
    starts = torch.zeros(1, 2)
    relative = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [0.0, -4.0]]])
    feature = torch.zeros(1, 4, 6)
    feature[0, :, 4] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    groups = fair.GroupedReferences(
        feature=feature,
        relative=relative,
        mask=torch.ones(1, 4, dtype=torch.bool),
        selectable=torch.ones(1, 4, dtype=torch.bool),
        useful=torch.zeros(1, 4, dtype=torch.bool),
        source_count=torch.ones(1, 4, dtype=torch.int64),
        anchor_group=torch.tensor([3]),
    )
    proposal, selected = fair.selector_proposal(
        "amplitude_top3_repeat",
        starts,
        groups,
        scorer=None,
        mean=None,
        scale=None,
        random_seed=1,
    )
    assert selected.shape == (1, 8)
    assert torch.unique(proposal.candidate[0], dim=0).shape[0] == 3
    assert torch.allclose(
        proposal.candidate, starts[:, None, :] + proposal.raw_direction
    )
