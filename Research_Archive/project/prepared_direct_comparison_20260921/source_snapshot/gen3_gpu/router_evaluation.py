"""Same-candidate selector evaluation and conservative-router diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from .router_useful_novelty import FrozenCandidateBatch, mean_spearman


@dataclass
class SelectorEvaluation:
    summary: dict[str, float]
    set_rows: list[dict[str, object]]
    selected_indices: torch.Tensor


def random_selector_scores(
    batch: FrozenCandidateBatch, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    indices = np.asarray(
        [int(rng.integers(0, batch.candidate_count)) for _ in range(batch.batch_size)]
    )
    scores = torch.as_tensor(
        rng.random((batch.batch_size, batch.candidate_count)),
        dtype=batch.quality.dtype,
        device=batch.quality.device,
    )
    selected = torch.as_tensor(indices, dtype=torch.long, device=batch.quality.device)
    return scores, selected


def useful_oracle_scores(
    batch: FrozenCandidateBatch,
    useful_threshold: float,
) -> torch.Tensor:
    useful = (
        (batch.regions >= 1)
        & (batch.regions <= batch.n_useful[:, None])
        & (batch.quality >= useful_threshold)
    )
    target = (batch.regions == batch.target_regions[:, None]) & (
        batch.quality >= useful_threshold
    )
    # Membership offsets prevent high-quality decoys from outranking useful hits.
    # Target hits get first priority; quality breaks ties within each group.
    return (
        batch.quality + useful.to(batch.quality.dtype) + target.to(batch.quality.dtype)
    )


def _selected(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, indices[:, None]).squeeze(1)


def _coverage_and_collapse(
    batch: FrozenCandidateBatch,
    selected_regions: torch.Tensor,
    selected_quality: torch.Tensor,
    useful_threshold: float,
) -> tuple[float, float]:
    coverages, collapses = [], []
    for task_id in torch.unique(batch.task_ids).detach().cpu().tolist():
        mask = batch.task_ids == int(task_id)
        regions = selected_regions[mask]
        qualities = selected_quality[mask]
        n_useful = int(batch.n_useful[mask][0].cpu())
        good = (regions >= 1) & (regions <= n_useful) & (qualities >= useful_threshold)
        chosen = regions[good].detach().cpu().tolist()
        coverages.append(len(set(chosen)) / n_useful)
        if len(chosen) < 2:
            collapses.append(1.0)
        else:
            counts = Counter(int(region) for region in chosen)
            same = sum(count * (count - 1) for count in counts.values())
            collapses.append(same / (len(chosen) * (len(chosen) - 1)))
    return float(np.mean(coverages)), float(np.mean(collapses))


@torch.no_grad()
def evaluate_selector(
    *,
    generator: str,
    selector: str,
    batch: FrozenCandidateBatch,
    scores: torch.Tensor,
    useful_novelty_target: torch.Tensor,
    useful_threshold: float,
    selected_indices: torch.Tensor | None = None,
) -> SelectorEvaluation:
    indices = (
        torch.argmax(scores, dim=1) if selected_indices is None else selected_indices
    )
    q_selected = _selected(batch.quality, indices)
    raw_q_selected = _selected(batch.raw_quality, indices)
    region_selected = _selected(batch.regions, indices)
    alpha_selected = _selected(batch.alphas, indices)
    novelty_selected = _selected(batch.novelty, indices)
    quality_oracle = torch.amax(batch.quality, dim=1)
    useful_scores = useful_oracle_scores(batch, useful_threshold)
    useful_indices = torch.argmax(useful_scores, dim=1)
    useful_oracle_quality = _selected(batch.quality, useful_indices)
    useful = (
        (region_selected >= 1)
        & (region_selected <= batch.n_useful)
        & (q_selected >= useful_threshold)
    )
    target = (region_selected == batch.target_regions) & (
        q_selected >= useful_threshold
    )
    decoy = region_selected > batch.n_useful
    anchor = region_selected == 0
    coverage, collapse = _coverage_and_collapse(
        batch, region_selected, q_selected, useful_threshold
    )
    set_rows: list[dict[str, object]] = []
    all_indices = torch.arange(batch.candidate_count, device=batch.quality.device)
    for set_index in range(batch.batch_size):
        chosen = int(indices[set_index].cpu())
        unselected_mask = all_indices != chosen
        row = {
            "generator": generator,
            "selector": selector,
            "set_index": set_index,
            "task_id": int(batch.task_ids[set_index].cpu()),
            "trial": int(batch.trials[set_index].cpu()),
            "selected_index": chosen,
            "selected_quality": float(q_selected[set_index].cpu()),
            "selected_raw_quality": float(raw_q_selected[set_index].cpu()),
            "selected_alpha": float(alpha_selected[set_index].cpu()),
            "selected_displacement": float(alpha_selected[set_index].cpu()),
            "selected_novelty": float(novelty_selected[set_index].cpu()),
            "selected_anchor": float(anchor[set_index].cpu()),
            "unselected_quality": float(
                torch.mean(batch.quality[set_index, unselected_mask]).cpu()
            ),
            "unselected_raw_quality": float(
                torch.mean(batch.raw_quality[set_index, unselected_mask]).cpu()
            ),
            "unselected_alpha": float(
                torch.mean(batch.alphas[set_index, unselected_mask]).cpu()
            ),
            "unselected_displacement": float(
                torch.mean(batch.alphas[set_index, unselected_mask]).cpu()
            ),
            "unselected_novelty": float(
                torch.mean(batch.novelty[set_index, unselected_mask]).cpu()
            ),
            "unselected_anchor_probability": float(
                torch.mean(
                    (batch.regions[set_index, unselected_mask] == 0).float()
                ).cpu()
            ),
        }
        for name in ["quality", "raw_quality", "alpha", "displacement", "novelty"]:
            row[f"selected_minus_unselected_{name}"] = float(
                row[f"selected_{name}"]
            ) - float(row[f"unselected_{name}"])
        row["selected_minus_unselected_anchor_probability"] = float(
            row["selected_anchor"]
        ) - float(row["unselected_anchor_probability"])
        set_rows.append(row)
    summary = {
        "selected_quality": float(torch.mean(q_selected).cpu()),
        "selected_raw_quality": float(torch.mean(raw_q_selected).cpu()),
        "useful_selection_rate": float(torch.mean(useful.float()).cpu()),
        "target_selection_rate": float(torch.mean(target.float()).cpu()),
        "decoy_selection_rate": float(torch.mean(decoy.float()).cpu()),
        "anchor_selection_rate": float(torch.mean(anchor.float()).cpu()),
        "coverage": coverage,
        "collapse": collapse,
        "quality_oracle_regret": float(torch.mean(quality_oracle - q_selected).cpu()),
        "useful_oracle_quality_gap": float(
            torch.mean(useful_oracle_quality - q_selected).cpu()
        ),
        "useful_oracle_hit_regret": float(
            torch.mean(
                (_selected(useful_scores, useful_indices) >= 1.0).float()
                - useful.float()
            ).cpu()
        ),
        "score_quality_spearman": mean_spearman(scores, batch.quality),
        "score_useful_novelty_spearman": mean_spearman(scores, useful_novelty_target),
        "selected_alpha": float(torch.mean(alpha_selected).cpu()),
        "selected_novelty": float(torch.mean(novelty_selected).cpu()),
        "selected_displacement": float(torch.mean(alpha_selected).cpu()),
        "candidate_sets": float(batch.batch_size),
    }
    return SelectorEvaluation(
        summary=summary, set_rows=set_rows, selected_indices=indices
    )


def conservative_diagnostic_summary(
    rows: list[Mapping[str, object]],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in [
        "quality",
        "raw_quality",
        "alpha",
        "displacement",
        "novelty",
        "anchor_probability",
    ]:
        key = f"selected_minus_unselected_{name}"
        output[key] = float(np.mean([float(row[key]) for row in rows]))
    return output
