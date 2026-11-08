"""Compare center-level selectors with saved Relational CSD models at d=32.

The script reconstructs the canonical IID test tasks, groups co-centered
component records using coordinates alone, and compares simple center ranking
with the saved models. It does not train or overwrite a CSD checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from creativity_toy.config import STUDY_SEEDS
from creativity_toy.experiments.study_v2 import _make_starts
from gen3_gpu.config_symmetry import SymmetryStudyConfig
from gen3_gpu.landscape_representation import LandscapeProposalBatch
from gen3_gpu.study_capacity_diagnostic import _aligned
from gen3_gpu.study_d16 import _datasets
from gen3_gpu.study_failure_diagnostics import _family
from gen3_gpu.study_symmetry import _benchmark, _equivariant_model, _load
from gen3_gpu.symmetry import (
    EquivariantComponentBatch,
    build_equivariant_components,
)
from gen3_gpu.training_landscape_supervised import (
    _matching,
    component_evaluation_labels,
)
from run_scripts.reproduction_support import fresh_output, verify_recorded_sources
from torch import nn

TOLERANCE = 1e-5
TOP_COUNT = 3
ALLOCATION = (0, 0, 0, 1, 1, 1, 2, 2)
METHODS = (
    "amplitude_top3_repeat",
    "amplitude_distance_top3_repeat",
    "descriptor_logistic_top3_repeat",
    "random_top3_repeat",
    "amplitude_unique8",
    "amplitude_distance_unique8",
    "descriptor_logistic_unique8",
    "random_unique8",
    "relational_csd",
)


@dataclass(frozen=True)
class Protocol:
    dimension: int = 32
    candidates: int = 8
    train_geometries: int = 80
    starts_per_task: int = 32
    test_geometries: int = 5
    trials_per_task: int = 2
    refinement_steps: int = 9
    refinement_rate: float = 0.28
    useful_threshold: float = 0.62
    scorer_steps: int = 500
    scorer_batch_tasks: int = 960
    scorer_learning_rate: float = 0.03
    scorer_weight_decay: float = 1e-4
    center_tolerance: float = TOLERANCE
    disclosed_useful_mode_count: int = TOP_COUNT
    repeated_slot_allocation: tuple[int, ...] = ALLOCATION


@dataclass
class GroupedReferences:
    feature: torch.Tensor  # Per-center summaries let scoring use observable descriptors without usefulness labels.
    relative: torch.Tensor  # One displacement per center prevents repeated records from duplicating distance scores.
    mask: torch.Tensor  # Padding has no candidate geometry, so it must not enter group selection.
    selectable: torch.Tensor  # Excluding padding and the anchor restricts selection to actual alternative centers.
    useful: torch.Tensor  # Separate labels support evaluation without exposing ground truth to selector scoring.
    source_count: torch.Tensor  # Per-group multiplicity reveals how many repeated records were merged into each center.
    anchor_group: torch.Tensor  # The anchor identity makes its exclusion consistent for each task.


class GroupLogisticScorer(nn.Module):
    """One affine score per center group; no candidate or cross-group path."""

    def __init__(self, feature_count: int, *, device: torch.device) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_count, 1, device=device, dtype=torch.float32)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.linear(feature).squeeze(-1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def group_references(
    tasks,
    starts: torch.Tensor,
    components: EquivariantComponentBatch,
) -> GroupedReferences:
    """Group exactly co-centered records without consulting component labels."""
    record_useful, _ = component_evaluation_labels(tasks)
    batch, records, dimension = components.relative.shape
    feature_count = components.invariant.shape[-1]
    record_index = torch.arange(records, device=starts.device)
    pair_distance = torch.linalg.vector_norm(
        tasks.component_centers[:, :, None, :] - tasks.component_centers[:, None, :, :],
        dim=-1,
    )
    same_center = (
        (pair_distance <= TOLERANCE)
        & components.mask[:, :, None]
        & components.mask[:, None, :]
    )
    first_member = torch.where(
        same_center,
        record_index[None, None, :],
        torch.full((1, 1, records), records, device=starts.device, dtype=torch.int64),
    ).amin(dim=-1)
    representative = components.mask & (first_member == record_index[None, :])
    group_count = representative.sum(dim=1)
    maximum_groups = int(group_count.max().detach().cpu())
    representative_order = torch.where(
        representative,
        record_index[None, :],
        torch.full((1, records), records, device=starts.device, dtype=torch.int64),
    )
    representative_index = torch.argsort(
        representative_order, dim=1, descending=False, stable=True
    )[:, :maximum_groups]
    mask = torch.gather(representative, 1, representative_index)
    membership = (
        torch.gather(
            same_center,
            1,
            representative_index[:, :, None].expand(-1, -1, records),
        )
        & mask[:, :, None]
    )
    source_count = membership.sum(dim=-1)
    membership_float = membership.to(components.invariant.dtype)
    mean_descriptor = torch.einsum(
        "bgj,bjf->bgf", membership_float, components.invariant
    ) / source_count.clamp_min(1).unsqueeze(-1)
    expanded_descriptor = components.invariant[:, None, :, :]
    minimum_descriptor = torch.amin(
        torch.where(
            membership[:, :, :, None],
            expanded_descriptor,
            torch.full_like(expanded_descriptor, torch.inf),
        ),
        dim=2,
    )
    maximum_descriptor = torch.amax(
        torch.where(
            membership[:, :, :, None],
            expanded_descriptor,
            torch.full_like(expanded_descriptor, -torch.inf),
        ),
        dim=2,
    )
    feature = torch.cat(
        [mean_descriptor, minimum_descriptor, maximum_descriptor], dim=-1
    )
    feature = torch.where(mask[:, :, None], feature, torch.zeros_like(feature))
    relative = torch.gather(
        components.relative,
        1,
        representative_index[:, :, None].expand(-1, -1, dimension),
    )
    relative = torch.where(mask[:, :, None], relative, torch.zeros_like(relative))
    useful = torch.any(membership & record_useful[:, None, :], dim=-1)
    distance = torch.linalg.vector_norm(relative, dim=-1)
    distance = torch.where(mask, distance, torch.full_like(distance, torch.inf))
    anchor_group = torch.argmin(distance, dim=1)
    row_index = torch.arange(batch, device=starts.device)
    inferred_anchor = starts + relative[row_index, anchor_group]
    if (
        float(torch.linalg.vector_norm(inferred_anchor - tasks.anchor, dim=-1).amax())
        > TOLERANCE
    ):
        raise AssertionError(
            "nearest observable center does not match the canonical anchor"
        )

    selectable = mask.clone()
    selectable[torch.arange(batch, device=starts.device), anchor_group] = False
    if torch.any(selectable.sum(dim=1) != 8):
        raise AssertionError(
            "canonical d=32 tasks must expose exactly eight non-anchor center groups"
        )
    if torch.any(mask.sum(dim=1) != 9):
        raise AssertionError(
            "canonical d=32 tasks must expose exactly nine unique center groups"
        )
    return GroupedReferences(
        feature=feature,
        relative=relative,
        mask=mask,
        selectable=selectable,
        useful=useful,
        source_count=source_count,
        anchor_group=anchor_group,
    )


def feature_normalization(
    groups: GroupedReferences,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = groups.feature[groups.selectable]
    return selected.mean(dim=0), selected.std(dim=0, correction=0).clamp_min(1e-4)


def train_group_scorer(
    groups: GroupedReferences,
    *,
    mean: torch.Tensor,
    scale: torch.Tensor,
    seed: int,
    protocol: Protocol,
) -> tuple[GroupLogisticScorer, list[dict[str, float]]]:
    scorer = GroupLogisticScorer(groups.feature.shape[-1], device=groups.feature.device)
    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=protocol.scorer_learning_rate,
        weight_decay=protocol.scorer_weight_decay,
    )
    generator = torch.Generator(device=groups.feature.device).manual_seed(seed)
    history: list[dict[str, float]] = []
    for step in range(1, protocol.scorer_steps + 1):
        count = min(protocol.scorer_batch_tasks, groups.feature.shape[0])
        row = torch.randperm(
            groups.feature.shape[0], generator=generator, device=groups.feature.device
        )[:count]
        feature = (groups.feature.index_select(0, row) - mean) / scale
        selectable = groups.selectable.index_select(0, row)
        target = groups.useful.index_select(0, row).float()
        logit = scorer(feature)
        loss = F.binary_cross_entropy_with_logits(logit[selectable], target[selectable])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {1, 100, 250, protocol.scorer_steps}:
            with torch.no_grad():
                accuracy = (
                    ((logit[selectable] >= 0) == target[selectable].bool())
                    .float()
                    .mean()
                )
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach()),
                    "batch_accuracy": float(accuracy.detach()),
                }
            )
    scorer.eval()
    return scorer, history


def scores_for(
    method: str,
    groups: GroupedReferences,
    *,
    scorer: GroupLogisticScorer | None,
    mean: torch.Tensor | None,
    scale: torch.Tensor | None,
    random_seed: int,
) -> torch.Tensor:
    # Use each group's largest amplitude so weaker co-centered records do not dilute amplitude-based scores.
    amplitude = groups.feature[..., 2 * (groups.feature.shape[-1] // 3)]
    distance = torch.linalg.vector_norm(groups.relative, dim=-1)
    if method.startswith("amplitude_distance"):
        score = amplitude / (distance + 1e-8)
    elif method.startswith("amplitude"):
        score = amplitude
    elif method.startswith("descriptor_logistic"):
        if scorer is None or mean is None or scale is None:
            raise RuntimeError("descriptor scorer and normalization are required")
        score = scorer((groups.feature - mean) / scale)
    elif method.startswith("random"):
        generator = torch.Generator(device=groups.feature.device).manual_seed(
            random_seed
        )
        score = torch.rand(
            groups.mask.shape,
            generator=generator,
            device=groups.feature.device,
            dtype=groups.feature.dtype,
        )
    else:
        raise ValueError(method)
    return torch.where(groups.selectable, score, torch.full_like(score, -torch.inf))


def selector_proposal(
    method: str,
    starts: torch.Tensor,
    groups: GroupedReferences,
    *,
    scorer: GroupLogisticScorer | None,
    mean: torch.Tensor | None,
    scale: torch.Tensor | None,
    random_seed: int,
) -> tuple[LandscapeProposalBatch, torch.Tensor]:
    score = scores_for(
        method,
        groups,
        scorer=scorer,
        mean=mean,
        scale=scale,
        random_seed=random_seed,
    )
    order = torch.argsort(score, dim=1, descending=True, stable=True)
    if method.endswith("top3_repeat"):
        top = order[:, :TOP_COUNT]
        allocation = torch.as_tensor(
            ALLOCATION, device=starts.device, dtype=torch.int64
        )
        selected = top.index_select(1, allocation)
    elif method.endswith("unique8"):
        selected = order[:, :8]
    else:
        raise ValueError(method)
    gather = selected.unsqueeze(-1).expand(-1, -1, groups.relative.shape[-1])
    relative = torch.gather(groups.relative, 1, gather)
    alpha = torch.linalg.vector_norm(relative, dim=-1)
    direction = F.normalize(relative, dim=-1, eps=1e-8)
    candidate = starts[:, None, :] + alpha.unsqueeze(-1) * direction
    return LandscapeProposalBatch(relative, direction, alpha, candidate, None), selected


@torch.no_grad()
def point_metrics(
    tasks,
    points: torch.Tensor,
    *,
    useful_threshold: float,
    prefix: str,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    quality = tasks.quality(points)
    region = tasks.regions(points)
    useful = (
        (region >= 1)
        & (region <= tasks.n_useful[:, None])
        & (quality >= useful_threshold)
    )
    decoy = region > tasks.n_useful[:, None]
    anchor = region == 0
    background = region < 0
    useful_ids = torch.arange(1, int(tasks.n_useful.max()) + 1, device=points.device)
    useful_count = torch.sum(
        torch.any(
            useful[:, :, None] & (region[:, :, None] == useful_ids[None, None, :]),
            dim=1,
        ).float(),
        dim=1,
    )
    unique_region_count = []
    repeated_region_fraction = []
    for row in range(region.shape[0]):
        unique = torch.unique(region[row])
        unique_region_count.append(float(len(unique)))
        repeated_region_fraction.append(float(1.0 - len(unique) / region.shape[1]))
    metrics = {
        f"{prefix}_proposal_recall": float(torch.any(useful, dim=1).float().mean()),
        f"{prefix}_mode_set_recall": float(
            (useful_count / tasks.n_useful.float()).mean()
        ),
        f"{prefix}_useful_region_count": float(useful_count.mean()),
        f"{prefix}_candidate_useful_rate": float(useful.float().mean()),
        f"{prefix}_candidate_decoy_rate": float(decoy.float().mean()),
        f"{prefix}_candidate_anchor_rate": float(anchor.float().mean()),
        f"{prefix}_candidate_background_rate": float(background.float().mean()),
        f"{prefix}_candidate_mean_quality": float(quality.mean()),
        f"{prefix}_distinct_region_count": float(np.mean(unique_region_count)),
        f"{prefix}_repeated_region_fraction": float(np.mean(repeated_region_fraction)),
    }
    return metrics, {
        "quality": quality,
        "region": region,
        "useful": useful,
        "useful_count": useful_count,
    }


@torch.no_grad()
def evaluate_proposal(
    proposal: LandscapeProposalBatch,
    tasks,
    starts: torch.Tensor,
    *,
    protocol: Protocol,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    matching = _matching(proposal, starts, tasks, lambda_alpha=0.10)
    before, before_tensor = point_metrics(
        tasks,
        proposal.candidate,
        useful_threshold=protocol.useful_threshold,
        prefix="before",
    )
    refined = tasks.refine(
        proposal.candidate,
        steps=protocol.refinement_steps,
        rate=protocol.refinement_rate,
    )
    after, after_tensor = point_metrics(
        tasks, refined, useful_threshold=protocol.useful_threshold, prefix="after"
    )
    pair_distance = torch.linalg.vector_norm(
        proposal.candidate[:, :, None, :] - proposal.candidate[:, None, :, :], dim=-1
    )
    upper = torch.triu_indices(
        protocol.candidates, protocol.candidates, offset=1, device=starts.device
    )
    exact_repeat_pair_rate = torch.mean(
        (pair_distance[:, upper[0], upper[1]] <= TOLERANCE).float(), dim=1
    )
    summary = {
        "mean_matched_cosine": float(matching["matched_cosine"].mean()),
        "mean_alpha_absolute_error": float(
            matching["matched_alpha_error"].abs().mean()
        ),
        **before,
        **after,
        "exact_repeat_pair_rate": float(exact_repeat_pair_rate.mean()),
        "refinement_gradient_queries_per_set": float(
            protocol.refinement_steps * protocol.candidates
        ),
        "quality_metric_queries_per_set": float(2 * protocol.candidates),
    }
    per_set: list[dict[str, float]] = []
    for row in range(tasks.batch_size):
        per_set.append(
            {
                "set_index": float(row),
                "before_proposal_recall": float(
                    torch.any(before_tensor["useful"][row]).float()
                ),
                "before_mode_set_recall": float(
                    before_tensor["useful_count"][row] / tasks.n_useful[row]
                ),
                "after_proposal_recall": float(
                    torch.any(after_tensor["useful"][row]).float()
                ),
                "after_mode_set_recall": float(
                    after_tensor["useful_count"][row] / tasks.n_useful[row]
                ),
                "after_candidate_mean_quality": float(
                    after_tensor["quality"][row].mean()
                ),
                "exact_repeat_pair_rate": float(exact_repeat_pair_rate[row]),
            }
        )
    return summary, per_set


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        key for key in rows[0] if key not in {"seed", "method", "elapsed_seconds"}
    ]
    output: list[dict[str, object]] = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        summary: dict[str, object] = {"method": method, "n_seeds": len(subset)}
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in subset], dtype=np.float64
            )
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sd_sample"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            summary[f"{metric}_min"] = float(values.min())
            summary[f"{metric}_max"] = float(values.max())
        output.append(summary)
    return output


@torch.no_grad()
def _validate_grouping(groups: GroupedReferences) -> dict[str, object]:
    counts = groups.source_count[groups.mask].detach().cpu().numpy()
    return {
        "unique_groups_per_set": 9,
        "selectable_nonanchor_groups_per_set": 8,
        "group_source_count_values": sorted(set(int(value) for value in counts)),
        "duplicated_groups_per_set": float(
            (groups.source_count > 1).sum(dim=1).float().mean()
        ),
        "useful_groups_per_set": float(
            (groups.useful & groups.selectable).sum(dim=1).float().mean()
        ),
    }


def run(output: Path, device: torch.device, limit_seeds: int | None) -> None:
    output = fresh_output(output)
    verify_recorded_sources(
        ROOT, ROOT / "new_verification/fair_selector_d32/manifest.json"
    )
    protocol = Protocol()
    config = SymmetryStudyConfig()
    benchmark = _benchmark(config)
    base = benchmark.base.base
    observed = Protocol(
        train_geometries=base.train_geometries,
        test_geometries=base.test_geometries,
        trials_per_task=base.trials_per_task,
        refinement_steps=base.training.refinement_steps,
        refinement_rate=base.training.refinement_rate,
        useful_threshold=base.landscape.useful_threshold,
    )
    if observed != protocol:
        raise AssertionError(f"canonical protocol drift: {observed!r} != {protocol!r}")
    if (
        config.dimension != protocol.dimension
        or config.candidates != protocol.candidates
    ):
        raise AssertionError("canonical d=32/K=8 configuration changed")

    if limit_seeds is not None and not 1 <= limit_seeds <= len(STUDY_SEEDS):
        raise ValueError("limit-seeds must be between 1 and the canonical seed count")
    seeds = list(STUDY_SEEDS if limit_seeds is None else STUDY_SEEDS[:limit_seeds])
    output.mkdir(parents=True, exist_ok=True)
    model_output = output / "group_scorers"
    model_output.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    per_set_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    checkpoint_hashes: dict[str, str] = {}
    grouping_audit: dict[str, object] | None = None

    for ordinal, seed in enumerate(seeds, 1):
        family = _family(benchmark)
        train, _, _, _, test, test_starts = _datasets(family, benchmark, seed)
        train_nested = _make_starts(
            train, trials=config.starts_per_task, seed=seed + 41_001
        )
        packed_train, h_train = _aligned(train, train_nested, device)
        packed_test, h_test = _aligned(test, test_starts, device)
        train_components = build_equivariant_components(packed_train, h_train)
        test_components = build_equivariant_components(packed_test, h_test)
        train_groups = group_references(packed_train, h_train, train_components)
        test_groups = group_references(packed_test, h_test, test_components)
        if grouping_audit is None:
            grouping_audit = _validate_grouping(test_groups)
        mean, scale = feature_normalization(train_groups)
        scorer, history = train_group_scorer(
            train_groups,
            mean=mean,
            scale=scale,
            seed=seed + 211_001,
            protocol=protocol,
        )
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in scorer.state_dict().items()
                },
                "mean": mean.detach().cpu(),
                "scale": scale.detach().cpu(),
                "seed": seed,
                "parameter_count": sum(
                    parameter.numel() for parameter in scorer.parameters()
                ),
                "protocol": protocol.__dict__,
            },
            model_output / f"group_logistic_seed_{seed}.pt",
        )
        training_rows.extend({"seed": seed, **point} for point in history)

        for method in METHODS[:-1]:
            started = perf_counter()
            proposal, _ = selector_proposal(
                method,
                h_test,
                test_groups,
                scorer=scorer,
                mean=mean,
                scale=scale,
                random_seed=seed + 231_001,
            )
            summary, per_set = evaluate_proposal(
                proposal, packed_test, h_test, protocol=protocol
            )
            elapsed = perf_counter() - started
            rows.append(
                {"seed": seed, "method": method, **summary, "elapsed_seconds": elapsed}
            )
            per_set_rows.extend(
                {"seed": seed, "method": method, **record} for record in per_set
            )

        checkpoint = (
            ROOT
            / "results"
            / "d32_equivariant_attention"
            / "models"
            / f"EquivariantAttention_seed_{seed}.pt"
        )
        checkpoint_hashes[str(checkpoint.relative_to(ROOT))] = sha256(checkpoint)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        state = payload["state_dict"]
        model, _, _ = _load(
            checkpoint,
            lambda: _equivariant_model(
                config,
                seed=seed,
                device=device,
                mean=state["invariant_mean"],
                scale=state["invariant_scale"],
            ),
            device,
        )
        started = perf_counter()
        proposal = model(h_test, test_components)
        summary, per_set = evaluate_proposal(
            proposal, packed_test, h_test, protocol=protocol
        )
        elapsed = perf_counter() - started
        rows.append(
            {
                "seed": seed,
                "method": "relational_csd",
                **summary,
                "elapsed_seconds": elapsed,
            }
        )
        per_set_rows.extend(
            {"seed": seed, "method": "relational_csd", **record} for record in per_set
        )

        write_csv(output / "per_seed.csv", rows)
        write_csv(output / "per_set.csv", per_set_rows)
        write_csv(output / "scorer_training.csv", training_rows)
        atomic_json(
            output / "progress.json",
            {
                "completed_seeds": seeds[:ordinal],
                "required_seeds": len(seeds),
                "complete": ordinal == len(seeds),
            },
        )
        print(
            f"fair selector verification: seed {seed} ({ordinal}/{len(seeds)})",
            flush=True,
        )

    write_csv(output / "summary.csv", summarize(rows))
    after_hashes = {path: sha256(ROOT / path) for path in checkpoint_hashes}
    mutation_check = checkpoint_hashes == after_hashes
    if not mutation_check:
        raise AssertionError("canonical checkpoint mutation detected")
    atomic_json(
        output / "manifest.json",
        {
            "status": "new retrospective verification on previously inspected IID tasks",
            "protocol_source": str((RECOVERY / "NEW_EXPERIMENTS.md").relative_to(ROOT)),
            "protocol": protocol.__dict__,
            "seeds": seeds,
            "methods": list(METHODS),
            "grouping": {
                "rule": "group component records when center distance <= 1e-5",
                "feature": "concat(elementwise mean, min, max) of invariant record descriptors",
                "anchor": "exclude group nearest to h; audit against canonical anchor coordinate",
                "label_use": "group useful labels are used only to train/evaluate the tiny scorer",
                **(grouping_audit or {}),
            },
            "fairness": {
                "candidate_count": 8,
                "known_useful_mode_count_disclosed_to_top3_selectors": 3,
                "repeat_allocation": list(ALLOCATION),
                "same_test_tasks_starts_and_refiner": True,
                "test_tuning": False,
                "relational_model_frozen": True,
            },
            "query_accounting": {
                "proposal_quality_queries": 0,
                "refinement_gradient_queries_per_set": protocol.refinement_steps
                * protocol.candidates,
                "reported_quality_metric_queries_per_set": 2 * protocol.candidates,
            },
            "source_hashes": {
                "gen3_gpu/symmetry.py": sha256(ROOT / "gen3_gpu" / "symmetry.py"),
                "gen3_gpu/training_landscape_supervised.py": sha256(
                    ROOT / "gen3_gpu" / "training_landscape_supervised.py"
                ),
                "gen3_gpu/task_batch.py": sha256(ROOT / "gen3_gpu" / "task_batch.py"),
                "creativity_toy/landscapes_d16.py": sha256(
                    ROOT / "creativity_toy" / "landscapes_d16.py"
                ),
            },
            "checkpoint_hashes_before": checkpoint_hashes,
            "checkpoint_hashes_after": after_hashes,
            "canonical_checkpoint_mutation_check": mutation_check,
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU",
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit-seeds", type=int, default=None)
    args = parser.parse_args()
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    run(args.output.resolve(), torch.device(device_name), args.limit_seeds)


if __name__ == "__main__":
    main()
