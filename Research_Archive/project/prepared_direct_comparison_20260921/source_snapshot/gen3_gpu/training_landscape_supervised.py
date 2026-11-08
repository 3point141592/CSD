"""Supervised non-oracle landscape-conditioned training and evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from .landscape_representation import (
    ComponentDescriptorBatch,
    ComponentUtilityProbe,
    LandscapeProposalBatch,
    random_component_permutation,
)
from .runtime import synchronize
from .supervised_capacity import exact_assignment_indices, useful_targets
from .task_batch import TorchTaskBatch


@dataclass
class LandscapeTrainingResult:
    history: list[dict[str, float]]
    elapsed_seconds: float
    examples_seen: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


@dataclass
class LandscapeEvaluation:
    summary: dict[str, float]
    per_set: list[dict[str, float]]
    attention_rows: list[dict[str, float]]


def component_evaluation_labels(
    tasks: TorchTaskBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluation/label path only: map component centers to useful centers."""
    distance = torch.linalg.vector_norm(
        tasks.component_centers[:, :, None, :] - tasks.useful_centers[:, None, :, :],
        dim=-1,
    )
    distance = torch.where(
        tasks.useful_mask[:, None, :], distance, torch.full_like(distance, torch.inf)
    )
    minimum, useful_index = torch.min(distance, dim=-1)
    is_useful = (minimum <= 1e-5) & tasks.component_mask
    useful_index = torch.where(
        is_useful, useful_index, torch.full_like(useful_index, -1)
    )
    return is_useful, useful_index


def _matching(
    proposal: LandscapeProposalBatch,
    h: torch.Tensor,
    tasks: TorchTaskBatch,
    lambda_alpha: float,
) -> dict[str, torch.Tensor]:
    useful_direction, useful_alpha, target_direction, _ = useful_targets(h, tasks)
    target_count = int(tasks.n_useful[0].detach().cpu())
    if torch.any(tasks.n_useful != target_count):
        raise ValueError("a supervised batch must have a constant useful target count")
    cosine_all = torch.einsum("bkd,bmd->bkm", proposal.direction, useful_direction)
    pair_alpha = proposal.alpha[:, :, None] - useful_alpha[:, None, :]
    pair_cost = 1.0 - cosine_all + lambda_alpha * pair_alpha.square()
    matched_candidate = exact_assignment_indices(pair_cost, target_count)
    matched_target = torch.arange(target_count, device=h.device)[None, :].expand(
        len(h), -1
    )
    batch = torch.arange(len(h), device=h.device)[:, None]
    matched_cosine = cosine_all[batch, matched_candidate, matched_target]
    matched_alpha_error = pair_alpha[batch, matched_candidate, matched_target]
    target_cosine = torch.einsum("bkd,bd->bk", proposal.direction, target_direction)
    return {
        "useful_direction": useful_direction,
        "useful_alpha": useful_alpha,
        "cosine_all": cosine_all,
        "matched_candidate": matched_candidate,
        "matched_target": matched_target,
        "matched_cosine": matched_cosine,
        "matched_alpha_error": matched_alpha_error,
        "target_cosine": target_cosine,
    }


def landscape_supervised_objective(
    model: torch.nn.Module,
    h: torch.Tensor,
    tasks: TorchTaskBatch,
    components: ComponentDescriptorBatch | None,
    *,
    lambda_alpha: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], LandscapeProposalBatch]:
    proposal = model(h, components)
    matching = _matching(proposal, h, tasks, lambda_alpha)
    direction_loss = torch.mean(1.0 - matching["matched_cosine"])
    alpha_loss = torch.mean(matching["matched_alpha_error"].square())
    loss = direction_loss + lambda_alpha * alpha_loss
    return (
        loss,
        {
            "direction_loss": direction_loss,
            "alpha_loss": alpha_loss,
            "mean_matched_cosine": torch.mean(matching["matched_cosine"]),
            "mean_angular_error_degrees": torch.mean(
                torch.rad2deg(
                    torch.acos(torch.clamp(matching["matched_cosine"], -1.0, 1.0))
                )
            ),
            "mean_alpha_absolute_error": torch.mean(
                torch.abs(matching["matched_alpha_error"])
            ),
            "best_target_cosine": torch.mean(
                torch.amax(matching["target_cosine"], dim=1)
            ),
            "best_useful_cosine": torch.mean(
                torch.amax(matching["cosine_all"], dim=(1, 2))
            ),
        },
        proposal,
    )


def train_landscape_policy(
    model: torch.nn.Module,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    components: ComponentDescriptorBatch | None,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    lambda_alpha: float,
    seed: int,
    curve_steps: list[int],
    evaluation_callback: Callable[[torch.nn.Module, int], dict[str, float]]
    | None = None,
) -> LandscapeTrainingResult:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    batch_generator = torch.Generator(device=starts.device).manual_seed(seed)
    permutation_generator = torch.Generator(device=starts.device).manual_seed(seed + 1)
    log_steps = {value for value in curve_steps if 0 < value <= steps}
    log_steps.add(steps)
    history: list[dict[str, float]] = []
    if evaluation_callback is not None and 0 in curve_steps:
        history.append({"step": 0.0, **evaluation_callback(model, 0)})
    if starts.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(starts.device)
    synchronize(starts.device)
    started = perf_counter()
    for step in range(1, steps + 1):
        count = min(batch_size, len(starts))
        index = torch.randperm(
            len(starts), generator=batch_generator, device=starts.device
        )[:count]
        batch_tasks = tasks.index_select(index)
        batch_h = starts.index_select(0, index)
        batch_components = (
            components.index_select(index) if components is not None else None
        )
        if batch_components is not None:
            permutation = random_component_permutation(
                batch_components.mask, generator=permutation_generator
            )
            batch_components = batch_components.permute(permutation)
        loss, metrics, _ = landscape_supervised_objective(
            model,
            batch_h,
            batch_tasks,
            batch_components,
            lambda_alpha=lambda_alpha,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if step in log_steps:
            point = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                **{
                    name: float(value.detach().cpu()) for name, value in metrics.items()
                },
            }
            if evaluation_callback is not None:
                point.update(evaluation_callback(model, step))
            history.append(point)
    synchronize(starts.device)
    return LandscapeTrainingResult(
        history=history,
        elapsed_seconds=perf_counter() - started,
        examples_seen=steps * min(batch_size, len(starts)),
        peak_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(starts.device))
            if starts.device.type == "cuda"
            else 0
        ),
        peak_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(starts.device))
            if starts.device.type == "cuda"
            else 0
        ),
    )


@torch.no_grad()
def evaluate_landscape_policy(
    model: torch.nn.Module,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    components: ComponentDescriptorBatch | None,
    *,
    lambda_alpha: float,
    refinement_steps: int,
    refinement_rate: float,
    useful_threshold: float,
    component_label_tasks: TorchTaskBatch | None = None,
    collect_per_set: bool = True,
    collect_attention_rows: bool = True,
) -> LandscapeEvaluation:
    proposal = model(starts, components)
    matching = _matching(proposal, starts, tasks, lambda_alpha)
    refined = tasks.refine(
        proposal.candidate, steps=refinement_steps, rate=refinement_rate
    )
    quality = tasks.quality(refined)
    region = tasks.regions(refined)
    useful = (
        (region >= 1)
        & (region <= tasks.n_useful[:, None])
        & (quality >= useful_threshold)
    )
    target = (region == tasks.target_region[:, None]) & (quality >= useful_threshold)
    decoy = region > tasks.n_useful[:, None]
    proposal_recall = torch.any(useful, dim=1)
    useful_ids = torch.arange(
        1, int(torch.max(tasks.n_useful).cpu()) + 1, device=starts.device
    )
    useful_count = torch.sum(
        torch.any(
            useful[:, :, None] & (region[:, :, None] == useful_ids[None, None, :]),
            dim=1,
        ).float(),
        dim=1,
    )
    mode_set_recall = useful_count / tasks.n_useful.float()
    matched_cosine = matching["matched_cosine"]
    angular = torch.rad2deg(torch.acos(torch.clamp(matched_cosine, -1.0, 1.0)))
    summary = {
        "mean_matched_cosine": float(torch.mean(matched_cosine).cpu()),
        "mean_angular_error_degrees": float(torch.mean(angular).cpu()),
        "mean_alpha_absolute_error": float(
            torch.mean(torch.abs(matching["matched_alpha_error"])).cpu()
        ),
        "best_target_cosine": float(
            torch.mean(torch.amax(matching["target_cosine"], dim=1)).cpu()
        ),
        "best_useful_cosine": float(
            torch.mean(torch.amax(matching["cosine_all"], dim=(1, 2))).cpu()
        ),
        "proposal_recall": float(torch.mean(proposal_recall.float()).cpu()),
        "mode_set_recall": float(torch.mean(mode_set_recall).cpu()),
        "useful_region_count": float(torch.mean(useful_count).cpu()),
        "target_proposal_recall": float(
            torch.mean(torch.any(target, dim=1).float()).cpu()
        ),
        "candidate_mean_quality": float(torch.mean(quality).cpu()),
        "candidate_decoy_rate": float(torch.mean(decoy.float()).cpu()),
        "candidate_anchor_rate": float(torch.mean((region == 0).float()).cpu()),
        "alpha_mean": float(torch.mean(proposal.alpha).cpu()),
        "alpha_sd": float(torch.std(proposal.alpha, correction=0).cpu()),
    }
    per_set = (
        [
            {
                "set_index": float(index),
                "mean_matched_cosine": float(torch.mean(matched_cosine[index]).cpu()),
                "mean_alpha_absolute_error": float(
                    torch.mean(torch.abs(matching["matched_alpha_error"][index])).cpu()
                ),
                "proposal_recall": float(proposal_recall[index].float().cpu()),
                "mode_set_recall": float(mode_set_recall[index].cpu()),
                "useful_region_count": float(useful_count[index].cpu()),
            }
            for index in range(len(starts))
        ]
        if collect_per_set
        else []
    )
    attention_rows: list[dict[str, float]] = []
    if proposal.attention is not None and components is not None:
        label_tasks = component_label_tasks or tasks
        token_useful, token_target_index = component_evaluation_labels(label_tasks)
        if components.permutation is not None:
            token_useful = torch.gather(token_useful, 1, components.permutation)
            token_target_index = torch.gather(
                token_target_index, 1, components.permutation
            )
        attention = proposal.attention
        useful_mass = torch.sum(attention * token_useful[:, None, :].float(), dim=-1)
        normalized_attention = F.normalize(attention, p=2, dim=-1, eps=1e-8)
        similarity = torch.matmul(
            normalized_attention, normalized_attention.transpose(-1, -2)
        )
        upper = torch.triu_indices(
            attention.shape[1], attention.shape[1], offset=1, device=starts.device
        )
        diversity = torch.mean(1.0 - similarity[:, upper[0], upper[1]], dim=1)
        entropy = -torch.sum(
            attention * torch.log(torch.clamp(attention, min=1e-12)), dim=-1
        )
        top_token = torch.argmax(attention, dim=-1)
        top_target = torch.gather(token_target_index, 1, top_token)
        batch = torch.arange(len(starts), device=starts.device)[:, None]
        matched_top = top_target[batch, matching["matched_candidate"]]
        specialization = matched_top == matching["matched_target"]
        summary.update(
            {
                "attention_candidate_diversity": float(torch.mean(diversity).cpu()),
                "attention_useful_mass": float(torch.mean(useful_mass).cpu()),
                "attention_entropy": float(torch.mean(entropy).cpu()),
                "attention_target_specialization": float(
                    torch.mean(specialization.float()).cpu()
                ),
            }
        )
        if collect_attention_rows:
            diversity_cpu = diversity.cpu().numpy()
            useful_mass_cpu = torch.mean(useful_mass, dim=1).cpu().numpy()
            entropy_cpu = torch.mean(entropy, dim=1).cpu().numpy()
            specialization_cpu = torch.mean(specialization.float(), dim=1).cpu().numpy()
            attention_rows = [
                {
                    "set_index": float(index),
                    "candidate_attention_diversity": float(diversity_cpu[index]),
                    "attention_useful_mass": float(useful_mass_cpu[index]),
                    "attention_entropy": float(entropy_cpu[index]),
                    "attention_target_specialization": float(specialization_cpu[index]),
                }
                for index in range(len(starts))
            ]
    return LandscapeEvaluation(summary, per_set, attention_rows)


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(np.sum(ranks[labels == 1]))
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def train_component_utility_probe(
    probe: ComponentUtilityProbe,
    train_components: ComponentDescriptorBatch,
    train_labels: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[list[dict[str, float]], float]:
    descriptor = train_components.values[train_components.mask]
    labels = train_labels[train_components.mask].float()
    positives = torch.sum(labels)
    negatives = len(labels) - positives
    positive_weight = negatives / torch.clamp(positives, min=1.0)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device=descriptor.device).manual_seed(seed)
    history = []
    synchronize(descriptor.device)
    started = perf_counter()
    for step in range(1, steps + 1):
        count = min(batch_size, len(descriptor))
        index = torch.randperm(
            len(descriptor), generator=generator, device=descriptor.device
        )[:count]
        logits = probe(descriptor.index_select(0, index))
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels.index_select(0, index),
            pos_weight=positive_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {1, max(1, steps // 4), max(1, steps // 2), steps}:
            history.append({"step": float(step), "loss": float(loss.detach().cpu())})
    synchronize(descriptor.device)
    return history, perf_counter() - started


@torch.no_grad()
def evaluate_component_utility_probe(
    probe: ComponentUtilityProbe,
    components: ComponentDescriptorBatch,
    labels_full: torch.Tensor,
) -> dict[str, float]:
    descriptor = components.values[components.mask]
    labels = labels_full[components.mask].long()
    probability = torch.sigmoid(probe(descriptor))
    prediction = probability >= 0.5
    true_positive = torch.sum(prediction & (labels == 1)).float()
    false_positive = torch.sum(prediction & (labels == 0)).float()
    true_negative = torch.sum((~prediction) & (labels == 0)).float()
    false_negative = torch.sum((~prediction) & (labels == 1)).float()
    precision = true_positive / torch.clamp(true_positive + false_positive, min=1.0)
    recall = true_positive / torch.clamp(true_positive + false_negative, min=1.0)
    specificity = true_negative / torch.clamp(true_negative + false_positive, min=1.0)
    label_numpy = labels.cpu().numpy()
    probability_numpy = probability.cpu().numpy()
    return {
        "accuracy": float(torch.mean((prediction == labels.bool()).float()).cpu()),
        "balanced_accuracy": float((0.5 * (recall + specificity)).cpu()),
        "auroc": _binary_auc(label_numpy, probability_numpy),
        "precision": float(precision.cpu()),
        "recall": float(recall.cpu()),
        "specificity": float(specificity.cpu()),
        "useful_nonuseful_score_separation": float(
            torch.mean(probability[labels == 1]).cpu()
            - torch.mean(probability[labels == 0]).cpu()
        ),
        "useful_count": float(torch.sum(labels == 1).cpu()),
        "nonuseful_count": float(torch.sum(labels == 0).cpu()),
    }
