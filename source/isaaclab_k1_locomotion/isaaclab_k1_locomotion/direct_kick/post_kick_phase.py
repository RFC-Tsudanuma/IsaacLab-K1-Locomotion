# Ported from rl_humanoid_htwk 3af2acc97f1081b4cbfd9556efd39408dea92cc6: utils/post_kick_phase.py
"""Auxiliary-loss helpers for the DirectKicking post-kick phase output."""

import torch
import torch.nn.functional as F


def class_balanced_phase_multipliers(
    targets: torch.Tensor,
    premature_weight: float,
    delayed_weight: float,
) -> torch.Tensor:
    """Return weights whose global mean is class-balanced across chunks."""
    if targets.ndim != 1:
        raise ValueError("post-kick phase targets must be one-dimensional")
    if premature_weight <= 0.0 or delayed_weight <= 0.0:
        raise ValueError("post-kick phase class weights must be positive")
    if torch.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("post-kick phase targets must be in [0, 1]")

    negative = targets < 0.5
    positive = ~negative
    negative_count = int(negative.sum().item())
    positive_count = int(positive.sum().item())
    sample_count = targets.numel()
    active_weight = (
        (premature_weight if negative_count else 0.0)
        + (delayed_weight if positive_count else 0.0)
    )
    if sample_count == 0 or active_weight <= 0.0:
        raise ValueError("post-kick phase targets must not be empty")

    multipliers = torch.zeros_like(targets)
    if negative_count:
        multipliers[negative] = (
            sample_count
            * premature_weight
            / negative_count
            / active_weight
        )
    if positive_count:
        multipliers[positive] = (
            sample_count
            * delayed_weight
            / positive_count
            / active_weight
        )
    return multipliers


def weighted_phase_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    multipliers: torch.Tensor,
) -> torch.Tensor:
    """Return the chunk mean used with the runner's batch-size weighting."""
    if logits.shape != targets.shape or multipliers.shape != targets.shape:
        raise ValueError("post-kick phase loss tensors must have matching shapes")
    per_sample = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    return torch.mean(per_sample * multipliers)
