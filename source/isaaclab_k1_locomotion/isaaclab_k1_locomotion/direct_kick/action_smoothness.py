# Ported from rl_humanoid_htwk 3af2acc97f1081b4cbfd9556efd39408dea92cc6: utils/action_smoothness.py
"""Action-smoothness reward helpers."""

import torch


def action_second_difference_l2(
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    previous_previous_actions: torch.Tensor,
) -> torch.Tensor:
    """Return the per-environment squared L2 action second difference."""
    if (
        actions.shape != previous_actions.shape
        or actions.shape != previous_previous_actions.shape
    ):
        raise ValueError("action history tensors must have matching shapes")
    if actions.ndim < 1:
        raise ValueError("action history tensors must have an action dimension")

    second_difference = (
        actions - 2.0 * previous_actions + previous_previous_actions
    )
    return torch.sum(torch.square(second_difference), dim=-1)
