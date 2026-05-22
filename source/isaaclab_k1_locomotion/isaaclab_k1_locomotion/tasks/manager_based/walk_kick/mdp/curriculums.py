# mdp/curriculums.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def scale_feet_landing_penalty(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    threshold: float = 1.4,
    scale: float = 2.0,
) -> torch.Tensor | None:
    episode_extras = env.extras.get("log", {})

    lin_rew = episode_extras.get("Episode_Reward/track_lin_vel_xy_exp", None)
    ang_rew = episode_extras.get("Episode_Reward/track_ang_vel_z_exp", None)

    if lin_rew is None or ang_rew is None:
        return None

    # 初回だけ元のweightを記憶
    if not hasattr(env, "_feet_landing_base_weight"):
        env._feet_landing_base_weight = env.reward_manager.get_term_cfg("feet_landing_velocity").weight

    # スカラーでも配列でも対応
    def to_scalar(x):
        if isinstance(x, torch.Tensor):
            return x.mean().item()
        return float(x)

    combined = to_scalar(lin_rew) + to_scalar(ang_rew)

    base_weight = env._feet_landing_base_weight
    target_weight = base_weight * scale if combined > threshold else base_weight

    term = env.reward_manager.get_term_cfg("feet_landing_velocity")
    if abs(term.weight - target_weight) > 1e-6:
        term.weight = target_weight

    return torch.tensor(combined)


def window_reward_weight(
    env: ManagerBasedRLEnv,
    _env_ids: torch.Tensor,
    term_name: str,
    weight: float,
    start_step: int,
    end_step: int,
    steps_per_iteration: int = 0,
) -> None:
    """start_step < step <= end_step の期間だけ weight を適用し、それ以外は 0 にする。"""
    if steps_per_iteration > 0:
        step = env.common_step_counter // steps_per_iteration
    else:
        step = env.common_step_counter

    new_weight = weight if start_step < step <= end_step else 0.0

    term = env.reward_manager.get_term_cfg(term_name)
    if abs(term.weight - new_weight) > 1e-8:
        term.weight = new_weight


def linear_reward_weight(
    env: ManagerBasedRLEnv,
    _env_ids: torch.Tensor,
    term_name: str,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int,
    steps_per_iteration: int = 0,
) -> None:
    """ステップ数（またはiteration数）に応じて報酬重みを線形補間するカリキュラム。

    steps_per_iteration > 0 の場合、start_step/end_step をiteration単位として解釈する。
    steps_per_iteration = 0（デフォルト）の場合、common_step_counter（物理ステップ数）を使う。
    """
    if steps_per_iteration > 0:
        step = env.common_step_counter // steps_per_iteration
    else:
        step = env.common_step_counter
    if step <= start_step:
        new_weight = start_weight
    elif step >= end_step:
        new_weight = end_weight
    else:
        alpha = (step - start_step) / (end_step - start_step)
        new_weight = start_weight + (end_weight - start_weight) * alpha

    term = env.reward_manager.get_term_cfg(term_name)
    if abs(term.weight - new_weight) > 1e-8:
        term.weight = new_weight