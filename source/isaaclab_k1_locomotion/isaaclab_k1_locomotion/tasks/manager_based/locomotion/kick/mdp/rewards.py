# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import math
import time

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


KICK_STAGE_DISCOVERY = 1
KICK_STAGE_POWER = 2
KICK_STAGE_RECOVERY = 3
KICK_STAGE_FINAL_POSE = 4
PROFILER_MAX_COMMON_STEPS = 2048
PROFILER_LOG_INTERVAL_STEPS = 128
PROFILER_PRINT_TOP_K = 8
K1_LEFT_LEG_BODY_NAMES = (
    "Left_Hip_Yaw",
    "Left_Shank",
    "Left_Ankle_Cross",
    "left_foot_link",
)
K1_RIGHT_LEG_BODY_NAMES = (
    "Right_Hip_Yaw",
    "Right_Shank",
    "Right_Ankle_Cross",
    "right_foot_link",
)


def _ball_contact_force(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    return torch.linalg.norm(sensor.data.force_matrix_w[:, 0, 0], dim=1)


def _ensure_kick_buffers(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    if not hasattr(env, "_kick_buffers"):
        env._kick_buffers = {}
    return env._kick_buffers


def _ensure_curriculum_buffers(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor | int | float]:
    kick_buffers = _ensure_kick_buffers(env)
    if "curriculum_stage" not in kick_buffers:
        kick_buffers["curriculum_stage"] = KICK_STAGE_DISCOVERY
    if "peak_ball_forward_speed" not in kick_buffers:
        kick_buffers["peak_ball_forward_speed"] = torch.zeros(env.num_envs, device=env.device)
    if "peak_ball_forward_distance" not in kick_buffers:
        kick_buffers["peak_ball_forward_distance"] = torch.zeros(env.num_envs, device=env.device)
    return kick_buffers


def _ensure_reward_profiler(env: ManagerBasedRLEnv) -> dict[str, dict[str, float] | int | bool]:
    if not hasattr(env, "_kick_reward_profiler"):
        env._kick_reward_profiler = {
            "stats": {},
            "last_log_step": -1,
            "printed_summary": False,
        }
    return env._kick_reward_profiler


def _get_step_cache(env: ManagerBasedRLEnv) -> dict[tuple, object]:
    current_step = int(getattr(env, "common_step_counter", 0))
    if not hasattr(env, "_kick_step_cache") or getattr(env, "_kick_step_cache_step", None) != current_step:
        env._kick_step_cache = {}
        env._kick_step_cache_step = current_step
    return env._kick_step_cache


def _profile_step_enabled(env: ManagerBasedRLEnv) -> bool:
    return getattr(env, "common_step_counter", 0) <= PROFILER_MAX_COMMON_STEPS


def _record_profile_stat(env: ManagerBasedRLEnv, name: str, elapsed_s: float) -> None:
    profiler = _ensure_reward_profiler(env)
    stats = profiler["stats"]
    if name not in stats:
        stats[name] = {"cum_s": 0.0, "max_s": 0.0, "calls": 0.0}
    entry = stats[name]
    entry["cum_s"] += elapsed_s
    entry["calls"] += 1.0
    entry["max_s"] = max(entry["max_s"], elapsed_s)


def _flush_profile_stats(env: ManagerBasedRLEnv) -> None:
    if not hasattr(env, "extras"):
        return
    profiler = _ensure_reward_profiler(env)
    current_step = int(getattr(env, "common_step_counter", 0))
    should_log = current_step <= PROFILER_MAX_COMMON_STEPS and (
        current_step <= 16 or current_step % PROFILER_LOG_INTERVAL_STEPS == 0 or current_step == PROFILER_MAX_COMMON_STEPS
    )
    if not should_log or profiler["last_log_step"] == current_step:
        return
    stats = profiler["stats"]
    if "log" not in env.extras:
        env.extras["log"] = {}
    for name, entry in stats.items():
        calls = max(entry["calls"], 1.0)
        env.extras["log"][f"RewardProfiler/{name}_avg_ms"] = 1000.0 * entry["cum_s"] / calls
        env.extras["log"][f"RewardProfiler/{name}_cum_ms"] = 1000.0 * entry["cum_s"]
        env.extras["log"][f"RewardProfiler/{name}_max_ms"] = 1000.0 * entry["max_s"]
        env.extras["log"][f"RewardProfiler/{name}_call_count"] = float(entry["calls"])
    profiler["last_log_step"] = current_step

    if current_step >= PROFILER_MAX_COMMON_STEPS and not profiler["printed_summary"] and stats:
        ranking = sorted(
            (
                name,
                1000.0 * entry["cum_s"] / max(entry["calls"], 1.0),
                1000.0 * entry["cum_s"],
                int(entry["calls"]),
            )
            for name, entry in stats.items()
        )
        ranking = sorted(ranking, key=lambda item: item[1], reverse=True)[:PROFILER_PRINT_TOP_K]
        print("[K1Kick RewardProfiler] avg-ms ranking over profiled steps:")
        for rank, (name, avg_ms, cum_ms, calls) in enumerate(ranking, start=1):
            print(f"  {rank}. {name}: avg={avg_ms:.4f} ms, cum={cum_ms:.2f} ms, calls={calls}")
        profiler["printed_summary"] = True


def _profile_named_call(env: ManagerBasedRLEnv, name: str, fn, *args, **kwargs):
    if not _profile_step_enabled(env):
        return fn(*args, **kwargs)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    _record_profile_stat(env, name, time.perf_counter() - start)
    _flush_profile_stats(env)
    return result


def _profile_term(name: str):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(env: ManagerBasedRLEnv, *args, **kwargs):
            return _profile_named_call(env, name, fn, env, *args, **kwargs)

        return wrapper

    return decorator


def get_curriculum_stage(env: ManagerBasedRLEnv) -> int:
    return int(_ensure_curriculum_buffers(env)["curriculum_stage"])


def set_curriculum_stage(env: ManagerBasedRLEnv, stage: int) -> None:
    _ensure_curriculum_buffers(env)["curriculum_stage"] = int(stage)


def _stage_active(env: ManagerBasedRLEnv, min_stage: int) -> bool:
    return get_curriculum_stage(env) >= min_stage


def _zero_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def _ball_spawn_world(env: ManagerBasedRLEnv, ball_spawn_pos: tuple[float, float, float]) -> torch.Tensor:
    spawn = torch.tensor(ball_spawn_pos, device=env.device, dtype=torch.float32)
    return env.scene.env_origins + spawn.unsqueeze(0)


def _episode_reset_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    # ManagerBasedRLEnv increments episode_length_buf before reward/termination terms run.
    # Therefore the first compute step of every new episode appears as length == 1.
    return env.episode_length_buf <= 1


def _time_out_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def _ball_contact_mask(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    left_sensor_name: str = "ball_contact_left_foot",
    right_sensor_name: str = "ball_contact_right_foot",
) -> torch.Tensor:
    left_contact, right_contact = _ball_contact_sides(env, threshold, left_sensor_name, right_sensor_name)
    return left_contact | right_contact


def _ball_contact_sides(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    left_sensor_name: str = "ball_contact_left_foot",
    right_sensor_name: str = "ball_contact_right_foot",
) -> tuple[torch.Tensor, torch.Tensor]:
    left_contact = _ball_contact_force(env, left_sensor_name) > threshold
    right_contact = _ball_contact_force(env, right_sensor_name) > threshold
    return left_contact, right_contact


def _ball_planar_speed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    ball: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm(ball.data.root_lin_vel_w[:, :2], dim=1)


def _ball_forward_distance(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    ball: RigidObject = env.scene[asset_cfg.name]
    spawn_pos = _ball_spawn_world(env, ball_spawn_pos)
    return ball.data.root_pos_w[:, 0] - spawn_pos[:, 0]


def _ball_forward_speed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    ball: RigidObject = env.scene[asset_cfg.name]
    return torch.clamp(ball.data.root_lin_vel_w[:, 0], min=0.0)


def _update_episode_ball_metrics(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> tuple[torch.Tensor, torch.Tensor]:
    buffers = _ensure_curriculum_buffers(env)
    reset_mask = _episode_reset_mask(env)
    peak_speed = buffers["peak_ball_forward_speed"]
    peak_distance = buffers["peak_ball_forward_distance"]
    assert isinstance(peak_speed, torch.Tensor)
    assert isinstance(peak_distance, torch.Tensor)
    peak_speed[reset_mask] = 0.0
    peak_distance[reset_mask] = 0.0
    current_speed = _ball_forward_speed(env, asset_cfg)
    current_distance = torch.clamp(_ball_forward_distance(env, ball_spawn_pos, asset_cfg), min=0.0)
    peak_speed[:] = torch.maximum(peak_speed, current_speed)
    peak_distance[:] = torch.maximum(peak_distance, current_distance)
    return peak_speed, peak_distance


def _body_contact_mask(
    env: ManagerBasedRLEnv,
    body_names: tuple[str, ...],
    sensor_name: str = "contact_forces",
    force_threshold: float = 5.0,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    body_ids = [sensor.find_bodies(name, preserve_order=True)[0][0] for name in body_names]
    contact_force = sensor.data.net_forces_w_history[:, :, body_ids, :].norm(dim=-1).max(dim=1)[0]
    return contact_force > force_threshold


def _double_support_mask(
    env: ManagerBasedRLEnv,
    foot_height_threshold: float = 0.18,
    foot_vertical_speed_threshold: float = 0.8,
    contact_force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    foot_ids = [robot.find_bodies(name)[0][0] for name in ("left_foot_link", "right_foot_link")]
    foot_heights = robot.data.body_pos_w[:, foot_ids, 2]
    foot_vertical_speeds = torch.abs(robot.data.body_lin_vel_w[:, foot_ids, 2])
    support_from_height = torch.all(foot_heights <= foot_height_threshold, dim=1) & torch.all(
        foot_vertical_speeds <= foot_vertical_speed_threshold, dim=1
    )
    support_from_contact = torch.all(
        _body_contact_mask(env, ("left_foot_link", "right_foot_link"), force_threshold=contact_force_threshold), dim=1
    )
    return support_from_height | support_from_contact


def _foot_contact_sides(
    env: ManagerBasedRLEnv,
    sensor_name: str = "contact_forces",
    force_threshold: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    contact = _body_contact_mask(env, ("left_foot_link", "right_foot_link"), sensor_name, force_threshold)
    return contact[:, 0], contact[:, 1]


def _foot_pose_metrics(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    robot: Articulation = env.scene[asset_cfg.name]
    left_foot_id = robot.find_bodies("left_foot_link")[0][0]
    right_foot_id = robot.find_bodies("right_foot_link")[0][0]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_id, :]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_id, :]
    return {
        "left_x": left_foot_pos[:, 0],
        "right_x": right_foot_pos[:, 0],
        "left_y": left_foot_pos[:, 1],
        "right_y": right_foot_pos[:, 1],
        "left_z": left_foot_pos[:, 2],
        "right_z": right_foot_pos[:, 2],
    }


def _root_roll_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    return roll, pitch


def _com_in_support_region_mask(
    env: ManagerBasedRLEnv,
    support_margin_x: float = 0.03,
    support_margin_y: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    left_rel_x, right_rel_x, left_rel_y, right_rel_y = _foot_positions_relative_to_com(env, asset_cfg)
    min_x = torch.minimum(left_rel_x, right_rel_x) - support_margin_x
    max_x = torch.maximum(left_rel_x, right_rel_x) + support_margin_x
    min_y = torch.minimum(left_rel_y, right_rel_y) - support_margin_y
    max_y = torch.maximum(left_rel_y, right_rel_y) + support_margin_y
    return (min_x <= 0.0) & (max_x >= 0.0) & (min_y <= 0.0) & (max_y >= 0.0)


def _quat_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    _, _, yaw = euler_xyz_from_quat(quat_wxyz)
    return yaw


def _get_initial_heading(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    heading = _quat_to_yaw(robot.data.root_quat_w).clone()

    kick_buffers = _ensure_kick_buffers(env)
    if "initial_heading" not in kick_buffers:
        kick_buffers["initial_heading"] = heading.clone()

    reset_mask = _episode_reset_mask(env)
    kick_buffers["initial_heading"][reset_mask] = heading[reset_mask]
    return kick_buffers["initial_heading"]


def _heading_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    current_heading = _quat_to_yaw(robot.data.root_quat_w)
    initial_heading = _get_initial_heading(env, asset_cfg)
    return torch.abs(wrap_to_pi(current_heading - initial_heading))


def _get_initial_base_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    base_xy = robot.data.root_pos_w[:, :2].clone()

    kick_buffers = _ensure_kick_buffers(env)
    if "initial_base_xy" not in kick_buffers:
        kick_buffers["initial_base_xy"] = base_xy.clone()

    reset_mask = _episode_reset_mask(env)
    kick_buffers["initial_base_xy"][reset_mask] = base_xy[reset_mask]
    return kick_buffers["initial_base_xy"]


def _base_step_offsets(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[asset_cfg.name]
    initial_base_xy = _get_initial_base_xy(env, asset_cfg)
    delta_xy = robot.data.root_pos_w[:, :2] - initial_base_xy
    return delta_xy[:, 0], delta_xy[:, 1]


def _get_initial_foot_pos_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[asset_cfg.name]
    left_foot_id = robot.find_bodies("left_foot_link")[0][0]
    right_foot_id = robot.find_bodies("right_foot_link")[0][0]
    left_pos = robot.data.body_pos_w[:, left_foot_id, :2].clone()
    right_pos = robot.data.body_pos_w[:, right_foot_id, :2].clone()

    kick_buffers = _ensure_kick_buffers(env)
    if "initial_left_foot_xy" not in kick_buffers:
        kick_buffers["initial_left_foot_xy"] = left_pos.clone()
    if "initial_right_foot_xy" not in kick_buffers:
        kick_buffers["initial_right_foot_xy"] = right_pos.clone()

    reset_mask = _episode_reset_mask(env)
    kick_buffers["initial_left_foot_xy"][reset_mask] = left_pos[reset_mask]
    kick_buffers["initial_right_foot_xy"][reset_mask] = right_pos[reset_mask]
    return kick_buffers["initial_left_foot_xy"], kick_buffers["initial_right_foot_xy"]


def _foot_step_offsets(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[asset_cfg.name]
    left_foot_id = robot.find_bodies("left_foot_link")[0][0]
    right_foot_id = robot.find_bodies("right_foot_link")[0][0]
    left_initial, right_initial = _get_initial_foot_pos_xy(env, asset_cfg)
    left_delta = robot.data.body_pos_w[:, left_foot_id, :2] - left_initial
    right_delta = robot.data.body_pos_w[:, right_foot_id, :2] - right_initial
    return left_delta[:, 0], right_delta[:, 0], left_delta[:, 1], right_delta[:, 1]


def _foot_positions_relative_to_com(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[asset_cfg.name]
    left_foot_id = robot.find_bodies("left_foot_link")[0][0]
    right_foot_id = robot.find_bodies("right_foot_link")[0][0]
    com_xy = robot.data.root_pos_w[:, :2]
    left_rel_w = robot.data.body_pos_w[:, left_foot_id, :2] - com_xy
    right_rel_w = robot.data.body_pos_w[:, right_foot_id, :2] - com_xy
    _, _, base_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    cos_yaw = torch.cos(-base_yaw)
    sin_yaw = torch.sin(-base_yaw)
    left_rel_x = cos_yaw * left_rel_w[:, 0] - sin_yaw * left_rel_w[:, 1]
    left_rel_y = sin_yaw * left_rel_w[:, 0] + cos_yaw * left_rel_w[:, 1]
    right_rel_x = cos_yaw * right_rel_w[:, 0] - sin_yaw * right_rel_w[:, 1]
    right_rel_y = sin_yaw * right_rel_w[:, 0] + cos_yaw * right_rel_w[:, 1]
    return left_rel_x, right_rel_x, left_rel_y, right_rel_y


def _support_foot_relative_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kick_sign = _kick_foot_sign(env)
    left_rel_x, right_rel_x, left_rel_y, right_rel_y = _foot_positions_relative_to_com(env, asset_cfg)
    support_rel_x = torch.where(kick_sign < 0, right_rel_x, left_rel_x)
    support_rel_y = torch.where(kick_sign < 0, right_rel_y, left_rel_y)
    valid_mask = kick_sign != 0
    return support_rel_x, support_rel_y, valid_mask


def _stance_x_separation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    left_rel_x, right_rel_x, _, _ = _foot_positions_relative_to_com(env, asset_cfg)
    return torch.abs(left_rel_x - right_rel_x)


def _stance_width_y(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    _, _, left_rel_y, right_rel_y = _foot_positions_relative_to_com(env, asset_cfg)
    return torch.abs(left_rel_y - right_rel_y)


def _foot_distance_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    left_rel_x, right_rel_x, left_rel_y, right_rel_y = _foot_positions_relative_to_com(env, asset_cfg)
    return torch.sqrt(torch.square(left_rel_x - right_rel_x) + torch.square(left_rel_y - right_rel_y))


def _leg_distance_body_ids(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    kick_buffers = _ensure_kick_buffers(env)
    cache_key = f"leg_distance_body_ids::{asset_cfg.name}"
    if cache_key not in kick_buffers:
        robot: Articulation = env.scene[asset_cfg.name]
        # K1 has no dedicated toe link, so the foot link is used as the terminal leg-link proxy.
        left_ids = [robot.find_bodies(name)[0][0] for name in K1_LEFT_LEG_BODY_NAMES]
        right_ids = [robot.find_bodies(name)[0][0] for name in K1_RIGHT_LEG_BODY_NAMES]
        kick_buffers[cache_key] = (
            torch.tensor(left_ids, device=env.device, dtype=torch.long),
            torch.tensor(right_ids, device=env.device, dtype=torch.long),
        )
    return kick_buffers[cache_key]


def _leg_pair_labels() -> list[str]:
    return [f"{left}__{right}" for left in K1_LEFT_LEG_BODY_NAMES for right in K1_RIGHT_LEG_BODY_NAMES]


def _pair_label_metric_suffix(label: str) -> str:
    return label.replace(".", "_").replace("/", "_")


def _same_leg_pair_labels() -> tuple[tuple[str, str], ...]:
    return (
        ("hip_link_distance", "Left_Hip_Yaw__Right_Hip_Yaw"),
        ("shank_distance", "Left_Shank__Right_Shank"),
        ("ankle_distance", "Left_Ankle_Cross__Right_Ankle_Cross"),
        ("foot_link_distance", "left_foot_link__right_foot_link"),
    )


def _min_leg_link_info(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = ("min_leg_link_info", asset_cfg.name)
    step_cache = _get_step_cache(env)
    if cache_key in step_cache:
        cached = step_cache[cache_key]
        return tuple(item.clone() for item in cached)

    robot: Articulation = env.scene[asset_cfg.name]
    left_ids, right_ids = _leg_distance_body_ids(env, asset_cfg)
    left_body_pos_w = robot.data.body_pos_w.index_select(1, left_ids)
    right_body_pos_w = robot.data.body_pos_w.index_select(1, right_ids)
    pairwise_distance = torch.linalg.norm(left_body_pos_w.unsqueeze(2) - right_body_pos_w.unsqueeze(1), dim=-1)
    flat_distance = pairwise_distance.view(env.num_envs, -1)
    min_distance, min_pair_index = torch.min(flat_distance, dim=1)
    result = (min_distance.clone(), min_pair_index.clone(), pairwise_distance.clone())
    step_cache[cache_key] = result
    return result


def _min_leg_link_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    min_distance, _, _ = _min_leg_link_info(env, asset_cfg)
    return min_distance


def _leg_self_contact_info(
    env: ManagerBasedRLEnv,
    sensor_name: str = "leg_self_contact_left",
    contact_force_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = ("leg_self_contact_info", sensor_name, float(contact_force_threshold))
    step_cache = _get_step_cache(env)
    if cache_key in step_cache:
        cached = step_cache[cache_key]
        return tuple(item.clone() for item in cached)

    sensor = env.scene.sensors[sensor_name]
    force_matrix_w = sensor.data.force_matrix_w
    if force_matrix_w is None:
        num_pairs = len(K1_LEFT_LEG_BODY_NAMES) * len(K1_RIGHT_LEG_BODY_NAMES)
        pair_contact_mask = torch.zeros((env.num_envs, num_pairs), device=env.device, dtype=torch.bool)
        pair_force_norm = torch.zeros((env.num_envs, num_pairs), device=env.device)
    else:
        left_body_indices = [sensor.find_bodies(name, preserve_order=True)[0][0] for name in K1_LEFT_LEG_BODY_NAMES]
        ordered_force_matrix_w = force_matrix_w.index_select(
            1, torch.tensor(left_body_indices, device=env.device, dtype=torch.long)
        )
        pair_force_norm = torch.linalg.norm(ordered_force_matrix_w, dim=-1).view(env.num_envs, -1)
        pair_contact_mask = pair_force_norm > contact_force_threshold
    any_contact = torch.any(pair_contact_mask, dim=1)
    contact_count = torch.sum(pair_contact_mask, dim=1)
    result = (any_contact.clone(), contact_count.clone(), pair_contact_mask.clone(), pair_force_norm.clone())
    step_cache[cache_key] = result
    return result


def _set_logged_metric(env: ManagerBasedRLEnv, key: str, value: float) -> None:
    if not hasattr(env, "extras"):
        return
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"][key] = float(value)


def _log_mask_rate(env: ManagerBasedRLEnv, key: str, mask: torch.Tensor) -> None:
    _set_logged_metric(env, key, torch.mean(mask.float()).item())


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.numel() == 0 or mask.numel() == 0 or not torch.any(mask):
        return 0.0
    return torch.mean(values[mask].float()).item()


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.numel() == 0 or mask.numel() == 0 or not torch.any(mask):
        return 0.0
    return torch.max(values[mask].float()).item()


def _selected_correlation(values_a: torch.Tensor, values_b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        if mask.numel() == 0 or not torch.any(mask):
            return 0.0
        values_a = values_a[mask]
        values_b = values_b[mask]
    if values_a.numel() < 2 or values_b.numel() < 2:
        return 0.0
    a = values_a.float()
    b = values_b.float()
    a_centered = a - torch.mean(a)
    b_centered = b - torch.mean(b)
    denom = torch.sqrt(torch.sum(a_centered * a_centered) * torch.sum(b_centered * b_centered))
    if torch.isnan(denom) or denom <= 1.0e-8:
        return 0.0
    corr = torch.sum(a_centered * b_centered) / denom
    return corr.item()


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return torch.quantile(values.float(), q).item()


def _paired_body_relative_y_in_heading_frame(
    env: ManagerBasedRLEnv,
    left_body_name: str,
    right_body_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[asset_cfg.name]
    left_id = robot.find_bodies(left_body_name)[0][0]
    right_id = robot.find_bodies(right_body_name)[0][0]
    com_xy = robot.data.root_pos_w[:, :2]
    left_rel_w = robot.data.body_pos_w[:, left_id, :2] - com_xy
    right_rel_w = robot.data.body_pos_w[:, right_id, :2] - com_xy
    _, _, base_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    cos_yaw = torch.cos(-base_yaw)
    sin_yaw = torch.sin(-base_yaw)
    left_rel_y = sin_yaw * left_rel_w[:, 0] + cos_yaw * left_rel_w[:, 1]
    right_rel_y = sin_yaw * right_rel_w[:, 0] + cos_yaw * right_rel_w[:, 1]
    return left_rel_y, right_rel_y


def _log_leg_collision_metrics(
    env: ManagerBasedRLEnv,
    min_leg_link_distance: torch.Tensor,
    min_pair_index: torch.Tensor,
    pairwise_distance: torch.Tensor,
    any_contact: torch.Tensor,
    contact_count: torch.Tensor,
    pair_contact_mask: torch.Tensor,
    pair_force_norm: torch.Tensor,
    log_detailed_pair_metrics: bool = False,
) -> None:
    pair_labels = _leg_pair_labels()
    _set_logged_metric(env, "Metrics/min_leg_link_distance", torch.min(min_leg_link_distance).item())
    _set_logged_metric(env, "Metrics/min_leg_link_distance_p10", _quantile(min_leg_link_distance, 0.10))
    _set_logged_metric(env, "Metrics/min_leg_link_distance_p05", _quantile(min_leg_link_distance, 0.05))
    _set_logged_metric(env, "Metrics/min_leg_link_distance_p01", _quantile(min_leg_link_distance, 0.01))
    _set_logged_metric(env, "Metrics/leg_self_contact_rate", torch.mean(any_contact.float()).item())
    _set_logged_metric(env, "Metrics/leg_self_contact_count", torch.mean(contact_count.float()).item())
    if pair_force_norm.numel() > 0:
        active_force_values = pair_force_norm[pair_contact_mask]
        _set_logged_metric(
            env, "Metrics/leg_self_contact_force", torch.mean(active_force_values).item() if active_force_values.numel() > 0 else 0.0
        )
        _set_logged_metric(env, "Metrics/max_leg_self_contact_force", torch.max(pair_force_norm).item())
    same_pair_names = _same_leg_pair_labels()
    pair_label_to_index = {label: idx for idx, label in enumerate(pair_labels)}
    flat_distance = pairwise_distance.view(env.num_envs, -1)
    for metric_name, pair_label in same_pair_names:
        pair_index = pair_label_to_index[pair_label]
        pair_values = flat_distance[:, pair_index]
        _set_logged_metric(env, f"Metrics/{metric_name}", torch.mean(pair_values).item())
    left_foot_y, right_foot_y = _paired_body_relative_y_in_heading_frame(env, "left_foot_link", "right_foot_link")
    left_ankle_y, right_ankle_y = _paired_body_relative_y_in_heading_frame(env, "Left_Ankle_Cross", "Right_Ankle_Cross")
    crossed_feet = left_foot_y < right_foot_y
    crossed_ankle = left_ankle_y < right_ankle_y
    crossed_leg = crossed_feet & crossed_ankle
    _set_logged_metric(env, "Metrics/crossed_feet_rate", torch.mean(crossed_feet.float()).item())
    _set_logged_metric(env, "Metrics/crossed_ankle_rate", torch.mean(crossed_ankle.float()).item())
    _set_logged_metric(env, "Metrics/crossed_leg_rate", torch.mean(crossed_leg.float()).item())
    if min_pair_index.numel() > 0:
        pair_index_counts = torch.bincount(min_pair_index.to(dtype=torch.long), minlength=len(pair_labels))
        dominant_min_pair_index = int(torch.argmax(pair_index_counts).item())
        _set_logged_metric(env, "Metrics/min_leg_link_pair", float(dominant_min_pair_index))
        if log_detailed_pair_metrics:
            for pair_index, pair_label in enumerate(pair_labels):
                pair_mask = min_pair_index == pair_index
                _set_logged_metric(
                    env,
                    f"Metrics/min_leg_link_pair_rate/{_pair_label_metric_suffix(pair_label)}",
                    torch.mean(pair_mask.float()).item(),
                )
                pair_values = flat_distance[:, pair_index]
                pair_suffix = _pair_label_metric_suffix(pair_label)
                _set_logged_metric(env, f"Metrics/leg_link_distance/{pair_suffix}_mean", torch.mean(pair_values).item())
                _set_logged_metric(env, f"Metrics/leg_link_distance/{pair_suffix}_min", torch.min(pair_values).item())
                _set_logged_metric(env, f"Metrics/leg_link_distance/{pair_suffix}_p05", _quantile(pair_values, 0.05))
                _set_logged_metric(env, f"Metrics/leg_link_distance/{pair_suffix}_p01", _quantile(pair_values, 0.01))
    if log_detailed_pair_metrics and pair_contact_mask.numel() > 0:
        for pair_index, pair_label in enumerate(pair_labels):
            _set_logged_metric(
                env,
                f"Metrics/leg_self_contact_pair_rate/{_pair_label_metric_suffix(pair_label)}",
                torch.mean(pair_contact_mask[:, pair_index].float()).item(),
            )


def _update_support_foot_metrics(
    env: ManagerBasedRLEnv,
    support_foot_x_drift: torch.Tensor,
    support_valid_mask: torch.Tensor,
    stance_x_separation: torch.Tensor,
    stance_width_y: torch.Tensor,
    foot_distance_xy: torch.Tensor,
    min_leg_link_distance: torch.Tensor | None = None,
) -> None:
    if torch.any(support_valid_mask):
        drift_values = support_foot_x_drift[support_valid_mask]
        stance_values = stance_x_separation[support_valid_mask]
        stance_width_values = stance_width_y[support_valid_mask]
        foot_distance_values = foot_distance_xy[support_valid_mask]
        _set_logged_metric(env, "Metrics/support_foot_x_drift_mean", torch.mean(drift_values).item())
        _set_logged_metric(env, "Metrics/support_foot_x_drift_max", torch.max(drift_values).item())
        _set_logged_metric(env, "Metrics/stance_x_separation_mean", torch.mean(stance_values).item())
        _set_logged_metric(env, "Metrics/stance_x_separation_max", torch.max(stance_values).item())
        _set_logged_metric(env, "Metrics/stance_width_y_mean", torch.mean(stance_width_values).item())
        _set_logged_metric(env, "Metrics/stance_width_y_max", torch.max(stance_width_values).item())
        _set_logged_metric(env, "Metrics/stance_width_y_min", torch.min(stance_width_values).item())
        _set_logged_metric(env, "Metrics/stance_width_y_p50", _quantile(stance_width_values, 0.50))
        _set_logged_metric(env, "Metrics/stance_width_y_p90", _quantile(stance_width_values, 0.90))
        _set_logged_metric(env, "Metrics/stance_width_y_p95", _quantile(stance_width_values, 0.95))
        _set_logged_metric(env, "Metrics/narrow_stance_rate", torch.mean((stance_width_values < 0.10).float()).item())
        _set_logged_metric(env, "Metrics/wide_stance_rate", torch.mean((stance_width_values > 0.25).float()).item())
        _set_logged_metric(env, "Metrics/foot_distance_xy_mean", torch.mean(foot_distance_values).item())
        _set_logged_metric(env, "Metrics/foot_distance_xy_min", torch.min(foot_distance_values).item())
    else:
        _set_logged_metric(env, "Metrics/support_foot_x_drift_mean", 0.0)
        _set_logged_metric(env, "Metrics/support_foot_x_drift_max", 0.0)
        _set_logged_metric(env, "Metrics/stance_x_separation_mean", 0.0)
        _set_logged_metric(env, "Metrics/stance_x_separation_max", 0.0)
        _set_logged_metric(env, "Metrics/stance_width_y_mean", 0.0)
        _set_logged_metric(env, "Metrics/stance_width_y_max", 0.0)
        _set_logged_metric(env, "Metrics/stance_width_y_min", 0.0)
        _set_logged_metric(env, "Metrics/stance_width_y_p50", 0.0)
        _set_logged_metric(env, "Metrics/stance_width_y_p90", 0.0)
        _set_logged_metric(env, "Metrics/stance_width_y_p95", 0.0)
        _set_logged_metric(env, "Metrics/narrow_stance_rate", 0.0)
        _set_logged_metric(env, "Metrics/wide_stance_rate", 0.0)
        _set_logged_metric(env, "Metrics/foot_distance_xy_mean", 0.0)
        _set_logged_metric(env, "Metrics/foot_distance_xy_min", 0.0)
    if min_leg_link_distance is not None:
        _set_logged_metric(env, "Metrics/min_leg_link_distance", torch.min(min_leg_link_distance).item())


def _support_foot_x_drift(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    kick_buffers = _ensure_kick_buffers(env)
    if "support_foot_initial_rel_x" not in kick_buffers:
        kick_buffers["support_foot_initial_rel_x"] = torch.zeros(env.num_envs, device=env.device)
    if "support_foot_initial_rel_y" not in kick_buffers:
        kick_buffers["support_foot_initial_rel_y"] = torch.zeros(env.num_envs, device=env.device)

    support_foot_x, _, valid_mask = _support_foot_relative_position(env, asset_cfg)
    initial_support_x = kick_buffers["support_foot_initial_rel_x"]
    drift = torch.abs(support_foot_x - initial_support_x)
    return drift, valid_mask


def _update_ball_contact_state(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    left_sensor_name: str = "ball_contact_left_foot",
    right_sensor_name: str = "ball_contact_right_foot",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = ("ball_contact_state", float(threshold), left_sensor_name, right_sensor_name)
    step_cache = _get_step_cache(env)
    if cache_key in step_cache:
        cached = step_cache[cache_key]
        return tuple(item.clone() for item in cached)

    kick_buffers = _ensure_kick_buffers(env)
    if "had_ball_contact" not in kick_buffers:
        kick_buffers["had_ball_contact"] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if "kick_foot_sign" not in kick_buffers:
        kick_buffers["kick_foot_sign"] = torch.zeros(env.num_envs, device=env.device, dtype=torch.int8)

    had_ball_contact = kick_buffers["had_ball_contact"]
    kick_foot_sign = kick_buffers["kick_foot_sign"]
    if "support_foot_initial_rel_x" not in kick_buffers:
        kick_buffers["support_foot_initial_rel_x"] = torch.zeros(env.num_envs, device=env.device)
    if "support_foot_initial_rel_y" not in kick_buffers:
        kick_buffers["support_foot_initial_rel_y"] = torch.zeros(env.num_envs, device=env.device)
    support_foot_initial_rel_x = kick_buffers["support_foot_initial_rel_x"]
    support_foot_initial_rel_y = kick_buffers["support_foot_initial_rel_y"]
    reset_mask = _episode_reset_mask(env)
    had_ball_contact[reset_mask] = False
    kick_foot_sign[reset_mask] = 0
    support_foot_initial_rel_x[reset_mask] = 0.0
    support_foot_initial_rel_y[reset_mask] = 0.0

    left_contact, right_contact = _ball_contact_sides(env, threshold, left_sensor_name, right_sensor_name)
    current_contact = left_contact | right_contact
    first_contact = current_contact & ~had_ball_contact
    if torch.any(first_contact):
        robot: Articulation = env.scene["robot"]
        left_foot_id = robot.find_bodies("left_foot_link")[0][0]
        right_foot_id = robot.find_bodies("right_foot_link")[0][0]
        left_x = robot.data.body_pos_w[:, left_foot_id, 0]
        right_x = robot.data.body_pos_w[:, right_foot_id, 0]
        both_contact = left_contact & right_contact & first_contact
        kick_foot_sign[first_contact & left_contact & ~right_contact] = -1
        kick_foot_sign[first_contact & right_contact & ~left_contact] = 1
        kick_foot_sign[both_contact] = torch.where(left_x[both_contact] >= right_x[both_contact], -1, 1).to(torch.int8)
        left_rel_x, right_rel_x, left_rel_y, right_rel_y = _foot_positions_relative_to_com(env)
        support_foot_initial_rel_x[first_contact] = torch.where(
            kick_foot_sign[first_contact] < 0, right_rel_x[first_contact], left_rel_x[first_contact]
        )
        support_foot_initial_rel_y[first_contact] = torch.where(
            kick_foot_sign[first_contact] < 0, right_rel_y[first_contact], left_rel_y[first_contact]
        )
    had_ball_contact |= current_contact
    result = (current_contact.clone(), first_contact.clone(), had_ball_contact.clone())
    step_cache[cache_key] = result
    return result


def _had_ball_contact_mask(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    left_sensor_name: str = "ball_contact_left_foot",
    right_sensor_name: str = "ball_contact_right_foot",
) -> torch.Tensor:
    _, _, had_ball_contact = _update_ball_contact_state(env, threshold, left_sensor_name, right_sensor_name)
    return had_ball_contact


def _get_ball_contact_steps(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    left_sensor_name: str = "ball_contact_left_foot",
    right_sensor_name: str = "ball_contact_right_foot",
) -> torch.Tensor:
    kick_buffers = _ensure_kick_buffers(env)
    if "ball_contact_step" not in kick_buffers:
        kick_buffers["ball_contact_step"] = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    contact_steps = kick_buffers["ball_contact_step"]
    reset_mask = _episode_reset_mask(env)
    contact_steps[reset_mask] = -1

    _, first_contact, _ = _update_ball_contact_state(env, threshold, left_sensor_name, right_sensor_name)
    contact_steps[first_contact] = env.episode_length_buf[first_contact].long()
    return contact_steps


def _ball_contact_elapsed_time(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    left_sensor_name: str = "ball_contact_left_foot",
    right_sensor_name: str = "ball_contact_right_foot",
) -> tuple[torch.Tensor, torch.Tensor]:
    contact_steps = _get_ball_contact_steps(env, threshold, left_sensor_name, right_sensor_name)
    contact_mask = contact_steps >= 0
    elapsed_s = (env.episode_length_buf.long() - contact_steps).float() * env.step_dt
    elapsed_s = torch.where(contact_mask, elapsed_s, torch.zeros_like(elapsed_s))
    return contact_mask, elapsed_s


def _kick_foot_sign(env: ManagerBasedRLEnv) -> torch.Tensor:
    kick_buffers = _ensure_kick_buffers(env)
    if "kick_foot_sign" not in kick_buffers:
        _update_ball_contact_state(env)
    return kick_buffers["kick_foot_sign"].clone()


def _update_kick_success_state(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = (
        "kick_success_state",
        tuple(ball_spawn_pos),
        float(success_distance),
        float(success_speed),
        float(min_success_distance_for_speed),
        asset_cfg.name,
    )
    step_cache = _get_step_cache(env)
    if cache_key in step_cache:
        cached = step_cache[cache_key]
        return tuple(item.clone() for item in cached)

    kick_buffers = _ensure_kick_buffers(env)
    if "had_kick_success" not in kick_buffers:
        kick_buffers["had_kick_success"] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    had_kick_success = kick_buffers["had_kick_success"]
    reset_mask = _episode_reset_mask(env)
    had_kick_success[reset_mask] = False

    _, _, had_ball_contact = _update_ball_contact_state(env)
    forward_distance = _ball_forward_distance(env, ball_spawn_pos, asset_cfg)
    forward_speed = _ball_forward_speed(env, asset_cfg)
    current_success = had_ball_contact & (
        (forward_distance >= success_distance)
        | ((forward_distance >= min_success_distance_for_speed) & (forward_speed >= success_speed))
        | (forward_speed >= success_speed)
    )
    first_success = current_success & ~had_kick_success
    had_kick_success |= current_success
    result = (current_success.clone(), first_success.clone(), had_kick_success.clone())
    step_cache[cache_key] = result
    return result


def _get_kick_success_steps(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    kick_buffers = _ensure_kick_buffers(env)
    if "kick_success_step" not in kick_buffers:
        kick_buffers["kick_success_step"] = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    success_steps = kick_buffers["kick_success_step"]
    reset_mask = _episode_reset_mask(env)
    success_steps[reset_mask] = -1

    _, first_success, _ = _update_kick_success_state(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, asset_cfg
    )
    success_steps[first_success] = env.episode_length_buf[first_success].long()
    return success_steps


def _post_success_mask(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    _, _, had_kick_success = _update_kick_success_state(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, asset_cfg
    )
    return had_kick_success


def _success_elapsed_time(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> tuple[torch.Tensor, torch.Tensor]:
    success_steps = _get_kick_success_steps(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, asset_cfg
    )
    success_mask = success_steps >= 0
    elapsed_s = (env.episode_length_buf.long() - success_steps).float() * env.step_dt
    elapsed_s = torch.where(success_mask, elapsed_s, torch.zeros_like(elapsed_s))
    return success_mask, elapsed_s


def _recovery_condition_mask(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    foot_contact_force_threshold: float = 1.0,
    support_margin_x: float = 0.03,
    support_margin_y: float = 0.02,
    max_foot_height_diff: float = 0.08,
    max_foot_height: float = 0.12,
    roll_pitch_threshold: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    cache_key = (
        "recovery_condition_mask",
        tuple(ball_spawn_pos),
        float(success_distance),
        float(success_speed),
        float(min_success_distance_for_speed),
        float(min_recovery_delay_s),
        float(foot_contact_force_threshold),
        float(support_margin_x),
        float(support_margin_y),
        float(max_foot_height_diff),
        float(max_foot_height),
        float(roll_pitch_threshold),
        asset_cfg.name,
    )
    step_cache = _get_step_cache(env)
    if cache_key in step_cache:
        return step_cache[cache_key].clone()
    diagnostics = get_recovery_condition_diagnostics(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        foot_contact_force_threshold,
        support_margin_x,
        support_margin_y,
        max_foot_height_diff,
        max_foot_height,
        roll_pitch_threshold,
        asset_cfg,
    )
    result = diagnostics["recovery_condition_mask"].clone()
    step_cache[cache_key] = result
    return result


def get_recovery_condition_diagnostics(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    foot_contact_force_threshold: float = 1.0,
    support_margin_x: float = 0.03,
    support_margin_y: float = 0.02,
    max_foot_height_diff: float = 0.08,
    max_foot_height: float = 0.12,
    roll_pitch_threshold: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    cache_key = (
        "recovery_condition_diagnostics",
        tuple(ball_spawn_pos),
        float(success_distance),
        float(success_speed),
        float(min_success_distance_for_speed),
        float(min_recovery_delay_s),
        float(foot_contact_force_threshold),
        float(support_margin_x),
        float(support_margin_y),
        float(max_foot_height_diff),
        float(max_foot_height),
        float(roll_pitch_threshold),
        asset_cfg.name,
    )
    step_cache = _get_step_cache(env)
    if cache_key in step_cache:
        return step_cache[cache_key]
    recovery_phase = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    robot: Articulation = env.scene[asset_cfg.name]
    foot_pose = _foot_pose_metrics(env, asset_cfg)
    double_support = _double_support_mask(
        env,
        foot_height_threshold=max_foot_height,
        foot_vertical_speed_threshold=1.0,
        contact_force_threshold=foot_contact_force_threshold,
        asset_cfg=asset_cfg,
    )
    feet_level = torch.abs(foot_pose["left_z"] - foot_pose["right_z"]) <= max_foot_height_diff
    # K1 has no dedicated toe link, so foot-link height is used as the toe-height proxy.
    both_feet_low = (foot_pose["left_z"] <= max_foot_height) & (foot_pose["right_z"] <= max_foot_height)
    com_in_support = _com_in_support_region_mask(env, support_margin_x, support_margin_y, asset_cfg)
    roll, pitch = _root_roll_pitch(env, asset_cfg)
    upright = (torch.abs(roll) <= roll_pitch_threshold) & (torch.abs(pitch) <= roll_pitch_threshold)
    left_rel_x, right_rel_x, left_rel_y, right_rel_y = _foot_positions_relative_to_com(env, asset_cfg)
    recovery_condition_mask = recovery_phase & double_support & com_in_support & feet_level & both_feet_low & upright
    support_min_x = torch.minimum(left_rel_x, right_rel_x) - support_margin_x
    support_max_x = torch.maximum(left_rel_x, right_rel_x) + support_margin_x
    support_min_y = torch.minimum(left_rel_y, right_rel_y) - support_margin_y
    support_max_y = torch.maximum(left_rel_y, right_rel_y) + support_margin_y
    com_support_violation_x = torch.maximum(torch.relu(support_min_x), torch.relu(-support_max_x))
    com_support_violation_y = torch.maximum(torch.relu(support_min_y), torch.relu(-support_max_y))
    com_support_violation = torch.sqrt(torch.square(com_support_violation_x) + torch.square(com_support_violation_y))
    upright_error_deg = torch.rad2deg(torch.maximum(torch.abs(roll), torch.abs(pitch)))
    stance_width_y = _stance_width_y(env, asset_cfg)
    stance_x_separation = _stance_x_separation(env, asset_cfg)
    max_foot_height_value = torch.maximum(foot_pose["left_z"], foot_pose["right_z"])
    result = {
        "recovery_phase": recovery_phase,
        "double_support": double_support,
        "com_in_support": com_in_support,
        "feet_level": feet_level,
        "both_feet_low": both_feet_low,
        "upright": upright,
        "recovery_condition_mask": recovery_condition_mask,
        "left_foot_height": foot_pose["left_z"],
        "right_foot_height": foot_pose["right_z"],
        "foot_height_difference": torch.abs(foot_pose["left_z"] - foot_pose["right_z"]),
        "roll": roll,
        "pitch": pitch,
        "upright_error_deg": upright_error_deg,
        "support_polygon_inside": com_in_support,
        "com_support_violation": com_support_violation,
        "com_support_distance": com_support_violation,
        "com_position_x": robot.data.root_pos_w[:, 0],
        "com_position_y": robot.data.root_pos_w[:, 1],
        "left_foot_rel_x": left_rel_x,
        "right_foot_rel_x": right_rel_x,
        "left_foot_rel_y": left_rel_y,
        "right_foot_rel_y": right_rel_y,
        "stance_width_y": stance_width_y,
        "stance_x_separation": stance_x_separation,
        "max_foot_height_value": max_foot_height_value,
        "support_min_x": support_min_x,
        "support_max_x": support_max_x,
        "support_min_y": support_min_y,
        "support_max_y": support_max_y,
    }
    step_cache[cache_key] = result
    return result


def get_recovery_failure_probe(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice | list[int] | None,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    foot_contact_force_threshold: float = 5.0,
    support_margin_x: float = 0.03,
    support_margin_y: float = 0.02,
    max_foot_height_diff: float = 0.05,
    max_foot_height: float = 0.10,
    roll_pitch_threshold: float = 0.20,
    max_items: int | None = 16,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    if env_ids is None:
        index = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif isinstance(env_ids, slice):
        index = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        index = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    diagnostics = get_recovery_condition_diagnostics(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        foot_contact_force_threshold,
        support_margin_x,
        support_margin_y,
        max_foot_height_diff,
        max_foot_height,
        roll_pitch_threshold,
        asset_cfg,
    )
    failure_mask = diagnostics["recovery_phase"] & ~diagnostics["recovery_condition_mask"]
    failure_ids = index[failure_mask[index]]
    if max_items is not None:
        failure_ids = failure_ids[:max_items]
    return {
        "env_ids": failure_ids.detach().cpu(),
        "left_foot_height": diagnostics["left_foot_height"][failure_ids].detach().cpu(),
        "right_foot_height": diagnostics["right_foot_height"][failure_ids].detach().cpu(),
        "foot_height_difference": diagnostics["foot_height_difference"][failure_ids].detach().cpu(),
        "roll": diagnostics["roll"][failure_ids].detach().cpu(),
        "pitch": diagnostics["pitch"][failure_ids].detach().cpu(),
        "support_polygon_inside": diagnostics["support_polygon_inside"][failure_ids].detach().cpu(),
        "com_position_x": diagnostics["com_position_x"][failure_ids].detach().cpu(),
        "com_position_y": diagnostics["com_position_y"][failure_ids].detach().cpu(),
        "left_foot_rel_x": diagnostics["left_foot_rel_x"][failure_ids].detach().cpu(),
        "right_foot_rel_x": diagnostics["right_foot_rel_x"][failure_ids].detach().cpu(),
        "left_foot_rel_y": diagnostics["left_foot_rel_y"][failure_ids].detach().cpu(),
        "right_foot_rel_y": diagnostics["right_foot_rel_y"][failure_ids].detach().cpu(),
        "double_support": diagnostics["double_support"][failure_ids].detach().cpu(),
        "com_in_support": diagnostics["com_in_support"][failure_ids].detach().cpu(),
        "feet_level": diagnostics["feet_level"][failure_ids].detach().cpu(),
        "both_feet_low": diagnostics["both_feet_low"][failure_ids].detach().cpu(),
        "upright": diagnostics["upright"][failure_ids].detach().cpu(),
    }


def _update_recovery_success_state(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    required_hold_time_s: float = 0.25,
    foot_contact_force_threshold: float = 5.0,
    support_margin_x: float = 0.03,
    support_margin_y: float = 0.02,
    max_foot_height_diff: float = 0.05,
    max_foot_height: float = 0.10,
    roll_pitch_threshold: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = (
        "recovery_success_state",
        tuple(ball_spawn_pos),
        float(success_distance),
        float(success_speed),
        float(min_success_distance_for_speed),
        float(min_recovery_delay_s),
        float(required_hold_time_s),
        float(foot_contact_force_threshold),
        float(support_margin_x),
        float(support_margin_y),
        float(max_foot_height_diff),
        float(max_foot_height),
        float(roll_pitch_threshold),
        asset_cfg.name,
    )
    step_cache = _get_step_cache(env)
    if cache_key in step_cache:
        cached = step_cache[cache_key]
        return tuple(item.clone() for item in cached)

    kick_buffers = _ensure_kick_buffers(env)
    if "had_recovery_success" not in kick_buffers:
        kick_buffers["had_recovery_success"] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if "recovery_hold_steps" not in kick_buffers:
        kick_buffers["recovery_hold_steps"] = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    had_recovery_success = kick_buffers["had_recovery_success"]
    recovery_hold_steps = kick_buffers["recovery_hold_steps"]
    reset_mask = _episode_reset_mask(env)
    had_recovery_success[reset_mask] = False
    recovery_hold_steps[reset_mask] = 0

    recovery_conditions_met = _recovery_condition_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        foot_contact_force_threshold,
        support_margin_x,
        support_margin_y,
        max_foot_height_diff,
        max_foot_height,
        roll_pitch_threshold,
        asset_cfg,
    )
    recovery_hold_steps[~recovery_conditions_met] = 0
    recovery_hold_steps[recovery_conditions_met] += 1

    required_hold_steps = max(1, int(math.ceil(required_hold_time_s / env.step_dt)))
    current_success = recovery_conditions_met & (recovery_hold_steps >= required_hold_steps)
    first_success = current_success & ~had_recovery_success
    had_recovery_success |= current_success
    result = (current_success.clone(), first_success.clone(), had_recovery_success.clone())
    step_cache[cache_key] = result
    return result


def _get_recovery_success_steps(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    required_hold_time_s: float = 0.25,
    foot_contact_force_threshold: float = 5.0,
    support_margin_x: float = 0.03,
    support_margin_y: float = 0.02,
    max_foot_height_diff: float = 0.05,
    max_foot_height: float = 0.10,
    roll_pitch_threshold: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    kick_buffers = _ensure_kick_buffers(env)
    if "recovery_success_step" not in kick_buffers:
        kick_buffers["recovery_success_step"] = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    success_steps = kick_buffers["recovery_success_step"]
    reset_mask = _episode_reset_mask(env)
    success_steps[reset_mask] = -1

    _, first_success, _ = _update_recovery_success_state(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        required_hold_time_s,
        foot_contact_force_threshold,
        support_margin_x,
        support_margin_y,
        max_foot_height_diff,
        max_foot_height,
        roll_pitch_threshold,
        asset_cfg,
    )
    success_steps[first_success] = env.episode_length_buf[first_success].long()
    return success_steps


def _recovery_success_elapsed_time(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    required_hold_time_s: float = 0.25,
    foot_contact_force_threshold: float = 5.0,
    support_margin_x: float = 0.03,
    support_margin_y: float = 0.02,
    max_foot_height_diff: float = 0.05,
    max_foot_height: float = 0.10,
    roll_pitch_threshold: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    success_steps = _get_recovery_success_steps(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        required_hold_time_s,
        foot_contact_force_threshold,
        support_margin_x,
        support_margin_y,
        max_foot_height_diff,
        max_foot_height,
        roll_pitch_threshold,
        asset_cfg,
    )
    success_mask = success_steps >= 0
    elapsed_s = (env.episode_length_buf.long() - success_steps).float() * env.step_dt
    elapsed_s = torch.where(success_mask, elapsed_s, torch.zeros_like(elapsed_s))
    return success_mask, elapsed_s


def _recovery_time_to_success(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    required_hold_time_s: float = 0.25,
    foot_contact_force_threshold: float = 5.0,
    support_margin_x: float = 0.03,
    support_margin_y: float = 0.02,
    max_foot_height_diff: float = 0.05,
    max_foot_height: float = 0.10,
    roll_pitch_threshold: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = (
        "recovery_time_to_success",
        tuple(ball_spawn_pos),
        float(success_distance),
        float(success_speed),
        float(min_success_distance_for_speed),
        float(min_recovery_delay_s),
        float(required_hold_time_s),
        float(foot_contact_force_threshold),
        float(support_margin_x),
        float(support_margin_y),
        float(max_foot_height_diff),
        float(max_foot_height),
        float(roll_pitch_threshold),
        asset_cfg.name,
    )
    step_cache = _get_step_cache(env)
    if cache_key in step_cache:
        cached = step_cache[cache_key]
        return tuple(item.clone() for item in cached)
    kick_buffers = _ensure_kick_buffers(env)
    if "recovery_time_s" not in kick_buffers:
        kick_buffers["recovery_time_s"] = torch.full((env.num_envs,), -1.0, device=env.device)

    recovery_time_s = kick_buffers["recovery_time_s"]
    reset_mask = _episode_reset_mask(env)
    recovery_time_s[reset_mask] = -1.0

    success_steps = _get_recovery_success_steps(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        required_hold_time_s,
        foot_contact_force_threshold,
        support_margin_x,
        support_margin_y,
        max_foot_height_diff,
        max_foot_height,
        roll_pitch_threshold,
        asset_cfg,
    )
    contact_steps = _get_ball_contact_steps(env)
    first_record_mask = (recovery_time_s < 0.0) & (success_steps >= 0) & (contact_steps >= 0)
    recovery_time_s[first_record_mask] = (success_steps[first_record_mask] - contact_steps[first_record_mask]).float() * env.step_dt
    success_mask = recovery_time_s >= 0.0
    result = (success_mask.clone(), torch.where(success_mask, recovery_time_s, torch.zeros_like(recovery_time_s)).clone())
    step_cache[cache_key] = result
    return result


def _recovery_timeout_mask(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    recovery_timeout_s: float = 2.75,
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
) -> torch.Tensor:
    cache_key = (
        "recovery_timeout_mask",
        tuple(ball_spawn_pos),
        float(recovery_timeout_s),
        float(success_distance),
        float(success_speed),
        float(min_success_distance_for_speed),
    )
    step_cache = _get_step_cache(env)
    if cache_key in step_cache:
        return step_cache[cache_key].clone()
    contact_mask, contact_elapsed_s = _ball_contact_elapsed_time(env)
    _, _, had_recovery_success = _update_recovery_success_state(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    result = contact_mask & ~had_recovery_success & (contact_elapsed_s >= recovery_timeout_s)
    step_cache[cache_key] = result.clone()
    return result


def _update_recovery_timing_metrics(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    recovery_timeout_s: float = 2.75,
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
) -> None:
    success_mask, recovery_time_s = _recovery_time_to_success(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    timeout_mask = _recovery_timeout_mask(
        env, ball_spawn_pos, recovery_timeout_s, success_distance, success_speed, min_success_distance_for_speed
    )
    if torch.any(success_mask):
        success_times = recovery_time_s[success_mask]
        _set_logged_metric(env, "Metrics/recovery_time_mean", torch.mean(success_times).item())
        _set_logged_metric(env, "Metrics/recovery_time_max", torch.max(success_times).item())
    else:
        _set_logged_metric(env, "Metrics/recovery_time_mean", 0.0)
        _set_logged_metric(env, "Metrics/recovery_time_max", 0.0)
    _set_logged_metric(env, "Metrics/recovery_timeout_rate", torch.mean(timeout_mask.float()).item())


def _post_recovery_success_window_mask(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    required_hold_time_s: float = 0.25,
    min_post_recovery_success_s: float = 0.0,
    max_post_recovery_success_s: float | None = None,
    foot_contact_force_threshold: float = 5.0,
    support_margin_x: float = 0.03,
    support_margin_y: float = 0.02,
    max_foot_height_diff: float = 0.05,
    max_foot_height: float = 0.10,
    roll_pitch_threshold: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    recovery_success, elapsed_s = _recovery_success_elapsed_time(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        required_hold_time_s,
        foot_contact_force_threshold,
        support_margin_x,
        support_margin_y,
        max_foot_height_diff,
        max_foot_height,
        roll_pitch_threshold,
        asset_cfg,
    )
    mask = recovery_success & (elapsed_s >= min_post_recovery_success_s)
    if max_post_recovery_success_s is not None:
        mask &= elapsed_s <= max_post_recovery_success_s
    return mask


def get_curriculum_episode_stats(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice | list[int] | None,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
) -> dict[str, torch.Tensor]:
    if env_ids is None:
        env_ids = slice(None)
    buffers = _ensure_curriculum_buffers(env)
    peak_speed, peak_distance = _update_episode_ball_metrics(env, ball_spawn_pos)
    _, _, had_ball_contact = _update_ball_contact_state(env)
    _, _, had_kick_success = _update_kick_success_state(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    recovery_diagnostics = get_recovery_condition_diagnostics(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s=0.1,
        foot_contact_force_threshold=5.0,
        support_margin_x=0.03,
        support_margin_y=0.02,
        max_foot_height_diff=0.05,
        max_foot_height=0.10,
        roll_pitch_threshold=0.20,
    )
    _, _, had_recovery_success = _update_recovery_success_state(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    success_mask, elapsed_s = _success_elapsed_time(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    recovery_success_mask, recovery_elapsed_s = _recovery_success_elapsed_time(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    terminated = getattr(env, "reset_terminated", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    time_outs = getattr(env, "reset_time_outs", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    recovery_success = had_recovery_success & ~terminated
    final_pose_success = recovery_success_mask & (recovery_elapsed_s >= 0.5) & ~terminated
    kick_success_steps = _get_kick_success_steps(env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed)
    recovery_success_steps = _get_recovery_success_steps(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    episode_end_time_s = env.episode_length_buf.float() * env.step_dt
    kick_time_s = torch.where(kick_success_steps >= 0, kick_success_steps.float() * env.step_dt, torch.zeros_like(episode_end_time_s))
    recovery_event_time_s = torch.where(
        recovery_success_steps >= 0, recovery_success_steps.float() * env.step_dt, torch.zeros_like(episode_end_time_s)
    )

    if isinstance(env_ids, slice):
        selected = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, torch.Tensor):
        selected = env_ids.to(device=env.device, dtype=torch.long)
    else:
        selected = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    def _selected_mean(values: torch.Tensor) -> float:
        if selected.numel() == 0:
            return 0.0
        return torch.mean(values[selected].float()).item()

    def _selected_max(values: torch.Tensor) -> float:
        if selected.numel() == 0:
            return 0.0
        return torch.max(values[selected].float()).item()

    def _selected_success_failure_stats(metric_name: str, values: torch.Tensor, active_mask: torch.Tensor | None = None) -> None:
        selected_values = values[selected].float()
        if active_mask is None:
            active_mask = torch.ones_like(selected_values, dtype=torch.bool)
        selected_success = recovery_success[selected] & active_mask
        selected_failure = (~recovery_success[selected]) & active_mask
        _set_logged_metric(env, f"Metrics/{metric_name}", _masked_mean(selected_values, active_mask))
        _set_logged_metric(env, f"Metrics/{metric_name}_max", _masked_max(selected_values, active_mask))
        _set_logged_metric(env, f"Metrics/{metric_name}_success_mean", _masked_mean(selected_values, selected_success))
        _set_logged_metric(env, f"Metrics/{metric_name}_failure_mean", _masked_mean(selected_values, selected_failure))

    _set_logged_metric(env, "Metrics/recovery_cond_double_support", _selected_mean(recovery_diagnostics["double_support"]))
    _set_logged_metric(env, "Metrics/recovery_cond_com_in_support", _selected_mean(recovery_diagnostics["com_in_support"]))
    _set_logged_metric(env, "Metrics/recovery_cond_feet_level", _selected_mean(recovery_diagnostics["feet_level"]))
    _set_logged_metric(env, "Metrics/recovery_cond_both_feet_low", _selected_mean(recovery_diagnostics["both_feet_low"]))
    _set_logged_metric(env, "Metrics/recovery_cond_upright", _selected_mean(recovery_diagnostics["upright"]))
    _set_logged_metric(env, "Metrics/recovery_cond_final_success", _selected_mean(recovery_success))
    _set_logged_metric(env, "Metrics/com_support_margin", max(0.03, 0.02))
    _set_logged_metric(env, "Metrics/com_support_margin_x", 0.03)
    _set_logged_metric(env, "Metrics/com_support_margin_y", 0.02)
    _set_logged_metric(env, "Metrics/feet_height_difference_threshold", 0.05)
    _set_logged_metric(env, "Metrics/feet_height_threshold", 0.10)
    _set_logged_metric(env, "Metrics/upright_threshold_deg", math.degrees(0.20))
    selected_post_contact = had_ball_contact[selected]
    _selected_success_failure_stats("com_support_distance", recovery_diagnostics["com_support_distance"], selected_post_contact)
    _selected_success_failure_stats("com_support_violation", recovery_diagnostics["com_support_violation"], selected_post_contact)
    _selected_success_failure_stats("upright_error_deg", recovery_diagnostics["upright_error_deg"], selected_post_contact)
    _selected_success_failure_stats("feet_height_difference", recovery_diagnostics["foot_height_difference"], selected_post_contact)
    _selected_success_failure_stats("max_foot_height_value", recovery_diagnostics["max_foot_height_value"], selected_post_contact)
    _selected_success_failure_stats("stance_width_y", recovery_diagnostics["stance_width_y"], selected_post_contact)
    _selected_success_failure_stats("stance_x_separation", recovery_diagnostics["stance_x_separation"], selected_post_contact)
    _set_logged_metric(
        env,
        "Metrics/stance_width_y_vs_com_support_violation_corr",
        _selected_correlation(
            recovery_diagnostics["stance_width_y"][selected].float(),
            recovery_diagnostics["com_support_violation"][selected].float(),
            selected_post_contact,
        ),
    )
    _set_logged_metric(env, "Metrics/kick_time_mean", _selected_mean(kick_time_s))
    _set_logged_metric(env, "Metrics/recovery_event_time_mean", _selected_mean(recovery_event_time_s))
    _set_logged_metric(env, "Metrics/episode_end_time_mean", _selected_mean(episode_end_time_s))

    return {
        "contact_success": had_ball_contact[env_ids].float(),
        "kick_success": had_kick_success[env_ids].float(),
        "peak_ball_forward_speed": peak_speed[env_ids].float(),
        "peak_ball_forward_distance": peak_distance[env_ids].float(),
        "recovery_success": recovery_success[env_ids].float(),
        "final_pose_success": final_pose_success[env_ids].float(),
        "double_support": recovery_diagnostics["double_support"][env_ids].float(),
        "com_in_support": recovery_diagnostics["com_in_support"][env_ids].float(),
        "feet_level": recovery_diagnostics["feet_level"][env_ids].float(),
        "both_feet_low": recovery_diagnostics["both_feet_low"][env_ids].float(),
        "upright": recovery_diagnostics["upright"][env_ids].float(),
        "terminated": terminated[env_ids].float(),
        "time_outs": time_outs[env_ids].float(),
        "success_elapsed_s": elapsed_s[env_ids].float(),
        "recovery_elapsed_s": recovery_elapsed_s[env_ids].float(),
        "success_mask": success_mask[env_ids].float(),
    }


def _recovery_phase_mask(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    success_mask, elapsed_s = _success_elapsed_time(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, asset_cfg
    )
    return success_mask & (elapsed_s >= min_recovery_delay_s)


def _recovery_window_mask(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    max_recovery_delay_s: float | None = None,
) -> torch.Tensor:
    success_mask, elapsed_s = _success_elapsed_time(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    window_mask = success_mask & (elapsed_s >= min_recovery_delay_s)
    if max_recovery_delay_s is not None:
        window_mask &= elapsed_s <= max_recovery_delay_s
    return window_mask


def _joint_symmetry_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    def joint_pos(name: str) -> torch.Tensor:
        return robot.data.joint_pos[:, robot.find_joints(name)[0][0]]

    def joint_default(name: str) -> torch.Tensor:
        idx = robot.find_joints(name)[0][0]
        return robot.data.default_joint_pos[:, idx]

    pitch_pairs = [
        ("Left_Hip_Pitch", "Right_Hip_Pitch"),
        ("Left_Knee_Pitch", "Right_Knee_Pitch"),
        ("Left_Ankle_Pitch", "Right_Ankle_Pitch"),
    ]
    mirrored_pairs = [
        ("Left_Hip_Roll", "Right_Hip_Roll"),
        ("Left_Hip_Yaw", "Right_Hip_Yaw"),
        ("Left_Ankle_Roll", "Right_Ankle_Roll"),
    ]

    error = torch.zeros(env.num_envs, device=env.device)
    for left_name, right_name in pitch_pairs:
        left_rel = joint_pos(left_name) - joint_default(left_name)
        right_rel = joint_pos(right_name) - joint_default(right_name)
        error += torch.square(left_rel - right_rel)
    for left_name, right_name in mirrored_pairs:
        left_rel = joint_pos(left_name) - joint_default(left_name)
        right_rel = joint_pos(right_name) - joint_default(right_name)
        error += torch.square(left_rel + right_rel)
    return error


def _joint_deviation_sum(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = [robot.find_joints(name)[0][0] for name in joint_names]
    joint_error = torch.abs(robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids])
    return torch.sum(joint_error, dim=1)


def _update_previous_ball_speed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> tuple[torch.Tensor, torch.Tensor]:
    kick_buffers = _ensure_kick_buffers(env)
    if "previous_ball_speed" not in kick_buffers:
        kick_buffers["previous_ball_speed"] = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    previous_ball_speed = kick_buffers["previous_ball_speed"]
    reset_mask = _episode_reset_mask(env)
    previous_ball_speed[reset_mask] = 0.0

    current_ball_speed = _ball_planar_speed(env, asset_cfg)
    delta_speed = torch.clamp(current_ball_speed - previous_ball_speed, min=0.0)
    previous_ball_speed[:] = current_ball_speed
    return current_ball_speed, delta_speed


def _get_kick_trigger_steps(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    trigger_speed: float,
    trigger_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    ball: RigidObject = env.scene[asset_cfg.name]
    spawn_pos = _ball_spawn_world(env, ball_spawn_pos)
    travel_distance = torch.linalg.norm(ball.data.root_pos_w[:, :2] - spawn_pos[:, :2], dim=1)
    forward_speed = torch.clamp(ball.data.root_lin_vel_w[:, 0], min=0.0)
    triggered = (forward_speed >= trigger_speed) | (travel_distance >= trigger_distance)

    kick_buffers = _ensure_kick_buffers(env)
    if "kick_trigger_step" not in kick_buffers:
        kick_buffers["kick_trigger_step"] = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    trigger_steps = kick_buffers["kick_trigger_step"]
    reset_mask = _episode_reset_mask(env)
    trigger_steps[reset_mask] = -1

    new_trigger_mask = (trigger_steps < 0) & triggered
    trigger_steps[new_trigger_mask] = env.episode_length_buf[new_trigger_mask].long()
    return trigger_steps


def reward_ball_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    left_sensor_name: str = "ball_contact_left_foot",
    right_sensor_name: str = "ball_contact_right_foot",
) -> torch.Tensor:
    current_contact, _, _ = _update_ball_contact_state(env, threshold, left_sensor_name, right_sensor_name)
    return current_contact.float()


def first_ball_contact_bonus(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    left_sensor_name: str = "ball_contact_left_foot",
    right_sensor_name: str = "ball_contact_right_foot",
) -> torch.Tensor:
    _, first_contact, _ = _update_ball_contact_state(env, threshold, left_sensor_name, right_sensor_name)
    return first_contact.float()


def reward_ball_speed_increase(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    _, delta_speed = _update_previous_ball_speed(env, asset_cfg)
    return delta_speed


def reward_ball_forward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    return _ball_forward_speed(env, asset_cfg)


def reward_kick_distance_progress(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_POWER):
        return _zero_reward(env)
    _update_episode_ball_metrics(env, ball_spawn_pos, asset_cfg)
    forward_distance = torch.clamp(_ball_forward_distance(env, ball_spawn_pos, asset_cfg), min=0.0)
    return torch.clamp(forward_distance / max(success_distance, 1.0e-6), max=1.0)


def reward_kick_speed_progress(
    env: ManagerBasedRLEnv,
    success_speed: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_POWER):
        return _zero_reward(env)
    forward_speed = _ball_forward_speed(env, asset_cfg)
    return torch.clamp(forward_speed / max(success_speed, 1.0e-6), max=1.0)


def reward_kick_success(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_POWER):
        return _zero_reward(env)
    _, first_success, _ = _update_kick_success_state(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, asset_cfg
    )
    return first_success.float()


def reward_stand_still(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_std: float = 0.15,
    ang_vel_std: float = 0.3,
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_FINAL_POSE):
        _set_logged_metric(env, "Metrics/reward_stance_x_alignment_fire_rate", 0.0)
        return _zero_reward(env)
    robot: Articulation = env.scene[asset_cfg.name]
    lin_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    ang_speed = torch.abs(robot.data.root_ang_vel_w[:, 2])
    stand_reward = torch.exp(-(lin_speed**2) / (lin_vel_std**2) - (ang_speed**2) / (ang_vel_std**2))
    return _post_success_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    ).float() * stand_reward


def reward_base_position_hold(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_forward_offset: float = 0.10,
    forward_std: float = 0.16,
    lateral_std: float = 0.06,
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_FINAL_POSE):
        _set_logged_metric(env, "Metrics/reward_stance_width_y_fire_rate", 0.0)
        return _zero_reward(env)
    forward_offset, lateral_offset = _base_step_offsets(env, asset_cfg)
    hold_reward = torch.exp(
        -torch.square(forward_offset - target_forward_offset) / (forward_std**2)
        - torch.square(lateral_offset) / (lateral_std**2)
    )
    return _post_success_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    ).float() * hold_reward


def penalty_base_position_drift(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_forward_offset: float = 0.35,
    max_lateral_offset: float = 0.12,
    backward_tolerance: float = 0.02,
) -> torch.Tensor:
    forward_offset, lateral_offset = _base_step_offsets(env, asset_cfg)
    excessive_forward = torch.relu(forward_offset - max_forward_offset)
    excessive_backward = torch.relu(-forward_offset - backward_tolerance)
    excessive_lateral = torch.relu(torch.abs(lateral_offset) - max_lateral_offset)
    return excessive_lateral + 0.5 * (excessive_forward + excessive_backward)


def penalty_unnecessary_walking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)


def reward_post_kick_stability(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_height: float = 0.42,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    robot: Articulation = env.scene[asset_cfg.name]
    success_mask = _post_success_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )

    lin_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    ang_speed = torch.linalg.norm(robot.data.root_ang_vel_w[:, :2], dim=1)
    height_ok = (robot.data.root_pos_w[:, 2] >= min_height).float()
    stability = torch.exp(-(lin_speed**2) / 0.12 - (ang_speed**2) / 0.5) * height_ok
    return success_mask.float() * stability


def reward_recover_to_stand(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    required_hold_time_s: float = 0.25,
    recovery_time_scale_s: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    _, first_recovery_success, _ = _update_recovery_success_state(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        required_hold_time_s,
        asset_cfg=asset_cfg,
    )
    success_mask, recovery_time_s = _recovery_time_to_success(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        required_hold_time_s,
        asset_cfg=asset_cfg,
    )
    timed_reward = torch.exp(-recovery_time_s / recovery_time_scale_s)
    return first_recovery_success.float() * success_mask.float() * timed_reward


def reward_symmetric_posture(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_FINAL_POSE):
        _set_logged_metric(env, "Metrics/reward_nominal_stance_width_fire_rate", 0.0)
        return _zero_reward(env)
    recovery_mask = _recovery_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        max_recovery_delay_s=None,
    )
    symmetry_error = _joint_symmetry_error(env, asset_cfg)
    return recovery_mask.float() * torch.exp(-symmetry_error / 0.20)


def reward_double_support(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    foot_height_threshold: float = 0.18,
    foot_vertical_speed_threshold: float = 0.8,
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    recovery_mask = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    double_support = _double_support_mask(env, foot_height_threshold, foot_vertical_speed_threshold)
    return recovery_mask.float() * double_support.float()


def reward_post_kick_balance(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    foot_height_threshold: float = 0.18,
    foot_vertical_speed_threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    robot: Articulation = env.scene[asset_cfg.name]
    recovery_mask = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    double_support = _double_support_mask(
        env,
        foot_height_threshold=foot_height_threshold,
        foot_vertical_speed_threshold=foot_vertical_speed_threshold,
        asset_cfg=asset_cfg,
    )
    upright = torch.exp(-torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1) / 0.08)
    low_motion = torch.exp(
        -torch.sum(torch.square(robot.data.root_lin_vel_w[:, :2]), dim=1) / 0.08
        - torch.sum(torch.square(robot.data.root_ang_vel_w[:, :2]), dim=1) / 0.2
    )
    return recovery_mask.float() * double_support.float() * upright * low_motion


def reward_return_to_default_pose(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    recovery_mask = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    joint_error = torch.mean(torch.abs(robot.data.joint_pos - robot.data.default_joint_pos), dim=1)
    return recovery_mask.float() * torch.exp(-joint_error / 0.18)


def reward_joint_symmetry(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_FINAL_POSE):
        return _zero_reward(env)
    recovery_mask = _recovery_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        max_recovery_delay_s=None,
    )
    symmetry_error = _joint_symmetry_error(env, asset_cfg)
    return recovery_mask.float() * torch.exp(-symmetry_error / 0.14)


def reward_feet_alignment(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    lateral_distance_ref: float = 0.18,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_FINAL_POSE):
        return _zero_reward(env)
    # Final-pose shaping starts once kick success has happened and the recovery phase is active.
    # This term is not evaluated before the kick.
    recovery_mask = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    left_rel_x, right_rel_x, left_rel_y, right_rel_y = _foot_positions_relative_to_com(env, asset_cfg)
    forward_alignment = torch.square(left_rel_x - right_rel_x)
    lateral_alignment = torch.square(torch.abs(left_rel_y - right_rel_y) - lateral_distance_ref)
    return recovery_mask.float() * torch.exp(-(forward_alignment + lateral_alignment) / 0.05)


def reward_post_kick_settling(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_FINAL_POSE):
        return _zero_reward(env)
    robot: Articulation = env.scene[asset_cfg.name]
    recovery_mask = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    joint_vel = torch.mean(torch.abs(robot.data.joint_vel), dim=1)
    yaw_rate = torch.abs(robot.data.root_ang_vel_w[:, 2])
    upright_error = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)
    return recovery_mask.float() * torch.exp(-(joint_vel / 6.0 + yaw_rate / 1.5 + upright_error / 0.08))


def reward_step_recovery(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_forward_offset: float = 0.10,
    forward_std: float = 0.08,
    lateral_std: float = 0.06,
    max_recovery_delay_s: float = 0.45,
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    recovery_mask = _recovery_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        max_recovery_delay_s,
    )
    _, _, had_recovery_success = _update_recovery_success_state(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    recovery_mask &= ~had_recovery_success
    forward_offset, lateral_offset = _base_step_offsets(env, asset_cfg)
    step_reward = torch.exp(
        -torch.square(forward_offset - target_forward_offset) / (forward_std**2)
        - torch.square(lateral_offset) / (lateral_std**2)
    )
    return recovery_mask.float() * step_reward


def reward_forward_step_stability(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_forward_offset: float = 0.10,
    forward_std: float = 0.10,
    lateral_std: float = 0.06,
    max_recovery_delay_s: float = 0.45,
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    robot: Articulation = env.scene[asset_cfg.name]
    recovery_mask = _recovery_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        max_recovery_delay_s,
    )
    _, _, had_recovery_success = _update_recovery_success_state(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    recovery_mask &= ~had_recovery_success
    forward_offset, lateral_offset = _base_step_offsets(env, asset_cfg)
    step_shape = torch.exp(
        -torch.square(forward_offset - target_forward_offset) / (forward_std**2)
        - torch.square(lateral_offset) / (lateral_std**2)
    )
    low_motion = torch.exp(
        -torch.sum(torch.square(robot.data.root_lin_vel_w[:, :2]), dim=1) / 0.18
        - torch.square(robot.data.root_ang_vel_w[:, 2]) / 0.60
    )
    upright = torch.exp(-torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1) / 0.10)
    return recovery_mask.float() * step_shape * low_motion * upright


def reward_opposite_foot_recovery(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.10,
    max_recovery_delay_s: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_recovery_step: float = 0.08,
    forward_std: float = 0.08,
    lateral_std: float = 0.06,
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    recovery_mask = _recovery_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        max_recovery_delay_s,
    )
    _, _, had_recovery_success = _update_recovery_success_state(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    recovery_mask &= ~had_recovery_success
    kick_sign = _kick_foot_sign(env)
    left_fx, right_fx, left_fy, right_fy = _foot_step_offsets(env, asset_cfg)
    opposite_forward = torch.where(kick_sign < 0, right_fx, left_fx)
    opposite_lateral = torch.where(kick_sign < 0, right_fy, left_fy)
    valid_mask = (kick_sign != 0).float()
    step_reward = torch.exp(
        -torch.square(opposite_forward - target_recovery_step) / (forward_std**2)
        - torch.square(opposite_lateral) / (lateral_std**2)
    )
    return recovery_mask.float() * valid_mask * step_reward


def reward_com_stability(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    robot: Articulation = env.scene[asset_cfg.name]
    recovery_mask = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    # Root linear/angular stabilization is used as a COM stability surrogate.
    com_like_stability = torch.exp(
        -torch.sum(torch.square(robot.data.root_lin_vel_w[:, :2]), dim=1) / 0.18
        - torch.sum(torch.square(robot.data.root_ang_vel_w[:, :2]), dim=1) / 0.30
        - torch.square(robot.data.root_ang_vel_w[:, 2]) / 0.60
        - torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1) / 0.10
    )
    return recovery_mask.float() * com_like_stability


def reward_stop_after_recovery(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.55,
    required_hold_time_s: float = 0.25,
    min_post_recovery_success_s: float = 0.0,
    max_post_recovery_success_s: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    robot: Articulation = env.scene[asset_cfg.name]
    settle_mask = _post_recovery_success_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        required_hold_time_s,
        min_post_recovery_success_s=min_post_recovery_success_s,
        max_post_recovery_success_s=max_post_recovery_success_s,
        asset_cfg=asset_cfg,
    )
    low_motion = torch.exp(
        -torch.sum(torch.square(robot.data.root_lin_vel_w[:, :2]), dim=1) / 0.18
        - torch.square(robot.data.root_ang_vel_w[:, 2]) / 0.40
        - torch.mean(torch.square(robot.data.joint_vel), dim=1) / 16.0
    )
    return settle_mask.float() * low_motion


def reward_zero_velocity_after_settle(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.55,
    required_hold_time_s: float = 0.25,
    min_post_recovery_success_s: float = 0.0,
    max_post_recovery_success_s: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    robot: Articulation = env.scene[asset_cfg.name]
    settle_mask = _post_recovery_success_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        required_hold_time_s,
        min_post_recovery_success_s=min_post_recovery_success_s,
        max_post_recovery_success_s=max_post_recovery_success_s,
        asset_cfg=asset_cfg,
    )
    base_vel_xy = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    yaw_rate = torch.abs(robot.data.root_ang_vel_w[:, 2])
    return settle_mask.float() * torch.exp(-(base_vel_xy**2) / 0.05 - (yaw_rate**2) / 0.12)


def reward_stable_double_support(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    foot_height_threshold: float = 0.18,
    foot_vertical_speed_threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    robot: Articulation = env.scene[asset_cfg.name]
    recovery_mask = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    double_support = _double_support_mask(
        env,
        foot_height_threshold=foot_height_threshold,
        foot_vertical_speed_threshold=foot_vertical_speed_threshold,
        asset_cfg=asset_cfg,
    ).float()
    low_motion = torch.exp(
        -torch.sum(torch.square(robot.data.root_lin_vel_w[:, :2]), dim=1) / 0.16
        - torch.square(robot.data.root_ang_vel_w[:, 2]) / 0.50
    )
    upright = torch.exp(-torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1) / 0.10)
    return recovery_mask.float() * double_support * low_motion * upright


def reward_final_double_support(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.55,
    foot_height_threshold: float = 0.18,
    foot_vertical_speed_threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_FINAL_POSE):
        return _zero_reward(env)
    settle_mask = _recovery_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        max_recovery_delay_s=None,
    )
    double_support = _double_support_mask(
        env,
        foot_height_threshold=foot_height_threshold,
        foot_vertical_speed_threshold=foot_vertical_speed_threshold,
        asset_cfg=asset_cfg,
    ).float()
    symmetry = torch.exp(-_joint_symmetry_error(env, asset_cfg) / 0.14)
    return settle_mask.float() * double_support * symmetry


def reward_feet_under_com_x(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.25,
    foot_x_std: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_FINAL_POSE):
        return _zero_reward(env)
    settle_mask = _recovery_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        max_recovery_delay_s=None,
    )
    left_rel_x, right_rel_x, _, _ = _foot_positions_relative_to_com(env, asset_cfg)
    under_com = torch.exp(-(torch.square(left_rel_x) + torch.square(right_rel_x)) / (foot_x_std**2))
    return settle_mask.float() * under_com


def reward_stance_x_alignment(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.25,
    target_max_x_separation: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_FINAL_POSE):
        return _zero_reward(env)
    settle_mask = _post_recovery_success_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        required_hold_time_s=0.25,
        min_post_recovery_success_s=0.0,
        max_post_recovery_success_s=None,
        asset_cfg=asset_cfg,
    )
    _log_mask_rate(env, "Metrics/reward_stance_x_alignment_fire_rate", settle_mask)
    stance_x = _stance_x_separation(env, asset_cfg)
    scale = max(target_max_x_separation, 1.0e-6)
    return settle_mask.float() * torch.exp(-torch.square(stance_x / scale))


def reward_stance_width_y(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.25,
    target_stance_width_y: float = 0.195,
    stance_width_std: float = 0.025,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_FINAL_POSE):
        return _zero_reward(env)
    settle_mask = _post_recovery_success_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        required_hold_time_s=0.25,
        min_post_recovery_success_s=0.0,
        max_post_recovery_success_s=None,
        asset_cfg=asset_cfg,
    )
    _log_mask_rate(env, "Metrics/reward_stance_width_y_fire_rate", settle_mask)
    width_y = _stance_width_y(env, asset_cfg)
    return settle_mask.float() * torch.exp(-torch.square(width_y - target_stance_width_y) / (stance_width_std**2))


def reward_nominal_stance_width(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.25,
    nominal_half_width: float = 0.11,
    stance_y_std: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_FINAL_POSE):
        return _zero_reward(env)
    # This settles the final stance only after the post-kick recovery window opens.
    settle_mask = _recovery_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        max_recovery_delay_s=None,
    )
    _log_mask_rate(env, "Metrics/reward_nominal_stance_width_fire_rate", settle_mask)
    _, _, left_rel_y, right_rel_y = _foot_positions_relative_to_com(env, asset_cfg)
    # Use width magnitude as the main term and keep sign preference soft to avoid a dead reward.
    left_width = torch.abs(left_rel_y)
    right_width = torch.abs(right_rel_y)
    width_reward = torch.exp(
        -torch.square(left_width - nominal_half_width) / (stance_y_std**2)
        - torch.square(right_width - nominal_half_width) / (stance_y_std**2)
    )
    sign_bonus = torch.sigmoid(12.0 * left_rel_y) * torch.sigmoid(-12.0 * right_rel_y)
    width_reward = width_reward * (0.5 + 0.5 * sign_bonus)
    return settle_mask.float() * width_reward


def penalty_yaw_rate(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    robot: Articulation = env.scene[asset_cfg.name]
    recovery_mask = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    return recovery_mask.float() * torch.abs(robot.data.root_ang_vel_w[:, 2])


def penalty_post_kick_yaw(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    recovery_mask = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    return recovery_mask.float() * _heading_error(env, asset_cfg)


def penalty_post_kick_walking(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.55,
    walking_speed_threshold: float = 0.12,
    yaw_rate_threshold: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    robot: Articulation = env.scene[asset_cfg.name]
    settle_mask = _recovery_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        max_recovery_delay_s=None,
    )
    base_vel_xy = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    yaw_rate = torch.abs(robot.data.root_ang_vel_w[:, 2])
    walking_penalty = torch.relu(base_vel_xy - walking_speed_threshold) + 0.5 * torch.relu(yaw_rate - yaw_rate_threshold)
    return settle_mask.float() * walking_penalty


def penalty_torso_pitch(
    env: ManagerBasedRLEnv,
    pitch_tolerance: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    _, pitch = _root_roll_pitch(env, asset_cfg)
    post_contact_mask = _had_ball_contact_mask(env)
    return post_contact_mask.float() * torch.relu(pitch - pitch_tolerance)


def penalty_hip_roll_deviation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    post_contact_mask = _had_ball_contact_mask(env)
    hip_roll_error = _joint_deviation_sum(env, ["Left_Hip_Roll", "Right_Hip_Roll"], asset_cfg)
    return post_contact_mask.float() * hip_roll_error


def penalty_hip_yaw_deviation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    post_contact_mask = _had_ball_contact_mask(env)
    hip_yaw_error = _joint_deviation_sum(env, ["Left_Hip_Yaw", "Right_Hip_Yaw"], asset_cfg)
    return post_contact_mask.float() * hip_yaw_error


def penalty_support_foot_drift(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.0,
    recovery_timeout_s: float = 2.75,
    no_penalty_support_foot_x_drift: float = 0.05,
    clear_penalty_support_foot_x_drift: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    support_foot_x_drift, support_valid_mask = _support_foot_x_drift(env, asset_cfg)
    stance_x_separation = _stance_x_separation(env, asset_cfg)
    stance_width_y = _stance_width_y(env, asset_cfg)
    foot_distance_xy = _foot_distance_xy(env, asset_cfg)
    min_leg_link_distance = _min_leg_link_distance(env, asset_cfg)
    _update_support_foot_metrics(
        env,
        support_foot_x_drift,
        support_valid_mask,
        stance_x_separation,
        stance_width_y,
        foot_distance_xy,
        min_leg_link_distance,
    )
    _update_recovery_timing_metrics(
        env, ball_spawn_pos, recovery_timeout_s, success_distance, success_speed, min_success_distance_for_speed
    )
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    _, _, had_recovery_success = _update_recovery_success_state(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    contact_to_recovery_mask = _had_ball_contact_mask(env) & ~had_recovery_success
    transition_width = max(clear_penalty_support_foot_x_drift - no_penalty_support_foot_x_drift, 1.0e-6)
    transition = torch.clamp((support_foot_x_drift - no_penalty_support_foot_x_drift) / transition_width, 0.0, 1.0)
    smooth_penalty = transition * transition * (3.0 - 2.0 * transition)
    tail_penalty = torch.relu(support_foot_x_drift - clear_penalty_support_foot_x_drift) / transition_width
    return contact_to_recovery_mask.float() * support_valid_mask.float() * (smooth_penalty + tail_penalty)


def penalty_split_stance(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.25,
    tolerated_split_x: float = 0.10,
    penalty_scale_x: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    # Split-stance suppression starts at first ball contact and remains active until recovery succeeds,
    # so support-foot stepping is discouraged during kick compensation rather than only at the end pose.
    _, _, had_recovery_success = _update_recovery_success_state(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    settle_mask = _had_ball_contact_mask(env) & ~had_recovery_success
    split_x = _stance_x_separation(env, asset_cfg)
    scale = max(penalty_scale_x, 1.0e-6)
    excess = torch.relu(split_x - tolerated_split_x) / scale
    return settle_mask.float() * (excess * excess + 0.5 * excess)


def penalty_narrow_stance_width(
    env: ManagerBasedRLEnv,
    target_stance_width_y: float = 0.19,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    stance_width_y = _stance_width_y(env, asset_cfg)
    post_contact_mask = _had_ball_contact_mask(env)
    return post_contact_mask.float() * torch.relu(target_stance_width_y - stance_width_y)


def penalty_wide_stance_width(
    env: ManagerBasedRLEnv,
    target_stance_width_y: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    stance_width_y = _stance_width_y(env, asset_cfg)
    post_contact_mask = _had_ball_contact_mask(env)
    wide_error = torch.relu(stance_width_y - target_stance_width_y)
    return post_contact_mask.float() * wide_error


def penalty_min_leg_link_distance(
    env: ManagerBasedRLEnv,
    no_penalty_distance: float = 0.15,
    strong_penalty_distance: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    min_leg_link_distance = _min_leg_link_distance(env, asset_cfg)
    post_contact_mask = _had_ball_contact_mask(env)
    mild_penalty = torch.relu(no_penalty_distance - min_leg_link_distance) / max(no_penalty_distance, 1.0e-6)
    strong_penalty = torch.relu(strong_penalty_distance - min_leg_link_distance) / max(strong_penalty_distance, 1.0e-6)
    return post_contact_mask.float() * (mild_penalty + 4.0 * torch.square(strong_penalty))


def penalty_foot_overlap(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.25,
    minimum_foot_distance_xy: float = 0.15,
    strong_penalty_distance_xy: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_FINAL_POSE):
        return _zero_reward(env)
    settle_mask = _post_recovery_success_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        required_hold_time_s=0.25,
        min_post_recovery_success_s=0.0,
        max_post_recovery_success_s=None,
        asset_cfg=asset_cfg,
    )
    foot_distance_xy = _foot_distance_xy(env, asset_cfg)
    mild_deficit = torch.relu(minimum_foot_distance_xy - foot_distance_xy)
    strong_deficit = torch.relu(strong_penalty_distance_xy - foot_distance_xy)
    mild_term = mild_deficit / max(minimum_foot_distance_xy, 1.0e-6)
    strong_term = torch.square(strong_deficit / max(strong_penalty_distance_xy, 1.0e-6))
    return settle_mask.float() * (mild_term + 4.0 * strong_term)


def penalty_final_pose_hip_deviation(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_FINAL_POSE):
        return _zero_reward(env)
    settle_mask = _post_recovery_success_window_mask(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        required_hold_time_s=0.25,
        min_post_recovery_success_s=0.0,
        max_post_recovery_success_s=None,
        asset_cfg=asset_cfg,
    )
    robot: Articulation = env.scene[asset_cfg.name]
    joint_names = ["Left_Hip_Roll", "Right_Hip_Roll", "Left_Hip_Yaw", "Right_Hip_Yaw"]
    joint_ids = [robot.find_joints(name)[0][0] for name in joint_names]
    joint_error = torch.abs(robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]).sum(dim=1)
    return settle_mask.float() * joint_error


def penalty_single_leg_freeze(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.25,
    single_leg_grace_time_s: float = 0.45,
    foot_contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    recovery_phase = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    _, _, had_recovery_success = _update_recovery_success_state(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    left_contact, right_contact = _foot_contact_sides(env, force_threshold=foot_contact_force_threshold)
    single_support = left_contact ^ right_contact
    freeze_candidate = recovery_phase & ~had_recovery_success & single_support

    kick_buffers = _ensure_kick_buffers(env)
    if "single_leg_freeze_steps" not in kick_buffers:
        kick_buffers["single_leg_freeze_steps"] = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    single_leg_freeze_steps = kick_buffers["single_leg_freeze_steps"]
    reset_mask = _episode_reset_mask(env)
    single_leg_freeze_steps[reset_mask] = 0
    single_leg_freeze_steps[~freeze_candidate] = 0
    single_leg_freeze_steps[freeze_candidate] += 1

    single_leg_duration_s = single_leg_freeze_steps.float() * env.step_dt
    return torch.relu(single_leg_duration_s - single_leg_grace_time_s)


def penalty_raised_kick_leg(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.20,
    kick_leg_height_threshold: float = 0.14,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    recovery_phase = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    _, _, had_recovery_success = _update_recovery_success_state(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed
    )
    kick_sign = _kick_foot_sign(env)
    foot_pose = _foot_pose_metrics(env, asset_cfg)
    kick_foot_height = torch.where(kick_sign < 0, foot_pose["left_z"], foot_pose["right_z"])
    valid_kick_foot = kick_sign != 0
    raised_leg_mask = recovery_phase & ~had_recovery_success & valid_kick_foot
    return raised_leg_mask.float() * torch.relu(kick_foot_height - kick_leg_height_threshold)


def reward_yaw_stabilization(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    yaw_rate_std: float = 0.6,
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    robot: Articulation = env.scene[asset_cfg.name]
    recovery_mask = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    yaw_rate = torch.abs(robot.data.root_ang_vel_w[:, 2])
    return recovery_mask.float() * torch.exp(-(yaw_rate**2) / (yaw_rate_std**2))


def reward_heading_recovery(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    heading_std: float = 0.35,
) -> torch.Tensor:
    if not _stage_active(env, KICK_STAGE_RECOVERY):
        return _zero_reward(env)
    recovery_mask = _recovery_phase_mask(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, min_recovery_delay_s
    )
    yaw_error = _heading_error(env, asset_cfg)
    return recovery_mask.float() * torch.exp(-(yaw_error**2) / (heading_std**2))


def penalty_no_kick_timeout(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    left_sensor_name: str = "ball_contact_left_foot",
    right_sensor_name: str = "ball_contact_right_foot",
) -> torch.Tensor:
    _, _, had_ball_contact = _update_ball_contact_state(env, threshold, left_sensor_name, right_sensor_name)
    return (_time_out_mask(env) & ~had_ball_contact).float()


def penalty_recovery_timeout(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    recovery_timeout_s: float = 2.75,
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
) -> torch.Tensor:
    return _recovery_timeout_mask(
        env, ball_spawn_pos, recovery_timeout_s, success_distance, success_speed, min_success_distance_for_speed
    ).float()


def terminate_ball_travel_distance(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    distance_threshold: float,
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_post_success_time_s: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    ball: RigidObject = env.scene[asset_cfg.name]
    spawn_pos = _ball_spawn_world(env, ball_spawn_pos)
    travel_distance = torch.linalg.norm(ball.data.root_pos_w[:, :2] - spawn_pos[:, :2], dim=1)
    success_mask, elapsed_s = _success_elapsed_time(
        env, ball_spawn_pos, success_distance, success_speed, min_success_distance_for_speed, asset_cfg
    )
    return success_mask & (elapsed_s >= min_post_success_time_s) & (travel_distance >= distance_threshold)


def terminate_post_kick_settle_time(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    settle_time_s: float = 0.4,
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
    min_recovery_delay_s: float = 0.1,
    required_hold_time_s: float = 0.25,
) -> torch.Tensor:
    recovery_success_mask, elapsed_s = _recovery_success_elapsed_time(
        env,
        ball_spawn_pos,
        success_distance,
        success_speed,
        min_success_distance_for_speed,
        min_recovery_delay_s,
        required_hold_time_s,
    )
    return recovery_success_mask & (elapsed_s >= settle_time_s)


def terminate_recovery_timeout(
    env: ManagerBasedRLEnv,
    ball_spawn_pos: tuple[float, float, float],
    recovery_timeout_s: float = 2.75,
    success_distance: float = 0.8,
    success_speed: float = 1.0,
    min_success_distance_for_speed: float = 0.12,
) -> torch.Tensor:
    return _recovery_timeout_mask(
        env, ball_spawn_pos, recovery_timeout_s, success_distance, success_speed, min_success_distance_for_speed
    )


def terminate_time_out(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    left_sensor_name: str = "ball_contact_left_foot",
    right_sensor_name: str = "ball_contact_right_foot",
) -> torch.Tensor:
    _, _, had_ball_contact = _update_ball_contact_state(env, threshold, left_sensor_name, right_sensor_name)
    return _time_out_mask(env) & had_ball_contact


def terminate_leg_self_collision(
    env: ManagerBasedRLEnv,
    minimum_leg_link_distance: float = 0.05,
    contact_force_threshold: float = 1.0,
    log_detailed_pair_metrics: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    min_leg_link_distance, min_pair_index, pairwise_distance = _min_leg_link_info(env, asset_cfg)
    any_contact, contact_count, pair_contact_mask, pair_force_norm = _leg_self_contact_info(
        env, sensor_name="leg_self_contact_left", contact_force_threshold=contact_force_threshold
    )
    _log_leg_collision_metrics(
        env,
        min_leg_link_distance,
        min_pair_index,
        pairwise_distance,
        any_contact,
        contact_count,
        pair_contact_mask,
        pair_force_norm,
        log_detailed_pair_metrics,
    )
    collision_mask = (min_leg_link_distance < minimum_leg_link_distance) | any_contact
    _log_mask_rate(env, "Episode_Termination/leg_self_collision", collision_mask)
    return collision_mask


def terminate_no_kick_timeout(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    left_sensor_name: str = "ball_contact_left_foot",
    right_sensor_name: str = "ball_contact_right_foot",
) -> torch.Tensor:
    _, _, had_ball_contact = _update_ball_contact_state(env, threshold, left_sensor_name, right_sensor_name)
    return _time_out_mask(env) & ~had_ball_contact


def _install_reward_profiling() -> None:
    profile_targets = [
        "reward_ball_contact",
        "first_ball_contact_bonus",
        "reward_ball_speed_increase",
        "reward_ball_forward_velocity",
        "reward_kick_distance_progress",
        "reward_kick_speed_progress",
        "reward_kick_success",
        "reward_stand_still",
        "reward_base_position_hold",
        "reward_post_kick_stability",
        "reward_recover_to_stand",
        "reward_symmetric_posture",
        "reward_double_support",
        "reward_post_kick_balance",
        "reward_return_to_default_pose",
        "reward_joint_symmetry",
        "reward_feet_alignment",
        "reward_post_kick_settling",
        "reward_step_recovery",
        "reward_forward_step_stability",
        "reward_opposite_foot_recovery",
        "reward_com_stability",
        "reward_stop_after_recovery",
        "reward_zero_velocity_after_settle",
        "reward_stable_double_support",
        "reward_final_double_support",
        "reward_feet_under_com_x",
        "reward_stance_x_alignment",
        "reward_stance_width_y",
        "reward_nominal_stance_width",
        "reward_yaw_stabilization",
        "reward_heading_recovery",
        "penalty_base_position_drift",
        "penalty_unnecessary_walking",
        "penalty_yaw_rate",
        "penalty_post_kick_yaw",
        "penalty_torso_pitch",
        "penalty_hip_roll_deviation",
        "penalty_hip_yaw_deviation",
        "penalty_post_kick_walking",
        "penalty_support_foot_drift",
        "penalty_split_stance",
        "penalty_narrow_stance_width",
        "penalty_wide_stance_width",
        "penalty_min_leg_link_distance",
        "penalty_foot_overlap",
        "penalty_final_pose_hip_deviation",
        "penalty_single_leg_freeze",
        "penalty_raised_kick_leg",
        "penalty_no_kick_timeout",
        "penalty_recovery_timeout",
        "terminate_ball_travel_distance",
        "terminate_leg_self_collision",
        "terminate_post_kick_settle_time",
        "terminate_time_out",
        "terminate_no_kick_timeout",
        "terminate_recovery_timeout",
        "_update_recovery_success_state",
        "_recovery_condition_mask",
        "_post_recovery_success_window_mask",
        "_recovery_time_to_success",
        "_min_leg_link_info",
        "_leg_self_contact_info",
        "get_curriculum_episode_stats",
    ]
    for name in profile_targets:
        fn = globals().get(name)
        if fn is None or getattr(fn, "_kick_profile_wrapped", False):
            continue
        wrapped = _profile_term(name)(fn)
        wrapped._kick_profile_wrapped = True
        globals()[name] = wrapped


_install_reward_profiling()
