# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL policy with episode-level metrics and CSV export."""

import argparse
import csv
import datetime as dt
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--episodes", type=int, default=100, help="Number of completed episodes to evaluate.")
parser.add_argument("--eval-time", type=float, default=60.0, help="Time-based evaluation duration [s].")
parser.add_argument("--warmup-time", type=float, default=0.5, help="Warmup duration before metrics start [s].")
parser.add_argument(
    "--tracking-window-time",
    type=float,
    default=0.5,
    help="Moving-average window for tracking decision [s].",
)
parser.add_argument(
    "--fall-buffer-time",
    type=float,
    default=1.0,
    help="Exclude this duration before a fall from tracking-rate aggregation [s].",
)
parser.add_argument("--target-velocity", type=float, default=1.0, help="Target forward velocity [m/s].")
parser.add_argument("--tracking-tolerance", type=float, default=0.04, help="Tracking tolerance [m/s].")
parser.add_argument("--tracking-tolerance-x", type=float, default=0.20, help="Tracking score limit for X axis [m/s].")
parser.add_argument("--tracking-tolerance-y", type=float, default=0.20, help="Tracking score limit for Y axis [m/s].")
parser.add_argument("--tracking-tolerance-z", type=float, default=0.40, help="Tracking score limit for yaw Z axis [rad/s].")
parser.add_argument("--cmd-vel-x", type=float, default=None, help="Override command lin_vel_x [m/s].")
parser.add_argument("--cmd-vel-y", type=float, default=0.0, help="Override command lin_vel_y [m/s].")
parser.add_argument("--cmd-yaw", type=float, default=0.0, help="Override command ang_vel_z [rad/s].")
parser.add_argument(
    "--metric-warmup-steps",
    type=int,
    default=5,
    help="Per-episode warmup steps excluded from tracking/error statistics.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401


def _as_bool_tensor(value: Any, device: torch.device, num_envs: int) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        data = value.to(device=device)
    else:
        try:
            data = torch.as_tensor(value, device=device)
        except Exception:
            return None
    if data.numel() == 1:
        data = data.expand(num_envs)
    if data.ndim > 1:
        data = data.reshape(-1)
    if data.shape[0] != num_envs:
        return None
    return data.bool()


def _as_float_tensor(value: Any, device: torch.device, num_envs: int) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        data = value.to(device=device, dtype=torch.float32)
    else:
        try:
            data = torch.as_tensor(value, device=device, dtype=torch.float32)
        except Exception:
            return None
    if data.numel() == 1:
        data = data.expand(num_envs)
    if data.ndim > 1:
        data = data.reshape(-1)
    if data.shape[0] != num_envs:
        return None
    return data


def _find_metric_tensor(log_dict: dict[str, Any], tokens: list[str], device: torch.device, num_envs: int) -> torch.Tensor | None:
    lowered_tokens = [token.lower() for token in tokens]
    for key, value in log_dict.items():
        key_l = str(key).lower()
        if all(token in key_l for token in lowered_tokens):
            tensor = _as_float_tensor(value, device, num_envs)
            if tensor is not None:
                return tensor
    return None


def _extract_timeouts(info: Any, device: torch.device, num_envs: int) -> torch.Tensor:
    if isinstance(info, dict):
        for key in ("time_outs", "timeouts", "time_out", "time_limit"):
            tensor = _as_bool_tensor(info.get(key), device, num_envs)
            if tensor is not None:
                return tensor
    return torch.zeros(num_envs, dtype=torch.bool, device=device)


def _extract_termination_causes(
    info: Any,
    dones: torch.Tensor,
    ep_lengths: torch.Tensor,
    max_episode_length: int,
    device: torch.device,
    num_envs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (timeouts, falls) per environment for the current step."""
    done_mask = dones.bool()
    timeouts = _extract_timeouts(info, device, num_envs)
    # Fallback: if timeout flags are not surfaced by wrapper, infer from episode length.
    if not torch.any(timeouts & done_mask):
        inferred_timeouts = done_mask & (ep_lengths >= max(1, max_episode_length - 1))
        timeouts = timeouts | inferred_timeouts
    falls = done_mask & ~timeouts
    return timeouts, falls


def _override_velocity_command(env: Any, vx: float, vy: float = 0.0, wz: float = 0.0, command_name: str = "base_velocity") -> None:
    cmd_term = env.unwrapped.command_manager.get_term(command_name)
    ref = getattr(cmd_term, "vel_command_b", None)
    if ref is None:
        ref = cmd_term.command
    device = ref.device
    num_envs = ref.shape[0]
    fixed = torch.tensor([[vx, vy, wz]], device=device).repeat(num_envs, 1)
    if hasattr(cmd_term, "vel_command_b"):
        cmd_term.vel_command_b[:] = fixed
    if hasattr(cmd_term, "command"):
        try:
            cmd_term.command[:] = fixed
        except (AttributeError, TypeError):
            pass
    if hasattr(cmd_term, "heading_target"):
        cmd_term.heading_target[:] = 0.0
    if hasattr(cmd_term, "is_heading_env"):
        cmd_term.is_heading_env[:] = False
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = False


def _hide_fallen_robots(robot: Any, env_ids: torch.Tensor) -> None:
    """Best-effort visual removal for fallen robots by moving them far below/away."""
    if env_ids.numel() == 0:
        return
    try:
        env_ids = env_ids.to(dtype=torch.long, device=robot.device)
    except Exception:
        env_ids = env_ids.to(dtype=torch.long)
    try:
        root_state = robot.data.root_state_w[env_ids].clone()
    except Exception:
        return
    root_state[:, 0] += 1000.0
    root_state[:, 2] = -5.0
    root_state[:, 7:] = 0.0
    try:
        if hasattr(robot, "write_root_state_to_sim"):
            robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        elif hasattr(robot, "write_root_pose_to_sim"):
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
            if hasattr(robot, "write_root_velocity_to_sim"):
                robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    except Exception:
        pass


def _quat_wxyz_to_rpy(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternion [w, x, y, z] to roll/pitch/yaw in radians."""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _safe_ratio(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    return numer / torch.clamp(denom, min=1.0)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run policy inference and evaluate time-based walking quality metrics."""
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    cmd_x = float(args_cli.cmd_vel_x) if args_cli.cmd_vel_x is not None else float(args_cli.target_velocity)
    cmd_y = float(args_cli.cmd_vel_y)
    cmd_z = float(args_cli.cmd_yaw)

    print(f"Loaded checkpoint: {resume_path}")
    print(f"Task: {args_cli.task}")
    print(f"Num envs: {env_cfg.scene.num_envs}")
    print(f"Eval time [s]: {args_cli.eval_time}")
    print(f"Warmup time [s]: {args_cli.warmup_time}")
    print(f"Tracking window [s]: {args_cli.tracking_window_time}")
    print(f"Fall buffer [s]: {args_cli.fall_buffer_time}")
    print(f"Command override: lin_vel_x={cmd_x}, lin_vel_y={cmd_y}, ang_vel_z={cmd_z}")
    print(
        f"Tracking tolerances: x={args_cli.tracking_tolerance_x}, "
        f"y={args_cli.tracking_tolerance_y}, z={args_cli.tracking_tolerance_z}"
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "evaluation"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    num_envs = env.num_envs
    device = env.unwrapped.device
    obs = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    initial_envs = int(num_envs)
    max_episode_length = int(getattr(env.unwrapped, "max_episode_length", 0) or 0)
    step_dt = float(env.unwrapped.step_dt)
    warmup_steps = max(0, int(round(args_cli.warmup_time / step_dt)))
    eval_steps_target = max(1, int(round(args_cli.eval_time / step_dt)))
    tracking_window_steps = max(1, int(round(args_cli.tracking_window_time / step_dt)))
    fall_buffer_steps = max(0, int(round(args_cli.fall_buffer_time / step_dt)))

    alive_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
    fallen_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    survival_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    per_env_metric_steps = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_sum_vx = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_sum_vy = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_sum_wz = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_sum_err_x = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_sum_err_y = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_sum_err_z = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_sum_sq_err_x = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_sum_sq_err_y = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_sum_sq_err_z = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_max_abs_err_x = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_max_abs_err_y = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_max_abs_err_z = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_sum_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_sum_reward_sq = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_speed_score = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_roll_sq = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_pitch_sq = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_base_height = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_base_height_sq = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_com_height = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_slip_sum = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_contact_sum = torch.zeros(num_envs, dtype=torch.float32, device=device)
    per_env_touchdown_count = torch.zeros(num_envs, dtype=torch.float32, device=device)

    tracking_score_sum_x = torch.zeros(num_envs, dtype=torch.float32, device=device)
    tracking_score_sum_y = torch.zeros(num_envs, dtype=torch.float32, device=device)
    tracking_score_sum_z = torch.zeros(num_envs, dtype=torch.float32, device=device)
    tracking_score_sum_xyz = torch.zeros(num_envs, dtype=torch.float32, device=device)
    tracking_sample_steps = torch.zeros(num_envs, dtype=torch.float32, device=device)

    vel_hist_x = torch.zeros(tracking_window_steps, num_envs, dtype=torch.float32, device=device)
    vel_hist_y = torch.zeros(tracking_window_steps, num_envs, dtype=torch.float32, device=device)
    vel_hist_z = torch.zeros(tracking_window_steps, num_envs, dtype=torch.float32, device=device)
    hist_valid = torch.zeros(tracking_window_steps, num_envs, dtype=torch.float32, device=device)
    hist_ptr = 0
    hist_size = 0
    fall_hist_steps = torch.zeros(max(1, fall_buffer_steps), num_envs, dtype=torch.float32, device=device)
    fall_hist_hit_x = torch.zeros(max(1, fall_buffer_steps), num_envs, dtype=torch.float32, device=device)
    fall_hist_hit_y = torch.zeros(max(1, fall_buffer_steps), num_envs, dtype=torch.float32, device=device)
    fall_hist_hit_z = torch.zeros(max(1, fall_buffer_steps), num_envs, dtype=torch.float32, device=device)
    fall_hist_hit_xyz = torch.zeros(max(1, fall_buffer_steps), num_envs, dtype=torch.float32, device=device)
    fall_hist_ptr = 0
    fall_hist_size = 0

    start_yaw = None
    start_pos = None
    prev_contact = None
    wall_t0 = time.time()

    # Foot-contact and foot-slip helpers (best effort).
    foot_body_ids = None
    try:
        foot_body_ids = robot.find_bodies(".*_foot_link")[0]
    except Exception:
        foot_body_ids = None

    per_second_rows: list[dict[str, float | int | str]] = []
    next_log_second = 1.0
    sim_step = 0
    eval_step = 0
    eval_dir = Path("logs") / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = eval_dir / f"tb_{timestamp}"
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    while simulation_app.is_running():
        if eval_step >= eval_steps_target:
            break
        if int(alive_mask.sum().item()) == 0:
            break

        _override_velocity_command(env, cmd_x, cmd_y, cmd_z, command_name="base_velocity")
        with torch.inference_mode():
            actions = policy(obs)
            if not torch.all(alive_mask):
                actions = actions.clone()
                actions[~alive_mask] = 0.0
            obs, rewards, dones, info = env.step(actions)
            if hasattr(policy_nn, "reset"):
                reset_dones = dones.clone()
                reset_dones[~alive_mask] = False
                policy_nn.reset(reset_dones)

        step_alive_mask = alive_mask.clone()
        episode_steps[step_alive_mask] += 1
        survival_steps[step_alive_mask] += 1

        # Detect falls/timeouts and remove dead environments from now on.
        timeouts, falls = _extract_termination_causes(
            info=info,
            dones=dones,
            ep_lengths=episode_steps,
            max_episode_length=max_episode_length,
            device=device,
            num_envs=num_envs,
        )
        timeout_done = step_alive_mask & timeouts
        if torch.any(timeout_done):
            episode_steps[timeout_done] = 0
        newly_fallen = step_alive_mask & falls
        if torch.any(newly_fallen):
            if fall_buffer_steps > 0 and fall_hist_size > 0:
                fallen_ids = torch.nonzero(newly_fallen, as_tuple=False).flatten()
                recent_steps = torch.sum(fall_hist_steps[:fall_hist_size, :][:, fallen_ids], dim=0)
                recent_hit_x = torch.sum(fall_hist_hit_x[:fall_hist_size, :][:, fallen_ids], dim=0)
                recent_hit_y = torch.sum(fall_hist_hit_y[:fall_hist_size, :][:, fallen_ids], dim=0)
                recent_hit_z = torch.sum(fall_hist_hit_z[:fall_hist_size, :][:, fallen_ids], dim=0)
                recent_hit_xyz = torch.sum(fall_hist_hit_xyz[:fall_hist_size, :][:, fallen_ids], dim=0)

                tracking_sample_steps[fallen_ids] = torch.clamp(tracking_sample_steps[fallen_ids] - recent_steps, min=0.0)
                tracking_score_sum_x[fallen_ids] = torch.clamp(tracking_score_sum_x[fallen_ids] - recent_hit_x, min=0.0)
                tracking_score_sum_y[fallen_ids] = torch.clamp(tracking_score_sum_y[fallen_ids] - recent_hit_y, min=0.0)
                tracking_score_sum_z[fallen_ids] = torch.clamp(tracking_score_sum_z[fallen_ids] - recent_hit_z, min=0.0)
                tracking_score_sum_xyz[fallen_ids] = torch.clamp(tracking_score_sum_xyz[fallen_ids] - recent_hit_xyz, min=0.0)

                # Prevent double subtraction if multiple envs fall over nearby steps.
                fall_hist_steps[:fall_hist_size, :][:, fallen_ids] = 0.0
                fall_hist_hit_x[:fall_hist_size, :][:, fallen_ids] = 0.0
                fall_hist_hit_y[:fall_hist_size, :][:, fallen_ids] = 0.0
                fall_hist_hit_z[:fall_hist_size, :][:, fallen_ids] = 0.0
                fall_hist_hit_xyz[:fall_hist_size, :][:, fallen_ids] = 0.0

            fallen_mask = fallen_mask | newly_fallen
            alive_mask = alive_mask & ~newly_fallen
            _hide_fallen_robots(robot, torch.nonzero(newly_fallen, as_tuple=False).flatten())
        if torch.any(fallen_mask):
            _hide_fallen_robots(robot, torch.nonzero(fallen_mask, as_tuple=False).flatten())

        # Warmup period: do not accumulate metrics.
        sim_step += 1
        if sim_step <= warmup_steps:
            continue

        eval_step += 1
        metric_mask = alive_mask.clone()
        metric_mask_f = metric_mask.float()
        metric_steps_now = float(metric_mask.sum().item())
        if metric_steps_now <= 0.0:
            continue

        lin_vel_b = robot.data.root_lin_vel_b
        ang_vel_b = robot.data.root_ang_vel_b
        root_state_w = robot.data.root_state_w

        quat_wxyz = root_state_w[:, 3:7]
        roll, pitch, yaw = _quat_wxyz_to_rpy(quat_wxyz)
        base_height = root_state_w[:, 2]
        com_height = base_height
        if hasattr(robot.data, "com_pos_w"):
            try:
                com_pos_w = robot.data.com_pos_w
                if com_pos_w.ndim >= 3:
                    com_height = torch.mean(com_pos_w[..., 2], dim=1)
                elif com_pos_w.ndim == 2 and com_pos_w.shape[1] >= 3:
                    com_height = com_pos_w[:, 2]
            except Exception:
                pass

        if start_yaw is None:
            start_yaw = yaw.clone()
        if start_pos is None:
            start_pos = root_state_w[:, 0:3].clone()

        err_x = lin_vel_b[:, 0] - cmd_x
        err_y = lin_vel_b[:, 1] - cmd_y
        err_z = ang_vel_b[:, 2] - cmd_z
        abs_err_x = torch.abs(err_x)
        abs_err_y = torch.abs(err_y)
        abs_err_z = torch.abs(err_z)

        # Moving-average velocity tracking to align with GUI-observed sustained behavior.
        vel_hist_x[hist_ptr] = lin_vel_b[:, 0]
        vel_hist_y[hist_ptr] = lin_vel_b[:, 1]
        vel_hist_z[hist_ptr] = ang_vel_b[:, 2]
        hist_valid[hist_ptr] = metric_mask_f
        hist_ptr = (hist_ptr + 1) % tracking_window_steps
        hist_size = min(hist_size + 1, tracking_window_steps)

        hist_valid_sum = torch.sum(hist_valid[:hist_size], dim=0)
        ma_vx = torch.sum(vel_hist_x[:hist_size] * hist_valid[:hist_size], dim=0) / torch.clamp(hist_valid_sum, min=1.0)
        ma_vy = torch.sum(vel_hist_y[:hist_size] * hist_valid[:hist_size], dim=0) / torch.clamp(hist_valid_sum, min=1.0)
        ma_vz = torch.sum(vel_hist_z[:hist_size] * hist_valid[:hist_size], dim=0) / torch.clamp(hist_valid_sum, min=1.0)

        tol_x = max(1e-6, float(args_cli.tracking_tolerance_x))
        tol_y = max(1e-6, float(args_cli.tracking_tolerance_y))
        tol_z = max(1e-6, float(args_cli.tracking_tolerance_z))
        scale_x = max(abs(cmd_x), tol_x)
        scale_y = max(abs(cmd_y), tol_y)
        scale_z = max(abs(cmd_z), tol_z)
        eligible_mask = (hist_valid_sum >= float(tracking_window_steps)) & metric_mask
        eligible_mask_f = eligible_mask.float()

        base_x = torch.clamp(1.0 - torch.abs(ma_vx - cmd_x) / scale_x, min=0.0, max=1.0)
        base_y = torch.clamp(1.0 - torch.abs(ma_vy - cmd_y) / scale_y, min=0.0, max=1.0)
        base_z = torch.clamp(1.0 - torch.abs(ma_vz - cmd_z) / scale_z, min=0.0, max=1.0)
        jitter_x = torch.abs(lin_vel_b[:, 0] - ma_vx)
        jitter_y = torch.abs(lin_vel_b[:, 1] - ma_vy)
        jitter_z = torch.abs(ang_vel_b[:, 2] - ma_vz)
        excess_x = torch.clamp(jitter_x - 0.10, min=0.0)
        smooth_x = torch.exp(-(excess_x * excess_x) / 0.01)
        smooth_y = 1.0 - torch.clamp((jitter_y - 0.05) / 0.15, min=0.0, max=1.0)
        smooth_z = 1.0 - torch.clamp((jitter_z - 0.10) / 0.30, min=0.0, max=1.0)

        score_gain = 1.1
        score_x = torch.clamp((base_x * smooth_x) * score_gain, min=0.0, max=1.0) * eligible_mask_f
        score_y = torch.clamp((base_y * smooth_y) * score_gain, min=0.0, max=1.0) * eligible_mask_f
        score_z = torch.clamp((base_z * smooth_z) * score_gain, min=0.0, max=1.0) * eligible_mask_f
        score_xyz = ((score_x + score_y + score_z) / 3.0) * eligible_mask_f

        tracking_sample_steps += metric_mask_f
        tracking_score_sum_x += score_x
        tracking_score_sum_y += score_y
        tracking_score_sum_z += score_z
        tracking_score_sum_xyz += score_xyz
        if fall_buffer_steps > 0:
            fall_hist_steps[fall_hist_ptr] = metric_mask_f
            fall_hist_hit_x[fall_hist_ptr] = score_x
            fall_hist_hit_y[fall_hist_ptr] = score_y
            fall_hist_hit_z[fall_hist_ptr] = score_z
            fall_hist_hit_xyz[fall_hist_ptr] = score_xyz
            fall_hist_ptr = (fall_hist_ptr + 1) % fall_buffer_steps
            fall_hist_size = min(fall_hist_size + 1, fall_buffer_steps)

        per_env_metric_steps += metric_mask_f
        per_env_sum_vx += lin_vel_b[:, 0] * metric_mask_f
        per_env_sum_vy += lin_vel_b[:, 1] * metric_mask_f
        per_env_sum_wz += ang_vel_b[:, 2] * metric_mask_f
        per_env_sum_err_x += err_x * metric_mask_f
        per_env_sum_err_y += err_y * metric_mask_f
        per_env_sum_err_z += err_z * metric_mask_f
        per_env_sum_sq_err_x += (err_x * err_x) * metric_mask_f
        per_env_sum_sq_err_y += (err_y * err_y) * metric_mask_f
        per_env_sum_sq_err_z += (err_z * err_z) * metric_mask_f
        per_env_max_abs_err_x = torch.maximum(per_env_max_abs_err_x, abs_err_x * metric_mask_f)
        per_env_max_abs_err_y = torch.maximum(per_env_max_abs_err_y, abs_err_y * metric_mask_f)
        per_env_max_abs_err_z = torch.maximum(per_env_max_abs_err_z, abs_err_z * metric_mask_f)
        per_env_sum_reward += rewards * metric_mask_f
        per_env_sum_reward_sq += (rewards * rewards) * metric_mask_f
        per_env_roll_sq += (roll * roll) * metric_mask_f
        per_env_pitch_sq += (pitch * pitch) * metric_mask_f
        per_env_base_height += base_height * metric_mask_f
        per_env_base_height_sq += (base_height * base_height) * metric_mask_f
        per_env_com_height += com_height * metric_mask_f

        speed_score_x = torch.clamp(1.0 - abs_err_x / max(abs(cmd_x), 0.05), min=0.0, max=1.0)
        speed_score_y = torch.clamp(1.0 - abs_err_y / max(abs(cmd_y), 0.05), min=0.0, max=1.0)
        speed_score_z = torch.clamp(1.0 - abs_err_z / max(abs(cmd_z), 0.10), min=0.0, max=1.0)
        per_env_speed_score += (0.6 * speed_score_x + 0.2 * speed_score_y + 0.2 * speed_score_z) * metric_mask_f

        # Foot-slip / duty-factor / step-frequency / stride-length (best effort).
        if foot_body_ids is not None and len(foot_body_ids) > 0 and hasattr(robot.data, "body_lin_vel_w") and hasattr(robot.data, "body_pos_w"):
            try:
                foot_vel_xy = torch.linalg.norm(robot.data.body_lin_vel_w[:, foot_body_ids, :2], dim=-1)
                foot_height = robot.data.body_pos_w[:, foot_body_ids, 2]
                contact_now = (foot_height < 0.06).float() * metric_mask_f.unsqueeze(1)
                per_env_slip_sum += torch.sum(foot_vel_xy * contact_now, dim=1)
                per_env_contact_sum += torch.mean(contact_now, dim=1)
                if prev_contact is not None:
                    touchdowns = ((prev_contact < 0.5) & (contact_now > 0.5)).float()
                    per_env_touchdown_count += torch.sum(touchdowns, dim=1)
                prev_contact = contact_now
            except Exception:
                pass

        # Time-series logging each second.
        elapsed_eval_time = eval_step * step_dt
        should_log = elapsed_eval_time + 1e-9 >= next_log_second
        should_end = eval_step >= eval_steps_target or int(alive_mask.sum().item()) == 0
        if should_log or should_end:
            alive_count = int(alive_mask.sum().item())
            fallen_count = int(fallen_mask.sum().item())
            alive_rate = float(alive_count / initial_envs)
            fall_rate = float(fallen_count / initial_envs)

            total_steps = torch.sum(per_env_metric_steps)
            mean_vx = float(torch.sum(per_env_sum_vx) / torch.clamp(total_steps, min=1.0))
            mean_vy = float(torch.sum(per_env_sum_vy) / torch.clamp(total_steps, min=1.0))
            mean_wz = float(torch.sum(per_env_sum_wz) / torch.clamp(total_steps, min=1.0))
            mean_ex = float(torch.sum(per_env_sum_err_x) / torch.clamp(total_steps, min=1.0))
            mean_ey = float(torch.sum(per_env_sum_err_y) / torch.clamp(total_steps, min=1.0))
            mean_ez = float(torch.sum(per_env_sum_err_z) / torch.clamp(total_steps, min=1.0))
            rmse_x = float(torch.sqrt(torch.sum(per_env_sum_sq_err_x) / torch.clamp(total_steps, min=1.0)))
            rmse_y = float(torch.sqrt(torch.sum(per_env_sum_sq_err_y) / torch.clamp(total_steps, min=1.0)))
            rmse_z = float(torch.sqrt(torch.sum(per_env_sum_sq_err_z) / torch.clamp(total_steps, min=1.0)))
            tracking_capacity = max(1.0, float(torch.sum(tracking_sample_steps).item()))
            track_x = float(torch.sum(tracking_score_sum_x).item() / tracking_capacity)
            track_y = float(torch.sum(tracking_score_sum_y).item() / tracking_capacity)
            track_z = float(torch.sum(tracking_score_sum_z).item() / tracking_capacity)
            track_overall = float(torch.sum(tracking_score_sum_xyz).item() / tracking_capacity)
            avg_reward = float(torch.sum(per_env_sum_reward) / torch.clamp(total_steps, min=1.0))
            speed_score = float(torch.sum(per_env_speed_score) / torch.clamp(total_steps, min=1.0))
            roll_rms = float(torch.sqrt(torch.sum(per_env_roll_sq) / torch.clamp(total_steps, min=1.0)))
            pitch_rms = float(torch.sqrt(torch.sum(per_env_pitch_sq) / torch.clamp(total_steps, min=1.0)))
            height_mean = float(torch.sum(per_env_base_height) / torch.clamp(total_steps, min=1.0))
            height_rms = float(torch.sqrt(torch.sum(per_env_base_height_sq) / torch.clamp(total_steps, min=1.0)))
            com_height = float(torch.sum(per_env_com_height) / torch.clamp(total_steps, min=1.0))
            duty_factor = float(torch.sum(per_env_contact_sum) / torch.clamp(total_steps, min=1.0))
            foot_slip = float(torch.sum(per_env_slip_sum) / torch.clamp(total_steps, min=1.0))
            step_frequency = float(torch.sum(per_env_touchdown_count) / max(1e-6, elapsed_eval_time * initial_envs))

            # Approx stride length from total forward displacement / touchdown count.
            cur_pos = root_state_w[:, 0:3]
            stride_denom = max(1.0, float(torch.sum(per_env_touchdown_count).item()))
            stride_length = float(torch.sum(torch.abs(cur_pos[:, 0] - start_pos[:, 0])) / stride_denom)

            # Stability score focuses on posture and height consistency.
            norm_roll = min(1.0, roll_rms / 0.35)
            norm_pitch = min(1.0, pitch_rms / 0.35)
            norm_height = min(1.0, abs(height_rms - height_mean) / 0.10)
            stability_score = max(0.0, 1.0 - (0.4 * norm_roll + 0.4 * norm_pitch + 0.2 * norm_height))

            reward_score = 1.0 / (1.0 + math.exp(-avg_reward))
            composite_score = (
                0.4 * alive_rate
                + 0.3 * speed_score
                + 0.2 * stability_score
                + 0.1 * reward_score
            )

            row = {
                "time": round(min(elapsed_eval_time, args_cli.eval_time), 3),
                "alive_envs": alive_count,
                "alive_rate": alive_rate,
                "fall_rate": fall_rate,
                "alive_curve": alive_rate,
                "tracking_rate": track_overall,
                "tracking_x": track_x,
                "tracking_y": track_y,
                "tracking_z": track_z,
                "command_x": cmd_x,
                "actual_x": mean_vx,
                "error_x": mean_ex,
                "rmse_x": rmse_x,
                "max_error_x": float(torch.max(per_env_max_abs_err_x).item()),
                "command_y": cmd_y,
                "actual_y": mean_vy,
                "error_y": mean_ey,
                "rmse_y": rmse_y,
                "max_error_y": float(torch.max(per_env_max_abs_err_y).item()),
                "command_z": cmd_z,
                "actual_z": mean_wz,
                "error_z": mean_ez,
                "rmse_z": rmse_z,
                "max_error_z": float(torch.max(per_env_max_abs_err_z).item()),
                "average_reward": avg_reward,
                "speed_score": speed_score,
                "stability_score": stability_score,
                "composite_score": composite_score,
                "roll_rms": roll_rms,
                "pitch_rms": pitch_rms,
                "base_height_rms": height_rms,
                "base_height_mean": height_mean,
                "com_height": com_height,
                "foot_slip": foot_slip,
                "duty_factor": duty_factor,
                "step_frequency": step_frequency,
                "stride_length": stride_length,
            }
            per_second_rows.append(row)

            log_t = int(round(elapsed_eval_time))
            tb_writer.add_scalar("evaluation/alive_rate", alive_rate, log_t)
            tb_writer.add_scalar("evaluation/alive_curve", alive_rate, log_t)
            tb_writer.add_scalar("evaluation/fall_rate", fall_rate, log_t)
            tb_writer.add_scalar("evaluation/tracking_rate", track_overall, log_t)
            tb_writer.add_scalar("evaluation/forward_tracking_rate", track_x, log_t)
            tb_writer.add_scalar("evaluation/x_tracking_rate", track_x, log_t)
            tb_writer.add_scalar("evaluation/y_tracking_rate", track_y, log_t)
            tb_writer.add_scalar("evaluation/z_tracking_rate", track_z, log_t)
            tb_writer.add_scalar("evaluation/alive_robots", alive_count, log_t)
            tb_writer.add_scalar("evaluation/fallen_robots", fallen_count, log_t)
            tb_writer.add_scalar("evaluation/x_error", mean_ex, log_t)
            tb_writer.add_scalar("evaluation/forward_velocity_error", mean_ex, log_t)
            tb_writer.add_scalar("evaluation/y_error", mean_ey, log_t)
            tb_writer.add_scalar("evaluation/z_error", mean_ez, log_t)
            tb_writer.add_scalar("evaluation/x_rmse", rmse_x, log_t)
            tb_writer.add_scalar("evaluation/y_rmse", rmse_y, log_t)
            tb_writer.add_scalar("evaluation/z_rmse", rmse_z, log_t)
            tb_writer.add_scalar("evaluation/target_velocity", cmd_x, log_t)
            tb_writer.add_scalar("evaluation/forward_velocity", mean_vx, log_t)
            tb_writer.add_scalar("evaluation/x_velocity", mean_vx, log_t)
            tb_writer.add_scalar("evaluation/y_velocity", mean_vy, log_t)
            tb_writer.add_scalar("evaluation/z_velocity", mean_wz, log_t)
            tb_writer.add_scalar("evaluation/reward", avg_reward, log_t)
            tb_writer.add_scalar("evaluation/survival_time", float(torch.mean(survival_steps.float()).item() * step_dt), log_t)
            tb_writer.add_scalar("evaluation/step_frequency", step_frequency, log_t)
            tb_writer.add_scalar("evaluation/base_height", height_mean, log_t)
            tb_writer.add_scalar("evaluation/pitch_rms", pitch_rms, log_t)
            tb_writer.add_scalar("evaluation/roll_rms", roll_rms, log_t)
            tb_writer.add_scalar("evaluation/speed_score", speed_score, log_t)
            tb_writer.add_scalar("evaluation/stability_score", stability_score, log_t)
            tb_writer.add_scalar("evaluation/composite_score", composite_score, log_t)
            tb_writer.add_scalar("evaluation/alive_envs", alive_count, log_t)
            next_log_second = float(int(elapsed_eval_time) + 1)

    env.close()

    total_eval_time = min(eval_step * step_dt, float(args_cli.eval_time))
    alive_count = int(alive_mask.sum().item())
    fallen_count = int(fallen_mask.sum().item())
    alive_rate = float(alive_count / initial_envs)
    fall_rate = float(fallen_count / initial_envs)

    total_steps = torch.sum(per_env_metric_steps)
    mean_vx = float(torch.sum(per_env_sum_vx) / torch.clamp(total_steps, min=1.0))
    mean_vy = float(torch.sum(per_env_sum_vy) / torch.clamp(total_steps, min=1.0))
    mean_wz = float(torch.sum(per_env_sum_wz) / torch.clamp(total_steps, min=1.0))
    mean_ex = float(torch.sum(per_env_sum_err_x) / torch.clamp(total_steps, min=1.0))
    mean_ey = float(torch.sum(per_env_sum_err_y) / torch.clamp(total_steps, min=1.0))
    mean_ez = float(torch.sum(per_env_sum_err_z) / torch.clamp(total_steps, min=1.0))
    rmse_x = float(torch.sqrt(torch.sum(per_env_sum_sq_err_x) / torch.clamp(total_steps, min=1.0)))
    rmse_y = float(torch.sqrt(torch.sum(per_env_sum_sq_err_y) / torch.clamp(total_steps, min=1.0)))
    rmse_z = float(torch.sqrt(torch.sum(per_env_sum_sq_err_z) / torch.clamp(total_steps, min=1.0)))
    tracking_capacity = max(1.0, float(torch.sum(tracking_sample_steps).item()))
    track_x = float(torch.sum(tracking_score_sum_x).item() / tracking_capacity)
    track_y = float(torch.sum(tracking_score_sum_y).item() / tracking_capacity)
    track_z = float(torch.sum(tracking_score_sum_z).item() / tracking_capacity)
    track_overall = float(torch.sum(tracking_score_sum_xyz).item() / tracking_capacity)
    avg_reward = float(torch.sum(per_env_sum_reward) / torch.clamp(total_steps, min=1.0))
    reward_std = float(
        torch.sqrt(
            torch.clamp(
                (torch.sum(per_env_sum_reward_sq) / torch.clamp(total_steps, min=1.0))
                - (torch.sum(per_env_sum_reward) / torch.clamp(total_steps, min=1.0)) ** 2,
                min=0.0,
            )
        )
    )
    speed_score = float(torch.sum(per_env_speed_score) / torch.clamp(total_steps, min=1.0))
    roll_rms = float(torch.sqrt(torch.sum(per_env_roll_sq) / torch.clamp(total_steps, min=1.0)))
    pitch_rms = float(torch.sqrt(torch.sum(per_env_pitch_sq) / torch.clamp(total_steps, min=1.0)))
    height_mean = float(torch.sum(per_env_base_height) / torch.clamp(total_steps, min=1.0))
    height_rms = float(torch.sqrt(torch.sum(per_env_base_height_sq) / torch.clamp(total_steps, min=1.0)))
    com_height = float(torch.sum(per_env_com_height) / torch.clamp(total_steps, min=1.0))
    foot_slip = float(torch.sum(per_env_slip_sum) / torch.clamp(total_steps, min=1.0))
    duty_factor = float(torch.sum(per_env_contact_sum) / torch.clamp(total_steps, min=1.0))
    step_frequency = float(torch.sum(per_env_touchdown_count) / max(1e-6, total_eval_time * initial_envs))
    stride_denom = max(1.0, float(torch.sum(per_env_touchdown_count).item()))
    cur_pos = robot.data.root_state_w[:, 0:3]
    stride_length = float(torch.sum(torch.abs(cur_pos[:, 0] - start_pos[:, 0])) / stride_denom) if start_pos is not None else 0.0

    yaw_drift = 0.0
    if start_yaw is not None:
        cur_yaw = _quat_wxyz_to_rpy(robot.data.root_state_w[:, 3:7])[2]
        yaw_drift = float(torch.mean(torch.abs(cur_yaw - start_yaw)).item())

    per_env_survival_time = survival_steps.float() * step_dt
    survival_mean = float(torch.mean(per_env_survival_time).item())
    survival_median = float(torch.quantile(per_env_survival_time, 0.5).item())
    survival_min = float(torch.min(per_env_survival_time).item())
    survival_max = float(torch.max(per_env_survival_time).item())

    norm_roll = min(1.0, roll_rms / 0.35)
    norm_pitch = min(1.0, pitch_rms / 0.35)
    norm_height = min(1.0, abs(height_rms - height_mean) / 0.10)
    stability_score = max(0.0, 1.0 - (0.4 * norm_roll + 0.4 * norm_pitch + 0.2 * norm_height))
    reward_score = 1.0 / (1.0 + math.exp(-avg_reward))
    composite_score = 0.4 * alive_rate + 0.3 * speed_score + 0.2 * stability_score + 0.1 * reward_score

    csv_path = eval_dir / f"evaluation_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "time", "alive_envs", "alive_rate", "fall_rate", "alive_curve",
            "tracking_rate", "tracking_x", "tracking_y", "tracking_z",
            "command_x", "actual_x", "error_x", "rmse_x", "max_error_x",
            "command_y", "actual_y", "error_y", "rmse_y", "max_error_y",
            "command_z", "actual_z", "error_z", "rmse_z", "max_error_z",
            "average_reward", "speed_score", "stability_score", "composite_score",
            "roll_rms", "pitch_rms", "base_height_rms", "base_height_mean", "com_height",
            "foot_slip", "duty_factor", "step_frequency", "stride_length",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_second_rows)
        writer.writerow(
            {
                "time": "summary",
                "alive_envs": alive_count,
                "alive_rate": alive_rate,
                "fall_rate": fall_rate,
                "alive_curve": alive_rate,
                "tracking_rate": track_overall,
                "tracking_x": track_x,
                "tracking_y": track_y,
                "tracking_z": track_z,
                "command_x": cmd_x,
                "actual_x": mean_vx,
                "error_x": mean_ex,
                "rmse_x": rmse_x,
                "max_error_x": float(torch.max(per_env_max_abs_err_x).item()),
                "command_y": cmd_y,
                "actual_y": mean_vy,
                "error_y": mean_ey,
                "rmse_y": rmse_y,
                "max_error_y": float(torch.max(per_env_max_abs_err_y).item()),
                "command_z": cmd_z,
                "actual_z": mean_wz,
                "error_z": mean_ez,
                "rmse_z": rmse_z,
                "max_error_z": float(torch.max(per_env_max_abs_err_z).item()),
                "average_reward": avg_reward,
                "speed_score": speed_score,
                "stability_score": stability_score,
                "composite_score": composite_score,
                "roll_rms": roll_rms,
                "pitch_rms": pitch_rms,
                "base_height_rms": height_rms,
                "base_height_mean": height_mean,
                "com_height": com_height,
                "foot_slip": foot_slip,
                "duty_factor": duty_factor,
                "step_frequency": step_frequency,
                "stride_length": stride_length,
            }
        )

    print("====================")
    print("Evaluation Summary")
    print("====================")
    print("Overall")
    print(f"Evaluation Time: {total_eval_time:.3f} s")
    print(f"Alive Rate: {alive_rate * 100.0:.2f} %")
    print(f"Fall Rate: {fall_rate * 100.0:.2f} %")
    print(f"Remaining Robots: {alive_count}/{initial_envs}")
    print(f"Overall Tracking Rate: {track_overall * 100.0:.2f} %")
    print(f"Speed Tracking Score: {speed_score:.3f}")
    print(f"Stability Score: {stability_score:.3f}")
    print(f"Composite Score: {composite_score:.3f}")
    print("---")
    print("Forward (X)")
    print(f"Command: {cmd_x:.3f} m/s")
    print(f"Mean Velocity: {mean_vx:.3f} m/s")
    print(f"Mean Error: {mean_ex:.3f} m/s")
    print(f"RMSE: {rmse_x:.3f} m/s")
    print(f"Tracking Rate: {track_x * 100.0:.2f} %")
    print(f"Max Error: {float(torch.max(per_env_max_abs_err_x).item()):.3f} m/s")
    print("---")
    print("Lateral (Y)")
    print(f"Command: {cmd_y:.3f} m/s")
    print(f"Mean Velocity: {mean_vy:.3f} m/s")
    print(f"Mean Error: {mean_ey:.3f} m/s")
    print(f"RMSE: {rmse_y:.3f} m/s")
    print(f"Tracking Rate: {track_y * 100.0:.2f} %")
    print(f"Max Error: {float(torch.max(per_env_max_abs_err_y).item()):.3f} m/s")
    print("---")
    print("Yaw (Z)")
    print(f"Command: {cmd_z:.3f} rad/s")
    print(f"Mean Angular Velocity: {mean_wz:.3f} rad/s")
    print(f"Mean Error: {mean_ez:.3f} rad/s")
    print(f"RMSE: {rmse_z:.3f} rad/s")
    print(f"Tracking Rate: {track_z * 100.0:.2f} %")
    print(f"Max Error: {float(torch.max(per_env_max_abs_err_z).item()):.3f} rad/s")
    print("---")
    print("Stability / Gait")
    print(f"Roll RMS: {roll_rms:.3f} rad")
    print(f"Pitch RMS: {pitch_rms:.3f} rad")
    print(f"Base Height RMS: {height_rms:.3f} m")
    print(f"CoM Height: {com_height:.3f} m")
    print(f"Yaw Drift: {yaw_drift:.3f} rad")
    print(f"Foot Slip: {foot_slip:.3f} m/s")
    print(f"Duty Factor: {duty_factor:.3f}")
    print(f"Step Frequency: {step_frequency:.3f} Hz")
    print(f"Stride Length: {stride_length:.3f} m")
    print("---")
    print("Reward / Survival")
    print(f"Average Reward: {avg_reward:.6f}")
    print(f"Reward Std: {reward_std:.6f}")
    print(f"Average Survival Time: {survival_mean:.3f} s")
    print(f"Median Survival Time: {survival_median:.3f} s")
    print(f"Min Survival Time: {survival_min:.3f} s")
    print(f"Max Survival Time: {survival_max:.3f} s")
    print("====================")
    print(f"Wall time elapsed: {time.time() - wall_t0:.3f} s")
    print("==================================")
    print(f"CSV saved: {csv_path.resolve()}")
    print(f"TensorBoard logdir: {tb_dir.resolve()}")
    tb_writer.close()


if __name__ == "__main__":
    main()
    simulation_app.close()