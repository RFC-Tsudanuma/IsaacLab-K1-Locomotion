# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL policy with episode-level metrics and CSV export."""

import argparse
import csv
import datetime as dt
import os
import sys
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
parser.add_argument("--target-velocity", type=float, default=0.4, help="Target forward velocity [m/s].")
parser.add_argument("--tracking-tolerance", type=float, default=0.04, help="Tracking tolerance [m/s].")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run policy inference and evaluate episode-level walking metrics."""
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

    print(f"Loaded checkpoint: {resume_path}")
    print(f"Task: {args_cli.task}")
    print(f"Num envs: {env_cfg.scene.num_envs}")
    print(f"Episodes: {args_cli.episodes}")
    print(f"Target velocity: {args_cli.target_velocity}")
    print(f"Tracking tolerance: {args_cli.tracking_tolerance}")

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

    ep_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
    ep_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
    sum_vx = torch.zeros(num_envs, dtype=torch.float32, device=device)
    sum_vy = torch.zeros(num_envs, dtype=torch.float32, device=device)
    sum_wz = torch.zeros(num_envs, dtype=torch.float32, device=device)
    sum_action_norm = torch.zeros(num_envs, dtype=torch.float32, device=device)
    tracking_hit_steps = torch.zeros(num_envs, dtype=torch.float32, device=device)
    sum_abs_vel_error = torch.zeros(num_envs, dtype=torch.float32, device=device)
    sum_sq_vel_error = torch.zeros(num_envs, dtype=torch.float32, device=device)

    # Optional reward components
    sum_command_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
    sum_orientation_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
    sum_feet_air_time_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
    command_reward_seen = False
    orientation_reward_seen = False
    feet_air_time_reward_seen = False

    rows: list[dict[str, float | int]] = []
    optional_episode_stats: dict[str, list[float]] = {
        "command_tracking_reward": [],
        "orientation_reward": [],
        "feet_air_time_reward": [],
    }
    step_count = 0
    eval_dir = Path("logs") / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = eval_dir / f"tb_{timestamp}"
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    while simulation_app.is_running() and len(rows) < args_cli.episodes:
        _override_velocity_command(env, args_cli.target_velocity, 0.0, 0.0, command_name="base_velocity")
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, info = env.step(actions)
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        robot = env.unwrapped.scene["robot"]
        lin_vel_b = robot.data.root_lin_vel_b
        ang_vel_b = robot.data.root_ang_vel_b
        action_norm = torch.linalg.norm(actions, dim=-1)
        vel_error = torch.abs(lin_vel_b[:, 0] - args_cli.target_velocity)
        track_hit = (vel_error <= args_cli.tracking_tolerance).float()

        ep_returns += rewards
        ep_lengths += 1
        sum_vx += lin_vel_b[:, 0]
        sum_vy += lin_vel_b[:, 1]
        sum_wz += ang_vel_b[:, 2]
        sum_action_norm += action_norm
        tracking_hit_steps += track_hit
        sum_abs_vel_error += vel_error
        sum_sq_vel_error += vel_error * vel_error

        log_dict = info.get("log", {}) if isinstance(info, dict) else {}
        if isinstance(log_dict, dict):
            command_rew = _find_metric_tensor(log_dict, ["track", "vel"], device, num_envs)
            orientation_rew = _find_metric_tensor(log_dict, ["orientation"], device, num_envs)
            feet_air_rew = _find_metric_tensor(log_dict, ["feet", "air"], device, num_envs)
            if command_rew is not None:
                sum_command_reward += command_rew
                command_reward_seen = True
            if orientation_rew is not None:
                sum_orientation_reward += orientation_rew
                orientation_reward_seen = True
            if feet_air_rew is not None:
                sum_feet_air_time_reward += feet_air_rew
                feet_air_time_reward_seen = True

        timeouts = _extract_timeouts(info, device, num_envs)
        done_ids = torch.nonzero(dones, as_tuple=False).flatten()
        for env_id_tensor in done_ids:
            if len(rows) >= args_cli.episodes:
                break
            env_id = int(env_id_tensor.item())
            ep_len = int(ep_lengths[env_id].item())
            if ep_len <= 0:
                continue
            fall = 0 if bool(timeouts[env_id].item()) else 1
            rows.append(
                {
                    "episode": len(rows) + 1,
                    "return": float(ep_returns[env_id].item()),
                    "length": ep_len,
                    "fall": fall,
                    "avg_forward_velocity": float((sum_vx[env_id] / ep_len).item()),
                    "avg_lateral_velocity": float((sum_vy[env_id] / ep_len).item()),
                    "avg_yaw_rate": float((sum_wz[env_id] / ep_len).item()),
                    "avg_action_norm": float((sum_action_norm[env_id] / ep_len).item()),
                    "tracking_rate": float((tracking_hit_steps[env_id] / ep_len).item()),
                    "mean_velocity_error": float((sum_abs_vel_error[env_id] / ep_len).item()),
                    "rms_velocity_error": float(torch.sqrt(sum_sq_vel_error[env_id] / ep_len).item()),
                }
            )
            if command_reward_seen:
                optional_episode_stats["command_tracking_reward"].append(float((sum_command_reward[env_id] / ep_len).item()))
            if orientation_reward_seen:
                optional_episode_stats["orientation_reward"].append(float((sum_orientation_reward[env_id] / ep_len).item()))
            if feet_air_time_reward_seen:
                optional_episode_stats["feet_air_time_reward"].append(float((sum_feet_air_time_reward[env_id] / ep_len).item()))

            ep_returns[env_id] = 0.0
            ep_lengths[env_id] = 0
            sum_vx[env_id] = 0.0
            sum_vy[env_id] = 0.0
            sum_wz[env_id] = 0.0
            sum_action_norm[env_id] = 0.0
            tracking_hit_steps[env_id] = 0.0
            sum_abs_vel_error[env_id] = 0.0
            sum_sq_vel_error[env_id] = 0.0
            sum_command_reward[env_id] = 0.0
            sum_orientation_reward[env_id] = 0.0
            sum_feet_air_time_reward[env_id] = 0.0

            completed_returns = np.array([float(r["return"]) for r in rows], dtype=np.float64)
            completed_lengths = np.array([float(r["length"]) for r in rows], dtype=np.float64)
            completed_falls = np.array([float(r["fall"]) for r in rows], dtype=np.float64)
            completed_track = np.array([float(r["tracking_rate"]) for r in rows], dtype=np.float64)
            completed_err = np.array([float(r["mean_velocity_error"]) for r in rows], dtype=np.float64)
            completed_vx = np.array([float(r["avg_forward_velocity"]) for r in rows], dtype=np.float64)
            tb_writer.add_scalar("evaluation/tracking_rate", float(np.mean(completed_track)), len(rows))
            tb_writer.add_scalar("evaluation/fall_rate", float(np.mean(completed_falls)), len(rows))
            tb_writer.add_scalar("evaluation/mean_velocity", float(np.mean(completed_vx)), len(rows))
            tb_writer.add_scalar("evaluation/velocity_error", float(np.mean(completed_err)), len(rows))
            tb_writer.add_scalar("evaluation/reward_mean", float(np.mean(completed_returns)), len(rows))
            tb_writer.add_scalar("evaluation/reward_std", float(np.std(completed_returns)), len(rows))
            tb_writer.add_scalar("evaluation/episode_length", float(np.mean(completed_lengths)), len(rows))

        step_count += 1
        if step_count % 200 == 0:
            print(f"[INFO] Collected episodes: {len(rows)}/{args_cli.episodes}")

    env.close()

    if not rows:
        raise RuntimeError("No episodes were completed during evaluation.")

    returns = np.array([float(row["return"]) for row in rows], dtype=np.float64)
    lengths = np.array([float(row["length"]) for row in rows], dtype=np.float64)
    falls = np.array([float(row["fall"]) for row in rows], dtype=np.float64)
    forward_vel = np.array([float(row["avg_forward_velocity"]) for row in rows], dtype=np.float64)
    lateral_vel = np.array([float(row["avg_lateral_velocity"]) for row in rows], dtype=np.float64)
    yaw_rates = np.array([float(row["avg_yaw_rate"]) for row in rows], dtype=np.float64)
    action_norms = np.array([float(row["avg_action_norm"]) for row in rows], dtype=np.float64)
    tracking_rates = np.array([float(row["tracking_rate"]) for row in rows], dtype=np.float64)
    mean_vel_errors = np.array([float(row["mean_velocity_error"]) for row in rows], dtype=np.float64)
    rms_vel_errors = np.array([float(row["rms_velocity_error"]) for row in rows], dtype=np.float64)

    avg_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    avg_length = float(np.mean(lengths))
    fall_rate = float(np.mean(falls))
    success_rate = float(1.0 - fall_rate)
    avg_forward_velocity = float(np.mean(forward_vel))
    avg_lateral_velocity = float(np.mean(lateral_vel))
    avg_yaw_rate = float(np.mean(yaw_rates))
    avg_action_norm = float(np.mean(action_norms))
    tracking_rate = float(np.mean(tracking_rates))
    mean_velocity_error = float(np.mean(mean_vel_errors))
    rms_velocity_error = float(np.mean(rms_vel_errors))
    command_tracking_score = float(np.mean(optional_episode_stats["command_tracking_reward"])) if optional_episode_stats["command_tracking_reward"] else float("nan")

    csv_path = eval_dir / f"evaluation_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "return",
                "length",
                "fall",
                "avg_forward_velocity",
                "avg_lateral_velocity",
                "avg_yaw_rate",
                "avg_action_norm",
                "tracking_rate",
                "mean_velocity_error",
                "rms_velocity_error",
            ],
        )
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    print(f"Checkpoint: {resume_path}")
    print(f"Task: {args_cli.task}")
    print(f"Episodes: {len(rows)}")
    print("")
    print(f"Tracking Rate: {tracking_rate:.6f}")
    print(f"Mean Velocity Error: {mean_velocity_error:.6f}")
    print(f"RMS Velocity Error: {rms_velocity_error:.6f}")
    print(f"Success Rate: {success_rate:.6f}")
    print(f"Average Return: {avg_return:.6f}")
    print(f"Reward Std: {std_return:.6f}")
    print(f"Average Length: {avg_length:.6f}")
    print(f"Fall Rate: {fall_rate:.6f}")
    print(f"Average Forward Velocity: {avg_forward_velocity:.6f}")
    print(f"Average Lateral Velocity: {avg_lateral_velocity:.6f}")
    print(f"Average Yaw Rate: {avg_yaw_rate:.6f}")
    print(f"Average Action Norm: {avg_action_norm:.6f}")
    if optional_episode_stats["command_tracking_reward"]:
        print(f"Command Tracking Score: {command_tracking_score:.6f}")
    if optional_episode_stats["orientation_reward"]:
        print(f"Average Orientation Reward: {np.mean(optional_episode_stats['orientation_reward']):.6f}")
    if optional_episode_stats["feet_air_time_reward"]:
        print(f"Average Feet Air Time Reward: {np.mean(optional_episode_stats['feet_air_time_reward']):.6f}")
    print("==================================")
    print(f"CSV saved: {csv_path.resolve()}")
    print(f"TensorBoard logdir: {tb_dir.resolve()}")
    tb_writer.close()


if __name__ == "__main__":
    main()
    simulation_app.close()