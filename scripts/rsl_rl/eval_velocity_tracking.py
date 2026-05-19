
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity tracking evaluation for RSL-RL policies on K1 (Isaac Lab)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Velocity tracking evaluation for RSL-RL agent.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--settle_time", type=float, default=0.0)
parser.add_argument("--measure_time", type=float, default=20.0)
parser.add_argument("--high_speed", type=float, default=1.0, help="高速コマンド [m/s]")
parser.add_argument("--low_speed", type=float, default=0.4, help="低速コマンド [m/s]")
parser.add_argument("--high_ang", type=float, default=6.24, help="高速角速度 [rad/s]")
parser.add_argument("--low_ang", type=float, default=0.4, help="低速角速度 [rad/s]")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

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

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401


def build_commands(high_speed: float, low_speed: float, high_ang: float, low_ang: float):
    """評価コマンドリストを構築: (label, vx, vy, wz)"""
    commands = []

    # 線速度のみ: high_speedは前のみ、low_speedは前後左右
    commands.append((f"前   {high_speed:.1f} m/s", +high_speed, 0.0, 0.0))
    for speed in (low_speed,):
        commands += [
            (f"前   {speed:.1f} m/s", +speed, 0.0, 0.0),
            (f"後   {speed:.1f} m/s", -speed, 0.0, 0.0),
            (f"左   {speed:.1f} m/s",  0.0, +speed, 0.0),
            (f"右   {speed:.1f} m/s",  0.0, -speed, 0.0),
        ]

    # 角速度のみ (その場旋回 × {±high_ang, ±low_ang})
    for wz in (+high_ang, -high_ang, +low_ang, -low_ang):
        sign = "左" if wz > 0 else "右"
        commands.append((f"旋回{sign} wz={wz:+.1f} rad/s", 0.0, 0.0, wz))

    # 前high_speed + wz ±low_ang
    for wz in (+low_ang, -low_ang):
        sign = "左" if wz > 0 else "右"
        commands.append((f"前+旋回{sign} vx={high_speed:.1f} wz={wz:+.1f}", +high_speed, 0.0, wz))

    # 前low_speed + wz ±low_ang
    for wz in (+low_ang, -low_ang):
        sign = "左" if wz > 0 else "右"
        commands.append((f"前+旋回{sign} vx={low_speed:.1f} wz={wz:+.1f}", +low_speed, 0.0, wz))

    # vy + wz の複合
    for wz in (+low_ang, -low_ang):
        sign = "左" if wz > 0 else "右"
        commands.append((f"左+旋回{sign} vy={low_speed:.1f} wz={wz:+.1f}", 0.0, +low_speed, wz))

    return commands


def override_command(env, vx: float, vy: float, wz: float) -> None:
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
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


def run_one_command(env, policy, policy_nn, label, vx, vy, wz, settle_steps, measure_steps):
    print(f"\n▶ コマンド: {label}  (vx={vx:+.2f}, vy={vy:+.2f}, wz={wz:+.2f})")

    obs, _ = env.reset()
    if hasattr(policy_nn, "reset"):
        policy_nn.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))

    for _ in range(settle_steps):
        override_command(env, vx, vy, wz)
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

    lin_log, ang_log = [], []
    for _ in range(measure_steps):
        override_command(env, vx, vy, wz)
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

        robot = env.unwrapped.scene["robot"]
        lin_log.append(robot.data.root_lin_vel_b[:, :2].detach().cpu().numpy())
        ang_log.append(robot.data.root_ang_vel_b[:, 2].detach().cpu().numpy())

    lins = np.stack(lin_log, axis=0)
    angs = np.stack(ang_log, axis=0)

    cmd_lin = np.array([vx, vy], dtype=np.float64)
    cmd_lin_norm = np.linalg.norm(cmd_lin)
    if cmd_lin_norm > 1e-6:
        axis = cmd_lin / cmd_lin_norm
        along = lins.reshape(-1, 2) @ axis
        mean_along = float(along.mean())
        lin_tracking = float(mean_along / cmd_lin_norm)
    else:
        mean_along = float("nan")
        lin_tracking = float("nan")

    mean_wz = float(angs.mean())
    ang_tracking = float(mean_wz / wz) if abs(wz) > 1e-6 else float("nan")

    return {
        "label": label,
        "cmd_vx": vx, "cmd_vy": vy, "cmd_wz": wz,
        "cmd_lin_norm": cmd_lin_norm,
        "mean_along": mean_along,
        "lin_tracking": lin_tracking,
        "mean_wz": mean_wz,
        "ang_tracking": ang_tracking,
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s,
        args_cli.settle_time + args_cli.measure_time + 1.0,
    )
    try:
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    except AttributeError:
        pass
    try:
        env_cfg.commands.base_velocity.heading_command = False
    except AttributeError:
        pass

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {resume_path}")
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

    dt = env.unwrapped.step_dt
    settle_steps = int(args_cli.settle_time / dt)
    measure_steps = int(args_cli.measure_time / dt)
    print(f"[INFO] dt={dt:.4f}s, settle={settle_steps} steps, measure={measure_steps} steps")
    print(f"[INFO] high_speed={args_cli.high_speed}, low_speed={args_cli.low_speed}, "
          f"high_ang={args_cli.high_ang}, low_ang={args_cli.low_ang}")

    COMMANDS = build_commands(args_cli.high_speed, args_cli.low_speed, args_cli.high_ang, args_cli.low_ang)

    results = []
    for label, vx, vy, wz in COMMANDS:
        row = run_one_command(env, policy, policy_nn, label, vx, vy, wz, settle_steps, measure_steps)
        results.append(row)

    # ---- 結果表示 --------------------------------------------------------
    header = f"  (settle {args_cli.settle_time:.1f}s / measure {args_cli.measure_time:.1f}s, num_envs={env.num_envs})"
    sep = "=" * 70

    def fmt_lin(r):
        if np.isnan(r["lin_tracking"]):
            return f"  {'---':>8}   {'---':>8}   {'---':>8}"
        return (f"  {r['cmd_lin_norm']:>7.2f}  "
                f"  {r['mean_along']:>+8.3f}   "
                f"  {r['lin_tracking']*100:>7.1f} %")

    def fmt_ang(r):
        if np.isnan(r["ang_tracking"]):
            return f"  {'---':>8}   {'---':>8}   {'---':>8}"
        return (f"  {r['cmd_wz']:>+7.2f}  "
                f"  {r['mean_wz']:>+8.3f}   "
                f"  {r['ang_tracking']*100:>7.1f} %")

    print(f"\n{sep}")
    print("  速度追従率 評価結果")
    print(header)
    print(sep)

    print("\n  ■ 線速度のみ (wz=0)")
    print(f"  {'コマンド':<22} {'指令[m/s]':>9} {'実測[m/s]':>10} {'追従率':>10}")
    print("  " + "-" * 54)
    for r in results:
        if abs(r["cmd_wz"]) < 1e-6 and r["cmd_lin_norm"] > 1e-6:
            print(f"  {r['label']:<22}" + fmt_lin(r))

    print("\n  ■ 角速度のみ (vx=vy=0, その場旋回)")
    print(f"  {'コマンド':<28} {'指令[r/s]':>9} {'実測[r/s]':>10} {'追従率':>10}")
    print("  " + "-" * 60)
    for r in results:
        if r["cmd_lin_norm"] < 1e-6 and abs(r["cmd_wz"]) > 1e-6:
            print(f"  {r['label']:<28}" + fmt_ang(r))

    print("\n  ■ 複合コマンド (線速度 + 角速度)")
    print(f"  {'コマンド':<28} "
          f"{'指令lin':>8} {'実測lin':>9} {'追従lin':>9}  "
          f"{'指令wz':>8} {'実測wz':>9} {'追従wz':>9}")
    print("  " + "-" * 82)
    for r in results:
        if r["cmd_lin_norm"] > 1e-6 and abs(r["cmd_wz"]) > 1e-6:
            lin_part = (f"  {r['cmd_lin_norm']:>7.2f}  {r['mean_along']:>+8.3f}  {r['lin_tracking']*100:>7.1f}%"
                        if not np.isnan(r["lin_tracking"]) else "  ---       ---        ---  ")
            ang_part = (f"  {r['cmd_wz']:>+7.2f}  {r['mean_wz']:>+8.3f}  {r['ang_tracking']*100:>7.1f}%"
                        if not np.isnan(r["ang_tracking"]) else "  ---       ---        ---  ")
            print(f"  {r['label']:<28}" + lin_part + ang_part)

    print(f"\n{sep}\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()