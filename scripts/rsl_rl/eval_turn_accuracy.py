# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""その場回転タスク (Isaac-Velocity-Flat-Turn) の「最終到達角の精度」を実測する。

学習ログの ``heading_error_deg`` はエピソード全ステップの平均なので、回転中の
大きな誤差を含み **最終精度ではない**。``success_rate`` も「5°以内に一度でも
入ったか」の二値なので、1°で止まるのか 4.9°で止まるのかを区別できない。

本スクリプトは各コマンドについて **再サンプリング直前 (= 保持しきった最後) の
残り角** を最終誤差として記録し、指令角の大きさ別に分布を出す。
MuJoCo での角度帯ごとの体感 (0-50 / 60-120 / 130-180) と直接比較できる。

Usage::

    bash /home/satoshi/workspace/IsaacLab-2.3.2/isaaclab.sh -p eval_turn_accuracy.py \\
        --task Isaac-Velocity-Flat-Turn --num_envs 256 --steps 4000 \\
        --checkpoint $PWD/logs/rsl_rl/k1_turn/<run>/model_N.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Measure final heading accuracy of the turn policy.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--steps", type=int, default=4000, help="Number of env steps to roll out.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401

# 指令角の大きさ [deg] で切る帯。MuJoCo での体感報告に合わせてある。
BANDS = [(0.0, 50.0), (50.0, 90.0), (90.0, 130.0), (130.0, 180.1)]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # play.py / export_policy.py と同じ安全策: runner.load が無言で no-op する
    # ケースがあるので state_dict を直接読み直す。
    raw_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    ckpt_msd = raw_ckpt.get("model_state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else None
    if isinstance(ckpt_msd, dict):
        policy_to_load = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        dev = next(policy_to_load.parameters()).device
        policy_to_load.load_state_dict(
            {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in ckpt_msd.items()}, strict=False
        )

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    dev = env.unwrapped.device
    n = env.unwrapped.num_envs

    # コマンド単位の記録。_elapsed はコマンド再サンプルで 0 に戻るので、
    # 減少を検出した時点の「直前ステップの残り角」が最終誤差になる。
    prev_elapsed = torch.zeros(n, device=dev)
    init_mag = torch.zeros(n, device=dev)      # そのコマンドの指令角 |Δψ_0|
    prev_err = torch.zeros(n, device=dev)      # 直前ステップの残り角
    fresh = torch.ones(n, dtype=torch.bool, device=dev)  # 再サンプル直後フラグ

    rec_mag: list[torch.Tensor] = []
    rec_err: list[torch.Tensor] = []

    # NOTE: この版の RslRlVecEnvWrapper は観測のみを返す (play.py と同じ)。
    obs = env.get_observations()
    print(f"[INFO] rolling out {args_cli.steps} steps x {n} envs ...")
    with torch.inference_mode():
        for i in range(args_cli.steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            err = cmd_term.heading_error.abs()
            elapsed = cmd_term._elapsed

            # 再サンプル検出 (elapsed が減った env)。ウォームアップ後のみ記録する。
            resampled = elapsed < prev_elapsed
            if i > 200 and bool(resampled.any()):
                rec_mag.append(init_mag[resampled].clone())
                rec_err.append(prev_err[resampled].clone())

            # 再サンプル直後の env は、この時点の残り角を指令角として記録する
            capture = resampled | fresh
            init_mag = torch.where(capture, err, init_mag)
            fresh = torch.zeros_like(fresh)

            prev_elapsed = elapsed.clone()
            prev_err = err.clone()

    env.close()

    if not rec_mag:
        print("[WARN] no completed commands recorded; increase --steps")
        return

    mag = torch.rad2deg(torch.cat(rec_mag)).cpu()
    fin = torch.rad2deg(torch.cat(rec_err)).cpu()

    def stats(m: torch.Tensor, f: torch.Tensor, label: str):
        if f.numel() == 0:
            print(f"  {label:>14s} :  (サンプルなし)")
            return
        q = torch.quantile(f, torch.tensor([0.5, 0.9, 0.95]))
        print(
            f"  {label:>14s} : n={f.numel():6d}  mean={f.mean():6.2f}  median={q[0]:6.2f}"
            f"  p90={q[1]:6.2f}  p95={q[2]:6.2f}  max={f.max():7.2f}"
            f"   <2°={100.0 * (f < 2).float().mean():5.1f}%"
            f"  <5°={100.0 * (f < 5).float().mean():5.1f}%"
            f"  >20°={100.0 * (f > 20).float().mean():5.1f}%"
        )

    print("\n" + "=" * 118)
    print("最終到達角の誤差 [deg] (各コマンドの再サンプリング直前 = 保持しきった時点)")
    print("=" * 118)
    stats(mag, fin, "全体")
    print("-" * 118)
    for lo, hi in BANDS:
        sel = (mag >= lo) & (mag < hi)
        stats(mag[sel], fin[sel], f"{lo:.0f}-{hi:.0f}°")
    print("=" * 118)


if __name__ == "__main__":
    main()
    simulation_app.close()
