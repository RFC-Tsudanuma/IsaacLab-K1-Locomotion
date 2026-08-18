# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール履歴版の ``is_engaged`` が意図通り判定しているかを実測する。

判定基準:
    * 静止ボール条件 (ball_speed 0) → **出動率はほぼ 0** であるべき
      (横ずれが大きい配置だけ出動する)
    * 通常のシュート条件           → 球が飛んでいる間は出動しているべき
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Check ballhist is_engaged behaviour.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="Isaac-GoalkeeperBallHist-K1-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--warmup", type=int, default=200)
parser.add_argument("--override_json", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401

from isaaclab_k1_locomotion.tasks.manager_based.goalkeeper.ballhist.observations import (
    ballhist_is_engaged,
)
from isaaclab_k1_locomotion.tasks.manager_based.goalkeeper.mdp.observations import gk_buffers


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.override_json is not None:
        from config_overrides import apply_overrides_from_file
        apply_overrides_from_file(args_cli.override_json, env_cfg=env_cfg, agent_cfg=agent_cfg)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else \
        get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    inner_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(inner_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=inner_env.unwrapped.device)
    raw = inner_env.unwrapped

    eng, act_ball = [], []
    obs = inner_env.get_observations()
    with torch.inference_mode():
        for i in range(args_cli.steps + args_cli.warmup):
            a = policy(obs)
            if i >= args_cli.warmup:
                eng.append(ballhist_is_engaged(raw).clone())
                act_ball.append(gk_buffers(raw)["ball_active"].clone())
            obs, _, _, _ = inner_env.step(a)

    e = torch.stack(eng).float()
    b = torch.stack(act_ball).float()
    print("\n" + "=" * 60)
    print(f"task           : {args_cli.task}")
    print(f"sampled        : {e.shape[0]} x {e.shape[1]} env")
    print(f"出動率 (全体)   : {e.mean().item():.1%}")
    print(f"ball_active 率  : {b.mean().item():.1%}")
    if b.sum() > 0:
        print(f"球が生きている間の出動率 : {(e * b).sum().item() / b.sum().item():.1%}")
    if (1 - b).sum() > 0:
        print(f"球が無い間の出動率       : {(e * (1 - b)).sum().item() / (1 - b).sum().item():.1%}")
    # --- 球が飛んでいる間に出動フラグが落ちる頻度 ---
    #
    # ★ ここが本題。is_engaged は「検出できている」を条件に含むので、知覚が
    #   途切れると false になり、gait_phase がゼロ埋めされて歩行が止まる。
    #   セーブの最中に止まると横移動を失う (1.3m/s x 停止時間) ので、
    #   落ちる頻度と長さが「is_engaged を外す価値」の判断材料になる。
    eb = e.bool() & b.bool()          # 球が生きていて出動中
    drop = (~e.bool()) & b.bool()     # 球が生きているのに非出動
    # on -> off の遷移回数 (球が生きている間だけ数える)
    trans = ((e[:-1].bool() & (~e[1:].bool())) & b[1:].bool()).sum().item()
    n_balls = max(1.0, (b[1:] > b[:-1]).sum().item())
    print(f"球の飛行中に非出動だった割合 : {drop.float().sum().item() / max(b.sum().item(), 1):.1%}")
    print(f"出動が落ちた回数 (on->off)   : {int(trans)}  = 1球あたり {trans / n_balls:.2f} 回")
    # 最長の連続 off ストリーク [step]
    longest = 0
    cur = torch.zeros(e.shape[1], device=e.device)
    for t in range(e.shape[0]):
        cur = torch.where(drop[t], cur + 1, torch.zeros_like(cur))
        longest = max(longest, int(cur.max().item()))
    print(f"最長の連続非出動             : {longest} step = {longest * 0.02:.2f} 秒")
    print("=" * 60 + "\n")
    inner_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
