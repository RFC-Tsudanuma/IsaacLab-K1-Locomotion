# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー Stage 1 ポリシーの実効横移動速度を計測する評価スクリプト。

Stage 1 (ランダム目標 y への到達と停止) の学習済みポリシーを Play 環境で回し、
「目標が再サンプルされてから到達するまで」の移動距離/所要時間から実効横移動速度
v_lat を求める。その値から Stage 3 の適応カリキュラム上限 (セーブ可能な限界初速の
9 割) を逆算して提案する:

    v_ball_cap = 0.9 * v_lat * spawn_dist_min / goal_half_width

使い方 (コンテナ内・リポジトリ直下):
    ./isaaclab.sh -p scripts/rsl_rl/eval_goalkeeper_speed.py \\
        --task Isaac-Goalkeeper-Stage1-K1-Play-v0 \\
        --frozen_checkpoint logs/rsl_rl/k1_flat/main_walk/0524_walk.pt \\
        --checkpoint logs/rsl_rl/k1_goalkeeper_stage1/<run>/model_XXXX.pt \\
        --low_level_obs_group low_level --high_action_clip 0.6 0.8 1.0 \\
        --num_envs 32 --duration_s 60 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate goalkeeper stage-1 lateral speed.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Goalkeeper-Stage1-K1-Play-v0", help="Task name.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--duration_s", type=float, default=60.0, help="Measurement duration [s].")
parser.add_argument("--reach_tol", type=float, default=0.15, help="Reach tolerance for timing [m].")
parser.add_argument(
    "--frozen_checkpoint", type=str, required=True, help="Frozen low-level walking checkpoint."
)
parser.add_argument(
    "--high_action_clip", type=float, nargs=3, default=[0.6, 0.8, 1.0], metavar=("VX", "VY", "WZ"),
    help="Per-axis high-level action clip. Must match training.",
)
parser.add_argument("--low_level_obs_group", type=str, default="low_level")
parser.add_argument("--low_level_cmd_term_name", type=str, default="velocity_commands")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

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

from goalkeeper_helpers import HierarchicalVecEnvWrapper, _build_frozen_policy


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    inner_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    frozen_policy = _build_frozen_policy(
        inner_env, agent_cfg, args_cli.frozen_checkpoint, agent_cfg.device, args_cli.low_level_obs_group
    )
    hier_env = HierarchicalVecEnvWrapper(
        inner_env,
        frozen_policy,
        low_level_obs_group=args_cli.low_level_obs_group,
        low_level_cmd_term_name=args_cli.low_level_cmd_term_name,
        action_clip=args_cli.high_action_clip,
        high_action_dim=3,
    )

    print(f"[INFO] Loading high-level checkpoint: {resume_path}")
    runner = OnPolicyRunner(hier_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=inner_env.unwrapped.device)

    raw_env = inner_env.unwrapped
    dt = raw_env.step_dt
    n = raw_env.num_envs
    device = raw_env.device
    n_steps = int(args_cli.duration_s / dt)

    # 計測バッファ: 目標が変わった時点の (目標との距離, 経過ステップ) を追跡
    prev_target = None
    start_dist = torch.zeros(n, device=device)
    elapsed = torch.zeros(n, dtype=torch.long, device=device)
    timing = torch.zeros(n, dtype=torch.bool, device=device)  # 計測中フラグ
    speeds: list[float] = []
    dists: list[float] = []

    obs = hier_env.get_observations()
    for _ in range(n_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = hier_env.step(actions)

        target = raw_env._gk_target_y.clone()
        robot_y = (raw_env.scene["robot"].data.root_pos_w[:, 1] - raw_env.scene.env_origins[:, 1])
        err = (robot_y - target).abs()

        if prev_target is None:
            prev_target = target
            continue

        # 目標が変わった env: 計測開始 (エピソードリセットでも target は変わる)
        changed = (target - prev_target).abs() > 1e-6
        start_dist[changed] = err[changed]
        elapsed[changed] = 0
        timing[changed] = start_dist[changed] > 0.3  # 近すぎる目標は計測対象外
        prev_target = target

        elapsed[timing] += 1
        # 到達した env: 実効速度 = 開始時距離 / 所要時間
        reached = timing & (err < args_cli.reach_tol) & (elapsed > 0)
        for i in torch.nonzero(reached).flatten().tolist():
            t = elapsed[i].item() * dt
            d = start_dist[i].item()
            speeds.append(d / t)
            dists.append(d)
        timing[reached] = False
        # エピソード終了した env は計測破棄
        if dones is not None:
            done_mask = dones.to(device=device, dtype=torch.bool).flatten()
            timing[done_mask] = False

    hier_env.close()

    if not speeds:
        print("[WARN] 計測サンプルが得られませんでした。duration を延ばすか到達判定を確認してください。")
        return

    v = torch.tensor(speeds)
    v_lat_mean = v.mean().item()
    v_lat_p10 = v.quantile(0.1).item()
    gk = raw_env.cfg.goalkeeper
    d_min = float(gk.spawn_dist_range[0])
    half_w = float(gk.goal_half_width)
    cap_mean = 0.9 * v_lat_mean * d_min / half_w
    cap_p10 = 0.9 * v_lat_p10 * d_min / half_w

    print("\n===== goalkeeper stage-1 lateral speed =====")
    print(f"samples            : {len(speeds)} (mean move dist {sum(dists)/len(dists):.2f} m)")
    print(f"v_lat mean         : {v_lat_mean:.3f} m/s")
    print(f"v_lat p10          : {v_lat_p10:.3f} m/s")
    print(f"suggested ball_speed_cap (mean-based): {cap_mean:.2f} m/s")
    print(f"suggested ball_speed_cap (p10-based) : {cap_p10:.2f} m/s   <- 保守的推奨")
    print("override 例: {\"env\": {\"goalkeeper.ball_speed_cap\": %.2f}}" % cap_p10)


if __name__ == "__main__":
    main()
    simulation_app.close()
