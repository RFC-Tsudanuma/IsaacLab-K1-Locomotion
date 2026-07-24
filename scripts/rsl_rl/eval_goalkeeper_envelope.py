# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパーの「どこまで止められるか」(セーブ可能範囲) を実測する評価スクリプト。

学習済みポリシーを通常の分布で回し、**1 エピソードごとに条件と結果を記録**して
後から集計する。条件を強制的に固定するのではなく実際の分布で回すので、
「ポリシーが実運用で直面する状況での成績」がそのまま出る。

記録する条件 (エピソード開始時点):
    * ball_dist  : ボールのスポーン距離 [m] (ゴール中央基準)
    * ball_angle : スポーン方位 |θ| [deg] (+x 正面を 0°)
    * ball_speed : ボール初速 [m/s]
    * offset     : **ロボットが横に動かねばならない距離** [m]
                   = |守備面でのボール到達予測 y − ロボットの y|
                   セーブ可否を最も強く決めるのはこの量、という仮説の検証用。

結果の分類:
    save   : save_success (ボールに触れて無害化 / 枠外へ弾いた)
    concede: goal_conceded (失点)
    fall   : base_contact (転倒)
    out    : out_of_bounds (守備範囲逸脱)
    other  : time_out など (ボールが自然停止した場合などを含む)

使い方 (コンテナ内・リポジトリ直下):
    /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/eval_goalkeeper_envelope.py \\
        --frozen_checkpoint logs/rsl_rl/k1_flat/main_walk/0524_walk.pt \\
        --checkpoint logs/rsl_rl/k1_goalkeeper_stage3/<run>/model_XXXXX.pt \\
        --episodes 2000 --num_envs 64 --headless

実運用に近い条件 (知覚ノイズ・押し外乱あり) で測りたいときは
``--task Isaac-Goalkeeper-K1-v0`` を指定する。
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Measure the goalkeeper's save envelope.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument(
    "--task", type=str, default="Isaac-Goalkeeper-K1-Play-v0",
    help="Task to evaluate on. Play = clean perception; use Isaac-Goalkeeper-K1-v0 for training conditions.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to collect.")
parser.add_argument(
    "--frozen_checkpoint", type=str, required=True, help="Frozen low-level walking checkpoint."
)
parser.add_argument(
    "--high_action_clip", type=float, nargs=3, default=[0.6, 0.8, 1.0], metavar=("VX", "VY", "WZ"),
    help="Per-axis high-level action clip. MUST match the value used during training.",
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
from isaaclab_k1_locomotion.tasks.manager_based.goalkeeper.mdp.observations import (
    ball_pos_goal,
    compute_target_y,
    robot_pos_goal,
)

from goalkeeper_helpers import HierarchicalVecEnvWrapper, _build_frozen_policy

# 集計のビン境界 (上限値のリスト。最後は inf 扱い)
SPEED_BINS = [1.0, 1.5, 2.0, 2.5, 3.0]
DIST_BINS = [1.5, 2.5, 3.5, 4.5]
ANGLE_BINS = [15.0, 30.0, 45.0, 60.0]
OFFSET_BINS = [0.25, 0.5, 0.75, 1.0, 1.5]


def _bin_label(bins: list[float], value: float, unit: str) -> str:
    lo = 0.0
    for hi in bins:
        if value < hi:
            return f"{lo:.2f}-{hi:.2f}{unit}"
        lo = hi
    return f"{lo:.2f}+{unit}"


def _print_table(title: str, bins: list[float], unit: str, records: list, key_idx: int) -> None:
    """指定した条件軸でビン集計してセーブ率を表示する。"""
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for r in records:
        label = _bin_label(bins, r[key_idx], unit)
        if label not in groups:
            groups[label] = []
            order.append(label)
        groups[label].append(r[4])

    def sort_key(lbl: str) -> float:
        return float(lbl.split("-")[0].replace("+", "").replace(unit, ""))

    print(f"\n--- {title} ---")
    print(f"{'range':>14} {'n':>6} {'save':>7} {'concede':>8} {'fall':>6} {'other':>6}")
    for label in sorted(order, key=sort_key):
        outs = groups[label]
        n = len(outs)
        save = outs.count("save") / n
        conc = outs.count("concede") / n
        fall = (outs.count("fall") + outs.count("out")) / n
        other = outs.count("other") / n
        print(f"{label:>14} {n:6d} {save:6.1%} {conc:7.1%} {fall:5.1%} {other:5.1%}")


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
    n = raw_env.num_envs
    device = raw_env.device
    tm = raw_env.termination_manager
    max_y = float(raw_env.cfg.goalkeeper.goal_half_width)

    def capture_conditions() -> torch.Tensor:
        """現時点 (エピソード開始直後) の条件を (n, 4) で返す: 距離/角度/速度/必要横移動量。"""
        bpos = ball_pos_goal(raw_env)
        bvel = raw_env.scene["soccer_ball"].data.root_com_vel_w[:, :3]
        dist = torch.norm(bpos[:, :2], dim=1)
        angle = torch.atan2(bpos[:, 1].abs(), bpos[:, 0].clamp(min=1e-3)) * 180.0 / torch.pi
        speed = torch.norm(bvel[:, :2], dim=1)
        offset = (compute_target_y(raw_env, max_y=max_y) - robot_pos_goal(raw_env)[:, 1]).abs()
        return torch.stack([dist, angle, speed, offset], dim=1)

    def term_fired(name: str) -> torch.Tensor:
        try:
            return tm.get_term(name)
        except (KeyError, AttributeError, ValueError):
            return torch.zeros(n, dtype=torch.bool, device=device)

    obs = hier_env.get_observations()
    # 1 step 進めてボールの初期状態を確定させてから条件を取る
    with torch.inference_mode():
        obs, _, _, _ = hier_env.step(policy(obs))
    pending = capture_conditions()

    records: list[tuple[float, float, float, float, str]] = []
    while len(records) < args_cli.episodes:
        with torch.inference_mode():
            obs, _, dones, _ = hier_env.step(policy(obs))
        if dones is None:
            continue
        done_mask = dones.to(device=device, dtype=torch.bool).flatten()
        if not bool(done_mask.any()):
            continue

        # 終了した env の結果を、そのエピソード開始時に記録した条件へ紐付ける
        conceded = term_fired("goal_conceded")
        saved = term_fired("save_success")
        fell = term_fired("base_contact")
        oob = term_fired("out_of_bounds")
        idx_list = torch.nonzero(done_mask).flatten().tolist()
        cond_cpu = pending.cpu()
        for i in idx_list:
            if bool(conceded[i]):
                outcome = "concede"
            elif bool(saved[i]):
                outcome = "save"
            elif bool(fell[i]):
                outcome = "fall"
            elif bool(oob[i]):
                outcome = "out"
            else:
                outcome = "other"
            d, a, s, o = cond_cpu[i].tolist()
            records.append((d, a, s, o, outcome))

        # リセット済みなので、終了した env の新しい条件を取り直す
        new_cond = capture_conditions()
        pending[done_mask] = new_cond[done_mask]

    hier_env.close()

    total = len(records)
    n_save = sum(1 for r in records if r[4] == "save")
    n_conc = sum(1 for r in records if r[4] == "concede")
    n_fall = sum(1 for r in records if r[4] in ("fall", "out"))
    n_other = total - n_save - n_conc - n_fall

    print("\n===== goalkeeper save envelope =====")
    print(f"task      : {args_cli.task}")
    print(f"checkpoint: {resume_path}")
    print(f"clip      : {tuple(args_cli.high_action_clip)}")
    print(f"episodes  : {total}")
    print(f"overall   : save {n_save/total:.1%} / concede {n_conc/total:.1%} / "
          f"fall+out {n_fall/total:.1%} / other {n_other/total:.1%}")

    _print_table("必要横移動量 offset [m] (セーブ可否を決める主因の検証)", OFFSET_BINS, "m", records, 3)
    _print_table("ボール初速 [m/s]", SPEED_BINS, "", records, 2)
    _print_table("スポーン距離 [m]", DIST_BINS, "m", records, 0)
    _print_table("スポーン方位 |θ| [deg]", ANGLE_BINS, "d", records, 1)

    print("\n[読み方]")
    print(" offset の表でセーブ率が急落する境界が「横に届く限界」。")
    print(" 速度/距離の表でセーブ率が平坦なら、勝敗を決めているのは速度や距離ではなく")
    print(" offset (横方向のズレ) だと確認できる (min_time_to_line で到達時間が揃うため)。")
    print(" 角度の表だけ悪い場合は、斜め球の到達予測が効いていない可能性がある。")


if __name__ == "__main__":
    main()
    simulation_app.close()
