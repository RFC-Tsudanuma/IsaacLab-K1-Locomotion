# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""**歩行から停止までの遷移**の振動を測る。

なぜ必要か (2026-09-02):
    実機フィードバックは「動いているときは振動しない / **止まるときに振動する**」。
    ところが既存の :mod:`diag_walk_jitter` は **指令をゼロに固定** して測るので
    「ずっと静止している状態」しか見ておらず、**遷移そのものを一度も測っていない**。
    対策 (減速区間ゲート / stop_prob / 角加速度罰) を 20000 イテレーション回す前に、
    まず **現状の数値と、シムで症状が再現するか** を押さえる。

    ☠ 後退不安定で 5 回繰り返した失敗を避けるため。シムで再現しないものを学習で
      潰そうとすると、当たったかどうかすら分からない。

測る量:
    胴体角速度の **2 階差分** ``||w_t - 2 w_{t-1} + w_{t-2}||`` [rad/s]。
    ``goalkeeper/mdp/rewards.py::body_jitter`` と同一の量で、実機と対応づけ済み:

        0.021  -12.0版  実機で振動しない
        0.082  08-41-39
        0.196  07-28    実機で常に振動する

区間の分け方 (1 サイクル = 走行 walk_s → 停止指令 stop_s):
    走行中     : 走行区間の後半 (立ち上がりを除く)
    ★停止遷移 : 指令ゼロ化から ``win_s`` 秒間  ← **実機で振動している区間**
    静止中     : ゼロ化から ``settle_s`` 秒経過後

使い方 (コンテナ内・リポジトリ直下):
    /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/diag_stop_transition.py \
        --task Isaac-GKLateralDH-K1-Play-v0 \
        --checkpoint logs/rsl_rl/k1_gk_lateral_dh/2026-08-23_18-05-55/model_17300.pt \
        --num_envs 64 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Stop-transition vibration diagnostic.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-GKLateral-K1-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--override_json", type=str, default=None)
parser.add_argument("--speed", type=float, default=1.2, help="走行時の指令速度 [m/s]。")
parser.add_argument("--walk_s", type=float, default=4.0, help="走行させる時間 [s]。")
parser.add_argument("--stop_s", type=float, default=3.0, help="停止指令を出す時間 [s]。")
parser.add_argument("--win_s", type=float, default=0.8,
                    help="★ 停止遷移とみなす窓 [s]。学習側の JITTER_STOP_WINDOW_S と揃える。")
parser.add_argument("--settle_s", type=float, default=1.5,
                    help="ここを過ぎたら「静止中」として集計する [s]。")
parser.add_argument("--cycles", type=int, default=6, help="走行→停止を繰り返す回数。")
parser.add_argument("--dirs", type=str, default="前進,後退,左横,右横",
                    help="測る方向 (カンマ区切り)。")
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

_DIR_VEC = {"前進": (1.0, 0.0), "後退": (-1.0, 0.0), "左横": (0.0, 1.0), "右横": (0.0, -1.0)}
_BIN_S = 0.2   # 経過時間ビンの幅 [s]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed

    if args_cli.override_json is not None:
        from config_overrides import apply_overrides_from_file

        apply_overrides_from_file(args_cli.override_json, env_cfg=env_cfg, agent_cfg=agent_cfg)

    dirs = [d for d in args_cli.dirs.split(",") if d in _DIR_VEC]
    cycle_s = args_cli.walk_s + args_cli.stop_s
    total_s = cycle_s * args_cli.cycles * len(dirs)

    vc = env_cfg.commands.base_velocity
    vc.heading_command = False
    vc.rel_standing_envs = 0.0
    # ☠ 指令はこちらで毎ステップ上書きするので、環境側の再サンプリングは止める。
    vc.resampling_time_range = (1.0e9, 1.0e9)
    if getattr(env_cfg.terminations, "out_of_bounds", None) is not None:
        env_cfg.terminations.out_of_bounds = None
    # ☠ 全サイクルを 1 エピソードで走り切らないと、リセット (原点へテレポート) が
    #   遷移の計測に混ざる。eval_dir_margin.py で一度これに嵌まっている。
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), total_s + 4.0)
    env_cfg.scene.env_spacing = max(float(getattr(env_cfg.scene, "env_spacing", 2.5)), 8.0)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = (
        retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint
        else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    inner_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(inner_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=inner_env.unwrapped.device)

    raw_env = inner_env.unwrapped
    robot = raw_env.scene["robot"]
    cmd_term = raw_env.command_manager.get_term("base_velocity")
    dt = raw_env.step_dt
    device = raw_env.device

    # 集計器: 方向 × 区間 → (合計, 件数)。速度も同時に見る (本当に止まったかの確認)。
    keys = ["走行中", "★停止遷移", "静止中"]
    acc = {d: {k: [0.0, 0] for k in keys} for d in dirs}
    vel = {d: {k: [0.0, 0] for k in keys} for d in dirs}
    peak = {d: 0.0 for d in dirs}   # 停止遷移中の最大値 (平均だと一瞬の跳ねが埋もれる)
    falls = {d: 0 for d in dirs}

    # ★★ 2026-09-02: **経過時間ビン**を追加した理由。
    #   初回計測 (凹凸地形) で「★停止遷移」が走行中とほぼ同値になった:
    #       前進 0.527 vs 0.508 / 後退 0.630 vs 0.620 / 左横 0.389 vs 0.518
    #   ☠ 窓の中でロボットはまだ 0.44〜0.59 m/s で走っており、**移動しているという
    #     事実だけで 0.5 級が出る** (body_jitter の docstring: 移動中は 0.57 級)。
    #   つまり窓の平均は「まだ歩いている」に支配され、**振動を分離できていない**。
    #   知りたいのは「止まった後に収まるか」なので、停止指令からの経過時間で
    #   ビン分けし、**減衰の様子そのもの**を見る。
    nb = int(round(args_cli.stop_s / _BIN_S))
    bins = {d: [[0.0, 0, 0.0] for _ in range(nb)] for d in dirs}   # [Σd2w, 件数, Σ速度]

    hist: list[torch.Tensor] = []
    obs, _ = env.reset()
    steps = int(round(total_s / dt))

    with torch.inference_mode():
        for i in range(steps):
            t = i * dt
            leg = int(t // (cycle_s * args_cli.cycles))
            leg = min(leg, len(dirs) - 1)
            d = dirs[leg]
            tc = t % cycle_s                      # サイクル内の時刻
            walking = tc < args_cli.walk_s
            ts = tc - args_cli.walk_s             # 指令ゼロ化からの経過

            vx, vy = _DIR_VEC[d]
            cmd_term.vel_command_b[:, 0] = vx * args_cli.speed if walking else 0.0
            cmd_term.vel_command_b[:, 1] = vy * args_cli.speed if walking else 0.0
            cmd_term.vel_command_b[:, 2] = 0.0

            actions = policy(obs)
            obs, _, dones, _ = inner_env.step(actions)
            falls[d] += int(dones.sum().item())

            w = robot.data.root_ang_vel_w
            hist.append(w.clone())
            if len(hist) > 3:
                hist.pop(0)
            if len(hist) < 3:
                continue
            d2w = torch.norm(hist[2] - 2.0 * hist[1] + hist[0], dim=1)
            lin = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)

            # 区間の判定。走行区間は立ち上がり 1.5s を除いて後半だけ見る。
            if not walking:
                bi = int(ts / _BIN_S)
                if 0 <= bi < nb:
                    b = bins[d][bi]
                    b[0] += float(d2w.sum().item()); b[1] += d2w.numel()
                    b[2] += float(lin.sum().item())

            if walking:
                if tc < 1.5:
                    continue
                k = "走行中"
            elif ts <= args_cli.win_s:
                k = "★停止遷移"
                peak[d] = max(peak[d], float(d2w.max().item()))
            elif ts >= args_cli.settle_s:
                k = "静止中"
            else:
                continue   # 窓と静止の間の緩衝はどちらにも入れない

            acc[d][k][0] += float(d2w.sum().item()); acc[d][k][1] += d2w.numel()
            vel[d][k][0] += float(lin.sum().item()); vel[d][k][1] += lin.numel()

    print()
    print("=" * 74)
    print(f"checkpoint: {resume_path}")
    print(f"task: {args_cli.task}   速度 {args_cli.speed} m/s   "
          f"{args_cli.cycles} サイクル × {args_cli.num_envs} env")
    print("=" * 74)
    print("胴体角速度の2階差分 [rad/s]   実機対応: 0.021 振動しない / "
          "0.082 停止時に振動 / 0.196 常に振動")
    print("-" * 74)
    print(f"{'方向':<6} {'走行中':>10} {'★停止遷移':>12} {'静止中':>10} "
          f"{'遷移の最大':>10} {'転倒':>6}")
    for d in dirs:
        def m(k):
            s, c = acc[d][k]
            return s / c if c else float("nan")
        print(f"{d:<6} {m('走行中'):>10.3f} {m('★停止遷移'):>12.3f} "
              f"{m('静止中'):>10.3f} {peak[d]:>10.3f} {falls[d]:>6d}")
    print("-" * 74)
    print("参考: 各区間の平均速度 [m/s] (★停止遷移で本当に減速しているかの確認)")
    print(f"{'方向':<6} {'走行中':>10} {'★停止遷移':>12} {'静止中':>10}")
    for d in dirs:
        def v(k):
            s, c = vel[d][k]
            return s / c if c else float("nan")
        print(f"{d:<6} {v('走行中'):>10.3f} {v('★停止遷移'):>12.3f} {v('静止中'):>10.3f}")
    print("-" * 74)
    print("★ 停止指令からの経過時間ごとの 2階差分 (上) と 速度 [m/s] (下)")
    print("   ← 「止まった後に収まるか」を直接見る。実機の目標は 0.021、"
          "0.082 で停止時に振動する")
    hdr = "".join(f"{(i + 1) * _BIN_S:>8.1f}s" for i in range(nb))
    print(f"{'方向':<6}{hdr}")
    for d in dirs:
        row = "".join(
            f"{(b[0] / b[1] if b[1] else float('nan')):>9.3f}" for b in bins[d]
        )
        vrow = "".join(
            f"{(b[2] / b[1] if b[1] else float('nan')):>9.3f}" for b in bins[d]
        )
        print(f"{d:<6}{row}")
        print(f"{'':<6}{vrow}")
    print("=" * 74)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
