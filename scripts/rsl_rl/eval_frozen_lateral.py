# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""凍結歩行ポリシー単体の移動性能 (前進・横・斜め) を計測する診断スクリプト。

上位ポリシーを **介さず**、frozen に一定の速度コマンド (vx, vy) を出し続けて、
実際に出る定常ドリフト速度を前後・左右成分に分けて測る。

目的: ゴールキーパーの「ゴールライン方向の移動」を速くする手段の評価。
frozen の横歩き (vy) は 0.6 m/s 程度で頭打ちだが、前進 (vx) は学習カリキュラムが
±1.8 m/s まで振られており遥かに速い。体を斜めに向ければ前進の速さを
ゴールライン方向に流用できるはずで、その効果を定量化する。

体を θ 傾けたときのゴールライン方向 (world-y) 速度は

    v_line(θ) = fwd_drift × sin θ + lat_drift × cos θ

で決まる。本スクリプトは各コマンド (vx, vy) について fwd_drift / lat_drift を実測し、
代表的な θ での v_line を表にして出力する。ロボットを実際に傾けて測るのではなく
成分から合成するので、旋回の過渡に汚されず、角度を変えるたびの再計測も不要。

計測方法:
    * ロボットを start_x へテレポートし yaw=0 に揃える (ゴールとの衝突回避)
    * 一定コマンドを出し続け、進行方向に travel_limit [m] 進んだらコマンドを反転
      (往復させて場外・ゴール衝突を防ぐ)
    * 過渡 (settle_s) を除いた区間で「正味変位 ÷ 時間」を 1 サンプルとする
      瞬間速度 (root_lin_vel) は歩行の左右揺れが支配的なので使わない
    * 転倒/タイムアウトでリセットが入った区間は破棄する

使い方 (コンテナ内・リポジトリ直下):
    /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/eval_frozen_lateral.py \\
        --frozen_checkpoint logs/rsl_rl/k1_flat/main_walk/0524_walk.pt \\
        --num_envs 32 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Measure the frozen walking policy's drift speeds.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments.")
parser.add_argument(
    "--task", type=str, default="Isaac-Goalkeeper-Stage1-K1-Play-v0",
    help="Task providing the scene/observations (Stage1 Play: ball parked, no perception noise).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument(
    "--frozen_checkpoint", type=str, required=True, help="Frozen low-level walking checkpoint."
)
parser.add_argument(
    "--cmd_list", type=str,
    default="0,0.9;0.6,0;1.0,0;1.4,0;1.8,0;0.9,0.9;1.4,0.9;1.4,0.5",
    help=(
        "Semicolon-separated 'vx,vy' command pairs to sweep. "
        "Defaults cover pure lateral / pure forward / diagonal combinations."
    ),
)
parser.add_argument("--hold_s", type=float, default=20.0, help="Measurement duration per command pair [s].")
parser.add_argument("--settle_s", type=float, default=1.5, help="Transient skipped after each direction flip [s].")
parser.add_argument("--travel_limit", type=float, default=1.5, help="Travel distance before the command flips [m].")
parser.add_argument("--start_x", type=float, default=2.5, help="Teleport x [m] (kept clear of the goal frame).")
parser.add_argument(
    "--min_run_s", type=float, default=1.0,
    help="Minimum steady duration for a run to count toward the average [s].",
)
parser.add_argument(
    "--angles", type=str, default="0,15,30,45,60,90",
    help="Body angles [deg] at which to report the implied along-the-goal-line speed.",
)
parser.add_argument(
    "--gait_hz", type=float, default=1.6,
    help=(
        "Gait frequency [Hz] used to window the foot-height measurement. "
        "One swing peak per foot occurs per gait cycle, so the per-cycle max is taken as the "
        "step's peak height (naive local-maxima detection over-counts stance jitter)."
    ),
)
parser.add_argument(
    "--high_action_clip", type=float, nargs=3, default=[1.8, 0.9, 1.0], metavar=("VX", "VY", "WZ"),
    help="Per-axis clip applied by the wrapper. Must be >= the largest value in --cmd_list.",
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
import math
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401

from goalkeeper_helpers import HierarchicalVecEnvWrapper, _build_frozen_policy


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed

    # 計測を邪魔する終了条件を無効化する。
    #   out_of_bounds : 広く往復させるので守備範囲判定は不要
    # 転倒 (base_contact) は残す — 転ぶこと自体が「その指令では走れない」証拠になる。
    if hasattr(env_cfg.terminations, "out_of_bounds"):
        env_cfg.terminations.out_of_bounds = None
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args_cli.hold_s * 2.0)

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

    raw_env = inner_env.unwrapped
    robot = raw_env.scene["robot"]
    # 足の高さ計測用の body index (スイング中の最大高さ = 「どれだけ足を上げているか」)
    left_foot_idx = robot.find_bodies("left_foot_link")[0][0]
    right_foot_idx = robot.find_bodies("right_foot_link")[0][0]
    dt = raw_env.step_dt
    n = raw_env.num_envs
    device = raw_env.device
    settle_steps = max(1, int(args_cli.settle_s / dt))
    n_steps = int(args_cli.hold_s / dt)
    min_run_steps = max(1, int(args_cli.min_run_s / dt))

    def teleport_to_start() -> None:
        """全 env のロボットを (start_x, 0, 既定高さ)・yaw=0 に置き直し、速度を 0 にする。

        ゴール枠 (x=0 付近) との衝突と、往復中心の左右への漂流を防ぐため、
        コマンドを切り替えるたびに基準姿勢へ戻す。

        ★ ``torch.inference_mode()`` で囲むこと。シーンの状態バッファは推論モード下で
        生成された "inference tensor" なので、その外から in-place 書き換えすると
        PyTorch に拒否される (RuntimeError: Inplace update to inference tensor ...)。
        """
        with torch.inference_mode():
            pose = torch.zeros(n, 7, device=device)
            pose[:, 0] = raw_env.scene.env_origins[:, 0] + float(args_cli.start_x)
            pose[:, 1] = raw_env.scene.env_origins[:, 1]
            pose[:, 2] = robot.data.default_root_state[:, 2]
            pose[:, 3] = 1.0  # 単位クォータニオン (w, x, y, z) = yaw 0
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=device))

    cmd_pairs = []
    for chunk in args_cli.cmd_list.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vx_s, vy_s = chunk.split(",")
        cmd_pairs.append((float(vx_s), float(vy_s)))

    clip = [float(c) for c in args_cli.high_action_clip]
    results = []

    hier_env.get_observations()
    for vx_cmd, vy_cmd in cmd_pairs:
        eff_vx = max(-clip[0], min(clip[0], vx_cmd))
        eff_vy = max(-clip[1], min(clip[1], vy_cmd))
        norm = math.hypot(eff_vx, eff_vy)
        if norm < 1e-6:
            continue
        axis_x, axis_y = eff_vx / norm, eff_vy / norm  # yaw=0 前提の進行方向 (world)

        teleport_to_start()
        anchor = (robot.data.root_pos_w[:, :2] - raw_env.scene.env_origins[:, :2]).clone()
        direction = torch.ones(n, device=device)
        since_flip = torch.zeros(n, dtype=torch.long, device=device)
        pos_settle = torch.zeros(n, 2, device=device)
        head_settle = torch.zeros(n, device=device)
        run_valid = torch.zeros(n, dtype=torch.bool, device=device)
        fwd_sum = torch.zeros((), device=device)
        lat_sum = torch.zeros((), device=device)
        cnt = torch.zeros((), device=device)
        head_drift_sum = torch.zeros((), device=device)
        resets = torch.zeros((), device=device)
        # 足の高さ: **歩行 1 周期ぶんの窓ごとに最大値を 1 つ**取り、それを 1 歩の
        # ピーク高さとする。単純な極大値検出 (前ステップより下がった点) だと接地中の
        # 微細な揺れまで拾って実歩数の 2〜3 倍を数えてしまい、平均が偽ピークに
        # 薄められて過小評価になる。1 周期には各足 1 回のスイングしか無いので、
        # 窓内最大値なら構造的に 1 歩 1 サンプルになる。
        cycle_steps = max(1, int(round(1.0 / (max(float(args_cli.gait_hz), 1e-3) * dt))))
        win_max = torch.zeros(n, 2, device=device)
        win_ctr = 0
        peak_sum = torch.zeros((), device=device)
        peak_cnt = torch.zeros((), device=device)
        peak_max = torch.zeros((), device=device)
        # 接地時の足リンク高さ (足裏オフセット)。持ち上げ量 = ピーク − この値。
        foot_floor = torch.full((), float("inf"), device=device)

        for _ in range(n_steps):
            action = torch.zeros(n, 3, device=device)
            action[:, 0] = eff_vx * direction
            action[:, 1] = eff_vy * direction
            with torch.inference_mode():
                _, _, dones, _ = hier_env.step(action)

            pos = robot.data.root_pos_w[:, :2] - raw_env.scene.env_origins[:, :2]
            heading = robot.data.heading_w

            # --- 足のピーク高さを集計 (地面 z=0 基準) ---
            foot_h = torch.stack(
                [
                    robot.data.body_pos_w[:, left_foot_idx, 2],
                    robot.data.body_pos_w[:, right_foot_idx, 2],
                ],
                dim=1,
            )
            # 歩行 1 周期の窓内で最大値を更新し、窓が閉じたら 1 歩ぶんのピークとして確定。
            # foot_h は (env, 足2本) なので、env ごとの過渡判定は (env, 1) に広げて合成する。
            steady_mask = (since_flip > settle_steps).unsqueeze(1)
            win_max = torch.maximum(win_max, torch.where(steady_mask, foot_h, torch.zeros_like(foot_h)))
            win_ctr += 1
            if win_ctr >= cycle_steps:
                valid = win_max > 0.0  # 過渡だけで窓が埋まった env は除外
                if bool(valid.any()):
                    vals = win_max[valid]
                    peak_sum += vals.sum()
                    peak_cnt += vals.numel()
                    peak_max = torch.maximum(peak_max, vals.max())
                win_max = torch.zeros_like(win_max)
                win_ctr = 0
            if bool((since_flip > settle_steps).any()):
                foot_floor = torch.minimum(foot_floor, foot_h[since_flip > settle_steps].min())

            # 過渡を抜けた瞬間の位置・向きを基準として記録
            newly_settled = since_flip == settle_steps
            if bool(newly_settled.any()):
                pos_settle[newly_settled] = pos[newly_settled]
                head_settle[newly_settled] = heading[newly_settled]
                run_valid[newly_settled] = True

            done_mask = (
                dones.to(device=device, dtype=torch.bool).flatten()
                if dones is not None
                else torch.zeros(n, dtype=torch.bool, device=device)
            )
            resets += done_mask.sum()

            # 進行方向に travel_limit まで進んだら 1 サンプル確定してコマンド反転。
            # リセット (転倒/タイムアウト) が起きた run は位置が飛ぶので破棄。
            rel = pos - anchor
            s = (rel[:, 0] * axis_x + rel[:, 1] * axis_y) * direction
            at_edge = s > args_cli.travel_limit
            measured = at_edge & run_valid & ~done_mask & (since_flip - settle_steps >= min_run_steps)
            if bool(measured.any()):
                elapsed = (since_flip[measured] - settle_steps).float() * dt
                d = pos[measured] - pos_settle[measured]
                h = head_settle[measured]
                # 基準時の体の向きに合わせて前後 (fwd) / 左右 (lat) 成分へ分解
                fwd = (d[:, 0] * torch.cos(h) + d[:, 1] * torch.sin(h)) * direction[measured]
                lat = (-d[:, 0] * torch.sin(h) + d[:, 1] * torch.cos(h)) * direction[measured]
                fwd_sum += (fwd / elapsed.clamp(min=1e-6)).sum()
                lat_sum += (lat / elapsed.clamp(min=1e-6)).sum()
                head_drift_sum += (heading[measured] - h).abs().sum()
                cnt += fwd.numel()

            flip = at_edge | done_mask
            direction[flip] *= -1.0
            run_valid[flip] = False
            since_flip += 1
            since_flip[flip] = 0

        denom = cnt.clamp(min=1)
        results.append({
            "cmd": (vx_cmd, vy_cmd),
            "sent": (eff_vx, eff_vy),
            "fwd": float((fwd_sum / denom).item()),
            "lat": float((lat_sum / denom).item()),
            "head_drift": float((head_drift_sum / denom).item()) * 180.0 / math.pi,
            "n": int(cnt.item()),
            "resets": int(resets.item()),
            "foot_peak": float((peak_sum / peak_cnt.clamp(min=1)).item()),
            "foot_peak_max": float(peak_max.item()),
            "foot_floor": float(foot_floor.item()),
            "foot_n": int(peak_cnt.item()),
        })

    hier_env.close()

    angles = [float(a) for a in args_cli.angles.split(",")]

    print("\n===== frozen policy drift speeds (変位ベース) =====")
    print(f"checkpoint: {args_cli.frozen_checkpoint}")
    print(f"clip (vx, vy, wz): {tuple(clip)}")
    print("\n--- 実測ドリフト速度 (体基準) ---")
    print(f"{'cmd(vx,vy)':>13} {'fwd':>7} {'lat':>7} {'|v|':>7} {'yaw drift':>10} {'n':>5} {'reset':>6}")
    for r in results:
        speed = math.hypot(r["fwd"], r["lat"])
        print(
            f"{r['cmd'][0]:6.2f},{r['cmd'][1]:5.2f} {r['fwd']:7.3f} {r['lat']:7.3f} {speed:7.3f} "
            f"{r['head_drift']:9.1f}° {r['n']:5d} {r['resets']:6d}"
        )

    print("\n--- 足の持ち上げ高さ (1 歩ごとのピーク) ---")
    print(f"{'cmd(vx,vy)':>13} {'持ち上げ平均':>12} {'最大':>9} {'接地時':>9} {'歩数':>7}")
    for r in results:
        lift = r["foot_peak"] - r["foot_floor"]
        lift_max = r["foot_peak_max"] - r["foot_floor"]
        print(
            f"{r['cmd'][0]:6.2f},{r['cmd'][1]:5.2f} {lift:11.3f}m {lift_max:8.3f}m"
            f" {r['foot_floor']:8.3f}m {r['foot_n']:6d}"
        )
    print(" 持ち上げ = スイング時のピーク高さ − 接地時の足リンク高さ (足裏オフセットを除いた実効値)。")
    print(" 目安: foot_clearance_ji_pen の target_clearance は足リンク高さで 0.10m 指定。")

    print("\n--- 体を θ 傾けたときのゴールライン方向 (world-y) 速度 [m/s] ---")
    print("    v_line(θ) = fwd × sin θ + lat × cos θ")
    header = "".join(f"{a:>8.0f}°" for a in angles)
    print(f"{'cmd(vx,vy)':>13}{header}")
    best = (None, 0.0, 0.0)
    for r in results:
        cells = ""
        for a in angles:
            rad = math.radians(a)
            v = r["fwd"] * math.sin(rad) + r["lat"] * math.cos(rad)
            cells += f"{v:9.3f}"
            if v > best[1]:
                best = (r["cmd"], v, a)
        print(f"{r['cmd'][0]:6.2f},{r['cmd'][1]:5.2f}{cells}")

    baseline = next((r["lat"] for r in results if abs(r["cmd"][0]) < 1e-6), None)
    if best[0] is not None:
        print(f"\n[最良] cmd=({best[0][0]:.2f}, {best[0][1]:.2f}) を θ={best[2]:.0f}° で使うと "
              f"{best[1]:.3f} m/s")
        if baseline:
            print(f"       真正面の横歩き {baseline:.3f} m/s に対して {best[1]/baseline:.2f} 倍")
    print("\n[注意] θ まで体を回す旋回コスト (wz=1.0 rad/s なら 45°で約0.79秒) は上表に含まれない。")
    print("       ボールが遠い/遅いうちに向きを作れる場合のみ、この速度が活きる。")
    print("       yaw drift が大きいコマンドは wz=0 でも向きが流れており、実運用では要注意。")


if __name__ == "__main__":
    main()
    simulation_app.close()
