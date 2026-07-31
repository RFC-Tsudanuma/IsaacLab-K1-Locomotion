# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""gk_direct (直接制御版) 単一ポリシーの移動性能 (前進・横・斜め) を計測する診断スクリプト。

eval_frozen_lateral.py の単一ポリシー版。frozen/階層ラッパーを介さず、gk_direct の
単一ポリシー (obs 59 → 12 関節) に一定の速度コマンド (vx, vy) を出し続けて、
実際に出る定常ドリフト速度を前後・左右成分に分けて測る。

コマンドは env の base_velocity コマンド項の ``vel_command_b`` に毎ステップ直接
書き込んで固定する (heading 制御・再サンプル・standing env はすべて無効化)。
観測の velocity_commands スロットはこの固定値を読むので、ポリシーには「一定速度で
横に動け」という指令が入り続ける。

計測方法は eval_frozen_lateral.py と同一 (変位ベース・往復・過渡除去・転倒破棄)。

使い方 (コンテナ内・リポジトリ直下):
    /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/eval_gk_direct_lateral.py \\
        --checkpoint logs/rsl_rl/k1_gk_direct_stage1/<run>/model_XXXX.pt \\
        --cmd_list "0,0.5;0,0.9;0,1.2;0,1.5" \\
        --num_envs 32 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Measure the gk_direct single policy's drift speeds.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments.")
parser.add_argument(
    "--task", type=str, default="Isaac-GoalkeeperDirect-Stage1-K1-Play-v0",
    help="Task providing the scene/observations (gk_direct Stage1 Play).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument(
    "--cmd_list", type=str,
    default="0,0.5;0,0.9;0,1.2;0,1.5;1.0,0;1.0,0.9;1.0,1.5",
    help=(
        "Semicolon-separated 'vx,vy' command pairs to sweep. "
        "Defaults cover pure lateral (up to vy=1.5) / pure forward / diagonal combinations."
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
    help="Gait frequency [Hz] used to window the foot-height measurement.",
)
parser.add_argument(
    "--cmd_clip", type=float, nargs=2, default=[1.0, 1.5], metavar=("VX", "VY"),
    help="Per-axis command clip (gk_direct Stage1 range: vx +/-1.0, vy +/-1.5). Must be >= cmd_list values.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed

    # --- コマンドを固定注入するための下準備 ---
    # heading 制御・再サンプル・standing env をすべて切って、vel_command_b への
    # 毎ステップ書き込みが上書きされないようにする。
    vc = env_cfg.commands.base_velocity
    vc.heading_command = False
    vc.rel_standing_envs = 0.0
    vc.resampling_time_range = (1.0e9, 1.0e9)

    # 計測を邪魔する終了条件を無効化 (転倒 base_contact は残す)。
    if hasattr(env_cfg.terminations, "out_of_bounds"):
        env_cfg.terminations.out_of_bounds = None
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args_cli.hold_s * 2.0)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    inner_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(inner_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=inner_env.unwrapped.device)

    raw_env = inner_env.unwrapped
    robot = raw_env.scene["robot"]
    cmd_term = raw_env.command_manager.get_term("base_velocity")
    left_foot_idx = robot.find_bodies("left_foot_link")[0][0]
    right_foot_idx = robot.find_bodies("right_foot_link")[0][0]
    dt = raw_env.step_dt
    n = raw_env.num_envs
    device = raw_env.device
    settle_steps = max(1, int(args_cli.settle_s / dt))
    n_steps = int(args_cli.hold_s / dt)
    min_run_steps = max(1, int(args_cli.min_run_s / dt))

    def set_command(vx: float, vy: float, direction: torch.Tensor) -> None:
        """base_velocity コマンドを (vx*dir, vy*dir, 0) に固定する。inference_mode 内で呼ぶ。"""
        cmd_term.vel_command_b[:, 0] = vx * direction
        cmd_term.vel_command_b[:, 1] = vy * direction
        cmd_term.vel_command_b[:, 2] = 0.0

    def teleport_to_start() -> None:
        """全 env を (start_x, 0, 既定高さ)・yaw=0 に置き直し速度 0 にする。inference_mode 必須。"""
        with torch.inference_mode():
            pose = torch.zeros(n, 7, device=device)
            pose[:, 0] = raw_env.scene.env_origins[:, 0] + float(args_cli.start_x)
            pose[:, 1] = raw_env.scene.env_origins[:, 1]
            pose[:, 2] = robot.data.default_root_state[:, 2]
            pose[:, 3] = 1.0  # 単位クォータニオン = yaw 0
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=device))

    cmd_pairs = []
    for chunk in args_cli.cmd_list.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vx_s, vy_s = chunk.split(",")
        cmd_pairs.append((float(vx_s), float(vy_s)))

    clip = [float(args_cli.cmd_clip[0]), float(args_cli.cmd_clip[1])]
    results = []

    obs = inner_env.get_observations()
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
        cycle_steps = max(1, int(round(1.0 / (max(float(args_cli.gait_hz), 1e-3) * dt))))
        win_max = torch.zeros(n, 2, device=device)
        win_ctr = 0
        # 足の高さ統計は **左右別** に取る (列 0 = 左足, 列 1 = 右足)。
        # 片足だけ跳ぶ/上がらないといった左右非対称な歩容は、両足の平均だと
        # 打ち消されて見えなくなるため。
        peak_sum = torch.zeros(2, device=device)
        peak_cnt = torch.zeros(2, device=device)
        peak_max = torch.zeros(2, device=device)
        foot_floor = torch.full((2,), float("inf"), device=device)

        for _ in range(n_steps):
            with torch.inference_mode():
                set_command(eff_vx, eff_vy, direction)
                action = policy(obs)
                obs, _, dones, _ = inner_env.step(action)
                # step 内の command 再計算後にも固定値を保証する
                set_command(eff_vx, eff_vy, direction)

            pos = robot.data.root_pos_w[:, :2] - raw_env.scene.env_origins[:, :2]
            heading = robot.data.heading_w

            foot_h = torch.stack(
                [
                    robot.data.body_pos_w[:, left_foot_idx, 2],
                    robot.data.body_pos_w[:, right_foot_idx, 2],
                ],
                dim=1,
            )
            steady_mask = (since_flip > settle_steps).unsqueeze(1)
            win_max = torch.maximum(win_max, torch.where(steady_mask, foot_h, torch.zeros_like(foot_h)))
            win_ctr += 1
            if win_ctr >= cycle_steps:
                valid = win_max > 0.0                      # (env, 2)
                if bool(valid.any()):
                    # 左右の列ごとに集計する (dim=0 は env 方向)
                    peak_sum += (win_max * valid).sum(dim=0)
                    peak_cnt += valid.sum(dim=0)
                    peak_max = torch.maximum(peak_max, win_max.max(dim=0).values)
                win_max = torch.zeros_like(win_max)
                win_ctr = 0
            steady = since_flip > settle_steps
            if bool(steady.any()):
                foot_floor = torch.minimum(foot_floor, foot_h[steady].min(dim=0).values)

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

            rel = pos - anchor
            s = (rel[:, 0] * axis_x + rel[:, 1] * axis_y) * direction
            at_edge = s > args_cli.travel_limit
            measured = at_edge & run_valid & ~done_mask & (since_flip - settle_steps >= min_run_steps)
            if bool(measured.any()):
                elapsed = (since_flip[measured] - settle_steps).float() * dt
                d = pos[measured] - pos_settle[measured]
                h = head_settle[measured]
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
            # 左右別 ([0]=左, [1]=右) と、両足込みの平均を両方持たせる
            "foot_peak_lr": (peak_sum / peak_cnt.clamp(min=1)).tolist(),
            "foot_peak_max_lr": peak_max.tolist(),
            "foot_floor_lr": foot_floor.tolist(),
            "foot_n_lr": [int(v) for v in peak_cnt.tolist()],
            "foot_peak": float((peak_sum.sum() / peak_cnt.sum().clamp(min=1)).item()),
            "foot_floor": float(foot_floor.min().item()),
            "foot_peak_max": float(peak_max.max().item()),
            "foot_n": int(peak_cnt.sum().item()),
        })

    inner_env.close()

    angles = [float(a) for a in args_cli.angles.split(",")]

    print("\n===== gk_direct policy drift speeds (変位ベース) =====")
    print(f"checkpoint: {resume_path}")
    print(f"cmd clip (vx, vy): {tuple(clip)}")
    print("\n--- 実測ドリフト速度 (体基準) ---")
    print(f"{'cmd(vx,vy)':>13} {'fwd':>7} {'lat':>7} {'|v|':>7} {'yaw drift':>10} {'n':>5} {'reset':>6}")
    for r in results:
        speed = math.hypot(r["fwd"], r["lat"])
        print(
            f"{r['cmd'][0]:6.2f},{r['cmd'][1]:5.2f} {r['fwd']:7.3f} {r['lat']:7.3f} {speed:7.3f} "
            f"{r['head_drift']:9.1f}° {r['n']:5d} {r['resets']:6d}"
        )

    print("\n--- 足の持ち上げ高さ (1 歩ごとのピーク) ---")
    print(f"{'cmd(vx,vy)':>13} {'左平均':>9} {'右平均':>9}")
    for r in results:
        lift_l = r["foot_peak_lr"][0] - r["foot_floor_lr"][0]
        lift_r = r["foot_peak_lr"][1] - r["foot_floor_lr"][1]
        print(f"{r['cmd'][0]:6.2f},{r['cmd'][1]:5.2f} {lift_l:8.3f}m {lift_r:8.3f}m")
    print(" 持ち上げ = スイング時のピーク高さ − 接地時の足リンク高さ (左右それぞれの接地高さで引く)。")

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


if __name__ == "__main__":
    main()
    simulation_app.close()
