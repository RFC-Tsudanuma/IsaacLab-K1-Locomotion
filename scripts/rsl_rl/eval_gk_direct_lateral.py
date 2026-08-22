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
    default="0,0.5;0,0.9;0,1.2;0,1.5;1.0,0;1.0,0.9;1.0,1.5;-0.6,0;-1.0,0;-0.7,0.7",
    help=(
        "Semicolon-separated 'vx,vy' command pairs to sweep. "
        "Defaults cover pure lateral (up to vy=1.5) / pure forward / diagonal / backward. "
        "★ 2026-08-20: 後退 (-0.6,0 / -1.0,0 / -0.7,0.7) を追加。それまで既定に負の vx が "
        "1 つも無く、実機で『後退指令だと人が支えないと転倒する』ことにデプロイするまで "
        "気づけなかった。測っていない量は直らない。"
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
    "--onset_reps", type=int, default=3,
    help="Number of stand-still -> full-command step responses per command (0 disables).",
)
parser.add_argument("--onset_pre_s", type=float, default=0.6, help="Stand-still duration before the step [s].")
parser.add_argument("--onset_max_s", type=float, default=2.0, help="Trace length after the step [s].")
parser.add_argument(
    "--onset_frac", type=float, default=0.9,
    help="Fraction of the steady speed used as the rise-time threshold (0.9 = t90).",
)
parser.add_argument(
    "--rev_reps", type=int, default=3,
    help=(
        "Number of full-speed reversal (+v -> -v) step responses per command (0 disables). "
        "★ 2026-08-21 追加。ゴールキーパーで実際に効くのは『静止 → 全開』ではなく "
        "**左右への振り直し** (+1.3 → -1.3 = 速度差 2.6 m/s)。それまで反転を測る指標が "
        "1 つも無く、実機で振られたときの挙動をデプロイするまで見られなかった。"
    ),
)
parser.add_argument(
    "--rev_pre_s", type=float, default=2.0,
    help="Hold the +v command this long (to reach steady state) before flipping [s].",
)
parser.add_argument(
    "--rev_max_s", type=float, default=3.0, help="Trace length after the reversal [s]."
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

    def _real_falls() -> torch.Tensor:
        """time_out を除いた「本当の終了」だけを bool (N,) で返す。

        ☠☠ 2026-08-22 修正。それまで ``step`` の返す ``dones`` をそのまま数えており、
          **エピソード満了 (time_out、20s) が「転倒」に混ざっていた**。反転の計測窓は
          3s あるので、転倒と無関係に一定割合で done が立つ。この汚染で
          「後退からの反転で転倒率 16〜26%」という誤った結論を出しかけた
          (同条件を eval_gk_fall_forensics.py で測ると **121 env-分で転倒 0 件**)。
          TerminationManager.terminated は time_out 項を除いた終了だけを返す。
        """
        tm = getattr(raw_env, "termination_manager", None)
        if tm is None or not hasattr(tm, "terminated"):
            return torch.zeros(n, dtype=torch.bool, device=device)
        return tm.terminated.to(device=device, dtype=torch.bool).flatten()

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

        # --- 立ち上がり (加速) の計測 ---
        # 静止 → 指令全開のステップ応答を撮り、定常速度の onset_frac (既定 90%) に
        # 達するまでの時間を測る。セーブに必要な横移動 0.3〜0.8m はまるごと加速区間に
        # 入るので、**定常速度より立ち上がりの方がセーブ率に効く**。
        # 定常速度はこの後の往復計測で求めるので、ここでは軌跡だけ貯めておく。
        onset_traces = []
        if args_cli.onset_reps > 0:
            pre_steps = max(1, int(args_cli.onset_pre_s / dt))
            trace_steps = max(1, int(args_cli.onset_max_s / dt))
            ones = torch.ones(n, device=device)
            for _rep in range(int(args_cli.onset_reps)):
                teleport_to_start()
                # 静止させる (コマンド 0)。歩容も stop 判定に入って足踏みが止まる。
                for _ in range(pre_steps):
                    with torch.inference_mode():
                        set_command(0.0, 0.0, ones)
                        action = policy(obs)
                        obs, _, _, _ = inner_env.step(action)
                        set_command(0.0, 0.0, ones)
                # ここでコマンドを全開に立ち上げる
                trace = torch.zeros(trace_steps, n, device=device)
                bad = torch.zeros(n, dtype=torch.bool, device=device)
                for k in range(trace_steps):
                    with torch.inference_mode():
                        set_command(eff_vx, eff_vy, ones)
                        action = policy(obs)
                        obs, _, dones, _ = inner_env.step(action)
                        set_command(eff_vx, eff_vy, ones)
                    # コマンドと同じ意味論 (base の yaw frame) で速度を測り、指令方向へ射影する
                    v_w = robot.data.root_lin_vel_w[:, :2]
                    h = robot.data.heading_w
                    v_fwd = v_w[:, 0] * torch.cos(h) + v_w[:, 1] * torch.sin(h)
                    v_lat = -v_w[:, 0] * torch.sin(h) + v_w[:, 1] * torch.cos(h)
                    trace[k] = v_fwd * axis_x + v_lat * axis_y
                    bad |= _real_falls()
                onset_traces.append((trace.cpu(), (~bad).cpu()))

        # --- 反転 (+v -> -v) の計測 ---
        # ★ 2026-08-21 追加。GK で効くのは静止からの立ち上がりより **左右の振り直し**。
        #   +v で定常まで持っていってから指令を反転させ、逆向きの定常速度の
        #   onset_frac に達するまでの時間と、その間の転倒率を測る。
        #   速度差は 2v (指令 1.3 なら 2.6 m/s) で、立ち上がりの倍の要求になる。
        # ☠ 転倒率は「反転させたから転んだ」に限らない (計測窓には反転後の走行も
        #   含まれる) が、同条件で run 間比較する分には有効。
        rev_traces = []
        rev_falls = 0
        rev_trials = 0
        if args_cli.rev_reps > 0:
            pre_steps = max(1, int(args_cli.rev_pre_s / dt))
            trace_steps = max(1, int(args_cli.rev_max_s / dt))
            ones = torch.ones(n, device=device)
            for _rep in range(int(args_cli.rev_reps)):
                teleport_to_start()
                # +v で定常まで持っていく
                for _ in range(pre_steps):
                    with torch.inference_mode():
                        set_command(eff_vx, eff_vy, ones)
                        action = policy(obs)
                        obs, _, _, _ = inner_env.step(action)
                        set_command(eff_vx, eff_vy, ones)
                # ここで指令を反転
                trace = torch.zeros(trace_steps, n, device=device)
                bad = torch.zeros(n, dtype=torch.bool, device=device)
                for k in range(trace_steps):
                    with torch.inference_mode():
                        set_command(-eff_vx, -eff_vy, ones)
                        action = policy(obs)
                        obs, _, dones, _ = inner_env.step(action)
                        set_command(-eff_vx, -eff_vy, ones)
                    v_w = robot.data.root_lin_vel_w[:, :2]
                    h = robot.data.heading_w
                    v_fwd = v_w[:, 0] * torch.cos(h) + v_w[:, 1] * torch.sin(h)
                    v_lat = -v_w[:, 0] * torch.sin(h) + v_w[:, 1] * torch.cos(h)
                    # 射影の軸は **反転前の指令方向**。したがって反転が成功すると負に振れる。
                    trace[k] = v_fwd * axis_x + v_lat * axis_y
                    bad |= _real_falls()
                rev_traces.append((trace.cpu(), (~bad).cpu()))
                rev_falls += int(bad.sum().item())
                rev_trials += n

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
            "axis": (axis_x, axis_y),
            "onset": onset_traces,
            "rev": rev_traces,
            "rev_falls": rev_falls,
            "rev_trials": rev_trials,
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

    if any(r["onset"] for r in results):
        frac = float(args_cli.onset_frac)

        def _rise_time(res: dict, level: float) -> tuple[float, float]:
            """定常速度の ``level`` 倍に達するまでの平均時間 [s] と到達率を返す。"""
            steady = res["fwd"] * res["axis"][0] + res["lat"] * res["axis"][1]
            if steady <= 1e-6:
                return float("nan"), 0.0
            hits, total, times = 0, 0, 0.0
            for trace, valid in res["onset"]:
                reached = trace >= level * steady          # (steps, env)
                ok = reached.any(dim=0) & valid
                idx = reached.float().argmax(dim=0)        # 最初に超えたステップ
                total += int(valid.sum().item())
                hits += int(ok.sum().item())
                if bool(ok.any()):
                    times += float(((idx[ok] + 1).float() * dt).sum().item())
            if hits == 0:
                return float("nan"), 0.0
            return times / hits, hits / max(total, 1)

        print("\n--- 立ち上がり (静止 → 指令全開のステップ応答) ---")
        print(f"{'cmd(vx,vy)':>13} {'定常':>8} {'t50':>8} {f't{int(frac*100)}':>8} {'到達率':>8}")
        for r in results:
            if not r["onset"]:
                continue
            steady = r["fwd"] * r["axis"][0] + r["lat"] * r["axis"][1]
            t50, _ = _rise_time(r, 0.5)
            t_hi, rate = _rise_time(r, frac)
            print(f"{r['cmd'][0]:6.2f},{r['cmd'][1]:5.2f} {steady:7.3f}m/s {t50:7.3f}s {t_hi:7.3f}s "
                  f"{rate * 100:7.1f}%")
        print(f"  定常 = 往復計測で得た速度を指令方向へ射影した値。t50/t{int(frac * 100)} は"
              f"その {50}% / {int(frac * 100)}% に達するまでの時間。")
        print(f"  試行数 = 指令あたり {args_cli.onset_reps} 回 × {args_cli.num_envs} env。"
              "到達率は転倒した env を除いた到達割合 (低いと計測窓 --onset_max_s が短い)。")
        print("  ★ 最重要指標。07-28 は約 0.6s。目標は 0.4s 台。")

    if any(r.get("rev") for r in results):
        frac = float(args_cli.onset_frac)

        def _rev_time(res: dict, level: float) -> tuple[float, float]:
            """反転後、逆向きの定常速度の ``level`` 倍に達するまでの平均時間 [s] と到達率。"""
            steady = res["fwd"] * res["axis"][0] + res["lat"] * res["axis"][1]
            if steady <= 1e-6:
                return float("nan"), 0.0
            hits, total, times = 0, 0, 0.0
            for trace, valid in res["rev"]:
                # 射影軸は反転前の向きなので、反転成功 = 十分に負へ振れること
                reached = trace <= -level * steady          # (steps, env)
                ok = reached.any(dim=0) & valid
                idx = reached.float().argmax(dim=0)
                total += int(valid.sum().item())
                hits += int(ok.sum().item())
                if bool(ok.any()):
                    times += float(((idx[ok] + 1).float() * dt).sum().item())
            if hits == 0:
                return float("nan"), 0.0
            return times / hits, hits / max(total, 1)

        print("\n--- 反転応答 (+v で定常 → 指令を反転) ---")
        print(f"{'cmd(vx,vy)':>13} {'定常':>8} {'t_rev50':>9} {f't_rev{int(frac*100)}':>9} "
              f"{'到達率':>8} {'転倒率':>8}")
        for r in results:
            if not r.get("rev"):
                continue
            steady = r["fwd"] * r["axis"][0] + r["lat"] * r["axis"][1]
            t50, _ = _rev_time(r, 0.5)
            t_hi, rate = _rev_time(r, frac)
            fall = r["rev_falls"] / max(r["rev_trials"], 1)
            print(f"{r['cmd'][0]:6.2f},{r['cmd'][1]:5.2f} {steady:7.3f}m/s {t50:8.3f}s {t_hi:8.3f}s "
                  f"{rate * 100:7.1f}% {fall * 100:7.1f}%")
        print(f"  +v で {args_cli.rev_pre_s}s 走ってから指令を反転し、**逆向き**の定常速度の"
              f" 50% / {int(frac * 100)}% に達するまでの時間を測る。")
        print("  速度差は 2v (指令 1.3 なら 2.6 m/s) で、立ち上がり (静止→全開) の倍の要求。")
        print(f"  試行数 = 指令あたり {args_cli.rev_reps} 回 × {args_cli.num_envs} env。")
        print("  ★ ゴールキーパーで実際に効くのはこちら。左右に振られたときの挙動を測る。")
        print("  ☠ 転倒率は反転そのものが原因とは限らない (窓には反転後の走行も含む)。"
              "同条件での run 間比較にのみ使うこと。")

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
