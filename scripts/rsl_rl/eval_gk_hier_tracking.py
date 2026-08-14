# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""階層版ゴールキーパー Stage 1 の「横移動の質」を計測する評価スクリプト。

:mod:`eval_goalkeeper_speed` が実効横移動速度 (到達時間) だけを見るのに対し、こちらは
**下位の定常ドリフトを上位が打ち消せているか** を主目的に測る。

背景 (eval_gk_direct_lateral.py で測った凍結下位 07-28 の素の挙動):
    * 横追従は優秀 (指令 1.3 → 実測 1.278 m/s)
    * ただし vy だけ指令すると **yaw が約 10°/s ドリフト**して円を描く
    * 同時に **約 -0.10 m/s で後退**する (指令 vy >= 0.9 のとき、速度に依らずほぼ一定)
上位は wz ≈ -0.175 rad/s と vx ≈ +0.10 m/s の定常オフセットでこれを打ち消すはずで、
本スクリプトの ``yaw drift`` と ``x drift`` がその答え合わせになる。**0 に近ければ成功、
10°/s・-0.10 m/s のままなら上位が補正を学べていない。**

計測単位は「レグ」= 目標 y が再サンプルされてから到達するまでの 1 区間。
Stage 1 は到達すると (エピソードを切らずに) 目標を採り直すので、1 エピソードから
複数のレグが取れる。移動距離でビン分けして出すので、立ち上がり支配の短距離と
定常速度が乗る長距離を分けて読める。

使い方 (コンテナ内・リポジトリ直下):
    /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/eval_gk_hier_tracking.py \\
        --task Isaac-GoalkeeperHier-Stage1-K1-Play-v0 \\
        --checkpoint logs/rsl_rl/k1_gk_hier_stage1/<run>/model_2999.pt \\
        --frozen_checkpoint logs/rsl_rl/k1_gk_direct_stage1/2026-07-28_17-13-15/exported/policy.pt \\
        --low_level_obs_group low_level \\
        --high_action_clip 1.0 1.3 1.0 --high_action_deadband 0.1 \\
        --num_envs 64 --duration_s 120 --headless

    # 学習時と同じ下位 DR を掛けて頑健性を見る場合
    ... --cmd_scale_range 0.8 1.0 --cmd_delay_range 1 3
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate hierarchical goalkeeper stage-1 tracking quality.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument(
    "--task", type=str, default="Isaac-GoalkeeperHier-Stage1-K1-Play-v0", help="Task name."
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--duration_s", type=float, default=120.0, help="Measurement duration [s].")
parser.add_argument(
    "--reach_tol", type=float, default=0.15,
    help="Reach tolerance [m]. Matches GoalkeeperParamsCfg.stage1_reach_tol by default.",
)
parser.add_argument(
    "--min_dist", type=float, default=0.3,
    help="Legs shorter than this [m] are skipped (they finish inside the acceleration transient and"
    " tell us nothing about steady-state tracking).",
)
parser.add_argument(
    "--settle_s", type=float, default=0.4,
    help="Window after reaching the target over which the stop quality is measured [s]. Keep it below"
    " stage1_hold_steps (0.5s by default): the env resamples the target once the hold completes, which"
    " cuts the window short.",
)
parser.add_argument(
    "--frozen_checkpoint", type=str, required=True, help="Frozen low-level walking checkpoint."
)
parser.add_argument(
    "--high_action_clip", type=float, nargs=3, default=[1.0, 1.3, 1.0], metavar=("VX", "VY", "WZ"),
    help="Per-axis high-level action clip. Must match training.",
)
parser.add_argument("--low_level_obs_group", type=str, default="low_level")
parser.add_argument("--low_level_cmd_term_name", type=str, default="velocity_commands")
parser.add_argument(
    "--high_action_deadband", type=float, default=0.1,
    help="Norm-based deadband on the high-level command. MUST match training (recorded in the run's"
    " params/goalkeeper_meta.txt) — the stop-quality numbers are meaningless otherwise.",
)
parser.add_argument(
    "--cmd_scale_range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
    help="Low-level gain DR. Off by default so the report shows nominal behaviour.",
)
parser.add_argument(
    "--cmd_delay_range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
    help="Command transport delay DR [ticks]. Off by default.",
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
import statistics
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

# 移動距離のビン [m)。短距離は加速区間が支配し、長距離ほど定常速度が乗る。
_DIST_BINS = [(0.3, 0.7), (0.7, 1.2), (1.2, 2.0), (2.0, 99.0)]

# 比較用の基準値: 凍結下位 07-28 を単体で走らせたときの実測 (eval_gk_direct_lateral.py)。
# 上位がドリフトを打ち消せていれば、下の yaw/x drift はこれより 0 に近づくはず。
_FROZEN_YAW_DRIFT_DPS = 10.0
_FROZEN_X_DRIFT_MPS = -0.10


def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
    """角度を (-pi, pi] に畳む。heading の差分を取るときに 2pi 跨ぎで壊れないように。"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _stat(vals: list[float]) -> tuple[float, float]:
    """(mean, p90) を返す。空なら (nan, nan)。"""
    if not vals:
        return float("nan"), float("nan")
    s = sorted(vals)
    p90 = s[min(len(s) - 1, int(0.9 * len(s)))]
    return statistics.fmean(vals), p90


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
        action_deadband=args_cli.high_action_deadband,
        cmd_scale_range=args_cli.cmd_scale_range,
        cmd_delay_range=args_cli.cmd_delay_range,
    )

    print(f"[INFO] Loading high-level checkpoint: {resume_path}")
    runner = OnPolicyRunner(hier_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=inner_env.unwrapped.device)

    raw_env = inner_env.unwrapped
    robot = raw_env.scene["robot"]
    dt = raw_env.step_dt
    n = raw_env.num_envs
    device = raw_env.device
    n_steps = int(args_cli.duration_s / dt)
    settle_steps = max(1, int(args_cli.settle_s / dt))
    guard_x = float(raw_env.cfg.goalkeeper.guard_x)

    # --- レグ (目標再サンプル → 到達 → 静定) の per-env 状態 ---
    PH_IDLE, PH_MOVE, PH_SETTLE = 0, 1, 2
    phase = torch.zeros(n, dtype=torch.long, device=device)
    y0 = torch.zeros(n, device=device)
    x0 = torch.zeros(n, device=device)
    h0 = torch.zeros(n, device=device)
    dist0 = torch.zeros(n, device=device)
    steps = torch.zeros(n, dtype=torch.long, device=device)
    yaw_abs_max = torch.zeros(n, device=device)
    xdev_max = torch.zeros(n, device=device)
    v_peak = torch.zeros(n, device=device)
    # 到達時点のスナップショット (静定フェーズ完了時にまとめて記録する)
    snap = {k: torch.zeros(n, device=device) for k in ("t", "v", "yawd", "xd", "yawmax", "xdevmax", "vpeak")}
    st_steps = torch.zeros(n, dtype=torch.long, device=device)
    st_cmd_sum = torch.zeros(n, device=device)
    st_zero_cnt = torch.zeros(n, device=device)
    st_speed_sum = torch.zeros(n, device=device)
    st_err_max = torch.zeros(n, device=device)

    legs: list[dict] = []
    falls = 0
    prev_target = None

    obs = hier_env.get_observations()
    for _ in range(n_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = hier_env.step(actions)

        target = raw_env._gk_target_y.clone()
        pos = robot.data.root_pos_w[:, :3] - raw_env.scene.env_origins
        y, x = pos[:, 1], pos[:, 0]
        heading = robot.data.heading_w
        err = (y - target).abs()
        # 上位が実際に下位へ渡した指令 (デッドバンド適用後)。停止品質の判定に使う。
        cmd_norm = torch.norm(raw_env._prev_high_action[:, :3], dim=1)
        speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
        v_y_abs = robot.data.root_lin_vel_w[:, 1].abs()

        done_mask = (
            dones.to(device=device, dtype=torch.bool).flatten()
            if dones is not None
            else torch.zeros(n, dtype=torch.bool, device=device)
        )
        falls += int(done_mask.sum())

        if prev_target is None:
            prev_target = target
            continue
        changed = (target - prev_target).abs() > 1e-6
        prev_target = target

        # --- 移動中の走行統計を積む ---
        moving = phase == PH_MOVE
        if bool(moving.any()):
            steps[moving] += 1
            yaw_abs_max = torch.where(moving, torch.maximum(yaw_abs_max, heading.abs()), yaw_abs_max)
            xdev_max = torch.where(moving, torch.maximum(xdev_max, (x - guard_x).abs()), xdev_max)
            v_peak = torch.where(moving, torch.maximum(v_peak, v_y_abs), v_peak)

        # --- 到達判定: MOVE -> SETTLE ---
        reached = moving & (err < args_cli.reach_tol) & (steps > 0)
        if bool(reached.any()):
            t = steps.float() * dt
            snap["t"] = torch.where(reached, t, snap["t"])
            snap["v"] = torch.where(reached, dist0 / t.clamp(min=1e-6), snap["v"])
            # ★ 主指標: 移動中に heading / x がどれだけ流れたかを「速度」に直したもの。
            #   凍結下位単体の +10°/s・-0.10 m/s と直接比較できる。
            snap["yawd"] = torch.where(
                reached, _wrap_pi(heading - h0) * (180.0 / math.pi) / t.clamp(min=1e-6), snap["yawd"]
            )
            snap["xd"] = torch.where(reached, (x - x0) / t.clamp(min=1e-6), snap["xd"])
            snap["yawmax"] = torch.where(reached, yaw_abs_max, snap["yawmax"])
            snap["xdevmax"] = torch.where(reached, xdev_max, snap["xdevmax"])
            snap["vpeak"] = torch.where(reached, v_peak, snap["vpeak"])
            phase = torch.where(reached, torch.full_like(phase, PH_SETTLE), phase)
            st_steps[reached] = 0
            st_cmd_sum[reached] = 0.0
            st_zero_cnt[reached] = 0.0
            st_speed_sum[reached] = 0.0
            st_err_max[reached] = 0.0

        # --- 静定フェーズの統計 ---
        settling = phase == PH_SETTLE
        if bool(settling.any()):
            st_steps[settling] += 1
            st_cmd_sum = torch.where(settling, st_cmd_sum + cmd_norm, st_cmd_sum)
            st_zero_cnt = torch.where(settling, st_zero_cnt + (cmd_norm < 1e-6).float(), st_zero_cnt)
            st_speed_sum = torch.where(settling, st_speed_sum + speed, st_speed_sum)
            st_err_max = torch.where(settling, torch.maximum(st_err_max, err), st_err_max)

        # --- レグの確定 ---
        # 静定窓を満了した env、または窓の途中で目標が再サンプルされた env
        # (stage1_target_tick が保持完了で採り直すので、こちらの方が普通に起きる)。
        settle_done = settling & ((st_steps >= settle_steps) | (changed & (st_steps >= 1)))
        finish = settle_done & ~done_mask
        for i in torch.nonzero(finish).flatten().tolist():
            k = float(st_steps[i])
            legs.append({
                "dist": float(dist0[i]),
                "t": float(snap["t"][i]),
                "v": float(snap["v"][i]),
                "v_peak": float(snap["vpeak"][i]),
                "yaw_drift": float(snap["yawd"][i]),
                "yaw_max": float(snap["yawmax"][i]) * 180.0 / math.pi,
                "x_drift": float(snap["xd"][i]),
                "x_dev_max": float(snap["xdevmax"][i]),
                "settle_err": float(st_err_max[i]),
                "settle_cmd": float(st_cmd_sum[i]) / k,
                "settle_zero": float(st_zero_cnt[i]) / k,
                "settle_speed": float(st_speed_sum[i]) / k,
            })
        phase = torch.where(settle_done, torch.full_like(phase, PH_IDLE), phase)
        # 転倒・タイムアウトで切れた env のレグは破棄する
        phase = torch.where(done_mask, torch.full_like(phase, PH_IDLE), phase)

        # --- 新しいレグの開始 (目標が変わった env) ---
        # ※ 上の確定処理より後に置くこと。同じステップで「静定完了 → 次の目標」が
        #    同時に起きるので、先に確定させてから新しいレグを開く。
        start = changed & ~done_mask & ((y - target).abs() >= args_cli.min_dist)
        if bool(start.any()):
            y0 = torch.where(start, y, y0)
            x0 = torch.where(start, x, x0)
            h0 = torch.where(start, heading, h0)
            dist0 = torch.where(start, (y - target).abs(), dist0)
            steps[start] = 0
            yaw_abs_max[start] = 0.0
            xdev_max[start] = 0.0
            v_peak[start] = 0.0
            phase = torch.where(start, torch.full_like(phase, PH_MOVE), phase)
        # 近すぎる目標を引いた env はレグを開かず待機に戻す
        skip = changed & ~start
        phase = torch.where(skip, torch.full_like(phase, PH_IDLE), phase)

    hier_env.close()

    # ---------------------------------------------------------------- 集計
    print("\n===== hierarchical GK stage-1 tracking quality =====")
    print(f"checkpoint : {resume_path}")
    print(f"clip (vx, vy, wz) : {tuple(args_cli.high_action_clip)}   deadband : {args_cli.high_action_deadband}")
    print(f"low-level DR      : scale {args_cli.cmd_scale_range} / delay {args_cli.cmd_delay_range}")
    print(f"guard_x           : {guard_x:.2f} m")

    if not legs:
        print("\n[WARN] レグが 1 本も取れませんでした。--duration_s を延ばすか、"
              "--reach_tol / --min_dist を確認してください。")
        return

    print(f"\nlegs : {len(legs)}   終了 (転倒/タイムアウト) : {falls} 回")

    print("\n--- 移動性能 (距離ビン別) ---")
    print(f"{'dist [m)':>12} {'n':>5} {'t [s]':>7} {'v_mean':>8} {'v_peak':>8}")
    for lo, hi in _DIST_BINS:
        sel = [g for g in legs if lo <= g["dist"] < hi]
        if not sel:
            continue
        t_m, _ = _stat([g["t"] for g in sel])
        v_m, _ = _stat([g["v"] for g in sel])
        vp_m, _ = _stat([g["v_peak"] for g in sel])
        label = f"{lo:.1f}-{hi:.1f}" if hi < 90 else f"{lo:.1f}+"
        print(f"{label:>12} {len(sel):5d} {t_m:7.2f} {v_m:8.3f} {vp_m:8.3f}")

    yawd_m, yawd_p90 = _stat([g["yaw_drift"] for g in legs])
    yawmax_m, yawmax_p90 = _stat([g["yaw_max"] for g in legs])
    xd_m, xd_p90 = _stat([g["x_drift"] for g in legs])
    xdev_m, xdev_p90 = _stat([g["x_dev_max"] for g in legs])

    print("\n--- 直進性 / 定位置維持 (★ 本スクリプトの主目的) ---")
    print(f"{'':<24}{'mean':>9}{'p90':>9}   凍結下位 07-28 単体")
    print(f"{'yaw ドリフト [deg/s]':<24}{yawd_m:9.2f}{yawd_p90:9.2f}   {_FROZEN_YAW_DRIFT_DPS:+.1f}")
    print(f"{'|yaw| 最大 [deg]':<24}{yawmax_m:9.2f}{yawmax_p90:9.2f}   (単体は累積して発散)")
    print(f"{'x ドリフト [m/s]':<24}{xd_m:9.3f}{xd_p90:9.3f}   {_FROZEN_X_DRIFT_MPS:+.2f}")
    print(f"{'|x - guard_x| 最大 [m]':<24}{xdev_m:9.3f}{xdev_p90:9.3f}   -")
    print("  → yaw / x ドリフトが 0 に近いほど、上位が下位の定常ドリフトを打ち消せている。")
    print("     右列の値に近いままなら、上位は補正を学べておらず下位の癖がそのまま出ている。")

    err_m, err_p90 = _stat([g["settle_err"] for g in legs])
    cmd_m, _ = _stat([g["settle_cmd"] for g in legs])
    zero_m, _ = _stat([g["settle_zero"] for g in legs])
    spd_m, spd_p90 = _stat([g["settle_speed"] for g in legs])

    print(f"\n--- 停止品質 (到達後 {args_cli.settle_s:.1f}s の平均) ---")
    print(f"{'':<24}{'mean':>9}{'p90':>9}")
    print(f"{'目標との誤差 最大 [m]':<24}{err_m:9.3f}{err_p90:9.3f}")
    print(f"{'ベース速度 [m/s]':<24}{spd_m:9.3f}{spd_p90:9.3f}")
    print(f"{'上位指令ノルム':<24}{cmd_m:9.3f}{'':>9}")
    print(f"{'指令ゼロ率 (デッドバンド)':<24}{zero_m:9.3f}{'':>9}")
    print("  → 指令ゼロ率が高いほど下位が停止モードに入れている (足踏みしていない)。")
    print("     ここが 0 に近いのに指令ノルムが小さい = デッドバンド閾値の直上に張り付いて")
    print("     足踏みし続けている状態なので、閾値を上げるか hold_at_target を強める。")


if __name__ == "__main__":
    main()
    simulation_app.close()
