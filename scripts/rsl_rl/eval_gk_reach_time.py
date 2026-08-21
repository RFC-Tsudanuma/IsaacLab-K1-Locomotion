# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""横移動ポリシーの **到達時間曲線 t(d)** を測る診断スクリプト (Phase 0)。

なぜ別スクリプトが要るか (2026-08-17):
    :file:`eval_gk_direct_lateral.py` の立ち上がり計測は ``t90`` = 「**その run 自身の**
    定常速度の 90% に達するまでの時間」なので、**run 間で比較できない**。遅い run ほど
    低い目標に速く届いて良く見える (2026-08-15 の 2 本目は t90 0.46s だが到達速度
    0.64m/s で、07-28 の 0.565s / 1.15m/s より圧倒的に遅い)。

    ゴールキーパーにとっての価値は速度そのものではなく「**必要横移動量 d を何秒で
    覆えるか**」で、これは絶対時間なのでそのまま run 間比較できる。セーブ率が必要横移動量
    でほぼ決まる (0〜0.75m 約90% → 1.0〜1.5m 63.7% → 1.5m+ 48.7%、2026-08-17 実測) 以上、
    横移動ポリシーの合否はこの曲線ひとつで判定すべき。

測るもの:
    t(d)      : 静止状態から指令全開にした瞬間を t=0 として、**ゴールライン方向 (world y)**
                の変位が d [m] に達するまでの時間 [s]。
    等価速度   : d / t(d)。「d を覆うのに平均何 m/s 出ていたか」= 立ち上がりを含む実効速度。
    x 逸れ     : d に到達した時点での前後方向 (world x) のずれ [m]。ガードラインから
                どれだけ離れてしまうか。
    yaw ずれ   : 同時点でのヨー角のずれ [deg]。弧を描く症状の絶対量。
    v_late    : 後半区間 (d_ref1 → d_ref2) の平均速度 [m/s]。加速を終えた定常速度の目安。

**変位は body frame ではなく world frame で測る**。ゴールに対して横に何 m 動けたかが
知りたい量であり、体が回ってしまえば body-y に速度が出ていてもゴールラインは覆えない。
つまりヨードリフトのペナルティがこの指標に自動的に含まれる (既存 eval は body frame)。

歩行位相について:
    ``phase_obs`` は ``episode_length_buf`` の関数なので、リセット直後は **全 env が同位相**
    になる。指令を立ち上げた瞬間の位相は立ち上がり時間に効く (どちらの足が浮いているか) ため、
    rep ごとに静止保持時間を 1 周期 / reps ずつずらして位相を一様にサンプルする。
    向きも rep ごとに ± を交互にして左右非対称の影響を打ち消す。

使い方 (コンテナ内・リポジトリ直下):
    /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/eval_gk_reach_time.py \\
        --checkpoint logs/rsl_rl/k1_gk_direct_stage1/2026-07-28_17-13-15/model_XXXX.pt \\
        --num_envs 64 --reps 6 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Measure the lateral policy's reach-time curve t(d).")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument(
    "--task", type=str, default="Isaac-GoalkeeperDirect-Stage1-K1-Play-v0",
    help="Task providing the scene/observations (Stage1 Play; observations match 07-28).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument(
    "--override_json", type=str, default=None, help="JSON file with dot-path overrides."
)
parser.add_argument(
    "--cmd_list", type=str, default="0,1.3;0,1.0",
    help="Semicolon-separated 'vx,vy' commands to step to. Default: full lateral and a mid one.",
)
parser.add_argument(
    "--dist_list", type=str, default="0.25,0.5,0.75,1.0,1.25,1.5,2.0",
    help="Lateral distances [m] at which to report the reach time.",
)
parser.add_argument("--reps", type=int, default=6, help="Step-response repetitions per command.")
parser.add_argument("--pre_s", type=float, default=1.0, help="Stand-still duration before the step [s].")
parser.add_argument(
    "--pre_cmd", type=float, nargs=2, default=[0.0, 0.0], metavar=("VX", "VY"),
    help=(
        "Command held during the pre-step phase = the goalkeeper's READY STANCE. "
        "Default (0,0) is a cold start from frozen standing: phase_obs zeroes the gait phase "
        "below |cmd|=0.05, so the policy must restart the gait from scratch. A real keeper is "
        "not frozen while waiting, so pass e.g. '0.3 0' to keep the phase running and measure "
        "the onset that actually applies in the task."
    ),
)
parser.add_argument(
    "--pre_alt_s", type=float, default=0.0,
    help=(
        "Flip the sign of --pre_cmd every this many seconds so the robot steps roughly in "
        "place instead of drifting away. 0 disables. Use ~0.4 with --pre_cmd 0.3 0."
    ),
)
parser.add_argument("--max_s", type=float, default=4.0, help="Trace length after the step [s].")
parser.add_argument("--start_x", type=float, default=2.5, help="Teleport x [m] (kept clear of the goal frame).")
parser.add_argument(
    "--phase_hz", type=float, default=1.6,
    help="Gait frequency [Hz]; used only to stagger the onset phase across reps.",
)
parser.add_argument(
    "--late_refs", type=float, nargs=2, default=[1.0, 2.0], metavar=("D1", "D2"),
    help="Distance pair used to estimate the post-acceleration steady speed [m].",
)
parser.add_argument(
    "--cmd_clip", type=float, nargs=2, default=[1.0, 1.3], metavar=("VX", "VY"),
    help="Per-axis command clip. 07-28 was trained on vx +/-1.0, vy +/-1.3.",
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


def _pct(values: list[float], q: float) -> float:
    """百分位数 (線形補間なしの近傍取り)。空なら nan。"""
    if not values:
        return float("nan")
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed

    if args_cli.override_json is not None:
        from config_overrides import apply_overrides_from_file

        apply_overrides_from_file(args_cli.override_json, env_cfg=env_cfg, agent_cfg=agent_cfg)

    # --- コマンドを固定注入するための下準備 ---
    vc = env_cfg.commands.base_velocity
    vc.heading_command = False
    vc.rel_standing_envs = 0.0
    vc.resampling_time_range = (1.0e9, 1.0e9)

    # 計測を邪魔する終了条件を無効化 (転倒 base_contact / base_height は残す)。
    if hasattr(env_cfg.terminations, "out_of_bounds"):
        env_cfg.terminations.out_of_bounds = None
    # 1 コマンドあたり reps 回のトレースを 1 エピソード内で撮り切る。
    total_s = (args_cli.pre_s + args_cli.max_s + 1.0) * max(int(args_cli.reps), 1) * 4.0
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), total_s)
    # 最大 2m 横に走るので、env 間の余裕を広げておく (Stage1 の既定は 2.5m)。
    env_cfg.scene.env_spacing = max(float(getattr(env_cfg.scene, "env_spacing", 2.5)), 6.0)

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
    dt = raw_env.step_dt
    n = raw_env.num_envs
    device = raw_env.device

    dists = sorted(float(v) for v in args_cli.dist_list.split(","))
    clip = [float(args_cli.cmd_clip[0]), float(args_cli.cmd_clip[1])]
    cmd_pairs = []
    for chunk in args_cli.cmd_list.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vx_s, vy_s = chunk.split(",")
        cmd_pairs.append((float(vx_s), float(vy_s)))

    pre_steps_base = max(1, int(args_cli.pre_s / dt))
    trace_steps = max(1, int(args_cli.max_s / dt))
    cycle_steps = max(1, int(round(1.0 / (max(float(args_cli.phase_hz), 1e-3) * dt))))
    reps = max(1, int(args_cli.reps))

    def set_command(vx: float, vy: float, sign: float) -> None:
        """base_velocity コマンドを (vx, vy, 0) * sign に固定する。inference_mode 内で呼ぶ。"""
        cmd_term.vel_command_b[:, 0] = vx * sign
        cmd_term.vel_command_b[:, 1] = vy * sign
        cmd_term.vel_command_b[:, 2] = 0.0

    def teleport_to_start() -> None:
        """全 env を (start_x, 0, 既定高さ)・yaw=0 に置き直し速度 0 にする。"""
        with torch.inference_mode():
            pose = torch.zeros(n, 7, device=device)
            pose[:, 0] = raw_env.scene.env_origins[:, 0] + float(args_cli.start_x)
            pose[:, 1] = raw_env.scene.env_origins[:, 1]
            pose[:, 2] = robot.data.default_root_state[:, 2]
            pose[:, 3] = 1.0  # 単位クォータニオン = yaw 0
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=device))

    results = []
    obs = inner_env.get_observations()

    for vx_cmd, vy_cmd in cmd_pairs:
        eff_vx = max(-clip[0], min(clip[0], vx_cmd))
        eff_vy = max(-clip[1], min(clip[1], vy_cmd))
        if abs(eff_vy) < 1e-6:
            print(f"[WARN] skip cmd ({vx_cmd}, {vy_cmd}): 横成分が 0 なので t(d) を定義できない")
            continue

        # d ごとの到達時間 [s] / その時点の x 逸れ [m] / yaw ずれ [deg] をためる
        times: dict[float, list[float]] = {d: [] for d in dists}
        xdrift: dict[float, list[float]] = {d: [] for d in dists}
        yawdrift: dict[float, list[float]] = {d: [] for d in dists}
        attempts = 0
        falls = 0
        onset_phases: list[float] = []

        for rep in range(reps):
            sign = 1.0 if rep % 2 == 0 else -1.0
            # 位相を 1 周期にわたって一様にずらす (rep ごとに cycle/reps ステップ足す)
            pre_steps = pre_steps_base + int(round(rep * cycle_steps / reps))

            teleport_to_start()
            # --- 待機姿勢 (ready stance) ---
            # pre_cmd=(0,0) なら従来通りの冷間始動 (位相がゼロ埋めされた完全静止)。
            # 非ゼロなら位相が回ったまま指令だけ切り替わる = 実タスクに近い条件。
            pre_vx, pre_vy = float(args_cli.pre_cmd[0]), float(args_cli.pre_cmd[1])
            alt_steps = max(1, int(args_cli.pre_alt_s / dt)) if args_cli.pre_alt_s > 0 else 0
            pre_sign = 1.0
            for k_pre in range(pre_steps):
                if alt_steps and k_pre % alt_steps == 0:
                    pre_sign = -pre_sign
                with torch.inference_mode():
                    set_command(pre_vx * pre_sign, pre_vy * pre_sign, 1.0)
                    action = policy(obs)
                    obs, _, _, _ = inner_env.step(action)
                    set_command(pre_vx * pre_sign, pre_vy * pre_sign, 1.0)

            # 立ち上げ直前の位相 (全 env 共通のはずだが念のため平均を記録)
            t_now = (raw_env.episode_length_buf.float() * dt).mean().item()
            onset_phases.append((float(args_cli.phase_hz) * t_now) % 1.0)

            pos0 = (robot.data.root_pos_w[:, :2] - raw_env.scene.env_origins[:, :2]).clone()
            yaw0 = robot.data.heading_w.clone()
            reach = {d: torch.full((n,), -1, dtype=torch.long, device=device) for d in dists}
            reach_x = {d: torch.zeros(n, device=device) for d in dists}
            reach_yaw = {d: torch.zeros(n, device=device) for d in dists}
            dead = torch.zeros(n, dtype=torch.bool, device=device)

            for k in range(trace_steps):
                with torch.inference_mode():
                    set_command(eff_vx, eff_vy, sign)
                    action = policy(obs)
                    obs, _, dones, _ = inner_env.step(action)
                    set_command(eff_vx, eff_vy, sign)

                # ☠ done_mask は **参照より先に** 反映する。IsaacLab は step の中で
                #   リセットまで済ませてから観測を返すので、done が立った env の
                #   root_pos_w はもう再スポーン後の座標になっている。後から dead に
                #   入れると、その 1 ステップぶんが偽の「到達」として混入する。
                done_mask = (
                    dones.to(device=device, dtype=torch.bool).flatten()
                    if dones is not None
                    else torch.zeros(n, dtype=torch.bool, device=device)
                )
                dead |= done_mask

                pos = robot.data.root_pos_w[:, :2] - raw_env.scene.env_origins[:, :2]
                rel = pos - pos0
                # ゴールライン方向 = world y。指令の向きへ射影する。
                s = rel[:, 1] * sign
                dx = rel[:, 0].abs()
                dyaw = (robot.data.heading_w - yaw0)
                dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw)).abs()

                for d in dists:
                    newly = (s >= d) & (reach[d] < 0) & ~dead
                    if bool(newly.any()):
                        reach[d][newly] = k
                        reach_x[d][newly] = dx[newly]
                        reach_yaw[d][newly] = dyaw[newly]

            attempts += n
            falls += int(dead.sum().item())
            for d in dists:
                ok = reach[d] >= 0
                if bool(ok.any()):
                    times[d].extend(((reach[d][ok] + 1).float() * dt).tolist())
                    xdrift[d].extend(reach_x[d][ok].tolist())
                    yawdrift[d].extend((reach_yaw[d][ok] * 180.0 / math.pi).tolist())

        results.append({
            "cmd": (vx_cmd, vy_cmd),
            "sent": (eff_vx, eff_vy),
            "times": times,
            "xdrift": xdrift,
            "yawdrift": yawdrift,
            "attempts": attempts,
            "falls": falls,
            "onset_phases": onset_phases,
        })

    inner_env.close()

    print("\n" + "=" * 82)
    print(f"checkpoint: {resume_path}")
    print(f"task: {args_cli.task} / envs {n} / reps {reps} (= 試行 {n * reps} / コマンド)")
    if args_cli.override_json:
        print(f"override: {args_cli.override_json}")
    pre_desc = "完全静止 (冷間始動)" if (abs(args_cli.pre_cmd[0]) + abs(args_cli.pre_cmd[1])) < 1e-9 else (
        f"待機指令 ({args_cli.pre_cmd[0]}, {args_cli.pre_cmd[1]})"
        + (f" を {args_cli.pre_alt_s}s ごとに反転" if args_cli.pre_alt_s > 0 else "")
    )
    print(f"cmd clip (vx, vy): {tuple(clip)} / 待機 {args_cli.pre_s}s [{pre_desc}] → 全開 {args_cli.max_s}s")
    print("=" * 82)

    d1, d2 = float(args_cli.late_refs[0]), float(args_cli.late_refs[1])

    for r in results:
        print(f"\n--- 到達時間 t(d)   cmd (vx,vy) = ({r['sent'][0]:.2f}, {r['sent'][1]:.2f}) ---")
        print(f"{'d [m]':>7} {'n':>6} {'t50 [s]':>9} {'t90 [s]':>9} {'平均 [s]':>10}"
              f" {'等価速度':>10} {'x逸れ':>8} {'yawずれ':>9} {'到達率':>8}")
        for d in dists:
            tv = r["times"][d]
            if not tv:
                print(f"{d:7.2f} {0:6d} {'--':>9} {'--':>9} {'--':>10} {'--':>10} {'--':>8} {'--':>9}"
                      f" {0.0:7.1f}%")
                continue
            mean_t = _mean(tv)
            print(
                f"{d:7.2f} {len(tv):6d} {_pct(tv, 0.5):8.3f}s {_pct(tv, 0.9):8.3f}s {mean_t:9.3f}s"
                f" {d / max(mean_t, 1e-6):9.3f} {_mean(r['xdrift'][d]):7.3f}m"
                f" {_mean(r['yawdrift'][d]):8.1f}° {len(tv) / max(r['attempts'], 1) * 100:7.1f}%"
            )
        t1, t2 = _mean(r["times"].get(d1, [])), _mean(r["times"].get(d2, []))
        if t1 == t1 and t2 == t2 and t2 > t1:  # nan チェック
            print(f"  v_late ({d1:.2f}→{d2:.2f}m) = {(d2 - d1) / (t2 - t1):.3f} m/s  (加速を終えた定常速度の目安)")
        print(f"  転倒/リセット: {r['falls']} / {r['attempts']} 試行 ({r['falls'] / max(r['attempts'], 1):.2%})")
        ph = ", ".join(f"{p:.2f}" for p in r["onset_phases"])
        print(f"  立ち上げ時の歩行位相 (0〜1): {ph}")

    print("\n  ★ 到達率が 100% を大きく割る d は「その距離まで届かなかった」= 転倒か速度不足。")
    print("  ★ 等価速度 = d / 平均到達時間。立ち上がりを含むので定常速度より必ず小さい。")
    print("  ★ x逸れ / yawずれ は world 基準。ゴールラインからの離れと弧の絶対量。")
    print("  ★ この表は **run 間でそのまま比較できる** (t90 のような自己相対の閾値ではない)。")


if __name__ == "__main__":
    main()
    simulation_app.close()
