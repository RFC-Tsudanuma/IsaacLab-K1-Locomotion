# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""gk_direct Stage1 の **足上げ (クリアランス)・位相整合・ヨードリフト** を実測する診断スクリプト。

なぜ要るか (2026-08-15):
    07-28 の足上げは 4.3cm で頭打ち。weight/目標値を上げる軸は探索し尽くしており
    (上げると跳躍に退行、跳躍を封じると 2.6〜3.3cm)、**報酬の測り方** を変える前に
    「何が足上げを制限しているのか」を切り分けるのが目的。

    測る仮説は 3 つ:
      (a) 位相同期のズレ: ``foot_clearance_ji`` は phase_freq を **固定値** で使うが
          (rewards.py:370)、``feet_phase`` / ``phase_obs`` は ``get_phase_freq`` 経由で
          ``randomize_phase_freq`` (±0.05Hz, startup 固定) に追従する。エピソード 20s では
          δ=0.05 の個体が t=10s で **逆位相** になる。逆位相でも取れる唯一の解が
          「体ごと持ち上げる」= 跳躍なので、跳躍への退行と症状が整合する。
          → 実際の遊脚区間と固定 1.6Hz のスイング窓の重なりを時間バケットごとに測る。
      (b) 可動/トルクの限界か、報酬バランスか: 遊脚中の関節余裕とトルク飽和率を測る。
          余裕があるなら報酬の作り替えで伸びる。飽和しているなら歩容タイミング側の話。
      (c) 測定点の問題: 既存指標は **足リンク原点** の高さ。実際につまずくのはつま先/かかと。
          底屈 5° でつま先は 1.0cm 下がる (足長 0.185m)。→ 足裏 4 隅の最小高さで測る。

    併せて「ずっと横移動していると曲がる」症状のために、ヨー角速度を **符号付き・
    指令方向別** に出す (既存 eval_gk_direct_lateral.py は abs で平均するので符号が消える)。

足の形状 (K1_locomotion.urdf の Left_Foot.STL バウンディングボックス):
    x: -0.0659 (かかと) 〜 +0.1195 (つま先) / y: ±0.040 / z 下端: -0.0382 (足裏)
    → 足リンク原点は足裏から 3.82cm 上。既存の「持ち上げ高さ」はこの原点基準。

使い方 (コンテナ内・リポジトリ直下):
    /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/eval_gk_clearance.py \\
        --checkpoint logs/rsl_rl/k1_gk_direct_stage1/2026-07-28_17-13-15/model_14999.pt \\
        --cmd_list "0,0.9;0,1.2;0,1.3" --num_envs 32 --headless

    位相 DR を殺した比較 (仮説 (a) の対照):
    ... --force_phase_freq 1.6
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Measure gk_direct foot clearance / phase alignment / yaw drift.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments.")
parser.add_argument(
    "--task", type=str, default="Isaac-GoalkeeperDirect-Stage1-K1-Play-v0",
    help="Task providing the scene/observations (gk_direct Stage1 Play).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument(
    "--cmd_list", type=str, default="0,0.9;0,1.2;0,1.3",
    help="Semicolon-separated 'vx,vy' command pairs to sweep.",
)
parser.add_argument("--hold_s", type=float, default=20.0, help="Measurement duration per command pair [s].")
parser.add_argument("--settle_s", type=float, default=1.5, help="Transient skipped after each direction flip [s].")
parser.add_argument("--travel_limit", type=float, default=1.5, help="Travel distance before the command flips [m].")
parser.add_argument("--start_x", type=float, default=2.5, help="Teleport x [m] (kept clear of the goal frame).")
parser.add_argument(
    "--cmd_clip", type=float, nargs=2, default=[1.0, 1.5], metavar=("VX", "VY"),
    help="Per-axis command clip (gk_direct Stage1 range: vx +/-1.0, vy +/-1.3).",
)
parser.add_argument("--gait_hz", type=float, default=1.6, help="Phase frequency used by foot_clearance_ji (fixed).")
parser.add_argument("--stance_ratio", type=float, default=0.5, help="Stance ratio used by foot_clearance_ji.")
parser.add_argument(
    "--force_phase_freq", type=float, default=None,
    help="Overwrite the per-env randomized phase frequency with this value (kills the phase DR).",
)
parser.add_argument("--contact_th", type=float, default=1.0, help="Contact force threshold for 'foot on ground' [N].")
parser.add_argument("--min_swing_s", type=float, default=0.08, help="Ignore airborne segments shorter than this [s].")
parser.add_argument(
    "--foot_box", type=float, nargs=4, default=[0.1195, -0.0659, 0.040, -0.0382],
    metavar=("TOE_X", "HEEL_X", "HALF_Y", "SOLE_Z"),
    help="Sole corner offsets in the foot-link frame [m] (from the collision mesh bbox).",
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
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401

# 1 遊脚区間あたりに保持するサンプル数の上限 (50Hz で 1.28s ぶん)。
_MAX_SEG = 64
_LEG_JOINTS = ["Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll"]
# 位相整合をエピソード時刻で刻むバケット境界 [s]。
_TIME_BUCKETS = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 1.0e9)]


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

    # コマンドを固定注入するため、heading 制御・再サンプル・standing env を切る。
    vc = env_cfg.commands.base_velocity
    vc.heading_command = False
    vc.rel_standing_envs = 0.0
    vc.resampling_time_range = (1.0e9, 1.0e9)

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
    contact = raw_env.scene.sensors["contact_forces"]
    cmd_term = raw_env.command_manager.get_term("base_velocity")
    dt = raw_env.step_dt
    n = raw_env.num_envs
    device = raw_env.device

    foot_names = ["left_foot_link", "right_foot_link"]
    foot_body_ids = [robot.find_bodies(nm)[0][0] for nm in foot_names]
    foot_sensor_ids = [contact.body_names.index(nm) for nm in foot_names]
    leg_joint_ids = [
        [robot.find_joints(f"{side}_{j}")[0][0] for j in _LEG_JOINTS]
        for side in ("Left", "Right")
    ]

    # 位相 DR。startup mode なので env ごとに固定。
    if args_cli.force_phase_freq is not None:
        setattr(
            raw_env, "_phase_freq_per_env",
            torch.full((n,), float(args_cli.force_phase_freq), device=device),
        )
    phase_freq_env = getattr(raw_env, "_phase_freq_per_env", None)
    if phase_freq_env is None:
        phase_freq_env = torch.full((n,), float(args_cli.gait_hz), device=device)
    phase_offset = (phase_freq_env - float(args_cli.gait_hz)).clone()

    # 足裏 4 隅のオフセット (足リンク座標系)。
    toe_x, heel_x, half_y, sole_z = [float(v) for v in args_cli.foot_box]
    corners = torch.tensor(
        [
            [toe_x, half_y, sole_z],
            [toe_x, -half_y, sole_z],
            [heel_x, half_y, sole_z],
            [heel_x, -half_y, sole_z],
        ],
        device=device,
    )  # (4, 3)

    settle_steps = max(1, int(args_cli.settle_s / dt))
    n_steps = int(args_cli.hold_s / dt)
    min_swing_steps = max(2, int(args_cli.min_swing_s / dt))
    stance_threshold = 2.0 * math.pi * float(args_cli.stance_ratio)

    joint_lim = robot.data.joint_pos_limits  # (n, J, 2)
    # トルク上限は **アクチュエータモデル側の effort_limit** を使う。data.joint_effort_limits は
    # physx の DOF max force で、明示アクチュエータ (DelayedPD / ActuatorNet) では
    # 大きな値のままクリップは actuator 側で行われるため、飽和率の分母にならない。
    effort_lim = robot.data.joint_effort_limits.clone()
    for act in robot.actuators.values():
        lim = getattr(act, "effort_limit", None)
        if lim is None:
            continue
        idx = act.joint_indices
        effort_lim[:, idx] = lim if torch.is_tensor(lim) else float(lim)
    effort_lim = effort_lim.clamp(min=1e-3)  # (n, J)

    def set_command(vx: float, vy: float, direction: torch.Tensor) -> None:
        cmd_term.vel_command_b[:, 0] = vx * direction
        cmd_term.vel_command_b[:, 1] = vy * direction
        cmd_term.vel_command_b[:, 2] = 0.0

    def teleport_to_start() -> None:
        with torch.inference_mode():
            pose = torch.zeros(n, 7, device=device)
            pose[:, 0] = raw_env.scene.env_origins[:, 0] + float(args_cli.start_x)
            pose[:, 1] = raw_env.scene.env_origins[:, 1]
            pose[:, 2] = robot.data.default_root_state[:, 2]
            pose[:, 3] = 1.0
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=device))

    def sole_min_z() -> torch.Tensor:
        """足裏 4 隅のうち **最も低い点** のワールド z を左右ぶん返す。 (n, 2)"""
        out = []
        for bid in foot_body_ids:
            pos = robot.data.body_pos_w[:, bid, :]            # (n, 3)
            quat = robot.data.body_quat_w[:, bid, :]          # (n, 4) wxyz
            q = quat.unsqueeze(1).expand(n, 4, 4).reshape(-1, 4)
            c = corners.unsqueeze(0).expand(n, 4, 3).reshape(-1, 3)
            world = quat_apply(q, c).reshape(n, 4, 3) + pos.unsqueeze(1)
            out.append(world[:, :, 2].min(dim=1).values)
        return torch.stack(out, dim=1)

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
        axis_x, axis_y = eff_vx / norm, eff_vy / norm

        teleport_to_start()
        anchor = (robot.data.root_pos_w[:, :2] - raw_env.scene.env_origins[:, :2]).clone()
        direction = torch.ones(n, device=device)
        since_flip = torch.zeros(n, dtype=torch.long, device=device)

        # --- 遊脚区間バッファ ---
        buf_sole = torch.zeros(n, 2, _MAX_SEG, device=device)   # 足裏最下点の絶対 z
        buf_orig = torch.zeros(n, 2, _MAX_SEG, device=device)   # 足リンク原点の絶対 z
        buf_win = torch.zeros(n, 2, _MAX_SEG, device=device)    # 報酬のスイング窓 (1/0)
        seg_len = torch.zeros(n, 2, dtype=torch.long, device=device)
        seg_valid = torch.zeros(n, 2, dtype=torch.bool, device=device)
        prev_air = torch.zeros(n, 2, dtype=torch.bool, device=device)

        segs: list[dict] = []          # 確定した遊脚区間の統計
        floor = torch.full((2,), float("inf"), device=device)   # 接地時の足裏最下点 (地面 z の実測)
        floor_orig = torch.full((2,), float("inf"), device=device)

        # --- 速度・ヨー (指令方向別) ---
        vel_sum = {1.0: torch.zeros(2, device=device), -1.0: torch.zeros(2, device=device)}
        yaw_sum = {1.0: torch.zeros((), device=device), -1.0: torch.zeros((), device=device)}
        vel_cnt = {1.0: torch.zeros((), device=device), -1.0: torch.zeros((), device=device)}

        # --- 位相整合 (エピソード時刻バケット × 位相オフセット群) ---
        off_group = torch.zeros(n, dtype=torch.long, device=device)
        off_group[phase_offset > 0.015] = 2
        off_group[phase_offset < -0.015] = 0
        off_group[(phase_offset >= -0.015) & (phase_offset <= 0.015)] = 1
        align_hit = torch.zeros(3, len(_TIME_BUCKETS), device=device)
        align_tot = torch.zeros(3, len(_TIME_BUCKETS), device=device)

        # --- 関節余裕・トルク飽和 (遊脚側のみ) ---
        margin_min = torch.full((2, len(_LEG_JOINTS)), float("inf"), device=device)
        torque_max = torch.zeros(2, len(_LEG_JOINTS), device=device)
        torque_sum = torch.zeros(2, len(_LEG_JOINTS), device=device)
        torque_cnt = torch.zeros(2, device=device)

        base_z_sum = torch.zeros((), device=device)
        base_vz_sq = torch.zeros((), device=device)
        base_cnt = torch.zeros((), device=device)
        resets = torch.zeros((), device=device)

        for _ in range(n_steps):
            with torch.inference_mode():
                set_command(eff_vx, eff_vy, direction)
                action = policy(obs)
                obs, _, dones, _ = inner_env.step(action)
                set_command(eff_vx, eff_vy, direction)

            steady = since_flip > settle_steps
            done_mask = (
                dones.to(device=device, dtype=torch.bool).flatten()
                if dones is not None
                else torch.zeros(n, dtype=torch.bool, device=device)
            )
            resets += done_mask.sum()

            forces = contact.data.net_forces_w[:, foot_sensor_ids, :]       # (n, 2, 3)
            on_ground = forces.norm(dim=-1) > float(args_cli.contact_th)    # (n, 2)
            air = ~on_ground

            sole_z_w = sole_min_z()                                          # (n, 2)
            orig_z_w = torch.stack(
                [robot.data.body_pos_w[:, bid, 2] for bid in foot_body_ids], dim=1
            )

            # 接地しているステップの最下点 = 地面 z の実測 (平面前提)
            grounded = on_ground & steady.unsqueeze(1)
            if bool(grounded.any()):
                masked = torch.where(grounded, sole_z_w, torch.full_like(sole_z_w, float("inf")))
                floor = torch.minimum(floor, masked.min(dim=0).values)
                masked_o = torch.where(grounded, orig_z_w, torch.full_like(orig_z_w, float("inf")))
                floor_orig = torch.minimum(floor_orig, masked_o.min(dim=0).values)

            # --- 報酬 (foot_clearance_ji) のスイング窓。位相は **固定 gait_hz** ---
            t_ep = raw_env.episode_length_buf.float() * dt                   # (n,)
            phase_l = (2.0 * math.pi * float(args_cli.gait_hz) * t_ep) % (2.0 * math.pi)
            phase_r = (phase_l + math.pi) % (2.0 * math.pi)
            win = torch.stack([phase_l >= stance_threshold, phase_r >= stance_threshold], dim=1)

            # 位相整合: 「報酬がスイングと言っているとき、実際に浮いているか」
            if bool(steady.any()):
                for bi, (lo, hi) in enumerate(_TIME_BUCKETS):
                    in_b = steady & (t_ep >= lo) & (t_ep < hi)
                    if not bool(in_b.any()):
                        continue
                    for gi in range(3):
                        sel = in_b & (off_group == gi)
                        if not bool(sel.any()):
                            continue
                        w = win[sel]
                        a = air[sel]
                        align_tot[gi, bi] += w.sum()
                        align_hit[gi, bi] += (w & a).sum()

            # --- 遊脚区間の切り出し (接触ベース) ---
            start = air & ~prev_air
            if bool(start.any()):
                seg_len[start] = 0
                seg_valid[start] = steady.unsqueeze(1).expand_as(air)[start]

            active = air & seg_valid & (seg_len < _MAX_SEG)
            if bool(active.any()):
                e_idx, f_idx = torch.nonzero(active, as_tuple=True)
                s_idx = seg_len[active]
                buf_sole[e_idx, f_idx, s_idx] = sole_z_w[active]
                buf_orig[e_idx, f_idx, s_idx] = orig_z_w[active]
                buf_win[e_idx, f_idx, s_idx] = win[active].float()
                seg_len[active] += 1

            ended = (~air) & prev_air & seg_valid & (seg_len >= min_swing_steps) & ~done_mask.unsqueeze(1)
            if bool(ended.any()):
                e_idx, f_idx = torch.nonzero(ended, as_tuple=True)
                for e, f in zip(e_idx.tolist(), f_idx.tolist()):
                    L = int(seg_len[e, f].item())
                    sole = buf_sole[e, f, :L]
                    orig = buf_orig[e, f, :L]
                    wseg = buf_win[e, f, :L]
                    lift_rel = orig - orig.min()                # 区間内の相対持ち上げ
                    apex_rel = float(lift_rel.max().item())
                    mid = lift_rel >= 0.5 * max(apex_rel, 1e-6)  # 上がりきっている区間
                    segs.append({
                        "foot": f,
                        "dur": L * dt,
                        "apex_abs": float(orig.max().item()),
                        "sole_min_abs": float(sole.min().item()),
                        "sole_min_mid_abs": float(sole[mid].min().item()),
                        "align": float(wseg.mean().item()),
                    })

            seg_valid &= air
            if bool(done_mask.any()):
                seg_valid[done_mask] = False
            prev_air = air

            # --- 遊脚側の関節余裕・トルク ---
            for f in range(2):
                sel = air[:, f] & steady & ~on_ground[:, f]
                if not bool(sel.any()):
                    continue
                ids = leg_joint_ids[f]
                q = robot.data.joint_pos[sel][:, ids]
                lo = joint_lim[sel][:, ids, 0]
                hi = joint_lim[sel][:, ids, 1]
                margin = torch.minimum(q - lo, hi - q)
                margin_min[f] = torch.minimum(margin_min[f], margin.min(dim=0).values)
                ratio = (robot.data.applied_torque[sel][:, ids].abs() / effort_lim[sel][:, ids])
                torque_max[f] = torch.maximum(torque_max[f], ratio.max(dim=0).values)
                torque_sum[f] += ratio.sum(dim=0)
                torque_cnt[f] += ratio.shape[0]

            # --- 速度・ヨー (指令方向別) ---
            if bool(steady.any()):
                for sgn in (1.0, -1.0):
                    sel = steady & (direction == sgn)
                    if not bool(sel.any()):
                        continue
                    v = robot.data.root_lin_vel_b[sel][:, :2]
                    vel_sum[sgn] += v.sum(dim=0)
                    yaw_sum[sgn] += robot.data.root_ang_vel_w[sel][:, 2].sum()
                    vel_cnt[sgn] += v.shape[0]
                base_z_sum += robot.data.root_pos_w[steady][:, 2].sum()
                base_vz_sq += robot.data.root_lin_vel_w[steady][:, 2].square().sum()
                base_cnt += int(steady.sum().item())

            # --- 往復 ---
            pos = robot.data.root_pos_w[:, :2] - raw_env.scene.env_origins[:, :2]
            rel = pos - anchor
            s = (rel[:, 0] * axis_x + rel[:, 1] * axis_y) * direction
            flip = (s > args_cli.travel_limit) | done_mask
            direction[flip] *= -1.0
            since_flip += 1
            since_flip[flip] = 0

        # --- 集計 ---
        fl = floor.tolist()
        flo = floor_orig.tolist()
        per_foot = []
        for f in range(2):
            sel = [s for s in segs if s["foot"] == f]
            lift = [s["apex_abs"] - flo[f] for s in sel]
            clr_mid = [s["sole_min_mid_abs"] - fl[f] for s in sel]
            clr_all = [s["sole_min_abs"] - fl[f] for s in sel]
            per_foot.append({
                "n": len(sel),
                "dur": _mean([s["dur"] for s in sel]),
                "lift": _mean(lift),
                "clr_mid_p50": _pct(clr_mid, 0.5),
                "clr_mid_p05": _pct(clr_mid, 0.05),
                "clr_all_p50": _pct(clr_all, 0.5),
                "clr_all_p05": _pct(clr_all, 0.05),
                "align": _mean([s["align"] for s in sel]),
            })

        results.append({
            "cmd": (vx_cmd, vy_cmd),
            "sent": (eff_vx, eff_vy),
            "feet": per_foot,
            "vel": {
                sgn: (vel_sum[sgn] / vel_cnt[sgn].clamp(min=1)).tolist() for sgn in (1.0, -1.0)
            },
            "yaw": {
                sgn: float((yaw_sum[sgn] / vel_cnt[sgn].clamp(min=1)).item()) for sgn in (1.0, -1.0)
            },
            "align_grid": (align_hit / align_tot.clamp(min=1)).tolist(),
            "align_tot": align_tot.tolist(),
            "margin": margin_min.tolist(),
            "torque_max": torque_max.tolist(),
            "torque_mean": (torque_sum / torque_cnt.clamp(min=1).unsqueeze(1)).tolist(),
            "base_z": float((base_z_sum / base_cnt.clamp(min=1)).item()),
            "base_vz_rms": float((base_vz_sq / base_cnt.clamp(min=1)).sqrt().item()),
            "resets": int(resets.item()),
        })

    inner_env.close()

    off_lo = float(phase_offset.min().item())
    off_hi = float(phase_offset.max().item())

    print("\n===== gk_direct clearance / phase / yaw diagnostics =====")
    print(f"checkpoint: {resume_path}")
    print(f"位相 DR オフセット: {off_lo:+.3f} 〜 {off_hi:+.3f} Hz (報酬側は固定 {args_cli.gait_hz} Hz)")
    print(f"足裏 4 隅オフセット: toe {args_cli.foot_box[0]:.4f} / heel {args_cli.foot_box[1]:.4f} "
          f"/ half_y {args_cli.foot_box[2]:.4f} / sole {args_cli.foot_box[3]:.4f} [m]")

    print("\n--- (1) 速度とヨードリフト (符号付き・指令方向別) ---")
    print(f"{'cmd(vx,vy)':>13} {'dir':>4} {'fwd':>7} {'lat':>7} {'yaw rate':>10} {'reset':>6}")
    for r in results:
        for sgn in (1.0, -1.0):
            v = r["vel"][sgn]
            print(f"{r['cmd'][0]:6.2f},{r['cmd'][1]:5.2f} {sgn:+4.0f} {v[0]:7.3f} {v[1]:7.3f} "
                  f"{math.degrees(r['yaw'][sgn]):9.2f}° {r['resets']:6d}")
    print("  dir = 指令の符号 (往復の向き)。yaw rate は world 系の角速度平均 [deg/s]。")
    print("  ★ dir で符号が反転 → 「進行方向側に曲がる」対称なドリフト (歩容の非対称性)。")
    print("    dir によらず同符号 → 機体/制御の固定バイアス。対策が変わるので必ず見ること。")

    print("\n--- (2) 足上げとつま先クリアランス ---")
    print(f"{'cmd(vx,vy)':>13} {'foot':>5} {'swing':>7} {'原点lift':>9} {'底面p50':>9} {'底面p05':>9} "
          f"{'離着地込p05':>11} {'n':>5}")
    for r in results:
        for f, name in enumerate(("L", "R")):
            d = r["feet"][f]
            print(f"{r['cmd'][0]:6.2f},{r['cmd'][1]:5.2f} {name:>5} {d['dur']:6.3f}s "
                  f"{d['lift']:8.3f}m {d['clr_mid_p50']:8.3f}m {d['clr_mid_p05']:8.3f}m "
                  f"{d['clr_all_p05']:10.3f}m {d['n']:5d}")
    print("  原点lift = 足リンク原点の最高点 − 接地時の原点高さ (既存指標と同じ定義)。")
    print("  底面 = 足裏 4 隅の最下点の地面からの高さ。上がりきった区間 (lift ≥ apex/2) の統計。")
    print("  ★ 人工芝のパイル高さ 20〜30mm と直接比較できるのは **底面 p05** の方。")

    print("\n--- (3) 位相整合: 報酬のスイング窓が実際の遊脚と重なっている割合 ---")
    labels = ["δ<0", "δ≈0", "δ>0"]
    head = "".join(f"{f'{lo:.0f}-{hi:.0f}s' if hi < 1e8 else f'{lo:.0f}s-':>10}" for lo, hi in _TIME_BUCKETS)
    for r in results:
        print(f" cmd=({r['cmd'][0]:.2f},{r['cmd'][1]:.2f})   " + head)
        for gi, lab in enumerate(labels):
            cells = "".join(
                f"{r['align_grid'][gi][bi]:9.2f} " if r["align_tot"][gi][bi] > 0 else f"{'--':>9} "
                for bi in range(len(_TIME_BUCKETS))
            )
            print(f"   {lab:>5}  {cells}")
    print("  値 = 報酬が「遊脚」と判定したステップのうち、実際に足が浮いていた割合。")
    print("  ★ エピソード時刻が進むほど δ≠0 群で 0.5 に近づく → 位相ズレの実証 (仮説 a)。")
    print("    ズレている間、報酬は接地脚の高さを要求する。それを満たす唯一の手段が跳躍。")

    print("\n--- (4) 遊脚側の関節余裕とトルク飽和 ---")
    print(f"{'cmd(vx,vy)':>13} {'foot':>5} " + "".join(f"{j.replace('_',''):>12}" for j in _LEG_JOINTS))
    for r in results:
        for f, name in enumerate(("L", "R")):
            m = "".join(f"{v:11.3f} " for v in r["margin"][f])
            print(f"{r['cmd'][0]:6.2f},{r['cmd'][1]:5.2f} {name:>5} {m}   ← 可動限界までの最小余裕 [rad]")
            t = "".join(f"{v:11.2f} " for v in r["torque_max"][f])
            print(f"{'':>13} {'':>5} {t}   ← トルク飽和率の最大 (1.0 = 上限)")
    print("  ★ 余裕が 0.2rad 以上あり飽和率 < 0.8 なら **報酬バランスの問題** → 作り替えで伸びる。")
    print("    どれかが張り付いていたら、その関節が足上げの物理的な律速。")

    print("\n--- (5) 体の上下動 (跳躍の有無) ---")
    print(f"{'cmd(vx,vy)':>13} {'base z':>9} {'vz rms':>9}")
    for r in results:
        print(f"{r['cmd'][0]:6.2f},{r['cmd'][1]:5.2f} {r['base_z']:8.3f}m {r['base_vz_rms']:8.3f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
