# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""足上げを **「許容できる地面の凹凸の高さ」** に換算する診断スクリプト (Phase 0 追加)。

なぜ要るか (2026-08-17):
    :file:`eval_gk_fall_forensics.py` で「シム平地では足上げ不足は転倒の原因ではない」
    (足裏クリアランスの先行時間 0.000s、逸脱率 12〜16% で全信号中の最下位) と分かった。
    しかし **「ある程度足を上げれば転倒率が下がる」という主張はこれでは反証できない**。
    平地には引っかかる対象が無いので、足上げの価値が原理的に現れないため。

    足上げの価値は「平地での転倒原因」ではなく **「どれだけの凹凸を踏み越せるか」** で、
    それは平地の歩容からでも測れる: 遊脚が地面すれすれを **水平にどれだけ移動したか**
    を積算すればよい。

測るもの — 露出率 f(h):
    f(h) = (遊脚の水平移動のうち、足裏クリアランスが h 未満だった距離) / (遊脚の総水平移動)

    高さ h の突起が遊脚の通り道に一様にあるとき、**その突起に足が当たる確率**そのもの。
    突起の密度を仮定しなくてもポリシー間の比較ができる (比が f(h) の比になる) のが利点。
    離陸/着地の前後は必ずクリアランス 0 を通るので f(0+) は 0 にならない。重要なのは
    「**すれすれのまま水平に流れる距離**」で、素早く上げて素早く下ろす歩容ほど f は小さい。

    1 歩あたりの遭遇確率 P(trip) も出す (1 歩の通り道に高さ h の突起がちょうど 1 個ある
    と仮定した場合)。**つまずき ≠ 転倒** なので、これは転倒率の上限側の目安。

比較の注意:
    跳躍 (両足同時浮き) 中は接触ベースの遊脚判定で両足とも「遊脚」になり、クリアランスが
    高く出るので f(h) は改善する。物理的には正しい (跳べば越えられる) が、跳躍自体に別の
    転倒リスクがあるので、跳躍率も併記する。

使い方 (コンテナ内・リポジトリ直下):
    /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/eval_gk_trip_margin.py \\
        --checkpoint logs/rsl_rl/k1_gk_direct_stage1/2026-07-28_17-13-15/model_14999.pt \\
        --fixed_cmd 0,1.3 --num_envs 64 --steps 6000 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Convert foot lift into tolerable ground-bump height.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument(
    "--task", type=str, default="Isaac-GoalkeeperDirect-Stage1-K1-Play-v0",
    help="Task to roll out (Stage1 Play by default so the command can be injected).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--steps", type=int, default=6000, help="Control steps to sample.")
parser.add_argument(
    "--override_json", type=str, default=None, help="JSON file with dot-path overrides."
)
parser.add_argument(
    "--fixed_cmd", type=str, default="0,1.3",
    help="'vx,vy' held constant. Pass 'none' to let the task drive (Stage2).",
)
parser.add_argument(
    "--flip_s", type=float, default=3.0, help="Flip the command sign every this many seconds [s]."
)
parser.add_argument("--settle_s", type=float, default=1.0, help="Skip this long after each reset [s].")
parser.add_argument("--contact_th", type=float, default=1.0, help="Contact force threshold [N].")
parser.add_argument(
    "--heights", type=str, default="0,2,5,8,10,15,20,25,30",
    help="Obstacle heights [mm] at which to report the exposure/trip probability.",
)
parser.add_argument(
    "--foot_box", type=float, nargs=4, default=[0.1195, -0.0659, 0.040, -0.0382],
    metavar=("TOE_X", "HEEL_X", "HALF_Y", "SOLE_Z"),
    help="Sole corner offsets in the foot-link frame [m] (same default as eval_gk_play_gait.py).",
)
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
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401

# 絶対 z のヒストグラム (1mm ビン)。地面基準は run 全体を見ないと決まらないので、
# 絶対値で貯めて最後にビンをずらす。
_HIST_LO, _HIST_HI, _HIST_BINS = -0.06, 0.20, 260
_BIN_M = (_HIST_HI - _HIST_LO) / _HIST_BINS      # 1mm


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed

    if args_cli.override_json is not None:
        from config_overrides import apply_overrides_from_file

        apply_overrides_from_file(args_cli.override_json, env_cfg=env_cfg, agent_cfg=agent_cfg)

    fixed_cmd = None
    if args_cli.fixed_cmd.strip().lower() not in ("none", ""):
        vx_s, vy_s = args_cli.fixed_cmd.split(",")
        fixed_cmd = (float(vx_s), float(vy_s))
        vc = env_cfg.commands.base_velocity
        vc.heading_command = False
        vc.rel_standing_envs = 0.0
        vc.resampling_time_range = (1.0e9, 1.0e9)

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

    toe_x, heel_x, half_y, sole_z = [float(v) for v in args_cli.foot_box]
    corners = torch.tensor(
        [[toe_x, half_y, sole_z], [toe_x, -half_y, sole_z],
         [heel_x, half_y, sole_z], [heel_x, -half_y, sole_z]],
        device=device,
    )

    def sole_min_z() -> torch.Tensor:
        out = []
        for bid in foot_body_ids:
            pos = robot.data.body_pos_w[:, bid, :]
            quat = robot.data.body_quat_w[:, bid, :]
            q = quat.unsqueeze(1).expand(n, 4, 4).reshape(-1, 4)
            c = corners.unsqueeze(0).expand(n, 4, 3).reshape(-1, 3)
            world = quat_apply(q, c).reshape(n, 4, 3) + pos.unsqueeze(1)
            out.append(world[:, :, 2].min(dim=1).values)
        return torch.stack(out, dim=1)

    settle_steps = max(1, int(args_cli.settle_s / dt))
    flip_steps = max(1, int(args_cli.flip_s / dt))

    # 遊脚の水平移動距離を、そのときの足裏絶対 z のビンに積む
    ds_hist = torch.zeros(2, _HIST_BINS, device=device)
    # 接地サンプルの絶対 z ヒストグラム (地面 p10 の推定用)
    gnd_hist = torch.zeros(2, _HIST_BINS, device=device)

    prev_foot_xy = torch.zeros(n, 2, 2, device=device)   # (env, foot, xy)
    have_prev = torch.zeros(n, dtype=torch.bool, device=device)
    swing_count = torch.zeros(2, device=device)
    prev_air = torch.zeros(n, 2, dtype=torch.bool, device=device)
    flight_steps = torch.zeros((), device=device)
    steady_steps = torch.zeros((), device=device)
    body_dist = torch.zeros((), device=device)

    obs = inner_env.get_observations()
    sign = 1.0
    for step in range(int(args_cli.steps)):
        if fixed_cmd is not None and step % flip_steps == 0:
            sign = -sign
        with torch.inference_mode():
            if fixed_cmd is not None:
                cmd_term.vel_command_b[:, 0] = fixed_cmd[0] * sign
                cmd_term.vel_command_b[:, 1] = fixed_cmd[1] * sign
                cmd_term.vel_command_b[:, 2] = 0.0
            action = policy(obs)
            obs, _, dones, _ = inner_env.step(action)
            if fixed_cmd is not None:
                cmd_term.vel_command_b[:, 0] = fixed_cmd[0] * sign
                cmd_term.vel_command_b[:, 1] = fixed_cmd[1] * sign
                cmd_term.vel_command_b[:, 2] = 0.0

        done_mask = (
            dones.to(device=device, dtype=torch.bool).flatten()
            if dones is not None
            else torch.zeros(n, dtype=torch.bool, device=device)
        )
        steady = (raw_env.episode_length_buf > settle_steps) & ~done_mask

        forces = contact.data.net_forces_w[:, foot_sensor_ids, :]
        on_ground = forces.norm(dim=-1) > float(args_cli.contact_th)
        air = ~on_ground
        sole_z_w = sole_min_z()                                        # (n, 2)
        foot_xy = torch.stack(
            [robot.data.body_pos_w[:, bid, :2] for bid in foot_body_ids], dim=1
        )                                                              # (n, 2, 2)

        # 水平移動量。リセット直後や過渡は捨てる。
        ds = (foot_xy - prev_foot_xy).norm(dim=-1)                     # (n, 2)
        valid = steady & have_prev & ~done_mask
        if bool(valid.any()):
            steady_steps += valid.sum()
            flight_steps += (air.all(dim=1) & valid).sum()
            body_dist += robot.data.root_lin_vel_w[valid, :2].norm(dim=-1).sum() * dt

            zi = ((sole_z_w - _HIST_LO) / (_HIST_HI - _HIST_LO) * _HIST_BINS).long()
            zi = zi.clamp(0, _HIST_BINS - 1)
            for f in range(2):
                sel = valid & air[:, f]
                if bool(sel.any()):
                    ds_hist[f] += torch.bincount(
                        zi[sel, f], weights=ds[sel, f], minlength=_HIST_BINS
                    ).float()
                gsel = valid & on_ground[:, f]
                if bool(gsel.any()):
                    gnd_hist[f] += torch.bincount(zi[gsel, f], minlength=_HIST_BINS).float()
            # 遊脚区間の開始をカウント (1 歩の数)
            started = valid.unsqueeze(1) & air & ~prev_air
            swing_count += started.sum(dim=0).float()

        prev_foot_xy = foot_xy.clone()
        have_prev = steady & ~done_mask
        prev_air = air

    inner_env.close()

    # --- 地面基準 (接地サンプルの p10) をビン番号で求める ---
    gnd_bin = []
    for f in range(2):
        total = float(gnd_hist[f].sum().item())
        if total <= 0:
            gnd_bin.append(0)
            continue
        cum = torch.cumsum(gnd_hist[f], dim=0)
        i = int(torch.searchsorted(cum, torch.tensor(0.10 * total, device=cum.device)).item())
        gnd_bin.append(min(max(i, 0), _HIST_BINS - 1))

    heights_mm = [float(v) for v in args_cli.heights.split(",")]
    ds_total = ds_hist.sum(dim=1)
    n_swing = swing_count.sum().item()

    print("\n" + "=" * 78)
    print(f"checkpoint: {resume_path}")
    print(f"task: {args_cli.task} / envs {n} / steps {args_cli.steps}")
    if fixed_cmd is not None:
        print(f"fixed_cmd: {fixed_cmd} (±{args_cli.flip_s}s ごとに反転)")
    if args_cli.override_json:
        print(f"override: {args_cli.override_json}")
    print(f"接地基準 p10 (足裏 z): 左 {_HIST_LO + (gnd_bin[0] + 0.5) * _BIN_M:.4f} m"
          f" / 右 {_HIST_LO + (gnd_bin[1] + 0.5) * _BIN_M:.4f} m")
    print("=" * 78)

    fl_frac = float((flight_steps / steady_steps.clamp(min=1)).item())
    dist = float(body_dist.item())
    print(f"\n遊脚の総水平移動: 左 {ds_total[0]:.1f} m / 右 {ds_total[1]:.1f} m"
          f"  (歩数 {int(n_swing)}、胴体移動 {dist:.1f} m)")
    print(f"跳躍 (両足同時浮き) の割合: {fl_frac:.2%}")

    print("\n--- 露出率 f(h) と 1 歩あたりのつまずき遭遇確率 ---")
    print(f"{'h [mm]':>8} {'f(h) 左':>9} {'f(h) 右':>9} {'f(h) 平均':>10}"
          f" {'P(trip)/歩':>12} {'平均何歩に1回':>14} {'何m進むと1回':>13}")
    for h_mm in heights_mm:
        fs = []
        for f in range(2):
            # クリアランス < h  ⇔  絶対 z のビン < 地面ビン + h
            cut = gnd_bin[f] + int(round(h_mm / 1000.0 / _BIN_M))
            cut = min(max(cut, 0), _HIST_BINS)
            below = float(ds_hist[f, :cut].sum().item())
            fs.append(below / max(float(ds_total[f].item()), 1e-9))
        f_mean = 0.5 * (fs[0] + fs[1])
        steps_per = (1.0 / f_mean) if f_mean > 1e-9 else float("inf")
        m_per = (dist / (n_swing * f_mean)) if (f_mean > 1e-9 and n_swing > 0) else float("inf")
        print(f"{h_mm:8.0f} {fs[0]:9.3f} {fs[1]:9.3f} {f_mean:10.3f}"
              f" {f_mean * 100:11.1f}% {steps_per:14.1f} {m_per:13.2f}")

    print("\n  f(h) = 遊脚の水平移動のうち足裏クリアランスが h 未満だった割合")
    print("       = 高さ h の突起が通り道に一様にあるとき、その突起に足が当たる確率。")
    print("  P(trip)/歩 は「1 歩の通り道に高さ h の突起がちょうど 1 個ある」と仮定した場合。")
    print("  ★ **つまずき ≠ 転倒**。踏ん張って復帰する場合もあるので転倒率の上限側の目安。")
    print("  ★ ポリシー間の比較は f(h) の比を見れば突起の密度を仮定せずに済む。")
    print("  ★ 跳躍中は両足とも遊脚と判定されクリアランスが高く出る (物理的には正しいが、")
    print("    跳躍自体の転倒リスクは別勘定)。跳躍率が高い run はその分を割り引くこと。")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
