# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ドリブル Stage1 の環境の系 (K1FlatEnvCfg 系 / K1FlatStancePlaneCfg 系) を、推論だけで比べる。

59 次元に広げた歩行ポリシーを Stage1 の環境で ``--steps`` ステップ動かし (学習の更新はしない)、
次を出す:

* 転倒率: 期間中に base_height / base_contact で終了した env の割合
* 速度の追従誤差: |v_xy(body yaw 座標) - cmd_xy| と |ω_z - cmd_ωz| の平均 (各 env のリセット後 50 ステップは除く)
* 足の滞空時間: 着地した瞬間の滞空時間 (last_air_time) の平均、両足が浮いているステップの割合 (fly%)
  (いずれもリセット後 50 ステップは除く)

``--variant stance_plane`` は、Stage1 の cfg に K1FlatStancePlaneCfg 系で変わる環境側の設定
(地形 PLANE_HEAVY・摩擦 0.6〜1.0) を当てる。報酬の違い (stance_ratio / no_fly など) は推論には
効かないので当てない。速度コマンドは両 variant とも同じ固定分布 (カリキュラム・extreme 無効) にそろえる。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--variant", choices=["flat", "stance_plane"], required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default=None, help="結果の JSON を書き出すファイル")
parser.add_argument(
    "--training_like",
    action="store_true",
    help="学習時の指標にそろえる: 行動ノイズ (std) を入れてサンプリング、エピソード長を最初にばらつかせる、"
    "速度コマンドを Stage1 の最後と同じ分布 (x ±2.0 / y ±0.9 / yaw ±1.0、extreme 0.35) にする。"
    "終わったエピソードのうち転倒で終わった割合 (Episode_Termination/base_height 相当) も出す",
)
parser.add_argument("--lin_vel_x", type=float, default=1.5)
parser.add_argument("--lin_vel_y", type=float, default=0.9)
parser.add_argument("--ang_vel_z", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402
import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import isaaclab_k1_locomotion.tasks  # noqa: E402, F401
from isaaclab_k1_locomotion.tasks.manager_based.dribble_target.flat_env_cfg import (  # noqa: E402
    PLANE_HEAVY_TERRAIN_CFG,
)

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import stage_transition_utils  # noqa: E402

TASK = "Isaac-Velocity-Flat-K1-Dribble-Target-WalkPhase-v0"


def main():
    env_cfg = parse_env_cfg(TASK, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if args_cli.variant == "stance_plane":
        env_cfg.scene.terrain.terrain_generator = PLANE_HEAVY_TERRAIN_CFG
        env_cfg.events.physics_material.params["static_friction_range"] = (0.6, 1.0)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.0)
    # 速度コマンドは両 variant で同じ固定分布にする
    env_cfg.curriculum.lin_vel_command = None
    env_cfg.curriculum.extreme_commands = None
    env_cfg.curriculum.command_resampling_time_range = None
    r = env_cfg.commands.base_velocity.ranges
    if args_cli.training_like:
        r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-2.0, 2.0), (-0.9, 0.9), (-1.0, 1.0)
        env_cfg.commands.base_velocity.extreme_prob = 0.35
        env_cfg.commands.base_velocity.extreme_frac = 0.7
        env_cfg.commands.base_velocity.resampling_time_range = (0.8, 8.0)
    else:
        r.lin_vel_x = (-args_cli.lin_vel_x, args_cli.lin_vel_x)
        r.lin_vel_y = (-args_cli.lin_vel_y, args_cli.lin_vel_y)
        r.ang_vel_z = (-args_cli.ang_vel_z, args_cli.ang_vel_z)
        env_cfg.commands.base_velocity.resampling_time_range = (4.0, 8.0)

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(TASK, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(
        wrapped, stage_transition_utils.filter_unsupported_alg_args(agent_cfg.to_dict()), log_dir=None, device="cuda:0"
    )
    runner.load(args_cli.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device="cuda:0")
    if args_cli.training_like:
        runner.alg.policy.eval()
        policy = runner.alg.policy.act  # 学習時と同じく行動ノイズ入りでサンプリング

    u = env.unwrapped
    robot = u.scene["robot"]
    contact = u.scene.sensors["contact_forces"]
    feet_ids, _ = contact.find_bodies(".*_foot_link")
    n = u.num_envs
    fell = torch.zeros(n, dtype=torch.bool, device=u.device)
    since_reset = torch.zeros(n, device=u.device)
    lin_err, ang_err, err_count = 0.0, 0.0, 0
    air_sum, air_count, fly_steps, total_steps = 0.0, 0, 0, 0

    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    if args_cli.training_like:  # rsl_rl の init_at_random_ep_len と同じ
        u.episode_length_buf = torch.randint_like(u.episode_length_buf, high=int(u.max_episode_length))
    ended, ended_fall = 0, 0
    obs = wrapped.get_observations()
    with torch.no_grad():  # inference_mode だと env.reset の in-place 操作でエラーになる
        for _ in range(args_cli.steps):
            obs, _, dones, _ = wrapped.step(policy(obs))
            term = u.termination_manager
            failed = (term.get_term("base_height") | term.get_term("base_contact")) & dones.bool()
            fell |= failed
            ended += int(dones.sum().item())
            ended_fall += int((term.get_term("base_height") & dones.bool()).sum().item())
            since_reset = torch.where(dones.bool(), torch.zeros_like(since_reset), since_reset + 1)

            cmd = u.command_manager.get_command("base_velocity")
            v_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)[:, :2]
            settled = since_reset > 50
            if settled.any():
                lin_err += torch.norm(v_b - cmd[:, :2], dim=1)[settled].sum().item()
                ang_err += (robot.data.root_ang_vel_w[:, 2] - cmd[:, 2]).abs()[settled].sum().item()
                err_count += int(settled.sum().item())

            first_contact = contact.compute_first_contact(u.step_dt)[:, feet_ids] & settled.unsqueeze(1)
            last_air = contact.data.last_air_time[:, feet_ids]
            air_sum += last_air[first_contact].sum().item()
            air_count += int(first_contact.sum().item())
            in_contact = contact.data.current_contact_time[:, feet_ids] > 0
            fly_steps += int(((~in_contact).all(dim=1) & settled).sum().item())
            total_steps += int(settled.sum().item())

    result = {
        "variant": args_cli.variant,
        "num_envs": n,
        "steps": args_cli.steps,
        "fall_rate": fell.float().mean().item(),
        "lin_vel_err_mean": lin_err / max(err_count, 1),
        "ang_vel_err_mean": ang_err / max(err_count, 1),
        "feet_air_time_mean": air_sum / max(air_count, 1),
        "fly_pct": 100.0 * fly_steps / max(total_steps, 1),
        "training_like": args_cli.training_like,
        "episodes_ended": ended,
        "ended_by_fall_fraction": ended_fall / max(ended, 1),
        "command": {"lin_vel_x": args_cli.lin_vel_x, "lin_vel_y": args_cli.lin_vel_y, "ang_vel_z": args_cli.ang_vel_z},
    }
    print("[RESULT] " + json.dumps(result), flush=True)
    if args_cli.output:
        with open(args_cli.output, "w") as f:
            json.dump(result, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
