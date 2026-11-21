# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""首安定化方策のブラー (露光 10 ms あたりの光軸の回転量 [deg]) を状況別に集計する (学習 env そのまま、headless)。

例::

    _labpython2 eval_neck.py --headless --checkpoint /abs/path/model_299.pt            # 学習した方策
    _labpython2 eval_neck.py --headless --mode ik                                      # 解析 IK のみ (残差 0)
    _labpython2 eval_neck.py --headless --mode none                                    # 安定化なし

実機ライブラリ (booster_k1_locomotion/src/neck_stabilizer.hpp) の一致テスト用の記録 (ノイズ・遅延なしで 1 env 分)::

    _labpython2 eval_neck.py --headless --num_envs 8 --steps 1300 --checkpoint /abs/path/model_1499.pt --trace out.csv \
        env.commands.neck.sensor_delay_s=0.0 env.commands.neck.rpy_noise=0.0 env.commands.neck.gyro_noise=0.0 \
        env.commands.neck.joint_pos_noise=0.0 env.commands.neck.joint_vel_noise=0.0
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Neck stabilizer evaluation.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--task", type=str, default="Isaac-Neck-K1-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=1500, help="評価 step 数 (50 Hz)")
parser.add_argument("--mode", choices=("policy", "ik", "none"), default="policy")
parser.add_argument("--no_push", action="store_true", help="押し外乱を無効化")
parser.add_argument("--trace", type=str, default=None, help="安定化器の入出力を CSV に記録する (docstring 参照)")
parser.add_argument("--trace_rows", type=int, default=250)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_k1_locomotion.tasks  # noqa: F401, E402
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi  # noqa: E402
from isaaclab_k1_locomotion.tasks.manager_based.locomotion.mdp.events import compute_cmd_phase_freq  # noqa: E402
from isaaclab_k1_locomotion.tasks.manager_based.locomotion.mdp.neck import NECK_DIM  # noqa: E402

TRANSIENT_S = 1.0  # 歩行コマンドが変わってからこの時間は「過渡」として分けて数える


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.no_push:
        env_cfg.events.push_robot = None
    if args_cli.mode == "none":
        env_cfg.commands.neck.stabilize = False

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if args_cli.mode == "policy":
        runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.alg.act_inference
    raw_env = env.unwrapped
    neck = raw_env.command_manager.get_term("neck")
    dt, n, dev = raw_env.step_dt, env.num_envs, raw_env.device

    walk_names = ("stand", "walk<1.0", "walk>=1.0")
    mode_names = ("const", "sweep", "gaze")
    stats = torch.zeros(2, 3, 3, 3, device=dev)  # [過渡?, 歩行, q_cmd モード, (blur の和, 件数, 0.5 deg/10ms 超の件数)]
    point_sum = 0.0
    n_done = n_fall = 0
    prev_cmd = raw_env.command_manager.get_command("base_velocity").clone()
    since_change = torch.zeros(n, device=dev)
    obs = env.get_observations()
    trace = []
    with torch.inference_mode():
        for step in range(args_cli.steps):
            actions = policy(obs)
            if args_cli.mode != "policy":
                actions[:, -NECK_DIM:] = 0.0
            if args_cli.trace:
                # 入力: [リセット直後か, rpy, 角速度, 首関節角, 首関節角速度, q_cmd, 歩行周波数] (歩行位相・コマンドは観測の末尾)
                d = neck.robot.data
                row = [
                    (raw_env.episode_length_buf == 0).float().unsqueeze(1),
                    wrap_to_pi(torch.stack(euler_xyz_from_quat(d.root_quat_w), dim=1)),
                    d.root_ang_vel_b, d.joint_pos[:, neck.joint_ids], d.joint_vel[:, neck.joint_ids], neck.q_cmd,
                    compute_cmd_phase_freq(raw_env, **neck.cfg.phase_freq_params).unsqueeze(1),
                    obs["neck_policy"], actions[:, -NECK_DIM:],
                ]
            obs, _, dones, _ = env.step(actions)
            if args_cli.trace:  # この step で実際に送られた首の関節目標
                trace.append(torch.cat(row + [raw_env.action_manager.get_term("neck").processed_actions], dim=1).cpu())
            fall = raw_env.termination_manager.get_term("base_height") | raw_env.termination_manager.get_term("base_contact")
            n_done += int(dones.sum())
            n_fall += int(fall.sum())

            cmd = raw_env.command_manager.get_command("base_velocity")
            since_change = torch.where((cmd - prev_cmd).norm(dim=1) > 0.1, 0.0, since_change + dt)
            prev_cmd = cmd.clone()
            lin = cmd[:, :2].norm(dim=1)
            walk = torch.where(cmd.norm(dim=1) < 0.05, 0, torch.where(lin < 1.0, 1, 2))
            transient = (since_change < TRANSIENT_S).long()
            # blur_speed は 1 step 前の値 (分類は 1 step ずれるが集計には影響しない)。リセット直後・整定待ちは除く
            counted = (raw_env.episode_length_buf > 1) & (neck._settle < 0.0)
            blur = torch.rad2deg(neck.blur_speed) * 0.01
            idx = (transient * 9 + walk * 3 + neck.mode)[counted]
            vals = torch.stack([blur, torch.ones_like(blur), (blur > 0.5).float()], dim=1)[counted]
            stats.view(18, 3).index_add_(0, idx, vals)
            point_sum += float(torch.rad2deg(neck.point_err).mean())

    def line(name, s):
        s = s.reshape(-1, 3).sum(dim=0)
        cnt = max(float(s[1]), 1.0)
        print(f"  {name:<28} blur {float(s[0]) / cnt:6.3f} deg/10ms   >0.5: {100.0 * float(s[2]) / cnt:5.1f}%   (n={int(s[1])})")

    print(f"\n=== mode={args_cli.mode} push={'off' if args_cli.no_push else 'on'} envs={n} steps={args_cli.steps} ===")
    line("all", stats)
    line("steady", stats[0])
    line("transient (cmd change <1s)", stats[1])
    for w, wn in enumerate(walk_names):
        line(f"steady / {wn}", stats[0, w])
    for m, mn in enumerate(mode_names):
        line(f"steady / q_cmd {mn}", stats[0, :, m])
    for w, wn in enumerate(walk_names):
        line(f"steady / {wn} / const", stats[0, w, 0])
    print(f"  point_err mean: {point_sum / args_cli.steps:.2f} deg")
    print(f"  episodes ended: {n_done}, falls: {n_fall}, falls per robot-minute: {n_fall / (n * args_cli.steps * dt / 60.0):.3f}")
    if args_cli.trace:
        # リセット直後 (平滑化の履歴が作り直される) から始まる区間を 1 env 分書き出す
        data = torch.stack(trace)  # (steps, envs, 列)
        # (最初の step は除く: 起動直後はコマンド項がまだ 1 度も更新されていない)
        steps, envs = torch.nonzero(data[1 : -args_cli.trace_rows, :, 0] > 0.5, as_tuple=True)
        assert len(steps) > 0, "リセットが起きなかった: --steps を増やすこと"
        out = data[steps[0] + 1 : steps[0] + 1 + args_cli.trace_rows, envs[0]]
        with open(args_cli.trace, "w") as f:
            f.write(f"# limits_lo {' '.join(map(str, neck.limits[:, 0].tolist()))} limits_hi {' '.join(map(str, neck.limits[:, 1].tolist()))}\n")
            f.write("# fresh, rpy(3), gyro(3), q(2), qd(2), q_cmd(2), gait_freq, neck_obs(130), action(2), target(2)\n")
            for r in out.tolist():
                f.write(",".join(f"{v:.7g}" for v in r) + "\n")
        print(f"  trace: {args_cli.trace} ({out.shape[0]} rows × {out.shape[1]} cols, resets: {int(out[:, 0].sum())})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
