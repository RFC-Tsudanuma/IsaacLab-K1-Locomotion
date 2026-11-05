# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""経由点追従方策の転倒を状況別に集計する (学習 env そのまま、headless)。

例::

    uv run eval_waypoint_falls.py --task Isaac-Waypoint-K1-v0 --headless --num_envs 2048 \
        --steps 3000 --checkpoint /abs/path/model_1499.pt [--no_push]
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Waypoint policy fall analysis.")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--task", type=str, default="Isaac-Waypoint-K1-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=3000, help="評価 step 数 (50 Hz)")
parser.add_argument("--no_push", action="store_true", help="押し外乱を無効化")
parser.add_argument("--expert_checkpoint", type=str, default=None, help="凍結歩行 expert を差し替える (model_*.pt)")
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


def hist(name, values, edges, unit=""):
    values = torch.as_tensor(values)
    print(f"  {name}:")
    lo = -float("inf")
    for hi in list(edges) + [float("inf")]:
        n = int(((values > lo) & (values <= hi)).sum())
        print(f"    ({lo:>6.2f}, {hi:>6.2f}]{unit}: {n:5d}  ({100.0 * n / max(len(values), 1):5.1f}%)")
        lo = hi


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

    resume_path = retrieve_file_path(args_cli.checkpoint)
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    if args_cli.expert_checkpoint:
        ckpt = torch.load(args_cli.expert_checkpoint, map_location=agent_cfg.device, weights_only=False)
        runner.alg.policy.expert.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[INFO] walk expert <- {args_cli.expert_checkpoint}")
    policy = runner.alg.act_inference
    raw_env = env.unwrapped
    wp = raw_env.command_manager.get_term("waypoints")
    robot = raw_env.scene["robot"]
    dt = raw_env.step_dt

    falls = {k: [] for k in ("t", "speed_cmd", "wz_cmd", "cmd_rate", "goal_dist", "all_passed", "speed_act", "ang_vel_xy")}
    n_done = n_fall = n_reach = 0
    ttg_sum = 0.0
    # モード別 (strict / fast) の到達・到達時間、未到達の内訳 (経由点で詰まる / ゴール段階で詰まる)
    mode_stats = {m: {"done": 0, "reach": 0, "ttg": 0.0, "stuck_wp": 0, "stuck_goal": 0, "miss_dist": 0.0} for m in ("strict", "fast")}
    prev_cmd = torch.zeros(env.num_envs, 3, device=raw_env.device)
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            actions = policy(obs)
            cmd = actions[:, 12:15]
            t_ep = (raw_env.episode_length_buf + 1).float() * dt
            goal_dist = wp.goal_dist()
            all_passed = wp.all_passed()
            speed_act = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
            ang_xy = torch.norm(robot.data.root_ang_vel_b[:, :2], dim=1)
            reached_pre, ttg_pre = wp.reached_goal.clone(), wp.time_to_goal.clone()
            strict_pre = wp.pass_radius < 0.5 * (wp.cfg.strict_radius + wp.cfg.loose_radius)
            at_goal_stage_pre = wp.all_passed().clone()
            obs, _, dones, _ = env.step(actions)
            fall = raw_env.termination_manager.get_term("base_height") | raw_env.termination_manager.get_term("base_contact")
            n_done += int(dones.sum())
            n_fall += int(fall.sum())
            d = dones.bool()
            n_reach += int((d & reached_pre).sum())
            ttg_sum += float(ttg_pre[d & reached_pre].sum())
            for name, mode_mask in (("strict", strict_pre), ("fast", ~strict_pre)):
                dm = d & mode_mask
                st = mode_stats[name]
                st["done"] += int(dm.sum())
                st["reach"] += int((dm & reached_pre).sum())
                st["ttg"] += float(ttg_pre[dm & reached_pre].sum())
                miss = dm & ~reached_pre & ~fall
                st["stuck_wp"] += int((miss & ~at_goal_stage_pre).sum())
                st["stuck_goal"] += int((miss & at_goal_stage_pre).sum())
                st["miss_dist"] += float(goal_dist[miss].sum())
            if fall.any():
                falls["t"] += t_ep[fall].tolist()
                falls["speed_cmd"] += torch.norm(cmd[fall, :2], dim=1).tolist()
                falls["wz_cmd"] += cmd[fall, 2].abs().tolist()
                falls["cmd_rate"] += torch.norm(cmd[fall] - prev_cmd[fall], dim=1).tolist()
                falls["goal_dist"] += goal_dist[fall].tolist()
                falls["all_passed"] += all_passed[fall].float().tolist()
                falls["speed_act"] += speed_act[fall].tolist()
                falls["ang_vel_xy"] += ang_xy[fall].tolist()
            prev_cmd = cmd.clone()
            prev_cmd[dones.bool()] = 0.0
            if step % 500 == 0:
                print(f"[step {step}] episodes={n_done} falls={n_fall}")

    print(f"\n=== push={'off' if args_cli.no_push else 'on'} envs={env.num_envs} steps={args_cli.steps} ===")
    print(f"  episodes ended: {n_done}, falls: {n_fall} ({100.0 * n_fall / max(n_done, 1):.2f}% / episode)")
    print(f"  reached goal: {100.0 * n_reach / max(n_done, 1):.2f}%  mean time_to_goal (reached only): {ttg_sum / max(n_reach, 1):.3f} s")
    for name, st in mode_stats.items():
        n_miss = st["stuck_wp"] + st["stuck_goal"]
        print(f"  [{name}] episodes {st['done']}: reached {100.0 * st['reach'] / max(st['done'], 1):.2f}%  "
              f"time_to_goal {st['ttg'] / max(st['reach'], 1):.3f} s  "
              f"timeout w/o reach: at waypoint {st['stuck_wp']}, at goal stage {st['stuck_goal']}, "
              f"mean goal dist {st['miss_dist'] / max(n_miss, 1):.2f} m")
    print(f"  falls per robot-minute: {n_fall / (env.num_envs * args_cli.steps * dt / 60.0):.3f}")
    if n_fall:
        hist("episode time at fall [s]", falls["t"], [0.5, 1.0, 2.0, 4.0, 8.0, 15.0])
        hist("|cmd xy| at fall [m/s]", falls["speed_cmd"], [0.05, 0.5, 1.0, 1.5, 1.79])
        hist("|cmd wz| at fall [rad/s]", falls["wz_cmd"], [0.1, 0.5, 0.9, 0.99])
        hist("cmd rate ||dc|| per step", falls["cmd_rate"], [0.05, 0.2, 0.5, 1.0, 2.0])
        hist("goal dist at fall [m]", falls["goal_dist"], [0.3, 0.6, 1.0, 2.0, 4.0])
        hist("actual speed at fall [m/s]", falls["speed_act"], [0.3, 0.8, 1.3, 1.8])
        print(f"  all waypoints passed at fall: {100.0 * sum(falls['all_passed']) / n_fall:.1f}%")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
