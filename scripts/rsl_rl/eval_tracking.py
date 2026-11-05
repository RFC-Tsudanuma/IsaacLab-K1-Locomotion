# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""経路追従 + 姿勢到達 / 見ながら到達方策 (Isaac-Tracking-K1) の評価 (headless)。

task_mode (pose / look) 別に到達率・到達時間・見つめ違反、pivot 使用率、切替回数、転倒を集計する。
既定は学習 env のまま (乱れたリセット)。``--standing`` で立位スタート (関節 ±5°, 初速 ±0.1) にする。

例::

    uv run eval_tracking.py --headless --num_envs 2048 --steps 3000 --standing \
        --checkpoint /abs/path/model_2999.pt
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Tracking policy evaluation.")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--task", type=str, default="Isaac-Tracking-K1-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=3000, help="評価 step 数 (50 Hz)")
parser.add_argument("--standing", action="store_true", help="立位スタート (関節 ±5°, 初速 ±0.1)")
parser.add_argument("--no_push", action="store_true", help="押し外乱を無効化")
parser.add_argument("--force_pivot_s", type=float, default=0.0,
                    help="診断用: 初期 |yaw 誤差| >= 120 deg の env で、エピソード開始からこの秒数だけ mode=turn (Δψ 残差 0) を強制する")
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


class Stat:
    def __init__(self):
        self.done = self.reach = self.fall = 0
        self.ttg = self.viol_steps = self.steps = self.switches = self.turn_steps = 0.0
        self.big_yaw_done = self.big_yaw_reach = 0
        self.big_yaw_ttg = 0.0
        self.miss_wp = self.miss_goal = 0
        self.miss_dist = 0.0

    def line(self, name):
        n = max(self.done, 1)
        s = (f"  [{name}] episodes {self.done}: reached {100.0 * self.reach / n:.2f}%  "
             f"time_to_goal {self.ttg / max(self.reach, 1):.3f} s  falls {100.0 * self.fall / n:.2f}%  "
             f"switches/ep {self.switches / n:.2f}  pivot time {100.0 * self.turn_steps / max(self.steps, 1):.1f}%")
        if name == "look":
            s += f"  look-at violation time {100.0 * self.viol_steps / max(self.steps, 1):.2f}%"
        n_miss = self.miss_wp + self.miss_goal
        s += (f"\n         timeout w/o reach: at waypoint {self.miss_wp}, at goal stage {self.miss_goal}, "
              f"mean goal dist {self.miss_dist / max(n_miss, 1):.2f} m")
        if self.big_yaw_done:
            s += (f"\n         initial |yaw err| >= 120 deg ({self.big_yaw_done} ep): reached "
                  f"{100.0 * self.big_yaw_reach / self.big_yaw_done:.2f}%  time_to_goal "
                  f"{self.big_yaw_ttg / max(self.big_yaw_reach, 1):.3f} s")
        return s


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
    if args_cli.standing:
        env_cfg.events.reset_robot_joints.params["position_range"] = (-math.radians(5.0), math.radians(5.0))
        env_cfg.events.reset_base.params["velocity_range"] = {k: (-0.1, 0.1) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        env_cfg.commands.base_velocity.initial_turn_prob = 0.0

    resume_path = retrieve_file_path(args_cli.checkpoint)
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.alg.act_inference
    raw_env = env.unwrapped
    goal = raw_env.command_manager.get_term("tracking")
    low = raw_env.command_manager.get_term("base_velocity")
    tm = raw_env.termination_manager
    dt = raw_env.step_dt

    stats = {"pose": Stat(), "look": Stat()}
    # エピソード開始時の「最初の目標点への yaw 誤差」(pivot が有効な状況の抽出用)。reset 直後の step で記録
    init_yaw_err = torch.zeros(env.num_envs, device=raw_env.device)
    ep_switches = torch.zeros(env.num_envs, device=raw_env.device)
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            fresh = raw_env.episode_length_buf == 0
            if fresh.any():
                first = goal.cmd_buf[:, :2]  # base yaw 系の次の点
                init_yaw_err[fresh] = torch.atan2(first[fresh, 1], first[fresh, 0]).abs()
            actions = policy(obs)
            if args_cli.force_pivot_s > 0.0:
                forced = (init_yaw_err >= math.radians(120.0)) & (raw_env.episode_length_buf.float() * dt < args_cli.force_pivot_s)
                actions[forced, 15] = 0.0
                actions[forced, 16] = 1.0
            reached_pre, ttg_pre = goal.reached_goal.clone(), goal.time_to_goal.clone()
            look_pre = goal.is_look().clone()
            turn_pre = low.is_turn.clone()
            viol_pre = look_pre & (goal.look_at_err().abs() > goal.cfg.look_at_limit)
            passed_pre, gdist_pre = goal.all_passed().clone(), goal.goal_dist().clone()
            obs, _, dones, _ = env.step(actions)
            ep_switches += low.switched.float()
            fall = tm.get_term("base_height") | tm.get_term("base_contact")
            d = dones.bool()
            for name, mask in (("pose", ~look_pre), ("look", look_pre)):
                st = stats[name]
                st.steps += float(mask.sum())
                st.turn_steps += float((mask & turn_pre).sum())
                st.viol_steps += float((mask & viol_pre).sum())
                dm = d & mask
                st.done += int(dm.sum())
                st.reach += int((dm & reached_pre).sum())
                st.ttg += float(ttg_pre[dm & reached_pre].sum())
                st.fall += int((dm & fall).sum())
                miss = dm & ~reached_pre & ~fall
                st.miss_wp += int((miss & ~passed_pre).sum())
                st.miss_goal += int((miss & passed_pre).sum())
                st.miss_dist += float(gdist_pre[miss].sum())
                st.switches += float(ep_switches[dm].sum())
                big = dm & (init_yaw_err >= math.radians(120.0))
                st.big_yaw_done += int(big.sum())
                st.big_yaw_reach += int((big & reached_pre).sum())
                st.big_yaw_ttg += float(ttg_pre[big & reached_pre].sum())
            ep_switches[d] = 0.0
            if step % 500 == 0:
                print(f"[step {step}] episodes={stats['pose'].done + stats['look'].done}")

    total = stats["pose"].done + stats["look"].done
    reach = stats["pose"].reach + stats["look"].reach
    falls = stats["pose"].fall + stats["look"].fall
    print(f"\n=== {'standing' if args_cli.standing else 'train-reset'} push={'off' if args_cli.no_push else 'on'} "
          f"envs={env.num_envs} steps={args_cli.steps} ===")
    print(f"  episodes {total}: reached {100.0 * reach / max(total, 1):.2f}%  falls {100.0 * falls / max(total, 1):.2f}%")
    for name, st in stats.items():
        print(st.line(name))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
