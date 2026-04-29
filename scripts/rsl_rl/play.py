# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Todo argparseとかで引数として操作モードを選べるように
# Todo 角速度を変えてもその場回転しない問題を解決する

import argparse
import sys
import os
import math

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import threading
import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks
import isaaclab_k1_soccer.tasks


class VelocityController:
    """別スレッドでキー入力を受け取り、速度・heading コマンドを管理するクラス"""

    VEL_STEP     = 0.1          # 速度の1ステップ量
    HEADING_STEP = math.pi / 12 # heading の1ステップ量 (15°)
    VEL_LIMIT    = 2.0
    HEADING_LOCK = True         # True: heading固定モード / False: heading制御モード

    def __init__(self):
        self.x_vel   = 0.0
        self.y_vel   = 0.0
        self.ang_vel = 0.0
        self.heading = 0.0      # -π ~ π (ラジアン)
        self.use_heading = False # Falseのとき heading は command_manager に渡さない
        self._lock   = threading.Lock()
        self._running = True

    def get_commands(self):
        with self._lock:
            return self.x_vel, self.y_vel, self.ang_vel, self.heading, self.use_heading

    def stop(self):
        self._running = False

    def _clamp_vel(self, v):
        return max(-self.VEL_LIMIT, min(self.VEL_LIMIT, v))

    def _wrap_angle(self, a):
        """角度を -π ~ π に正規化"""
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _print_status(self):
        h_deg = math.degrees(self.heading)
        mode  = "HEADING" if self.use_heading else "ANG_VEL"
        print(
            f"[CMD] x={self.x_vel:+.2f}  y={self.y_vel:+.2f}  "
            f"ang={self.ang_vel:+.2f}  heading={h_deg:+.1f}°  mode={mode}"
        )

    def run(self):
        print("\n" + "="*56)
        print("  キーボードコントローラー起動")
        print("="*56)
        print("  W / S   : 前後速度 (x_vel)")
        print("  A / D   : 左右速度 (y_vel)")
        print("  Q / E   : 角速度   (ang_vel)  ← heading無効時に使用")
        print("  Z / C   : heading を左/右回転  (15°ずつ)")
        print("  H       : heading制御モード ON/OFF 切替")
        print("  R       : 全パラメータをリセット")
        print("="*56 + "\n")

        try:
            import tty, termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while self._running:
                    ch = sys.stdin.read(1).lower()
                    with self._lock:
                        changed = True
                        if   ch == 'w': self.x_vel   = self._clamp_vel(self.x_vel   + self.VEL_STEP)
                        elif ch == 's': self.x_vel   = self._clamp_vel(self.x_vel   - self.VEL_STEP)
                        elif ch == 'd': self.y_vel   = self._clamp_vel(self.y_vel   + self.VEL_STEP)
                        elif ch == 'a': self.y_vel   = self._clamp_vel(self.y_vel   - self.VEL_STEP)
                        elif ch == 'e': self.ang_vel = self._clamp_vel(self.ang_vel + self.VEL_STEP)
                        elif ch == 'q': self.ang_vel = self._clamp_vel(self.ang_vel - self.VEL_STEP)
                        elif ch == 'z': self.heading = self._wrap_angle(self.heading + self.HEADING_STEP)
                        elif ch == 'c': self.heading = self._wrap_angle(self.heading - self.HEADING_STEP)
                        elif ch == 'h': self.use_heading = not self.use_heading
                        elif ch == 'r':
                            self.x_vel = self.y_vel = self.ang_vel = 0.0
                            self.heading = 0.0
                        else:
                            changed = False
                        if changed:
                            self._print_status()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

        except ImportError:
            import msvcrt
            while self._running:
                if msvcrt.kbhit():
                    ch = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                    with self._lock:
                        changed = True
                        if   ch == 'w': self.x_vel   = self._clamp_vel(self.x_vel   + self.VEL_STEP)
                        elif ch == 's': self.x_vel   = self._clamp_vel(self.x_vel   - self.VEL_STEP)
                        elif ch == 'd': self.y_vel   = self._clamp_vel(self.y_vel   + self.VEL_STEP)
                        elif ch == 'a': self.y_vel   = self._clamp_vel(self.y_vel   - self.VEL_STEP)
                        elif ch == 'e': self.ang_vel = self._clamp_vel(self.ang_vel + self.VEL_STEP)
                        elif ch == 'q': self.ang_vel = self._clamp_vel(self.ang_vel - self.VEL_STEP)
                        elif ch == 'z': self.heading = self._wrap_angle(self.heading + self.HEADING_STEP)
                        elif ch == 'c': self.heading = self._wrap_angle(self.heading - self.HEADING_STEP)
                        elif ch == 'h': self.use_heading = not self.use_heading
                        elif ch == 'r':
                            self.x_vel = self.y_vel = self.ang_vel = 0.0
                            self.heading = 0.0
                        else:
                            changed = False
                        if changed:
                            self._print_status()
                time.sleep(0.02)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    controller = VelocityController()
    ctrl_thread = threading.Thread(target=controller.run, daemon=True)
    ctrl_thread.start()

    while simulation_app.is_running():
        start_time = time.time()

        x_vel, y_vel, ang_vel, heading, use_heading = controller.get_commands()

        def apply_commands():
            try:
                vel_term = env.unwrapped.command_manager.get_term("base_velocity")
                vel_term.command[:, 0] = x_vel
                vel_term.command[:, 1] = y_vel
                if use_heading:
                    vel_term.command[:, 2] = 0.0
                    vel_term.command[:, 3] = heading
                else:
                    vel_term.command[:, 2] = ang_vel
                # resampleで上書きされないよう時間を伸ばす
                vel_term.cfg.resampling_time_range = (1e9, 1e9)
            except Exception:
                pass

        apply_commands()  # step前

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)  # ← ここでresampleが走る
            policy_nn.reset(dones)

        apply_commands()  # step後に再書き込みで確実に上書き

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    controller.stop()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()