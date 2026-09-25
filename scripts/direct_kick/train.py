"""Train the IsaacLab DirectKick port with its source PPO implementation."""
import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=8192)
parser.add_argument('--max_iterations', type=int, default=200000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--rl_device', default=None)
parser.add_argument('--checkpoint', default=None, help='Resume a checkpoint made by this port.')
parser.add_argument('--log_dir', default=None)
parser.add_argument('--no_wandb', action='store_true')
parser.add_argument('--no_csv', action='store_true', help='Disable environment-0 reward/debug CSV.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import isaaclab_k1_locomotion.tasks  # noqa: F401
from isaaclab_k1_locomotion.tasks.direct.direct_kick.env_cfg import DirectKickEnvCfg
from isaaclab_k1_locomotion.direct_kick.runner import DirectKickRunner


def main():
    cfg = DirectKickEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.log_dir = str(Path(args.log_dir or ('logs/direct_kick/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))).resolve())
    cfg.enable_csv_logging = not args.no_csv
    env = gym.make('Isaac-K1-DirectKick-v0', cfg=cfg)
    try:
        task_cfg = env.unwrapped.task_cfg
        task_cfg['runner']['use_wandb'] = not args.no_wandb
        task_cfg['basic'].update(seed=args.seed, max_iterations=args.max_iterations,
                                 sim_device=args.device, rl_device=args.rl_device or args.device,
                                 headless=args.headless)
        runner = DirectKickRunner(env, task_cfg, cfg.log_dir, args.rl_device or args.device)
        if args.checkpoint:
            runner.load(args.checkpoint)
        checkpoint = runner.train(args.max_iterations)
        runner.export(Path(cfg.log_dir) / 'policy.pt')
        print(f'Checkpoint: {checkpoint}')
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
