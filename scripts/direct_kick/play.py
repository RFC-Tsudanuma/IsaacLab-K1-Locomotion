"""Evaluate a DirectKick checkpoint; the environment receives 12 action outputs."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--steps', type=int, default=2000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--export', default=None, help='Also export the 13-output TorchScript policy here.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import torch
import gymnasium as gym
import isaaclab_k1_locomotion.tasks  # noqa: F401
from isaaclab_k1_locomotion.tasks.direct.direct_kick.env_cfg import DirectKickEnvCfg
from isaaclab_k1_locomotion.direct_kick.runner import DirectKickRunner


def main():
    cfg = DirectKickEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.enable_csv_logging = False
    env = gym.make('Isaac-K1-DirectKick-v0', cfg=cfg)
    try:
        runner = DirectKickRunner(env, env.unwrapped.task_cfg, Path(args.checkpoint).parent / 'evaluation', args.device)
        runner.load(args.checkpoint, resume=False)
        if args.export:
            runner.export(args.export)
        obs, _ = env.reset()
        with torch.no_grad():
            for _ in range(args.steps):
                if not app.is_running():
                    break
                output = runner.model.actor(obs['policy'])
                obs, _, _, _, _ = env.step(output[:, :12])
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
