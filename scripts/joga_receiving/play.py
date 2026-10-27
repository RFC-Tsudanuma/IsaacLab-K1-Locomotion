"""Evaluate a K1 receiving checkpoint using its saved task settings."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--export", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
from tensordict import TensorDict
import isaaclab_k1_locomotion.tasks  # noqa: F401
from isaaclab_k1_locomotion.joga_receiving.config import TASK_ID
from isaaclab_k1_locomotion.joga_receiving.runner import ReceivingRunner
from isaaclab_k1_locomotion.tasks.direct.joga_receiving.env_cfg import JogaReceiveEnvCfg


def main():
    settings = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
    cfg = JogaReceiveEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.task_settings = settings
    cfg.episode_length_s = settings["completion"]["episode_s"]
    env = gym.make(TASK_ID, cfg=cfg)
    try:
        directory = Path(args.checkpoint).parent / "evaluation"
        runner = ReceivingRunner(env, settings, directory, args.device)
        runner.load(args.checkpoint, resume=False)
        runner.model.eval()
        if args.export:
            runner.export(args.export)
        observations = runner.observations
        with torch.inference_mode():
            for _ in range(args.steps):
                if not app.is_running():
                    break
                actions = runner.model.act_inference(observations)
                observations, _, _, _, _ = env.step(actions)
                observations = TensorDict(observations, batch_size=[args.num_envs])
        metrics = env.unwrapped.drain_episode_metrics()
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(metrics)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
