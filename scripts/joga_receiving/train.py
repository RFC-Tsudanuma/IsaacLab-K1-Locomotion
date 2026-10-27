"""Train the Joga-paper receiving adaptation on K1/IsaacLab."""
import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=30000)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--log_dir", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
import isaaclab_k1_locomotion.tasks  # noqa: F401
from isaaclab_k1_locomotion.joga_receiving.config import TASK_ID, load_config
from isaaclab_k1_locomotion.joga_receiving.runner import ReceivingRunner
from isaaclab_k1_locomotion.tasks.direct.joga_receiving.env_cfg import JogaReceiveEnvCfg


def main():
    settings = (torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
                if args.checkpoint else load_config())
    if args.num_envs is not None:
        settings["num_envs"] = args.num_envs
    if args.seed is not None:
        settings["seed"] = args.seed
    cfg = JogaReceiveEnvCfg()
    cfg.scene.num_envs = settings["num_envs"]
    cfg.seed = settings["seed"]
    cfg.sim.device = args.device
    cfg.task_settings = settings
    cfg.episode_length_s = settings["completion"]["episode_s"]
    directory = Path(args.log_dir or ("logs/joga_receiving/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    env = gym.make(TASK_ID, cfg=cfg)
    try:
        runner = ReceivingRunner(env, settings, directory, args.device)
        if args.checkpoint:
            runner.load(args.checkpoint)
        checkpoint = runner.train(args.max_iterations)
        runner.export(directory / "policy.pt")
        print(f"Checkpoint: {checkpoint}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
