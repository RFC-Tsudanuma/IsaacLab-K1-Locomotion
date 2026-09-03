# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical RL training script.

A frozen low-level walking policy (trained on K1FlatEnvCfg) is loaded and held
fixed. A new high-level policy is trained on top: the high-level action is the
walking command (vx, vy, omega_z) fed into the frozen policy. The frozen policy
then outputs joint targets that get stepped into the env.

Key design point (vs. injecting into the env's command_manager):

- The env's ``command_manager`` is left alone. It is expected to generate the
  high-level command the dribble policy is asked to follow (e.g. dribble target
  direction). That command shows up in the high-level policy's observation
  through the env's normal obs terms.
- The high-level action is injected only into the observation slice that the
  frozen policy reads as its "velocity_commands". Concretely: we look up the
  index/length of the term in the env's ``observation_manager`` and overwrite
  that slice in a *copy* of the observation, before passing it to the frozen
  policy. The original obs returned to the runner is unchanged.

To make this work cleanly with a dribble env, the env cfg should expose a
dedicated observation group whose terms exactly match the frozen policy's
training-time observation structure (same term order, same dims, with a
``velocity_commands`` slot of the right size). Pass that group's name via
``--low_level_obs_group`` (default ``policy``).

Example (with the dribble env's ``low_level`` group)::

    train_dribble.py --task Isaac-Dribble-K1-v0 \
        --frozen_checkpoint logs/.../model_XXX.pt \
        --low_level_obs_group low_level \
        --headless --num_envs 4096
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Hierarchical (dribble) RL training with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--override_json",
    type=str,
    default=None,
    help="JSON file with dot-path overrides for env_cfg / agent_cfg (used by Optuna tuning).",
)
# -- hierarchical-specific arguments
parser.add_argument(
    "--frozen_checkpoint",
    type=str,
    required=True,
    help="Path to the frozen low-level (walking) policy checkpoint (model_*.pt).",
)
parser.add_argument(
    "--high_action_clip",
    type=float,
    nargs=3,
    default=[1.0, 0.5, 1.0],
    metavar=("VX", "VY", "WZ"),
    help="Per-axis clipping range for the high-level action (vx, vy, wz) — should match the frozen"
    " walking policy's training-time velocity command range. Default mirrors lin_vel_command's"
    " final stage: vx=±1.0, vy=±0.5, wz=±1.0.",
)
parser.add_argument(
    "--low_level_obs_group",
    type=str,
    default="policy",
    help="Name of the env observation group fed to the frozen low-level policy. The group's term order/dims must"
    " match what the frozen policy was trained on.",
)
parser.add_argument(
    "--low_level_cmd_term_name",
    type=str,
    default="velocity_commands",
    help="Name of the observation term within --low_level_obs_group whose slice gets overwritten by the high-level"
    " action before being passed to the frozen policy.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import shutil
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

import isaaclab_k1_locomotion.tasks  # noqa: F401

from dribble_helpers import HierarchicalVecEnvWrapper, _build_frozen_policy  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train the high-level policy on top of a frozen walking policy."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # apply Optuna-style JSON overrides last so they win over Hydra/CLI defaults
    if args_cli.override_json is not None:
        from config_overrides import apply_overrides_from_file

        apply_overrides_from_file(args_cli.override_json, env_cfg=env_cfg, agent_cfg=agent_cfg)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training is not supported when using CPU device.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning("IO descriptors only supported for manager based RL envs; skipping.")

    env_cfg.log_dir = log_dir

    # Build env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # Inner env: consumes joint-target actions output by the frozen policy.
    inner_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Frozen low-level policy.
    print(f"[INFO] Loading frozen low-level checkpoint: {args_cli.frozen_checkpoint}")
    frozen_policy = _build_frozen_policy(
        inner_env,
        agent_cfg,
        args_cli.frozen_checkpoint,
        agent_cfg.device,
        args_cli.low_level_obs_group,
    )

    # Hierarchical wrapper presents num_actions=3 (walking command) to the runner.
    hier_env = HierarchicalVecEnvWrapper(
        inner_env,
        frozen_policy,
        low_level_obs_group=args_cli.low_level_obs_group,
        low_level_cmd_term_name=args_cli.low_level_cmd_term_name,
        action_clip=args_cli.high_action_clip,
        high_action_dim=3,
    )

    # High-level runner.
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(hier_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(
            f"train_dribble.py only supports OnPolicyRunner; got '{agent_cfg.class_name}'."
        )
    runner.add_git_repo_to_log(__file__)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "dribble_meta.txt"), "w") as f:
        f.write(f"frozen_checkpoint: {os.path.abspath(args_cli.frozen_checkpoint)}\n")
        f.write(f"high_action_clip (vx, vy, wz): {tuple(args_cli.high_action_clip)}\n")
        f.write(f"low_level_obs_group: {args_cli.low_level_obs_group}\n")
        f.write(f"low_level_cmd_term_name: {args_cli.low_level_cmd_term_name}\n")

    # 学習時の frozen checkpoint をログに保存。後で play / 再学習する際に「どの歩行ポリシーで
    # 学習したか」を取り違えないようにするための保険。
    frozen_src = os.path.abspath(args_cli.frozen_checkpoint)
    frozen_dst = os.path.join(log_dir, "params", "frozen_low_level.pt")
    try:
        shutil.copy(frozen_src, frozen_dst)
        print(f"[INFO] Copied frozen checkpoint to: {frozen_dst}")
    except OSError as e:
        print(f"[WARN] Failed to copy frozen checkpoint ({frozen_src} -> {frozen_dst}): {e}")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
