# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a hierarchical dribble agent.

Loads two checkpoints:

- ``--checkpoint``        : the trained high-level (dribble) policy that outputs
                            a 3-D walking command (vx, vy, omega_z).
- ``--frozen_checkpoint`` : the frozen low-level walking policy whose
                            velocity_commands slot is overwritten by the
                            high-level action.

Both are wired through the same ``HierarchicalVecEnvWrapper`` used in training.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a hierarchical (dribble) RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
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
    default=1.0,
    help="Clipping range for the high-level action (walking command).",
)
parser.add_argument(
    "--low_level_obs_group",
    type=str,
    default="low_level",
    help="Env observation group fed to the frozen low-level policy.",
)
parser.add_argument(
    "--low_level_cmd_term_name",
    type=str,
    default="velocity_commands",
    help="Observation term whose slice gets overwritten by the high-level action.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401

# Reuse the wrapper + frozen-policy builder so the play inference path is
# byte-for-byte identical to training. Imported from a side-effect-free helper
# module (not train_dribble.py, which would re-run its top-level argparse).
from dribble_helpers import HierarchicalVecEnvWrapper, _build_frozen_policy


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play a trained hierarchical dribble agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    inner_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Frozen low-level (walking) policy.
    print(f"[INFO] Loading frozen low-level checkpoint: {args_cli.frozen_checkpoint}")
    frozen_policy = _build_frozen_policy(
        inner_env,
        agent_cfg,
        args_cli.frozen_checkpoint,
        agent_cfg.device,
        args_cli.low_level_obs_group,
    )

    # Hierarchical wrapper (presents num_actions=3 to the runner).
    hier_env = HierarchicalVecEnvWrapper(
        inner_env,
        frozen_policy,
        low_level_obs_group=args_cli.low_level_obs_group,
        low_level_cmd_term_name=args_cli.low_level_cmd_term_name,
        action_clip=args_cli.high_action_clip,
        high_action_dim=3,
    )

    # High-level (dribble) policy runner — load from --checkpoint.
    print(f"[INFO] Loading high-level (dribble) checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(hier_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"play_dribble.py only supports OnPolicyRunner; got '{agent_cfg.class_name}'.")
    runner.load(resume_path)

    # Safeguard: re-load state_dict directly (mirrors play.py).
    raw_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    ckpt_msd = raw_ckpt.get("model_state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else None
    if isinstance(ckpt_msd, dict):
        policy_to_load = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        target_device = next(policy_to_load.parameters()).device
        ckpt_on_device = {
            k: (v.to(target_device) if isinstance(v, torch.Tensor) else v) for k, v in ckpt_msd.items()
        }
        policy_to_load.load_state_dict(ckpt_on_device, strict=False)

    policy = runner.get_inference_policy(device=inner_env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = inner_env.unwrapped.step_dt

    obs = hier_env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = hier_env.step(actions)
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    hier_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
