# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record continuous walk rollouts from a frozen RSL-RL policy for AMP references.

This does *not* record video. It rolls the frozen lower-body walk policy in sim
under a fixed velocity command and logs, every control step, the robot's
**root pose + leg joint angles/velocities** into a ``.pkl`` — a continuous
trajectory the AMP discriminator can consume as an expert clip.

One env per command: the schedule below assigns each parallel env a distinct
(vx, vy, wz) so a single episode yields several clips (forward + turning at a
range of speeds). Each env is saved as its own ``walk_policy_<i>.pkl`` with the
GMR-retargeter key layout expected by ``build_amp_corpus.py``:

    {fps, root_pos (T,3), root_rot (T,4, xyzw), dof_pos (T,12), joint_names, command}

Upper-body (arm/head) joints are *not* in this URDF (legs-only). They are filled
in later, on the booster side, from ``_DEPLOY_DEFAULT_UPPER_BODY``.

.. code-block:: bash

    python scripts/rsl_rl/record_amp_rollout.py \
        --task Isaac-Velocity-Flat-K1-Play-v0 \
        --checkpoint /path/to/0524_walk.pt \
        --out_dir outputs/amp_rollout --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record walk rollouts for AMP references.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None, help="Overridden to len(SCHEDULE).")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-K1-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--out_dir", type=str, default="outputs/amp_rollout",
                    help="Directory to write walk_policy_<i>.pkl clips.")
parser.add_argument("--settle_time", type=float, default=2.0,
                    help="Seconds to run (and discard) before recording, to let the gait stabilize.")
parser.add_argument("--record_time", type=float, default=8.0, help="Seconds to record per clip.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import pickle

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401


# Per-env velocity command schedule: (vx [m/s], vy [m/s], wz [rad/s]).
# Forward + turning at a range of speeds (approach = ~1 m forward w/ turn).
SCHEDULE: list[tuple[float, float, float]] = [
    (0.3, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.5, 0.0, 0.4),
    (0.5, 0.0, -0.4),
    (0.5, 0.0, 0.8),
    (0.5, 0.0, -0.8),
    (0.8, 0.0, 0.5),
    (0.8, 0.0, -0.5),
]


def _infer_mlp_hidden(msd: dict, prefix: str) -> list[int]:
    """Derive an MLP's hidden-layer sizes from a checkpoint's Linear weights.

    ``actor``/``critic`` are ``nn.Sequential`` with Linear at even indices; each
    ``<prefix>.<i>.weight`` has shape (out, in). Hidden dims = out_features of
    every Linear except the final output layer.
    """
    layers = sorted(
        (int(k.split(".")[1]), v)
        for k, v in msd.items()
        if k.startswith(prefix + ".") and k.endswith(".weight") and k.split(".")[1].isdigit()
    )
    outs = [int(v.shape[0]) for _, v in layers]
    return outs[:-1]


def set_commands(env, cmd: torch.Tensor) -> None:
    """Force the ``base_velocity`` command to a per-env (N,3) tensor and pin it."""
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    if hasattr(cmd_term, "vel_command_b"):
        cmd_term.vel_command_b[:] = cmd
    if hasattr(cmd_term, "command"):
        try:
            cmd_term.command[:] = cmd
        except (AttributeError, TypeError):
            pass
    # Stop heading control / standing-zeroing from overwriting our command.
    if hasattr(cmd_term, "is_heading_env"):
        cmd_term.is_heading_env[:] = False
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = False
    if hasattr(cmd_term, "heading_target"):
        cmd_term.heading_target[:] = 0.0


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    n_clips = len(SCHEDULE)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = n_clips
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    # Keep the command fixed for the whole episode; no heading resampling.
    env_cfg.episode_length_s = max(env_cfg.episode_length_s,
                                   args_cli.settle_time + args_cli.record_time + 2.0)
    try:
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    except AttributeError:
        pass
    # Disable perturbations that would corrupt a clean gait clip.
    for ev in ("push_robot", "base_external_force_torque"):
        if hasattr(env_cfg.events, ev):
            setattr(env_cfg.events, ev, None)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    # Match the policy architecture to the checkpoint (0524_walk.pt was trained
    # with different hidden sizes than the current default agent cfg), else
    # ``runner.load`` raises a size-mismatch. Also enable the empirical obs
    # normalizer iff the checkpoint carries one (needed for correct inference).
    _raw = torch.load(resume_path, map_location="cpu", weights_only=False)
    _msd = _raw.get("model_state_dict", _raw) if isinstance(_raw, dict) else {}
    if isinstance(_msd, dict) and _msd:
        actor_hidden = _infer_mlp_hidden(_msd, "actor")
        critic_hidden = _infer_mlp_hidden(_msd, "critic")
        if actor_hidden:
            agent_cfg.policy.actor_hidden_dims = actor_hidden
        if critic_hidden:
            agent_cfg.policy.critic_hidden_dims = critic_hidden
        if any(k.startswith("actor_obs_normalizer") for k in _msd):
            agent_cfg.empirical_normalization = True
        print(f"[INFO] policy dims from ckpt: actor={actor_hidden} critic={critic_hidden} "
              f"empirical_norm={getattr(agent_cfg, 'empirical_normalization', None)}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # Safeguard: re-load the state_dict directly (runner.load can silently no-op).
    raw_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    ckpt_msd = raw_ckpt.get("model_state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else None
    if isinstance(ckpt_msd, dict):
        policy_to_load = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        target_device = next(policy_to_load.parameters()).device
        ckpt_on_device = {k: (v.to(target_device) if isinstance(v, torch.Tensor) else v)
                          for k, v in ckpt_msd.items()}
        policy_to_load.load_state_dict(ckpt_on_device, strict=False)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.joint_names)
    env_origins = env.unwrapped.scene.env_origins  # (N, 3)
    dt = env.unwrapped.step_dt
    fps = 1.0 / dt
    settle_steps = int(args_cli.settle_time / dt)
    record_steps = int(args_cli.record_time / dt)

    cmd = torch.tensor(SCHEDULE, dtype=torch.float32, device=device)  # (N, 3)
    print(f"[INFO] dt={dt:.4f}s fps={fps:.1f} | {n_clips} clips | "
          f"settle={settle_steps} rec={record_steps} steps | joints={joint_names}")

    obs = env.get_observations()
    if hasattr(policy_nn, "reset"):
        policy_nn.reset(torch.ones(env.num_envs, dtype=torch.bool, device=device))

    # Settle (discarded).
    for _ in range(settle_steps):
        set_commands(env, cmd)
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

    # Record.
    root_pos_log, root_rot_log, jp_log, jv_log = [], [], [], []
    done_step = [None] * n_clips
    for t in range(record_steps):
        set_commands(env, cmd)
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        rs = robot.data.root_state_w  # (N, 13)
        root_pos_log.append((rs[:, 0:3] - env_origins).detach().cpu().numpy())
        quat_wxyz = rs[:, 3:7].detach().cpu().numpy()
        root_rot_log.append(quat_wxyz[:, [1, 2, 3, 0]])  # wxyz -> xyzw
        jp_log.append(robot.data.joint_pos.detach().cpu().numpy())
        jv_log.append(robot.data.joint_vel.detach().cpu().numpy())

        dflags = dones.detach().cpu().numpy().astype(bool).reshape(-1)
        for i in range(n_clips):
            if dflags[i] and done_step[i] is None:
                done_step[i] = t

    root_pos = np.stack(root_pos_log, axis=0)  # (T, N, 3)
    root_rot = np.stack(root_rot_log, axis=0)  # (T, N, 4)
    jp = np.stack(jp_log, axis=0)              # (T, N, 12)
    jv = np.stack(jv_log, axis=0)              # (T, N, 12)

    os.makedirs(args_cli.out_dir, exist_ok=True)
    for i, (vx, vy, wz) in enumerate(SCHEDULE):
        # Truncate at first termination (a fall would teleport the root on reset).
        length = done_step[i] if done_step[i] is not None else record_steps
        if length < int(1.0 / dt):  # < 1s of clean gait
            print(f"[WARN] clip {i} (vx={vx} wz={wz}) only {length} frames before reset — likely fell; skipping.")
            continue
        clip = {
            "fps": float(fps),
            "root_pos": root_pos[:length, i].astype(np.float32),
            "root_rot": root_rot[:length, i].astype(np.float32),  # xyzw
            "dof_pos": jp[:length, i].astype(np.float32),
            "dof_vel": jv[:length, i].astype(np.float32),
            "joint_names": joint_names,
            "command": [float(vx), float(vy), float(wz)],
        }
        out_path = os.path.join(args_cli.out_dir, f"walk_policy_{i}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(clip, f)
        print(f"[SAVE] {out_path}: {length} frames  cmd=(vx={vx}, vy={vy}, wz={wz})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
