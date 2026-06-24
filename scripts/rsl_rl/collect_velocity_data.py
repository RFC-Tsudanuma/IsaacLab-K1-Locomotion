# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect (command, actual velocity) rollouts from a trained RSL-RL policy and store as zarr."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect velocity rollouts for the velocity-predictor model.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--num_rollouts", type=int, default=100)
parser.add_argument("--episode_length", type=int, default=250, help="Steps per rollout (5s @ 50Hz default).")
parser.add_argument("--output", type=str, default="data/rollouts.zarr",
                    help="Output zarr path (relative to cwd; default writes under scripts/rsl_rl/data/).")
parser.add_argument("--collect_proprio", action="store_true")
parser.add_argument("--proprio_keys", nargs="+", default=[],
                    choices=["joint_pos", "joint_vel", "base_quat", "projected_gravity",
                             "base_lin_vel_b", "base_ang_vel_b"])
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from pathlib import Path

import gymnasium as gym
import torch
import zarr
from numcodecs import Blosc
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
from isaaclab_k1_locomotion.velocity_model import sample_commands


def override_command(env, cmd_n3: torch.Tensor) -> None:
    """Overwrite the ``base_velocity`` command with a (num_envs, 3) tensor.

    Also disables heading-based ang_vel_z recomputation and the standing-env zeroing
    so the override survives the next ``_update_command`` call.
    """
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    if hasattr(cmd_term, "vel_command_b"):
        cmd_term.vel_command_b[:] = cmd_n3
    if hasattr(cmd_term, "command"):
        try:
            cmd_term.command[:] = cmd_n3
        except (AttributeError, TypeError):
            pass
    if hasattr(cmd_term, "heading_target"):
        cmd_term.heading_target[:] = 0.0
    if hasattr(cmd_term, "is_heading_env"):
        cmd_term.is_heading_env[:] = False
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = False


PROPRIO_SOURCE_KEYS = (
    "joint_pos",
    "joint_vel",
    "base_quat",
    "projected_gravity",
    "base_lin_vel_b",
    "base_ang_vel_b",
)


def _proprio_source(robot, key: str) -> torch.Tensor:
    if key == "joint_pos":
        return robot.data.joint_pos
    if key == "joint_vel":
        return robot.data.joint_vel
    if key == "base_quat":
        return robot.data.root_quat_w
    if key == "projected_gravity":
        return robot.data.projected_gravity_b
    if key == "base_lin_vel_b":
        return robot.data.root_lin_vel_b
    if key == "base_ang_vel_b":
        return robot.data.root_ang_vel_b
    raise KeyError(f"Unknown proprio key: {key}")


class VelocityDataCollector:
    def __init__(
        self,
        env,
        policy,
        episode_length: int,
        output_path: str,
        collect_proprio: bool = False,
        proprio_keys: list[str] | None = None,
    ):
        self.env = env
        self.policy = policy
        self.T = episode_length
        self.num_envs = env.num_envs
        self.device = env.device
        self.collect_proprio = collect_proprio
        self.proprio_keys = list(proprio_keys or [])
        self.dt = float(env.unwrapped.step_dt)

        self._setup_writer(output_path)
        self._setup_buffers()

    def _setup_writer(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        self.root = zarr.open_group(str(out_path), mode="w")
        chunk_eps = min(512, self.num_envs)
        T = self.T

        self.cmd_arr = self.root.create_dataset(
            "cmd", shape=(0, T, 3), chunks=(chunk_eps, T, 3),
            dtype="float32", compressor=compressor,
        )
        self.vel_arr = self.root.create_dataset(
            "vel", shape=(0, T, 3), chunks=(chunk_eps, T, 3),
            dtype="float32", compressor=compressor,
        )
        self.done_arr = self.root.create_dataset(
            "done", shape=(0, T), chunks=(chunk_eps, T),
            dtype="bool", compressor=compressor,
        )

        self.proprio_group = self.root.create_group("proprio")
        self.proprio_arrs: dict[str, zarr.Array] = {}
        if self.collect_proprio:
            self._setup_proprio_arrays(chunk_eps, compressor)

        self.root.attrs["num_envs"] = self.num_envs
        self.root.attrs["episode_length"] = T
        self.root.attrs["dt"] = self.dt

    def _setup_proprio_arrays(self, chunk_eps: int, compressor) -> None:
        T = self.T
        robot = self.env.unwrapped.scene["robot"]
        for key in self.proprio_keys:
            if key not in PROPRIO_SOURCE_KEYS:
                raise ValueError(f"Unknown proprio key: {key}")
            dim = _proprio_source(robot, key).shape[1]
            self.proprio_arrs[key] = self.proprio_group.create_dataset(
                key, shape=(0, T, dim), chunks=(chunk_eps, T, dim),
                dtype="float32", compressor=compressor,
            )

    def _setup_buffers(self) -> None:
        T, N, d = self.T, self.num_envs, self.device
        self.buf_cmd = torch.zeros(N, T, 3, device=d)
        self.buf_vel = torch.zeros(N, T, 3, device=d)
        self.buf_done = torch.zeros(N, T, dtype=torch.bool, device=d)
        self.buf_proprio: dict[str, torch.Tensor] = {}
        if self.collect_proprio:
            robot = self.env.unwrapped.scene["robot"]
            for key in self.proprio_keys:
                dim = _proprio_source(robot, key).shape[1]
                self.buf_proprio[key] = torch.zeros(N, T, dim, device=d)

    def _store_velocity(self, t: int) -> None:
        robot = self.env.unwrapped.scene["robot"]
        self.buf_vel[:, t, :2] = robot.data.root_lin_vel_b[:, :2]
        self.buf_vel[:, t, 2] = robot.data.root_ang_vel_b[:, 2]

    def _store_proprio(self, t: int) -> None:
        if not self.collect_proprio:
            return
        robot = self.env.unwrapped.scene["robot"]
        for key in self.proprio_keys:
            self.buf_proprio[key][:, t] = _proprio_source(robot, key)

    def collect_one_rollout(self) -> None:
        obs = self.env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        commands = sample_commands(self.num_envs, self.T, self.dt, self.device)

        for t in range(self.T):
            cmd_t = commands[:, t]
            override_command(self.env, cmd_t)
            with torch.inference_mode():
                action = self.policy(obs)
            obs, _, dones, _ = self.env.step(action)

            self.buf_cmd[:, t] = cmd_t
            self._store_velocity(t)
            self.buf_done[:, t] = dones
            self._store_proprio(t)

        self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        self.cmd_arr.append(self.buf_cmd.cpu().numpy())
        self.vel_arr.append(self.buf_vel.cpu().numpy())
        self.done_arr.append(self.buf_done.cpu().numpy())
        for key, buf in self.buf_proprio.items():
            self.proprio_arrs[key].append(buf.cpu().numpy())

    def collect(self, num_rollouts: int) -> None:
        for i in range(num_rollouts):
            self.collect_one_rollout()
            done_rate = float(self.buf_done.float().mean().item())
            print(
                f"[{i+1}/{num_rollouts}] total_episodes={self.cmd_arr.shape[0]:,} "
                f"done_rate={done_rate:.3f}"
            )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    # Keep episodes long enough that termination is unlikely to interrupt a rollout.
    # dt is sim.dt * decimation; computed lazily later, but a 0.02 estimate is exact for K1.
    needed_s = args_cli.episode_length * 0.02 + 1.0
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, needed_s)

    # Disable command resampling and heading-based ang_vel recomputation so override sticks.
    try:
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    except AttributeError:
        pass
    try:
        env_cfg.commands.base_velocity.heading_command = False
    except AttributeError:
        pass

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

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
    # `runner.load` does a strict load of the full actor-critic; it raises if the
    # critic obs dim differs from the checkpoint (e.g. the critic observation grew
    # after this policy was trained). For rollout collection we only need the actor,
    # which the strict=False safeguard below loads regardless, so tolerate the error.
    try:
        runner.load(resume_path)
    except RuntimeError as exc:
        print(f"[WARN] runner.load strict load failed ({exc}); relying on strict=False safeguard.")

    # Safeguard against `runner.load` no-op (see play.py:281-290).
    raw_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    ckpt_msd = raw_ckpt.get("model_state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else None
    if isinstance(ckpt_msd, dict):
        policy_to_load = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        target_device = next(policy_to_load.parameters()).device
        ckpt_on_device = {k: (v.to(target_device) if isinstance(v, torch.Tensor) else v)
                          for k, v in ckpt_msd.items()}
        # strict=False ignores missing/extra keys but NOT shape mismatches, so drop any
        # checkpoint tensor whose shape differs from the current model (e.g. a critic that
        # grew after this policy was trained). The actor is unaffected and loads fine.
        model_sd = policy_to_load.state_dict()
        filtered = {k: v for k, v in ckpt_on_device.items()
                    if not (isinstance(v, torch.Tensor) and k in model_sd and v.shape != model_sd[k].shape)}
        skipped = [k for k in ckpt_on_device if k not in filtered]
        if skipped:
            print(f"[WARN] skipping {len(skipped)} checkpoint tensors with shape mismatch: {skipped}")
        policy_to_load.load_state_dict(filtered, strict=False)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    print(
        f"[INFO] dt={env.unwrapped.step_dt:.4f}s, num_envs={args_cli.num_envs}, "
        f"episode_length={args_cli.episode_length}, num_rollouts={args_cli.num_rollouts}, "
        f"output={args_cli.output}"
    )

    collector = VelocityDataCollector(
        env=env,
        policy=policy,
        episode_length=args_cli.episode_length,
        output_path=args_cli.output,
        collect_proprio=args_cli.collect_proprio,
        proprio_keys=args_cli.proprio_keys,
    )
    collector.collect(args_cli.num_rollouts)

    print(f"[INFO] Wrote rollouts to {args_cli.output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
