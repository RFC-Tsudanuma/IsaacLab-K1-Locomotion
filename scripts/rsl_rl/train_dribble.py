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
    help="Clipping range for the high-level action (interpreted as the walking command for the frozen policy).",
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
import math
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups

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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _term_slot(observation_manager, group: str, term_name: str) -> tuple[int, int]:
    """Return (start, end) column indices of ``term_name`` within the concatenated tensor of ``group``."""
    if group not in observation_manager.active_terms:
        raise ValueError(
            f"Observation group '{group}' not found. Available: {list(observation_manager.active_terms.keys())}"
        )
    term_names = list(observation_manager.active_terms[group])
    term_dims = list(observation_manager.group_obs_term_dim[group])
    if term_name not in term_names:
        raise ValueError(
            f"Term '{term_name}' not found in group '{group}'. Available terms: {term_names}"
        )
    idx = term_names.index(term_name)
    start = sum(int(math.prod(d)) for d in term_dims[:idx])
    end = start + int(math.prod(term_dims[idx]))
    return start, end


class HierarchicalVecEnvWrapper:
    """rsl_rl VecEnv-compatible wrapper exposing a high-level (walking command) action space.

    Each step:
        1. Read the current env observation.
        2. Build a copy of the chosen ``low_level_obs_group`` tensor with the
           ``low_level_cmd_term_name`` slice overwritten by the high-level action.
        3. Run the frozen low-level policy on the modified observation → joint targets.
        4. Step the inner env with those joint targets.

    The env's ``command_manager`` is not touched.
    """

    def __init__(
        self,
        inner_env: RslRlVecEnvWrapper,
        low_level_policy: torch.nn.Module,
        low_level_obs_group: str = "policy",
        low_level_cmd_term_name: str = "velocity_commands",
        action_clip: float = 1.0,
        high_action_dim: int = 3,
    ):
        self.env = inner_env
        self.low_level_policy = low_level_policy
        self.low_level_obs_group = low_level_obs_group
        self.action_clip = float(action_clip)
        # rsl_rl VecEnv-required attrs
        self.num_envs = inner_env.num_envs
        self.device = inner_env.device
        self.max_episode_length = inner_env.max_episode_length
        self.num_actions = int(high_action_dim)
        # Resolve the slice of the walking-command observation term inside the chosen group.
        om = inner_env.unwrapped.observation_manager
        self._cmd_slot = _term_slot(om, low_level_obs_group, low_level_cmd_term_name)
        slot_dim = self._cmd_slot[1] - self._cmd_slot[0]
        if slot_dim != self.num_actions:
            raise ValueError(
                f"Slot dim of obs term '{low_level_cmd_term_name}' in group '{low_level_obs_group}' is"
                f" {slot_dim}, but high_action_dim is {self.num_actions}."
            )

    # -- pass-through properties expected by rsl_rl / IsaacLab utilities --
    @property
    def cfg(self):
        return self.env.cfg

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def render_mode(self):
        return self.env.render_mode

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.env.episode_length_buf = value

    # -- VecEnv API --
    def reset(self):
        return self.env.reset()

    def get_observations(self):
        return self.env.get_observations()

    def step(self, action: torch.Tensor):
        cmd = action.to(self.device).clamp(-self.action_clip, self.action_clip)

        # Snapshot of the current env observation (used as the *base* for both policies).
        env_obs = self.env.get_observations()

        # Build a modified low-level obs without mutating env_obs.
        low_group_tensor = env_obs[self.low_level_obs_group].clone()
        s, e = self._cmd_slot
        low_group_tensor[:, s:e] = cmd
        low_obs = {k: env_obs[k] for k in env_obs.keys()}
        low_obs[self.low_level_obs_group] = low_group_tensor

        # Frozen low-level policy → joint targets.
        joint_action = self.low_level_policy.act_inference(low_obs)

        # Step the inner env with joint targets; return its outputs to the runner.
        return self.env.step(joint_action)

    def close(self):
        return self.env.close()


class _JitFrozenPolicy:
    """TorchScript wrapper exposing the same ``act_inference`` API as RSL-RL policies.

    Used when the frozen-checkpoint ``.pt`` is a TorchScript archive produced by
    ``export_policy_as_jit`` (i.e. its ``forward(obs) -> action`` already bakes
    in the normalizer).
    """

    def __init__(self, jit_module: torch.jit.ScriptModule, low_level_obs_group: str, device: str):
        self._module = jit_module.to(device).eval()
        for p in self._module.parameters():
            p.requires_grad_(False)
        self._low_level_obs_group = low_level_obs_group
        self._device = device

    @torch.inference_mode()
    def act_inference(self, obs: dict) -> torch.Tensor:
        return self._module(obs[self._low_level_obs_group].to(self._device))


class _OnnxFrozenPolicy:
    """ONNX-runtime wrapper exposing the same ``act_inference`` API as RSL-RL policies.

    Reads the concatenated tensor of ``low_level_obs_group`` from the obs dict,
    runs ONNX inference, and returns the resulting action tensor on ``device``.
    Recurrent policies (LSTM/GRU with extra ``h_in`` / ``c_in`` inputs) are not
    supported.
    """

    def __init__(self, onnx_path: str, low_level_obs_group: str, device: str):
        import onnxruntime as ort  # local import: only needed for the ONNX branch

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device else ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        input_names = [i.name for i in self._session.get_inputs()]
        if input_names != ["obs"]:
            raise ValueError(
                f"ONNX frozen policy expects a single input named 'obs' (non-recurrent export),"
                f" got inputs={input_names}. Recurrent ONNX frozen policies are not supported."
            )
        self._output_name = self._session.get_outputs()[0].name
        self._low_level_obs_group = low_level_obs_group
        self._device = device

    def act_inference(self, obs: dict) -> torch.Tensor:
        x = obs[self._low_level_obs_group]
        x_np = x.detach().cpu().numpy().astype("float32", copy=False)
        out = self._session.run([self._output_name], {"obs": x_np})[0]
        return torch.from_numpy(out).to(self._device)


def _build_frozen_policy(
    env: RslRlVecEnvWrapper,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint_path: str,
    device: str,
    low_level_obs_group: str,
):
    """Construct a low-level policy and load weights from a ``.pt`` or ``.onnx`` checkpoint.

    For ``.pt`` checkpoints (raw state_dict), an ``ActorCritic`` mirroring
    ``agent_cfg.policy`` is built and weights are loaded. For ``.onnx``
    checkpoints (e.g. produced by ``export_policy.py``), an ONNX-runtime
    wrapper is returned.

    The policy is configured to read its input from ``low_level_obs_group`` so
    that its concatenated input matches its training structure regardless of
    how the high-level "policy" group is shaped.
    """
    ext = os.path.splitext(checkpoint_path)[1].lower()
    if ext == ".onnx":
        return _OnnxFrozenPolicy(checkpoint_path, low_level_obs_group, device)

    # ``.pt`` can be either a TorchScript archive (export_policy_as_jit) or a
    # raw state_dict (rsl_rl ``model_*.pt``). Try TorchScript first.
    try:
        jit_module = torch.jit.load(checkpoint_path, map_location=device)
        return _JitFrozenPolicy(jit_module, low_level_obs_group, device)
    except RuntimeError:
        pass

    agent_dict = agent_cfg.to_dict()
    policy_cfg = dict(agent_dict["policy"])
    policy_class_name = policy_cfg.pop("class_name", "ActorCritic")
    policy_class = {"ActorCritic": ActorCritic, "ActorCriticRecurrent": ActorCriticRecurrent}[policy_class_name]

    obs = env.get_observations()
    forced_groups = {"policy": [low_level_obs_group], "critic": [low_level_obs_group]}
    obs_groups = resolve_obs_groups(obs, forced_groups, ["critic"])

    frozen = policy_class(obs, obs_groups, env.num_actions, **policy_cfg).to(device)

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    # Frozen policy is only used for inference; critic_* tensors may have a
    # different shape (privileged obs) and would fail the size check even with
    # strict=False, so drop them entirely.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(("critic", "critic_obs_normalizer"))}
    # rsl_rl's ActorCritic.load_state_dict overrides the base method and returns a bool.
    frozen.load_state_dict(state_dict, strict=False)

    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    return frozen


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train the high-level policy on top of a frozen walking policy."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

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
        f.write(f"high_action_clip: {args_cli.high_action_clip}\n")
        f.write(f"low_level_obs_group: {args_cli.low_level_obs_group}\n")
        f.write(f"low_level_cmd_term_name: {args_cli.low_level_cmd_term_name}\n")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
