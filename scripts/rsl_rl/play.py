# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
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
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--viser", action="store_true", default=False, help="Enable viser-based visualization.")
parser.add_argument("--viser_port", type=int, default=8080, help="Port for the viser server.")
parser.add_argument(
    "--viser_urdf",
    type=str,
    default=None,
    help="Path to the URDF used for viser visualization. Defaults to the K1 locomotion URDF.",
)
parser.add_argument(
    "--viser_env_idx", type=int, default=0, help="Index of the environment to visualize in viser."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time
from pathlib import Path

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
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401


DEFAULT_VISER_URDF = str(
    Path(__file__).resolve().parent
    / "../../assets_soccer/booster_robotics_robots/K1/K1_locomotion.urdf"
)


def setup_viser(env, urdf_path: str, port: int):
    """Spin up a viser server and load the given URDF for visualization.

    Returns a tuple ``(server, base_frame, viser_urdf, joint_indices)`` where
    ``joint_indices[k]`` is the index in Isaac Lab's joint state for the k-th
    actuated joint reported by ``viser_urdf.get_actuated_joint_names()``. An
    entry is ``None`` when no matching joint exists.
    """
    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=port)
    server.scene.add_grid(
        "/ground",
        width=20.0,
        height=20.0,
        cell_size=0.5,
        section_size=2.0,
        plane="xy",
        plane_color=(0.85, 0.85, 0.85),
        plane_opacity=1.0,
        shadow_opacity=0.3,
        infinite_grid=True,
    )
    server.scene.add_frame("/world", show_axes=True, axes_length=0.3, axes_radius=0.01)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    viser_urdf = ViserUrdf(server, urdf_or_path=Path(urdf_path), root_node_name="/base")

    urdf_joint_names = viser_urdf.get_actuated_joint_names()
    isaac_joint_names = list(env.unwrapped.scene["robot"].joint_names)
    joint_indices = []
    for name in urdf_joint_names:
        if name in isaac_joint_names:
            joint_indices.append(isaac_joint_names.index(name))
        else:
            print(f"[WARNING] Viser: joint '{name}' not found in Isaac robot; will use 0.0.")
            joint_indices.append(None)
    print(f"[INFO] Viser visualization available at http://localhost:{port}")
    return server, base_frame, viser_urdf, joint_indices


def update_viser(env, base_frame, viser_urdf, joint_indices, env_idx: int = 0):
    """Push the current robot state from Isaac Lab into the viser scene."""
    robot = env.unwrapped.scene["robot"]
    root_state = robot.data.root_state_w[env_idx]
    pos = root_state[0:3].detach().cpu().numpy()
    quat_wxyz = root_state[3:7].detach().cpu().numpy()
    joint_pos = robot.data.joint_pos[env_idx].detach().cpu().numpy()

    cfg = np.array(
        [joint_pos[i] if i is not None else 0.0 for i in joint_indices], dtype=np.float32
    )

    base_frame.position = pos.astype(np.float32)
    base_frame.wxyz = quat_wxyz.astype(np.float32)
    viser_urdf.update_cfg(cfg)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        # Keep reference to the RecordVideo wrapper for manual frame capture fallback
        _record_video_wrapper = env

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # set up viser visualization (optional)
    viser_state = None
    if args_cli.viser:
        urdf_path = args_cli.viser_urdf or DEFAULT_VISER_URDF
        viser_state = setup_viser(env, urdf_path, args_cli.viser_port)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # manual frame buffer for robust video recording
    _manual_frames = [] if args_cli.video else None
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if viser_state is not None:
            try:
                _, base_frame, viser_urdf, joint_indices = viser_state
                update_viser(env, base_frame, viser_urdf, joint_indices, env_idx=args_cli.viser_env_idx)
            except Exception as e:
                print(f"[WARNING] Viser update failed: {e}")
        if args_cli.video:
            # Manually capture frame as fallback in case RecordVideo fails
            try:
                frame = env.unwrapped.render()
            except Exception:
                frame = None
            if frame is not None and hasattr(frame, "shape") and frame.size > 0:
                _manual_frames.append(frame.copy())
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Save video from manual frames if RecordVideo wrapper produced no output
    if args_cli.video and _manual_frames:
        video_dir = os.path.join(log_dir, "videos", "play")
        os.makedirs(video_dir, exist_ok=True)
        # Check if RecordVideo already saved a file
        existing = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        if not existing:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
                import numpy as np

                frames = [f if f.ndim == 3 else np.stack([f] * 3, axis=-1) for f in _manual_frames]
                clip = ImageSequenceClip(frames, fps=env.unwrapped.metadata.get("render_fps", 30))
                out_path = os.path.join(video_dir, "rl-video-step-0.mp4")
                clip.write_videofile(out_path, logger=None)
                del clip
                print(f"[INFO] Video saved to: {out_path}")
            except Exception as e:
                print(f"[WARNING] Could not save video: {e}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()