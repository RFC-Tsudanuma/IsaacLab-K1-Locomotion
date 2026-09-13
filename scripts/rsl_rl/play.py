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

# when viser is enabled, force headless to avoid spinning up the Isaac Sim viewer.
if args_cli.viser:
    args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
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
from isaaclab_k1_locomotion.tasks.manager_based.locomotion.agents.history_policy_exporter import (
    export_history_policy_as_jit,
    export_history_policy_as_onnx,
    is_history_policy,
)


DEFAULT_VISER_URDF = str(
    Path(__file__).resolve().parent
    / "../../assets_soccer/booster_robotics_robots/K1/K1_locomotion.urdf"
)


def setup_viser(env, urdf_path: str, port: int):
    """Spin up a viser server and load the given URDF for visualization.

    Returns a tuple ``(server, base_frame, viser_urdf, joint_indices, gui)`` where
    ``joint_indices[k]`` is the index in Isaac Lab's joint state for the k-th
    actuated joint reported by ``viser_urdf.get_actuated_joint_names()``. An
    entry is ``None`` when no matching joint exists. ``gui`` is a dict of viser
    GUI handles for the velocity command sliders.
    """
    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=port)
    # NOTE: plane_color / plane_opacity / shadow_opacity / infinite_grid は新しめの viser のみ。
    # isaacsim が websockets==12 を要求するため viser は 0.2.7 (websockets12 互換) を使う必要があり、
    # そのバージョンの add_grid には無いので基本 kwargs のみにする (新しい viser でも動く)。
    server.scene.add_grid(
        "/ground",
        width=20.0,
        height=20.0,
        cell_size=0.5,
        section_size=2.0,
        plane="xy",
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

    # GUI for velocity command override
    with server.gui.add_folder("Velocity Command"):
        gui_vx = server.gui.add_slider("lin_vel_x [m/s]", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
        gui_vy = server.gui.add_slider("lin_vel_y [m/s]", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
        gui_wz = server.gui.add_slider("ang_vel_z [rad/s]", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
        gui_reset = server.gui.add_button("Reset to 0")

    @gui_reset.on_click
    def _(_):
        gui_vx.value = 0.0
        gui_vy.value = 0.0
        gui_wz.value = 0.0

    gui = {"vx": gui_vx, "vy": gui_vy, "wz": gui_wz}

    # ------------------------------------------------------------------
    # Turn task (Isaac-Velocity-Flat-Turn) specific GUI + scene handles.
    # Only built when the active command term is a TargetHeadingCommand.
    # ------------------------------------------------------------------
    if _is_turn_task(env):
        with server.gui.add_folder("Turn Task (target heading)"):
            gui_free_run = server.gui.add_checkbox(
                "free run (env resamples)",
                initial_value=True,
                hint="ON: the env picks new targets on its own timer (3-5 s) - this shows the"
                " trained behaviour. OFF: hold the target at the slider below.",
            )
            gui_yaw = server.gui.add_slider(
                "target_yaw [deg]", min=-180.0, max=180.0, step=5.0, initial_value=0.0,
                hint="Absolute world yaw to hold. Only used when 'free run' is OFF.",
            )
            gui_new_target = server.gui.add_button(
                "new random target", hint="Force an immediate resample (free run only)."
            )
            # Read-only readouts.
            gui_err = server.gui.add_number("remaining dpsi [deg]", initial_value=0.0, disabled=True)
            gui_tgt = server.gui.add_number("target psi [deg]", initial_value=0.0, disabled=True)
            gui_cur = server.gui.add_number("current psi [deg]", initial_value=0.0, disabled=True)
            gui_in_tgt = server.gui.add_checkbox("in target (<5 deg)", initial_value=False, disabled=True)
            gui_elapsed = server.gui.add_number("time since target [s]", initial_value=0.0, disabled=True)

        # Pending-resample flag, consumed by override_command_from_viser.
        pending = {"resample": False}

        @gui_new_target.on_click
        def _(_):
            pending["resample"] = True

        # Persistent scene handles (positions/text updated every frame).
        target_marker = server.scene.add_icosphere(
            "/turn/target_marker", radius=0.07, color=(0, 220, 60), position=(0.0, 0.0, 0.06)
        )
        readout_label = server.scene.add_label("/turn/label", "dpsi = 0.0 deg", position=(0.0, 0.0, 1.1))

        gui.update({
            "turn": True,
            "free_run": gui_free_run,
            "target_yaw": gui_yaw,
            "pending": pending,
            "err": gui_err,
            "tgt": gui_tgt,
            "cur": gui_cur,
            "in_tgt": gui_in_tgt,
            "elapsed": gui_elapsed,
            "target_marker": target_marker,
            "readout_label": readout_label,
        })
        print("[INFO] Viser: turn-task visualization enabled (target/current heading rays + dpsi arc).")

    # NOTE: viser falls back to the next free port when the requested one is taken
    # (e.g. a previous play.py still holding it), so report the port the server
    # actually bound rather than the requested one.
    actual_port = server.get_port() if hasattr(server, "get_port") else port
    if actual_port != port:
        print(f"[WARNING] Viser: port {port} was busy; bound to {actual_port} instead.")
    print(f"[INFO] Viser visualization available at http://localhost:{actual_port}")
    return server, base_frame, viser_urdf, joint_indices, gui


def _is_turn_task(env) -> bool:
    """True when the active ``base_velocity`` term is a ``TargetHeadingCommand``.

    Detected structurally (``heading_target`` buffer + ``turn_angle_range`` cfg field)
    rather than by task id, so it also covers derived configs.
    """
    try:
        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    except Exception:
        return False
    return hasattr(cmd_term, "heading_target") and hasattr(cmd_term.cfg, "turn_angle_range")


def _update_turn_viser(env, server, gui, env_idx: int = 0):
    """Draw the target/current heading and the remaining angle for the turn task.

    Scene contents:
      * green ray  = target heading (psi_target)
      * blue ray   = current heading (psi_current)
      * orange arc = the remaining angle dpsi the robot still has to rotate
      * green ball = tip of the target ray
      * label      = signed dpsi in degrees
    """
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]

    psi_t = float(cmd_term.heading_target[env_idx])
    psi_c = float(robot.data.heading_w[env_idx])
    # Signed remaining angle, wrapped to (-pi, pi]. Reuse the command term's own
    # property so the displayed value matches exactly what the policy observes.
    d_psi = float(cmd_term.heading_error[env_idx])

    base = robot.data.root_pos_w[env_idx, :3].detach().cpu().numpy().astype(np.float32)
    origin = np.array([base[0], base[1], 0.06], dtype=np.float32)

    ray_len = 1.2
    arc_r = 0.8

    def ray(psi, length):
        return origin + np.array([math.cos(psi) * length, math.sin(psi) * length, 0.0], dtype=np.float32)

    # -- target / current heading rays (re-added by name each frame; viser replaces in place)
    server.scene.add_line_segments(
        "/turn/target_ray",
        points=np.array([[origin, ray(psi_t, ray_len)]], dtype=np.float32),
        colors=(0, 220, 60),
        line_width=5.0,
    )
    server.scene.add_line_segments(
        "/turn/current_ray",
        points=np.array([[origin, ray(psi_c, ray_len)]], dtype=np.float32),
        colors=(60, 130, 255),
        line_width=5.0,
    )

    # -- arc sweeping from the current heading to the target, i.e. the work still to do
    n_seg = 24
    angles = psi_c + d_psi * np.linspace(0.0, 1.0, n_seg + 1)
    arc_pts = origin + np.stack(
        [np.cos(angles) * arc_r, np.sin(angles) * arc_r, np.zeros_like(angles)], axis=-1
    ).astype(np.float32)
    server.scene.add_line_segments(
        "/turn/dpsi_arc",
        points=np.stack([arc_pts[:-1], arc_pts[1:]], axis=1).astype(np.float32),
        colors=(255, 160, 0),
        line_width=3.0,
    )

    # -- marker + label
    gui["target_marker"].position = ray(psi_t, ray_len)
    gui["readout_label"].text = f"dpsi = {math.degrees(d_psi):+.1f} deg"
    gui["readout_label"].position = np.array([base[0], base[1], base[2] + 0.6], dtype=np.float32)

    # -- GUI readouts
    gui["err"].value = round(math.degrees(d_psi), 1)
    gui["tgt"].value = round(math.degrees(psi_t), 1)
    gui["cur"].value = round(math.degrees(psi_c), 1)
    gui["in_tgt"].value = bool(abs(d_psi) < float(getattr(cmd_term.cfg, "success_threshold", 0.087)))
    elapsed = getattr(cmd_term, "_elapsed", None)
    if elapsed is not None:
        gui["elapsed"].value = round(float(elapsed[env_idx]), 2)


def override_command_from_viser(env, gui):
    """Overwrite the ``base_velocity`` command tensor with viser GUI values.

    Also disables the heading-based ang_vel_z recomputation and the standing-env
    zeroing so the GUI values survive the next ``_update_command`` call.
    """
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")

    # TargetHeadingCommand (Isaac-Velocity-Flat-Turn): the command is the remaining
    # angle Δψ, recomputed from ``heading_target`` every step. Writing vel_command_b
    # here would just be overwritten by ``_update_command``, so drive
    # ``heading_target`` instead.
    if gui.get("turn"):
        # "new random target" button: zeroing time_left makes the env resample on the
        # next command_manager.compute(), which goes through the real _resample_command
        # path and so also resets the per-command metric buffers (_elapsed/_settled).
        if gui["pending"]["resample"]:
            gui["pending"]["resample"] = False
            if hasattr(cmd_term, "time_left"):
                cmd_term.time_left[:] = 0.0
            return

        if bool(gui["free_run"].value):
            # Let the env drive: targets resample on their own timer (3-5 s).
            # This is what the trained policy actually sees, so it is the default.
            return

        # Hold mode: pin the target to the slider and freeze resampling.
        cmd_term.heading_target[:] = math.radians(float(gui["target_yaw"].value))
        if hasattr(cmd_term, "time_left"):
            cmd_term.time_left[:] = 1.0e9
        return

    ref = getattr(cmd_term, "vel_command_b", None)
    if ref is None:
        ref = getattr(cmd_term, "command", None)
    if ref is None:
        return
    device = ref.device
    num_envs = ref.shape[0]
    fixed = torch.tensor(
        [[float(gui["vx"].value), float(gui["vy"].value), float(gui["wz"].value)]],
        device=device,
    ).repeat(num_envs, 1)

    if hasattr(cmd_term, "vel_command_b"):
        cmd_term.vel_command_b[:] = fixed
    # Disable heading-based ang_vel_z recomputation (overwrites vel_command_b[:, 2]).
    if hasattr(cmd_term, "is_heading_env"):
        cmd_term.is_heading_env[:] = False
    # Disable standing-env zeroing of the whole command vector.
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = False


def update_viser(env, base_frame, viser_urdf, joint_indices, env_idx: int = 0, server=None, gui=None):
    """Push the current robot state from Isaac Lab into the viser scene.

    When ``server``/``gui`` are given and the active task is the turn task, also
    updates the target-heading visualization (see :func:`_update_turn_viser`).
    """
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

    if server is not None and gui is not None and gui.get("turn"):
        _update_turn_viser(env, server, gui, env_idx=env_idx)


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

    # `runner.load` can silently no-op for some checkpoints, leaving the live policy at
    # initialization values. Re-load the state_dict directly into the policy as a safeguard.
    raw_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    ckpt_msd = raw_ckpt.get("model_state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else None
    if isinstance(ckpt_msd, dict):
        policy_to_load = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        target_device = next(policy_to_load.parameters()).device
        ckpt_on_device = {k: (v.to(target_device) if isinstance(v, torch.Tensor) else v)
                          for k, v in ckpt_msd.items()}
        policy_to_load.load_state_dict(ckpt_on_device, strict=False)

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
    if is_history_policy(policy_nn):
        # 履歴 + CNN 構成 (K1 Flat): 入力は command(N,3) + obs_history(N,H,C)。
        # 詳細は agents/history_policy_exporter.py の docstring を参照。
        export_history_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_history_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    else:
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
            # override velocity command from viser GUI before policy inference
            if viser_state is not None:
                try:
                    override_command_from_viser(env, viser_state[4])
                except Exception as e:
                    print(f"[WARNING] Viser command override failed: {e}")
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if viser_state is not None:
            try:
                server, base_frame, viser_urdf, joint_indices, gui = viser_state
                update_viser(
                    env, base_frame, viser_urdf, joint_indices,
                    env_idx=args_cli.viser_env_idx, server=server, gui=gui,
                )
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