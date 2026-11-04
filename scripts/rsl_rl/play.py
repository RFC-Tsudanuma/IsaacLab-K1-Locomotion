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
import trimesh
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

    # ------------------------------------------------------------------
    # Waypoint task (Isaac-Waypoint-K1) specific GUI. Scene objects are (re)drawn
    # every frame in _update_waypoint_viser.
    # ------------------------------------------------------------------
    if _is_waypoint_task(env):
        wp_cfg = env.unwrapped.command_manager.get_term("waypoints").cfg
        with server.gui.add_folder("Waypoint Task"):
            gui_wp_mode = server.gui.add_dropdown(
                "mode", options=("random", "fast", "strict"), initial_value="random",
                hint=f"fast: pass radius {wp_cfg.loose_radius} m (may cut corners), strict: {wp_cfg.strict_radius} m."
                " random: the env samples the mode at every reset (training distribution).",
            )
            gui_wp_new = server.gui.add_button("new random path", hint="Resample the path now, keeping the robot pose.")
            gui_wp_next = server.gui.add_number("next point idx", initial_value=0, disabled=True)
            gui_wp_n = server.gui.add_number("num waypoints", initial_value=0, disabled=True)
            gui_wp_rem = server.gui.add_number("remaining path [m]", initial_value=0.0, disabled=True)
            gui_wp_gd = server.gui.add_number("goal dist [m]", initial_value=0.0, disabled=True)
            gui_wp_gy = server.gui.add_number("goal yaw err [deg]", initial_value=0.0, disabled=True)
            gui_wp_reached = server.gui.add_checkbox("reached goal", initial_value=False, disabled=True)
            gui_wp_t = server.gui.add_number("time to goal [s]", initial_value=0.0, disabled=True)
            gui_wp_ep = server.gui.add_number("episode time [s]", initial_value=0.0, disabled=True)
            gui_wp_cmd = server.gui.add_text("command vx, vy, wz", initial_value="+0.00, +0.00, +0.00", disabled=True)

        pending_wp = {"resample": False}

        @gui_wp_new.on_click
        def _(_):
            pending_wp["resample"] = True

        gui.update({
            "waypoint": True,
            "wp_mode": gui_wp_mode,
            "wp_pending": pending_wp,
            "wp_next": gui_wp_next,
            "wp_n": gui_wp_n,
            "wp_rem": gui_wp_rem,
            "wp_gd": gui_wp_gd,
            "wp_gy": gui_wp_gy,
            "wp_reached": gui_wp_reached,
            "wp_t": gui_wp_t,
            "wp_ep": gui_wp_ep,
            "wp_cmd": gui_wp_cmd,
            "wp_trail": [],  # robot xy trace of the current episode
            "wp_spheres": {},  # name -> handle, so stale waypoint spheres can be removed
        })
        print("[INFO] Viser: waypoint-task visualization enabled (path / waypoints / goal pose / trail)."
              " The velocity sliders are ignored: base_velocity is driven by the policy.")

    # ------------------------------------------------------------------
    # Block task (Isaac-Block-K1) specific GUI. Scene objects are (re)drawn
    # every frame in _update_block_viser.
    # ------------------------------------------------------------------
    if _is_block_task(env):
        with server.gui.add_folder("Block Task (goalkeeper)"):
            gui_bl_new = server.gui.add_button("new ball", hint="Re-launch a ball now (goal line = current pose).")
            gui_bl_last = server.gui.add_text("last outcome", initial_value="-", disabled=True)
            gui_bl_cnt = server.gui.add_text("conceded / resolved / fell", initial_value="0 / 0 / 0", disabled=True)
            gui_bl_speed = server.gui.add_number("launch speed [m/s]", initial_value=0.0, disabled=True)
            gui_bl_lat = server.gui.add_number("target lateral [m]", initial_value=0.0, disabled=True)
            gui_bl_bspeed = server.gui.add_number("ball speed [m/s]", initial_value=0.0, disabled=True)
            gui_bl_bx = server.gui.add_number("ball x on line [m]", initial_value=0.0, disabled=True)
            gui_bl_tta = server.gui.add_number("time to line [s]", initial_value=0.0, disabled=True)
            gui_bl_ep = server.gui.add_number("episode time [s]", initial_value=0.0, disabled=True)
            gui_bl_cmd = server.gui.add_text("command vx, vy, wz", initial_value="+0.00, +0.00, +0.00", disabled=True)
            gui_bl_res = server.gui.add_number("|residual| max", initial_value=0.0, disabled=True)

        pending_bl = {"relaunch": False}

        @gui_bl_new.on_click
        def _(_):
            pending_bl["relaunch"] = True

        gui.update({
            "block": True,
            "bl_pending": pending_bl,
            "bl_last": gui_bl_last,
            "bl_cnt": gui_bl_cnt,
            "bl_counts": [0, 0, 0],
            "bl_speed": gui_bl_speed,
            "bl_lat": gui_bl_lat,
            "bl_bspeed": gui_bl_bspeed,
            "bl_bx": gui_bl_bx,
            "bl_tta": gui_bl_tta,
            "bl_ep": gui_bl_ep,
            "bl_cmd": gui_bl_cmd,
            "bl_res": gui_bl_res,
            "bl_trail": [],  # robot xy trace of the current episode
            "bl_ball_trail": [],  # true ball trace of the current episode
        })
        print("[INFO] Viser: block-task visualization enabled (ball true/observed, goal line, intercept, trails)."
              " The velocity sliders are ignored: base_velocity is driven by the policy.")

    # ------------------------------------------------------------------
    # Steal task (Isaac-Steal-K1) specific GUI. Scene objects are (re)drawn
    # every frame in _update_steal_viser.
    # ------------------------------------------------------------------
    if _is_steal_task(env):
        with server.gui.add_folder("Steal Task"):
            gui_st_new = server.gui.add_button("new placement", hint="Re-place the obstacle and the ball now.")
            gui_st_last = server.gui.add_text("last outcome", initial_value="-", disabled=True)
            gui_st_cnt = server.gui.add_text("stolen / lost / no_foot / fell", initial_value="0 / 0 / 0 / 0", disabled=True)
            gui_st_state = server.gui.add_text("state", initial_value="approach", disabled=True)
            gui_st_body = server.gui.add_number("body-ball [m]", initial_value=0.0, disabled=True)
            gui_st_foot = server.gui.add_number("foot-ball [m]", initial_value=0.0, disabled=True)
            gui_st_br = server.gui.add_number("ball-obstacle [m]", initial_value=0.0, disabled=True)
            gui_st_away = server.gui.add_number("ball away speed [m/s]", initial_value=0.0, disabled=True)
            gui_st_ep = server.gui.add_number("episode time [s]", initial_value=0.0, disabled=True)
            gui_st_cmd = server.gui.add_text("command vx, vy, wz", initial_value="+0.00, +0.00, +0.00", disabled=True)
            gui_st_res = server.gui.add_number("|residual| max", initial_value=0.0, disabled=True)

        pending_st = {"replace": False}

        @gui_st_new.on_click
        def _(_):
            pending_st["replace"] = True

        gui.update({
            "steal": True,
            "st_pending": pending_st,
            "st_last": gui_st_last,
            "st_cnt": gui_st_cnt,
            "st_counts": [0, 0, 0, 0],
            "st_state": gui_st_state,
            "st_body": gui_st_body,
            "st_foot": gui_st_foot,
            "st_br": gui_st_br,
            "st_away": gui_st_away,
            "st_ep": gui_st_ep,
            "st_cmd": gui_st_cmd,
            "st_res": gui_st_res,
            "st_trail": [],  # robot xy trace of the current episode
            "st_ball_trail": [],  # ball trace of the current episode
        })
        print("[INFO] Viser: steal-task visualization enabled (obstacle, steal radius, ball, foot contact, trails)."
              " The velocity sliders are ignored: base_velocity is driven by the policy.")

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


def _add_polyline(server, name: str, pts: np.ndarray, color, line_width: float):
    """Straight polyline through ``pts`` (N, 3). viser 0.2.7 has no add_line_segments, so use a
    cubic Bezier whose control points sit on the chord of each segment (= straight lines)."""
    pts = np.asarray(pts, dtype=np.float32)
    ctrl = np.stack([pts[:-1] * 2 / 3 + pts[1:] / 3, pts[:-1] / 3 + pts[1:] * 2 / 3], axis=1).reshape(-1, 3)
    return server.scene.add_spline_cubic_bezier(name, positions=pts, control_points=ctrl, line_width=line_width, color=color)


def _is_waypoint_task(env) -> bool:
    """True when the env has a ``waypoints`` command term (``WaypointCommand``)."""
    try:
        return hasattr(env.unwrapped.command_manager.get_term("waypoints"), "points_w")
    except Exception:
        return False


def _update_waypoint_viser(env, server, gui, env_idx: int = 0):
    """Draw the waypoint task state for one env.

    Scene contents:
      * orange spheres / polyline = remaining waypoints and the path robot -> next -> ... -> goal
      * grey spheres              = already passed waypoints
      * green sphere + ray        = goal position and goal heading
      * thin circle               = pass radius around the next waypoint (goal tolerance at the goal)
      * blue trace                = robot xy trajectory of the current episode
    """
    wp = env.unwrapped.command_manager.get_term("waypoints")
    robot = env.unwrapped.scene["robot"]
    z = 0.05
    n_wp = int(wp.num_waypoints[env_idx])
    tgt = int(wp.target_idx[env_idx])
    pts = wp.points_w[env_idx, : n_wp + 1].detach().cpu().numpy()  # (n_wp + 1, 2); last row = goal
    pts3 = np.concatenate([pts, np.full((len(pts), 1), z)], axis=1).astype(np.float32)
    base = robot.data.root_pos_w[env_idx, :3].detach().cpu().numpy().astype(np.float32)
    here = np.array([base[0], base[1], z], dtype=np.float32)

    # -- waypoint spheres (re-added by name = replaced in place; extra ones from a longer path removed)
    spheres = gui["wp_spheres"]
    for i, p in enumerate(pts3):
        color = (0, 220, 60) if i == n_wp else ((140, 140, 140) if i < tgt else (255, 160, 0))
        spheres[f"/waypoint/pt_{i}"] = server.scene.add_icosphere(f"/waypoint/pt_{i}", radius=0.06, color=color, position=p)
    for name in [k for k in spheres if int(k.rsplit("_", 1)[1]) > n_wp]:
        spheres.pop(name).remove()

    # -- remaining path: robot -> next waypoint -> ... -> goal
    chain = np.concatenate([here[None], pts3[tgt:]], axis=0)
    _add_polyline(server, "/waypoint/path", chain, (255, 160, 0), 4.0)

    # -- goal heading ray
    yaw = float(wp.goal_yaw_w[env_idx])
    tip = pts3[-1] + np.array([math.cos(yaw) * 0.4, math.sin(yaw) * 0.4, 0.0], dtype=np.float32)
    _add_polyline(server, "/waypoint/goal_yaw", np.stack([pts3[-1], tip]), (0, 220, 60), 5.0)

    # -- pass radius circle around the next waypoint (goal tolerance once all waypoints are passed)
    r = float(wp.pass_radius[env_idx]) if tgt < n_wp else float(wp.cfg.goal_pos_tolerance)
    ang = np.linspace(0.0, 2.0 * np.pi, 33)
    circ = pts3[tgt] + np.stack([np.cos(ang) * r, np.sin(ang) * r, np.zeros_like(ang)], axis=-1).astype(np.float32)
    _add_polyline(server, "/waypoint/pass_radius", circ, (255, 160, 0) if tgt < n_wp else (0, 220, 60), 2.0)

    # -- trail of the current episode
    trail = gui["wp_trail"]
    if int(env.unwrapped.episode_length_buf[env_idx]) <= 1:
        trail.clear()
    trail.append(here)
    if len(trail) > 1500:
        del trail[0]
    if len(trail) >= 2:
        _add_polyline(server, "/waypoint/trail", np.stack(trail), (60, 130, 255), 3.0)

    # -- readouts
    mode = "strict" if float(wp.pass_radius[env_idx]) < 0.5 * (wp.cfg.strict_radius + wp.cfg.loose_radius) else "fast"
    rem = float(wp.path_length()[env_idx])
    server.scene.add_label(
        "/waypoint/label",
        f"{mode} (r={float(wp.pass_radius[env_idx]):.2f}) | remaining {rem:.2f} m",
        position=np.array([base[0], base[1], base[2] + 0.6], dtype=np.float32),
    )
    gui["wp_next"].value = tgt
    gui["wp_n"].value = n_wp
    gui["wp_rem"].value = round(rem, 2)
    gui["wp_gd"].value = round(float(wp.goal_dist()[env_idx]), 2)
    gui["wp_gy"].value = round(math.degrees(float(wp.goal_yaw_err()[env_idx])), 1)
    gui["wp_reached"].value = bool(wp.reached_goal[env_idx])
    gui["wp_t"].value = round(float(wp.time_to_goal[env_idx]), 2)
    gui["wp_ep"].value = round(float(env.unwrapped.episode_length_buf[env_idx]) * env.unwrapped.step_dt, 2)
    cmd = env.unwrapped.command_manager.get_command("base_velocity")[env_idx].tolist()
    gui["wp_cmd"].value = f"{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}"


def _is_block_task(env) -> bool:
    """True when the env has a ``ball_launch`` command term (``BallLaunchCommand``)."""
    try:
        return hasattr(env.unwrapped.command_manager.get_term("ball_launch"), "line_pos_w")
    except Exception:
        return False


def _update_block_viser(env, server, gui, env_idx: int = 0):
    """Draw the block (goalkeeper) task state for one env.

    Scene contents:
      * white sphere + grey trace  = true ball and its trajectory this episode
      * orange sphere + ray        = ball as the policy sees it (delayed / 30 fps / noisy) and its velocity
      * red line                   = goal line (half width = goal_half_width); thin red line = conceded threshold
      * yellow sphere              = predicted crossing point of the ball on the goal line (while approaching)
      * blue trace                 = robot xy trajectory of the current episode
    """
    bl = env.unwrapped.command_manager.get_term("ball_launch")
    robot = env.unwrapped.scene["robot"]
    ball = env.unwrapped.scene["soccer_ball"]
    cfg = bl.cfg
    z = 0.02
    ep_steps = int(env.unwrapped.episode_length_buf[env_idx])
    base = robot.data.root_pos_w[env_idx, :3].detach().cpu().numpy().astype(np.float32)
    here = np.array([base[0], base[1], z], dtype=np.float32)

    # -- outcome of the step that just finished (dones are computed before the reset)
    tm = env.unwrapped.termination_manager
    done_name = None
    for name, label in (("ball_conceded", "conceded"), ("ball_resolved", "resolved"), ("base_contact", "fell"), ("base_height", "fell")):
        try:
            if bool(tm.get_term(name)[env_idx]):
                done_name = label
                break
        except Exception:
            pass
    if done_name is not None:
        idx = ("conceded", "resolved", "fell").index(done_name)
        gui["bl_counts"][idx] += 1
        detail = done_name
        if done_name == "resolved":
            parts = [k for k in ("out_wide", "away", "stopped") if bool(bl.metrics[k][env_idx])]
            detail += " (" + ", ".join(parts) + ")" if parts else ""
        gui["bl_last"].value = detail
        gui["bl_cnt"].value = " / ".join(str(c) for c in gui["bl_counts"])
        gui["bl_trail"].clear()
        gui["bl_ball_trail"].clear()

    # -- goal line (fixed at the episode start pose): x = line heading (forward), y = left
    lp = bl.line_pos_w[env_idx].detach().cpu().numpy()
    h = float(bl.line_heading[env_idx])
    fwd = np.array([math.cos(h), math.sin(h), 0.0], dtype=np.float32)
    left = np.array([-math.sin(h), math.cos(h), 0.0], dtype=np.float32)
    origin = np.array([lp[0], lp[1], z], dtype=np.float32)
    w = float(cfg.goal_half_width)
    _add_polyline(server, "/block/goal_line", np.stack([origin - left * w, origin + left * w]), (230, 40, 40), 6.0)
    behind = origin - fwd * float(cfg.behind_x)
    _add_polyline(server, "/block/conceded_line", np.stack([behind - left * w, behind + left * w]), (230, 40, 40), 2.0)
    _add_polyline(server, "/block/line_fwd", np.stack([origin, origin + fwd * 0.4]), (230, 40, 40), 3.0)

    # -- true ball + trail
    bp = ball.data.root_pos_w[env_idx, :3].detach().cpu().numpy().astype(np.float32)
    bv = ball.data.root_lin_vel_w[env_idx, :3].detach().cpu().numpy()
    server.scene.add_icosphere("/block/ball", radius=float(cfg.ball_radius), color=(245, 245, 245), position=bp)
    trail = gui["bl_ball_trail"]
    if ep_steps <= 1:
        trail.clear()
    trail.append(np.array([bp[0], bp[1], z], dtype=np.float32))
    if len(trail) > 1000:
        del trail[0]
    if len(trail) >= 2:
        _add_polyline(server, "/block/ball_trail", np.stack(trail), (160, 160, 160), 2.0)

    # -- observed (delayed / noisy) ball: relative pos + vel in the base yaw frame -> world
    try:
        obs_term = next(c.func for c in env.unwrapped.observation_manager._group_obs_class_term_cfgs["ball_noisy"] if c.func.__class__.__name__ == "DelayedBallObs")
        rel = obs_term._out[env_idx].detach().cpu().numpy()
        qw, qx, qy, qz = robot.data.root_quat_w[env_idx].detach().cpu().numpy().tolist()
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        c, s_ = math.cos(yaw), math.sin(yaw)
        obs_p = np.array([base[0] + c * rel[0] - s_ * rel[1], base[1] + s_ * rel[0] + c * rel[1], float(cfg.ball_radius)], dtype=np.float32)
        obs_v = np.array([c * rel[2] - s_ * rel[3], s_ * rel[2] + c * rel[3], 0.0], dtype=np.float32)
        server.scene.add_icosphere("/block/ball_obs", radius=float(cfg.ball_radius) * 0.6, color=(255, 160, 0), position=obs_p)
        _add_polyline(server, "/block/ball_obs_vel", np.stack([obs_p, obs_p + obs_v * 0.3]), (255, 160, 0), 3.0)
    except Exception:
        pass
    _add_polyline(server, "/block/ball_vel", np.stack([bp, bp + bv.astype(np.float32) * 0.3]), (245, 245, 245), 3.0)

    # -- predicted crossing point on the goal line (same formula as foot_on_intercept)
    x, y, vx, vy = (float(t[env_idx]) for t in bl.ball_line_state())
    approaching = vx < -0.1 and x > 0.0
    if approaching:
        y_int = y + vy * (x / max(-vx, 0.1))
        p_int = origin + left * float(np.clip(y_int, -w, w))
        server.scene.add_icosphere("/block/intercept", radius=0.06, color=(255, 230, 0), position=p_int)
    else:
        server.scene.add_icosphere("/block/intercept", radius=0.001, color=(255, 230, 0), position=origin)

    # -- robot trail
    rtrail = gui["bl_trail"]
    if ep_steps <= 1:
        rtrail.clear()
    rtrail.append(here)
    if len(rtrail) > 1000:
        del rtrail[0]
    if len(rtrail) >= 2:
        _add_polyline(server, "/block/trail", np.stack(rtrail), (60, 130, 255), 3.0)

    # -- readouts
    speed = float(np.linalg.norm(bv[:2]))
    server.scene.add_label(
        "/block/label",
        f"ball {speed:.1f} m/s, x {x:+.2f} m | last: {gui['bl_last'].value}",
        position=np.array([base[0], base[1], base[2] + 0.6], dtype=np.float32),
    )
    gui["bl_speed"].value = round(float(bl.metrics["launch_speed"][env_idx]), 2)
    gui["bl_lat"].value = round(float(bl.launch_lateral[env_idx]), 2)
    gui["bl_bspeed"].value = round(speed, 2)
    gui["bl_bx"].value = round(x, 2)
    gui["bl_tta"].value = round(x / -vx, 2) if approaching else 0.0
    gui["bl_ep"].value = round(ep_steps * env.unwrapped.step_dt, 2)
    cmd = env.unwrapped.command_manager.get_command("base_velocity")[env_idx].tolist()
    gui["bl_cmd"].value = f"{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}"
    res = env.unwrapped.action_manager.get_term("joint_residual").raw_actions[env_idx]
    gui["bl_res"].value = round(float(res.abs().max()), 3)


def _is_steal_task(env) -> bool:
    """True when the env has a ``steal`` command term (``StealCommand``)."""
    try:
        return hasattr(env.unwrapped.command_manager.get_term("steal"), "stolen_at")
    except Exception:
        return False


def _update_steal_viser(env, server, gui, env_idx: int = 0):
    """Draw the steal task state for one env.

    Scene contents:
      * red cylinder               = obstacle (enemy); thin red circle = contact threshold (lost when the ball enters)
      * orange / green circle      = steal radius around the obstacle (green once the ball is outside = stolen latch)
      * white sphere + grey trace  = ball and its trajectory this episode; white ray = ball velocity
      * yellow sphere              = ball turns yellow while a foot touches it (contact sensor)
      * blue trace                 = robot xy trajectory of the current episode
    """
    st = env.unwrapped.command_manager.get_term("steal")
    robot = env.unwrapped.scene["robot"]
    ball = env.unwrapped.scene["soccer_ball"]
    cfg = st.cfg
    z = 0.02
    ep_steps = int(env.unwrapped.episode_length_buf[env_idx])
    base = robot.data.root_pos_w[env_idx, :3].detach().cpu().numpy().astype(np.float32)
    here = np.array([base[0], base[1], z], dtype=np.float32)

    # -- outcome of the step that just finished (dones are computed before the reset)
    tm = env.unwrapped.termination_manager
    done_name = None
    for name, label in (("ball_stolen", "stolen"), ("ball_lost", "lost"), ("ball_no_foot", "no_foot"), ("base_contact", "fell"), ("base_height", "fell")):
        try:
            if bool(tm.get_term(name)[env_idx]):
                done_name = label
                break
        except Exception:
            pass
    if done_name is not None:
        gui["st_counts"][("stolen", "lost", "no_foot", "fell").index(done_name)] += 1
        gui["st_last"].value = done_name
        gui["st_cnt"].value = " / ".join(str(c) for c in gui["st_counts"])
        gui["st_trail"].clear()
        gui["st_ball_trail"].clear()

    # -- obstacle cylinder + contact threshold + steal radius
    op = st.obstacle.data.root_pos_w[env_idx, :3].detach().cpu().numpy().astype(np.float32)
    r_obs = float(cfg.obstacle_radius)
    cyl = trimesh.creation.cylinder(radius=r_obs, height=float(cfg.obstacle_height), sections=24)
    cyl.visual.face_colors = (200, 50, 50, 255)
    server.scene.add_mesh_trimesh("/steal/obstacle", cyl, position=op)
    th = np.linspace(0.0, 2.0 * math.pi, 49)
    ring = lambda r: np.stack([op[0] + r * np.cos(th), op[1] + r * np.sin(th), np.full_like(th, z)], axis=1).astype(np.float32)
    _add_polyline(server, "/steal/lost_ring", ring(r_obs + float(cfg.ball_radius) + float(cfg.touch_margin)), (230, 40, 40), 2.0)
    latched = float(st.stolen_at[env_idx]) >= 0.0
    _add_polyline(server, "/steal/steal_ring", ring(float(cfg.steal_radius)), (0, 220, 60) if latched else (255, 160, 0), 3.0)

    # -- ball (yellow while a foot touches it) + velocity ray + trail
    bp = ball.data.root_pos_w[env_idx, :3].detach().cpu().numpy().astype(np.float32)
    bv = ball.data.root_lin_vel_w[env_idx, :3].detach().cpu().numpy().astype(np.float32)
    touching = bool(st.foot_contact_now()[env_idx])
    server.scene.add_icosphere("/steal/ball", radius=float(cfg.ball_radius), color=(255, 230, 0) if touching else (245, 245, 245), position=bp)
    _add_polyline(server, "/steal/ball_vel", np.stack([bp, bp + bv * 0.3]), (245, 245, 245), 3.0)
    trail = gui["st_ball_trail"]
    if ep_steps <= 1:
        trail.clear()
    trail.append(np.array([bp[0], bp[1], z], dtype=np.float32))
    if len(trail) > 1000:
        del trail[0]
    if len(trail) >= 2:
        _add_polyline(server, "/steal/ball_trail", np.stack(trail), (160, 160, 160), 2.0)

    # -- robot trail
    rtrail = gui["st_trail"]
    if ep_steps <= 1:
        rtrail.clear()
    rtrail.append(here)
    if len(rtrail) > 1000:
        del rtrail[0]
    if len(rtrail) >= 2:
        _add_polyline(server, "/steal/trail", np.stack(rtrail), (60, 130, 255), 3.0)

    # -- readouts
    if latched:
        state = f"hold {float(st.elapsed()[env_idx]) - float(st.stolen_at[env_idx]):.1f}/{float(cfg.hold_s):.1f} s"
    elif bool(st.foot_touched[env_idx]):
        state = "touched"
    else:
        state = "approach"
    server.scene.add_label(
        "/steal/label",
        f"{state} | ball-obstacle {float(st.ball_obstacle_dist()[env_idx]):.2f} m | last: {gui['st_last'].value}",
        position=np.array([base[0], base[1], base[2] + 0.6], dtype=np.float32),
    )
    gui["st_state"].value = state
    gui["st_body"].value = round(float(st.body_ball_dist()[env_idx]), 2)
    gui["st_foot"].value = round(float(st.foot_ball_dist()[env_idx]), 2)
    gui["st_br"].value = round(float(st.ball_obstacle_dist()[env_idx]), 2)
    gui["st_away"].value = round(float(st.ball_away_speed()[env_idx]), 2)
    gui["st_ep"].value = round(ep_steps * env.unwrapped.step_dt, 2)
    cmd = env.unwrapped.command_manager.get_command("base_velocity")[env_idx].tolist()
    gui["st_cmd"].value = f"{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}"
    res = env.unwrapped.action_manager.get_term("joint_residual").raw_actions[env_idx]
    gui["st_res"].value = round(float(res.abs().max()), 3)


def override_command_from_viser(env, gui):
    """Overwrite the ``base_velocity`` command tensor with viser GUI values.

    Also disables the heading-based ang_vel_z recomputation and the standing-env
    zeroing so the GUI values survive the next ``_update_command`` call.
    """
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")

    # WaypointCommand (Isaac-Waypoint-K1): base_velocity is the policy output (walk_command
    # action), so the sliders must not touch it. Only the path / mode are driven from the GUI.
    if gui.get("waypoint"):
        wp = env.unwrapped.command_manager.get_term("waypoints")
        if gui["wp_pending"]["resample"]:
            gui["wp_pending"]["resample"] = False
            wp.time_left[:] = 0.0  # resampled on the next command_manager.compute()
            gui["wp_trail"].clear()
        mode = gui["wp_mode"].value
        if mode != "random":
            # pinned every step: _resample_command re-randomizes pass_radius at reset
            wp.pass_radius[:] = wp.cfg.loose_radius if mode == "fast" else wp.cfg.strict_radius
        return

    # BallLaunchCommand (Isaac-Block-K1): base_velocity is the policy output, so the sliders
    # must not touch it. "new ball" re-launches from the current pose (goal line moves with it).
    if gui.get("block"):
        if gui["bl_pending"]["relaunch"]:
            gui["bl_pending"]["relaunch"] = False
            bl = env.unwrapped.command_manager.get_term("ball_launch")
            bl._resample_command(torch.tensor([gui.get("bl_env_idx", 0)], device=env.unwrapped.device))
            gui["bl_trail"].clear()
            gui["bl_ball_trail"].clear()
        return

    # StealCommand (Isaac-Steal-K1): base_velocity is the policy output, so the sliders
    # must not touch it. "new placement" re-places the obstacle and the ball around the current pose.
    if gui.get("steal"):
        if gui["st_pending"]["replace"]:
            gui["st_pending"]["replace"] = False
            st = env.unwrapped.command_manager.get_term("steal")
            st._resample_command(torch.tensor([gui.get("st_env_idx", 0)], device=env.unwrapped.device))
            gui["st_trail"].clear()
            gui["st_ball_trail"].clear()
        return

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
    if server is not None and gui is not None and gui.get("waypoint"):
        _update_waypoint_viser(env, server, gui, env_idx=env_idx)
    if server is not None and gui is not None and gui.get("block"):
        gui["bl_env_idx"] = env_idx
        _update_block_viser(env, server, gui, env_idx=env_idx)
    if server is not None and gui is not None and gui.get("steal"):
        gui["st_env_idx"] = env_idx
        _update_steal_viser(env, server, gui, env_idx=env_idx)


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
    # マルチ expert (歩行 ⇄ 回転の遷移学習) の場合は、学習方策と凍結 expert を env の
    # モードで振り分けた実行アクションを使う (--frozen_ckpt で凍結側を渡すこと)。
    # ゲート方策 (GatingPPO) も同じフックで、凍結 expert を混合した 12 次元アクションを返す。
    if hasattr(runner.alg, "act_inference"):
        frozen = getattr(runner.alg, "frozen", None)
        if frozen is not None:
            for expert in frozen.values():
                expert.to(env.unwrapped.device)
        policy = runner.alg.act_inference

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
    if getattr(policy_nn, "is_gating", False):
        # ゲート方策 (凍結 expert 2 本 + ゲート MLP) の ONNX 化は未対応。
        print("[INFO] Gating policy: skipping JIT/ONNX export (not supported yet).")
    elif getattr(policy_nn, "is_hierarchical", False):
        # 上位コマンド方策 (凍結歩行 expert を内包) の ONNX 化は未対応。
        print("[INFO] Hierarchical (waypoint) policy: skipping JIT/ONNX export (not supported yet).")
    elif is_history_policy(policy_nn):
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