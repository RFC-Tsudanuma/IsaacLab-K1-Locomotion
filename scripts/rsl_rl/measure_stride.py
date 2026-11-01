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

    print(f"[INFO] Viser visualization available at http://localhost:{port}")
    return server, base_frame, viser_urdf, joint_indices, gui


def override_command_from_viser(env, gui):
    """Overwrite the ``base_velocity`` command tensor with viser GUI values.

    Also disables the heading-based ang_vel_z recomputation and the standing-env
    zeroing so the GUI values survive the next ``_update_command`` call.
    """
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
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

    # ------------------------------------------------------------------
    # 歩幅計測 (2026-08-25): 固定前進コマンドで足位置・接地・base 速度を直接記録し、
    # Metrics/base_velocity/stride_touchdown の妥当性と歩容の実態 (歩幅/ケイデンス/滑り)
    # を確認する。--speeds で与えた vx を env に巡回割り当てする。
    # ------------------------------------------------------------------
    uenv = env.unwrapped
    dev = uenv.device
    N = uenv.num_envs
    speeds = [float(s) for s in os.environ.get("MEASURE_SPEEDS", "0.5,1.0,1.5,1.8").split(",")]
    steps_total = int(os.environ.get("MEASURE_STEPS", "600"))
    warmup = int(os.environ.get("MEASURE_WARMUP", "150"))
    sp = torch.tensor(speeds, device=dev)
    group = torch.arange(N, device=dev) % len(speeds)
    fixed = torch.zeros(N, 3, device=dev)
    fixed[:, 0] = sp[group]
    cmd_term = uenv.command_manager.get_term("base_velocity")
    robot = uenv.scene["robot"]
    sensor = uenv.scene.sensors["contact_forces"]
    sid, names = sensor.find_bodies(".*_foot_link")
    rid = [robot.find_bodies(n)[0][0] for n in names]
    from isaaclab.utils.math import yaw_quat, quat_apply_inverse

    z = lambda: torch.zeros(N, device=dev)
    stride_sum, stride_cnt, vel_sum, vel_cnt, gap_absmax, air_sum = z(), z(), z(), z(), z(), z()
    slide_sum, slide_cnt = z(), z()
    fly_sum = z()  # 両足同時空中だったステップ数 (odometry 要件: 0 が理想)
    obs = env.get_observations()
    for step in range(steps_total):
        cmd_term.vel_command_b[:] = fixed
        cmd_term.is_heading_env[:] = False
        cmd_term.is_standing_env[:] = False
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
        if step < warmup:
            continue
        valid = uenv.episode_length_buf > 50
        base_pos = robot.data.root_pos_w[:, :3]
        qy = yaw_quat(robot.data.root_quat_w)
        rel = [quat_apply_inverse(qy, robot.data.body_pos_w[:, i, :3] - base_pos) for i in rid]
        gap_x = rel[0][:, 0] - rel[1][:, 0]
        gap_absmax = torch.maximum(gap_absmax, gap_x.abs() * valid)
        first = sensor.compute_first_contact(dt)[:, sid]
        first = first & (sensor.data.last_air_time[:, sid] > 0.1)
        hit = (first[:, 0] | first[:, 1]) & valid
        stride = torch.where(first[:, 0], gap_x, -gap_x)
        stride_sum += torch.where(hit, stride, z())
        stride_cnt += hit.float()
        air_sum += torch.where(hit, sensor.data.last_air_time[:, sid].max(dim=1)[0], z())
        vel_sum += robot.data.root_lin_vel_b[:, 0] * valid
        vel_cnt += valid.float()
        in_contact = sensor.data.net_forces_w_history[:, :, sid, :].norm(dim=-1).max(dim=1)[0] > 1.0
        foot_v = torch.stack([robot.data.body_lin_vel_w[:, i, :2].norm(dim=1) for i in rid], dim=1)
        slide_sum += (foot_v * in_contact).sum(dim=1) * valid
        slide_cnt += in_contact.sum(dim=1).float() * valid
        fly_sum += (~in_contact.any(dim=1)).float() * valid
    T = (steps_total - warmup) * dt
    print("\n=== stride measurement (per commanded vx) ===")
    print("vx_cmd  vx_act  stride_td  gap_amp  cadence[Hz]  swing[s]  slide[m/s]  fly%  n_td")
    for gi, s in enumerate(speeds):
        m = group == gi
        n_td = stride_cnt[m].sum().item()
        st = (stride_sum[m].sum() / max(n_td, 1)).item()
        va = (vel_sum[m].sum() / vel_cnt[m].sum().clamp(min=1)).item()
        amp = gap_absmax[m].mean().item()
        cad = n_td / (m.sum().item() * T) / 2.0
        sw = (air_sum[m].sum() / max(n_td, 1)).item()
        sl = (slide_sum[m].sum() / slide_cnt[m].sum().clamp(min=1)).item()
        fly = (fly_sum[m].sum() / vel_cnt[m].sum().clamp(min=1)).item() * 100.0
        print(f"{s:6.2f}  {va:6.3f}  {st:8.3f}  {amp:7.3f}  {cad:10.2f}  {sw:7.3f}  {sl:9.3f}  {fly:4.1f}  {int(n_td)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
