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
parser.add_argument(
    "--kick_speed",
    type=float,
    default=None,
    help=(
        "Fix the commanded ball speed [m/s] for walk_kick family tasks "
        "(walk_kick / walk_pass / walk_loop_pass / walk_loop_shoot). Without this the speed is sampled "
        "per episode from the task's target_speed_range."
    ),
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
from isaaclab_k1_locomotion.tasks.manager_based.locomotion.networks import (
    ActorCriticHistoryCNN,
    export_history_policy_as_jit,
    export_history_policy_as_onnx,
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


def log_terminations(env, step: int, counts: dict, episode_start: torch.Tensor) -> None:
    """Print which termination term fired this step, for which env, and after how long.

    ``termination_manager._term_dones`` is only overwritten by ``compute()`` and is not
    cleared by ``_reset_idx``, so it still holds this step's values once ``env.step()``
    returns. Several terms can fire for the same env on the same step (e.g. a robot that
    falls exactly on the timeout), so every active term is checked rather than just the first.
    """
    term_mgr = getattr(env.unwrapped, "termination_manager", None)
    if term_mgr is None:  # direct-workflow envs have no termination manager
        return

    for name in term_mgr.active_terms:
        fired = term_mgr.get_term(name).nonzero(as_tuple=False).squeeze(-1)
        if fired.numel() == 0:
            continue
        counts[name] = counts.get(name, 0) + int(fired.numel())
        for env_idx in fired.tolist():
            duration = step - int(episode_start[env_idx].item())
            print(f"[TERM] step {step:6d} | env {env_idx:3d} | {name:16s} | episode {duration:4d} steps")
        episode_start[fired] = step


# キック検出フラグ (walk_long_pass_flag 系) の累計成績。
# 「エピソード平均」ではなく **セッション全体の全ステップ** を通した累計にしている。
# CommandTerm 側の flag_accuracy はエピソード内の走査平均で、しかも _reset_idx の直後に
# 累積器が巻き戻るため、play のループから読むと終了直後の env が 0 に見えてしまう
# (order: termination -> reward -> _reset_idx -> command_manager.compute -> observation)。
# ここで生の値から積み直す方が素直で、読み方も「何割当たったか」そのものになる。
_FLAG_STATS = {"ok": 0.0, "n": 0.0, "pre_sum": 0.0, "pre_n": 0.0, "post_sum": 0.0, "post_n": 0.0}
_FLAG_ACTION_NAME = "kick_flag"


def _accumulate_flag_stats(env, state) -> bool:
    """フラグ出力の正誤を積む。フラグ行動を持たないタスクでは False を返して何もしない。"""
    action_mgr = env.unwrapped.action_manager
    if _FLAG_ACTION_NAME not in action_mgr.active_terms:
        return False

    p = torch.sigmoid(action_mgr.get_term(_FLAG_ACTION_NAME).raw_actions[:, 0])
    y = state["kick_done"].float()
    post = y > 0.5
    correct = (p > 0.5) == post

    _FLAG_STATS["ok"] += float(correct.sum().item())
    _FLAG_STATS["n"] += float(correct.numel())
    _FLAG_STATS["pre_sum"] += float(p[~post].sum().item())
    _FLAG_STATS["pre_n"] += float((~post).sum().item())
    _FLAG_STATS["post_sum"] += float(p[post].sum().item())
    _FLAG_STATS["post_n"] += float(post.sum().item())
    return True


def log_flag_metrics(step: int) -> None:
    """フラグ検出の累計成績を出す。``[FLAG]`` 行。"""
    if _FLAG_STATS["n"] == 0:
        return
    acc = _FLAG_STATS["ok"] / _FLAG_STATS["n"]
    pre = _FLAG_STATS["pre_sum"] / max(_FLAG_STATS["pre_n"], 1.0)
    post = _FLAG_STATS["post_sum"] / max(_FLAG_STATS["post_n"], 1.0)
    print(
        f"[FLAG] step {step:6d} | accuracy {acc:5.3f} "
        f"| pre-latch p {pre:5.3f} (低いほど良い) | post-latch p {post:5.3f} (高いほど良い) "
        f"| {int(_FLAG_STATS['n']):d} env-steps 累計"
    )


def log_kick_metrics(env, step: int, every: int = 100) -> None:
    """Print kick diagnostics periodically (walk_kick family tasks only).

    Metrics such as ``sole_height_at_kick`` are computed every step by the command
    manager, but only the training loop writes them to TensorBoard. This prints the
    same quantities during play, straight from the shared latch state, so a policy can
    be diagnosed without launching a training run.

    Values are averaged over the environments whose kick has latched.

    ``kick_flag`` 行動を持つタスク (walk_long_pass_flag) では、フラグ検出の累計成績を
    ``[FLAG]`` 行として併せて出す。累計なので毎ステップ積む必要があり、``every`` の
    間引きより前に集計している。
    """
    state = getattr(env.unwrapped, "_kick_latch_state", None)
    if state is None:  # not a walk_kick family task
        return

    has_flag = _accumulate_flag_stats(env, state)

    if step % every != 0:
        return
    if has_flag:
        log_flag_metrics(step)

    done = state["kick_done"]
    n_kicked = int(done.sum().item())
    num_envs = env.unwrapped.num_envs
    touches = float(state["touch_count"].mean().item())
    if n_kicked == 0:
        print(f"[KICK] step {step:6d} | kicked   0/{num_envs:3d} | touches {touches:4.2f}")
        return

    mask = done.float()

    def avg(key: str) -> float:
        return float((state[key] * mask).sum().item()) / n_kicked

    sole_cm = avg("sole_height_at_kick") * 100.0
    phi_deg = math.degrees(avg("phi_frozen"))
    dir_deg = math.degrees(avg("tau_direction_frozen"))
    # 符号付き方向誤差 (正 = ボールが指令より右) と右足で蹴った割合。
    # dir が小さくても dirs が同じ大きさなら「誤差ではなく片側バイアス」。
    # Rfoot が 0/1 に張り付いていれば片足でしか蹴っていない。
    dirs_deg = math.degrees(avg("tau_signed_frozen"))
    r_foot = avg("kick_foot_frozen")
    v3d = avg("v_ball_3d_frozen")
    # apex は接地時のボール中心 (= 半径) を引いて「浮き」に直す
    ball_radius = float(env.unwrapped.scene["soccer_ball"].cfg.spawn.radius)
    loft_cm = (avg("apex_height") - ball_radius) * 100.0
    print(
        f"[KICK] step {step:6d} | kicked {n_kicked:3d}/{num_envs:3d} "
        f"| sole {sole_cm:5.1f}cm | phi {phi_deg:5.1f}deg "
        f"| dir {dir_deg:5.1f}deg (signed {dirs_deg:+6.1f}) | Rfoot {r_foot:4.2f} "
        f"| v3d {v3d:4.2f}m/s | loft {loft_cm:5.1f}cm | touches {touches:4.2f}"
    )


def print_termination_summary(counts: dict) -> None:
    """Print how often each termination term fired over the whole play session."""
    total = sum(counts.values())
    print("\n[TERM] ---- termination summary ----")
    if total == 0:
        print("[TERM] no terminations recorded")
        return
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"[TERM] {name:16s} {n:6d}  ({100.0 * n / total:5.1f}%)")
    print(f"[TERM] {'total':16s} {total:6d}")


def _apply_kick_speed(env_cfg, speed: float) -> None:
    """指令ボール速度を単一値に固定する (walk_kick / walk_pass / walk_loop_pass / walk_loop_shoot 用)。

    キック速度はエピソードごとに ``target_speed_range`` から一様サンプリングされるので、
    PLAY で「この強さのキックを見たい」ときはレンジを (v, v) に潰す。

    NOTE: 値 latch のトリガー速度 (v_thresh) は下回れない。v_thresh 未満の指令を与えると
          ``kick_state`` の latch が永久に発火せず、キック報酬が全て 0 のまま
          ``kick_finished`` も出ずに time_out まで走る (= 蹴ったように見えない)。
          タスクごとの既定は walk_kick/walk_loop_* が 0.8、walk_pass が 0.40。
          ここでは黙って値を書き換えず、警告だけ出して指定値をそのまま通す。
    """
    cmd = getattr(env_cfg.commands, "kick_direction", None)
    if cmd is None:
        raise ValueError(
            f"--kick_speed is only supported for walk_kick family tasks "
            f"(the task '{args_cli.task}' has no 'kick_direction' command)."
        )

    old = cmd.target_speed_range
    cmd.target_speed_range = (speed, speed)
    print(f"[INFO] kick speed: {old} -> ({speed}, {speed}) [m/s]")

    # v_thresh は kick_state を参照する全項に同じ値が配られている。代表として
    # termination 側 (常に有効な項) から読む。
    done_term = getattr(env_cfg.terminations, "kick_finished", None)
    v_thresh = done_term.params.get("v_thresh") if done_term is not None else None
    if v_thresh is not None and speed <= v_thresh:
        print(
            f"[WARN] --kick_speed {speed} is at or below the latch threshold "
            f"v_thresh={v_thresh}. The kick will likely never latch, so the episode "
            f"will run to time_out and no kick reward/termination will fire."
        )


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

    if args_cli.kick_speed is not None:
        _apply_kick_speed(env_cfg, args_cli.kick_speed)

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
    run_dir = os.path.dirname(resume_path)
    export_model_dir = os.path.join(run_dir, "exported")
    onnx_filename = f"{agent_cfg.experiment_name}_{os.path.basename(run_dir)}.onnx"
    if isinstance(policy_nn, ActorCriticHistoryCNN):
        # 観測履歴を見る actor は nn.Sequential ではないので標準 exporter が使えない
        # (actor[0].in_features を見に行く)。入力は (1, history, obs_dim)。
        export_history_policy_as_jit(policy_nn, path=export_model_dir, filename="policy.pt")
        export_history_policy_as_onnx(policy_nn, path=export_model_dir, filename=onnx_filename)
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename=onnx_filename)
    print(f"[INFO]: Exported policy to: {os.path.join(export_model_dir, onnx_filename)}")

    dt = env.unwrapped.step_dt

    # set up viser visualization (optional)
    viser_state = None
    if args_cli.viser:
        urdf_path = args_cli.viser_urdf or DEFAULT_VISER_URDF
        viser_state = setup_viser(env, urdf_path, args_cli.viser_port)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # termination logging: cumulative per-term counts + the step each env's episode started on
    term_counts: dict[str, int] = {}
    step_count = 0
    episode_start = torch.zeros(env.unwrapped.num_envs, dtype=torch.long)
    # manual frame buffer for robust video recording
    _manual_frames = [] if args_cli.video else None
    # simulate environment
    try:
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
            step_count += 1
            log_terminations(env, step_count, term_counts, episode_start)
            log_kick_metrics(env, step_count)
            if viser_state is not None:
                try:
                    _, base_frame, viser_urdf, joint_indices, _ = viser_state
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
    finally:
        # the play loop normally runs until interrupted, so print the summary on the way out
        print_termination_summary(term_counts)
        # フラグ検出タスクなら最終成績も出す (Ctrl-C で抜けたときが本番)
        if _FLAG_STATS["n"] > 0:
            print("\n[FLAG] ---- kick flag summary ----")
            log_flag_metrics(step_count)

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
