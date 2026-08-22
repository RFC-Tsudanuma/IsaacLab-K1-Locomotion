# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play with an ONNX-exported RSL-RL policy in Isaac Sim (構造は play.py と同じ)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import export_naming  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with an ONNX policy in Isaac Sim.")
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
    "--onnx",
    type=str,
    default=None,
    help="ONNX モデルへのパス。未指定時は logs から自動解決し、.../exported/ の最新 *.onnx を読む。",
)
parser.add_argument(
    "--onnx_provider",
    type=str,
    default="cpu",
    choices=["cpu", "cuda"],
    help="onnxruntime の実行 provider。",
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
import onnxruntime as ort
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401


DEFAULT_VISER_URDF = str(
    Path(__file__).resolve().parent
    / "../../assets_soccer/booster_robotics_robots/K1/K1_22dof.urdf"
)


def setup_viser(env, urdf_path: str, port: int):
    """play.py と同じ viser セットアップ。"""
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


class OnnxPolicy:
    """ONNX session を runner.get_inference_policy() と同じ signature でラップする。

    入出力契約 (export_policy_as_onnx 由来):
        - 入力 "obs": (N, obs_dim) float32, 生の観測 (正規化はモデル内部に焼込み済み)
        - 出力 "actions": (N, num_actions) float32
    LSTM/GRU を含む再帰モデルには未対応 (h_in/c_in 入力があれば例外)。
    """

    def __init__(self, onnx_path: str, device: str, provider: str = "cpu") -> None:
        if provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.device = device

        input_names = [i.name for i in self.session.get_inputs()]
        output_names = [o.name for o in self.session.get_outputs()]
        if "obs" not in input_names or "actions" not in output_names:
            raise RuntimeError(
                f"想定外の ONNX I/O: inputs={input_names}, outputs={output_names}"
            )
        if len(input_names) > 1:
            raise NotImplementedError(
                f"再帰モデル (入力 {input_names}) には未対応。MLP ポリシーのみサポート。"
            )

        # 入力のバッチ次元が固定 (= 1 など) かどうかを検査。
        obs_input = self.session.get_inputs()[0]
        batch_dim = obs_input.shape[0] if obs_input.shape else None
        # int で具体的な値が入っていれば固定。文字列/None なら動的。
        self._fixed_batch = batch_dim if isinstance(batch_dim, int) else None
        if self._fixed_batch is not None and self._fixed_batch != 1:
            print(
                f"[WARNING] ONNX のバッチ次元が {self._fixed_batch} に固定されています。"
                " num_envs を一致させるか再エクスポートしてください。"
            )

    def __call__(self, obs) -> torch.Tensor:
        # RslRlVecEnvWrapper は TensorDict を返すので "policy" グループを取り出す。
        if not isinstance(obs, torch.Tensor):
            obs = obs["policy"]
        np_obs = obs.detach().cpu().numpy().astype(np.float32, copy=False)

        if self._fixed_batch == 1 and np_obs.shape[0] != 1:
            # バッチ固定 1 の ONNX では env 毎に逐次推論する。
            outs = [
                self.session.run(["actions"], {"obs": np_obs[i : i + 1]})[0]
                for i in range(np_obs.shape[0])
            ]
            np_act = np.concatenate(outs, axis=0)
        else:
            np_act = self.session.run(["actions"], {"obs": np_obs})[0]

        return torch.from_numpy(np_act).to(self.device)

    def reset(self, dones=None) -> None:  # MLP は内部状態なし
        pass


def _resolve_onnx_path(args_cli, agent_cfg) -> tuple[str, str]:
    """ONNX のパスとログディレクトリを返す。"""
    if args_cli.onnx:
        onnx_path = os.path.abspath(args_cli.onnx)
        # exported/ の親をログディレクトリ扱いに
        log_dir = os.path.dirname(os.path.dirname(onnx_path))
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
        # 成果物は固定名ではなく <experiment>_<checkpoint>_<時刻>.onnx になったので
        # (export_naming の docstring)、名前を決め打ちせず mtime が一番新しいものを拾う。
        export_dir = os.path.join(log_dir, "exported")
        found = export_naming.latest_artifact(export_dir, ".onnx")
        if found is None:
            raise FileNotFoundError(
                f"ONNX モデルが見つかりません: {export_dir}/*.onnx\n"
                "学習後 play.py か export_policy.py を 1 度走らせて exported/ に"
                " 書き出しておくか、--onnx <path> で明示指定してください。"
            )
        onnx_path = found
        print(f"[INFO] 最新の ONNX を使用します: {onnx_path}")

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX モデルが見つかりません: {onnx_path}")
    return onnx_path, log_dir


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with an ONNX policy."""
    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ONNX とログディレクトリの解決
    onnx_path, log_dir = _resolve_onnx_path(args_cli, agent_cfg)

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
            "video_folder": os.path.join(log_dir, "videos", "play_onnx"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading ONNX policy from: {onnx_path}")
    policy = OnnxPolicy(onnx_path, device=env.unwrapped.device, provider=args_cli.onnx_provider)

    dt = env.unwrapped.step_dt

    # set up viser visualization (optional)
    viser_state = None
    if args_cli.viser:
        urdf_path = args_cli.viser_urdf or DEFAULT_VISER_URDF
        viser_state = setup_viser(env, urdf_path, args_cli.viser_port)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    _manual_frames = [] if args_cli.video else None
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
        if viser_state is not None:
            try:
                _, base_frame, viser_urdf, joint_indices = viser_state
                update_viser(env, base_frame, viser_urdf, joint_indices, env_idx=args_cli.viser_env_idx)
            except Exception as e:
                print(f"[WARNING] Viser update failed: {e}")
        if args_cli.video:
            try:
                frame = env.unwrapped.render()
            except Exception:
                frame = None
            if frame is not None and hasattr(frame, "shape") and frame.size > 0:
                _manual_frames.append(frame.copy())
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Save video from manual frames if RecordVideo wrapper produced no output
    if args_cli.video and _manual_frames:
        video_dir = os.path.join(log_dir, "videos", "play_onnx")
        os.makedirs(video_dir, exist_ok=True)
        existing = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        if not existing:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

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
    main()
    simulation_app.close()
