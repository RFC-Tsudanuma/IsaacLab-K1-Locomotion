"""RSL-RL の export_policy_as_onnx で書き出された policy.onnx を使って推論するサンプル。

- ONNX 入力: "obs" shape=(N, 79), float32 (生の観測。正規化器はモデル内部に焼き込み済み)
- ONNX 出力: "actions" shape=(N, 22), float32 (生のアクション。スケール・オフセット未適用)

観測ベクトル (79 次元) の構成 (rough_env_cfg.py の K1PolicyCfg と一致):
  [0:3]    base_ang_vel        ボディ系での角速度 (rad/s)
  [3:6]    projected_gravity   ボディ系に射影した重力ベクトル (単位ベクトル)
  [6:9]    velocity_commands   [lin_vel_x, lin_vel_y, ang_vel_z] のコマンド
  [9:31]   joint_pos_rel       q - q_default  (JOINT_NAMES_K1 の順, 22 次元)
  [31:53]  joint_vel_rel       dq             (JOINT_NAMES_K1 の順, 22 次元)
  [53:75]  last_action         前ステップの ONNX 出力 (生値, 22 次元)
  [75:79]  gait_phase          [sin(φL), cos(φL), sin(φR), cos(φR)],  φL = 2π·f·t,  φR = φL + π

アクション → 関節位置目標値:
  target_q[i] = q_default[i] + 0.5 * action[i]   (scale=0.5, use_default_offset=True)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


# play.py と同じ既定 URDF
DEFAULT_VISER_URDF = str(
    Path(__file__).resolve().parent
    / "../../assets_soccer/booster_robotics_robots/K1/K1_22dof.urdf"
)


# JointPositionActionCfg(scale=0.5, use_default_offset=True)
ACTION_SCALE: float = 0.5

# 歩行位相の周波数 (Hz)。rough_env_cfg.py の phase_obs(phase_freq=2.0) と同じ。
PHASE_FREQ: float = 2.0

# JOINT_NAMES_K1 (velocity_env_cfg.py と同じ順序を厳守)
JOINT_NAMES_K1: tuple[str, ...] = (
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
NUM_JOINTS = len(JOINT_NAMES_K1)  # 22

# K1_LOCOMOTION_CFG.init_state.joint_pos と同じ値 (JOINT_NAMES_K1 の順)
_SHOULDER_ROLL = 0.7853981634 * 1.75  # ≈ 1.3744467860
DEFAULT_JOINT_POS: np.ndarray = np.array(
    [
        0.0, 0.0,                                  # AAHead_yaw, Head_pitch
        0.0, -_SHOULDER_ROLL, 0.0, 0.0,            # 左肩・肘
        0.0,  _SHOULDER_ROLL, 0.0, 0.0,            # 右肩・肘
        -0.26, 0.0, 0.0,                           # 左股 (Pitch, Roll, Yaw)
        0.52, -0.26, 0.0,                          # 左膝・足首 (Knee_Pitch, Ankle_Pitch, Ankle_Roll)
        -0.26, 0.0, 0.0,                           # 右股
        0.52, -0.26, 0.0,                          # 右膝・足首
    ],
    dtype=np.float32,
)
assert DEFAULT_JOINT_POS.shape == (NUM_JOINTS,)


class K1OnnxPolicy:
    """policy.onnx をラップして 1 ステップ分の推論を行う。"""

    def __init__(self, onnx_path: str | Path, providers: list[str] | None = None) -> None:
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)

        # 期待する入出力の検証
        input_names = [i.name for i in self.session.get_inputs()]
        output_names = [o.name for o in self.session.get_outputs()]
        if "obs" not in input_names:
            raise RuntimeError(f"ONNX に 'obs' 入力が見つかりません (got {input_names})")
        if "actions" not in output_names:
            raise RuntimeError(f"ONNX に 'actions' 出力が見つかりません (got {output_names})")
        # LSTM/GRU 版にも対応したい場合は h_in/c_in のハンドリングを追加すること。
        if len(input_names) > 1:
            raise NotImplementedError(
                f"再帰モデル (入力 {input_names}) には未対応。MLP ポリシーのみサポート。"
            )

        self.last_action: np.ndarray = np.zeros(NUM_JOINTS, dtype=np.float32)

    @staticmethod
    def gait_phase(t: float, freq: float = PHASE_FREQ) -> np.ndarray:
        phi_l = 2.0 * math.pi * freq * t
        phi_r = phi_l + math.pi
        return np.array(
            [math.sin(phi_l), math.cos(phi_l), math.sin(phi_r), math.cos(phi_r)],
            dtype=np.float32,
        )

    def build_observation(
        self,
        base_ang_vel: np.ndarray,        # (3,)
        projected_gravity: np.ndarray,    # (3,)
        velocity_command: np.ndarray,     # (3,) [vx, vy, wz]
        joint_pos: np.ndarray,            # (22,) 現在角 [rad]
        joint_vel: np.ndarray,            # (22,) 角速度 [rad/s]
        episode_time: float,              # エピソード開始からの経過秒
    ) -> np.ndarray:
        """生の観測を 79 次元ベクトルに組み立てる。正規化はモデル内で行うので不要。"""
        joint_pos_rel = joint_pos.astype(np.float32) - DEFAULT_JOINT_POS
        joint_vel_f = joint_vel.astype(np.float32)
        phase = self.gait_phase(episode_time)

        obs = np.concatenate(
            [
                base_ang_vel.astype(np.float32).reshape(3),
                projected_gravity.astype(np.float32).reshape(3),
                velocity_command.astype(np.float32).reshape(3),
                joint_pos_rel.reshape(NUM_JOINTS),
                joint_vel_f.reshape(NUM_JOINTS),
                self.last_action.reshape(NUM_JOINTS),
                phase,
            ],
            dtype=np.float32,
        )
        assert obs.shape == (79,), f"obs shape mismatch: {obs.shape}"
        return obs

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """obs (79,) を入力して raw action (22,) を返す。last_action も更新する。"""
        if obs.ndim == 1:
            obs_batched = obs[None, :].astype(np.float32)
        else:
            obs_batched = obs.astype(np.float32)
        action = self.session.run(["actions"], {"obs": obs_batched})[0]  # (1, 22)
        action = action[0]
        self.last_action = action.astype(np.float32)
        return action

    @staticmethod
    def action_to_target_joint_pos(action: np.ndarray) -> np.ndarray:
        """ONNX の生アクションを実際の関節位置目標 [rad] に変換。"""
        return DEFAULT_JOINT_POS + ACTION_SCALE * action.astype(np.float32)

    def step(
        self,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        velocity_command: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        episode_time: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """センサ値を入れて (raw_action, target_joint_pos) を返す高水準 API。"""
        obs = self.build_observation(
            base_ang_vel, projected_gravity, velocity_command,
            joint_pos, joint_vel, episode_time,
        )
        action = self.infer(obs)
        target = self.action_to_target_joint_pos(action)
        return action, target

    def reset(self) -> None:
        self.last_action[:] = 0.0


def _setup_viser(urdf_path: str, port: int):
    """viser サーバを立ち上げて URDF を読み込む。

    返り値の `joint_indices[k]` は URDF の actuated joint 名を JOINT_NAMES_K1 上の
    インデックスに対応付けたもの。一致しない関節は ``None``。
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
    joint_indices: list[int | None] = []
    for name in urdf_joint_names:
        if name in JOINT_NAMES_K1:
            joint_indices.append(JOINT_NAMES_K1.index(name))
        else:
            print(f"[WARNING] Viser: joint '{name}' は JOINT_NAMES_K1 に無いので 0.0 を割当てます。")
            joint_indices.append(None)
    print(f"[INFO] Viser visualization available at http://localhost:{port}")
    return server, base_frame, viser_urdf, joint_indices


def _add_command_sliders(server) -> dict:
    """vx/vy/wz とリセットボタンを GUI に追加。"""
    handles = {}
    with server.gui.add_folder("Velocity Command"):
        handles["vx"] = server.gui.add_slider("lin_vel_x [m/s]", min=-1.0, max=1.0, step=0.05, initial_value=0.5)
        handles["vy"] = server.gui.add_slider("lin_vel_y [m/s]", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
        handles["wz"] = server.gui.add_slider("ang_vel_z [rad/s]", min=-1.5, max=1.5, step=0.05, initial_value=0.0)
        handles["reset"] = server.gui.add_button("Reset")
    return handles


def _push_to_viser(base_frame, viser_urdf, joint_indices, joint_pos: np.ndarray) -> None:
    cfg = np.array(
        [joint_pos[i] if i is not None else 0.0 for i in joint_indices],
        dtype=np.float32,
    )
    # 物理シミュなしの open-loop 可視化なので、ベースは原点固定。
    base_frame.position = np.zeros(3, dtype=np.float32)
    base_frame.wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    viser_urdf.update_cfg(cfg)


def _run(
    onnx_path: str,
    num_steps: int,
    dt: float,
    use_viser: bool,
    viser_urdf_path: str,
    viser_port: int,
    real_time: bool,
    track_alpha: float,
) -> None:
    """ポリシーを走らせるメインループ。

    物理シミュレータを持たないため、可視化は擬似的な一次遅れ追従モデル:
      - ベースは原点固定 (角速度・重力は直立想定で固定値)
      - joint_pos += track_alpha * (target - joint_pos)
        (track_alpha=1.0 で target を即時反映、0 に近いほど鈍く追従)
      - joint_vel は差分から計算
    PD 制御の物理を入れていないため、policy が想定する閉ループとは異なる点に注意。
    """
    policy = K1OnnxPolicy(onnx_path)
    policy.reset()

    base_ang_vel = np.zeros(3, dtype=np.float32)
    projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    velocity_command = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    joint_pos = DEFAULT_JOINT_POS.copy()
    joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)

    print(f"[INFO] ONNX policy loaded from: {onnx_path}")

    viser_state = None
    cmd_handles = None
    if use_viser:
        viser_state = _setup_viser(viser_urdf_path, viser_port)
        cmd_handles = _add_command_sliders(viser_state[0])
        # 起動直後にデフォルト姿勢を一度反映
        _push_to_viser(viser_state[1], viser_state[2], viser_state[3], DEFAULT_JOINT_POS)

    print(f"[INFO] Running {num_steps if num_steps > 0 else 'infinite'} steps (dt={dt:.4f} s)")

    step = 0
    episode_time = 0.0
    try:
        while True:
            if num_steps > 0 and step >= num_steps:
                break
            loop_start = time.time()

            # GUI からコマンド更新
            if cmd_handles is not None:
                velocity_command[0] = float(cmd_handles["vx"].value)
                velocity_command[1] = float(cmd_handles["vy"].value)
                velocity_command[2] = float(cmd_handles["wz"].value)
                if cmd_handles["reset"].value:
                    cmd_handles["reset"].value = False
                    policy.reset()
                    joint_pos = DEFAULT_JOINT_POS.copy()
                    joint_vel[:] = 0.0
                    episode_time = 0.0
                    step = 0
                    continue

            action, target = policy.step(
                base_ang_vel, projected_gravity, velocity_command,
                joint_pos, joint_vel, episode_time=episode_time,
            )

            if not use_viser and (step < 3 or step == num_steps - 1):
                print(f"step={step:4d}  t={episode_time:6.3f}s  "
                      f"|a|_inf={np.abs(action).max():.3f}  "
                      f"target_q [{target.min():+.3f}, {target.max():+.3f}] rad")

            # 可視化に流し込む
            if viser_state is not None:
                try:
                    _push_to_viser(viser_state[1], viser_state[2], viser_state[3], target)
                except Exception as e:
                    print(f"[WARNING] Viser update failed: {e}")

            # open-loop で次ステップの状態を更新
            joint_vel = (target - joint_pos) / dt
            joint_pos = target.copy()
            episode_time += dt
            step += 1

            if real_time:
                sleep_time = dt - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")


def main() -> None:
    parser = argparse.ArgumentParser(description="K1 ONNX policy inference (with optional viser viz)")
    parser.add_argument(
        "--onnx", type=str, required=True,
        help="logs/.../exported/policy.onnx へのパス",
    )
    parser.add_argument("--steps", type=int, default=0,
                        help="走らせるステップ数 (0 で無限ループ。viser 利用時は 0 推奨)")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="制御周期 [s] (env.step_dt と一致させる)")
    parser.add_argument("--viser", action="store_true", default=False,
                        help="viser ベースの可視化を有効化する")
    parser.add_argument("--viser_port", type=int, default=8080, help="viser サーバのポート")
    parser.add_argument("--viser_urdf", type=str, default=None,
                        help=f"可視化に使う URDF (既定: {DEFAULT_VISER_URDF})")
    parser.add_argument("--real-time", dest="real_time", action="store_true", default=False,
                        help="dt に合わせてリアルタイム再生 (viser 時は基本オン推奨)")
    args = parser.parse_args()

    urdf_path = args.viser_urdf or DEFAULT_VISER_URDF
    real_time = args.real_time or args.viser  # viser 時は既定でリアルタイム
    _run(
        onnx_path=args.onnx,
        num_steps=args.steps,
        dt=args.dt,
        use_viser=args.viser,
        viser_urdf_path=urdf_path,
        viser_port=args.viser_port,
        real_time=real_time,
    )


if __name__ == "__main__":
    main()
