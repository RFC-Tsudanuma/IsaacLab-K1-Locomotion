# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""首 (ヨー・ピッチ) 安定化タスク (Isaac-Neck-K1) の MDP 部品 (2026-11-20)。

設計は scripts/rsl_rl/neck_compensate.md。凍結した歩行 expert が歩いている上で、首 2 軸の
関節目標 ``q_neck = q_base + 残差`` の残差だけを学習する。

* :class:`NeckCommand` (コマンド項 ``neck``) が安定化器の本体。毎 step (50 Hz):
  遅延 + ノイズ付きの IMU / 首エンコーダ値を作る → 胴体姿勢 (roll, pitch, yaw) を歩行 1 周期分の
  移動平均で平滑化 (設計書 3.1 の第 1 候補。窓長は歩行周波数から決まる。yaw は遅れを補正) →
  首指令 ``q_cmd`` を生成 (一定 / ステップ / 見渡し / 凝視風、重複・欠落あり) →
  ``d_target = R_smooth · FK(q_cmd)``、``q_base = IK(R_body^T · d_target)``。
  コマンド値 (= 方策の観測, 17 次元) は ``[角速度 (3), 平滑化姿勢からの偏差 (3), 首関節角 (2),
  首関節角速度 (2), q_cmd (2), q_base (2), カメラ座標系の角速度 (3)]``。
* :class:`NeckAction` (action 項 ``neck``, 2 次元) が ``q_base + scale × 残差`` を関節目標にする。
  物理 step ごとに呼ばれるので、センサ遅延用の 1 物理 step 前の値とカメラ角速度の積算もここで行う。
* 報酬用の評価量 (``blur_speed`` / ``point_err``) は真値から `NeckCommand` が計算する。コマンド更新は
  報酬計算の後なので、**報酬は 1 step 遅れて** 渡る (リセット直後の step は 0)。

仮定: 光軸 = Head_2 の +x 軸 (首 0 で胴体の正面)。カメラの取付角が違っても、光軸に垂直な
角速度を見る報酬の性質はほぼ変わらない。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .events import compute_cmd_phase_freq

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

NECK_JOINTS = ["AAHead_yaw", "Head_pitch"]
NECK_DIM = 2
CAMERA_BODY = "Head_2"
NECK_OBS_DIM = 17
_SMOOTH_HISTORY = 32  # 平滑化の履歴長 [step]。歩行周波数の下限 1.75 Hz (窓 28.6 step) + 補間 1 点が入る


def neck_fk(q: torch.Tensor) -> torch.Tensor:
    """首関節角 (N, 2) [yaw, pitch] → 胴体座標系の光軸方向 (N, 3)。Rz(yaw)·Ry(pitch)·e_x。"""
    cy, sy, cp, sp = torch.cos(q[:, 0]), torch.sin(q[:, 0]), torch.cos(q[:, 1]), torch.sin(q[:, 1])
    return torch.stack([cy * cp, sy * cp, -sp], dim=1)


def neck_ik(d: torch.Tensor) -> torch.Tensor:
    """胴体座標系の光軸方向 (N, 3) → 首関節角 (N, 2)。`neck_fk` の逆 (光軸がヨー軸と平行な特異姿勢は除く)。"""
    return torch.stack([torch.atan2(d[:, 1], d[:, 0]), torch.atan2(-d[:, 2], torch.hypot(d[:, 0], d[:, 1]))], dim=1)


def camera_gyro(ang_vel_b: torch.Tensor, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    """胴体角速度 (胴体座標系) と首の関節角・角速度 → カメラ (Head_2) の角速度 (カメラ座標系, N×3)。
    y / z 成分が光軸の動き (ブラー)、x 成分は画像ロール。実機では IMU のジャイロと首エンコーダから同じ式で作れる。"""
    cy, sy, cp, sp = torch.cos(q[:, 0]), torch.sin(q[:, 0]), torch.cos(q[:, 1]), torch.sin(q[:, 1])
    x1 = cy * ang_vel_b[:, 0] + sy * ang_vel_b[:, 1]
    y1 = -sy * ang_vel_b[:, 0] + cy * ang_vel_b[:, 1]
    z1 = ang_vel_b[:, 2] + qd[:, 0]
    return torch.stack([cp * x1 - sp * z1, y1 + qd[:, 1], sp * x1 + cp * z1], dim=1)


def _uniform(n: int, lo: float, hi: float, device, dim: int = NECK_DIM) -> torch.Tensor:
    return torch.empty(n, dim, device=device).uniform_(lo, hi)


class NeckCommand(CommandTerm):
    """安定化器 (平滑化姿勢・q_cmd 生成・q_base) + センサ遅延 + 評価量。モジュール docstring 参照。"""

    cfg: "NeckCommandCfg"

    def __init__(self, cfg: "NeckCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.joint_ids = self.robot.find_joints(NECK_JOINTS, preserve_order=True)[0]
        self._cam_id = self.robot.find_bodies(CAMERA_BODY)[0][0]
        self.limits = self.robot.data.soft_joint_pos_limits[0, self.joint_ids]  # (2, [lo, hi])
        q = torch.tensor([[0.4, -0.2], [-0.7, 0.6]], device=dev)
        assert torch.allclose(neck_ik(neck_fk(q)), q, atol=1e-5), "neck_fk / neck_ik が逆になっていない"

        # センサ遅延 (env ごと、q_cmd のパターンと一緒に再抽選)。1 物理 step 前の値との線形補間で作る
        self._delay_frac = torch.zeros(n, 1, device=dev)
        self._sensed_prev = torch.zeros(n, 11, device=dev)
        # 平滑化姿勢: (roll, pitch, 巻き戻した yaw) の履歴 (新しい → 古い)
        self._yaw = torch.zeros(n, device=dev)
        self._hist = torch.zeros(n, _SMOOTH_HISTORY, 3, device=dev)
        # q_cmd = center + amp × sin(2π freq t + phase) (軸ごと)
        self._center, self._amp, self._freq, self._phase = (torch.zeros(n, NECK_DIM, device=dev) for _ in range(4))
        self._clock = torch.zeros(n, device=dev)
        self._stale = torch.zeros(n, dtype=torch.bool, device=dev)  # 受け取った q_cmd が 1 step 古いか
        self.mode = torch.zeros(n, dtype=torch.long, device=dev)  # 0 一定 / 1 見渡し / 2 凝視風 (評価の集計用)

        self.q_cmd = torch.zeros(n, NECK_DIM, device=dev)
        self.q_base = torch.zeros(n, NECK_DIM, device=dev)
        self.d_target_w = torch.zeros(n, 3, device=dev)
        self.d_target_w[:, 0] = 1.0
        self._obs = torch.zeros(n, NECK_OBS_DIM, device=dev)
        # 評価量 (真値)
        self._cam_vel_sum = torch.zeros(n, 3, device=dev)
        self._cam_vel_sq = torch.zeros(n, device=dev)
        self._win_vel = torch.zeros(n, 3, device=dev)  # 露光 1 回分 (exposure_substeps 物理 step) の光軸速度の和
        self._win_count = 0
        self._num_win = 0
        self.blur_speed = torch.zeros(n, device=dev)
        """目標方向に対する光軸の相対角速度 [rad/s]。露光 1 回分ごとの平均速度の、制御 1 step 内の RMS。"""
        self.point_err = torch.zeros(n, device=dev)
        """光軸と目標方向のなす角 [rad]。"""
        self._settle = torch.zeros(n, device=dev)  # q_cmd を引き直してからの整定待ち [s] (この間は blur を数えない)
        self._ep_sum = torch.zeros(n, 3, device=dev)  # [blur の和, blur を数えた step 数, point_err の和]
        for name in ("blur_deg_per_10ms", "point_err_deg"):
            self.metrics[name] = torch.zeros(n, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return self._obs

    def _update_metrics(self):
        pass  # _update_command でまとめて更新する

    # -- センサ ------------------------------------------------------------
    def _read_sensors(self) -> torch.Tensor:
        """[胴体 quat (4), 胴体角速度 (3), 首関節角 (2), 首関節角速度 (2)] の真値。"""
        d = self.robot.data
        return torch.cat([d.root_quat_w, d.root_ang_vel_b, d.joint_pos[:, self.joint_ids], d.joint_vel[:, self.joint_ids]], dim=1)

    def _camera_axis(self) -> torch.Tensor:
        e_x = torch.zeros(self.num_envs, 3, device=self.device)
        e_x[:, 0] = 1.0
        return math_utils.quat_apply(self.robot.data.body_quat_w[:, self._cam_id], e_x)

    def record_substep(self):
        """物理 step ごとに `NeckAction.apply_actions` から呼ぶ (sim.step の直前の状態を記録)。"""
        self._sensed_prev = self._read_sensors()
        self._win_vel += torch.cross(self.robot.data.body_ang_vel_w[:, self._cam_id], self._camera_axis(), dim=1)
        self._win_count += 1
        if self._win_count == self.cfg.exposure_substeps:
            # 露光時間内の平均速度 (= 露光中の回転量 / 露光時間)。それより速い振動は像の上で打ち消し合う
            vel = self._win_vel / self._win_count
            self._cam_vel_sum += vel
            self._cam_vel_sq += vel.square().sum(dim=1)
            self._num_win += 1
            self._win_vel.zero_()
            self._win_count = 0

    # -- q_cmd の生成 --------------------------------------------------------
    def _resample_command(self, env_ids: Sequence[int]):
        cfg, dev, n = self.cfg, self.device, len(env_ids)
        self._delay_frac[env_ids] = (_uniform(n, 0.0, cfg.sensor_delay_s, dev, 1) / self._env.physics_dt).clamp(max=1.0)
        # 0 = 一定 (再抽選がステップ変化になる) / 1 = 見渡し (yaw の往復) / 2 = 凝視風 (両軸がゆっくり動く)
        mode = torch.randint(0, 3, (n, 1), device=dev)
        self.mode[env_ids] = mode.squeeze(1)
        sweep = torch.cat([_uniform(n, *cfg.sweep_amp, dev, 1), torch.zeros(n, 1, device=dev)], dim=1)
        amp = torch.where(mode == 1, sweep, _uniform(n, 0.0, cfg.gaze_amp, dev)) * (mode != 0)
        lo, hi = self.limits[:, 0] + cfg.range_margin, self.limits[:, 1] - cfg.range_margin
        center = lo + (hi - lo) * torch.rand(n, NECK_DIM, device=dev)
        self._center[env_ids] = torch.maximum(torch.minimum(center, hi - amp), lo + amp)
        self._amp[env_ids] = amp
        self._freq[env_ids] = torch.where(mode == 1, _uniform(n, *cfg.sweep_freq, dev), _uniform(n, *cfg.gaze_freq, dev))
        self._phase[env_ids] = _uniform(n, 0.0, 2.0 * math.pi, dev)
        self._settle[env_ids] = cfg.settle_time_s

    # -- 毎 step の更新 ------------------------------------------------------
    def _update_command(self):
        env, cfg, dt, dev, n = self._env, self.cfg, self._env.step_dt, self.device, self.num_envs
        fresh = env.episode_length_buf == 0
        fresh1 = fresh.unsqueeze(1)

        # --- 遅延 + ノイズ付きのセンサ値 ---
        cur = self._read_sensors()
        prev = torch.where(fresh1, cur, self._sensed_prev)
        prev[:, :4] *= torch.where((cur[:, :4] * prev[:, :4]).sum(dim=1, keepdim=True) < 0.0, -1.0, 1.0)
        sensed = torch.lerp(cur, prev, self._delay_frac)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(math_utils.normalize(sensed[:, :4]))
        rpy = math_utils.wrap_to_pi(torch.stack([roll, pitch, yaw], dim=1)) + _uniform(n, -cfg.rpy_noise, cfg.rpy_noise, dev, 3)
        ang_vel = sensed[:, 4:7] + _uniform(n, -cfg.gyro_noise, cfg.gyro_noise, dev, 3)
        q = sensed[:, 7:9] + _uniform(n, -cfg.joint_pos_noise, cfg.joint_pos_noise, dev)
        qd = sensed[:, 9:11] + _uniform(n, -cfg.joint_vel_noise, cfg.joint_vel_noise, dev)
        quat_body = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])

        # --- 平滑化胴体姿勢: 歩行 1 周期分の移動平均 (窓長 = 1 / (歩行周波数 × dt) step、端は小数重み) ---
        self._yaw = torch.where(fresh, rpy[:, 2], self._yaw + math_utils.wrap_to_pi(rpy[:, 2] - self._yaw))
        rpy[:, 2] = self._yaw
        self._hist = torch.where(
            fresh.view(n, 1, 1), rpy.unsqueeze(1), torch.cat([rpy.unsqueeze(1), self._hist[:, :-1]], dim=1)
        )
        freq = compute_cmd_phase_freq(env, **cfg.phase_freq_params)
        window = (1.0 / (freq * dt)).clamp(2.0, _SMOOTH_HISTORY - 2.0).unsqueeze(1)
        age = torch.arange(_SMOOTH_HISTORY, device=dev).unsqueeze(0)
        weight = (window - age).clamp(0.0, 1.0) / window
        smooth = (weight.unsqueeze(2) * self._hist).sum(dim=1)
        # yaw の遅れ補正: 窓の平均角速度 (yaw[0] − yaw[窓]) / 窓 × 平均遅れ (窓 − 1) / 2。周期的な揺れは打ち消される
        i0 = window.floor().long()
        yaw_hist = self._hist[:, :, 2]
        yaw_old = torch.lerp(yaw_hist.gather(1, i0), yaw_hist.gather(1, i0 + 1), window - i0).squeeze(1)
        smooth[:, 2] += (self._yaw - yaw_old) * ((window - 1.0) / (2.0 * window)).squeeze(1)
        quat_smooth = math_utils.quat_from_euler_xyz(smooth[:, 0], smooth[:, 1], smooth[:, 2])

        # --- 首指令 q_cmd: 首制御パッケージとの非同期を模擬する。受け取る値は最新か 1 step 古いかのどちらかで、
        # 切り替わる瞬間に重複 (古くなる) / 欠落 (最新に戻る) が起きる。評価の基準は乱れのない q_ref ---
        self._clock += dt
        self._stale ^= torch.rand(n, device=dev) < cfg.cmd_skip_prob
        clock = torch.stack([self._clock, self._clock - dt * self._stale], dim=0).unsqueeze(2)
        q_ref, self.q_cmd = self._center + self._amp * torch.sin(2.0 * math.pi * self._freq * clock + self._phase)

        # --- 目標視線方向と解析 IK ---
        d_ref = math_utils.quat_apply(quat_smooth, neck_fk(q_ref))
        d_target = math_utils.quat_apply(quat_smooth, neck_fk(self.q_cmd))
        if cfg.stabilize:
            # 先読み: lead_time_s 先の胴体姿勢 (今の角速度で外挿) に対して IK を解く
            speed = ang_vel.norm(dim=1)
            lead = math_utils.quat_from_angle_axis(speed * cfg.lead_time_s, ang_vel / (speed.unsqueeze(1) + 1e-9))
            q_base = neck_ik(math_utils.quat_apply_inverse(math_utils.quat_mul(quat_body, lead), d_target))
            self.q_base = torch.maximum(torch.minimum(q_base, self.limits[:, 1]), self.limits[:, 0])
        else:
            self.q_base = self.q_cmd
        self._obs = torch.cat([ang_vel, rpy - smooth, q, qd, self.q_cmd, self.q_base, camera_gyro(ang_vel, q, qd)], dim=1)

        # --- 評価量 (真値): この step の間のカメラ角速度 − 目標方向の角速度 (光軸に垂直な成分) の RMS ---
        target_vel = (d_ref - self.d_target_w) / dt
        num_win = max(self._num_win, 1)
        mean_sq = (
            self._cam_vel_sq / num_win
            - 2.0 * (self._cam_vel_sum / num_win * target_vel).sum(dim=1)
            + target_vel.square().sum(dim=1)
        )
        d_cam = self._camera_axis()
        angle = torch.atan2(torch.cross(d_cam, d_ref, dim=1).norm(dim=1), (d_cam * d_ref).sum(dim=1))
        # q_cmd のステップ変化の直後は目標方向が飛ぶので数えない (意図した首振りを罰しない。その間は point_err だけが効く)
        counted = ~fresh & (self._settle <= 0.0)
        self._settle -= dt
        self.blur_speed = torch.where(counted, mean_sq.clamp(min=0.0).sqrt(), 0.0)
        self.point_err = torch.where(fresh, 0.0, angle)
        self.d_target_w = d_ref
        self._cam_vel_sum.zero_()
        self._cam_vel_sq.zero_()
        self._num_win = 0

        # エピソード平均 (リセット時に Metrics/neck/* として記録される)
        step = torch.stack([self.blur_speed, counted.float(), self.point_err], dim=1)
        self._ep_sum = torch.where(fresh1, 0.0, self._ep_sum + step)
        # 露光 10 ms あたりの光軸の回転量 [deg]
        self.metrics["blur_deg_per_10ms"] = torch.rad2deg(self._ep_sum[:, 0] / self._ep_sum[:, 1].clamp(min=1.0)) * 0.01
        self.metrics["point_err_deg"] = torch.rad2deg(self._ep_sum[:, 2] / env.episode_length_buf.clamp(min=1))


@configclass
class NeckCommandCfg(CommandTermCfg):
    class_type: type = NeckCommand

    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (2.0, 6.0)
    """q_cmd のパターン (一定 / 見渡し / 凝視風) を引き直す間隔 [s]。"""

    stabilize: bool = True
    """False で安定化なし (q_base = q_cmd)。比較用 (設計書 7 章の「安定化なし」)。"""

    lead_time_s: float = 0.12
    """解析 IK の先読み時間 [s]。首 PD は目標との差に比例した速度で動く (時定数 Kd / Kp = 0.12 s) ので、その分を
    先読みで補う。0 だと歩行の揺れ (1〜6 Hz) に対して補償が遅れてほぼ効かない。指令遅延 (10〜35 ms) の分は
    6 Hz 以上を逆に増やすので、そこは残差方策に任せる (2026-11-20 の周波数帯別の計測)。"""

    phase_freq_params: dict = {}
    """`compute_cmd_phase_freq` の引数。env cfg が歩行 expert の gait_phase 観測と同じ値を入れる。"""

    sensor_delay_s: float = 0.005
    """IMU・首エンコーダの観測遅延の上限 [s] (env ごとに U(0, 上限))。
    ponytail: 1 物理 step 前との線形補間なので物理 dt (5 ms) が上限。それ以上にするなら物理 step の履歴を持つ。"""

    # 観測ノイズ (毎 step 独立の一様乱数の振幅)。実機ログで合わせること
    rpy_noise: float = 0.002  # IMU 姿勢 [rad]。q_base に直接乗るので歩行側の projected_gravity ノイズより小さくしている
    gyro_noise: float = 0.05  # [rad/s]
    joint_pos_noise: float = 0.002  # [rad]
    joint_vel_noise: float = 0.1  # [rad/s]

    # q_cmd の生成
    range_margin: float = 0.15
    """可動範囲 (ソフトリミット) の両端に残す余裕 [rad] (設計書 3.3: 端で飽和すると安定化が効かない)。"""
    sweep_amp: tuple[float, float] = (0.2, 0.6)  # 見渡し (yaw) の振幅 [rad]
    sweep_freq: tuple[float, float] = (0.2, 0.6)  # 見渡しの周波数 [Hz]
    gaze_amp: float = 0.2  # 凝視風の振幅の上限 [rad]
    gaze_freq: tuple[float, float] = (0.05, 0.3)  # [Hz]
    settle_time_s: float = 1.0
    """q_cmd のパターンを引き直した (ステップ変化) 後、blur の報酬・metrics から外す時間 [s]。
    首 PD の時定数 Kd / Kp = 0.12 s で、1 rad のステップは 0.4 s 後でもまだ 0.3 rad/s で動いている。"""
    exposure_substeps: int = 2
    """露光時間 [物理 step]。2 × 5 ms = 10 ms (設計書 3.4 の仮定)。decimation の約数にすること。"""
    cmd_skip_prob: float = 0.1
    """各 step で指令の重複または欠落が起きる確率。"""


class NeckAction(ActionTerm):
    """首の関節目標 = ``q_base + scale × 残差`` (残差とソフトリミットでクリップ)。"""

    cfg: "NeckActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "NeckActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._cmd: NeckCommand = env.command_manager.get_term(cfg.command_name)
        self._raw = torch.zeros(self.num_envs, NECK_DIM, device=self.device)
        self._prev = torch.zeros_like(self._raw)
        self._target = torch.zeros_like(self._raw)
        # Kd は PhysX の関節ドライブ (陰解法) に入れる。DelayedPD の陽的な damping だと yaw 軸は Kd·dt / I ≈ 1.5
        # (質量 DR 込みで発散限界 2 の近く) になり、25 Hz 以上で首が細かく振動した。実機の Kd もモータ内の速い
        # ループで効くので、指令遅延の掛からないこの形の方が近い。
        self._asset.write_joint_damping_to_sim(cfg.damping, joint_ids=self._cmd.joint_ids)

    @property
    def action_dim(self) -> int:
        return NECK_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def prev_actions(self) -> torch.Tensor:
        return self._prev

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._target

    def process_actions(self, actions: torch.Tensor):
        self._prev[:] = self._raw
        self._raw[:] = actions
        residual = (self.cfg.scale * actions).clamp(-self.cfg.max_residual, self.cfg.max_residual)
        limits = self._cmd.limits
        self._target = torch.maximum(torch.minimum(self._cmd.q_base + residual, limits[:, 1]), limits[:, 0])

    def apply_actions(self):
        self._asset.set_joint_position_target(self._target, joint_ids=self._cmd.joint_ids)
        self._cmd.record_substep()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        self._prev[env_ids] = 0.0
        return {}


@configclass
class NeckActionCfg(ActionTermCfg):
    class_type: type = NeckAction

    command_name: str = "neck"
    scale: float = 0.1
    """方策の出力 1 単位あたりの残差 [rad]。0 にすると解析 IK のみ (比較用)。"""
    max_residual: float = 0.3
    """残差のクリップ [rad]。"""
    damping: float = 1.2
    """首関節の Kd [N·m·s/rad] (実機 HEAD_KD)。アクチュエータ cfg 側の damping は 0 にすること。"""


# ---------------------------------------------------------------------------
# 報酬・観測
# ---------------------------------------------------------------------------
def neck_blur_speed(env: "ManagerBasedRLEnv", command_name: str = "neck") -> torch.Tensor:
    """目標方向に対する光軸の相対角速度 [rad/s] (ブラー量に比例)。意図した首振り (q_cmd の動き) は含まない。"""
    return env.command_manager.get_term(command_name).blur_speed


def neck_pointing_error(env: "ManagerBasedRLEnv", command_name: str = "neck") -> torch.Tensor:
    """光軸と目標方向のなす角 [rad]。"""
    return env.command_manager.get_term(command_name).point_err


def neck_eval(env: "ManagerBasedRLEnv", command_name: str = "neck") -> torch.Tensor:
    """critic 用の特権観測: [blur_speed, point_err]。"""
    term: NeckCommand = env.command_manager.get_term(command_name)
    return torch.stack([term.blur_speed, term.point_err], dim=1)


def neck_blur_speed_sq(env: "ManagerBasedRLEnv", command_name: str = "neck") -> torch.Tensor:
    """`neck_blur_speed` の 2 乗 [(rad/s)²]。大きく揺れた瞬間を重く見る。"""
    return env.command_manager.get_term(command_name).blur_speed.square()
