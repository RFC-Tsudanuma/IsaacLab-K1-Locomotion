"""Virtual perception module for the Booster K1 goalkeeper task.

NOTE (2026-07-24): booster_amp_lab の beyond_mimic/mdp/soccer_perception.py
(defend/kick タスクが使う実機標準のカメラ知覚モデル) を goalkeeper 用に **コピー**
したもの。ユーザー指示で goalkeeper を独立させるため、他リポジトリへの import 依存を
持たせずここに複製する。カメラ仕様 (D-Robotics RDK Stereo: FOV 150°/80°、Head_2
マウント下向き40°、レイテンシ116ms、25Hz、距離依存ノイズ σ(d)=0.124d+0.149) は
soccer_vision_train_cfg がそのまま持つ。本家の更新を取り込みたいときは再コピーする。

Ports the SPIRIT of mjlab's ``VirtualPerception`` (see
``mjlab/src/mjlab/tasks/kick/mdp/perception.py``) to the Isaac Lab managers-
based stack:

  * FOV check on a body-mounted optical-frame camera (default: Head_2 link
    on K1 with the RealSense D435i offset).
  * Distance-dependent Bernoulli detection (in-FOV AND in-range), per-env DR.
  * Distance-dependent Gaussian noise sigma(d) = a*d + b on world xy, with
    per-env DR multipliers on (a, b).
  * Latency ring buffer (per-env latency expressed in control steps).
  * Update-frequency decimation (per-env Hz mean); reset-on-miss fast-lock so
    a fresh detection arrives within one step after a dropout streak.
  * Blind episodes (per-env Bernoulli flag at reset zeroing detection prob).
  * Hold-on-miss flag (default False for K1 — emit zeros when ball_mask=0).
  * ``last_seen_dt`` counter (seconds since the most recent detection).

All operations are fully vectorized over ``num_envs``. Quaternions are wxyz,
matching the Isaac Lab convention.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class VirtualPerceptionCfg:
    """Parameters for the simulated head-camera ball detector.

    Defaults track the mjlab K1 RealSense D435i deployment, with the V1.49
    DR knobs (lower detection-prob floor + 10% blind episodes).
    """

    # ---------------------------------------------------------------- camera
    camera_body_name: str = "Head_2"
    """Robot body the camera is rigidly attached to (post-yaw, post-pitch)."""

    # Isaac Lab K1 URDF: Head_2 origin sits at the pitch joint with
    # +X = head forward, +Z = head up (verified by the inertial CoM at
    # (0.011, -0.001, 0.081)). The camera sits on the forehead — about
    # 6 cm forward and 10 cm above the joint.
    # The previous defaults (0.00124, 0.04553, -0.01582) and the rotated
    # quaternion are mjlab MJCF values (+Y forward, +Z down) and do NOT
    # match the Isaac Lab body-axis convention.
    camera_offset_pos: tuple[float, float, float] = (0.06, 0.0, 0.10)
    """Camera optical-frame origin offset in the camera body's local frame (m)."""

    camera_offset_quat: tuple[float, float, float, float] = (
        1.0, 0.0, 0.0, 0.0,
    )
    """Camera optical-frame orientation relative to the camera body (wxyz).

    Identity by default — the camera looks forward along Head_2's +X axis.
    """

    # ------------------------------------------------------------------ FOV
    # D-Robotics RDK Stereo Camera Module (SC230AI, 2.28 mm lens):
    # 178° diagonal / 150° horizontal / 80° vertical.
    fov_h_deg: float = 150.0
    """Horizontal FOV (degrees, full angle). Default: D-Robotics RDK Stereo Camera."""

    fov_v_deg: float = 80.0
    """Vertical FOV (degrees, full angle). Default: D-Robotics RDK Stereo Camera."""

    # --------------------------------------------------------------- detection
    max_detection_range: float = 7.0
    """Distance beyond which detection probability decays toward zero (m)."""

    range_decay: float = 2.0
    """Soft falloff distance for range-based detection probability (m)."""

    detection_prob_in_fov_range: tuple[float, float] = (0.30, 0.95)
    """Per-env absolute range for in-FOV detection probability."""

    blind_prob: float = 0.10
    """Per-episode probability that an env is fully blind for the entire
    episode (detection_prob forced to 0)."""

    # ----------------------------------------------------------- occlusion
    # Temporal occlusion: contiguous dropout streaks that model the ball being
    # briefly hidden (by the robot's own body/hands, another player, etc.).
    # Distinct from the per-step Bernoulli dropout — once an event starts, the
    # ball is forced undetected for a whole sampled duration.
    occlusion_prob: float = 0.002
    """Per-step hazard probability of *starting* a new occlusion event on an
    env that is not already occluded. 0 disables temporal occlusion.
    At the 0.02 s control step this is ~1 event per 10 s."""

    occlusion_duration_range: tuple[float, float] = (0.2, 0.8)
    """Uniform range (s) for the duration of each occlusion event."""

    # -------------------------------------------------------- FOV dead zone
    # Static blind region inside the FOV (smudged lens / fixed occluder /
    # self-body in view). Sampled per episode in the camera angular frame:
    # whenever the ball's (yaw, pitch) falls inside the rectangle it is
    # undetected, even if otherwise in-FOV and in-range. Unlike temporal
    # occlusion, the ball reappears as soon as it leaves the dead region.
    deadzone_prob: float = 0.5
    """Per-episode probability that an env has a FOV dead zone."""

    deadzone_half_h_range: tuple[float, float] = (0.10, 0.30)
    """Dead-zone half-width as a fraction of the horizontal FOV half-angle."""

    deadzone_half_v_range: tuple[float, float] = (0.10, 0.30)
    """Dead-zone half-height as a fraction of the vertical FOV half-angle."""

    # -------------------------------------------------------------- xy noise
    noise_a: float = 0.05
    """Linear coefficient in distance-dependent noise model sigma(d) = a*d + b."""

    noise_b: float = 0.08
    """Offset coefficient (m) in sigma(d) = a*d + b."""

    # ------------------------------------------------- attitude-driven noise
    # 実機のボール位置は「内部パラメータ + 首の姿勢 + 地面平面」で出す (地面投影法)。
    # この方式の誤差は等方ガウスではなく、姿勢推定の角度誤差が支配する:
    #   奥行き誤差 = (r^2 + h^2)/h * pitch誤差   ← 距離の 2 乗で増える
    #   横誤差     = r * yaw誤差                 ← 距離に比例
    # 画素ノイズ由来は姿勢誤差の 1/4 程度で、実質無視できる (2026-08-03 試算)。
    # attitude_noise=True でこのモデルを使い、等方 sigma(d) は使わない (二重計上になる)。
    attitude_noise: bool = False
    """True: 姿勢誤差→地面投影の誤差モデルを使う。False: 等方 sigma(d)=a*d+b。"""

    attitude_bias_deg_range: tuple[float, float] = (0.0, 1.2)
    """エピソード固定の姿勢バイアス [deg]。キャリブ残差 0.28° + 首のバックラッシュ相当。"""

    attitude_osc_deg_range: tuple[float, float] = (0.0, 1.5)
    """歩行同期の姿勢振動の振幅 [deg]。白色でない相関ノイズなので均されない。"""

    attitude_osc_hz_range: tuple[float, float] = (1.2, 2.0)
    """姿勢振動の周波数 [Hz]。歩行周期 (1.6 Hz) 前後。"""

    pixel_noise_px: float = 2.0
    """ボール中心の画素ノイズ [px]。"""

    focal_px: float = 208.3
    """焦点距離 [px] (実機 d-robotics カメラ fx=208.263)。画素ノイズの角度換算に使う。"""

    noise_a_range: tuple[float, float] = (0.7, 1.5)
    """Per-env multiplicative range on ``noise_a``."""

    noise_b_range: tuple[float, float] = (0.7, 1.5)
    """Per-env multiplicative range on ``noise_b``."""

    # --------------------------------------------------------------- latency
    latency_mean_range: tuple[float, float] = (0.080, 0.160)
    """Per-env uniform range for the latency Gaussian mean (s)."""

    latency_std_s: float = 0.018
    """Std of the per-env latency Gaussian (s)."""

    update_hz_mean_range: tuple[float, float] = (20.0, 30.0)
    """Per-env uniform range for the detector update-rate Gaussian mean (Hz)."""

    update_hz_std: float = 1.06
    """Std of the detector update-rate Gaussian (Hz)."""

    buffer_size: int = 16
    """Latency ring buffer length, in control steps. Should cover
    ``ceil((max latency mean + 3*latency_std)/dt)``."""

    # -------------------------------------------------------------- misc
    hold_last_on_miss: bool = False
    """When True, hold the most recent valid value on a miss. When False
    (default for K1), emit zeros. ``ball_mask`` is 0 either way."""

    head_tracks_ball: bool = False
    """When True, the camera is assumed to be actively pointed at the ball by
    the strategy layer (head-tracking), so the FOV check uses a virtual camera
    orientation aimed at the ball instead of the fixed head-body orientation.

    Goalkeeper use: the real robot's strategy keeps the head on the ball, but
    the RL policy here does not control the neck joints. Without this flag the
    fixed-forward head would drop any off-axis ball as out-of-FOV, which does
    not match deployment. With it, the FOV gate effectively always passes
    (unless the ball is behind the robot), while latency / noise / detection
    rate / occlusion / dead-zone are still applied — i.e. "always looking at
    the ball, but with real-camera quality degradation"."""


def _pitch_down_quat(deg: float) -> tuple[float, float, float, float]:
    """Quaternion rotating the camera optical +X axis downward about local +Y."""
    half = math.radians(float(deg)) * 0.5
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def soccer_vision_train_cfg(
    *,
    hold_last_on_miss: bool = True,
    detection_prob: float = 0.90,
    blind_prob: float = 0.0,
    max_detection_range: float = 7.0,
) -> VirtualPerceptionCfg:
    """Shared soccer virtual-camera preset for kicker/receiver/defender roles.

    The values track the LVDRS/output-sc perception model: structured ball
    detections, 90% in-FOV detection up to 7 m, distance-dependent xy noise,
    roughly 25 Hz detector updates, and ~116 ms perception latency.
    """
    return VirtualPerceptionCfg(
        camera_offset_quat=_pitch_down_quat(40.0),
        hold_last_on_miss=hold_last_on_miss,
        blind_prob=blind_prob,
        max_detection_range=max_detection_range,
        detection_prob_in_fov_range=(detection_prob, detection_prob),
        noise_a=0.124,
        noise_b=0.149,
        noise_a_range=(0.8, 1.2),
        noise_b_range=(0.8, 1.2),
        latency_mean_range=(0.116, 0.116),
        latency_std_s=0.018,
        update_hz_mean_range=(25.36, 25.36),
        update_hz_std=1.06,
    )


def soccer_vision_repair_cfg() -> VirtualPerceptionCfg:
    """Easy shared-camera preset for short approach-repair curricula.

    This keeps the same API and last-seen semantics as the training preset but
    reduces latency/noise/dropout so the policy can relearn clean approach and
    foot placement before harder perception randomization is reintroduced.
    """
    return VirtualPerceptionCfg(
        camera_offset_quat=_pitch_down_quat(40.0),
        hold_last_on_miss=True,
        blind_prob=0.0,
        max_detection_range=8.0,
        detection_prob_in_fov_range=(0.95, 1.0),
        noise_a=0.02,
        noise_b=0.03,
        noise_a_range=(0.8, 1.2),
        noise_b_range=(0.8, 1.2),
        latency_mean_range=(0.020, 0.060),
        latency_std_s=0.006,
        update_hz_mean_range=(30.0, 45.0),
        update_hz_std=1.0,
    )


class VirtualPerception:
    """Stateful simulator of a head-camera ball detection pipeline.

    Each ``update`` call:
      1. Compose camera world pose from the head body pose + static offset.
      2. Transform the ball world position into the camera optical frame.
      3. Run FOV check (forward AND |yaw|<half AND |pitch|<half).
      4. Apply distance-attenuated Bernoulli detection.
      5. Inject Gaussian xy noise scaled by camera-distance.
      6. Express the noisy xy in the robot body-yaw frame.
      7. Decimate to the per-env detector rate (reset-on-miss fast-lock).
      8. Push (pos, mask) into the latency ring buffer and read out the
         buffer entry corresponding to each env's sampled latency.
      9. Update the per-env ``last_seen_dt`` counter.
    """

    def __init__(
        self,
        cfg: VirtualPerceptionCfg,
        robot: "Articulation",
        num_envs: int,
        dt: float,
        device: torch.device | str,
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = torch.device(device)

        # Resolve camera body index once. (Re-resolved on first update if
        # ``robot`` happens to be re-bound — kept here for the fast path.)
        head_ids, _ = robot.find_bodies([cfg.camera_body_name], preserve_order=True)
        if len(head_ids) == 0:
            raise ValueError(
                f"camera_body_name {cfg.camera_body_name!r} not found on robot "
                f"(have {robot.body_names!r})"
            )
        self._head_idx = int(head_ids[0])

        # Static optical-frame offset (camera body -> camera optical).
        self._cam_offset_pos = torch.tensor(
            cfg.camera_offset_pos, dtype=torch.float32, device=self.device
        )
        self._cam_offset_quat = torch.tensor(
            cfg.camera_offset_quat, dtype=torch.float32, device=self.device
        )

        # Pre-converted FOV half-angles (radians).
        self._fov_h_half = math.radians(cfg.fov_h_deg) * 0.5
        self._fov_v_half = math.radians(cfg.fov_v_deg) * 0.5

        # Per-env DR coefficients ------------------------------------------
        N = self.num_envs
        d = self.device
        self._noise_a_per_env = torch.full((N,), cfg.noise_a, dtype=torch.float32, device=d)
        self._noise_b_per_env = torch.full((N,), cfg.noise_b, dtype=torch.float32, device=d)
        self._detection_prob_per_env = torch.full(
            (N,), 0.5 * (cfg.detection_prob_in_fov_range[0] + cfg.detection_prob_in_fov_range[1]),
            dtype=torch.float32, device=d,
        )
        self._blind_per_env = torch.zeros((N,), dtype=torch.bool, device=d)

        # Latency / update-rate per-env counters ---------------------------
        self._latency_steps = torch.zeros(N, dtype=torch.long, device=d)
        self._update_period_steps = torch.ones(N, dtype=torch.long, device=d)
        self._steps_since_update = torch.zeros(N, dtype=torch.long, device=d)

        # Temporal occlusion: steps remaining in the active occlusion event.
        self._occlusion_steps_remaining = torch.zeros(N, dtype=torch.long, device=d)

        # Attitude error state (pitch/yaw): episode-fixed bias + gait-synchronous
        # oscillation. The oscillation is a correlated disturbance, so downstream
        # filters cannot average it away — unlike the white pixel noise.
        self._att_bias = torch.zeros(N, 2, dtype=torch.float32, device=d)
        self._att_osc_amp = torch.zeros(N, 2, dtype=torch.float32, device=d)
        self._att_osc_w = torch.zeros(N, dtype=torch.float32, device=d)
        self._att_osc_phase = torch.zeros(N, 2, dtype=torch.float32, device=d)
        self._att_t = torch.zeros(N, dtype=torch.float32, device=d)

        # Static per-episode FOV dead zone (camera angular frame, radians).
        self._deadzone_active = torch.zeros(N, dtype=torch.bool, device=d)
        self._deadzone_yaw_c = torch.zeros(N, dtype=torch.float32, device=d)
        self._deadzone_pitch_c = torch.zeros(N, dtype=torch.float32, device=d)
        self._deadzone_yaw_h = torch.zeros(N, dtype=torch.float32, device=d)
        self._deadzone_pitch_h = torch.zeros(N, dtype=torch.float32, device=d)

        # Ring buffer (head -> newest entry) -------------------------------
        self._buffer_pos = torch.zeros(cfg.buffer_size, N, 2, dtype=torch.float32, device=d)
        self._buffer_mask = torch.zeros(cfg.buffer_size, N, dtype=torch.float32, device=d)
        self._buffer_head = 0

        # Measurement sequence number, pushed through the same latency buffer so
        # downstream filters can tell a genuinely new detection from a held one.
        # Updating an alpha-beta / Kalman filter on a held value would over-weight
        # the same measurement at the control rate (50 Hz) instead of the detector
        # rate (~25 Hz).
        self._buffer_seq = torch.zeros(cfg.buffer_size, N, dtype=torch.long, device=d)
        self._meas_seq = torch.zeros(N, dtype=torch.long, device=d)
        self._prev_out_seq = torch.zeros(N, dtype=torch.long, device=d)
        self._fresh = torch.zeros(N, dtype=torch.float32, device=d)

        # Most recent (pre-buffer) detection state -------------------------
        self._last_pos = torch.zeros(N, 2, dtype=torch.float32, device=d)
        self._last_mask = torch.zeros(N, dtype=torch.float32, device=d)

        # Outputs (post-buffer) --------------------------------------------
        self._ball_pos_b = torch.zeros(N, 2, dtype=torch.float32, device=d)
        self._ball_mask = torch.zeros(N, dtype=torch.float32, device=d)
        self._last_seen_dt = torch.zeros(N, dtype=torch.float32, device=d)
        self._in_fov = torch.zeros(N, dtype=torch.float32, device=d)
        self._occluded = torch.zeros(N, dtype=torch.float32, device=d)
        self._in_deadzone = torch.zeros(N, dtype=torch.float32, device=d)
        self._range_prob = torch.zeros(N, dtype=torch.float32, device=d)
        self._detect_prob = torch.zeros(N, dtype=torch.float32, device=d)
        self._raw_detected = torch.zeros(N, dtype=torch.float32, device=d)
        self._ball_in_cam = torch.zeros(N, 3, dtype=torch.float32, device=d)

        # Cached camera pose (useful for debug viz) ------------------------
        self._cam_pos_w = torch.zeros(N, 3, dtype=torch.float32, device=d)
        self._cam_quat_w = torch.zeros(N, 4, dtype=torch.float32, device=d)
        self._cam_quat_w[:, 0] = 1.0

        # Initial per-env sampling -----------------------------------------
        self._sample_per_env(torch.arange(N, dtype=torch.long, device=d))

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset per-env buffers and resample latency / rate / DR coefficients."""
        if env_ids is None or (hasattr(env_ids, "numel") and env_ids.numel() == 0):
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._buffer_pos[:, env_ids] = 0.0
        self._buffer_mask[:, env_ids] = 0.0
        self._buffer_seq[:, env_ids] = 0
        self._meas_seq[env_ids] = 0
        self._prev_out_seq[env_ids] = 0
        self._fresh[env_ids] = 0.0
        self._att_t[env_ids] = 0.0
        self._steps_since_update[env_ids] = 0
        self._occlusion_steps_remaining[env_ids] = 0
        self._last_pos[env_ids] = 0.0
        self._last_mask[env_ids] = 0.0
        self._ball_pos_b[env_ids] = 0.0
        self._ball_mask[env_ids] = 0.0
        self._last_seen_dt[env_ids] = 0.0
        self._in_fov[env_ids] = 0.0
        self._occluded[env_ids] = 0.0
        self._in_deadzone[env_ids] = 0.0
        self._range_prob[env_ids] = 0.0
        self._detect_prob[env_ids] = 0.0
        self._raw_detected[env_ids] = 0.0
        self._ball_in_cam[env_ids] = 0.0
        self._sample_per_env(env_ids)

    # ---------------------------------------------------------------- sampling
    def _sample_per_env(self, env_ids: torch.Tensor) -> None:
        cfg = self.cfg
        n = env_ids.numel()
        if n == 0:
            return

        def _uniform(lo: float, hi: float) -> torch.Tensor:
            return torch.rand(n, device=self.device) * (hi - lo) + lo

        # Per-env xy-noise coefficients (multiplicative on cfg base values).
        self._noise_a_per_env[env_ids] = cfg.noise_a * _uniform(*cfg.noise_a_range)
        self._noise_b_per_env[env_ids] = cfg.noise_b * _uniform(*cfg.noise_b_range)

        # Attitude error: (pitch, yaw) bias + oscillation. Signs are independent
        # per axis so a run can bias the projection either near or far.
        if cfg.attitude_noise:
            deg = math.pi / 180.0
            sign = lambda: torch.where(  # noqa: E731
                torch.rand(n, 2, device=self.device) < 0.5, 1.0, -1.0
            )
            bias_lo, bias_hi = cfg.attitude_bias_deg_range
            osc_lo, osc_hi = cfg.attitude_osc_deg_range
            self._att_bias[env_ids] = (
                (torch.rand(n, 2, device=self.device) * (bias_hi - bias_lo) + bias_lo) * deg * sign()
            )
            self._att_osc_amp[env_ids] = (
                torch.rand(n, 2, device=self.device) * (osc_hi - osc_lo) + osc_lo
            ) * deg
            self._att_osc_w[env_ids] = 2.0 * math.pi * _uniform(*cfg.attitude_osc_hz_range)
            self._att_osc_phase[env_ids] = torch.rand(n, 2, device=self.device) * (2.0 * math.pi)
        else:
            self._att_bias[env_ids] = 0.0
            self._att_osc_amp[env_ids] = 0.0

        # Per-env in-FOV detection probability (absolute, not multiplicative).
        det_p = _uniform(*cfg.detection_prob_in_fov_range)

        # Blind episodes: force detection probability to zero.
        if cfg.blind_prob > 0.0:
            blind = torch.rand(n, device=self.device) < cfg.blind_prob
            self._blind_per_env[env_ids] = blind
            det_p = torch.where(blind, torch.zeros_like(det_p), det_p)
        else:
            self._blind_per_env[env_ids] = False
        self._detection_prob_per_env[env_ids] = det_p

        # Per-env latency in steps.
        latency_mean = _uniform(*cfg.latency_mean_range)
        latency = (
            latency_mean + torch.randn(n, device=self.device) * cfg.latency_std_s
        ).clamp_min(0.0)
        self._latency_steps[env_ids] = (
            (latency / self.dt).round().long().clamp_(0, cfg.buffer_size - 1)
        )

        # Per-env update-rate period in steps.
        hz_mean = _uniform(*cfg.update_hz_mean_range)
        hz = (
            hz_mean + torch.randn(n, device=self.device) * cfg.update_hz_std
        ).clamp_min(1.0)
        period_s = 1.0 / hz
        self._update_period_steps[env_ids] = (
            (period_s / self.dt).round().long().clamp_min_(1)
        )

        # Per-episode FOV dead zone (camera angular frame). Sample a half-size,
        # then a center such that the box stays fully inside the FOV.
        if cfg.deadzone_prob > 0.0:
            active = torch.rand(n, device=self.device) < cfg.deadzone_prob
            yaw_h = _uniform(*cfg.deadzone_half_h_range) * self._fov_h_half
            pitch_h = _uniform(*cfg.deadzone_half_v_range) * self._fov_v_half
            yaw_c = (torch.rand(n, device=self.device) * 2.0 - 1.0) * (
                self._fov_h_half - yaw_h
            )
            pitch_c = (torch.rand(n, device=self.device) * 2.0 - 1.0) * (
                self._fov_v_half - pitch_h
            )
            self._deadzone_active[env_ids] = active
            self._deadzone_yaw_c[env_ids] = yaw_c
            self._deadzone_pitch_c[env_ids] = pitch_c
            self._deadzone_yaw_h[env_ids] = yaw_h
            self._deadzone_pitch_h[env_ids] = pitch_h
        else:
            self._deadzone_active[env_ids] = False

    # ------------------------------------------------------- projection error
    def _projection_error(
        self, cam_pos_w: torch.Tensor, ball_pos_w: torch.Tensor
    ) -> torch.Tensor:
        """Ground-projection position error from camera attitude error. Shape (N, 2).

        The ball's world position comes from intersecting the image ray with the
        z=0 plane. Writing r for the horizontal camera-ball distance and h for the
        camera height, an elevation error d_pitch moves the intersection along the
        bearing by (r^2 + h^2)/h * d_pitch, and a bearing error d_yaw moves it
        sideways by r * d_yaw. The first term is quadratic in distance, which is
        why far balls are poorly localised in depth but well localised laterally.
        """
        self._att_t += self.dt
        t = self._att_t.unsqueeze(-1)
        att = self._att_bias + self._att_osc_amp * torch.sin(
            self._att_osc_w.unsqueeze(-1) * t + self._att_osc_phase
        )
        d_pitch, d_yaw = att[:, 0], att[:, 1]

        # Pixel noise enters as an extra angular error on both axes.
        if self.cfg.pixel_noise_px > 0.0 and self.cfg.focal_px > 0.0:
            px_ang = self.cfg.pixel_noise_px / self.cfg.focal_px
            d_pitch = d_pitch + torch.randn_like(d_pitch) * px_ang
            d_yaw = d_yaw + torch.randn_like(d_yaw) * px_ang

        rel = ball_pos_w[:, :2] - cam_pos_w[:, :2]
        r = rel.norm(dim=-1).clamp_min(1e-3)
        h = cam_pos_w[:, 2].clamp_min(0.2)
        u = rel / r.unsqueeze(-1)
        v = torch.stack([-u[:, 1], u[:, 0]], dim=-1)

        depth_err = (r * r + h * h) / h * d_pitch
        lat_err = r * d_yaw
        return u * depth_err.unsqueeze(-1) + v * lat_err.unsqueeze(-1)

    # ----------------------------------------------------------------- update
    @torch.no_grad()
    def update(self, robot: "Articulation", ball_pos_w: torch.Tensor) -> None:
        """Advance one control step.

        Args:
            robot: Articulation holding the camera body.
            ball_pos_w: World-frame ball position. Shape (num_envs, 3).
        """
        cfg = self.cfg
        N = self.num_envs
        d = self.device

        # ----- Camera world pose ----------------------------------------
        body_pos = robot.data.body_pos_w[:, self._head_idx]
        body_quat = robot.data.body_quat_w[:, self._head_idx]

        offset_pos_b = self._cam_offset_pos.expand(N, -1)
        offset_quat_b = self._cam_offset_quat.expand(N, -1)

        cam_pos_w = body_pos + quat_apply(body_quat, offset_pos_b)
        cam_quat_w = quat_mul(body_quat, offset_quat_b)

        # 頭追従モード: カメラ位置は頭のままだが、向きを「カメラ→ボール」に上書きする。
        # 戦略層が首を振ってボールを画面中心に捉える実機挙動を模擬する。これにより
        # FOV ゲートは実質常にパス (ボールが真後ろでない限り) し、遅延・ノイズ・検出率・
        # occlusion・dead-zone の品質劣化だけが効く。yaw のみ回す (光軸をボール方位へ)。
        if cfg.head_tracks_ball:
            to_ball = ball_pos_w - cam_pos_w
            yaw_to_ball = torch.atan2(to_ball[:, 1], to_ball[:, 0])
            half = 0.5 * yaw_to_ball
            look_quat = torch.zeros(N, 4, device=d, dtype=cam_quat_w.dtype)
            look_quat[:, 0] = torch.cos(half)  # w
            look_quat[:, 3] = torch.sin(half)  # z (world yaw)
            # 光軸をボール方位へ向けたうえで、元のカメラ姿勢の pitch/roll (下向き40°) を
            # 合成する。world-yaw の look_quat に、body 由来の下向きオフセットを右から掛ける。
            cam_quat_w = quat_mul(look_quat, offset_quat_b)

        self._cam_pos_w.copy_(cam_pos_w)
        self._cam_quat_w.copy_(cam_quat_w)

        # ----- Ball in camera optical frame -----------------------------
        ball_in_cam = quat_apply_inverse(cam_quat_w, ball_pos_w - cam_pos_w)
        self._ball_in_cam = ball_in_cam
        bx = ball_in_cam[:, 0]
        by = ball_in_cam[:, 1]
        bz = ball_in_cam[:, 2]
        distance = ball_in_cam.norm(dim=-1)

        # ----- FOV --------------------------------------------------------
        forward = bx > 1e-3
        yaw = torch.atan2(by, bx)
        pitch = torch.atan2(bz, bx)
        in_fov = forward & (yaw.abs() < self._fov_h_half) & (pitch.abs() < self._fov_v_half)

        # ----- FOV dead zone (static per-episode blind region) ----------
        # A ball inside the angular rectangle is blocked even though it is
        # geometrically in-FOV. Reappears as soon as it leaves the region.
        in_dead = (
            self._deadzone_active
            & ((yaw - self._deadzone_yaw_c).abs() < self._deadzone_yaw_h)
            & ((pitch - self._deadzone_pitch_c).abs() < self._deadzone_pitch_h)
        )
        self._in_deadzone = in_dead.float()
        in_fov = in_fov & (~in_dead)
        self._in_fov = in_fov.float()

        # ----- Range probability (linear falloff) -----------------------
        range_prob = torch.where(
            distance < cfg.max_detection_range,
            torch.ones_like(distance),
            torch.clamp(
                1.0 - (distance - cfg.max_detection_range) / max(cfg.range_decay, 1e-6),
                min=0.0,
            ),
        )

        # ----- Temporal occlusion (contiguous dropout streaks) ----------
        # Decrement active timers, then start new events on currently-clear
        # envs at the per-step hazard rate. While occluded, detection is
        # forced to zero for the whole sampled duration.
        if cfg.occlusion_prob > 0.0:
            was_active = self._occlusion_steps_remaining > 0
            self._occlusion_steps_remaining = (
                self._occlusion_steps_remaining - 1
            ).clamp_min(0)
            start = (~was_active) & (torch.rand(N, device=d) < cfg.occlusion_prob)
            lo, hi = cfg.occlusion_duration_range
            dur_steps = (
                (torch.rand(N, device=d) * (hi - lo) + lo) / self.dt
            ).round().long().clamp_min(1)
            self._occlusion_steps_remaining = torch.where(
                start, dur_steps, self._occlusion_steps_remaining
            )
            occluded = self._occlusion_steps_remaining > 0
        else:
            occluded = torch.zeros(N, dtype=torch.bool, device=d)
        self._occluded = occluded.float()

        # ----- Bernoulli detection --------------------------------------
        p_detect = (
            self._detection_prob_per_env * range_prob * in_fov.float()
            * (~occluded).float()
        )
        detected = torch.bernoulli(p_detect.clamp(0.0, 1.0)) > 0.5
        detected_f = detected.float()
        self._range_prob = range_prob.clamp(0.0, 1.0)
        self._detect_prob = p_detect.clamp(0.0, 1.0)
        self._raw_detected = detected_f

        # ----- xy noise (world frame, then yaw-rotate into body frame) --
        if cfg.attitude_noise:
            ball_xy_world = ball_pos_w[:, :2] + self._projection_error(cam_pos_w, ball_pos_w)
        else:
            sigma = self._noise_a_per_env * distance + self._noise_b_per_env
            noise = torch.randn_like(ball_pos_w[:, :2]) * sigma.unsqueeze(-1)
            ball_xy_world = ball_pos_w[:, :2] + noise

        robot_pos_w = robot.data.root_pos_w
        robot_quat_w = robot.data.root_quat_w
        yq = yaw_quat(robot_quat_w)
        rel_xyz = torch.zeros(N, 3, device=d, dtype=ball_xy_world.dtype)
        rel_xyz[:, :2] = ball_xy_world - robot_pos_w[:, :2]
        rel_xy_b = quat_apply_inverse(yq, rel_xyz)[:, :2]

        # ----- Decimation + reset-on-miss fast lock ---------------------
        new_pos = torch.where(detected.unsqueeze(-1), rel_xy_b, self._last_pos)

        self._steps_since_update += 1
        do_update = (self._steps_since_update >= self._update_period_steps) | (
            self._last_mask < 0.5
        )

        # On an update tick where we actually detected, replace last_pos.
        self._last_pos = torch.where(
            (do_update & detected).unsqueeze(-1), new_pos, self._last_pos
        )
        # On an update tick, mask = current detection result (could be 0).
        self._last_mask = torch.where(do_update, detected_f, self._last_mask)
        # Reset the counter on update ticks.
        self._steps_since_update = torch.where(
            do_update, torch.zeros_like(self._steps_since_update), self._steps_since_update
        )

        # Bump the sequence number only when a genuinely new detection landed.
        self._meas_seq = self._meas_seq + (do_update & detected).long()

        # ----- Ring buffer write ----------------------------------------
        self._buffer_head = (self._buffer_head + 1) % cfg.buffer_size
        self._buffer_pos[self._buffer_head] = self._last_pos
        self._buffer_mask[self._buffer_head] = self._last_mask
        self._buffer_seq[self._buffer_head] = self._meas_seq

        # ----- Ring buffer read at per-env latency ----------------------
        env_idx = torch.arange(N, device=d)
        read_head = (self._buffer_head - self._latency_steps) % cfg.buffer_size
        out_pos = self._buffer_pos[read_head, env_idx]
        out_mask = self._buffer_mask[read_head, env_idx]
        out_seq = self._buffer_seq[read_head, env_idx]

        self._fresh = ((out_seq != self._prev_out_seq) & (out_mask > 0.5)).float()
        self._prev_out_seq = out_seq

        if not cfg.hold_last_on_miss:
            out_pos = out_pos * out_mask.unsqueeze(-1)

        self._ball_pos_b = out_pos
        self._ball_mask = out_mask

        # ----- last_seen_dt counter -------------------------------------
        # Reset when a detection lands (out_mask == 1), increment otherwise.
        self._last_seen_dt = torch.where(
            out_mask > 0.5,
            torch.zeros_like(self._last_seen_dt),
            self._last_seen_dt + self.dt,
        )

    # --------------------------------------------------------------- output
    @property
    def ball_pos_b(self) -> torch.Tensor:
        """Noisy ball xy in robot body-yaw frame. Shape (num_envs, 2)."""
        return self._ball_pos_b

    @property
    def ball_mask(self) -> torch.Tensor:
        """Float mask in {0, 1}; 1 when a (delayed) detection is available."""
        return self._ball_mask

    @property
    def fresh(self) -> torch.Tensor:
        """1 on the step a *new* detection surfaces from the latency buffer.

        Between detector ticks the buffer keeps emitting the held value, so
        downstream estimators must gate their measurement update on this flag.
        """
        return self._fresh

    @property
    def last_seen_dt(self) -> torch.Tensor:
        """Seconds since the most recent detection (per env)."""
        return self._last_seen_dt

    @property
    def in_fov(self) -> torch.Tensor:
        """Current, non-latent FOV gate before Bernoulli/dropout."""
        return self._in_fov

    @property
    def occluded(self) -> torch.Tensor:
        """Current temporal-occlusion gate (1 while an occlusion event is active)."""
        return self._occluded

    @property
    def in_deadzone(self) -> torch.Tensor:
        """Current FOV dead-zone gate (1 while the ball sits in the blind region)."""
        return self._in_deadzone

    @property
    def range_prob(self) -> torch.Tensor:
        """Range attenuation factor used in the current detection probability."""
        return self._range_prob

    @property
    def detect_prob(self) -> torch.Tensor:
        """Current Bernoulli detection probability before latency buffering."""
        return self._detect_prob

    @property
    def raw_detected(self) -> torch.Tensor:
        """Current unbuffered Bernoulli result before decimation/latency."""
        return self._raw_detected

    @property
    def ball_in_cam(self) -> torch.Tensor:
        """Current ball position in the camera optical frame."""
        return self._ball_in_cam

    @property
    def cam_pos_w(self) -> torch.Tensor:
        return self._cam_pos_w

    @property
    def cam_quat_w(self) -> torch.Tensor:
        return self._cam_quat_w
