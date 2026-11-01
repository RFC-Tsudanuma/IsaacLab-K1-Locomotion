# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# observations.py

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from .events import get_gait_phase
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# Helper Functions
##

def phase_obs(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
    low_speed: float = 1.0,
    high_speed: float = 1.8,
    low_freq: float = 1.5,
    high_freq: float = 2.0,
) -> torch.Tensor:
    """現在の歩行位相を sin/cos で返す (左足, 右足の計4次元)。

    位相の周波数は速度コマンドのノルムに応じて線形遷移する
    (:func:`mdp.events.compute_cmd_phase_freq` 参照)。
    コマンド速度が ``cmd_threshold`` 未満のときは位相をゼロで埋め、
    停止すべき状況であることをポリシーに明示する。
    """
    phase_left = get_gait_phase(env, command_name, low_speed, high_speed, low_freq, high_freq)
    phase_right = phase_left + math.pi

    phase = torch.stack([
        torch.sin(phase_left), torch.cos(phase_left),
        torch.sin(phase_right), torch.cos(phase_right),
    ], dim=1)

    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :3], dim=1, keepdim=True)
    is_stopped = cmd_speed < cmd_threshold
    phase = torch.where(is_stopped, torch.zeros_like(phase), phase)

    return phase

def ball_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ボールの線速度を、ロボットの base yaw frame で返す (x: 前後, y: 左右)。"""
    ball: Articulation = env.scene["soccer_ball"]
    robot: Articulation = env.scene[asset_cfg.name]
    vel_w = ball.data.root_com_vel_w[:, :3]
    vel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), vel_w)
    return vel_b[:, :2]

def ball_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ボールとロボットの相対位置を、ロボットの base yaw frame で返す (x: 前後, y: 左右)。"""
    ball: Articulation = env.scene["soccer_ball"]
    robot: Articulation = env.scene[asset_cfg.name]
    offset_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    offset_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), offset_w)
    return offset_b[:, :2]


def last_high_action(
    env: ManagerBasedRLEnv,
    action_dim: int = 3,
) -> torch.Tensor:
    """前回ステップで上位ポリシーが出力した行動 (歩行コマンド) を返す。

    ``HierarchicalVecEnvWrapper`` が毎ステップ ``env._prev_high_action``
    (shape ``(num_envs, action_dim)``) を更新する。リセット時は
    :func:`mdp.events.reset_prev_high_action` で対象 env のスロットが 0 にされる。
    未初期化のときは 0 を返す (起動時など)。
    """
    buf = getattr(env, "_prev_high_action", None)
    if buf is None or buf.shape != (env.num_envs, action_dim):
        return torch.zeros((env.num_envs, action_dim), device=env.device)
    return buf


def kick_direction_b(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ワールド座標系のキック方向コマンドをロボットの base yaw frame に回転して返す (2D 単位ベクトル)。"""
    robot: Articulation = env.scene[asset_cfg.name]
    kick_dir_w_xy = env.command_manager.get_term(command_name).command  # (N, 2)
    # 3D に拡張 (z=0) してから base yaw frame に回転
    z = torch.zeros_like(kick_dir_w_xy[:, :1])
    kick_dir_w = torch.cat([kick_dir_w_xy, z], dim=1)
    kick_dir_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), kick_dir_w)
    return kick_dir_b[:, :2]


def transition_mode(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """`TransitionCommand` の現在モード (0=walk / 1=turn) を (num_envs, 1) の float で返す。

    方策の入力には使わない (obs_groups に含めない) 補助観測。``MultiExpertPPO`` が
    「この env のアクションをどの expert が出すか」を決めるために読む。
    """
    return env.command_manager.get_term(command_name).mode.to(torch.float32).unsqueeze(1)


# ---------------------------------------------------------------------------
# ドリブル (Isaac-Velocity-Flat-K1-Dribble-Target-v0) 用の観測
# ---------------------------------------------------------------------------
# 座標系はすべて「body yaw 座標」: ロボットの yaw だけを打ち消した座標系
# (roll/pitch はそのまま残る)。offset_b = quat_apply_inverse(yaw_quat(robot_quat_w), p_w - robot_pos_w)
# の xy 成分。実機側は vision_filter / localization の値を同じ座標系に直して入力する。


def _to_body_yaw_xy(robot: Articulation, vec_w_xy: torch.Tensor) -> torch.Tensor:
    """ワールド xy ベクトル (N, 2) を body yaw 座標の xy に回す。"""
    vec_w = torch.cat([vec_w_xy, torch.zeros_like(vec_w_xy[:, :1])], dim=1)
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), vec_w)[:, :2]


# ---------------------------------------------------------------------------
# 観測の遅れ (2026-10-26): ボール検出の遅れ・自己位置の遅れの再現
# ---------------------------------------------------------------------------
# 実機では、ボールの観測は検出の遅れ (約 100ms)、目標点の相対位置は自己位置の遅れ (約 170ms) を含む。観測ベクトルを
# そのまま遅らせるのではなく、毎制御周期ワールド座標の量 (ボールの位置・速度、ロボットの位置・向き) を履歴に残し、
# 遅れた量と姿勢から body yaw 座標の値を作り直す (遅れている間もロボットは動いたり向きを変えたりしているため):
#
#   目標点:         R(ψ(t - d_loc))ᵀ (W(t) - p(t - d_loc)) … 今の目標点を、遅れた自己位置 (位置・向き) で見る (実機と同じ)
#   ボールの観測 (ball_delay_mode で選ぶ):
#     "capture" (実機・MuJoCo と同じ、2026-10-26 確認): 撮影した時刻のロボット座標で見た、撮影した時刻のボール
#         ball_pos_rel = R(ψ(t - d))ᵀ (b(t - d) - p(t - d))、ball_vel = R(ψ(t - d))ᵀ v_b(t - d)
#         foot_*_ball_pos_rel = (上の ball_pos_rel) - (今の足の位置を今のロボット座標で見たもの)  … 時刻のずれた 2 つが混ざる
#         (実機: vision_filter は撮影時刻の姿勢でフィールド座標に直してフィルタし、同じ姿勢でロボット座標に戻す。
#          推論ノードはその値をそのまま使い、足の位置は今の関節角の FK)
#     "current" (2026-10-25 のランで使った作り方): 遅れたボール位置を、今のロボットの姿勢で見る R(ψ(t))ᵀ (b(t - d) - p(t))
#
# d_ball / d_loc は env ごと・エピソードごとに mdp.events.randomize_observation_delays が引く (env._obs_delay_s)。
# 20ms (制御周期) の間は線形補間する。critic は遅れの無い真値のまま。

_OBS_HIST_LEN = 16  # 制御周期 × 15 = 300ms まで


class _ObsDelayHistory:
    """ワールド座標の量の履歴 (index 0 が今)。1 制御周期に 1 回だけ進める。"""

    def __init__(self, env: ManagerBasedRLEnv, ball_name: str, robot_name: str):
        n, dev = env.num_envs, env.device
        # 列: ボール位置 xy, ボール速度 xy, ロボット位置 xy, ロボットの向き (cos ψ, sin ψ)
        self.buf = torch.zeros(n, _OBS_HIST_LEN, 8, device=dev)
        self.step = -1
        self.ball_name, self.robot_name = ball_name, robot_name

    def _current(self, env) -> torch.Tensor:
        ball = env.scene[self.ball_name]
        robot = env.scene[self.robot_name]
        q = robot.data.root_quat_w  # (w, x, y, z)
        yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        return torch.cat([
            ball.data.root_pos_w[:, :2], ball.data.root_com_vel_w[:, :2],
            robot.data.root_pos_w[:, :2], torch.stack([torch.cos(yaw), torch.sin(yaw)], 1),
        ], 1)

    def update(self, env):
        cur = self._current(env)
        step = int(env.common_step_counter)
        if step != self.step:
            self.buf = torch.roll(self.buf, shifts=1, dims=1)
            self.step = step
        self.buf[:, 0] = cur
        fresh = env.episode_length_buf == 0  # エピソードの開始直後は履歴全体を今の値で埋める
        if fresh.any():
            self.buf[fresh] = cur[fresh].unsqueeze(1)

    def delayed(self, delay_s: torch.Tensor, step_dt: float) -> torch.Tensor:
        """delay_s (N,) 秒前の値 (N, 8) を線形補間で返す。"""
        d = (delay_s / step_dt).clamp(0.0, _OBS_HIST_LEN - 1.001)
        i0 = torch.floor(d).long()
        f = (d - i0.float()).unsqueeze(1)
        idx = torch.arange(len(d), device=d.device)
        return (1.0 - f) * self.buf[idx, i0] + f * self.buf[idx, i0 + 1]


def _obs_history(env: ManagerBasedRLEnv, ball_name: str = "soccer_ball", robot_name: str = "robot") -> _ObsDelayHistory:
    hist = env.__dict__.get("_obs_delay_history")
    if hist is None:
        hist = _ObsDelayHistory(env, ball_name, robot_name)
        env.__dict__["_obs_delay_history"] = hist
    hist.update(env)
    return hist


def _obs_delays(env: ManagerBasedRLEnv) -> torch.Tensor:
    """env ごとの遅れ [秒] (N, 2): ボール検出, 自己位置。未設定なら 0。"""
    d = env.__dict__.get("_obs_delay_s")
    return d if d is not None else torch.zeros(env.num_envs, 2, device=env.device)


def _rotate_to_yaw(vec_xy: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """ワールド xy ベクトルを、向き (cos ψ, sin ψ) の body yaw 座標に回す。"""
    c, s_ = cos_sin[:, 0], cos_sin[:, 1]
    norm = torch.sqrt(c * c + s_ * s_).clamp(min=1e-6)  # 補間で長さが 1 から少しずれるので正規化
    c, s_ = c / norm, s_ / norm
    return torch.stack([c * vec_xy[:, 0] + s_ * vec_xy[:, 1], -s_ * vec_xy[:, 0] + c * vec_xy[:, 1]], 1)


def _corruption_enabled(env: ManagerBasedRLEnv) -> bool:
    policy_cfg = getattr(env.cfg.observations, "policy", None)
    return bool(getattr(policy_cfg, "enable_corruption", True))


def delayed_ball_xy(env: ManagerBasedRLEnv, what: str = "pos") -> torch.Tensor:
    """ボール検出の遅れ d_ball だけ前のボールのワールド位置 (what="pos") / 速度 (what="vel") の xy。"""
    hist = _obs_history(env)
    past = hist.delayed(_obs_delays(env)[:, 0], env.step_dt)
    return past[:, 0:2] if what == "pos" else past[:, 2:4]


def ball_rel_capture(env: ManagerBasedRLEnv, what: str = "pos") -> torch.Tensor:
    """撮影した時刻 (d_ball 前) のロボット座標で見た、撮影した時刻のボールの位置 / 速度 (body yaw 座標の xy)。"""
    hist = _obs_history(env)
    past = hist.delayed(_obs_delays(env)[:, 0], env.step_dt)
    vec = past[:, 0:2] - past[:, 4:6] if what == "pos" else past[:, 2:4]
    return _rotate_to_yaw(vec, past[:, 6:8])


# ---------------------------------------------------------------------------
# vision_filter のボールのフィルタと推論ノードの模倣 (2026-11-01、ball_delay_mode="filter")
# ---------------------------------------------------------------------------
# 実機・MuJoCo の vision_filter (config/vision_filter.yaml) と推論ノードのコードで確認した仕様:
# - 状態はフィールド座標の [x, y, vx, vy]。観測は検出の位置だけ。検出が届くたび (25 Hz = 2 制御周期に 1 回、
#   env ごとに位相はランダム) に、撮影時刻の差を dt として予測・補正する。出力は撮影時刻の推定 (今の時刻へは外挿しない)。
# - 等速モデル。4 仮説: 停止 (v=0 固定) / 転がり / 高速 (転がりと同じ計算) / 反発 (MAP でない間は MAP 状態から v' = -v
#   で作り直す)。出力は「事前確率 × 補正時の尤度」が最大の 1 つ (MAP)。遷移確率は対角 0.95、他 0.0167。
# - Q: 白色加速度 σa。R = d[m] × R_cm2 [cm²] (フィールド座標)。初期化は位置 = 観測値、速度 0、σ 0.25 m / 2.5 m/s。
# - 座標変換の姿勢: 撮影時刻ではなく、検出が届いた制御周期に見えている遅れた自己位置 (target_pos_rel の
#   localization_delay と同じもの)。実機では自己位置の遅れがボールの遅れより大きく、撮影時刻の自己位置はまだ届いて
#   いないため、その時点で最新の自己位置が使われる。検出 → フィールド座標も、出力 → ロボット座標も同じ姿勢 (速度は回転だけ)。
# - NIS ゲート (全仮説で超えたら捨てる)。公開中の推定があれば、捨てた回は予測だけ進めて、その予測 (位置・速度) を出す。
#   並行して捨てた観測から新しい推定を作り、2 回続けてゲートを通ったら入れ替えて公開する。
# - lost_s 受け入れなければ LOST。公開中の推定が無い間は、推論ノードは生の検出の位置を使い、速度は 0。
# - 推論ノードは次のメッセージまで最後の値を使い続ける。エピソードの開始では、その時のボールで初期化してすぐ公開する
#   (MuJoCo は 1.5 s 静止させてから始めるので同じ)。
# 後から足す予定の 2 つ (既定は 0 で切ってある):
# - 検出の位置のノイズ: 共分散 s² · d · R、s は env ごとに U(det_noise_scale_range)
# - 距離に応じた検出の取りこぼし: 確率は near_drop_prob (d <= near_distance) → far_drop_prob (d >= far_distance) を線形補間
# critic は真値のまま。policy の enable_corruption=False (PLAY など) では使わない。

BALL_FILTER_DEFAULTS = {
    "period_steps": 2,  # 25 Hz
    "sigma_a": 0.8,
    "r_cm2": ((477.7, -4.1), (-4.1, 153.1)),
    "p0_pos": 0.25, "p0_vel": 2.5,
    "gate": 9.21,
    "stay_prob": 0.95, "switch_prob": 0.0167,
    "lost_s": 3.0,
    "det_noise_scale_range": (0.0, 0.0),
    "near_distance": 0.35, "far_distance": 2.0, "near_drop_prob": 0.0, "far_drop_prob": 0.0,
}
_KF_STOP, _KF_ROLL, _KF_FAST, _KF_BOUNCE = 0, 1, 2, 3
_KF_H = 4


def _from_yaw(vec_xy: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """body yaw 座標の xy をワールドに回す (_rotate_to_yaw の逆)。"""
    c, s_ = cos_sin[:, 0], cos_sin[:, 1]
    norm = torch.sqrt(c * c + s_ * s_).clamp(min=1e-6)
    c, s_ = c / norm, s_ / norm
    return torch.stack([c * vec_xy[:, 0] - s_ * vec_xy[:, 1], s_ * vec_xy[:, 0] + c * vec_xy[:, 1]], 1)


class _BallKFBank:
    """4 仮説のフィルタ 1 つ分 (env ごと)。"""

    def __init__(self, n: int, dev, p: dict):
        self.n, self.dev, self.p = n, dev, p
        self.x = torch.zeros(n, _KF_H, 4, device=dev)
        self.P = torch.zeros(n, _KF_H, 4, 4, device=dev)
        self.logp = torch.full((n, _KF_H), -math.log(_KF_H), device=dev)
        self.map = torch.full((n,), _KF_ROLL, dtype=torch.long, device=dev)
        self.t = torch.zeros(n, device=dev)  # 状態の時刻 (撮影時刻) [s]
        T = torch.full((_KF_H, _KF_H), p["switch_prob"], device=dev)
        T.fill_diagonal_(p["stay_prob"])
        self.logT = torch.log(T / T.sum(1, keepdim=True))
        self.P0 = torch.diag(torch.tensor([p["p0_pos"] ** 2] * 2 + [p["p0_vel"] ** 2] * 2, device=dev))
        self.idx = torch.arange(n, device=dev)

    def init(self, m: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        mx = m.view(-1, 1, 1)
        x0 = torch.zeros_like(self.x)
        x0[..., 0:2] = z.unsqueeze(1)
        self.x = torch.where(mx, x0, self.x)
        self.P = torch.where(mx.unsqueeze(-1), self.P0.expand_as(self.P), self.P)
        self.logp = torch.where(m.unsqueeze(1), torch.full_like(self.logp, -math.log(_KF_H)), self.logp)
        self.map = torch.where(m, torch.full_like(self.map, _KF_ROLL), self.map)
        self.t = torch.where(m, t, self.t)
        self._clamp_stop()

    def copy_from(self, other: "_BallKFBank", m: torch.Tensor):
        mx = m.view(-1, 1, 1)
        self.x = torch.where(mx, other.x, self.x)
        self.P = torch.where(mx.unsqueeze(-1), other.P, self.P)
        self.logp = torch.where(m.unsqueeze(1), other.logp, self.logp)
        self.map = torch.where(m, other.map, self.map)
        self.t = torch.where(m, other.t, self.t)

    def _clamp_stop(self):
        self.x[:, _KF_STOP, 2:] = 0.0
        self.P[:, _KF_STOP, 2:, :] = 0.0
        self.P[:, _KF_STOP, :, 2:] = 0.0

    def map_state(self) -> torch.Tensor:
        return self.x[self.idx, self.map]

    def predict(self, m: torch.Tensor, t: torch.Tensor):
        """m の env を時刻 t まで予測する。反発の仮説は MAP でなければ MAP 状態から v' = -v で作り直す。"""
        dt = torch.where(m, t - self.t, torch.zeros_like(t)).clamp(min=0.0)
        rb = (m & (self.map != _KF_BOUNCE)).view(-1, 1)
        xb = self.map_state().clone()
        xb[:, 2:] *= -1.0
        self.x[:, _KF_BOUNCE] = torch.where(rb, xb, self.x[:, _KF_BOUNCE])
        self.P[:, _KF_BOUNCE] = torch.where(rb.unsqueeze(-1), self.P[self.idx, self.map], self.P[:, _KF_BOUNCE])
        F = torch.eye(4, device=self.dev).repeat(self.n, 1, 1)
        F[:, 0, 2] = dt
        F[:, 1, 3] = dt
        q = self.p["sigma_a"] ** 2
        Q = torch.zeros(self.n, 4, 4, device=self.dev)
        for a in (0, 1):
            Q[:, a, a] = dt ** 4 / 4 * q
            Q[:, a, a + 2] = dt ** 3 / 2 * q
            Q[:, a + 2, a] = dt ** 3 / 2 * q
            Q[:, a + 2, a + 2] = dt ** 2 * q
        self.x = torch.einsum("nij,nhj->nhi", F, self.x)
        self.P = F.unsqueeze(1) @ self.P @ F.transpose(1, 2).unsqueeze(1) + Q.unsqueeze(1)
        self.t = torch.where(m, t, self.t)
        self._clamp_stop()

    def correct(self, m: torch.Tensor, z: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """m の env を補正する。ゲートを通らなかった env は変えない。戻り値: 受け入れたか (N,)。"""
        y = z.unsqueeze(1) - self.x[..., 0:2]
        S = self.P[..., 0:2, 0:2] + R.unsqueeze(1)
        Si = torch.linalg.inv(S)
        nis = torch.einsum("nhi,nhij,nhj->nh", y, Si, y)
        ok = (nis <= self.p["gate"]).any(1) & m
        loglik = -0.5 * nis - 0.5 * torch.logdet(S)
        prior = torch.logsumexp(self.logp.unsqueeze(2) + self.logT.unsqueeze(0), dim=1)
        post = prior + loglik
        post = post - torch.logsumexp(post, dim=1, keepdim=True)
        K = self.P[..., :, 0:2] @ Si
        o = ok.view(-1, 1, 1)
        self.x = torch.where(o, self.x + torch.einsum("nhij,nhj->nhi", K, y), self.x)
        self.P = torch.where(o.unsqueeze(-1), self.P - K @ self.P[..., 0:2, :], self.P)
        self.logp = torch.where(ok.unsqueeze(1), post, self.logp)
        self.map = torch.where(ok, post.argmax(1), self.map)
        self._clamp_stop()
        return ok


class _BallVisionFilter:
    """検出 → vision_filter → 推論ノードが使う値 (ロボット座標のボールの位置・速度) を、1 制御周期に 1 回作る。"""

    def __init__(self, env: ManagerBasedRLEnv, params: dict | None):
        p = dict(BALL_FILTER_DEFAULTS)
        p.update(params or {})
        n, dev = env.num_envs, env.device
        self.p, self.n, self.dev = p, n, dev
        self.main, self.cand = _BallKFBank(n, dev, p), _BallKFBank(n, dev, p)
        self.has_main = torch.zeros(n, dtype=torch.bool, device=dev)  # 公開中の推定がある
        self.has_cand = torch.zeros(n, dtype=torch.bool, device=dev)
        self.cand_n = torch.zeros(n, dtype=torch.long, device=dev)  # 新しい推定がゲートを続けて通った回数
        self.t_seen = torch.zeros(n, device=dev)
        self.phase = torch.randint(0, p["period_steps"], (n,), device=dev)
        self.noise_scale = torch.zeros(n, device=dev)
        self.R_cm2 = torch.tensor(p["r_cm2"], device=dev) * 1e-4  # m²
        self.out_pos = torch.zeros(n, 2, device=dev)
        self.out_vel = torch.zeros(n, 2, device=dev)
        self.step = -1

    def update(self, env: ManagerBasedRLEnv):
        step = int(env.common_step_counter)
        fresh = env.episode_length_buf == 0
        # 同じ制御周期の 2 回目以降は、エピソードの開始の初期化だけ行う (観測の次元を調べる呼び出しと
        # 最初のリセットが同じ common_step_counter で来るため、ここで飛ばすと古いボールで始まる)
        same_step = step == self.step
        if same_step and not bool(fresh.any()):
            return
        self.step = step
        p = self.p
        hist = _obs_history(env)
        delays = _obs_delays(env)
        cap = hist.delayed(delays[:, 0], env.step_dt)  # 撮影時刻
        rel_c = _rotate_to_yaw(cap[:, 0:2] - cap[:, 4:6], cap[:, 6:8])  # 検出 (撮影時刻のロボット座標)
        d = torch.norm(rel_c, dim=1)
        t_cap = env.episode_length_buf.float() * env.step_dt - delays[:, 0]
        loc = hist.delayed(delays[:, 1], env.step_dt)  # その時点で最新の自己位置
        p_l, cs_l = loc[:, 4:6], loc[:, 6:8]
        R = self.R_cm2.unsqueeze(0) * d.clamp(min=0.05).view(-1, 1, 1)  # (N, 2, 2) フィールド座標
        lo, hi = p["det_noise_scale_range"]
        self.phase = torch.where(fresh, torch.randint_like(self.phase, 0, p["period_steps"]), self.phase)
        self.noise_scale = torch.where(fresh, torch.empty_like(self.noise_scale).uniform_(lo, hi), self.noise_scale)
        z = p_l + _from_yaw(rel_c, cs_l)  # フィールド座標の検出
        if hi > 0.0:  # 検出の位置のノイズ: 共分散 s² · d · R
            L = torch.linalg.cholesky(R)
            z = z + self.noise_scale.unsqueeze(1) * (L @ torch.randn_like(z).unsqueeze(-1)).squeeze(-1)
        det = ((env.episode_length_buf + self.phase) % p["period_steps"] == 0) & ~fresh & ~torch.full_like(fresh, same_step)
        if p["near_drop_prob"] > 0.0 or p["far_drop_prob"] > 0.0:  # 距離に応じた取りこぼし
            alpha = ((d - p["near_distance"]) / max(p["far_distance"] - p["near_distance"], 1e-6)).clamp(0.0, 1.0)
            drop = torch.rand_like(d) < p["near_drop_prob"] + alpha * (p["far_drop_prob"] - p["near_drop_prob"])
            det = det & ~drop
        # エピソードの開始: その時のボールで初期化して、すぐ公開する
        self.main.init(fresh, z, t_cap)
        self.has_main |= fresh
        self.has_cand &= ~fresh
        self.t_seen = torch.where(fresh, t_cap, self.t_seen)
        # 公開中の推定: 予測して補正 (ゲートで捨てた回は予測のまま)
        live = det & self.has_main
        self.main.predict(live, t_cap)
        ok_m = self.main.correct(live, z, R)
        self.t_seen = torch.where(ok_m, t_cap, self.t_seen)
        # 新しい推定: 公開中の推定が受け入れなかった検出で作る・育てる
        need = det & ~ok_m
        c_live = need & self.has_cand
        self.cand.predict(c_live, t_cap)
        ok_c = self.cand.correct(c_live, z, R)
        self.cand_n = torch.where(ok_c, self.cand_n + 1, self.cand_n)
        restart = need & ~ok_c
        self.cand.init(restart, z, t_cap)
        self.has_cand |= restart
        self.cand_n = torch.where(restart, torch.zeros_like(self.cand_n), self.cand_n)
        swap = need & ok_c & (self.cand_n >= 2)
        self.main.copy_from(self.cand, swap)
        self.has_main |= swap
        self.t_seen = torch.where(swap, t_cap, self.t_seen)
        clear = ok_m | swap
        self.has_cand &= ~clear
        self.cand_n = torch.where(clear, torch.zeros_like(self.cand_n), self.cand_n)
        lost = det & self.has_main & (t_cap - self.t_seen > p["lost_s"])
        self.has_main &= ~lost
        # 推論ノードが使う値 (次の検出まで保つ)。公開中の推定が無ければ、生の検出の位置と速度 0
        new = det | fresh
        xm = self.main.map_state()
        use_kf = (new & self.has_main).unsqueeze(1)
        use_raw = (new & ~self.has_main).unsqueeze(1)
        pos_kf = _rotate_to_yaw(xm[:, 0:2] - p_l, cs_l)
        vel_kf = _rotate_to_yaw(xm[:, 2:4], cs_l)
        pos_raw = _rotate_to_yaw(z - p_l, cs_l)
        self.out_pos = torch.where(use_kf, pos_kf, torch.where(use_raw, pos_raw, self.out_pos))
        self.out_vel = torch.where(use_kf, vel_kf, torch.where(use_raw, torch.zeros_like(vel_kf), self.out_vel))


def ball_rel_filtered(env: ManagerBasedRLEnv, what: str = "pos", filter_params: dict | None = None) -> torch.Tensor:
    """vision_filter と推論ノードを通したボールの位置 / 速度 (body yaw 座標の xy)。filter は env に 1 つだけ作る。"""
    filt = env.__dict__.get("_ball_vision_filter")
    if filt is None:
        filt = _BallVisionFilter(env, filter_params)
        env.__dict__["_ball_vision_filter"] = filt
    filt.update(env)
    return filt.out_pos if what == "pos" else filt.out_vel


def robot_yaw_cos_sin(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _obs_history(env).buf[:, 0, 6:8]


def zero_obs(env: ManagerBasedRLEnv, dim: int = 2) -> torch.Tensor:
    """常に 0 を返す観測 (Stage1 でドリブル用スロットを埋めるため)。

    観測ベクトルの形を Stage2 と同一に保つために使う。ObsTerm にノイズを付けたまま
    使うと「0 + ノイズ」が観測されて正規化器の統計が壊れるので、zero_obs に差し替える
    項は必ず noise=None にすること (k1_dribble_target_env_cfg の Stage1 参照)。
    """
    return torch.zeros((env.num_envs, dim), device=env.device)


def target_pos_rel(
    env: ManagerBasedRLEnv,
    command_name: str = "dribble_target",
    localization_delay: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """目標点 (ワールド xy、現在のウェイポイント) のロボットからの相対位置 [m] (body yaw 座標)。

    localization_delay=True (policy 用) では、自己位置の遅れ d_loc を入れる:
    今の目標点 W を、d_loc 前のロボットの位置・向きで見る R(ψ(t - d_loc))ᵀ (W - p(t - d_loc))。
    policy の enable_corruption=False (再生など) では遅れを入れない。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    target_w = env.command_manager.get_term(command_name).pos_command_w[:, :2]
    if localization_delay and _corruption_enabled(env):
        past = _obs_history(env).delayed(_obs_delays(env)[:, 1], env.step_dt)
        return _rotate_to_yaw(target_w - past[:, 4:6], past[:, 6:8])
    return _to_body_yaw_xy(robot, target_w - robot.data.root_pos_w[:, :2])


def foot_to_ball_pos_rel(
    env: ManagerBasedRLEnv,
    foot_body_name: str,
    ball_delay: bool = False,
    ball_delay_mode: str = "current",
    filter_params: dict | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """足 (foot_link の原点) からボール中心への相対位置 [m] (body yaw 座標、xy のみ)。

    ball_delay=True (policy 用) では、ボールの位置だけ検出の遅れ d_ball 前のものを使う (足の位置と向きは今のもの)。

    ``offset_b = yaw_quat(robot_quat)^-1 · (ball_pos_w[:2] − foot_link_pos_w[:2])``。
    足の点は足裏中心ではなく foot_link の原点、ボールは中心の xy のみ (z は含めない)、
    符号は ball − foot。既定姿勢での foot_link の Trunk 基準位置は
    左 (-0.0155, +0.0962, -0.4996) / 右 (-0.0155, -0.0962, -0.4996) m (実機 FK と一致済み)。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    cache = env.__dict__.setdefault("_foot_body_idx_cache", {})
    if foot_body_name not in cache:
        ids, _ = robot.find_bodies(foot_body_name)
        if len(ids) != 1:
            raise ValueError(f"foot_to_ball_pos_rel: body '{foot_body_name}' が一意に見つかりません: {ids}")
        cache[foot_body_name] = ids[0]
    foot_pos_w = robot.data.body_pos_w[:, cache[foot_body_name], :2]
    if ball_delay and _corruption_enabled(env):
        if ball_delay_mode == "filter":  # 実機と同じ: (vision_filter のボール) - (今のロボット座標で見た今の足)
            foot_b = _to_body_yaw_xy(robot, foot_pos_w - robot.data.root_pos_w[:, :2])
            return ball_rel_filtered(env, "pos", filter_params) - foot_b
        if ball_delay_mode == "capture":
            # 実機と同じ: (撮影時刻のロボット座標で見たボール) - (今のロボット座標で見た今の足)
            foot_b = _to_body_yaw_xy(robot, foot_pos_w - robot.data.root_pos_w[:, :2])
            return ball_rel_capture(env, "pos") - foot_b
        return _to_body_yaw_xy(robot, delayed_ball_xy(env, "pos") - foot_pos_w)
    return _to_body_yaw_xy(robot, ball.data.root_pos_w[:, :2] - foot_pos_w)


def obstacle_pos_rel(
    env: ManagerBasedRLEnv,
    obstacle_cfgs: list[SceneEntityCfg],
    max_distance: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """最も近い障害物 (中心) のロボットからの相対位置 [m] (body yaw 座標)。critic 専用の特権情報。

    障害物の数に依らず「最も近い 1 個」だけを返すので次元は 2 で固定。障害物の無い
    レベルでは遠方に退避しているため、ノルムを ``max_distance`` にクリップして
    正規化器の統計が退避位置 (~100m) に引きずられないようにする。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    robot_xy = robot.data.root_pos_w[:, :2]
    rel = torch.stack([env.scene[cfg.name].data.root_pos_w[:, :2] - robot_xy for cfg in obstacle_cfgs], dim=1)
    dist = torch.norm(rel, dim=-1)  # (N, K)
    nearest = torch.argmin(dist, dim=1)
    rel_nearest = rel[torch.arange(env.num_envs, device=env.device), nearest]
    norm = torch.norm(rel_nearest, dim=1, keepdim=True).clamp(min=1e-6)
    rel_nearest = rel_nearest * torch.clamp(max_distance / norm, max=1.0)
    return _to_body_yaw_xy(robot, rel_nearest)


class _BallVisionTerm(ManagerTermBase):
    """実機 vision_filter のボール観測の劣化を再現する観測の共通部分。

    ボールまでの距離 d に応じて劣化の確率を線形補間する
    (d <= near_distance で near_*、d >= far_distance で far_*)。
    足元のボールはカメラの死角に入るため、近いほど劣化が大きい。

    * ホールド: 確率 p_hold で「前回出力した値」をそのまま出す。毎ステップ評価するので連鎖しうる。
    * 白色ノイズは観測関数の中で掛ける (ObsTerm の noise は使わない)。ObsTerm の noise は
      関数の戻り値の後に掛かるため、ホールドした値や「厳密に 0」の値にもノイズが乗って
      しまい、実機 (同じ値の繰り返し / ちょうど 0) と食い違うため。
    * policy グループの enable_corruption=False (PLAY など) のときは劣化させず真値を返す。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_out = torch.zeros((env.num_envs, 2), device=env.device)
        self._has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._has_prev[env_ids] = False
        self._prev_out[env_ids] = 0.0

    def _corruption_enabled(self, env: ManagerBasedRLEnv) -> bool:
        policy_cfg = getattr(env.cfg.observations, "policy", None)
        return bool(getattr(policy_cfg, "enable_corruption", True))

    @staticmethod
    def _lerp_prob(dist: torch.Tensor, near_distance, far_distance, near_prob, far_prob) -> torch.Tensor:
        alpha = ((dist - near_distance) / max(far_distance - near_distance, 1e-6)).clamp(0.0, 1.0)
        return near_prob + alpha * (far_prob - near_prob)

    def _ball_dist_and_robot(self, env: ManagerBasedRLEnv, asset_cfg, ball_cfg):
        robot: Articulation = env.scene[asset_cfg.name]
        ball = env.scene[ball_cfg.name]
        offset_w = ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
        return robot, ball, offset_w, torch.norm(offset_w, dim=1)

    def _hold(self, out: torch.Tensor, p_hold: torch.Tensor) -> torch.Tensor:
        hold = (torch.rand_like(p_hold) < p_hold) & self._has_prev
        out = torch.where(hold.unsqueeze(-1), self._prev_out, out)
        self._prev_out = out.clone()
        self._has_prev[:] = True
        return out


class ball_pos_rel_vision(_BallVisionTerm):
    """ボール中心の xy 相対位置 [m] (body yaw 座標)。距離依存のホールドとノイズ付き。"""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        near_distance: float = 0.35,
        far_distance: float = 2.0,
        near_hold_prob: float = 0.6,
        far_hold_prob: float = 0.05,
        noise_range: float = 0.05,
        ball_delay: bool = False,
        ball_delay_mode: str = "current",
        filter_params: dict | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    ) -> torch.Tensor:
        robot, _, offset_w, dist = self._ball_dist_and_robot(env, asset_cfg, ball_cfg)
        if not self._corruption_enabled(env):
            return _to_body_yaw_xy(robot, offset_w)
        if ball_delay and ball_delay_mode == "filter":  # vision_filter と推論ノードの模倣
            out = ball_rel_filtered(env, "pos", filter_params)
            dist = torch.norm(out, dim=1)
        elif ball_delay and ball_delay_mode == "capture":  # 撮影時刻のロボット座標で見た、撮影時刻のボール
            out = ball_rel_capture(env, "pos")
            dist = torch.norm(out, dim=1)
        elif ball_delay:  # d_ball 前のボール位置を、今のロボットの姿勢で見る
            offset_w = delayed_ball_xy(env, "pos") - robot.data.root_pos_w[:, :2]
            dist = torch.norm(offset_w, dim=1)
            out = _to_body_yaw_xy(robot, offset_w)
        else:
            out = _to_body_yaw_xy(robot, offset_w)
        out = out + torch.empty_like(out).uniform_(-noise_range, noise_range)
        p_hold = self._lerp_prob(dist, near_distance, far_distance, near_hold_prob, far_hold_prob)
        return self._hold(out, p_hold)


class ball_vel_rel_vision(_BallVisionTerm):
    """ボールのワールド速度を body yaw 座標に回したもの [m/s]。距離依存のホールド・無効化・ノイズ付き。

    速度が推定できない回 (確率 p_invalid) は厳密に 0 を返し、その 0 もホールド用の
    状態に保存する (実機 vision_filter と同じ)。
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        near_distance: float = 0.35,
        far_distance: float = 2.0,
        near_hold_prob: float = 0.6,
        far_hold_prob: float = 0.05,
        near_invalid_prob: float = 0.4,
        far_invalid_prob: float = 0.02,
        noise_range: float = 0.1,
        ball_delay: bool = False,
        ball_delay_mode: str = "current",
        filter_params: dict | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    ) -> torch.Tensor:
        robot, ball, _, dist = self._ball_dist_and_robot(env, asset_cfg, ball_cfg)
        if not self._corruption_enabled(env):
            return _to_body_yaw_xy(robot, ball.data.root_com_vel_w[:, :2])
        if ball_delay and ball_delay_mode == "filter":  # vision_filter と推論ノードの模倣
            out = ball_rel_filtered(env, "vel", filter_params)
            dist = torch.norm(ball_rel_filtered(env, "pos", filter_params), dim=1)
        elif ball_delay and ball_delay_mode == "capture":  # 撮影時刻のロボットの向きで回した、撮影時刻の速度
            out = ball_rel_capture(env, "vel")
            dist = torch.norm(ball_rel_capture(env, "pos"), dim=1)
        elif ball_delay:  # d_ball 前のボール速度を、今のロボットの向きで見る
            dist = torch.norm(delayed_ball_xy(env, "pos") - robot.data.root_pos_w[:, :2], dim=1)
            out = _to_body_yaw_xy(robot, delayed_ball_xy(env, "vel"))
        else:
            out = _to_body_yaw_xy(robot, ball.data.root_com_vel_w[:, :2])
        out = out + torch.empty_like(out).uniform_(-noise_range, noise_range)
        p_invalid = self._lerp_prob(dist, near_distance, far_distance, near_invalid_prob, far_invalid_prob)
        invalid = torch.rand_like(p_invalid) < p_invalid
        out = torch.where(invalid.unsqueeze(-1), torch.zeros_like(out), out)
        p_hold = self._lerp_prob(dist, near_distance, far_distance, near_hold_prob, far_hold_prob)
        return self._hold(out, p_hold)
