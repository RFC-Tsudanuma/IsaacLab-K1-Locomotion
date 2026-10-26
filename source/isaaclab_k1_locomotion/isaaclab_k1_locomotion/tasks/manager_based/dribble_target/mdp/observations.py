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
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    ) -> torch.Tensor:
        robot, _, offset_w, dist = self._ball_dist_and_robot(env, asset_cfg, ball_cfg)
        if not self._corruption_enabled(env):
            return _to_body_yaw_xy(robot, offset_w)
        if ball_delay and ball_delay_mode == "capture":  # 撮影時刻のロボット座標で見た、撮影時刻のボール
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
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    ) -> torch.Tensor:
        robot, ball, _, dist = self._ball_dist_and_robot(env, asset_cfg, ball_cfg)
        if not self._corruption_enabled(env):
            return _to_body_yaw_xy(robot, ball.data.root_com_vel_w[:, :2])
        if ball_delay and ball_delay_mode == "capture":  # 撮影時刻のロボットの向きで回した、撮影時刻の速度
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
