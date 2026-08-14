# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール回り込み (around_ball) タスク専用の観測関数。"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 上位アクション駆動の歩行位相アキュムレータ用バッファ名 (locomotion 側の
# _gait_phase_per_env とは独立させ、base_velocity 駆動の位相と混ざらないようにする)
_HIGH_PHASE_ATTR = "_around_ball_gait_phase"
_HIGH_PHASE_STEP_ATTR = "_around_ball_gait_phase_last_step"

# fixed_freq=None (アキュムレータモード) 用の速度依存ケイデンス則。
# locomotion 側の新規約 (get_gait_phase) と同じ値のローカルコピー。
# NOTE: locomotion 側から import しない — 歩行コードの版によっては存在せず
# ImportError で around_ball 全体が読めなくなるため、意図的に自己完結にしている。
# 新規約の歩行 pt を frozen に使うときは、学習時の値とここが一致しているか確認すること。
_GAIT_FREQ_BASE = 1.7
_GAIT_FREQ_SLOPE = 0.5
_GAIT_FREQ_MIN = 1.7
_GAIT_FREQ_MAX = 2.6
_GAIT_FREQ_YAW_WEIGHT = 0.25
# randomize_phase_freq イベント (存在すれば) が書き込む per-env 周波数オフセットのバッファ名
_PHASE_FREQ_OFFSET_ATTR = "_phase_freq_offset_per_env"


def ball_offset_and_bearing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """ボールの base yaw frame 相対位置 (N, 2) と方位角の絶対値 |bearing| (N,) を返すヘルパ。"""
    ball: Articulation = env.scene["soccer_ball"]
    robot: Articulation = env.scene[asset_cfg.name]
    offset_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    offset_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), offset_w)[:, :2]
    bearing = torch.atan2(offset_b[:, 1], offset_b[:, 0]).abs()
    return offset_b, bearing


def ball_pos_rel_fov(
    env: ManagerBasedRLEnv,
    fov_half_angle_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """視野 (±fov_half_angle_deg) 内にあるときだけ更新されるボール相対位置 (base yaw frame, 2D)。

    実機のカメラ視野を模して、ボールの方位角が視野外のときは「最後に見えたときの値」を
    保持して返す (hold-last-seen)。保持値は見えた時点の base yaw frame 座標のままなので、
    その後ロボットが動くと古い値になる — これも「見失った」状況の近似として意図的。
    バッファはエピソードリセット時に :func:`reset_ball_last_seen` イベントで 0 にする
    (ボールは視野内にスポーンするので、リセット直後の観測計算で即座に真値へ更新される)。
    """
    offset_b, bearing = ball_offset_and_bearing(env, asset_cfg)
    visible = bearing <= math.radians(fov_half_angle_deg)

    buf = getattr(env, "_ball_last_seen_pos_b", None)
    if buf is None or buf.shape != offset_b.shape:
        buf = torch.zeros_like(offset_b)
        env._ball_last_seen_pos_b = buf
    buf[visible] = offset_b[visible]
    # ObsManager のノイズ付加でバッファ本体が汚れないように clone を返す
    return buf.clone()


def ball_in_fov(
    env: ManagerBasedRLEnv,
    timeout_s: float = 0.2,
    max_latency: int = 8,
    fov_half_angle_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """認識トラッキングが「生きているか」のフラグ (N, 1)。1=直近 ``timeout_s`` 秒以内に
    ボール検出が更新された。

    **フレーム単位ではなくタイムアウト方式** (teammate の kick policy の
    ``BALL_TIMEOUT_S=0.2`` と同一セマンティクス)。低更新レートDR (カメラ30fps相当) や
    単発ドロップでは「新しい検出が来ないtick」が常時発生するが、それで 0 に落とすと
    1,1,0,1,1,0... と点滅して信号にならない。そこで「最後に検出が届いてからの経過時間」
    で判定する:
        * 通常動作 (数tickごとに検出更新)          → 1 のまま
        * 1フレーム落ち程度                          → 1 のまま
        * 視野外 / 連続ドロップ / ボール追い越し     → timeout_s 途絶えたら 0

    ``_ball_perc_last_update_step`` は :func:`ball_pos_rel_perceived` が do_update
    (= 遅延サンプルのFOV内 & 非ドロップ & 更新tick) のたびに現在stepで更新する。
    本関数は必ず ``ball_pos_rel_perceived`` より後の obs term として評価されること
    (policy グループの並び: ball_pos_rel → ball_in_fov → kick_direction_b)。

    バッファ未初期化時は真値FOVにフォールバック (起動直後の保険)。
    """
    _ensure_perc_buffers(env, max_latency)
    last = getattr(env, "_ball_perc_last_update_step", None)
    if last is None:
        _, bearing = ball_offset_and_bearing(env, asset_cfg)
        return (bearing <= math.radians(fov_half_angle_deg)).float().unsqueeze(1)
    timeout_ticks = timeout_s / env.step_dt
    age = (int(env.common_step_counter) - last).float()  # 最終検出からの経過 [tick]
    fresh = age <= timeout_ticks
    return fresh.float().unsqueeze(1)


# ============================================================================
# 知覚 (認識チーム出力) の sim-to-real DR
# ボール位置は別チームの認識パイプラインから来るので、その不完全さ
# (レイテンシ・低更新レート・検出ドロップ・距離依存ノイズ・バイアス) を
# ここで模擬してポリシーを頑健にする。実測スペックが出たら各パラメータ
# (下の cfg 側 params) を差し替えるだけでよい。
# ============================================================================

_PERC_STEP_ATTR = "_ball_perc_last_step"


def _ensure_perc_buffers(env: ManagerBasedRLEnv, max_latency: int) -> None:
    """知覚DR用の per-env バッファを (無ければ) 生成する。"""
    n = env.num_envs
    hist = getattr(env, "_ball_perc_hist", None)
    if hist is None or hist.shape != (n, max_latency + 1, 2):
        env._ball_perc_hist = torch.zeros(n, max_latency + 1, 2, device=env.device)  # 真値の履歴 (0=現在)
        env._ball_perceived = torch.zeros(n, 2, device=env.device)                    # 認識出力 (更新間はホールド)
        env._ball_perc_latency = torch.ones(n, dtype=torch.long, device=env.device)   # 遅延 [tick] (per-episode)
        env._ball_perc_update_period = torch.ones(n, dtype=torch.long, device=env.device)  # 次の更新までの間隔 [tick] (毎更新で引き直す)
        # 更新間隔の [lo, hi] レンジ (per-episode 固定)。ビジョンは 30Hz が上限だが常に
        # 30Hz とは限らず負荷で frame 間隔が揺れるので、間隔自体をこのレンジ内で毎回サンプルする。
        env._ball_perc_update_period_lo = torch.full((n,), 2, dtype=torch.long, device=env.device)
        env._ball_perc_update_period_hi = torch.full((n,), 3, dtype=torch.long, device=env.device)
        env._ball_perc_update_ctr = torch.zeros(n, dtype=torch.long, device=env.device)
        env._ball_perc_bias = torch.zeros(n, 2, device=env.device)                    # 系統バイアス (per-episode)
        # 検出が最後に更新された global step (common_step_counter)。ball_in_fov の
        # タイムアウト判定に使う。負の大きな値で「未検出」を表す (起動直後の誤フレッシュ回避)。
        env._ball_perc_last_update_step = torch.full((n,), -(10**9), dtype=torch.long, device=env.device)


def ball_pos_rel_perceived(
    env: ManagerBasedRLEnv,
    fov_half_angle_deg: float = 60.0,
    max_latency: int = 8,
    dropout_prob: float = 0.1,
    noise_along_sigma: float = 0.04,
    noise_along_per_m: float = 0.03,
    noise_lat_sigma: float = 0.02,
    noise_lat_per_m: float = 0.015,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """認識パイプラインの不完全さを模擬したボール相対位置 (body-yaw frame, 2D)。

    ``ball_pos_rel_fov`` (FOV+hold-last-seen) の上に sim-to-real 用の知覚DRを重ねる:

    * **レイテンシ**: ``latency`` tick 前に撮影した位置を使う (per-episode 固定)。
    * **更新レート**: ``update_period`` tick に 1 回だけ認識出力を更新、間はホールド
      (カメラfps < 制御50Hz の再現)。
    * **ドロップアウト**: 更新tickでも ``dropout_prob`` で検出失敗 → ホールド。
    * **距離依存・異方性ノイズ**: 視線方向 (レンジ誤差, 大) と横方向 (ベアリング誤差, 小)
      に分け, それぞれ ``base + per_m * 距離`` の σ でガウス付加。
    * **系統バイアス**: per-episode 固定オフセット (キャリブずれ)。

    視野外/ドロップ/更新なしの間は前回出力を保持する (hold-last-seen)。
    全パラメータを 0 (dropout_prob=0, sigma=0, latency=1, update_period=1) にすると
    ``ball_pos_rel_fov`` とほぼ同じ挙動に縮退する安全な上位互換。
    critic 側は真値 (ball_pos_rel) を見るので、非対称 actor-critic として機能する。
    """
    offset_b, _ = ball_offset_and_bearing(env, asset_cfg)  # 現在の真値 (body-yaw)
    _ensure_perc_buffers(env, max_latency)

    # 同一制御tick内の複数回呼び出しでは確率的更新を進めない (policyグループのみ使用だが保険)
    if getattr(env, _PERC_STEP_ATTR, -1) == int(env.common_step_counter):
        return env._ball_perceived.clone()
    setattr(env, _PERC_STEP_ATTR, int(env.common_step_counter))

    n = env.num_envs
    ar = torch.arange(n, device=env.device)

    # 真値履歴を1つずらして先頭に現在値を積む
    hist = torch.roll(env._ball_perc_hist, shifts=1, dims=1)
    hist[:, 0, :] = offset_b
    env._ball_perc_hist = hist

    # per-env レイテンシ分だけ過去の撮影値
    idx = env._ball_perc_latency.clamp(0, max_latency)
    delayed = hist[ar, idx, :]  # (n, 2)

    # 撮影時点でのFOV内判定
    delayed_bearing = torch.atan2(delayed[:, 1], delayed[:, 0]).abs()
    visible = delayed_bearing <= math.radians(fov_half_angle_deg)

    # 更新レートゲート。次の更新までの間隔は「更新のたびに」[lo, hi] から引き直す。
    # ビジョンは 30Hz が上限だが常に 30Hz とは限らず、負荷で frame 間隔が揺れる。
    # per-episode 固定だとその episode 内ジッタ (30Hz→一時的に遅延) を再現できないので、
    # ここで毎回サンプルし直して「速いフレームと遅いフレームが混ざる」状況を作る。
    ctr = env._ball_perc_update_ctr - 1
    is_update = ctr <= 0
    lo = env._ball_perc_update_period_lo
    span = (env._ball_perc_update_period_hi - lo + 1).clamp(min=1).float()
    new_period = (lo + (torch.rand(n, device=env.device) * span).long()).clamp(min=1)
    env._ball_perc_update_ctr = torch.where(is_update, new_period, ctr)

    # ドロップアウト
    dropped = torch.rand(n, device=env.device) < dropout_prob
    do_update = is_update & visible & (~dropped)

    # 検出が届いた env の最終更新時刻を記録 (ball_in_fov のタイムアウト判定用)。
    # do_update は delayed_bearing (遅延サンプルのFOV) で判定しているので、
    # 「遅延後の時間軸で検出が届いたか」の鮮度になり、位置の時間軸と揃う。
    env._ball_perc_last_update_step[do_update] = int(env.common_step_counter)

    # 距離依存・異方性ノイズ (視線方向=レンジ, 横方向=ベアリング)
    r = torch.norm(delayed, dim=1, keepdim=True).clamp(min=1e-3)
    ray = delayed / r
    perp = torch.stack([-ray[:, 1], ray[:, 0]], dim=1)
    s_along = noise_along_sigma + noise_along_per_m * r.squeeze(1)
    s_lat = noise_lat_sigma + noise_lat_per_m * r.squeeze(1)
    noise = (torch.randn(n, device=env.device) * s_along).unsqueeze(1) * ray + (
        torch.randn(n, device=env.device) * s_lat
    ).unsqueeze(1) * perp
    measured = delayed + env._ball_perc_bias + noise

    env._ball_perceived = torch.where(do_update.unsqueeze(1), measured, env._ball_perceived)
    return env._ball_perceived.clone()


def kick_direction_b_perceived(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """キック方向 (body-yaw frame 単位ベクトル) にロボットのヨー推定誤差を乗せた版。

    キック方向は戦略チームからワールド(フィールド)座標で来て、ロボットの**推定ヨー**で
    body frame に変換される。実機ではこの推定に誤差があり、その分だけ kick_direction_b が
    回る。誤差 ``_kick_yaw_err`` は per-episode 固定 (reset_ball_perception で採番)。

    NOTE: ボール位置はカメラが body 直付けで直接測るため、ヨー推定誤差では回らない
    (ノイズは ball_pos_rel_perceived 側)。ヨー誤差はキック方向のみに効く、が正しい非対称。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    kick_dir_w_xy = env.command_manager.get_term(command_name).command  # (N, 2) ワールド単位ベクトル
    z = torch.zeros_like(kick_dir_w_xy[:, :1])
    kick_dir_w = torch.cat([kick_dir_w_xy, z], dim=1)
    kick_dir_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), kick_dir_w)[:, :2]

    yaw_err = getattr(env, "_kick_yaw_err", None)
    if yaw_err is None:
        return kick_dir_b
    c, s = torch.cos(yaw_err), torch.sin(yaw_err)
    x = kick_dir_b[:, 0] * c - kick_dir_b[:, 1] * s
    y = kick_dir_b[:, 0] * s + kick_dir_b[:, 1] * c
    return torch.stack([x, y], dim=1)


def _high_action_cmd(env: ManagerBasedRLEnv) -> torch.Tensor:
    """上位ポリシーが frozen に注入した歩行コマンド (vx, vy, wz) を返す。未初期化なら 0。"""
    buf = getattr(env, "_prev_high_action", None)
    if buf is None or buf.shape != (env.num_envs, 3):
        return torch.zeros(env.num_envs, 3, device=env.device)
    return buf


def _high_action_gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """上位アクションの速度から周波数を決めて積算した左足の歩行位相 φ∈[0,2π) (per-env)。

    locomotion の :func:`get_gait_phase` と同一のアキュムレータ・周波数則
    (freq = clamp(BASE + SLOPE * speed) + per-env オフセット) だが、速度の出所を
    ``base_velocity`` コマンドではなく ``_prev_high_action`` (frozen が実際に受け取る
    歩行コマンド) にする。階層構成では base_velocity は使われないダミーなので、
    そこから位相を作ると frozen が学習時に見た「コマンド速度と位相テンポの対応」が
    崩れる — 本関数はその整合性を回復する。
    """
    phase = getattr(env, _HIGH_PHASE_ATTR, None)
    if phase is None:
        phase = torch.zeros(env.num_envs, device=env.device)
        setattr(env, _HIGH_PHASE_ATTR, phase)
        setattr(env, _HIGH_PHASE_STEP_ATTR, -1)

    # 同一ステップ内の複数回呼び出し (policy/critic/low_level) では 1 回だけ積算
    if getattr(env, _HIGH_PHASE_STEP_ATTR) != int(env.common_step_counter):
        cmd = _high_action_cmd(env)
        speed = torch.norm(cmd[:, :2], dim=1) + _GAIT_FREQ_YAW_WEIGHT * cmd[:, 2].abs()
        freq = _GAIT_FREQ_BASE + _GAIT_FREQ_SLOPE * speed
        offset = getattr(env, _PHASE_FREQ_OFFSET_ATTR, None)
        if offset is not None:
            freq = freq + offset
        freq = freq.clamp(_GAIT_FREQ_MIN, _GAIT_FREQ_MAX)
        phase = (phase + 2.0 * math.pi * freq * env.step_dt) % (2.0 * math.pi)
        # リセット直後の env は位相 0 から再開 (locomotion と同じ規約)
        phase = torch.where(env.episode_length_buf <= 1, torch.zeros_like(phase), phase)
        setattr(env, _HIGH_PHASE_ATTR, phase)
        setattr(env, _HIGH_PHASE_STEP_ATTR, int(env.common_step_counter))
    return getattr(env, _HIGH_PHASE_ATTR)


def high_action_phase_obs(
    env: ManagerBasedRLEnv,
    cmd_threshold: float = 0.05,
    fixed_freq: float | None = None,
) -> torch.Tensor:
    """上位アクション駆動の歩行位相を sin/cos で返す (左足, 右足の計4次元)。

    locomotion の :func:`phase_obs` と同一フォーマット (frozen 歩行ポリシーの
    ``gait_phase`` スロットにそのまま入る)。停止判定も同じ規約で、上位アクションの
    ノルムが ``cmd_threshold`` 未満なら位相をゼロ埋めして「停止すべき」と伝える。

    ``fixed_freq`` は frozen ポリシーの学習時期に合わせて選ぶ:
        * None (既定): 速度依存周波数のアキュムレータ (2026-07 以降の歩行 pt 用)。
        * 数値 (例 1.6): 旧規約 ``φ = 2π·f·t`` の固定周波数 (0524_walk.pt など
          2026-05 時点の歩行 pt はこちらで学習されている)。
    """
    if fixed_freq is not None:
        t = env.episode_length_buf * env.step_dt
        phase_left = (2.0 * math.pi * fixed_freq * t) % (2.0 * math.pi)
    else:
        phase_left = _high_action_gait_phase(env)
    phase_right = phase_left + math.pi

    phase = torch.stack([
        torch.sin(phase_left), torch.cos(phase_left),
        torch.sin(phase_right), torch.cos(phase_right),
    ], dim=1)

    cmd = _high_action_cmd(env)
    cmd_speed = torch.norm(cmd[:, :3], dim=1, keepdim=True)
    is_stopped = cmd_speed < cmd_threshold
    phase = torch.where(is_stopped, torch.zeros_like(phase), phase)

    return phase
