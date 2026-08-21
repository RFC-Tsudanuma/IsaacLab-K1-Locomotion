# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_kick 用の観測関数。

B-Human "A Modular Ball Kicking Behavior with Reinforcement Learning"
(Reichenberg & Frese) の observation 空間を K1 向けに移植したもの。
Flags (3次元) は使わない。
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

from ...locomotion.mdp.events import get_phase_freq, get_phase_offset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_pos_rel(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ロボットベースフレーム（yaw aligned）でのボール相対位置。shape: (N, 3)

    policy には遅延なしの現在位置は渡さない (:func:`prev_ball_pos_b` を使う)。
    critic の特権情報としてのみ使用する。
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]

    rel_pos_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    rel_pos_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)
    return rel_pos_b


def ball_vel_b(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ロボットベースフレーム（yaw aligned）でのボール水平速度。shape: (N, 2)"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]

    vel_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), ball.data.root_lin_vel_w[:, :3])
    return vel_b[:, :2]


def sole_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_foot_link"),
) -> torch.Tensor:
    """足裏 (sole) のベース相対 3D 位置。shape: (N, 3)

    論文の評価表と同じく左足裏を既定とする。ベースの姿勢を完全に打ち消した
    ボディフレーム (yaw だけでなく roll/pitch も除去) で表現する。
    """
    robot = env.scene[asset_cfg.name]
    body_idx = asset_cfg.body_ids[0]

    offset_w = robot.data.body_pos_w[:, body_idx, :] - robot.data.root_pos_w[:, :3]
    return quat_rotate_inverse(robot.data.root_quat_w, offset_w)


def gait_phase_sincos(
    env: ManagerBasedRLEnv,
    phase_freq: float = 1.6,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
) -> torch.Tensor:
    """歩行位相を (sin, cos) で返す。shape: (N, 2)

    locomotion 側の ``phase_obs`` は左右両脚の位相 (4次元) を返すが、
    右脚位相は左脚位相 + π で一意に決まるため、ここでは左脚位相のみを渡す。
    コマンド速度が ``cmd_threshold`` 未満のときは ``phase_obs`` と同様にゼロで
    埋め、停止すべき状況であることを明示する (feet_phase 報酬のゲートと揃える)。

    ``randomize_phase_offset`` イベントが登録されている環境 (両足キック系タスク) では
    env ごとの初期オフセットが乗る。方策はこの sin/cos から現在位相を直接読めるので、
    オフセット自体を別スロットで開示する必要はない。
    """
    t = env.episode_length_buf * env.step_dt
    pf = get_phase_freq(env, phase_freq)
    phase = 2.0 * math.pi * pf * t + get_phase_offset(env)

    phase_sincos = torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)

    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :3], dim=1, keepdim=True)
    is_stopped = cmd_speed < cmd_threshold
    return torch.where(is_stopped, torch.zeros_like(phase_sincos), phase_sincos)


def gait_phase_factor_offset(
    env: ManagerBasedRLEnv,
    base_phase_freq: float = 1.6,
) -> torch.Tensor:
    """この env の歩行位相周波数が基準値からどれだけずれているか。shape: (N, 1)

    ``randomize_phase_freq`` イベントが startup で env ごとに ±offset を振るので、
    その offset (= pf - base_phase_freq) をポリシーに開示する。イベントが無い
    構成ではゼロを返す。
    """
    pf = get_phase_freq(env, base_phase_freq)
    if not isinstance(pf, torch.Tensor):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return (pf - base_phase_freq).unsqueeze(-1)


def kick_dir_b(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
) -> torch.Tensor:
    """目標蹴り方向のベース相対 2D 単位ベクトル。shape: (N, 2)

    ``KickDirectionCommand`` は world frame の角度 θ を [sin θ, cos θ, v] で持つので、
    world の方向ベクトル (cos θ, sin θ) をロボットの yaw frame に変換して返す。
    """
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)  # (N, 3) [sin θ, cos θ, v]

    dir_w = torch.zeros(env.num_envs, 3, device=env.device)
    dir_w[:, 0] = command[:, 1]  # cos θ
    dir_w[:, 1] = command[:, 0]  # sin θ

    dir_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), dir_w)
    return dir_b[:, :2]


def target_kick_velocity(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
) -> torch.Tensor:
    """目標ボール速度 [m/s]。shape: (N, 1)

    ``KickDirectionCommand`` が command[:, 2] に格納する。
    """
    command = env.command_manager.get_command(command_name)
    return command[:, 2].unsqueeze(-1)


# --------------------------------------------------------------------------- #
# walk phase 用: ボール/キック由来のスロットを歩行コマンドで置き換える
#
# 観測の次元と並びを kick phase と 1 bit も違わないように保ったまま、中身だけ差し替える。
# こうすると walk phase の checkpoint をそのまま kick phase に引き継げる。
#
# スロットの対応は kick phase の BallFollowVelocityCommand の式から決めている:
#   vx, vy = clamp(<G のロボット相対位置>, ±max_vel)  ← prev_ball_pos スロットと同じ土俵
#   wz     = clamp(<kick_dir_b の偏角>,   ±max_ang_vel) ← kick_direction スロットそのもの
# なので walk phase では「仮想的な目標点」を prev_ball_pos に、「仮想的な目標向き」を
# kick_direction に載せる。policy から見た「スロットが指す方へ歩き、指す向きへ回る」という
# 入力→挙動の対応が両 phase で共通になるので、歩容がそのまま転移する。
#
# NOTE: kick phase の vx,vy は G (ボールの後方 reach の点) を指すので、対応は厳密な恒等では
#       なく reach 分のオフセットが乗る。歩容そのものの転移が目的なので許容し、その差は
#       kick phase 側のフェードイン期間で吸収させる。
# --------------------------------------------------------------------------- #


def walk_command_xy(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """walk phase: 速度コマンドの (vx, vy) を prev_ball_pos スロットに載せる。shape: (N, 2)"""
    return env.command_manager.get_command(command_name)[:, :2]


def walk_command_xyz(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """walk phase: 速度コマンド (vx, vy, 0) を 3 次元のボール位置スロットに載せる。shape: (N, 3)

    :func:`walk_command_xy` の 3 次元版。両足キック系タスクはスロット 3 が
    「ボール 3D 位置」なので、walk phase でもここに仮想的な目標点を載せて
    「スロットが指す方へ歩く」という入力→挙動の対応を kick phase と共通にする。
    z は歩行中は意味を持たないので 0 (kick phase でも蹴るまでは ≈ ボール半径で
    ほぼ一定なので、この差はフェードイン期間で吸収される)。
    """
    cmd = env.command_manager.get_command(command_name)[:, :2]
    return torch.cat([cmd, torch.zeros_like(cmd[:, :1])], dim=-1)


def walk_command_yaw_dir(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """walk phase: 角速度コマンド wz を kick_direction スロットに載せる。shape: (N, 2)

    kick phase の kick_dir_b は単位ベクトルで、その偏角が wz 指令に対応する。
    合わせて (cos wz, sin wz) の単位ベクトルとして渡す。
    """
    wz = env.command_manager.get_command(command_name)[:, 2]
    return torch.stack([torch.cos(wz), torch.sin(wz)], dim=-1)


def zero_obs(env: ManagerBasedRLEnv, dim: int = 1) -> torch.Tensor:
    """walk phase: ボールが存在しないスロットをゼロで埋める。shape: (N, dim)"""
    return torch.zeros(env.num_envs, dim, device=env.device)


def noisy_ball_pos_b(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    delay_step_range: tuple[int, int] = (2, 6),
    camera_hz: float = 30.0,
    jitter_std: float = 0.067,
    jitter_clip: float = 0.2,
    dim: int = 2,
    frame_lag: int = 0,
) -> torch.Tensor:
    """実機の認識パイプラインを模したボール位置（ベース相対）。shape: (N, ``dim``)

    :func:`prev_ball_pos_b` (固定 1 ステップ遅延 + 毎ステップ独立の Unoise) の置き換え。
    実機では vision が制御 (50Hz) より遅い周期で動き、画像処理・通信の遅延を経て届くので、
    観測は「古い値が数ステップ保持され、フレーム境界で不連続に更新される」系列になる。
    毎ステップ独立な白色ノイズより濾しにくく、これを学習時に見せておくのが目的。

    3 つの誤差を重ねてモデル化する:

    1. **遅延**: 真の相対位置を毎ステップリングバッファに積み、``delay`` ステップ前の値を
       観測として使う。``delay`` は ``delay_step_range`` (両端含む) からエピソードごと・
       env ごとに一様サンプルする。既定 (2, 6) は 50Hz 制御で 40-120ms。
    2. **サンプル&ホールド**: 観測の更新はカメラフレーム到来時 (``camera_hz``) のみ。
       50Hz 制御 × 30Hz カメラなら 2,2,1 ステップ間隔の繰り返しで更新され、間は前の値を
       保持する。フレーム位相はエピソードごとにランダム。ホールドのぶん実効遅延は
       ``delay`` に加えて 0〜1 フレーム分 (30Hz なら 0-33ms) 伸びる。
    3. **ジッタ**: フレーム更新時に軸ごと独立なガウス乱数 N(0, ``jitter_std``²) [m] を
       引き直して載せ、±``jitter_clip`` でクリップする。ホールド中は同じノイズ実現値が
       保持される (毎ステップ独立な Unoise とはここが違う。cfg 側でこの項の Unoise は
       外すこと)。
       一様分布ではなくガウスにするのは、実機の認識誤差が「ほとんどの時刻は小さく、
       たまに大きく外す」分布だから。一様 ±20cm は「常時 20cm 級に外れている」ことに
       なり、ボール半径 0.11m を超える誤差が定常化して位置信号そのものが壊れる。
       既定の std 0.067 は ``jitter_clip`` (0.2) の 1/3 で、3σ = クリップ点。
       サンプルの 99.7% がクリップされずに通り、クリップは分布の裾を切るだけになる
       (std をこれより大きく取るとクリップ点に確率質量が溜まり、実質「±clip の
       二値ノイズ」に近づいてしまう)。

    遅延はベース相対位置に対して掛ける。実機はオドメトリアンカーで自己移動分を補償して
    いるので、これは実機より保守的 (アンカーの不完全さ・オドメトリドリフトも被覆する) な
    モデルになる。

    リセット時はバッファ全体を現在位置で埋め、ジッタ付きの現在位置から始める
    (ゼロ埋めするとボールがベース原点にあるという誤った観測になるため)。

    同一ステップ内で複数グループから呼ばれても状態が二重に進まないよう、
    ``common_step_counter`` でステップ境界を検出する。リセット処理も同じガードの中で
    行う (ジッタの引き直しがあるので、:func:`prev_ball_pos_b` と違い毎呼び出しで
    実行すると冪等にならない)。

    Args:
        dim: 3 なら (x, y, z)、2 なら水平成分のみ。既定 2 は従来の
            ``prev_ball_pos`` スロット用。3 は both_feet 系の観測スロット 3
            (Current Ball 3D Position) 用で、z も同じ遅延・サンプル&ホールド・
            ジッタのパイプラインを通る。
        frame_lag: 0 = 最新のカメラフレームの推定値、1 = その 1 フレーム前。
            スロット 3 (現在) とスロット 12 (直前) を **同じカメラの違うフレーム**
            から取るための引数。

    .. warning::
        状態 (``env._noisy_ball_pos_state``) は **1 台のカメラ** として全呼び出しで
        共有する。遅延・フレーム位相・ジッタの実現値を共有するのが目的なので、
        これは意図した設計。ただしその副作用として、``delay_step_range`` /
        ``camera_hz`` / ``jitter_std`` / ``jitter_clip`` は **そのステップで最初に
        呼ばれた項の値** (= ObsGroup の宣言順で先にある項) が使われる。同じ env の
        複数スロットでこの関数を使うときは、これら 4 つを必ず同じ値にすること
        (``dim`` と ``frame_lag`` はスロットごとに違ってよい。状態を読むだけなので)。

    NOTE: 内部バッファは常に 3D で持ち、返す直前に ``[:, :dim]`` で切る。既定
          ``dim=2`` の返り値の**分布**は従来と同一だが、ジッタの乱数を 2 列 → 3 列
          引くようになったので、同一 seed でのビット単位の再現性は無い
          (DR ノイズなので学習結果には影響しない)。
    """
    if dim not in (2, 3):
        raise ValueError(f"noisy_ball_pos_b: dim は 2 か 3 (指定: {dim})。")
    if frame_lag not in (0, 1):
        raise ValueError(f"noisy_ball_pos_b: frame_lag は 0 か 1 (指定: {frame_lag})。")

    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]

    rel_pos_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    cur = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)  # (N, 3)

    buf_len = int(delay_step_range[1]) + 1
    frame_dt = 1.0 / camera_hz
    device = env.device

    def _sample_delay(n: int) -> torch.Tensor:
        return torch.randint(
            int(delay_step_range[0]), int(delay_step_range[1]) + 1, (n,), device=device
        )

    def _jitter(n: int) -> torch.Tensor:
        """クリップ済みガウスジッタ。shape: (n, 3)"""
        return (torch.randn(n, 3, device=device) * jitter_std).clamp_(-jitter_clip, jitter_clip)

    step = int(env.common_step_counter)
    state = getattr(env, "_noisy_ball_pos_state", None)
    if state is None:
        state = {
            # (N, buf_len, 3): 真の相対位置の履歴。head が最新の書き込み位置。
            "buf": cur.unsqueeze(1).repeat(1, buf_len, 1),
            "head": 0,
            "delay": _sample_delay(env.num_envs),
            # カメラフレーム位相 [s]。frame_dt を超えたらフレーム到来。
            "acc": torch.rand(env.num_envs, device=device) * frame_dt,
            "held": cur + _jitter(env.num_envs),
            # 1 フレーム前の推定値 (frame_lag=1 用)。初期値は held と独立に引く。
            "held_prev": cur + _jitter(env.num_envs),
            "env_ids": torch.arange(env.num_envs, device=device),
            "step": step,
        }
        env._noisy_ball_pos_state = state
    elif state["step"] != step:
        state["step"] = step

        # 1. 真の相対位置をリングバッファに積む
        head = (state["head"] + 1) % buf_len
        state["head"] = head
        state["buf"][:, head] = cur

        # 2. カメラフレーム到来判定 (step_dt < frame_dt なので 1 ステップ最大 1 回)
        state["acc"] += env.step_dt
        new_frame = state["acc"] >= frame_dt
        state["acc"][new_frame] -= frame_dt

        # 3. フレームが来た env だけ、delay ステップ前の位置 + ジッタで観測を更新。
        #    更新前の held は held_prev へ送る (スロット 12 が読む「1 フレーム前」)。
        if new_frame.any():
            idx = (head - state["delay"]) % buf_len
            meas = state["buf"][state["env_ids"], idx]
            meas = meas + _jitter(env.num_envs)
            mask = new_frame.unsqueeze(-1)
            state["held_prev"] = torch.where(mask, state["held"], state["held_prev"])
            state["held"] = torch.where(mask, meas, state["held"])

        # 4. リセット直後の env は履歴を現在位置で埋め直し、delay とフレーム位相を再サンプル。
        #    episode_length_buf は step() 内で加算された後に _reset_idx で 0 に戻されるので、
        #    「今このステップでリセットされた env」だけが 0 になる。
        just_reset = env.episode_length_buf == 0
        if just_reset.any():
            n = int(just_reset.sum())
            state["buf"][just_reset] = cur[just_reset].unsqueeze(1)
            state["delay"][just_reset] = _sample_delay(n)
            state["acc"][just_reset] = torch.rand(n, device=device) * frame_dt
            state["held"][just_reset] = cur[just_reset] + _jitter(n)
            state["held_prev"][just_reset] = cur[just_reset] + _jitter(n)

    return state["held" if frame_lag == 0 else "held_prev"][:, :dim]


def prev_ball_pos_b(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """前ステップのボール水平位置（ベース相対）。shape: (N, 2)

    知覚遅延を模すため、現在位置ではなく 1 ステップ前の値を渡す。
    エピソード開始直後は前ステップが存在しないので現在位置で初期化する
    (ゼロ埋めするとボールがベース原点にあるという誤った観測になるため)。

    同一ステップ内で policy / critic の両グループから呼ばれても値が二重に
    進まないよう、``common_step_counter`` でステップ境界を検出する。
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]

    rel_pos_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    cur = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)[:, :2]

    step = int(env.common_step_counter)
    state = getattr(env, "_prev_ball_pos_state", None)
    if state is None:
        state = {"prev": cur.clone(), "cur": cur.clone(), "step": step}
        env._prev_ball_pos_state = state
    elif state["step"] != step:
        # ステップが進んだ: 前ステップに観測した「現在位置」が今ステップの「前回位置」になる
        state["prev"] = state["cur"]
        state["cur"] = cur.clone()
        state["step"] = step

    # リセット直後の env は前エピソードの値を引き継がせない。
    # episode_length_buf は step() 内で加算された後に _reset_idx で 0 に戻されるので、
    # 「今このステップでリセットされた env」だけが 0 になる。
    just_reset = env.episode_length_buf == 0
    state["prev"][just_reset] = cur[just_reset]
    state["cur"][just_reset] = cur[just_reset]

    return state["prev"]


# --------------------------------------------------------------------------- #
# 遅延つきボール位置 (履歴バッファ版)
#
# :func:`prev_ball_pos_b` は「1 ステップ前の 2D 位置」1 種類しか出せない。論文の観測は
# **Current Ball 3D Position (3) と Previous Ball 2D Position (2) の 2 スロット**を持つ
# ので、遅延の違う 2 つの標本を同じ履歴から取り出せるようにしたのがこの関数。
#
# 遅延を残す理由は :func:`ball_pos_rel` の NOTE と同じ (実機の vision は制御より遅い)。
# 論文どおり「遅延ゼロの現在位置」を渡すと critic の特権情報と区別が付かなくなるので、
# **現在位置スロットも 1 ステップ遅延**とし、previous スロットをその 1 つ前にする。
#
# 履歴は「そのステップの base yaw フレームで測った相対位置」をそのまま積む
# (:func:`prev_ball_pos_b` と同じ規約)。過去の計測を過去のロボット姿勢で表した値になり、
# 実機の vision 出力の扱いと一致する。
# --------------------------------------------------------------------------- #
_BALL_HIST_ATTR = "_ball_pos_hist_state"
_BALL_HIST_LEN = 3  # 現在 (delay 0) + 過去 2 ステップぶん


def delayed_ball_pos_b(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    delay_steps: int = 1,
    dim: int = 3,
) -> torch.Tensor:
    """``delay_steps`` ステップ前のボール位置（ベース相対）。shape: (N, ``dim``)

    Args:
        delay_steps: 何ステップ前の値を返すか。0 = 遅延なし。``_BALL_HIST_LEN`` 未満。
        dim: 3 なら (x, y, z)、2 なら水平成分のみ。

    同一ステップ内で policy / critic の両グループ・複数スロットから呼ばれても履歴が
    二重に進まないよう、``common_step_counter`` でステップ境界を検出する
    (:func:`prev_ball_pos_b` と同じ)。エピソード開始直後は履歴が無いので現在位置で
    全段を埋める (ゼロ埋めするとボールがベース原点にあるという誤った観測になる)。
    """
    if not 0 <= delay_steps < _BALL_HIST_LEN:
        raise ValueError(
            f"delayed_ball_pos_b: delay_steps は 0-{_BALL_HIST_LEN - 1} の範囲 (指定: {delay_steps})。"
            " これより長い遅延が要るなら _BALL_HIST_LEN を増やすこと。"
        )

    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]

    rel_pos_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    cur = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)  # (N, 3)

    step = int(env.common_step_counter)
    state = getattr(env, _BALL_HIST_ATTR, None)
    if state is None:
        # (N, _BALL_HIST_LEN, 3)。index 0 が最新 (delay 0)。
        hist = cur.unsqueeze(1).repeat(1, _BALL_HIST_LEN, 1).clone()
        state = {"hist": hist, "step": step}
        setattr(env, _BALL_HIST_ATTR, state)
    elif state["step"] != step:
        # ステップが進んだ: 全段を 1 つ古い側へずらして、先頭に今の値を入れる
        state["hist"] = torch.roll(state["hist"], shifts=1, dims=1)
        state["hist"][:, 0] = cur
        state["step"] = step

    # リセット直後の env は前エピソードの履歴を引き継がせない
    just_reset = env.episode_length_buf == 0
    if just_reset.any():
        state["hist"][just_reset] = cur[just_reset].unsqueeze(1)

    return state["hist"][:, delay_steps, :dim]


# --------------------------------------------------------------------------- #
# センサ遅延の domain randomization
#
# NOTE: 「delayed_」で始まる関数がこのファイルには 2 系統ある。混同しないこと。
#
#   * :func:`delayed_ball_pos_b` / :func:`noisy_ball_pos_b` (上)
#       = **ボール知覚** の遅延。実機の vision パイプライン (30Hz サンプル&ホールド +
#         エピソードごとランダムな **整数ステップ** 遅延 + ガウスジッタ) を模す。
#         対象は観測スロットの「ボール位置」だけ。
#   * :func:`delayed_projected_gravity` ほか 4 つ (下)
#       = **IMU / 関節エンコーダ** の遅延。ロボット自身の自己受容感覚が遅れて届く
#         状況を模す。遅延は過去フレームの線形補間で **連続値**、対象は
#         projected_gravity / base_ang_vel / joint_pos / joint_vel。
#         こちらは fewa/walk_kick_dual_encoder_tune からの移植で、dual encoder 系
#         (walk_kick_dual / walk_weak_kick_dual / walk_middle_kick_dual) の最終
#         stage が :func:`~...walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_delay`
#         経由で使う。
#
# 実機では IMU も関節エンコーダも「測ってから policy に届くまで」に遅れがある
# (バス転送・フィルタ・制御ループの位相)。sim で遅延ゼロのまま学習すると、実機の
# 遅れた観測に対して過剰に反応する (特に base_ang_vel は歩行の安定化に直結する)。
#
# 実装は「過去フレームの線形補間」。制御周期 dt = 0.02 s に対して遅延 0.02 s は
# ちょうど 1 ステップなので、整数ステップの遅延だと 0 か 1 の 2 値にしかならない。
# hist[i0] と hist[i0+1] を補間することで [0, max_delay_s] の連続値を表現する。
#
#   lag [steps] = delay_s / dt,  i0 = floor(lag),  w = lag - i0
#   out = (1 - w) * hist[i0] + w * hist[i0 + 1]
#
# 遅延量は **env ごと・エピソードごと** に一様サンプリングする (エピソード内では
# 一定)。実機のレイテンシは機体・起動ごとにほぼ一定で、ステップ単位で揺れるもの
# ではないため。ジッタまでは模擬していない。
#
# ``group`` が同じ項は **同じ遅延量を共有する**。projected_gravity と base_ang_vel は
# どちらも同じ IMU から来るので独立に遅れることはなく、joint_pos と joint_vel も
# 同じエンコーダ読み出しから来る。独立に引くと物理的にあり得ない組み合わせ
# (重力は最新・角速度だけ 1 ステップ古い) を学習させることになる。
#
# NOTE: policy 観測にだけ掛けること。critic は特権情報なので遅延させない
#       (遅延した観測から価値を推定させる理由が無く、学習が難しくなるだけ)。
# NOTE: ObservationManager のノイズはこの関数の **後** に乗る。実機の
#       「遅れて届いた値にセンサノイズが乗る」順序と一致する。
# NOTE: 観測の次元も並びも変えないので、遅延の有無で checkpoint はそのまま繋がる。
# --------------------------------------------------------------------------- #
_OBS_DELAY_STATE_ATTR = "_obs_delay_state"


def _delayed_signal(
    env: ManagerBasedRLEnv,
    key: str,
    group: str,
    value: torch.Tensor,
    max_delay_s: float,
    base_delay_s: float = 0.0,
) -> torch.Tensor:
    """``value`` を ``base_delay_s + [0, max_delay_s]`` だけ遅延させて返す。

    Args:
        key: 項ごとの履歴バッファを引くキー (項ごとに一意にすること)。
        group: 遅延量を共有するセンサ名 ("imu" / "encoder" / "vision" など)。
            同じ group の項は同じ乱数を引く。
        value: 今ステップの生の観測 (num_envs, dim)。
        max_delay_s: ランダム成分の上限 [s]。
        base_delay_s: 全 env 共通の固定遅延 [s]。同じセンサから来るのに片方の項だけ
            設計上すでに遅れている場合 (:func:`delayed_ball_pos_b` の整数ステップ)、
            遅れていない方にこれを与えて実効遅延を揃える。
    """
    if max_delay_s <= 0.0 and base_delay_s <= 0.0:
        return value

    dt = env.step_dt
    max_lag = max_delay_s / dt
    base_lag = base_delay_s / dt
    # 補間には hist[i0] と hist[i0+1] が要るので、最大遅延ぶん + 1 フレーム持つ。
    n_frames = int(math.ceil(base_lag + max_lag)) + 1
    step = int(env.common_step_counter)
    num_envs = value.shape[0]

    root = getattr(env, _OBS_DELAY_STATE_ATTR, None)
    if root is None:
        root = {"groups": {}, "terms": {}}
        setattr(env, _OBS_DELAY_STATE_ATTR, root)

    # prev_ball_pos_b と同じ判定: episode_length_buf は step() 内で加算された後に
    # _reset_idx で 0 に戻るので、「今このステップでリセットされた env」だけが 0 になる。
    just_reset = env.episode_length_buf == 0

    # -- 1. グループ単位の遅延量。エピソード開始時に引き直す。
    gate = root["groups"].get(group)
    if gate is None or gate["lag"].shape[0] != num_envs:
        gate = {"lag": torch.rand(num_envs, device=value.device) * max_lag, "step": step}
        root["groups"][group] = gate
    elif gate["step"] != step:
        # 同じグループの 2 項目以降が同じステップで引き直さないよう step で守る。
        gate["step"] = step
        n_reset = int(just_reset.sum())
        if n_reset > 0:
            gate["lag"][just_reset] = torch.rand(n_reset, device=value.device) * max_lag
    lag = gate["lag"]

    # -- 2. 項ごとの履歴。hist[0] が現在フレーム、hist[k] が k ステップ前。
    hist_state = root["terms"].get(key)
    if hist_state is None or hist_state["hist"].shape[1:] != value.shape:
        hist_state = {"hist": value.unsqueeze(0).repeat(n_frames, 1, 1), "step": step}
        root["terms"][key] = hist_state
    elif hist_state["step"] != step:
        hist_state["hist"] = torch.roll(hist_state["hist"], shifts=1, dims=0)
        hist_state["hist"][0] = value
        hist_state["step"] = step
    else:
        # 同一ステップ内で 2 回呼ばれても履歴をずらさない (先頭を上書きするだけ)。
        hist_state["hist"][0] = value
    hist = hist_state["hist"]

    # リセット直後は前エピソードの値を引きずらせない (全フレームを現在値で埋める)。
    if bool(just_reset.any()):
        hist[:, just_reset] = value[just_reset].unsqueeze(0)

    # -- 3. 線形補間 (固定遅延ぶんを足してから)
    total_lag = lag + base_lag
    i0 = torch.floor(total_lag).long().clamp_(min=0, max=n_frames - 2)
    weight = (total_lag - i0.to(total_lag.dtype)).unsqueeze(-1)
    env_idx = torch.arange(num_envs, device=value.device)
    return (1.0 - weight) * hist[i0, env_idx] + weight * hist[i0 + 1, env_idx]


def delayed_projected_gravity(
    env: ManagerBasedRLEnv,
    max_delay_s: float,
    group: str = "imu",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """IMU 由来の重力方向を遅延させたもの。"""
    value = base_mdp.projected_gravity(env, asset_cfg=asset_cfg)
    return _delayed_signal(env, "projected_gravity", group, value, max_delay_s)


def delayed_base_ang_vel(
    env: ManagerBasedRLEnv,
    max_delay_s: float,
    group: str = "imu",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """IMU 由来の base 角速度を遅延させたもの。"""
    value = base_mdp.base_ang_vel(env, asset_cfg=asset_cfg)
    return _delayed_signal(env, "base_ang_vel", group, value, max_delay_s)


def delayed_joint_pos_rel(
    env: ManagerBasedRLEnv,
    max_delay_s: float,
    group: str = "encoder",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """エンコーダ由来の関節角 (デフォルト姿勢からの相対) を遅延させたもの。"""
    value = base_mdp.joint_pos_rel(env, asset_cfg=asset_cfg)
    return _delayed_signal(env, "joint_pos_rel", group, value, max_delay_s)


def delayed_joint_vel_rel(
    env: ManagerBasedRLEnv,
    max_delay_s: float,
    group: str = "encoder",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """エンコーダ由来の関節速度 (デフォルトからの相対) を遅延させたもの。"""
    value = base_mdp.joint_vel_rel(env, asset_cfg=asset_cfg)
    return _delayed_signal(env, "joint_vel_rel", group, value, max_delay_s)


def delayed_ball_vel_b(
    env: ManagerBasedRLEnv,
    max_delay_s: float,
    group: str = "vision",
    base_delay_s: float = 0.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """視覚由来のボール速度を遅延させたもの。

    ``base_delay_s`` はボール位置スロットが設計上持っている整数ステップの遅延と
    実効遅延を揃えるためのもの。同じカメラフレームから出る量なので、レイテンシが
    ずれているのは実機ではあり得ない。

    移植元: ``fewa/walk_kick_dual_encoder_tune`` の 47b8863。
    """
    value = ball_vel_b(env, ball_cfg=ball_cfg)
    return _delayed_signal(env, "ball_vel_b", group, value, max_delay_s, base_delay_s)


def delayed_ball_pos_vision_b(
    env: ManagerBasedRLEnv,
    max_delay_s: float,
    group: str = "vision",
    base_delay_s: float = 0.0,
    delay_steps: int = 1,
    dim: int = 3,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """:func:`delayed_ball_pos_b` の値にさらに視覚レイテンシの DR を掛ける。

    both_feet 系の観測は同じボール位置履歴から 2 つのスロットを取る:

    * スロット 3 (Current Ball 3D Position): ``delay_steps=1``, ``dim=3``
    * スロット 12 (Previous Ball 2D Position): ``delay_steps=2``, ``dim=2``

    この **整数ステップの設計遅延の上に**、``group`` (既定 "vision") で共有される
    連続遅延 ``base_delay_s + [0, max_delay_s]`` が乗る。乱数は group 単位で共有される
    ので 2 つのスロットには同じ遅延量が掛かり、**「スロット 12 は常にスロット 3 の
    1 ステップ前」という both_feet の設計関係は保たれる**。

    既定の設定 (``max_delay_s = 0.06``, ``base_delay_s = 0``) での実効遅延:

    * スロット 3  : 0.02 + [0, 0.06] = 0.02-0.08 s
    * スロット 12 : 0.04 + [0, 0.06] = 0.04-0.10 s

    ``key`` は ``delay_steps`` ごとに分けるので、2 スロットの履歴バッファは別々に
    持たれる (中身は 1 ステップずれた同じ系列)。

    移植元: ``fewa/walk_kick_dual_encoder_tune`` の 47b8863 (あちらの
    ``delayed_prev_ball_pos_b`` を both_feet の 2 スロット構成に適合させたもの)。

    NOTE: ``base_delay_s`` は既定の 0 のまま使うこと。整数ステップの遅延を元から
          持っているので、足すと二重になる。固定ぶんを足すのは ``ball_vel``
          (:func:`delayed_ball_vel_b`) の側。
    """
    value = delayed_ball_pos_b(env, ball_cfg=ball_cfg, delay_steps=delay_steps, dim=dim)
    return _delayed_signal(
        env, f"ball_pos_vision_{delay_steps}", group, value, max_delay_s, base_delay_s
    )


# --------------------------------------------------------------------------- #
# 自己位置推定の遅延
#
# 上の 2 系統 (vision / imu・encoder) とはまた別枠。こちらは **ロボット自身の向き
# (ヨー角) がいつの時点のものか** の遅れを模す。
#
# 実機の自己位置推定はカメラのランドマーク認識と InEKF (FK + IMU) でできており、
# その出力が policy に届くまでに遅れがある。蹴り方向はフィールド地図上の座標
# (ゴールなど) で与えるので、体基準に直すにはこのヨー角が要る。遅れたヨー角で
# 変換すると、policy が見る蹴り方向は実際とずれる。
#
# 実装は「``kick_dir_b`` の出力そのものを遅延させる」だけ。これで
# 「古いヨー角で今の θ を変換した値」と **完全に同じ値**になる:
#
#   kick_dir_b(t) = R(yaw_t)^-1 · dir_w   で dir_w は θ から決まる
#   θ は KickDirectionCommand の resampling_time_range = (1e9, 1e9) により
#   エピソード中いっさい変わらない  →  dir_w も定数
#   よって kick_dir_b(t - Δ) = R(yaw_{t-Δ})^-1 · dir_w
#
# ヨー角の履歴バッファを別に持つ必要はなく、既存の :func:`_delayed_signal` に
# 載せられる。遅延量は env ごと・エピソードごとに引き直される。
#
# NOTE: θ をエピソード途中で引き直す設定にしたら、この等価性は壊れる
#       (指令の変化そのものまで遅れることになる)。そのときはヨー角側を
#       遅延させる実装に書き換えること。
# NOTE: ボール位置・速度には掛けないこと。あれはカメラが体基準で直接測る量で
#       自己位置推定を通らず、別枠の vision 遅延を既に持っている。
#       自己位置の遅延を受ける policy 観測は ``kick_direction`` の 1 スロットだけ。
# NOTE: policy 観測にだけ掛けること (critic は特権情報)。
# --------------------------------------------------------------------------- #


def delayed_kick_dir_b(
    env: ManagerBasedRLEnv,
    max_delay_s: float,
    command_name: str = "kick_direction",
    group: str = "localization",
    base_delay_s: float = 0.0,
) -> torch.Tensor:
    """自己位置推定の遅延ぶんだけ古いヨー角で変換した蹴り方向。shape: (N, 2)

    実効遅延は ``base_delay_s + [0, max_delay_s]``。150-300 ms を狙うなら
    ``base_delay_s=0.15, max_delay_s=0.15``。

    Args:
        max_delay_s: ランダム成分の上限 [s]。
        command_name: :class:`~.commands.KickDirectionCommand` の名前。
        group: 遅延量を共有するセンサ名。ボールの "vision" とは別にすること
            (カメラの生画像と自己位置推定では出所も経路も違う)。
        base_delay_s: 全 env 共通の固定遅延 [s]。

    NOTE: 線形補間で単位ベクトルの長さがわずかに縮むので、返す前に正規化する。
          `kick_dir_b` は単位ベクトルを返す規約なので、長さの情報を持たせない。
    """
    value = kick_dir_b(env, command_name=command_name)
    out = _delayed_signal(env, "kick_dir_b", group, value, max_delay_s, base_delay_s)
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def prev_joint_request_rsi(env: ManagerBasedRLEnv) -> torch.Tensor:
    """前ステップの関節指令。RSI でリセットした env は 1 ステップ目だけ注入値を返す。

    ``last_action`` は ``ActionManager`` のバッファを読むが、そのバッファは reset 時に
    0 クリアされる (しかもクリアは event の **後**)。そのため
    :func:`~.events.reset_from_walk_states` が action_manager に直接書いても消される。

    実機の walk → walk_kick 切り替えでは、この観測には walk ポリシーが直前に出した
    関節指令が入っている。そこを 0 (= default 姿勢を要求) で始めると学習時と実機で
    初期フレームが食い違うので、リセット直後の 1 フレームだけ注入値に差し替える。
    イベントが未登録の環境では ``last_action`` と完全に同じ値を返す。
    """
    action = env.action_manager.action
    buf = getattr(env, "_rsi_last_action", None)
    if buf is None:
        return action
    first_step = (env.episode_length_buf == 0).unsqueeze(-1)
    return torch.where(first_step, buf, action)
