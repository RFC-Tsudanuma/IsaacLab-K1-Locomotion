# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (goalkeeper) タスク専用の終了条件と save 状態機。

失点判定 (ルールブック §1.9 準拠):
    ボール全体がゴールライン (ゴール側エッジ) を完全に越えたとき
    = ボール中心 x < -(ボール半径)、かつポスト内側 |y| < goal_half_width。
    ポストに当たって跳ね返ったボールはセーブ扱いにせず、跳ね返り後に
    上記を満たせば失点になる (判定は毎ステップの位置ベースなので自然に成立)。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from .events import _gk_params
from .observations import ball_pos_goal, gk_buffers, robot_pos_goal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# update_save_state の 1 ステップ 1 回ガード
_SAVE_STEP_ATTR = "_gk_save_state_step"
# 「このステップで新規にボールに触れた」フラグ (save_touch_bonus 報酬が消費)
_NEWLY_TOUCHED_ATTR = "_gk_newly_touched"


def _feet_ball_contact_force(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """左右の足-ボール接触センサから接触力ノルムの最大値 (N,) を返す (ball_kick と同方式)。"""
    f_r = torch.norm(env.scene.sensors["contact_balls_right"].data.force_matrix_w[:, 0, 0], p=2, dim=1)
    f_l = torch.norm(env.scene.sensors["contact_balls_left"].data.force_matrix_w[:, 0, 0], p=2, dim=1)
    return torch.maximum(f_r, f_l)


def update_save_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """タッチ検出とセーブ成功カウントダウンを 1 ステップ 1 回だけ更新する (冪等)。

    戻り値: 「このステップで新規にタッチした」bool マスク (N,)。

    タッチ検出 (いずれか):
        1. 足-ボール接触センサの力 > touch_force_threshold
        2. フォールバック: ボールがロボット近傍 (< touch_proximity) で
           ゴールから遠ざかる向き (vx > 0.1) に転じた
           (足以外の部位で弾いた場合のセンサ取りこぼし対策)

    カウントダウン開始条件: タッチ済み & ボールが脅威でない
    (vx ≥ -0.05: 接近していない、または速度 < 0.15: ほぼ停止)。
    開始後 ``save_delay_steps`` 経過で :func:`save_success` が成功終了を発火する。
    カウントダウン中にボールが再度ゴールへ向かえばカウントダウンを解除する
    (弾きが弱く転がり直した場合はセーブ未確定)。

    報酬 (save_touch_bonus)・終了 (save_success) のどちらから呼んでも、
    同一ステップ内の 2 回目以降は更新をスキップする。
    """
    bufs = gk_buffers(env)
    step = int(env.common_step_counter)
    if getattr(env, _SAVE_STEP_ATTR, None) == step:
        return getattr(env, _NEWLY_TOUCHED_ATTR)
    setattr(env, _SAVE_STEP_ATTR, step)

    p = _gk_params(env)
    ball = env.scene["soccer_ball"]
    robot = env.scene["robot"]
    active = bufs["ball_active"]

    vel = ball.data.root_com_vel_w[:, :3]
    speed_xy = torch.norm(vel[:, :2], dim=1)
    dist = torch.norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)

    contact = _feet_ball_contact_force(env) > float(p.touch_force_threshold)
    deflected = (dist < float(p.touch_proximity)) & (vel[:, 0] > 0.1)
    # ★ 2026-08-21: **守るべきシュートに限る**。
    #
    #   これが無いと `newly` は「アクティブな球に触った」だけで立つので、
    #   状況の多様化 (_diversify_situations) で入れた
    #     静止球 / 横に転がる球 / ゴールから遠ざかる球
    #   に走って行って触るだけで save_touch_bonus (+100) が満額になり、さらに
    #   no_threat が即成立して save_cd が回り save_success で「成功」終了する。
    #   密報酬が ±1 スケールなのに対して +100 なので、**あらゆるボールを追うことが
    #   最適解**になる。実機で観測された「持って離れていくボールにも反応する」
    #   「ゴールと反対方向へ行く球にも反応する」の学習側の原因。
    #
    #   is_shot は reset_ball_shot が立て、_diversify_situations が差し替えた球で
    #   False に戻す。cfg で False にすると従来どおり全ての球を採点する (A/B 用)。
    require_shot = bool(getattr(p, "save_requires_shot", True))
    eligible = active & bufs["is_shot"] if require_shot else active
    newly = eligible & (~bufs["touched"]) & (contact | deflected)
    bufs["touched"][newly] = True
    setattr(env, _NEWLY_TOUCHED_ATTR, newly)

    # セーブ確定カウントダウン。「無害化」に加えて **ボールがゴールラインの外側
    # (フィールド側, x > 0)** にあることを要求する — ロボットは guard_x でライン前に
    # 立つので、正しく守れていればボールはゴールの外で止まる。ライン上に乗った
    # 微妙な球はセーブ確定にせず、タイムアウトまで様子を見る (ルール上は失点では
    # ないが「外で止めた」とは言えないため)。
    ball_outside = ball_pos_goal(env)[:, 0] > 0.0
    no_threat = (vel[:, 0] >= -0.05) | (speed_xy < 0.15)
    start = active & bufs["touched"] & no_threat & ball_outside & (bufs["save_cd"] < 0)
    bufs["save_cd"][start] = int(p.save_delay_steps)
    # 再度脅威になった / ボールがライン内側へ入ったら解除
    rearmed = active & (bufs["save_cd"] >= 0) & ((~no_threat) | (~ball_outside))
    bufs["save_cd"][rearmed] = -1
    bufs["save_cd"][bufs["save_cd"] > 0] -= 1
    return newly


def goal_conceded(
    env: "ManagerBasedRLEnv",
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """失点: ボール全体がポスト間のゴールラインを越えた (失敗終了)。

    ``DoneTerm(time_out=False)`` で登録し、``termination_penalty``
    (mdp.is_terminated) の大きな負報酬の対象にする。
    """
    p = _gk_params(env)
    bufs = gk_buffers(env)
    pos = ball_pos_goal(env, ball_cfg)
    crossed = pos[:, 0] < -float(p.ball_radius)
    inside_posts = pos[:, 1].abs() < float(p.goal_half_width)
    return bufs["ball_active"] & crossed & inside_posts


def save_success(
    env: "ManagerBasedRLEnv",
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """セーブ成功による正常終了。``DoneTerm(time_out=True)`` で登録すること。

    発火条件 (いずれか):
        1. :func:`update_save_state` のカウントダウンが 0 に達した
           (タッチしてボールを無害化し、一定時間保持)
        2. ボールがゴールラインをポストの外側で越えた (枠外に弾き出した)

    ``time_out=True`` により「成功の区切り」として扱われ、termination_penalty の
    対象外になる (around_ball の ball_kicked と同じ流儀)。
    """
    p = _gk_params(env)
    bufs = gk_buffers(env)
    update_save_state(env)

    cd_fire = bufs["save_cd"] == 0
    pos = ball_pos_goal(env, ball_cfg)
    out_wide = bufs["ball_active"] & (pos[:, 0] < -float(p.ball_radius)) & (
        pos[:, 1].abs() >= float(p.goal_half_width)
    )
    fire = cd_fire | out_wide
    bufs["save_cd"][fire] = -1
    return fire


def robot_out_of_bounds(
    env: "ManagerBasedRLEnv",
    x_range: tuple[float, float] = (-0.6, 2.5),
    y_abs_max: float = 2.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ロボットがゴール前の守備範囲から逸脱した (失敗終了)。

    ゴール裏 (x < x_range[0])・前方への飛び出し (x > x_range[1])・
    ポストの外側への迷走 (|y| > y_abs_max) を打ち切って学習信号を密にする。
    """
    pos = robot_pos_goal(env, asset_cfg)
    return (
        (pos[:, 0] < float(x_range[0]))
        | (pos[:, 0] > float(x_range[1]))
        | (pos[:, 1].abs() > float(y_abs_max))
    )
