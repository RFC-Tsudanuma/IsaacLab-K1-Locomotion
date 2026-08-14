# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール回り込み (around_ball) タスク専用の終了 (termination) 関数。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from .events import _ball_tracking_buffers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_kicked(
    env: "ManagerBasedRLEnv",
    kick_dist_threshold: float = 0.3,
    delay_steps: int = 150,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ボールを蹴れたら ``delay_steps`` 後にエピソードを終了させる (成功終了)。

    ``DoneTerm(func=ball_kicked, time_out=True)`` で登録すること。``time_out=True``
    にすることで、この終了は失敗ではなく「区切り」として扱われ、
    ``termination_penalty`` (mdp.is_terminated) の対象外になる (蹴れたのに罰する
    のを防ぐ)。終了すると通常のリセットイベントが全部走る (reset_base で
    ロボット姿勢、reset_ball_in_front_cone でボール配置・キック方向) ので、
    「ボールもロボットも」リセットされる。

    毎ステップの処理:
        1. 蹴り検出: ボールが直近スポーン位置 (reset_ball_in_front_cone が記録)
           から ``kick_dist_threshold`` [m] 以上動いたらカウントダウン開始
           (接触センサー不要の変位ベース検出)。
        2. カウントダウンが 0 になったステップで True を返してエピソード終了。
           カウントダウン中 (蹴った直後〜終了まで) は転がるボールが
           ``ball_moved_along_kick`` 報酬を稼ぐ猶予になる。

    スポーン位置・カウントダウンのバッファはリセット時に
    reset_ball_in_front_cone が初期化するので、別要因 (タイムアウト/転倒) で
    途中終了しても幽霊カウントダウンは残らない。
    """
    ball = env.scene[ball_cfg.name]
    spawn, cd = _ball_tracking_buffers(env)

    # 1. 蹴り検出 → カウントダウン開始 (まだ発火していない env のみ)
    moved = torch.norm(ball.data.root_pos_w[:, :2] - spawn, dim=1) > kick_dist_threshold
    cd[moved & (cd < 0)] = int(delay_steps)

    # 2. カウントダウン進行 → 0 になった env を終了させる
    cd[cd > 0] -= 1
    fire = cd == 0
    # 発火した env の cd は -1 に戻す (このステップでリセットされ再初期化もされるが念のため)
    cd[fire] = -1
    return fire
