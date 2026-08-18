# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール履歴版のイベント。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .observations import ballhist_is_engaged

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def sync_engaged_command(
    env: "ManagerBasedEnv",
    # ★ env_ids にデフォルト値を付けないこと (EventManager の引数検証の都合。
    #   直接版の sync_task_command と同じ規約)。
    env_ids: torch.Tensor | None,
    command_name: str = "base_velocity",
    nominal: float = 1.0,
):
    """``base_velocity`` コマンドを **出動フラグ** で上書きする毎ステップイベント。

    直接版の ``sync_task_command`` の置き換え。あちらは手書きの制御則
    (``task_drive_vector``) をそのまま書いていたが、ここでは

        出動中   → (0, sign(ball_y) * nominal, 0)
        非出動   → (0, 0, 0)

    とする。**方策はこのコマンドを観測しない** (velocity_commands はゼロ埋め)。
    書く目的は、コマンドを参照する既存の報酬・判定をそのまま動かすこと:

        * ``feet_phase`` / ``foot_clearance`` の停止判定 (ノルムがしきい値を超えるか)
        * ``foot_clearance`` の速度ゲート (指令方向へ実際に出ている速度の割合)
        * action ペナルティの待機ブースト (ノルム 0 なら待機)

    ノルムは「0 か nominal」の二値なので、しきい値付近をうろつく状態が構造的に
    存在しない。直接版で問題になった「小さい非ゼロ指令」の帯域が消える。

    ★ 横成分に符号を入れてあるのは ``foot_clearance`` の速度ゲートのため。
      あれは指令方向へ実際に出ている速度の割合しか払わないので、方向が無いと
      「動かずに足だけ上げる」解を塞げない。
    """
    from .observations import ballhist_ball_history, FRAME_DIM

    cmd_term = env.command_manager.get_term(command_name)
    buf = getattr(cmd_term, "vel_command_b", None)
    if buf is None:
        return

    engaged = ballhist_is_engaged(env)
    hist = ballhist_ball_history(env, frames=1, stride=1).view(env.num_envs, 1, FRAME_DIM)
    by = hist[:, 0, 1]
    lateral_dir = torch.sign(by)
    lateral_dir = torch.where(lateral_dir == 0, torch.ones_like(lateral_dir), lateral_dir)

    buf[:, 0] = 0.0
    buf[:, 1] = torch.where(engaged, lateral_dir * float(nominal), torch.zeros_like(by))
    buf[:, 2] = 0.0
