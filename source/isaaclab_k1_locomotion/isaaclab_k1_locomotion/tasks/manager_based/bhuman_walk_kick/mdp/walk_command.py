# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""土台の walk に渡す速度コマンド (vx, vy, wz) をボールとキック方向から作る。

土台の walk の方策は速度コマンドを入力に持ち (command グループと観測履歴の両方)、位相の
周波数もコマンドの大きさで決まる。B-Human のキック環境には速度コマンドが無いので、
B-Human の報酬 ``ball_walk_speed`` が向かわせる点 (``_helper_walk_target``) をそのまま
目標にしてコマンドを作る:

* 目標点: キック方向の手前のキック立ち位置 (蹴り足側へ ``kick_pose_y_off`` ずらす)。
  キック姿勢が揃っていくと立ち位置はボールの中へ移り (B-Human ``kick_pose_x``)、
  そのまま歩けばボールを蹴る。ロボットがボールの前にいるときは回り込み点。
* 向き: 遠いうちは目標点の方、近づいたらキック方向。

コマンドは方策への入力 (「どっちへ歩くか」の手がかり) で、追従は報酬にしない。どう歩くかは
B-Human の報酬 (ball_walk_speed など) と walk の報酬で決まる。学習開始時は歩行の重みのまま
なので、このコマンドに従ってボールの方へ歩く状態から始まる。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .kick_command import wrap_to_pi
from .rewards import _walk_target

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BallWalkVelocityCommand(CommandTerm):
    cfg: BallWalkVelocityCommandCfg

    def __init__(self, cfg: BallWalkVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    def __str__(self) -> str:
        return "BallWalkVelocityCommand: B-Human の歩行目標へ向かう (vx, vy, wz)"

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        return {}

    def compute(self, dt: float):
        self._update_command()

    def _update_command(self):
        cfg = self.cfg
        k = self._env.command_manager.get_term(cfg.kick_command_name)
        k.ensure_updated()
        target = _walk_target(k)[:, :2]
        head_far = torch.atan2(target[:, 1], target[:, 0])
        head_near = wrap_to_pi(k.direction_angle_current)
        ball_dist = torch.norm(k.ball_pos[:, :2], dim=-1)
        w = ((ball_dist - cfg.near_dist) / cfg.blend_dist).clamp(0.0, 1.0)  # 1 = 遠い
        err = wrap_to_pi(head_near + w * wrap_to_pi(head_far - head_near))
        vx = (cfg.gain_xy * target[:, 0]).clamp(*cfg.vx_range)
        vx = vx * ((1.0 - w) + w * torch.cos(head_far).clamp(min=0.0))
        # キック姿勢に入ったら (ball_close_to_kick_pose < 0) 速度を落とさず、kick_through_speed で
        # ボールへ歩き込ませる。目標点がボールに近づくと gain_xy * 距離 では止まりかけの速度になり、
        # 足がボールに当たる時の勢いが無くなる (ボールを押すだけになる) ため。
        # B-Human の ball_walk_speed もボールの近くで 0.5 m/s (clip_max_speed_near_ball) までは歩く速さを褒める。
        through = (-k.ball_close_to_kick_pose).clamp(0.0, 1.0) * cfg.kick_through_speed
        through = through * torch.cos(head_near).clamp(min=0.0) * (k.ball_pos[:, 0] > 0).float()
        vx = torch.maximum(vx, through)
        vy = (cfg.gain_xy * target[:, 1]).clamp(-cfg.max_vy, cfg.max_vy)
        wz = (cfg.gain_yaw * err).clamp(-cfg.max_wz, cfg.max_wz)
        self.vel_command_b[:] = torch.stack((vx, vy, wz), dim=-1)


@configclass
class BallWalkVelocityCommandCfg(CommandTermCfg):
    class_type: type = BallWalkVelocityCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    kick_command_name: str = "kick"
    gain_xy: float = 1.5
    vx_range: tuple[float, float] = (-0.5, 1.0)
    max_vy: float = 0.5
    gain_yaw: float = 1.5
    max_wz: float = 1.0
    near_dist: float = 0.6
    """ボールまでこれより近いとキック方向を向く [m]。"""
    blend_dist: float = 0.4
    """near_dist からこの距離かけて「目標点を向く」へ切り替える [m]。"""
    kick_through_speed: float = 0.0
    """キック姿勢に入った後の最低前進速度 [m/s] (0 で無効 = s1 まで)。"""
