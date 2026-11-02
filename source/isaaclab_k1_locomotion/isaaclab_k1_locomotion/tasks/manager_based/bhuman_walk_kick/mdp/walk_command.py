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

import math
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
        target = _walk_target(k, cfg.min_side_off)[:, :2]
        head_far = torch.atan2(target[:, 1], target[:, 0])
        # near_yaw_off_deg (既定 0): 近くでは キック方向 - walk_side * near_yaw_off を向く (cfg の説明)
        head_near = wrap_to_pi(k.direction_angle_current - k.walk_side * math.radians(cfg.near_yaw_off_deg))
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
        if k.handoff is not None:
            # 受け渡し前 (歩行方策が動かしている env) は歩行の速度指令。観測の速度指令の欄と位相の周波数もこれになる
            # (実機の observe_walk_tick と同じ。kick_command.handoff_prob、walk_handoff.py)。
            wm = k.handoff.walk_mode
            self.vel_command_b[:] = torch.where(wm.unsqueeze(-1), k.handoff.walk_command(), self.vel_command_b)


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
    min_side_off: float = 0.0
    """(既定 0 = 今までどおり) 目標点 (キック立ち位置) の横ずれ kick_pose_y_off + extra の下限 [m]。
    報酬 (B-Human ball_walk_speed) の目標点は変えない。near_yaw_off_deg と合わせて 2026-10-31 Stage 9 で使う。

    理由 (2026-10-31、s7b model_9400、MuJoCo、学習と同じ Stage 7 の指令): うまく蹴れた 132 場面 (向きのずれ 5° 以下) では
    胴体 (両足の中点) がボールとキック方向の線から 0.6 秒前 0.16〜0.18 m・当たる時 0.13〜0.15 m 横にいる。指令の目標は
    kick_pose_y_off 0.06 m (ロボットがボールの真後ろにいると extra = 0) で、真後ろから来た場面ではロボットが指令どおり
    0.05〜0.08 m に寄り、内側の足 (蹴り足) がボールの真後ろを前向きのまま歩いてつま先・すねで先にボールを押す
    (rel -10〜+10 の 10°超・失敗 82 場面のうち約 35)。"""
    near_yaw_off_deg: float = 0.0
    """(既定 0 = 今までどおり) ボールの近く (near_dist 以内) で向く方向を、キック方向から蹴り足の側へこの角度 [deg] 回す
    (walk_side > 0 = ボールの左に立って右足で蹴る → 右へ回す)。

    理由: 上と同じ 132 場面で、胴体の向きは当たる 0.6 秒前・0.3 秒前ともキック方向から蹴り足の側へ 18〜20°、当たる時 37〜39°
    (仕様の B-Human の姿勢「胴体をキック方向から蹴り足の側へ約 30° 回す」と同じ向き)。今までの指令は 0° を向かせるので、
    うまく蹴れた場面では方策が指令に逆らっていて、真後ろから来た場面では指令どおり 0° のまま歩いてつま先で押していた。"""
