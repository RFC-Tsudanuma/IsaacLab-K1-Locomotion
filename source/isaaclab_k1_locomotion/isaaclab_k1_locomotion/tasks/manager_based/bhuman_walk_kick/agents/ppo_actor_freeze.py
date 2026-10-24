# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""最初の数 update だけ actor を固定して critic を慣らす PPO (rsl_rl)。

歩行の重みから始めると、critic も歩行の報酬で学習した値のまま。報酬 (ボール関係) が変わった
直後に PPO を回すと、合わない価値の推定で actor が大きく動いて歩き方が崩れる
(feat/bhuman_kick で実測: 約 100 iteration で全エピソード転倒に戻った)。
``actor_freeze_iterations`` 回の update は actor (と std) の勾配を止め、学習率も固定する。
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO


class PPOActorFreeze(PPO):
    """``std_range`` (B-Human の entropy_range に相当): update のたびに方策のノイズ (std) をこの範囲に clamp する。
    B-Human は「エントロピーを範囲内に保つ」損失で探索の広さを段ごとに決めている (Stage 1 は広く、後段で狭く)。
    うちの std は scalar (noise_std_type="scalar") なので、値を直接 clamp する。None なら何もしない。"""

    def __init__(self, policy, actor_freeze_iterations: int = 0, std_range: tuple | None = None, **kwargs):
        super().__init__(policy, **kwargs)
        self.actor_freeze_iterations = int(actor_freeze_iterations)
        self.std_range = None if std_range is None else (float(std_range[0]), float(std_range[1]))
        self._num_updates = 0
        self._clamp_std()

    def _clamp_std(self):
        if self.std_range is None or self.policy.noise_std_type != "scalar":
            return
        with torch.no_grad():
            self.policy.std.clamp_(*self.std_range)

    def _actor_params(self):
        return [p for n, p in self.policy.named_parameters() if not n.startswith("critic")]

    def update(self):
        freeze = self._num_updates < self.actor_freeze_iterations
        if freeze:
            schedule = self.schedule
            self.schedule = "fixed"
            for p in self._actor_params():
                p.requires_grad_(False)
        try:
            loss_dict = super().update()
        finally:
            if freeze:
                self.schedule = schedule
                for p in self._actor_params():
                    p.requires_grad_(True)
        self._num_updates += 1
        self._clamp_std()
        return loss_dict
