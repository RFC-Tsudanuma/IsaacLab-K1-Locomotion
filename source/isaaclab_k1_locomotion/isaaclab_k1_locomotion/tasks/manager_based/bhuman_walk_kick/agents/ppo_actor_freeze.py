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

from rsl_rl.algorithms import PPO


class PPOActorFreeze(PPO):
    def __init__(self, policy, actor_freeze_iterations: int = 0, **kwargs):
        super().__init__(policy, **kwargs)
        self.actor_freeze_iterations = int(actor_freeze_iterations)
        self._num_updates = 0

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
        return loss_dict
