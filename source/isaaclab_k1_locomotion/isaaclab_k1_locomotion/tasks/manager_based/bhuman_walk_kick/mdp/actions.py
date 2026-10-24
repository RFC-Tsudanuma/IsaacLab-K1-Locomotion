# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""B-Human の 13 番目の action: 歩行周波数。

B-Human のキック方策は 12 関節に加えて歩行の周波数も自分で決める
(``freq = clip(a, -0.5, 0.5) + target_frequence``)。足を出すタイミングと、蹴る一歩の長さを
方策が変えられるようにするためのもの。

土台の walk は「速度コマンドから決まる周波数 (1.8〜2.0 Hz) を毎ステップ積分した位相」で
歩いている (locomotion.mdp.events.get_gait_phase)。ここではその周波数に action を足す:

    freq = compute_cmd_phase_freq(cmd) + clip(a, -0.5, 0.5)

足す量は ``env._gait_freq_action_offset`` に書き、``compute_cmd_phase_freq`` が読む。
a = 0 なら歩行と同じ周波数・同じ位相になるので、歩行の重みから始めたとき
(13 番目の出力の重みを 0 にしたとき) の動きは歩行のまま変わらない。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ...locomotion.mdp.events import GAIT_FREQ_ACTION_ATTR

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GaitFrequencyAction(ActionTerm):
    cfg: GaitFrequencyActionCfg

    def __init__(self, cfg: GaitFrequencyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, 1, device=self.device)
        self._offset = torch.zeros(self.num_envs, device=self.device)
        setattr(env, GAIT_FREQ_ACTION_ATTR, self._offset)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._offset.unsqueeze(-1)

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        lo, hi = self.cfg.frequency_offset_limit
        self._offset[:] = actions[:, 0].clamp(lo, hi)

    def apply_actions(self):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        self._offset[env_ids] = 0.0


@configclass
class GaitFrequencyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = GaitFrequencyAction
    frequency_offset_limit: tuple[float, float] = (-0.5, 0.5)
    """周波数オフセットとして使う action の範囲 [Hz] (B-Human: normalization.action_frequence_limit)。"""
