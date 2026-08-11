# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""キック検出フラグを出力するためのダミー行動項。

物理には一切作用しない 1 次元の出力を行動空間に足すだけの項。値は
:func:`..rewards.kick_flag` が報酬で「このエピソードで蹴ったか (kick_done)」に
一致するよう教育する。

なぜ「補助ヘッド」ではなく行動空間なのか
----------------------------------------
kick_done は latch (0 → 1 の単調フラグ) なので、判定には **記憶が要る**。ポリシーは
feedforward MLP なので、latch から 2 秒経った時点の観測 (ボールが遠い・遅い) だけでは
「蹴った後」と「一度も蹴っていない」を区別できない。

行動空間に入れると前ステップの出力が ``prev_kick_flag`` 観測として戻ってくるので、
この自己ループが latch を保持する 1 bit のメモリになる。GRU に切り替えるより遥かに安い。

代償として、この 1 次元が PPO の Gaussian に含まれるため log_prob / entropy / KL に
混ざる (adaptive LR がわずかに汚れる)。実測で std が潰れない・KL が暴れないかを
Loss/learning_rate と Policy/mean_noise_std で見ること。
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class KickFlagAction(ActionTerm):
    """物理に作用しない 1 次元の出力。生値をそのまま保持するだけ。

    解釈は ``sigmoid(a) > 0.5`` で「蹴った」。報酬側も sigmoid を通して採点する
    (:func:`..rewards.kick_flag`)。
    """

    cfg: KickFlagActionCfg

    def __init__(self, cfg: KickFlagActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        # 何も加工しない。ActionTerm の抽象プロパティを満たすためだけの実装。
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        """何もしない。フラグはシーンに一切作用しない。"""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # ActionManager.reset() は自身の _action バッファはゼロにするが、各項の
        # raw_actions までは触らない (JointAction も同様)。この項の raw_actions は
        # メトリクスから直接読むので、前エピソードの値を持ち越さないようここで消す。
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0


@configclass
class KickFlagActionCfg(ActionTermCfg):
    """:class:`KickFlagAction` の設定。

    ``asset_name`` は ActionTerm 基底クラスがシーンから引くために必要なだけで、
    実際には使わない。
    """

    class_type: type[ActionTerm] = KickFlagAction
    asset_name: str = "robot"
