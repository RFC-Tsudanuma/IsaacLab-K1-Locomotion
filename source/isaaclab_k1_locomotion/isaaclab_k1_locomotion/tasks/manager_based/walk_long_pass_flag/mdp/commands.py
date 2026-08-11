# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""キック検出フラグのメトリクスを足した KickDirectionCommand。

キック側のメトリクス (kick_rate / kick_dir_error_deg / …) は
:class:`~...walk_kick.mdp.commands.KickDirectionCommand` が既に全部持っているので、
フラグの 3 指標だけを追加する。

cfg クラスは新設しない。親の :class:`KickDirectionCommandCfg` の ``class_type`` を
差し替えるだけで有効になる::

    self.commands.kick_direction.class_type = mdp.KickFlagDirectionCommand

こうすると walk_kick → loop_pass → long_pass が積み上げてきた設定値
(target_speed_range / ranges / v_thresh …) を一切書き写さずに済む。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from ...walk_kick.mdp.commands import KickDirectionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from ...walk_kick.mdp.commands import KickDirectionCommandCfg

#: フラグを出力している行動項の名前 (env cfg の ActionsCfg と一致させること)。
FLAG_ACTION_NAME = "kick_flag"


class KickFlagDirectionCommand(KickDirectionCommand):
    """親のメトリクスに加えて、フラグ出力の当たり具合を記録する。

    追加されるのは ``Metrics/kick_direction/`` 以下の 3 つ:

    ``flag_pred_final``
      エピソード最終ステップの ``sigmoid(a_flag)``。**``kick_rate`` と並べて読む**。
      両方 0.99 付近なら「蹴ったエピソードの最後にちゃんと 1 を出せている」。
      kick_rate だけ高くて flag_pred_final が低いなら latch を検出できていない。

    ``flag_err_mean``
      エピソード全体を通した ``|sigmoid(a_flag) − kick_done|`` の平均。総合精度。
      「常に 0 を出す」に潰れると正例の割合 (0.2-0.3 程度) に張り付く。

    ``flag_pre_latch_pred``
      latch **前** の ``sigmoid(a_flag)`` の平均 = 誤検出率。0 に近いほど良い。
      ``flag_pred_final`` が高くてもこれが高いなら「常に 1 を出しているだけ」。

    3 つ揃えて初めて「latch を検出できている」と言える。1 つだけでは必ず潰れ方を見逃す。
    """

    cfg: KickDirectionCommandCfg

    def __init__(self, cfg: KickDirectionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        def _z() -> torch.Tensor:
            return torch.zeros(self.num_envs, device=self.device)

        self.metrics["flag_pred_final"] = _z()
        self.metrics["flag_err_mean"] = _z()
        self.metrics["flag_pre_latch_pred"] = _z()

        # エピソード内の累積。episode_length_buf == 0 (= 今リセットされた env) で
        # 自動的に巻き戻すので、明示的な reset フックは要らない。
        self._flag_err_sum = _z()
        self._flag_pre_sum = _z()
        self._flag_pre_cnt = _z()

    def _update_metrics(self):
        super()._update_metrics()

        state = getattr(self._env, "_kick_latch_state", None)
        if state is None:
            return
        if FLAG_ACTION_NAME not in self._env.action_manager.active_terms:
            return

        p = torch.sigmoid(self._env.action_manager.get_term(FLAG_ACTION_NAME).raw_actions[:, 0])
        y = state["kick_done"].float()

        # CommandManager.compute() は _reset_idx の **後** に走るので、今リセットされた
        # env は episode_length_buf == 0 になっている。そこで累積を巻き戻す
        # (これらの env の値は _reset_idx の中で既にログ済みなので上書きして構わない)。
        just_reset = self._env.episode_length_buf == 0
        zero = torch.zeros_like(p)

        err = (p - y).abs()
        self._flag_err_sum = torch.where(just_reset, zero, self._flag_err_sum + err)

        pre = (y < 0.5).float()
        self._flag_pre_sum = torch.where(just_reset, zero, self._flag_pre_sum + p * pre)
        self._flag_pre_cnt = torch.where(just_reset, zero, self._flag_pre_cnt + pre)

        n = self._env.episode_length_buf.clamp(min=1).float()
        self.metrics["flag_pred_final"] = p
        self.metrics["flag_err_mean"] = self._flag_err_sum / n
        self.metrics["flag_pre_latch_pred"] = self._flag_pre_sum / self._flag_pre_cnt.clamp(min=1.0)
