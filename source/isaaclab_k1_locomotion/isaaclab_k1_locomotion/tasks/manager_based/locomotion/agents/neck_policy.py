# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""首安定化の残差方策 (Isaac-Neck-K1, 2026-11-20)。環境は `K1NeckEnvCfg`。

* 方策: 小さな MLP。入力は ``neck_policy`` / ``neck_critic`` 観測グループ、出力は首 2 軸の残差 (2 次元)。
* 歩行 expert: `WaypointActorCritic` と同じく frozen_policies/best_walk の `HistoryActorCritic` を凍結して
  内包する。歩行コマンドは環境が乱数で生成するので、観測はそのまま expert に渡す。
* 環境へは ``[expert の関節指令 (12), 残差 (2)]`` を渡す。PPO は `WaypointPPO` をそのまま使う
  (storage には neck_* グループだけを保存する)。
"""

from __future__ import annotations

import os

import torch
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from rsl_rl.modules import ActorCritic

from ..mdp.neck import NECK_DIM
from .gating_policy import DEFAULT_EXPERT_CHECKPOINTS
from .history_actor_critic import HistoryActorCritic, RslRlHistoryActorCriticCfg
from .waypoint_policy import EXPERT_OBS_GROUPS, RslRlWaypointActorCriticCfg


class NeckActorCritic(ActorCritic):
    """首の残差 (2 次元) を出す方策 + 凍結歩行 expert。"""

    is_hierarchical = True

    def __init__(self, obs, obs_groups, num_actions, expert_checkpoint: str = "", expert_policy: dict | None = None, **kwargs):
        self.num_command = NECK_DIM  # WaypointPPO が storage のアクション次元として読む
        super().__init__(obs, obs_groups, NECK_DIM, **kwargs)

        expert_cfg = dict(expert_policy or {})
        expert_cfg.pop("class_name", None)
        path = expert_checkpoint or DEFAULT_EXPERT_CHECKPOINTS.get("walk", "")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"NeckActorCritic: 歩行 expert の checkpoint が見つかりません: {path}")
        self.expert = HistoryActorCritic(obs, EXPERT_OBS_GROUPS, int(num_actions) - NECK_DIM, **expert_cfg)
        self.expert.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        self.expert.eval()
        for p in self.expert.parameters():
            p.requires_grad_(False)
        print(f"[NeckActorCritic] walk expert <- {path}")

    @torch.no_grad()
    def env_actions(self, obs: TensorDict, raw: torch.Tensor) -> torch.Tensor:
        """環境に渡す 14 次元 ``[関節指令 (12), 首の残差 (2)]``。"""
        return torch.cat([self.expert.act_inference(obs), raw], dim=-1)

    # 凍結 expert は checkpoint に含めない (block_policy.py と同じ対処)
    def state_dict(self, *args, **kwargs):
        return {k: v for k, v in super().state_dict(*args, **kwargs).items() if not k.startswith("expert.")}

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict({k: v for k, v in state_dict.items() if not k.startswith("expert.")}, strict=False)


@configclass
class RslRlNeckActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "NeckActorCritic"

    init_noise_std: float = 0.1  # × env 側 scale 0.1 → 残差の探索ノイズ 0.01 rad (首 PD を通ると約 0.08 rad/s の揺れになる)
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [128, 64]
    critic_hidden_dims: list[int] = [128, 64]
    activation: str = "elu"

    expert_checkpoint: str = DEFAULT_EXPERT_CHECKPOINTS.get("walk", "")
    """凍結する歩行 expert。既定は frozen_policies/best_walk/model_*.pt (番号最大)。"""
    expert_policy: RslRlHistoryActorCriticCfg = RslRlWaypointActorCriticCfg().expert_policy
    """歩行 expert のアーキテクチャ (waypoint と共通)。"""


# OnPolicyRunner は eval(class_name) を on_policy_runner モジュールの名前空間で評価する。
import rsl_rl.runners.on_policy_runner as _rsl_rl_on_policy_runner  # noqa: E402

_rsl_rl_on_policy_runner.NeckActorCritic = NeckActorCritic
