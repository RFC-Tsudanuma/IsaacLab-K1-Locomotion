# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール奪取方策 (Isaac-Steal-K1) とその PPO (2026-11-03)。

Block (agents/block_policy.py) からボール encoder・latent 蒸留を外した構成:

* 歩行 expert: `WaypointActorCritic` と同じく frozen_policies/best_walk を凍結して内包 (事前分布)。
* 方策の出力 (15 次元) = ``[歩行コマンド (3), 関節残差 (12)]``。環境へは
  ``[expert の関節指令 + residual_scale × 残差 (12), コマンド (3), 残差 (12)]`` の 27 次元を渡す。
  残差 0 なら expert の歩行そのもの。
* actor / critic は MLP。入力は ``steal_policy`` / ``steal_critic`` 観測グループ (内界 + ボール + 障害物)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from rsl_rl.networks import MLP
from rsl_rl.storage import RolloutStorage

from ..mdp.block import JOINT_DIM
from .waypoint_policy import RslRlWaypointActorCriticCfg, WaypointActorCritic, WaypointPPO


class StealActorCritic(WaypointActorCritic):
    """歩行コマンド (3) + 関節残差 (12) を出す MLP 方策 + 凍結歩行 expert。"""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,  # 環境のアクション次元 (12 + 3 + 12)。方策の出力次元ではない。
        residual_scale: float = 0.5,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        self.num_policy_actions = 3 + JOINT_DIM
        # 基底は「num_actions − 3」を expert の出力次元にするので 15 を渡す (expert は 12 出力)。
        # normalizer / 凍結 expert / command() / expert_actions() を流用し、actor・critic・std は置き換える。
        super().__init__(
            obs,
            obs_groups,
            self.num_policy_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        self.num_env_actions = int(num_actions)
        self.residual_scale = float(residual_scale)
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        self.actor = MLP(num_actor_obs, self.num_policy_actions, list(actor_hidden_dims), activation)
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(self.num_policy_actions))
        else:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_policy_actions)))
        print(f"Actor (steal): {self.actor}")
        print(f"Critic (steal): {self.critic}")

    def env_actions(self, obs: TensorDict, raw: torch.Tensor) -> torch.Tensor:
        """環境に渡す 27 次元 ``[expert + 残差 (12), コマンド (3), 残差 (12)]``。"""
        cmd = self.command(raw[:, :3])
        res = self.residual_scale * raw[:, 3:]
        return torch.cat([self.expert_actions(obs, cmd) + res, cmd, res], dim=-1)

    # 凍結 expert は checkpoint に含めない (block_policy.py / gating_policy.py と同じ対処)
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        for k in [k for k in sd if k.startswith("expert.")]:
            del sd[k]
        return sd

    def load_state_dict(self, state_dict, strict=True):
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("expert.")}
        missing = [k for k in self.state_dict() if k not in filtered]
        if missing:
            raise KeyError(f"checkpoint に方策側のキーが無い: {missing[:4]} ...")
        return super().load_state_dict(filtered, strict=False)


class StealPPO(WaypointPPO):
    """15 次元 (コマンド + 残差) を PPO で学習する (storage の action 次元だけ WaypointPPO と異なる)。"""

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.storage = RolloutStorage(
            training_type, num_envs, num_transitions_per_env, self._slim(obs), [self.policy.num_policy_actions], self.device
        )


@configclass
class RslRlStealActorCriticCfg(RslRlWaypointActorCriticCfg):
    """`StealActorCritic` 用の設定 (expert の checkpoint / アーキテクチャ / command_limits は waypoint と共通)。"""

    class_name: str = "StealActorCritic"

    init_noise_std: float = 0.5
    actor_hidden_dims: list[int] = [256, 128]
    critic_hidden_dims: list[int] = [256, 128]

    residual_scale: float = 0.5
    """方策の残差出力 → 関節指令 (raw action 単位) の倍率。Block で 1.0 は転倒が増えて逆効果だった。"""

    command_limits: tuple[float, float, float] = (2.0, 0.9, 2.0)
    """waypoint 側の既定が変わっても Steal は固定 (Block と同じ上限)。"""


@configclass
class RslRlStealPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "StealPPO"


# OnPolicyRunner は eval(class_name) を on_policy_runner モジュールの名前空間で評価する。
import rsl_rl.runners.on_policy_runner as _rsl_rl_on_policy_runner  # noqa: E402

_rsl_rl_on_policy_runner.StealActorCritic = StealActorCritic
_rsl_rl_on_policy_runner.StealPPO = StealPPO
