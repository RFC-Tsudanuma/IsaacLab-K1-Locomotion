# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール阻止方策 (Isaac-Block-K1) とその PPO (2026-11-01)。

構成 (環境は `K1BlockEnvCfg`):

* 歩行 expert: `WaypointActorCritic` と同じく frozen_policies/best_walk の `HistoryActorCritic`
  を凍結して内包する (事前分布)。
* 方策の出力 (15 次元) = ``[歩行コマンド (3), 関節残差 (12)]``。環境へは
  ``[expert の関節指令 + residual_scale × 残差 (12), コマンド (3), 残差 (12)]`` の 27 次元を渡す
  (`BlockPPO.act` が組み立てる)。残差 0 なら expert の歩行そのもの。
* ボール encoder: causal transformer。履歴 (T, 4) → 各時刻の latent (T, L)。actor は
  **ノイズ付き観測** (``ball_noisy``) の encoder、critic は **真値** (``ball_true``) の encoder を持ち、
  それぞれ最後の時刻の latent を内界観測と連結して MLP に入れる。
* latent 蒸留: `BlockPPO.update` が PPO 更新後に 1 epoch、
  ``MSE(enc_noisy(ball_noisy), stopgrad(enc_true(ball_true)))`` を全時刻で最小化する
  (真値 encoder は critic の価値損失だけで学習される)。ノイズ付き encoder は PPO の方策勾配も受ける。
* 観測の正規化は rsl_rl 基底の EmpiricalNormalization を「内界 + ボール履歴」の連結全体に掛ける
  (`HistoryActorCritic` と同じ流儀)。

rsl_rl の OnPolicyRunner は方策・アルゴリズムを ``eval(class_name)`` で解決するため、
本モジュール末尾で `BlockActorCritic` / `BlockPPO` を注入している。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from rsl_rl.storage import RolloutStorage

from ..mdp.block import BALL_DIM, BALL_HISTORY, JOINT_DIM
from .block_nets import BlockHead, CausalTransformerEncoder
from .waypoint_policy import RslRlWaypointActorCriticCfg, WaypointActorCritic, WaypointPPO


class BlockActorCritic(WaypointActorCritic):
    """歩行コマンド (3) + 関節残差 (12) を出す方策 + 凍結歩行 expert + ボール encoder ×2。"""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,  # 環境のアクション次元 (12 + 3 + 12)。方策の出力次元ではない。
        ball_history: int = BALL_HISTORY,
        latent_dim: int = 32,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        residual_scale: float = 0.5,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        self.num_policy_actions = 3 + JOINT_DIM
        # 基底 (WaypointActorCritic) は「num_actions − 3」を expert の出力次元にするので 15 を渡す
        # (expert は 12 出力)。normalizer / 凍結 expert / command() / expert_actions() を流用し、
        # actor・critic・std は下で置き換える。
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

        heads = {}
        for set_name in ("policy", "critic"):
            groups = obs_groups[set_name]
            ball_dim = obs[groups[-1]].shape[-1]
            if ball_dim != ball_history * BALL_DIM:
                raise ValueError(
                    f"obs_groups['{set_name}'] の末尾グループ '{groups[-1]}' はボール履歴 ({ball_history} × {BALL_DIM}"
                    f" = {ball_history * BALL_DIM}) のはずですが {ball_dim} 次元です。"
                )
            heads[set_name] = sum(obs[g].shape[-1] for g in groups[:-1])

        def encoder():
            return CausalTransformerEncoder(BALL_DIM, ball_history, d_model, nhead, num_layers, latent_dim)

        self.actor = BlockHead(heads["policy"], ball_history, encoder(), self.num_policy_actions, list(actor_hidden_dims), activation)
        self.critic = BlockHead(heads["critic"], ball_history, encoder(), 1, list(critic_hidden_dims), activation)
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(self.num_policy_actions))
        else:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_policy_actions)))
        print(f"Actor (block): {self.actor}")
        print(f"Critic (block): {self.critic}")

    def env_actions(
        self, obs: TensorDict, raw: torch.Tensor, blend: torch.Tensor | None = None, cmd_ext: torch.Tensor | None = None
    ) -> torch.Tensor:
        """環境に渡す 27 次元 ``[expert + 残差 (12), コマンド (3), 残差 (12)]``。

        ``blend`` (N,) を与えると歩行との混合 (推論用、学習では常に 1): w=0 で外部コマンド ``cmd_ext`` (N, 3)
        による純歩行、w=1 で阻止方策そのもの。阻止方策は歩行 expert を内包し残差 0 で歩行と一致するので、
        別のゲート網は要らない (2026-11-02 の判断)。
        """
        cmd = self.command(raw[:, :3])
        res = self.residual_scale * raw[:, 3:]
        if blend is not None:
            w = blend.unsqueeze(1)
            cmd = (1.0 - w) * (cmd_ext if cmd_ext is not None else torch.zeros_like(cmd)) + w * cmd
            res = w * res
        return torch.cat([self.expert_actions(obs, cmd) + res, cmd, res], dim=-1)

    # 凍結 expert は checkpoint に含めない (expert を差し替えて resume したとき古い expert が復元されるのを防ぐ。
    # gating_policy.py と同じ対処)。古い checkpoint の expert.* キーは load 時に無視する。
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

    def noisy_latents(self, obs: TensorDict) -> torch.Tensor:
        return self.actor.latents(self.actor_obs_normalizer(self.get_actor_obs(obs)))

    def true_latents(self, obs: TensorDict) -> torch.Tensor:
        return self.critic.latents(self.critic_obs_normalizer(self.get_critic_obs(obs)))


class BlockPPO(WaypointPPO):
    """15 次元 (コマンド + 残差) を PPO で学習し、PPO 更新後にノイズ付き encoder を真値 encoder へ蒸留する。"""

    def __init__(self, policy: BlockActorCritic, latent_loss_coef: float = 1.0, **kwargs):
        super().__init__(policy, **kwargs)
        self.latent_loss_coef = float(latent_loss_coef)

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.storage = RolloutStorage(
            training_type, num_envs, num_transitions_per_env, self._slim(obs), [self.policy.num_policy_actions], self.device
        )

    def update(self):
        loss_dict = super().update()
        total, count = 0.0, 0
        for obs_batch, *_ in self.storage.mini_batch_generator(self.num_mini_batches, 1):
            target = self.policy.true_latents(obs_batch).detach()
            loss = self.latent_loss_coef * F.mse_loss(self.policy.noisy_latents(obs_batch), target)
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total += loss.item()
            count += 1
        loss_dict["latent"] = total / max(count, 1)
        return loss_dict


@configclass
class RslRlBlockActorCriticCfg(RslRlWaypointActorCriticCfg):
    """`BlockActorCritic` 用の設定 (expert の checkpoint / アーキテクチャ / command_limits は waypoint と共通)。"""

    class_name: str = "BlockActorCritic"

    init_noise_std: float = 0.5
    actor_hidden_dims: list[int] = [256, 128]
    critic_hidden_dims: list[int] = [256, 128]

    ball_history: int = BALL_HISTORY
    """ボール観測の履歴長 (環境側 BALL_HISTORY と一致させる)。"""
    latent_dim: int = 32
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    residual_scale: float = 0.5
    """方策の残差出力 → 関節指令 (raw action 単位) の倍率。env 側 scale 0.5 と合わせて 1 単位 ≈ 0.25 rad。"""


@configclass
class RslRlBlockPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "BlockPPO"

    latent_loss_coef: float = 1.0
    """ノイズ付き encoder → 真値 encoder の latent 蒸留損失の係数。"""


# OnPolicyRunner は eval(class_name) を on_policy_runner モジュールの名前空間で評価する。
import rsl_rl.runners.on_policy_runner as _rsl_rl_on_policy_runner  # noqa: E402

_rsl_rl_on_policy_runner.BlockActorCritic = BlockActorCritic
_rsl_rl_on_policy_runner.BlockPPO = BlockPPO
