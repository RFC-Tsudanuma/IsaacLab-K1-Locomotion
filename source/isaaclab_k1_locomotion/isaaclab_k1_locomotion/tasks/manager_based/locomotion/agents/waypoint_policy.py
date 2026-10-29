# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""凍結歩行 expert に歩行コマンドを出す上位方策 (経由点追従) とその PPO (2026-10-29)。

構成 (環境は `K1WaypointEnvCfg`):

* 上位 actor / critic: 小さな MLP。入力は ``nav_policy`` / ``nav_critic`` 観測グループ
  (経由点・ゴールの相対位置、現在の歩行コマンド、base 速度)。出力は 3 次元の歩行コマンド
  ``(vx, vy, ωz)`` (物理単位)。``command_limits`` でクランプし、ノルムが
  ``command_deadzone`` 未満なら 0 (= 歩行 expert の停止指令) にする。
* 歩行 expert: `HistoryActorCritic` を frozen_policies/best_walk から復元して凍結
  (``nn.Module`` として保持するので上位の checkpoint だけで自己完結する)。
  expert には環境観測の ``command`` グループと ``policy`` 履歴の最新フレームのコマンド枠を
  今 step のコマンドで差し替えた観測を渡す (歩行学習時と同じ「現在コマンドが観測に入って
  いる」見え方になる)。
* PPO のアクション空間は 3 次元コマンド。環境には ``[expert の関節指令 (12), コマンド (3)]``
  の 15 次元を渡し、後ろ 3 つを action 項 ``walk_command`` が受け取る
  (`WaypointPPO.act` が差し替える)。rollout storage には ``nav_*`` グループだけを保存し、
  巨大な履歴グループ (policy 4900 / critic 5400 次元) は保存しない。

rsl_rl の OnPolicyRunner は方策・アルゴリズムを ``eval(class_name)`` で解決するため、
本モジュール末尾で `WaypointActorCritic` / `WaypointPPO` を注入している。
"""

from __future__ import annotations

import os

import torch
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

from ..history_layout import COMMAND_DIM, HISTORY_LENGTH, POLICY_TERM_SPECS
from .gating_policy import COMMAND_TERM_NAME, DEFAULT_EXPERT_CHECKPOINTS, _term_block
from .history_actor_critic import HistoryActorCritic, RslRlHistoryActorCriticCfg

EXPERT_OBS_GROUPS = {"policy": ["command", "policy"], "critic": ["command", "critic"]}


class WaypointActorCritic(ActorCritic):
    """歩行コマンド (3 次元) を出す上位方策 + 凍結歩行 expert。"""

    is_hierarchical = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,  # 環境のアクション次元 (12 + 3)。上位の出力次元ではない。
        expert_checkpoint: str = "",
        expert_policy: dict | None = None,
        command_limits=(1.8, 0.9, 1.0),
        command_deadzone: float = 0.1,
        **kwargs,
    ):
        self.num_env_actions = int(num_actions)
        self.num_command = COMMAND_DIM
        # 基底は obs_groups (nav_policy / nav_critic) の次元で actor / critic / normalizer を作る
        super().__init__(obs, obs_groups, self.num_command, **kwargs)

        # --- 凍結 expert ---
        expert_cfg = dict(expert_policy or {})
        expert_cfg.pop("class_name", None)
        path = expert_checkpoint or DEFAULT_EXPERT_CHECKPOINTS.get("walk", "")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"WaypointActorCritic: 歩行 expert の checkpoint が見つかりません: {path}")
        self.expert = HistoryActorCritic(obs, EXPERT_OBS_GROUPS, self.num_env_actions - self.num_command, **expert_cfg)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.expert.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.expert.eval()
        for p in self.expert.parameters():
            p.requires_grad_(False)
        print(f"[WaypointActorCritic] walk expert <- {path}")

        # policy 履歴グループ内の「最新フレームのコマンド枠」の添字 (ブロック内は 古い → 新しい)
        cmd_block = _term_block(POLICY_TERM_SPECS, COMMAND_TERM_NAME, HISTORY_LENGTH)
        self.register_buffer("_policy_cmd_latest_idx", cmd_block[-COMMAND_DIM:], persistent=False)
        self.register_buffer("command_limits", torch.tensor(command_limits, dtype=torch.float), persistent=False)
        self.command_deadzone = float(command_deadzone)

    def command(self, raw: torch.Tensor) -> torch.Tensor:
        """上位の生アクション → 歩行 expert に渡すコマンド (クランプ + デッドゾーン)。"""
        cmd = torch.maximum(torch.minimum(raw, self.command_limits), -self.command_limits)
        # 歩行 expert は ||cmd|| < cmd_threshold で位相を止めて立つ (離散的な停止モード)。
        # ガウス方策の出力がちょうどその範囲に入ることはほぼ無いので、小さいコマンドは 0 に落とす。
        stop = torch.norm(cmd, dim=-1, keepdim=True) < self.command_deadzone
        return torch.where(stop, torch.zeros_like(cmd), cmd)

    @torch.no_grad()
    def expert_actions(self, obs: TensorDict, cmd: torch.Tensor) -> torch.Tensor:
        """コマンド枠を ``cmd`` に差し替えた観測で歩行 expert を評価した関節指令 (N, 12)。"""
        policy = obs["policy"].clone()
        policy[:, self._policy_cmd_latest_idx] = cmd
        groups = {key: value for key, value in obs.items()}
        groups["command"] = cmd
        groups["policy"] = policy
        return self.expert.act_inference(TensorDict(groups, batch_size=obs.batch_size))

    def env_actions(self, obs: TensorDict, raw: torch.Tensor) -> torch.Tensor:
        """環境に渡す 15 次元 ``[関節指令 (12), コマンド (3)]``。"""
        cmd = self.command(raw)
        return torch.cat([self.expert_actions(obs, cmd), cmd], dim=-1)


class WaypointPPO(PPO):
    """3 次元コマンドを PPO で学習し、環境には expert の関節指令 + コマンドを渡す。"""

    def __init__(self, policy: WaypointActorCritic, **kwargs):
        super().__init__(policy, **kwargs)
        if not getattr(policy, "is_hierarchical", False):
            raise TypeError("WaypointPPO には WaypointActorCritic を使うこと。")
        # storage に保存する観測グループ (上位 actor / critic が使うものだけ)
        self._storage_groups = sorted(set(policy.obs_groups["policy"]) | set(policy.obs_groups["critic"]))

    def _slim(self, obs: TensorDict) -> TensorDict:
        return TensorDict({key: obs[key] for key in self._storage_groups}, batch_size=obs.batch_size)

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.storage = RolloutStorage(
            training_type, num_envs, num_transitions_per_env, self._slim(obs), [self.policy.num_command], self.device
        )

    def act(self, obs):
        raw = super().act(obs)  # transition には 3 次元コマンド (生) が保存される
        self.transition.observations = self._slim(obs)
        return self.policy.env_actions(obs, raw)

    def act_inference(self, obs):
        return self.policy.env_actions(obs, self.policy.act_inference(obs))


@configclass
class RslRlWaypointActorCriticCfg(RslRlPpoActorCriticCfg):
    """`WaypointActorCritic` 用の設定。"""

    class_name: str = "WaypointActorCritic"

    init_noise_std: float = 0.5
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [256, 128]
    critic_hidden_dims: list[int] = [256, 128]
    activation: str = "elu"

    expert_checkpoint: str = DEFAULT_EXPERT_CHECKPOINTS.get("walk", "")
    """凍結する歩行 expert の checkpoint。既定は frozen_policies/best_walk/model_*.pt (番号最大)。
    上書きは hydra 形式 ``agent.policy.expert_checkpoint=/abs/path/model.pt``。"""

    expert_policy: RslRlHistoryActorCriticCfg = RslRlHistoryActorCriticCfg(
        init_noise_std=0.7207805082202461,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    """歩行 expert (HistoryActorCritic) のアーキテクチャ。checkpoint と一致させること。"""

    command_limits: tuple[float, float, float] = (1.8, 0.9, 1.0)
    """コマンドのクランプ (|vx|, |vy|, |ωz|)。歩行 expert の学習レンジ (2.0 / 0.9 / 1.0) 以内にする。"""

    command_deadzone: float = 0.1
    """||cmd|| がこれ未満なら 0 (停止指令) にする。"""


@configclass
class RslRlWaypointPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """`WaypointPPO` 用の設定。"""

    class_name: str = "WaypointPPO"


# OnPolicyRunner は eval(class_name) を on_policy_runner モジュールの名前空間で評価する。
import rsl_rl.runners.on_policy_runner as _rsl_rl_on_policy_runner  # noqa: E402

_rsl_rl_on_policy_runner.WaypointActorCritic = WaypointActorCritic
_rsl_rl_on_policy_runner.WaypointPPO = WaypointPPO
