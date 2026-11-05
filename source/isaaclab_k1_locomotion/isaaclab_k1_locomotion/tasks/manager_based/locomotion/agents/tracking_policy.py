# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""凍結ゲート (歩行 expert + 回転 expert + ゲート網) に 5 次元 ``[vx, vy, ωz, Δψ, mode]`` を出す
上位方策 (経路追従 + 姿勢到達 / 見ながら到達, 2026-11-05)。

* 下位は `GatingActorCritic` を frozen_policies/best_gating の checkpoint から復元して凍結する。
  expert 2 本はゲートと同じくパス (best_walk / best_turn の番号最大) から読む。実機の融合 ONNX
  (gate_v4 = walk 52700 + turn 7999 + gate 5897) と同一構成。
* 上位の生出力 → `command`: 歩行コマンドはクランプ + デッドゾーン、Δψ 残差は ±pivot_limit、mode は ``> 0`` で 1。
  環境に渡す 17 次元は ``[ゲート混合の関節指令 12, command 5]``。
* 上位出力は 1 step 遅れて下位に反映される (ゲートには env が出した観測をそのまま渡す。
  ゲートの入力 (expert_mode / command / 履歴) を学習時と同じ見え方に保つため)。
* PPO は `WaypointPPO` をそのまま使う (``num_command`` / ``env_actions`` の duck typing)。
* ゲートは checkpoint に含めない (凍結、常にパスから読む)。
"""

from __future__ import annotations

import os

import torch
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from rsl_rl.modules import ActorCritic

from ..mdp.tracking import TRACK_ACTION_DIM
from .gating_policy import GatingActorCritic, RslRlGatingActorCriticCfg, _default_expert_checkpoint
from .waypoint_policy import RslRlWaypointPpoAlgorithmCfg  # noqa: F401  (WaypointPPO を runner に注入する副作用)

GATE_OBS_GROUPS = {"policy": ["command", "policy"], "critic": ["command", "critic"]}
DEFAULT_GATE_CHECKPOINT = _default_expert_checkpoint("gating")
"""frozen_policies/best_gating/model_*.pt (番号最大)。"""


class TrackingActorCritic(ActorCritic):
    """5 次元の上位コマンドを出す方策 + 凍結ゲート (walk / turn expert 込み)。"""

    is_hierarchical = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,  # 環境のアクション次元 (12 + 5)。上位の出力次元ではない。
        gate_checkpoint: str = "",
        gate_policy: dict | None = None,
        command_limits=(2.0, 0.9, 1.0),
        command_deadzone: float = 0.1,
        pivot_limit: float = 1.0,
        min_noise_std=(0.0, 0.0, 0.0, 0.0, 0.3),
        **kwargs,
    ):
        self.num_env_actions = int(num_actions)
        self.num_command = TRACK_ACTION_DIM
        super().__init__(obs, obs_groups, self.num_command, **kwargs)

        gate_cfg = dict(gate_policy or {})
        gate_cfg.pop("class_name", None)
        path = gate_checkpoint or DEFAULT_GATE_CHECKPOINT
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"TrackingActorCritic: ゲートの checkpoint が見つかりません: {path}")
        self.gate = GatingActorCritic(obs, GATE_OBS_GROUPS, self.num_env_actions - self.num_command, **gate_cfg)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.gate.load_state_dict(ckpt["model_state_dict"])  # experts.* は無視され expert はパスから読まれる
        self.gate.eval()
        for p in self.gate.parameters():
            p.requires_grad_(False)
        print(f"[TrackingActorCritic] gate <- {path}")

        self.register_buffer("command_limits", torch.tensor(command_limits, dtype=torch.float), persistent=False)
        self.register_buffer("min_noise_std", torch.tensor(min_noise_std, dtype=torch.float), persistent=False)
        self.command_deadzone = float(command_deadzone)
        self.pivot_limit = float(pivot_limit)

    def update_distribution(self, obs):
        # mode 次元の探索ノイズに下限を置く。v2/v3 では std が 0.1 まで潰れた後 mode の平均が walk 側に
        # 寄ると pivot を試す機会が無くなり、pivot の価値を学べなかった。しきい値 (0) 付近でだけ効き、
        # 平均が十分離れれば実質無効。推論 (act_inference) には影響しない。
        mean = self.actor(obs)
        std = self.std.expand_as(mean) if self.noise_std_type == "scalar" else torch.exp(self.log_std).expand_as(mean)
        self.distribution = torch.distributions.Normal(mean, torch.maximum(std, self.min_noise_std))

    def command(self, raw: torch.Tensor) -> torch.Tensor:
        """生出力 → env に渡す 5 次元 [vx, vy, ωz (クランプ + デッドゾーン), Δψ 残差 (±pivot_limit), mode (0/1)]。"""
        cmd = torch.maximum(torch.minimum(raw[:, :3], self.command_limits), -self.command_limits)
        stop = torch.norm(cmd, dim=-1, keepdim=True) < self.command_deadzone
        cmd = torch.where(stop, torch.zeros_like(cmd), cmd)
        dpsi = raw[:, 3:4].clamp(-self.pivot_limit, self.pivot_limit)
        mode = (raw[:, 4:5] > 0.0).float()
        return torch.cat([cmd, dpsi, mode], dim=-1)

    @torch.no_grad()
    def env_actions(self, obs: TensorDict, raw: torch.Tensor) -> torch.Tensor:
        """環境に渡す 17 次元 ``[ゲート混合の関節指令 (12), command (5)]``。"""
        return torch.cat([self.gate.act_inference_blended(obs), self.command(raw)], dim=-1)

    # ── 凍結ゲートは checkpoint に含めない (パスから読む) ───────────────────
    _GATE_PREFIX = "gate."

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        for k in [k for k in sd if self._GATE_PREFIX in k]:
            del sd[k]
        return sd

    def load_state_dict(self, state_dict, strict=True):
        filtered = {k: v for k, v in state_dict.items() if self._GATE_PREFIX not in k}
        missing = [k for k in self.state_dict() if k not in filtered]
        if missing and strict:
            raise RuntimeError(f"TrackingActorCritic: checkpoint に無いキー: {missing[:6]}")
        return super().load_state_dict(filtered, strict=False)


@configclass
class RslRlTrackingActorCriticCfg(RslRlPpoActorCriticCfg):
    """`TrackingActorCritic` 用の設定。"""

    class_name: str = "TrackingActorCritic"

    init_noise_std: float = 0.5
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [256, 128]
    critic_hidden_dims: list[int] = [256, 128]
    activation: str = "elu"

    gate_checkpoint: str = DEFAULT_GATE_CHECKPOINT
    """凍結ゲートの checkpoint。既定は frozen_policies/best_gating/model_*.pt (番号最大)。"""

    gate_policy: RslRlGatingActorCriticCfg = RslRlGatingActorCriticCfg()
    """ゲート (GatingActorCritic) のアーキテクチャと expert のパス。gate_v4 の学習設定と一致させること。"""

    command_limits: tuple[float, float, float] = (2.0, 0.9, 1.0)
    """歩行コマンドのクランプ (|vx|, |vy|, |ωz|) = 歩行 expert の学習レンジ。"""

    command_deadzone: float = 0.1
    """||cmd|| がこれ未満なら 0 (停止指令)。"""

    pivot_limit: float = 1.0
    """Δψ 残差のクランプ [rad] (pivot 目標 = env の既定目標 + 残差)。"""

    min_noise_std: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.3)
    """学習時の探索ノイズ std の下限 (次元ごと)。mode 次元だけ 0.3 を残して pivot を試し続けさせる。"""


# OnPolicyRunner は eval(class_name) を on_policy_runner モジュールの名前空間で評価する。
import rsl_rl.runners.on_policy_runner as _rsl_rl_on_policy_runner  # noqa: E402

_rsl_rl_on_policy_runner.TrackingActorCritic = TrackingActorCritic
