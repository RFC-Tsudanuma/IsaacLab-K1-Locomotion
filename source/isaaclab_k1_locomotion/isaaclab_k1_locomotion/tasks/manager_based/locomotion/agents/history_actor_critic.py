# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""履歴バッファ付き観測から「最新コマンド + 直近数ステップ + CNN 履歴特徴」を MLP に入力する ActorCritic。

環境側 (K1FlatEnvCfg) は以下の観測グループを出力する:

* ``command``: 最新の歩行コマンド (3)
* ``policy``:  コマンドを除いた actor 観測の HISTORY_LENGTH ステップ分の履歴 (ノイズあり)
* ``critic``:  コマンドを除いた critic 観測の HISTORY_LENGTH ステップ分の履歴 (ノイズなし・特権情報込み)

actor / critic それぞれのネットワーク (:class:`HistoryEncoderHead`) は、正規化済みの
「command + 全履歴」ベクトルを受け取り、

1. 履歴から各項の直近 ``mlp_history_steps`` ステップ分をスライス
2. 全履歴 (HISTORY_LENGTH ステップ) を (チャネル=1ステップ分の観測次元, 時間) に
   並べ替えて 1-D CNN でエンコード
3. ``[command, 直近ステップ, CNN特徴]`` を連結して MLP に入力

する。観測の正規化 (EmpiricalNormalization) は rsl_rl 基底クラスの仕組みのまま
「command + 全履歴」全体に掛かるため、CNN も正規化済みの入力を受け取る。

rsl_rl の OnPolicyRunner はポリシークラスを ``eval(class_name)`` で
on_policy_runner モジュールの名前空間から解決するため、本モジュール末尾で
名前空間に注入している (本モジュールは agents/rsl_rl_ppo_cfg.py の import 時に
読み込まれるので、runner 生成前に必ず実行される)。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from rsl_rl.utils import resolve_nn_activation

from ..history_layout import (
    CRITIC_TERM_SPECS,
    HISTORY_LENGTH,
    LATEST_FRAME_GROUP,
    MLP_HISTORY_STEPS,
    POLICY_TERM_SPECS,
    term_specs_dim,
)

# 履歴バッファを持つ観測グループ名 → 項レイアウト
_HISTORY_GROUP_SPECS = {
    "policy": POLICY_TERM_SPECS,
    "critic": CRITIC_TERM_SPECS,
}

# 1-D CNN のデフォルト構成: 各層 [kernel size, filter size (out channels), stride]
CNN_KERNEL_SIZES = (8, 4)
CNN_FILTER_SIZES = (32, 16)
CNN_STRIDE_SIZES = (4, 2)


def _recent_step_indices(term_specs, history_length: int, num_steps: int) -> torch.Tensor:
    """flatten された履歴グループから「各項の直近 num_steps ステップ」を取り出す添字を作る。

    レイアウトは項ごとに (history_length × dim) のブロックが並び、ブロック内は
    古い → 新しい の順 (history_layout.py 参照)。
    """
    indices: list[int] = []
    offset = 0
    for _, dim, _ in term_specs:
        start = offset + (history_length - num_steps) * dim
        indices.extend(range(start, offset + history_length * dim))
        offset += history_length * dim
    return torch.tensor(indices, dtype=torch.long)


class HistoryEncoderHead(nn.Module):
    """「直接入力 (command 等) + 履歴」から行動平均 / 価値を出すネットワーク。

    入力は ``[直接入力 (direct_dim), 履歴 (history_length × step_dim)]`` の順で
    flatten されたベクトル (正規化済み)。
    """

    def __init__(
        self,
        direct_dim: int,
        term_specs,
        output_dim: int,
        hidden_dims,
        activation: str,
        history_length: int,
        mlp_history_steps: int,
        cnn_kernel_sizes,
        cnn_filter_sizes,
        cnn_stride_sizes,
    ):
        super().__init__()
        self.direct_dim = direct_dim
        self.history_length = history_length
        self.term_dims = tuple(dim for _, dim, _ in term_specs)
        step_dim = term_specs_dim(term_specs)

        # 履歴部分 (先頭 direct_dim を除いた部分) に対する直近ステップの添字
        self.register_buffer(
            "_recent_idx",
            _recent_step_indices(term_specs, history_length, mlp_history_steps),
            persistent=False,
        )

        # 1-D CNN: 入力 (B, step_dim, history_length)、時間軸に沿って畳み込む
        act_fn = resolve_nn_activation(activation)
        cnn_layers: list[nn.Module] = []
        in_channels = step_dim
        seq_len = history_length
        for kernel, filters, stride in zip(cnn_kernel_sizes, cnn_filter_sizes, cnn_stride_sizes):
            cnn_layers.append(nn.Conv1d(in_channels, filters, kernel_size=kernel, stride=stride))
            cnn_layers.append(act_fn)
            in_channels = filters
            seq_len = (seq_len - kernel) // stride + 1
            if seq_len < 1:
                raise ValueError(
                    f"CNN の出力系列長が {seq_len} になりました。kernel/stride が履歴長"
                    f" ({history_length}) に対して大きすぎます。"
                )
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_dim = in_channels * seq_len

        self.mlp = MLP(
            direct_dim + step_dim * mlp_history_steps + self.cnn_out_dim,
            output_dim,
            hidden_dims,
            activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direct = x[..., : self.direct_dim]
        history = x[..., self.direct_dim :]
        # 各項の直近ステップ分
        recent = history.index_select(-1, self._recent_idx)
        # 項ごとの (history_length × dim) ブロックを (B, history_length, dim) に戻して
        # チャネル方向に連結 → (B, step_dim, history_length)
        blocks = torch.split(history, [self.history_length * d for d in self.term_dims], dim=-1)
        frames = torch.cat(
            [block.reshape(-1, self.history_length, d) for block, d in zip(blocks, self.term_dims)],
            dim=-1,
        )
        features = self.cnn(frames.transpose(1, 2)).flatten(1)
        return self.mlp(torch.cat([direct, recent, features], dim=-1))


class HistoryActorCritic(ActorCritic):
    """履歴グループを CNN + 直近ステップスライスで処理する ActorCritic。"""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation: str = "elu",
        history_length: int = HISTORY_LENGTH,
        mlp_history_steps: int = MLP_HISTORY_STEPS,
        cnn_kernel_sizes=CNN_KERNEL_SIZES,
        cnn_filter_sizes=CNN_FILTER_SIZES,
        cnn_stride_sizes=CNN_STRIDE_SIZES,
        **kwargs,
    ):
        if mlp_history_steps > history_length:
            raise ValueError(
                f"mlp_history_steps ({mlp_history_steps}) は history_length ({history_length}) 以下にすること。"
            )
        # 観測グループの構成を検証する。各セットは「直接入力グループ (command 等) が先、
        # 履歴グループがちょうど 1 つで末尾」という並びを前提とする。
        head_args = {}
        for set_name in ("policy", "critic"):
            groups = obs_groups[set_name]
            history_groups = [g for g in groups if g in _HISTORY_GROUP_SPECS]
            if len(history_groups) != 1 or groups[-1] != history_groups[0]:
                raise ValueError(
                    f"obs_groups['{set_name}'] ({groups}) は履歴グループ"
                    f" ({list(_HISTORY_GROUP_SPECS)}) をちょうど 1 つ、末尾に置くこと。"
                )
            specs = _HISTORY_GROUP_SPECS[history_groups[0]]
            expected = term_specs_dim(specs) * history_length
            if obs[history_groups[0]].shape[-1] != expected:
                raise ValueError(
                    f"観測グループ '{history_groups[0]}' の次元 {obs[history_groups[0]].shape[-1]} が想定"
                    f" {expected} (= {term_specs_dim(specs)} × 履歴長 {history_length}) と一致しません。"
                    " history_layout.py の項リストを環境の観測定義に合わせて更新してください。"
                )
            head_args[set_name] = {
                "direct_dim": sum(obs[g].shape[-1] for g in groups[:-1]),
                "term_specs": specs,
            }

        # 基底クラスは get_actor_obs / get_critic_obs の連結次元 (command + 全履歴) で
        # 正規化器を作る。actor / critic の MLP は下で置き換える。
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            **kwargs,
        )

        common = {
            "history_length": history_length,
            "mlp_history_steps": mlp_history_steps,
            "cnn_kernel_sizes": cnn_kernel_sizes,
            "cnn_filter_sizes": cnn_filter_sizes,
            "cnn_stride_sizes": cnn_stride_sizes,
            "activation": activation,
        }
        self.actor = HistoryEncoderHead(
            output_dim=num_actions, hidden_dims=actor_hidden_dims, **head_args["policy"], **common
        )
        self.critic = HistoryEncoderHead(
            output_dim=1, hidden_dims=critic_hidden_dims, **head_args["critic"], **common
        )
        print(f"Actor CNN+MLP: {self.actor}")
        print(f"Critic CNN+MLP: {self.critic}")

        # mirror loss 用: symmetry 関数が返す「直近 1 ステップの生の policy 観測」
        # (LATEST_FRAME_GROUP, 1 ステップ分の次元) を履歴長ぶんタイル展開して
        # policy グループ相当のベクトルを作るための添字。
        tile_indices: list[int] = []
        frame_offset = 0
        for _, dim, _ in POLICY_TERM_SPECS:
            tile_indices.extend(list(range(frame_offset, frame_offset + dim)) * history_length)
            frame_offset += dim
        self.register_buffer(
            "_policy_latest_tile_idx", torch.tensor(tile_indices, dtype=torch.long), persistent=False
        )

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            if obs_group == "policy" and obs_group not in obs.keys() and LATEST_FRAME_GROUP in obs.keys():
                # 直近フレームのみのコンパクト観測 (mirror loss 用)。履歴を最新フレームで
                # 埋めた形 (エピソード開始直後の CircularBuffer バックフィルと同じ) に展開する。
                obs_list.append(obs[LATEST_FRAME_GROUP].index_select(-1, self._policy_latest_tile_idx))
            else:
                obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)


@configclass
class RslRlHistoryActorCriticCfg(RslRlPpoActorCriticCfg):
    """HistoryActorCritic 用の設定。"""

    class_name: str = "HistoryActorCritic"

    history_length: int = HISTORY_LENGTH
    """観測履歴バッファの長さ (環境側の history_length と一致させること)。"""

    mlp_history_steps: int = MLP_HISTORY_STEPS
    """MLP に直接入力する直近ステップ数。"""

    cnn_kernel_sizes: tuple = CNN_KERNEL_SIZES
    """履歴エンコーダ CNN の各層の kernel size。"""

    cnn_filter_sizes: tuple = CNN_FILTER_SIZES
    """履歴エンコーダ CNN の各層の filter 数 (out channels)。"""

    cnn_stride_sizes: tuple = CNN_STRIDE_SIZES
    """履歴エンコーダ CNN の各層の stride。"""


# OnPolicyRunner は eval(class_name) を on_policy_runner モジュールの名前空間で
# 評価するため、そこにクラスを注入する。
import rsl_rl.runners.on_policy_runner as _rsl_rl_on_policy_runner  # noqa: E402

_rsl_rl_on_policy_runner.HistoryActorCritic = HistoryActorCritic
