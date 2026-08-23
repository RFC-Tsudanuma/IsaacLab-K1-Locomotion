# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LateralHistoryActorCritic (履歴 + CNN 構成) 用の JIT / ONNX エクスポータ。

★ 2026-08-23: feat/inoue_walk_double_encoder の history_policy_exporter.py を移植。

isaaclab_rl 標準の exporter (``export_policy_as_onnx`` 等) は「actor が単純な MLP
(nn.Sequential) で入力が 1 本の観測ベクトル」を前提にしており
(``actor[0].in_features`` 参照・``torch.jit.script``)、HistoryEncoderHead では動かない。
本モジュールはデプロイしやすいフレーム形式のインターフェースで書き出す。

エクスポートされるモデルの入出力:

* 入力 ``direct``:      (N, 13) 履歴なしの直接入力 = velocity_commands(3) [vx, vy, ωz]
  + GK タスクスロット(10, Stage1 では全て 0)
* 入力 ``obs_history``: (N, HISTORY_LENGTH, STEP_DIM)  コマンドを除いた観測の履歴。
  時間軸 (dim=1) は 古い → 新しい。チャネル (dim=2) は history_layout.POLICY_TERM_SPECS
  の順で、K1 Flat では
  [base_ang_vel(3), projected_gravity(3), velocity_commands(3), joint_pos(12),
   joint_vel(12), last_action(12), gait_phase(4)] の 49 次元。
  起動直後など履歴が足りない間は、環境の CircularBuffer と同様に「最新フレームで
  全履歴を埋める」形でバッファを初期化すること。
* 出力 ``actions``:     (N, 12)  生のアクション (スケール・オフセット未適用)

観測の正規化 (EmpiricalNormalization) はモデル内部に焼き込み済みなので、
生の観測をそのまま入力してよい。
"""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn

from .history_layout import DIRECT_DIM, HISTORY_LENGTH, POLICY_TERM_SPECS, term_specs_dim

__all__ = ["is_history_policy", "export_history_policy_as_jit", "export_history_policy_as_onnx"]


def is_history_policy(policy: object) -> bool:
    """policy が LateralHistoryActorCritic (履歴 + CNN 構成) かどうかを返す。"""
    from .networks import LateralHistoryActorCritic

    return isinstance(policy, LateralHistoryActorCritic)


class _HistoryPolicyExporter(nn.Module):
    """(direct, obs_history) を受け取り actor を評価するラッパ。

    obs_history (N, H, C) を history_layout の「項ごとの (H × dim) ブロック」flatten
    レイアウトへ並べ替え、direct と連結して正規化器 + actor に通す。
    """

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(policy.actor)
        if normalizer is not None:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = nn.Identity()
        self.history_length = HISTORY_LENGTH
        self.term_dims = tuple(dim for _, dim, _ in POLICY_TERM_SPECS)
        self.step_dim = term_specs_dim(POLICY_TERM_SPECS)

    def forward(self, direct: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        # (N, H, C) → 項ごとに (N, H, dim) を切り出して (N, H*dim) に flatten (ステップ優先)
        parts = [direct]
        channel = 0
        for dim in self.term_dims:
            block = obs_history[:, :, channel : channel + dim]
            parts.append(block.reshape(obs_history.shape[0], -1))
            channel += dim
        x = torch.cat(parts, dim=-1)
        return self.actor(self.normalizer(x))

    def _example_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        direct = torch.zeros(1, DIRECT_DIM)
        obs_history = torch.zeros(1, self.history_length, self.step_dim)
        return direct, obs_history


def export_history_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """履歴方策を TorchScript (trace) で書き出す。"""
    os.makedirs(path, exist_ok=True)
    exporter = _HistoryPolicyExporter(policy, normalizer)
    exporter.to("cpu")
    exporter.eval()
    with torch.no_grad():
        traced = torch.jit.trace(exporter, exporter._example_inputs())
    traced.save(os.path.join(path, filename))


def export_history_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """履歴方策を ONNX で書き出す。"""
    os.makedirs(path, exist_ok=True)
    exporter = _HistoryPolicyExporter(policy, normalizer)
    exporter.to("cpu")
    exporter.eval()
    torch.onnx.export(
        exporter,
        exporter._example_inputs(),
        os.path.join(path, filename),
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=["direct", "obs_history"],
        output_names=["actions"],
        dynamic_axes={},
    )
