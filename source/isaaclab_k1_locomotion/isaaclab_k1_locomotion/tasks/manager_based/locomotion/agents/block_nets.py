# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール阻止方策のネットワーク部品 (causal transformer encoder + head)。

torch と rsl_rl.networks.MLP だけに依存し、Isaac Sim / isaaclab を import しない。
scripts/rsl_rl/export_block_policy.py がこのファイルをパス指定で読み込んで ONNX 化するため、
パッケージ相対 import を置かないこと。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.networks import MLP


class CausalTransformerEncoder(nn.Module):
    """(B, T, in_dim) → (B, T, latent_dim)。時刻 t の出力は t 以前のフレームだけを見る。"""

    def __init__(self, in_dim: int, seq_len: int, d_model: int, nhead: int, num_layers: int, latent_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2 * d_model, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, latent_dim)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(seq_len), persistent=False)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(frames) + self.pos
        return self.out(self.encoder(h, mask=self.mask, is_causal=True))


class BlockHead(nn.Module):
    """正規化済み ``[内界 (direct_dim), ボール履歴 (T × ball_dim)]`` → encoder → ``[内界, latent_T]`` → MLP。"""

    def __init__(self, direct_dim: int, seq_len: int, encoder: CausalTransformerEncoder, output_dim: int, hidden_dims, activation):
        super().__init__()
        self.direct_dim = direct_dim
        self.seq_len = seq_len
        self.encoder = encoder
        self.mlp = MLP(direct_dim + encoder.out.out_features, output_dim, hidden_dims, activation)

    def latents(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x[..., self.direct_dim :].reshape(-1, self.seq_len, self.encoder.in_proj.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x[..., : self.direct_dim], self.latents(x)[:, -1]], dim=-1))


def steal_mirror(hist: int = 30) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """奪取方策 (StealHist) の左右反転 (2026-11-24)。actor 入力 (69 + hist × 8) と出力 (コマンド 3 + 残差 12) について
    ``mirror(v) = v[..., idx] * sign`` となる (入力の idx, sign, 出力の idx, sign) を返す。規約は mdp/symmetry.py と同じ。"""
    jp, js = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], [1.0, -1.0, -1.0, 1.0, 1.0, -1.0] * 2
    idx, sign = [], []

    def add(perm, sg):
        base = len(idx)
        idx.extend(base + p for p in perm)
        sign.extend(sg)

    add(range(3), [-1.0, 1.0, -1.0])  # base_ang_vel
    add(range(3), [1.0, -1.0, 1.0])  # projected_gravity
    add(range(3), [1.0, -1.0, -1.0])  # 歩行コマンド
    for _ in range(3):  # joint_pos, joint_vel, 前回の関節指令
        add(jp, js)
    add([2, 3, 0, 1], [1.0] * 4)  # 歩行位相 [sin L, cos L, sin R, cos R]
    add(jp, js)  # 前回の残差
    for _ in range(4):  # ボール位置, ボール速度, 敵位置, 目標方向
        add(range(2), [1.0, -1.0])
    for _ in range(hist):  # [ボール位置, ボール速度, 敵位置, 見えている, 仮想]
        add(range(8), [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
    m = torch.tensor(idx), torch.tensor(sign), torch.tensor([0, 1, 2] + [3 + p for p in jp]), torch.tensor([1.0, -1.0, -1.0] + js)
    for i, s in (m[:2], m[2:]):  # 2 回反転すると元に戻る
        assert torch.equal(i[i], torch.arange(len(i))) and torch.all(s[i] * s == 1.0)
    return m


def steal_target_left(x: torch.Tensor) -> torch.Tensor:
    """目標方向がロボット→ボールの線の左にあるか (N,) (actor 入力のボール相対位置 × 目標方向)。"""
    return x[:, 61] * x[:, 68] - x[:, 62] * x[:, 67] > 0.0


def steal_symmetric(actor, x: torch.Tensor, m, left: torch.Tensor | None = None) -> torch.Tensor:
    """左右対称にした奪取方策の出力 (N, 15)。学習した方策は「左足で右へ払う」蹴りしか持たず、目標方向が左のときは
    ボールの向こう側へ回り込む。目標方向がロボット→ボールの線の左にあるときは、観測を左右反転して方策に通し、
    出力を反転して戻す (右足で左へ払う)。``actor`` は正規化込みの関数、``m`` は `steal_mirror` の戻り値 (x と同じ device)。``left`` を渡すとその判定を使う (起動時に固定する場合)。"""
    idx, sign, oidx, osign = m
    left = steal_target_left(x) if left is None else left
    return torch.where(left.unsqueeze(1), actor(x[:, idx] * sign)[:, oidx] * osign, actor(x))
