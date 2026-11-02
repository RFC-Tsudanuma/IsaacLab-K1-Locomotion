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
