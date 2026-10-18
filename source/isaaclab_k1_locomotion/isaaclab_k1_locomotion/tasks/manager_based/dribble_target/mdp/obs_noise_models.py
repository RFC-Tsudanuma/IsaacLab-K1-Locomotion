# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""実機センサの「履歴の質」の劣化を模擬するノイズモデル。

背景: dual encoder (100 ステップ履歴 + CNN) ポリシーは、観測フレーム列の
時間的な質 (位相遅れ・重複・定常オフセット) に敏感である。実機 (dual_first)
では後退歩行で転倒したが、クリーンな観測の mujoco sim2sim では再現しなかった。
毎ステップ独立な白色ノイズ (従来の Unoise) はこの種の「構造化された」劣化を
カバーしないため、以下のアーティファクトを per-env ランダム化して学習時に注入し、
履歴の質の変化に頑健なポリシーを得る。

:class:`SensorArtifactNoiseModel` は 1 つの観測項に対して次を順に適用する:

0. **センサ遅延** (per-env で delay_range [秒] から抽選、リセット時に再抽選)
   — 通信・ドライバ由来の伝送遅延相当。遅延 d = delay / step_dt [ステップ] を
   整数部 i と小数部 β に分け、過去の生値の線形補間で近似する:
   x_delayed = (1-β)·x_{t-i} + β·x_{t-i-1}。
   50Hz (step_dt=0.02) では delay_range (0, 0.010) → i=0, β ∈ [0, 0.5]。
   1 ステップを超える遅延 (例: 自己位置推定の 0〜0.1 秒) も扱える (2026-10-18 拡張。
   1 ステップ以下の遅延では拡張前と同じ値になる)。
1. **定数バイアス** (成分ごと、エピソードリセット時に ±bias_range で再抽選)
   — IMU 取付誤差・姿勢推定器バイアス・エンコーダオフセット相当。
2. **EMA ローパスフィルタ** (係数 α を per-env で filter_alpha_range から抽選)
   — 実機センサの内蔵 LPF やデプロイ側のアンチエイリアス移動平均による
   位相遅れ・なまり相当。y_t = α·y_{t-1} + (1-α)·x_t。
   50Hz では α=0.6 でカットオフ ≒ 5Hz。
3. **白色ノイズ** (noise_cfg、従来の Unoise と同じ毎ステップ独立ノイズ)。
4. **フレームホールド** (確率 p を per-env で hold_prob_range から抽選し、
   各ステップ確率 p で「前ステップに出力した値」をそのまま再出力する。
   項の全成分が同時にホールドされる) — 受信タイミングのずれによる重複・
   stale フレーム相当。ホールドは白色ノイズ適用後の出力を複製するので、
   実機の「同じ値が 2 回サンプリングされる」現象と一致する。

ノイズは ObservationManager が履歴バッファへ push する前に適用されるため、
これらのアーティファクトはそのまま履歴に固定される (実機と同じ)。
critic グループは enable_corruption=False なので影響を受けない。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg, NoiseModel, NoiseModelCfg


class SensorArtifactNoiseModel(NoiseModel):
    """バイアス + EMA フィルタ + 白色ノイズ + フレームホールドのノイズモデル。"""

    def __init__(self, noise_model_cfg: SensorArtifactNoiseCfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self._cfg = noise_model_cfg
        # per-env パラメータ (成分数 D が判明する初回 __call__ まで bias は未確保)
        self._alpha = torch.zeros((num_envs, 1), device=device)
        self._hold_prob = torch.zeros((num_envs, 1), device=device)
        self._delay_frac = torch.zeros((num_envs, 1), device=device)
        self._bias: torch.Tensor | None = None
        # 時系列状態
        self._ema: torch.Tensor | None = None
        self._prev_out: torch.Tensor | None = None
        # 過去の生値のリングバッファ (index 0 が最新)。遅延の上限ステップ数 + 2 だけ持つ。
        self._max_delay_steps = max(self._cfg.delay_range[1], 0.0) / max(self._cfg.step_dt, 1e-6)
        self._raw_hist_len = int(torch.ceil(torch.tensor(self._max_delay_steps)).item()) + 2
        self._raw_hist: torch.Tensor | None = None
        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._sample_env_params(slice(None))

    def _sample_env_params(self, env_ids):
        lo, hi = self._cfg.filter_alpha_range
        self._alpha[env_ids] = torch.empty_like(self._alpha[env_ids]).uniform_(lo, hi)
        lo, hi = self._cfg.hold_prob_range
        self._hold_prob[env_ids] = torch.empty_like(self._hold_prob[env_ids]).uniform_(lo, hi)
        # 遅延 [秒] → ステップ数 d = delay / step_dt (整数部 + 小数部、バッファ長で clamp)
        lo, hi = self._cfg.delay_range
        delays = torch.empty_like(self._delay_frac[env_ids]).uniform_(lo, hi)
        self._delay_frac[env_ids] = (delays / max(self._cfg.step_dt, 1e-6)).clamp(0.0, self._max_delay_steps)
        if self._bias is not None:
            r = self._cfg.bias_range
            self._bias[env_ids] = torch.empty_like(self._bias[env_ids]).uniform_(-r, r)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._sample_env_params(env_ids)
        # EMA / prev の状態はリセット後の初回観測で初期化し直す
        self._initialized[env_ids] = False

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self._bias is None:
            # 初回呼び出しで成分数が判明したら bias を確保して抽選する
            num_components = data.shape[-1]
            r = self._cfg.bias_range
            self._bias = torch.empty(
                (self._num_envs, num_components), device=self._device, dtype=data.dtype
            ).uniform_(-r, r)
            self._ema = torch.zeros_like(self._bias)
            self._prev_out = torch.zeros_like(self._bias)
            self._raw_hist = torch.zeros(
                (self._num_envs, self._raw_hist_len, num_components), device=self._device, dtype=data.dtype
            )

        # 生値の履歴を 1 つずらして最新値を先頭に入れる。リセット直後の env は履歴全体を
        # 現在値で埋める (フィルタ/遅延の過渡を作らない)
        fresh = ~self._initialized
        self._raw_hist = torch.roll(self._raw_hist, shifts=1, dims=1)
        self._raw_hist[:, 0] = data
        if fresh.any():
            self._raw_hist[fresh] = data[fresh].unsqueeze(1)

        # センサ遅延: d = i + β ステップ前の値を、i / i+1 ステップ前の生値の線形補間で近似
        steps = self._delay_frac.squeeze(-1)
        i0 = torch.floor(steps).long().clamp(max=self._raw_hist_len - 2)
        beta = (steps - i0.float()).unsqueeze(-1)
        env_idx = torch.arange(self._num_envs, device=self._device)
        delayed = (1.0 - beta) * self._raw_hist[env_idx, i0] + beta * self._raw_hist[env_idx, i0 + 1]

        x = delayed + self._bias
        if fresh.any():
            self._ema[fresh] = x[fresh]
        self._ema = self._alpha * self._ema + (1.0 - self._alpha) * x
        # 白色ノイズ (従来の Unoise 相当)
        out = self._cfg.noise_cfg.func(self._ema, self._cfg.noise_cfg)
        # フレームホールド: 確率 p で前回出力をそのまま再出力 (項の全成分同時)
        hold = (torch.rand((self._num_envs, 1), device=self._device) < self._hold_prob) & (
            self._initialized.unsqueeze(-1)
        )
        out = torch.where(hold, self._prev_out, out)
        # ObservationManager は返り値に clip_/mul_ を in-place 適用しうるので、
        # 次ステップのホールド用コピーは clone して切り離す
        self._prev_out = out.clone()
        self._initialized[:] = True
        return out


@configclass
class SensorArtifactNoiseCfg(NoiseModelCfg):
    """SensorArtifactNoiseModel の設定。"""

    class_type: type = SensorArtifactNoiseModel

    noise_cfg: NoiseCfg = MISSING
    """毎ステップの白色ノイズ (従来の Unoise をそのまま指定する)。"""

    bias_range: float = 0.0
    """定数バイアスの振幅。成分ごとに ±bias_range の一様分布からリセット時に再抽選。"""

    filter_alpha_range: tuple[float, float] = (0.0, 0.0)
    """EMA フィルタ係数 α の per-env 抽選範囲。0 でフィルタなし。"""

    hold_prob_range: tuple[float, float] = (0.0, 0.0)
    """フレームホールド確率の per-env 抽選範囲。"""

    delay_range: tuple[float, float] = (0.0, 0.0)
    """センサ遅延 [秒] の per-env 抽選範囲 (リセット時に再抽選)。
    過去の生値の線形補間で分数ステップの遅延を近似する。(0, 0) で遅延なし。"""

    step_dt: float = 0.02
    """制御周期 [秒]。遅延の分数ステップ換算 β = delay / step_dt に使う (50Hz → 0.02)。"""
