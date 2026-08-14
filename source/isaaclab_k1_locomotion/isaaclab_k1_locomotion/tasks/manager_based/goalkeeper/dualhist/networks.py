# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""デュアルヒストリー方策のネットワーク (arXiv:2401.16889 の構造をこのタスク向けに移植)。

観測ベクトルの並び (:mod:`.env_cfg` が作る policy グループ)::

    [ 0 : 59 ]                    既存の gk 観測 (歩行 49 + タスク 10)
    [ 59 : 59+Ts*7 ]              短期履歴 (既定 5 frame = 0.1s @50Hz)
    [ 59+Ts*7 : +Tl*7 ]           長期履歴 (既定 50 frame = 1.0s @50Hz)

処理:

    * 既存観測 + 短期履歴 → **そのまま** MLP へ (論文の short I/O history と同じ扱い)
    * 長期履歴 → ``(N, 7, Tl)`` に直して 1D CNN で圧縮 → latent を MLP に concat

CNN のカーネル構成は論文と同じ (k6/c32/s3 → k4/c16/s2)。論文は 66 frame → latent 16×?
だったが、こちらは frame 数が違うので出力長は実際の畳み込み計算から求める。

critic には履歴を入れていない (特権情報の真値を単一フレームで持っているため) ので、
critic 側は素の :class:`~rsl_rl.modules.ActorCritic` の MLP のまま。

TorchScript 互換について:
    ``play_goalkeeper.py`` は学習後に ``export_policy_as_jit`` / ``export_policy_as_onnx``
    を呼び、``policy.actor`` を deepcopy して ``torch.jit.script`` する。そのため
    :class:`DualHistoryActor` の ``forward`` は script 可能な書き方に限定してある
    (int 属性 + narrow/reshape/transpose のみ)。実機デプロイもこの経路。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from types import SimpleNamespace

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from rsl_rl.utils import resolve_nn_activation


def _conv_out_len(length: int, kernels, strides) -> int:
    """padding 無し 1D 畳み込みを重ねたあとの系列長。"""
    for k, s in zip(kernels, strides):
        if length < k:
            raise ValueError(
                f"長期履歴が短すぎます: 入力長 {length} < カーネル {k}。"
                " hist_long_frames を増やすか kernel/stride を小さくしてください。"
            )
        length = (length - k) // s + 1
    return length


class DualHistoryActor(nn.Module):
    """[既存観測 + 短期履歴] と [CNN 圧縮した長期履歴] を結合して行動を出す actor。"""

    def __init__(
        self,
        num_direct: int,
        frame_dim: int,
        long_frames: int,
        num_actions: int,
        hidden_dims,
        activation: str,
        channels,
        kernels,
        strides,
    ) -> None:
        super().__init__()
        self.num_direct = int(num_direct)          # 既存観測 + 短期履歴 (そのまま通す分)
        self.frame_dim = int(frame_dim)
        self.long_frames = int(long_frames)
        self.num_long = int(long_frames) * int(frame_dim)

        layers: list[nn.Module] = []
        in_ch = int(frame_dim)
        for out_ch, k, s in zip(channels, kernels, strides):
            layers.append(nn.Conv1d(in_ch, int(out_ch), kernel_size=int(k), stride=int(s)))
            layers.append(resolve_nn_activation(activation))
            in_ch = int(out_ch)
        self.encoder = nn.Sequential(*layers)

        out_len = _conv_out_len(self.long_frames, kernels, strides)
        self.latent_dim = int(in_ch) * int(out_len)
        self.mlp = MLP(self.num_direct + self.latent_dim, num_actions, hidden_dims, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direct = x.narrow(1, 0, self.num_direct)
        long_flat = x.narrow(1, self.num_direct, self.num_long)
        # (N, T*D) -> (N, T, D) -> (N, D, T)。並びは observations.gk_io_history と対で決まる。
        seq = long_flat.reshape(-1, self.long_frames, self.frame_dim).transpose(1, 2)
        latent = self.encoder(seq).flatten(1)
        return self.mlp(torch.cat([direct, latent], dim=1))

    def __getitem__(self, index: int):
        """``actor[0].in_features`` 互換 (ONNX エクスポータ用)。

        isaaclab_rl の ``_OnnxPolicyExporter.export`` は actor が nn.Sequential である前提で::

            obs = torch.zeros(1, self.actor[0].in_features)

        としてダミー入力を作る。この actor は Sequential ではないので、入力次元だけを
        答えられる最小限の互換を用意する。TorchScript は forward から到達しないメソッドを
        コンパイルしないので、jit export 側には影響しない。
        """
        if index != 0:
            raise IndexError(
                "DualHistoryActor が提供するのは actor[0].in_features 互換のみです。"
            )
        return SimpleNamespace(in_features=self.num_direct + self.num_long)


class ActorCriticDualHistory(ActorCritic):
    """actor だけデュアルヒストリー構造に差し替えた ActorCritic。

    rsl-rl の ``OnPolicyRunner`` は ``policy.class_name`` を ``eval()`` で解決するので、
    このモジュールの import 時に ``rsl_rl.runners.on_policy_runner`` の名前空間へ
    クラスを注入する (末尾の :func:`register_with_rsl_rl`)。

    Args:
        hist_frame_dim: 履歴 1 フレームの次元 (:data:`.observations.GK_HIST_FRAME_DIM`)。
        hist_short_frames: 短期履歴のフレーム数。生のまま MLP へ入る。
        hist_long_frames: 長期履歴のフレーム数。CNN で圧縮される。
        hist_long_channels / hist_long_kernels / hist_long_strides: CNN の構成。
            既定は論文と同じ (32ch k6 s3 → 16ch k4 s2)。

    ★ ``actor_obs_normalization`` は履歴ブロックにも**次元ごとに**掛かる。同じ信号の
      異なるフレームが別々の統計で正規化されることになるが、定常な信号なので統計は
      ほぼ同じ値に収束する (実害は無いと判断してそのままにしてある)。
    """

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        hist_frame_dim: int = 7,
        hist_short_frames: int = 5,
        hist_long_frames: int = 50,
        hist_long_channels=(32, 16),
        hist_long_kernels=(6, 4),
        hist_long_strides=(3, 2),
        **kwargs,
    ) -> None:
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        if kwargs.get("state_dependent_std", False):
            raise NotImplementedError(
                "ActorCriticDualHistory は state_dependent_std に対応していません。"
            )

        num_actor_obs = 0
        for group in self.obs_groups["policy"]:
            num_actor_obs += obs[group].shape[-1]

        num_hist = (int(hist_short_frames) + int(hist_long_frames)) * int(hist_frame_dim)
        num_direct = num_actor_obs - int(hist_long_frames) * int(hist_frame_dim)
        num_base = num_actor_obs - num_hist
        if num_base <= 0:
            raise ValueError(
                f"actor 観測 {num_actor_obs} 次元に対して履歴が {num_hist} 次元あり、"
                " 既存観測の分が残りません。hist_* の設定と観測グループの定義"
                " (dualhist/env_cfg.py) が食い違っています。"
            )

        # super() が作った素の MLP actor を差し替える (critic はそのまま使う)。
        self.actor = DualHistoryActor(
            num_direct=num_direct,
            frame_dim=int(hist_frame_dim),
            long_frames=int(hist_long_frames),
            num_actions=num_actions,
            hidden_dims=kwargs.get("actor_hidden_dims", [256, 256, 256]),
            activation=kwargs.get("activation", "elu"),
            channels=hist_long_channels,
            kernels=hist_long_kernels,
            strides=hist_long_strides,
        )
        print(
            f"Actor (dual history): base {num_base} + short {int(hist_short_frames)}x{int(hist_frame_dim)}"
            f" = {num_direct} direct, long {int(hist_long_frames)}x{int(hist_frame_dim)}"
            f" -> latent {self.actor.latent_dim}\n{self.actor}"
        )


def register_with_rsl_rl() -> None:
    """``OnPolicyRunner`` の ``eval(class_name)`` から見えるように名前を注入する。

    rsl-rl 側 (on_policy_runner.py) は::

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))

    としており、``eval`` は呼び出し元モジュールの globals を参照する。``rsl_rl.modules``
    に登録しても見えないので、モジュールの属性として直接差し込む必要がある。
    """
    import rsl_rl.runners.on_policy_runner as _opr

    setattr(_opr, "ActorCriticDualHistory", ActorCriticDualHistory)


register_with_rsl_rl()
