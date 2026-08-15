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
from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation

# 観測正規化の std がこれ未満の次元は「そのステージで一度も動かなかった列」とみなす。
# :meth:`ActorCriticDualHistory._sanitize_obs_normalizer` 参照。
_DEGENERATE_STD = 1e-3


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

    # -- 観測正規化の縮退対策 ------------------------------------------------

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """重みを読み込み、続けて観測正規化の縮退列を無害化する。

        ``OnPolicyRunner.load`` (= ``--resume``) から呼ばれる。Stage 1 → Stage 2 の
        受け渡しでここが効く。
        """
        resumed = super().load_state_dict(state_dict, strict=strict)
        self._sanitize_obs_normalizer("actor", self.actor_obs_normalizer)
        self._sanitize_obs_normalizer("critic", self.critic_obs_normalizer)
        return resumed

    @staticmethod
    def _sanitize_obs_normalizer(tag: str, norm) -> None:
        """Stage 1 で一度も動かなかった観測列の正規化統計をリセットする。

        ★ これが無いと Stage 2 が壊れる (2026-08-15 に実際に起きた)。仕組み:

        Stage 1 はボールを検出範囲外 (park_pos = 9m) に置くので、ボール由来の観測列は
        **全ステップ厳密に 0**。:class:`~rsl_rl.networks.EmpiricalNormalization` は
        そこから ``std = 0`` を学習する。正規化は::

            (x - mean) / (std + eps)      # eps = 1e-2

        なので、Stage 2 でボールが見え始めた瞬間、その列は **最大 100 倍** に増幅されて
        ネットワークへ入る (フィールド座標のボール x は最大 9m → 900)。

        しかも自力では戻らない。EmpiricalNormalization の更新率は
        ``rate = batch / 累積count`` で、Stage 1 を 5000 iter 回した時点で
        count ≈ 10 億、rate ≈ 1e-4。分散が実値に追いつくまで **1 万 iter 級**かかる。
        その間ずっと入力スケールが 2 桁ずれたままなので、方策は壊れ、PPO の適応学習率は
        KL を抑えようと潰れ、セーブ成功率が上がらず**適応カリキュラムが 1 段も進まない**。

        直接制御版・既存階層版でこれが表面化しなかったのは、park_pos が 5m だった頃の
        Stage 1 ckpt を使っていて、ボール列に実分散が入っていたため
        (``goalkeeper_hier_env_cfg.py`` の park_pos 変更は 2026-08-14、DH 版の Stage 1 が
        その後に回した最初の学習だった)。デュアルヒストリー版は履歴のボール 3ch × 55 frame
        = 165 列がまとめて縮退するので、同じ地雷を桁違いに強く踏む。

        対処: **縮退した列だけ** mean=0 / var=1 / std=1 に戻す。健全な列 (歩行 49 次元など)
        の学習済み統計はそのまま残すこと。全体をリセットすると、joint_pos のように
        std=0.03 で 33 倍に増幅されて学習された列のスケールまで変わり、同じ理由で壊れる。
        """
        if not isinstance(norm, EmpiricalNormalization):
            return
        bad = (norm._std.squeeze(0) < _DEGENERATE_STD)
        n_bad = int(bad.sum().item())
        if n_bad == 0:
            return
        idx = torch.nonzero(bad).flatten().tolist()
        norm._mean[0, bad] = 0.0
        norm._var[0, bad] = 1.0
        norm._std[0, bad] = 1.0
        preview = idx[:12] + (["..."] if len(idx) > 12 else [])
        print(
            f"[dualhist] {tag} 観測正規化: 縮退した {n_bad} 列を mean=0/var=1 にリセットしました"
            f" (Stage 1 で一度も動かなかった列)。列番号: {preview}"
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
