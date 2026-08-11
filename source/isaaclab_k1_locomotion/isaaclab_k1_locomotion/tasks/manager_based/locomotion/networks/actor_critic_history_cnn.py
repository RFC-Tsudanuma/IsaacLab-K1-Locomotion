# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""観測履歴を 1D-CNN で符号化する ActorCritic (rsl_rl 用)。

Actor の入力を「1 フレームの観測」から「50 フレーム分の観測バッファ」に差し替える::

    obs_history: (N, H=50, D)     ← 古い順。obs_history[:, -1] が現在フレーム
        ├─ 直近 K=5 フレームをそのまま平坦化      → 5 * D
        └─ H=50 フレーム全部を 1D-CNN で符号化    → latent
                                                    ↓ concat
                                             actor MLP → action

CNN は 1 次元・隠れ層 2 つで、[kernel, filter, stride] = [6, 32, 3], [4, 16, 2]。
チャンネルが観測次元 D、系列長が履歴長 H になるので (N, D, H) に転置してから通す::

    L0 = 50 → conv1 (k=6, s=3) → L1 = (50-6)//3 + 1 = 15
            → conv2 (k=4, s=2) → L2 = (15-4)//2 + 1 = 6
    latent = 16 * 6 = 96

D = 55 (walk_kick 系の policy 観測) なら actor MLP の入力は 5*55 + 96 = 371 次元。

環境側の要求
------------
policy 観測グループに履歴を持たせ、**平坦化しない**こと::

    self.observations.policy.history_length = 50
    self.observations.policy.flatten_history_dim = False

``flatten_history_dim = True`` にすると ObservationManager は項ごとに (H, d_i) を
平坦化してから連結するので、並びが「項ごとのフレーム列」になってしまい
(N, H, D) には戻せない。ここでは 3 次元のまま受け取り、2 次元が来たら弾く。

Critic には履歴を付けていない (特権情報付きの 1 フレーム観測のまま)。critic 側の
グループにも ``history_length`` を設定すると 3 次元になって MLP に入らないので、
その場合はこのクラスの critic 側も拡張する必要がある。

実機デプロイ時の入力の作り方
----------------------------
55 次元の観測を毎制御周期リングバッファに積み、古い順に並べた (1, 50, 55) を
ONNX の "obs" に渡す。エピソード開始直後はバッファを現在フレームで埋める
(IsaacLab の CircularBuffer が reset 後に同じことをする)。
"""

from __future__ import annotations

import contextlib
import copy
import io
import os
import torch
import torch.nn as nn

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from rsl_rl.utils import resolve_nn_activation


@configclass
class RslRlPpoActorCriticHistoryCnnCfg(RslRlPpoActorCriticCfg):
    """:class:`ActorCriticHistoryCNN` 用の policy 設定。

    履歴長 H は環境の観測形状から自動で読むので、ここには持たせない
    (``observations.policy.history_length`` が唯一の情報源)。
    """

    class_name: str = "ActorCriticHistoryCNN"

    num_recent_frames: int = 5
    """CNN を通さずそのまま actor に入れる直近フレーム数 K。"""

    cnn_kernel_sizes: list[int] = [6, 4]
    """各 conv 層の kernel size。"""

    cnn_filters: list[int] = [32, 16]
    """各 conv 層の出力チャンネル数 (filter size)。"""

    cnn_strides: list[int] = [3, 2]
    """各 conv 層の stride。"""


class ObsHistoryEncoder(nn.Module):
    """(N, H, D) の観測履歴を「直近 K フレーム + CNN 潜在」に符号化する。"""

    def __init__(
        self,
        obs_dim: int,
        history_length: int,
        num_recent_frames: int,
        kernel_sizes: list[int],
        filters: list[int],
        strides: list[int],
        activation: str = "elu",
    ):
        super().__init__()

        if not len(kernel_sizes) == len(filters) == len(strides):
            raise ValueError(
                "cnn_kernel_sizes / cnn_filters / cnn_strides は同じ長さにすること。"
                f" 受け取った長さ: {len(kernel_sizes)}, {len(filters)}, {len(strides)}"
            )
        if not 1 <= num_recent_frames <= history_length:
            raise ValueError(
                f"num_recent_frames は 1 以上 history_length ({history_length}) 以下にすること。"
                f" 受け取った値: {num_recent_frames}"
            )

        self.obs_dim = obs_dim
        self.history_length = history_length
        self.num_recent_frames = num_recent_frames
        # TorchScript で負のスライスを避けるため、開始インデックスを持っておく
        self.recent_start = history_length - num_recent_frames

        layers: list[nn.Module] = []
        in_channels = obs_dim
        length = history_length
        for kernel, out_channels, stride in zip(kernel_sizes, filters, strides):
            if length < kernel:
                raise ValueError(
                    f"conv 層の kernel size {kernel} が入力系列長 {length} を超えている。"
                    " history_length を伸ばすか kernel/stride を小さくすること。"
                )
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride))
            layers.append(resolve_nn_activation(activation))
            length = (length - kernel) // stride + 1
            in_channels = out_channels
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        self.latent_dim = in_channels * length
        self.recent_dim = num_recent_frames * obs_dim
        self.output_dim = self.recent_dim + self.latent_dim

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        # 直近 K フレーム（古い順のまま平坦化）
        recent = obs_history[:, self.recent_start :, :].flatten(1)
        # 全 H フレームを CNN へ。(N, H, D) -> (N, D, H) でチャンネル = 観測次元
        latent = self.cnn(obs_history.transpose(1, 2))
        return torch.cat((recent, latent), dim=-1)


class HistoryCnnActor(nn.Module):
    """観測履歴 (N, H, D) を受けて action を返す actor 本体。"""

    def __init__(
        self,
        obs_dim: int,
        history_length: int,
        output_dim: int | list[int],
        hidden_dims: list[int],
        activation: str,
        num_recent_frames: int,
        kernel_sizes: list[int],
        filters: list[int],
        strides: list[int],
    ):
        super().__init__()
        self.encoder = ObsHistoryEncoder(
            obs_dim=obs_dim,
            history_length=history_length,
            num_recent_frames=num_recent_frames,
            kernel_sizes=kernel_sizes,
            filters=filters,
            strides=strides,
            activation=activation,
        )
        # output_dim は素の num_actions か、state_dependent_std のときの
        # [2, num_actions] (平均と標準偏差)。親の ActorCritic が actor に期待する形を
        # そのまま渡す。
        self.mlp = MLP(self.encoder.output_dim, output_dim, hidden_dims, activation)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(obs_history))


class ActorCriticHistoryCNN(ActorCritic):
    """Actor が観測履歴を、critic が 1 フレームの特権観測を見る ActorCritic。

    分布まわり (act / act_inference / evaluate / entropy など) は
    :class:`~rsl_rl.modules.ActorCritic` の実装をそのまま使う。

    ``__init__`` も **親をそのまま通す**。親は「観測は 2 次元」を assert するので、
    履歴の最新フレームだけを取り出した proxy を食わせて初期化し、actor だけを後から
    履歴版に差し替える。critic・観測正規化・action noise・分布まわりは親の初期化を
    そのまま使う。

    親を通さず自前で組み直すと、rsl_rl が ``__init__`` で増やした属性を取りこぼす。
    実際 rsl-rl-lib 3.1 で入った ``state_dependent_std`` を落として
    ``AttributeError: 'ActorCriticHistoryCNN' object has no attribute
    'state_dependent_std'`` で学習が起動しなくなった (3.0 系では同名の属性が無いので
    再現せず、バージョン差で片方だけ壊れる)。属性の面倒は親に見させること。
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        num_recent_frames: int = 5,
        cnn_kernel_sizes: list[int] = [6, 4],
        cnn_filters: list[int] = [32, 16],
        cnn_strides: list[int] = [3, 2],
        **kwargs,
    ):
        # -- actor 側: 履歴付きの 1 グループだけを受け付ける
        policy_groups = obs_groups["policy"]
        if len(policy_groups) != 1:
            raise ValueError(
                "ActorCriticHistoryCNN は履歴付きの観測グループ 1 つだけを actor 入力として扱う。"
                f" 受け取ったグループ: {policy_groups}"
            )
        policy_group = policy_groups[0]
        actor_obs = obs[policy_group]
        if actor_obs.dim() != 3:
            raise ValueError(
                f"actor の観測 '{policy_group}' は (N, history, dim) の 3 次元である必要がある"
                f" (受け取った形状: {tuple(actor_obs.shape)})。環境側で"
                " observations.policy.history_length を設定し、flatten_history_dim = False に"
                " すること。"
            )
        history_length = int(actor_obs.shape[1])
        obs_dim = int(actor_obs.shape[2])

        # -- 親の __init__ を通す
        #
        # 履歴の最新フレームだけにした proxy を渡す。親はここから
        #   * actor MLP (入力 obs_dim) … 後で履歴版に差し替えるので捨てる
        #   * actor_obs_normalizer (obs_dim 次元) … **これはそのまま使える**
        #     (統計は 1 フレーム分。(N, H, D) には最終軸でブロードキャストされる)
        #   * critic / critic_obs_normalizer / std / distribution / その他の属性
        # を作る。捨てる actor の構成を print されるとログが読めなくなるので、
        # 親の出力は飲み込んで下で本物を出し直す。
        proxy_obs = {
            key: (value[:, -1, :] if key == policy_group else value) for key, value in obs.items()
        }
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__(
                proxy_obs,
                obs_groups,
                num_actions,
                actor_obs_normalization=actor_obs_normalization,
                critic_obs_normalization=critic_obs_normalization,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                activation=activation,
                init_noise_std=init_noise_std,
                noise_std_type=noise_std_type,
                **kwargs,
            )

        self.history_length = history_length
        self.obs_dim = obs_dim

        # rsl_rl のバージョン差の吸収。3.1 以降の ActorCritic は
        # state_dependent_std を __init__ で生やし、update_distribution / act_inference が
        # それを読む。3.0 系には無いので、無ければ False を入れておく。
        if not hasattr(self, "state_dependent_std"):
            if kwargs.get("state_dependent_std"):
                raise ValueError(
                    "state_dependent_std=True が指定されたが、この rsl_rl の ActorCritic は"
                    " 対応していない (親の __init__ が属性を作らなかった)。"
                )
            self.state_dependent_std = False

        # -- actor を履歴版に差し替える
        #
        # 出力の形は親に合わせる: state_dependent_std なら [2, num_actions]
        # (平均と標準偏差)、そうでなければ num_actions。
        actor_output_dim = [2, num_actions] if self.state_dependent_std else num_actions
        self.actor = HistoryCnnActor(
            obs_dim=obs_dim,
            history_length=history_length,
            output_dim=actor_output_dim,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            num_recent_frames=num_recent_frames,
            kernel_sizes=cnn_kernel_sizes,
            filters=cnn_filters,
            strides=cnn_strides,
        )

        print(f"Actor history encoder: {self.actor.encoder.cnn}")
        print(
            f"Actor input: {num_recent_frames} x {obs_dim} (recent frames)"
            f" + {self.actor.encoder.latent_dim} (CNN latent over {history_length} frames)"
            f" = {self.actor.encoder.output_dim}"
        )
        print(f"Actor MLP: {self.actor.mlp}")
        print(f"Critic MLP: {self.critic}")

    def update_normalization(self, obs):
        """観測統計の更新。

        actor 側は履歴の全フレームではなく **現在フレームだけ** で更新する。
        バッファには同じ遷移が 50 回入っているので、全部使うと 1 ステップを 50 回
        数えることになり、リセット直後の複製フレームも重複して効いてしまう。
        """
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs[:, -1, :])
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)


class _HistoryPolicyExporter(nn.Module):
    """JIT / ONNX 書き出し用のラッパ。入力は (N, H, D) の生の観測履歴。

    正規化器を内部に焼き込むので、実機側は生の観測をそのまま積むだけでよい
    (isaaclab_rl の標準 exporter と同じ約束)。標準 exporter は actor が
    ``nn.Sequential`` である前提 (``actor[0].in_features``) なので使えない。
    """

    def __init__(self, policy: ActorCriticHistoryCNN):
        super().__init__()
        self.actor = copy.deepcopy(policy.actor)
        if getattr(policy, "actor_obs_normalizer", None) is not None:
            self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        else:
            self.normalizer = torch.nn.Identity()
        self.history_length = int(policy.history_length)
        self.obs_dim = int(policy.obs_dim)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(obs_history))

    @torch.jit.export
    def reset(self):
        pass


def export_history_policy_as_jit(policy: ActorCriticHistoryCNN, path: str, filename: str = "policy.pt"):
    """:class:`ActorCriticHistoryCNN` を TorchScript で書き出す。"""
    exporter = _HistoryPolicyExporter(policy)
    exporter.to("cpu")
    exporter.eval()
    os.makedirs(path, exist_ok=True)
    torch.jit.script(exporter).save(os.path.join(path, filename))


def export_history_policy_as_onnx(
    policy: ActorCriticHistoryCNN, path: str, filename: str = "policy.onnx", verbose: bool = False
):
    """:class:`ActorCriticHistoryCNN` を ONNX で書き出す。入力 "obs" は (1, H, D)。"""
    exporter = _HistoryPolicyExporter(policy)
    exporter.to("cpu")
    exporter.eval()
    os.makedirs(path, exist_ok=True)
    obs = torch.zeros(1, exporter.history_length, exporter.obs_dim)
    torch.onnx.export(
        exporter,
        obs,
        os.path.join(path, filename),
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
