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

import copy
import os
import torch
import torch.nn as nn
from torch.distributions import Normal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization
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
        num_actions: int,
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
        self.mlp = MLP(self.encoder.output_dim, num_actions, hidden_dims, activation)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(obs_history))


class ActorCriticHistoryCNN(ActorCritic):
    """Actor が観測履歴を、critic が 1 フレームの特権観測を見る ActorCritic。

    分布まわり (act / act_inference / evaluate / entropy など) は
    :class:`~rsl_rl.modules.ActorCritic` の実装をそのまま使う。``__init__`` だけは
    親が「観測は 2 次元」を assert するので通さず、ここで組み直す。
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
        if kwargs:
            print(
                "ActorCriticHistoryCNN.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        # 親の __init__ は 2 次元観測を前提にしているので飛ばす
        nn.Module.__init__(self)

        self.obs_groups = obs_groups

        # -- actor 側: 履歴付きの 1 グループだけを受け付ける
        policy_groups = obs_groups["policy"]
        if len(policy_groups) != 1:
            raise ValueError(
                "ActorCriticHistoryCNN は履歴付きの観測グループ 1 つだけを actor 入力として扱う。"
                f" 受け取ったグループ: {policy_groups}"
            )
        actor_obs = obs[policy_groups[0]]
        if actor_obs.dim() != 3:
            raise ValueError(
                f"actor の観測 '{policy_groups[0]}' は (N, history, dim) の 3 次元である必要がある"
                f" (受け取った形状: {tuple(actor_obs.shape)})。環境側で"
                " observations.policy.history_length を設定し、flatten_history_dim = False に"
                " すること。"
            )
        self.history_length = int(actor_obs.shape[1])
        self.obs_dim = int(actor_obs.shape[2])

        # -- critic 側: 従来どおり 1 フレームの平坦な観測
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            if obs[obs_group].dim() != 2:
                raise ValueError(
                    f"critic の観測 '{obs_group}' は 2 次元である必要がある"
                    f" (受け取った形状: {tuple(obs[obs_group].shape)})。"
                )
            num_critic_obs += obs[obs_group].shape[-1]

        # actor
        self.actor = HistoryCnnActor(
            obs_dim=self.obs_dim,
            history_length=self.history_length,
            num_actions=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            num_recent_frames=num_recent_frames,
            kernel_sizes=cnn_kernel_sizes,
            filters=cnn_filters,
            strides=cnn_strides,
        )
        # actor observation normalization
        # 統計は「1 フレームの観測次元」に対して持つ。(N, H, D) には最後の軸で
        # ブロードキャストされるので、全フレームが同じ統計で正規化される。
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor history encoder: {self.actor.encoder.cnn}")
        print(
            f"Actor input: {num_recent_frames} x {self.obs_dim} (recent frames)"
            f" + {self.actor.encoder.latent_dim} (CNN latent over {self.history_length} frames)"
            f" = {self.actor.encoder.output_dim}"
        )
        print(f"Actor MLP: {self.actor.mlp}")

        # critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

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
