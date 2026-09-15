# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""複数 expert 同居環境で 1 つの expert だけを学習する PPO (歩行 ⇄ 回転の遷移学習用)。

RPG (Robust Policy Gating) の「全 expert を同一 env にロードし、アクティブな 1 本だけを
更新する」枝を rsl_rl の PPO に載せたもの。環境側は `TransitionCommand` が env ごとの
モード (0=walk / 1=turn) を持ち、補助観測グループ ``expert_mode`` にそれを出す。

* ``learner_mode`` の env: 学習中の方策 (``self.policy``) がアクションを出し、遷移を学習に使う。
* それ以外の env: 凍結 expert (``frozen_checkpoints[mode]``) の決定論アクションを実行する。
  これらの遷移は「保存アクション ≠ 実行アクション」で PPO 的に破損しているので、
  損失から完全に除外する。

除外の実装は `MaskedRolloutStorage`:
  - ``mask[step, env]`` = その step で学習方策がアクティブだったか
  - ``mini_batch_generator`` は mask==True の遷移だけを並べ替えて出す
    → surrogate / value / entropy / KL / mirror loss の全てから自然に外れる (PPO.update は無改造)
  - advantage の正規化も mask 内で行う

学習方策 → 凍結 expert への引き渡し step は、学習方策にとって「軌跡の打ち切り」なので
``dones=1, time_outs=1`` を注入して value でブートストラップする (rsl_rl 標準の time-out
処理をそのまま流用)。凍結 expert → 学習方策の引き渡しは学習側の新しい区間の開始で、
GAE は後ろ向き再帰なので前の (マスク済み) 区間から漏れてくるものはない。

観測正規化 (EmpiricalNormalization) の統計更新も「次 step で学習方策がアクティブな env」
の観測に限定する。凍結 expert は ``copy.deepcopy(policy)`` に checkpoint を読み込んだもので、
自身の正規化統計を持ち更新されない (eval / no_grad)。

rsl_rl の OnPolicyRunner はアルゴリズムを ``eval(class_name)`` で on_policy_runner モジュールの
名前空間から解決するため、本モジュール末尾で ``MultiExpertPPO`` を注入している
(HistoryActorCritic と同じ方式)。
"""

from __future__ import annotations

import copy
import os

import torch
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from ..mdp.commands import MODE_NAMES


def resolve_mode(mode: int | str) -> int:
    """モード指定 ("walk"/"turn" または 0/1) を整数に正規化する。"""
    if isinstance(mode, str):
        key = mode.strip().lower()
        if key in MODE_NAMES:
            return MODE_NAMES[key]
        return int(key)
    return int(mode)


class MaskedRolloutStorage(RolloutStorage):
    """per-transition のマスクを持ち、mask==True の遷移だけをミニバッチに出す RolloutStorage。"""

    def __init__(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape, device="cpu"):
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.mask = torch.ones(num_transitions_per_env, num_envs, dtype=torch.bool, device=self.device)

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        # GAE は全 env で計算 (マスク外の値は使われないだけ)。正規化はマスク内で行う。
        super().compute_returns(last_values, gamma, lam, normalize_advantage=False)
        if normalize_advantage:
            valid = self.advantages[self.mask]
            if valid.numel() > 1:
                self.advantages = (self.advantages - valid.mean()) / (valid.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        valid_idx = self.mask.flatten().nonzero(as_tuple=False).flatten()
        num_valid = int(valid_idx.numel())
        mini_batch_size = num_valid // num_mini_batches
        if mini_batch_size == 0:
            raise RuntimeError(
                f"MaskedRolloutStorage: 学習方策がアクティブな遷移が {num_valid} 件しかなく、"
                f" {num_mini_batches} ミニバッチに分けられません。learner_mode / モード遷移の設定を確認してください。"
            )

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            perm = valid_idx[torch.randperm(num_valid, device=self.device)]
            for i in range(num_mini_batches):
                batch_idx = perm[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield observations[batch_idx], actions[batch_idx], values[batch_idx], advantages[batch_idx], returns[
                    batch_idx
                ], old_actions_log_prob[batch_idx], old_mu[batch_idx], old_sigma[batch_idx], (None, None), None

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        raise NotImplementedError("MaskedRolloutStorage は再帰方策 (RNN) に対応していません。")


class MultiExpertPPO(PPO):
    """学習中 expert と凍結 expert を env のモードで振り分ける PPO。"""

    def __init__(
        self,
        policy,
        learner_mode: int | str = 0,
        frozen_checkpoints: dict | None = None,
        mode_obs_group: str = "expert_mode",
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        self.learner_mode = resolve_mode(learner_mode)
        self.mode_obs_group = mode_obs_group
        self.frozen: dict[int, torch.nn.Module] = {}
        for mode, path in (frozen_checkpoints or {}).items():
            mode_idx = resolve_mode(mode)
            if mode_idx == self.learner_mode:
                raise ValueError(f"MultiExpertPPO: 凍結 expert のモード {mode!r} が learner_mode と同じです。")
            if not path:
                continue
            if not os.path.isfile(path):
                raise FileNotFoundError(f"MultiExpertPPO: 凍結 expert の checkpoint が見つかりません: {path}")
            expert = copy.deepcopy(policy)
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            expert.load_state_dict(ckpt["model_state_dict"], strict=True)
            expert.eval()
            for p in expert.parameters():
                p.requires_grad_(False)
            self.frozen[mode_idx] = expert
            print(f"[MultiExpertPPO] frozen expert for mode {mode_idx}: {path}")
        # 反対モードの凍結 expert が無いと、学習方策が両モードを駆動しつつ非学習モードの
        # 遷移だけがマスクで捨てられ、「自分自身を相手役にして遷移を学ぶ」状態に黙って入る。
        # 遷移学習の前提が崩れるので構築時に止める。
        missing = sorted(set(MODE_NAMES.values()) - {self.learner_mode} - set(self.frozen))
        if missing:
            names = [name for name, idx in MODE_NAMES.items() if idx in missing]
            raise ValueError(
                f"MultiExpertPPO: モード {names} の凍結 expert が指定されていません。"
                " --frozen_ckpt MODE=/abs/path/model.pt (または cfg の frozen_checkpoints) で与えてください。"
            )
        print(f"[MultiExpertPPO] learner mode: {self.learner_mode}, frozen modes: {sorted(self.frozen)}")
        # act() 時点の「学習方策がアクティブか」(process_env_step で参照)
        self._active: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # storage
    # ------------------------------------------------------------------
    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.storage = MaskedRolloutStorage(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, self.device
        )

    # ------------------------------------------------------------------
    # rollout
    # ------------------------------------------------------------------
    def _modes(self, obs: TensorDict) -> torch.Tensor:
        return obs[self.mode_obs_group][:, 0].round().long()

    def _mux_actions(self, obs: TensorDict, learner_actions: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
        actions = learner_actions
        for mode_idx, expert in self.frozen.items():
            sel = modes == mode_idx
            if sel.any():
                expert_actions = expert.act_inference(obs)
                actions = torch.where(sel.unsqueeze(1), expert_actions, actions)
        return actions

    def act(self, obs):
        modes = self._modes(obs)
        self._active = modes == self.learner_mode
        # 学習方策は全 env でサンプルし (保存される)、実行アクションだけを差し替える
        learner_actions = super().act(obs)
        return self._mux_actions(obs, learner_actions, modes)

    def act_inference(self, obs):
        """play 用: 学習方策 (決定論) と凍結 expert をモードで振り分けた実行アクション。"""
        modes = self._modes(obs)
        return self._mux_actions(obs, self.policy.act_inference(obs), modes)

    def process_env_step(self, obs, rewards, dones, extras):
        active = self._active
        if active is None:
            active = torch.ones(dones.shape[0], dtype=torch.bool, device=self.device)
        modes_next = self._modes(obs)
        active_next = modes_next == self.learner_mode

        # 学習方策 → 凍結 expert への引き渡し: 学習側の軌跡はここで打ち切り (value でブートストラップ)
        handoff_out = active & ~active_next & (dones == 0)
        dones = dones.clone()
        dones[handoff_out] = 1
        time_outs = extras.get("time_outs")
        if time_outs is None:
            time_outs = torch.zeros_like(dones, dtype=torch.bool)
        else:
            time_outs = time_outs.to(self.device).clone()
        time_outs[handoff_out] = True

        # 正規化統計は「次 step で学習方策がアクティブな env」の観測だけで更新
        if active_next.any():
            self.policy.update_normalization(obs[active_next])
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards
        # time-out ブートストラップ (rsl_rl 標準: V(s_t) を使う近似)
        self.transition.rewards += self.gamma * torch.squeeze(
            self.transition.values * time_outs.unsqueeze(1).to(self.device), 1
        )

        self.storage.mask[self.storage.step] = active
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)


@configclass
class RslRlMultiExpertPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """`MultiExpertPPO` 用の設定。"""

    class_name: str = "MultiExpertPPO"

    learner_mode: str = "walk"
    """学習する expert のモード ("walk" / "turn")。"""

    frozen_checkpoints: dict = {}
    """凍結 expert の checkpoint パス。キーはモード名 ("walk" / "turn")。
    train.py の ``--frozen_ckpt turn=/path/to/model.pt`` で上書きできる。"""

    mode_obs_group: str = "expert_mode"
    """モードを載せている補助観測グループ名。"""


# OnPolicyRunner は eval(class_name) を on_policy_runner モジュールの名前空間で
# 評価するため、そこにクラスを注入する。
import rsl_rl.runners.on_policy_runner as _rsl_rl_on_policy_runner  # noqa: E402

_rsl_rl_on_policy_runner.MultiExpertPPO = MultiExpertPPO
