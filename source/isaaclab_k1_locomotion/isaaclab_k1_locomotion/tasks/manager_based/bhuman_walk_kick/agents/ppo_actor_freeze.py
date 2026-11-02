# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""最初の数 update だけ actor を固定して critic を慣らす PPO (rsl_rl)。

歩行の重みから始めると、critic も歩行の報酬で学習した値のまま。報酬 (ボール関係) が変わった
直後に PPO を回すと、合わない価値の推定で actor が大きく動いて歩き方が崩れる
(feat/bhuman_kick で実測: 約 100 iteration で全エピソード転倒に戻った)。
``actor_freeze_iterations`` 回の update は actor (と std) の勾配を止め、学習率も固定する。
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage


class MaskedRolloutStorage(RolloutStorage):
    """標本ごとの「キック方策の標本か」(valid) を持つ RolloutStorage。全部 valid なら rsl_rl のものと同じ動き。

    歩行の受け渡し (bhuman_walk_kick/mdp/walk_handoff.py、既定は無効) で、凍結した歩行方策が動かしたステップは
    キック方策の action ではないので、方策の損失 (surrogate)・価値の損失・mirror loss の標本から外す。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid = torch.ones(self.num_transitions_per_env, self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.masked_frac = 0.0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        """歩行が動かしたステップを「時間だけ進む env の中の動き」として飛ばした GAE。

        受け渡しの境目の考え方: 歩行の区間 (報酬 0、env 側で 0 にしている) はキック方策が選べない動きなので、
        キック方策の標本 t の次の状態は「次のキック方策の標本 t'」(歩行の区間を挟むなら受け渡しの瞬間) とし、
        その間の時間 Δ = t' - t だけ割り引く (δ_t = r_t + γ^Δ V(s_t') - V(s_t)、A_t = δ_t + (γλ の代わりに γ^Δ λ) A_t')。
        歩行の状態の価値 V は価値の損失で学習しないので、ブートストラップに使わない (使うとまだ合っていない値が
        キックの後の標本に入る)。歩行の区間の途中でエピソードが終わった (歩行の転倒・時間切れ) ときは、その前の
        キック方策の標本はそこで終わり (次の価値 0) とする。めったに起きない (歩行の転倒はまれ、時間切れは 40 秒)。
        rollout の最後が歩行の区間なら、最後の観測の価値 (last_values) を使う (歩行の状態の価値で、まだ合っていない
        ことがある。各 env の rollout 1 回につき最後の 1 標本だけ)。
        """
        if bool(self.valid.all()):
            self.masked_frac = 0.0
            return super().compute_returns(last_values, gamma, lam, normalize_advantage)
        adv = torch.zeros_like(last_values)
        next_v = last_values
        cont = torch.ones_like(last_values)  # 次のキック方策の標本までにエピソードの終わりが無い
        disc = torch.ones_like(last_values)  # 次のキック方策の標本までに飛ばした時間の割引 γ^(Δ-1)
        for step in reversed(range(self.num_transitions_per_env)):
            v = self.valid[step].float()
            not_done = 1.0 - self.dones[step].float()
            g = gamma * not_done * cont * disc
            delta = self.rewards[step] + g * next_v - self.values[step]
            a = delta + g * lam * adv
            self.returns[step] = torch.where(v > 0, a + self.values[step], self.values[step])
            # キック方策の標本: ここを次の標本にする。歩行の標本: 1 ステップ分の割引とエピソードの終わりだけ足す
            adv = torch.where(v > 0, a, adv)
            next_v = torch.where(v > 0, self.values[step], next_v)
            cont = torch.where(v > 0, torch.ones_like(cont), cont * not_done)
            disc = torch.where(v > 0, torch.ones_like(disc), disc * gamma)
        self.advantages = torch.where(self.valid, self.returns - self.values, torch.zeros_like(self.values))
        vm = self.valid
        self.masked_frac = 1.0 - vm.float().mean().item()
        if normalize_advantage:
            a = self.advantages[vm]
            self.advantages = torch.where(vm, (self.advantages - a.mean()) / (a.std() + 1e-8), self.advantages)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if bool(self.valid.all()):
            yield from super().mini_batch_generator(num_mini_batches, num_epochs)
            return
        # キック方策の標本だけからミニバッチを作る
        idx = self.valid.flatten().nonzero(as_tuple=False).flatten()
        mini_batch_size = len(idx) // num_mini_batches
        if mini_batch_size == 0:
            return
        indices = idx[torch.randperm(len(idx), device=self.device)[: num_mini_batches * mini_batch_size]]
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                b = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield (
                    observations[b], actions[b], values[b], advantages[b], returns[b], old_actions_log_prob[b],
                    old_mu[b], old_sigma[b], (None, None), None,
                )


class PPOActorFreeze(PPO):
    """``std_range`` (B-Human の entropy_range に相当): update のたびに方策のノイズ (std) をこの範囲に clamp する。
    B-Human は「エントロピーを範囲内に保つ」損失で探索の広さを段ごとに決めている (Stage 1 は広く、後段で狭く)。
    うちの std は scalar (noise_std_type="scalar") なので、値を直接 clamp する。None なら何もしない。"""

    def __init__(self, policy, actor_freeze_iterations: int = 0, std_range: tuple | None = None, **kwargs):
        super().__init__(policy, **kwargs)
        self.actor_freeze_iterations = int(actor_freeze_iterations)
        self.std_range = None if std_range is None else (float(std_range[0]), float(std_range[1]))
        self._num_updates = 0
        self._clamp_std()

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.storage = MaskedRolloutStorage(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, self.device
        )

    def process_env_step(self, obs, rewards, dones, extras):
        """歩行の受け渡し (env.extras の bwk_walk_driven / bwk_walk_mode) があれば、歩行が動かしたステップを標本から外す。
        無ければ rsl_rl の PPO と同じ。"""
        driven = extras.get("bwk_walk_driven")
        if driven is None:
            return super().process_env_step(obs, rewards, dones, extras)
        self.storage.valid[self.storage.step] = ~driven.to(self.device).view(-1, 1)
        # 観測の正規化は、次にキック方策が動く env の観測だけで更新する (歩行で近づいている間の状態は入れない)
        walk_next = extras["bwk_walk_mode"].to(self.device)
        if bool((~walk_next).any()):
            self.policy.update_normalization(obs[~walk_next])
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)
        self._handoff_seen = True

    def _clamp_std(self):
        if self.std_range is None or self.policy.noise_std_type != "scalar":
            return
        with torch.no_grad():
            self.policy.std.clamp_(*self.std_range)

    def _actor_params(self):
        return [p for n, p in self.policy.named_parameters() if not n.startswith("critic")]

    def update(self):
        freeze = self._num_updates < self.actor_freeze_iterations
        if freeze:
            schedule = self.schedule
            self.schedule = "fixed"
            for p in self._actor_params():
                p.requires_grad_(False)
        try:
            loss_dict = super().update()
        finally:
            if freeze:
                self.schedule = schedule
                for p in self._actor_params():
                    p.requires_grad_(True)
        self._num_updates += 1
        self._clamp_std()
        if getattr(self, "_handoff_seen", False):
            loss_dict["walk_masked_frac"] = self.storage.masked_frac  # 歩行が動かして標本から外した割合
        return loss_dict
