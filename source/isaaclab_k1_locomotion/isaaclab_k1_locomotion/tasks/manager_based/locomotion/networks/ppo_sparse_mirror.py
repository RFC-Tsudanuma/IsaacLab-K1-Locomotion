# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""mirror loss を数ミニバッチに 1 回だけ掛ける PPO (rsl_rl 用)。

なぜ要るか
----------
dual 系タスク (walk_kick_dual / walk_weak_kick_dual / walk_middle_kick_dual) は
履歴観測 (N, 100, 55) + :class:`~.ActorCriticHistoryCNN` + mirror loss で学習する。
rsl_rl の mirror loss は **ミニバッチごとに** 「観測を左右反転して 2 倍に増やし、
actor をもう一度通す」ので、追加コストが actor の forward + backward 1 回分ずつ
乗る。履歴 CNN の actor は learning 時間の支配項なので、これがそのまま出る。

実測 (2026-08-17)::

    mirror loss 無し   learning 0.78 s / iteration
    mirror loss 有り   learning 2.54 s / iteration     ← ×3.3

1 update あたりのミニバッチ数は ``num_learning_epochs * num_mini_batches``
= 5 * 8 = 40 で、mirror loss はその全部に掛かっていた。これを
:data:`PPOSparseMirror.mirror_loss_interval` = 5 で間引くと **40 回中 8 回**
だけになり、learning は 2.54 s → 1.1 s 程度まで戻る見込み
(0.78 + (2.54 - 0.78) / 5 ≈ 1.13)。

間引き方は「update() の冒頭でカウンタを 0 に戻し、ミニバッチ番号が 5 の倍数の
ときだけ掛ける」。**毎 update 決定論的に同じ番号のミニバッチ (0, 5, 10, ..., 35)**
に当たる。ミニバッチの中身は毎回シャッフルされるので、特定のサンプルに偏ることは
無い。「5 update に 1 回だけ全ミニバッチに掛ける」でも平均コストは同じだが、
そちらは 40 回連続で対称化圧が掛かってから 160 回何も掛からない、という
バースト的な勾配になるので採らない。

.. warning::
    **このモジュールは rsl_rl の :meth:`PPO.update` を逐語コピーしている。**
    コピー元は **rsl-rl-lib 3.1.1** の ``rsl_rl/algorithms/ppo.py``。
    **rsl_rl を更新したら、このコピーも取り直すこと。** 差分は
    :meth:`PPOSparseMirror.update` の docstring に列挙した 5 箇所だけで、
    それ以外は 1 文字も変えていない。

NOTE:
    **なぜ 3.1.1 なのか (IsaacLab 2.3.2 の setup.py は 3.1.2 を pin しているのに)。**
    学習マシン (vast コンテナ) で CUDA OOM が出たときの **生のトレースバック**の
    行番号が、独立に 2 点とも 3.1.1 系と一致し、3.1.2 とは一致しなかった::

        ppo.py:328              act_inference(obs_batch.detach().clone())
                                → 3.0.1/3.1.0/3.1.1 で 328 行、3.1.2/3.1.3 では 326 行
        on_policy_runner.py:149 loss_dict = self.alg.update()
                                → 3.1.1 で 149 行、3.1.2 では 148 行

    ``setup.py`` の ``rsl-rl-lib==3.1.2`` は「あるべき姿」であって「実際に入って
    いるもの」ではない (isaac-sim の kit python イメージに別版が同梱されている
    ことは普通にある)。**実機の直接観測を正とした。**
    なお ``ppo.py`` は 3.0.1 / 3.1.0 / 3.1.1 でバイト同一なので、この 3 版の
    どれであってもこのコピーはそのまま使える。

.. warning::
    3.1.2 以降とは非互換で、``self.policy.act`` / ``self.policy.evaluate`` の
    キーワードが ``hidden_states=`` (3.1.1) → ``hidden_state=`` (3.1.2) に、
    ループ変数が ``hid_states_batch`` → ``hidden_states_batch`` に変わっている。
    **3.1.2 以降で回すと最初の update で
    ``TypeError: act() got an unexpected keyword argument 'hidden_states'``
    で落ちる。** 起動時にバージョンを照合して警告を出しているので、ログの
    ``[PPOSparseMirror]`` 行を確認すること。

NOTE:
    間引きと mirror_loss_coeff の関係。適用頻度が 1/5 になるぶん、**平均的な
    対称化の圧も 1/5 になる**。係数側で
    (``walk_kick_both_feet/agents/rsl_rl_ppo_cfg.py`` の ``_MIRROR_LOSS_COEFF``)
    0.5 → 2.5 に上げれば期待勾配は元と等価になるが、「1 回あたりの勾配が 5 倍に
    なって更新が荒れる」ので等価とは言い切れない。**ここでは係数は 0.5 のまま
    据え置く。** 実際の学習を見て判断すること:

    * 圧が足りない兆候 = ``Metrics/kick_direction/kick_foot_right_frac`` が
      0 / 1 に張り付く (片足に戻る) → ``_MIRROR_LOSS_COEFF`` を 2.5 まで上げる。
    * 上げすぎの兆候 = ``Metrics/kick_direction/kick_dir_error_deg`` の悪化
      (「左右対称に振る舞う」が「指令どおり蹴る」より優先されている)。

NOTE:
    副次効果として GPU メモリも楽になる。mirror loss を掛けるミニバッチでだけ
    ``obs_batch.detach().clone()`` の 2 倍バッチが確保されるので、5 回に 4 回は
    そのピークが消える (``_NUM_MINI_BATCHES`` = 8 の理由だった OOM 対策と同じ話)。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO

# コピー元の rsl-rl-lib のバージョン。この 3 版は ppo.py がバイト同一なので
# どれでも match 扱いにする。実行時に照合して、外れていたら警告する。
_COPIED_FROM_RSL_RL_VERSIONS = ("3.0.1", "3.1.0", "3.1.1")


def _warn_on_version_mismatch() -> None:
    """インストール済みの rsl-rl-lib が :data:`_COPIED_FROM_RSL_RL_VERSIONS` か照合する。

    外れていても止めはしない (前方互換のことがある) が、**3.1.2 以降なら最初の
    update で ``TypeError`` になる**ので、起動ログに残るように警告する。
    """
    try:
        from importlib.metadata import version

        installed = version("rsl-rl-lib")
    except Exception:  # メタデータが引けない環境では黙って諦める
        return
    if installed in _COPIED_FROM_RSL_RL_VERSIONS:
        print(f"[PPOSparseMirror] rsl-rl-lib {installed} (コピー元と一致)")
        return
    print(
        f"[PPOSparseMirror] 警告: rsl-rl-lib {installed} がインストールされているが、"
        f"PPO.update() のコピー元は {'/'.join(_COPIED_FROM_RSL_RL_VERSIONS)} である。\n"
        f"[PPOSparseMirror]   ppo_sparse_mirror.py の update() を "
        f"{installed} の rsl_rl/algorithms/ppo.py から取り直すこと。\n"
        f"[PPOSparseMirror]   3.1.2 以降では self.policy.act() のキーワードが "
        f"hidden_state= なので、このまま回すと最初の update で TypeError になる。"
    )


class PPOSparseMirror(PPO):
    """mirror loss を :data:`mirror_loss_interval` ミニバッチに 1 回だけ掛ける PPO。

    :class:`~rsl_rl.algorithms.PPO` との違いは :meth:`update` の 5 箇所だけ。
    コンストラクタは触っていないので、``RslRlPpoAlgorithmCfg`` の引数はそのまま
    通る (``class_name = "PPOSparseMirror"`` を指定するだけで差し替わる)。

    ``mirror_loss_interval`` を cfg のフィールドにはしていない。IsaacLab の
    ``RslRlPpoAlgorithmCfg`` は固定フィールドで、余分な値は ``**self.alg_cfg`` に
    載って ``PPO.__init__`` に渡され ``TypeError`` になるため。値を変えるときは
    このクラス属性を書き換えること。
    """

    # ----------------------------------------------------------------------- #
    # mirror loss を掛けるミニバッチの間隔
    #
    # 1 update = num_learning_epochs * num_mini_batches = 5 * 8 = 40 ミニバッチ。
    # 5 なら 0, 5, 10, ..., 35 の **8 回**だけ掛かる (40 回中 8 回 = 20 %)。
    # learning は実測 2.54 s から ~1.1 s に戻る見込み。
    #
    # 1 にすれば素の PPO と同じ挙動になるが、**このクラスを使う時点で間引きが
    # 目的**なので既定は 5。素の挙動が欲しいときは class_name を "PPO" に戻すこと。
    # ----------------------------------------------------------------------- #
    mirror_loss_interval: int = 5

    def update(self):  # noqa: C901
        """rsl-rl-lib 3.1.1 の :meth:`PPO.update` の逐語コピー + 間引きのゲート。

        元との差分は以下の 5 箇所だけ (``# [差分]`` のコメントを付けてある)。
        それ以外は 1 文字も変えていない。

        1. 冒頭でミニバッチカウンタを 0 に戻し、実際に mirror loss を掛けた回数
           ``num_mirror_updates`` を用意する。
        2. ミニバッチの先頭で ``apply_mirror`` を決めてカウンタを進める。
        3. ``if self.symmetry:`` → ``if self.symmetry and apply_mirror:``。
           **ブロックごと飛ばす**のが要点で、``use_mirror_loss`` の分岐だけを
           塞いでも ``act_inference`` の forward は走ってしまい、狙った分だけ
           速くならない。
        4. ``mean_symmetry_loss`` の加算を ``apply_mirror`` でも守る。掛けなかった
           ミニバッチでは ``symmetry_loss`` が前のミニバッチの値のまま残っている
           (最初のミニバッチでは未定義) ため、素通しにすると誤った値を足す。
           併せて ``num_mirror_updates`` を数える。
        5. ``mean_symmetry_loss`` を ``num_updates`` ではなく
           ``num_mirror_updates`` で割る。**間引く前と同じ尺度** (掛けた
           ミニバッチでの平均) でログに出したいため。``num_updates`` で割ると
           値だけが 1/5 になって、間引き前の run と比べられなくなる。

        なお ``num_aug`` はこのメソッド内で読まれない (代入だけ) ので、symmetry
        ブロックを飛ばしても後段に影響しない。``obs_batch`` を 2 倍に差し替える
        のも同ブロック内だけで、後段で参照するのは RND (未使用) の
        ``obs_batch[:original_batch_size]`` だけなので、どちらでも同じ。
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # [差分 1] mirror loss の間引き用。update ごとに 0 に戻すので、毎回同じ
        #          番号のミニバッチ (0, 5, 10, ...) に当たる。
        mirror_minibatch_counter = 0
        num_mirror_updates = 0

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # [差分 2] このミニバッチで mirror loss を掛けるかどうか。
            apply_mirror = mirror_minibatch_counter % self.mirror_loss_interval == 0
            mirror_minibatch_counter += 1

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            # we assume policy group is always there and needs augmentation
            original_batch_size = obs_batch.batch_size[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # compute number of augmentations per sample
                # we assume policy group is always there and needs augmentation
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            # [差分 3] 間引きのゲート。元は ``if self.symmetry:``。
            if self.symmetry and apply_mirror:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            # [差分 4] 掛けなかったミニバッチでは symmetry_loss が前の値のまま
            #          (最初のミニバッチでは未定義) なので、apply_mirror でも守る。
            if mean_symmetry_loss is not None and apply_mirror:
                mean_symmetry_loss += symmetry_loss.item()
                num_mirror_updates += 1

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            # [差分 5] 間引き前と同じ尺度で出すため num_updates ではなく
            #          実際に掛けた回数で割る。
            mean_symmetry_loss /= max(num_mirror_updates, 1)
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict


_warn_on_version_mismatch()
