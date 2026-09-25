"""DirectKick full-batch PPO ported from rl_humanoid_htwk 3af2acc9.

The update below preserves the original timeout target, loss reductions,
chunk accumulation, and post-epoch KL learning-rate adaptation.
"""

import torch
import torch.nn.functional as F

from .post_kick_phase import class_balanced_phase_multipliers, weighted_phase_binary_cross_entropy
from .utils import discount_values, surrogate_loss


class DirectKickPPO:
    def __init__(self, model, cfg, device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = torch.device(device)
        self.learning_rate = float(cfg["algorithm"]["learning_rate"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._init_post_kick_phase_auxiliary()

    def _init_post_kick_phase_auxiliary(self):
        phase_cfg = self.cfg["algorithm"].get(
            "post_kick_phase_auxiliary",
            {},
        )
        self.post_kick_phase_auxiliary_enabled = bool(
            phase_cfg.get("enabled", False)
        )
        self.post_kick_phase_loss_coefficient = float(
            phase_cfg.get("coefficient", 0.0)
        )
        self.post_kick_phase_premature_weight = float(
            phase_cfg.get("premature_weight", 3.0)
        )
        self.post_kick_phase_delayed_weight = float(
            phase_cfg.get("delayed_weight", 1.0)
        )
        if self.post_kick_phase_loss_coefficient < 0.0:
            raise ValueError(
                "post-kick phase loss coefficient must be non-negative"
            )
        if (
            self.post_kick_phase_auxiliary_enabled
            and self.post_kick_phase_loss_coefficient == 0.0
        ):
            raise ValueError(
                "enabled post-kick phase training requires a positive "
                "loss coefficient"
            )
        if (
            self.post_kick_phase_premature_weight <= 0.0
            or self.post_kick_phase_delayed_weight <= 0.0
        ):
            raise ValueError("post-kick phase class weights must be positive")
        if self.post_kick_phase_auxiliary_enabled and not callable(
            getattr(self.model, "post_kick_phase_logit", None)
        ):
            raise ValueError(
                "post-kick phase auxiliary training requires a compatible model"
            )

    def update(self, rollout, obs, privileged_obs):
        """Update from [time, env, feature] tensors and the final observations."""
        self.buffer = rollout
        flat_obses = self.buffer["obses"].reshape(-1, self.model.num_observations)
        flat_privileged_obses = self.buffer["privileged_obses"].reshape(
            -1,
            self.model.num_privileged_observations,
        )
        flat_actions = self.buffer["actions"].reshape(-1, self.model.num_actions)
        sample_count = flat_obses.shape[0]
        if self.post_kick_phase_auxiliary_enabled:
            flat_post_kick_phase_targets = self.buffer[
                "post_kick_phase_targets"
            ].reshape(-1)
            post_kick_phase_multipliers = (
                class_balanced_phase_multipliers(
                    flat_post_kick_phase_targets,
                    self.post_kick_phase_premature_weight,
                    self.post_kick_phase_delayed_weight,
                )
            )
        else:
            flat_post_kick_phase_targets = None
            post_kick_phase_multipliers = None
        configured_chunk_size = self.cfg["runner"].get(
            "optimization_chunk_size",
            sample_count,
        )
        optimization_chunk_size = (
            sample_count
            if configured_chunk_size is None
            else int(configured_chunk_size)
        )
        if optimization_chunk_size <= 0:
            raise ValueError("runner.optimization_chunk_size must be positive")
        optimization_chunk_size = min(optimization_chunk_size, sample_count)

        min_entropy = self.cfg["algorithm"]["min_entropy"]
        max_entropy = self.cfg["algorithm"]["max_entropy"]
        if (
            optimization_chunk_size < sample_count
            and min_entropy is not None
            and max_entropy is not None
        ):
            raise ValueError(
                "optimization_chunk_size does not support the global entropy "
                "range penalty"
            )

        with torch.no_grad():
            old_actions_log_prob = torch.empty(
                sample_count,
                device=self.device,
            )
            old_values_flat = torch.empty(sample_count, device=self.device)
            old_action_mean = torch.empty_like(flat_actions)
            old_action_std = torch.empty_like(flat_actions)
            for start in range(0, sample_count, optimization_chunk_size):
                end = min(start + optimization_chunk_size, sample_count)
                batch_slice = slice(start, end)
                old_dist = self.model.act(flat_obses[batch_slice])
                old_actions_log_prob[batch_slice] = old_dist.log_prob(
                    flat_actions[batch_slice]
                ).sum(dim=-1)
                old_action_mean[batch_slice] = old_dist.loc
                old_action_std[batch_slice] = old_dist.scale
                old_values_flat[batch_slice] = self.model.est_value(
                    flat_obses[batch_slice],
                    flat_privileged_obses[batch_slice],
                )

            old_values = old_values_flat.reshape_as(self.buffer["rewards"])
            old_last_values = self.model.est_value(obs, privileged_obs)
            # Compute returns once using old values (they shouldn't change during mini epochs)
            self.buffer["rewards"][self.buffer["time_outs"]] = old_values[self.buffer["time_outs"]]
            advantages = discount_values(
                self.buffer["rewards"],
                self.buffer["dones"] | self.buffer["time_outs"],
                old_values,
                old_last_values,
                self.cfg["algorithm"]["gamma"],
                self.cfg["algorithm"]["lam"],
            )
            returns = old_values + advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        flat_old_values = old_values.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)

        # Get value clip parameter (default to None for no clipping, for backwards compatibility)
        value_clip_param = self.cfg["algorithm"].get("value_clip_param", None)
        symmetric_coef = float(
            self.cfg["algorithm"].get("symmetric_coef", 0.0)
        )
        if symmetric_coef < 0.0:
            raise ValueError("algorithm.symmetric_coef must be non-negative")
        symmetry_loss_fn = getattr(self.model, "compute_symmetry_loss", None)
        use_symmetry_loss = (
            symmetric_coef > 0.0
            and callable(symmetry_loss_fn)
            and bool(getattr(self.model, "mirror_consistency_enabled", True))
        )

        mean_value_loss = 0
        mean_actor_loss = 0
        mean_bound_loss = 0
        mean_entropy = 0
        mean_symmetry_loss = 0
        mean_symmetry_weight = 0
        mean_post_kick_phase_loss = 0
        for n in range(self.cfg["runner"]["mini_epochs"]):
            self.optimizer.zero_grad()
            epoch_value_loss = 0.0
            epoch_actor_loss = 0.0
            epoch_bound_loss = 0.0
            epoch_entropy = 0.0
            epoch_symmetry_loss = 0.0
            epoch_symmetry_weight = 0.0
            epoch_post_kick_phase_loss = 0.0
            for start in range(0, sample_count, optimization_chunk_size):
                end = min(start + optimization_chunk_size, sample_count)
                batch_slice = slice(start, end)
                batch_weight = float(end - start) / float(sample_count)
                values = self.model.est_value(
                    flat_obses[batch_slice],
                    flat_privileged_obses[batch_slice],
                )

                if value_clip_param is not None:
                    values_clipped = flat_old_values[batch_slice] + torch.clamp(
                        values - flat_old_values[batch_slice],
                        -value_clip_param,
                        value_clip_param,
                    )
                    value_loss_unclipped = (
                        values - flat_returns[batch_slice]
                    ).pow(2)
                    value_loss_clipped = (
                        values_clipped - flat_returns[batch_slice]
                    ).pow(2)
                    value_loss = 0.5 * torch.max(
                        value_loss_unclipped,
                        value_loss_clipped,
                    ).mean()
                else:
                    value_loss = F.mse_loss(
                        values,
                        flat_returns[batch_slice],
                    )

                dist = self.model.act(flat_obses[batch_slice])
                actions_log_prob = dist.log_prob(
                    flat_actions[batch_slice]
                ).sum(dim=-1)
                actor_loss = surrogate_loss(
                    old_actions_log_prob[batch_slice],
                    actions_log_prob,
                    flat_advantages[batch_slice],
                )
                bound_loss = (
                    torch.clip(dist.loc - 1.0, min=0.0).square().mean()
                    + torch.clip(dist.loc + 1.0, max=0.0).square().mean()
                )
                entropy = dist.entropy().sum(dim=-1)
                entropy_mean = entropy.mean()
                symmetry_loss = dist.loc.new_zeros(())
                symmetry_weight = dist.loc.new_zeros(())
                if use_symmetry_loss:
                    symmetry_loss, symmetry_weight = symmetry_loss_fn(
                        flat_obses[batch_slice],
                        dist.loc,
                    )
                post_kick_phase_loss = dist.loc.new_zeros(())
                if self.post_kick_phase_auxiliary_enabled:
                    post_kick_phase_logits = (
                        self.model.post_kick_phase_logit(
                            flat_obses[batch_slice]
                        )
                    )
                    post_kick_phase_loss = (
                        weighted_phase_binary_cross_entropy(
                            post_kick_phase_logits,
                            flat_post_kick_phase_targets[batch_slice],
                            post_kick_phase_multipliers[batch_slice],
                        )
                    )

                if min_entropy is not None and max_entropy is not None:
                    loss_entropy = torch.mean(
                        (
                            torch.clamp(
                                entropy_mean,
                                min=min_entropy,
                                max=max_entropy,
                            )
                            - entropy_mean
                        )
                        ** 2
                    )
                else:
                    loss_entropy = 0.0
                loss = (
                    value_loss
                    + actor_loss
                    + self.cfg["algorithm"]["bound_coef"] * bound_loss
                    + self.cfg["algorithm"]["entropy_coef"] * entropy_mean
                    + 0.01 * loss_entropy
                    + symmetric_coef * symmetry_loss
                    + self.post_kick_phase_loss_coefficient
                    * post_kick_phase_loss
                )
                (loss * batch_weight).backward()
                epoch_value_loss += value_loss.item() * batch_weight
                epoch_actor_loss += actor_loss.item() * batch_weight
                epoch_bound_loss += bound_loss.item() * batch_weight
                epoch_entropy += entropy_mean.item() * batch_weight
                epoch_symmetry_loss += symmetry_loss.item() * batch_weight
                epoch_symmetry_weight += symmetry_weight.item() * batch_weight
                epoch_post_kick_phase_loss += (
                    post_kick_phase_loss.item() * batch_weight
                )

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            mean_value_loss += epoch_value_loss
            mean_actor_loss += epoch_actor_loss
            mean_bound_loss += epoch_bound_loss
            mean_entropy += epoch_entropy
            mean_symmetry_loss += epoch_symmetry_loss
            mean_symmetry_weight += epoch_symmetry_weight
            mean_post_kick_phase_loss += epoch_post_kick_phase_loss

        # Calculate KL divergence after all mini epochs (between old and final policy)
        with torch.no_grad():
            kl_sum = torch.zeros((), device=self.device)
            post_kick_phase_probability_sum = torch.zeros(
                (),
                device=self.device,
            )
            post_kick_phase_ready_probability_sum = torch.zeros(
                (),
                device=self.device,
            )
            post_kick_phase_ready_count = torch.zeros(
                (),
                device=self.device,
            )
            for start in range(0, sample_count, optimization_chunk_size):
                end = min(start + optimization_chunk_size, sample_count)
                batch_slice = slice(start, end)
                final_dist = self.model.act(flat_obses[batch_slice])
                kl = torch.sum(
                    torch.log(
                        final_dist.scale / old_action_std[batch_slice]
                    )
                    + 0.5
                    * (
                        torch.square(old_action_std[batch_slice])
                        + torch.square(
                            final_dist.loc - old_action_mean[batch_slice]
                        )
                    )
                    / torch.square(final_dist.scale)
                    - 0.5,
                    dim=-1,
                )
                kl_sum += kl.sum()
                if self.post_kick_phase_auxiliary_enabled:
                    phase_probability = torch.sigmoid(
                        self.model.post_kick_phase_logit(
                            flat_obses[batch_slice]
                        )
                    )
                    post_kick_phase_probability_sum += (
                        phase_probability.sum()
                    )
                    ready = (
                        flat_post_kick_phase_targets[batch_slice] >= 0.5
                    )
                    post_kick_phase_ready_probability_sum += (
                        phase_probability[ready].sum()
                    )
                    post_kick_phase_ready_count += ready.sum()
            kl_mean = kl_sum / float(sample_count)
            post_kick_phase_probability_mean = (
                post_kick_phase_probability_sum / float(sample_count)
            )
            post_kick_phase_ready_probability_mean = (
                post_kick_phase_ready_probability_sum
                / torch.clamp(post_kick_phase_ready_count, min=1.0)
            )

            # Adapt learning rate based on KL divergence
            if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        mean_value_loss /= self.cfg["runner"]["mini_epochs"]
        mean_actor_loss /= self.cfg["runner"]["mini_epochs"]
        mean_bound_loss /= self.cfg["runner"]["mini_epochs"]
        mean_entropy /= self.cfg["runner"]["mini_epochs"]
        mean_symmetry_loss /= self.cfg["runner"]["mini_epochs"]
        mean_symmetry_weight /= self.cfg["runner"]["mini_epochs"]
        mean_post_kick_phase_loss /= self.cfg["runner"]["mini_epochs"]
        return {
            "value_loss": mean_value_loss,
            "actor_loss": mean_actor_loss,
            "bound_loss": mean_bound_loss,
            "entropy": mean_entropy,
            "symmetry_loss": mean_symmetry_loss,
            "symmetry_weight": mean_symmetry_weight,
            "phase_loss": mean_post_kick_phase_loss,
            "phase_probability": post_kick_phase_probability_mean.item(),
            "phase_ready_probability": post_kick_phase_ready_probability_mean.item(),
            "kl": kl_mean.item(),
            "learning_rate": self.learning_rate,
        }
