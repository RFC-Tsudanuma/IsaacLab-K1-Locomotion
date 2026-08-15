"""RSL-RL runner wiring for the WalkKick likelihood policy."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.runners import OnPolicyRunner

from .model import DirectKickingActorCritic
from .ppo import DirectKickingPPO


_POLICY_CLASSES = {
    "ActorCritic": ActorCritic,
    "ActorCriticRecurrent": ActorCriticRecurrent,
    "DirectKickingActorCritic": DirectKickingActorCritic,
}
_ALGORITHM_CLASSES = {
    "PPO": PPO,
    "DirectKickingPPO": DirectKickingPPO,
}


def _resolve_class(class_name: str, classes: Mapping[str, type], kind: str) -> type:
    try:
        return classes[class_name]
    except KeyError as error:
        available = ", ".join(sorted(classes))
        raise ValueError(
            f"Unsupported {kind} class {class_name!r}; available classes: {available}"
        ) from error


def _validate_direct_checkpoint_metadata(
    policy: DirectKickingActorCritic,
    checkpoint: Mapping[str, Any],
) -> None:
    """Require an exact match for the source model's explicit input schema."""
    actual = checkpoint.get("model_metadata")
    if actual is None:
        raise ValueError(
            "DirectKicking checkpoint has no model_metadata and cannot be loaded safely"
        )
    expected = policy.checkpoint_metadata()
    if actual != expected:
        raise ValueError(
            "DirectKicking checkpoint model_metadata does not match the configured model. "
            f"Expected {expected}, got {actual}"
        )


class DirectKickingOnPolicyRunner(OnPolicyRunner):
    """Resolve local policy/PPO classes and load source-format checkpoints."""

    def _construct_algorithm(self, obs) -> PPO:
        """Construct RSL-RL exactly as upstream, with two local class names."""
        self.alg_cfg = resolve_rnd_config(
            self.alg_cfg,
            obs,
            self.cfg["obs_groups"],
            self.env,
        )
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set "
                "`actor_obs_normalization` and `critic_obs_normalization` as part of "
                "the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg[
                    "empirical_normalization"
                ]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg[
                    "empirical_normalization"
                ]

        policy_class_name = self.policy_cfg.pop("class_name")
        actor_critic_class = _resolve_class(
            policy_class_name,
            _POLICY_CLASSES,
            "policy",
        )
        actor_critic = actor_critic_class(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        algorithm_class_name = self.alg_cfg.pop("class_name")
        algorithm_class = _resolve_class(
            algorithm_class_name,
            _ALGORITHM_CLASSES,
            "algorithm",
        )
        algorithm = algorithm_class(
            actor_critic,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        algorithm.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return algorithm

    def save(self, path: str | Path, infos=None) -> None:
        """Save the native RSL-RL payload plus the DirectKicking schema metadata."""
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        metadata_getter = getattr(self.alg.policy, "checkpoint_metadata", None)
        if callable(metadata_getter):
            saved_dict["model_metadata"] = metadata_getter()

        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # Preserve RSL-RL 3.0.1's external logger upload behavior.
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str | Path,
        load_optimizer: bool = True,
        map_location: str | None = None,
    ):
        """Load source ``model`` checkpoints or preserve upstream RSL loading.

        Source DirectKicking checkpoints use ``model`` and ``model_metadata``;
        native RSL-RL checkpoints use ``model_state_dict``.  The latter are
        delegated unchanged to :class:`OnPolicyRunner`.
        """
        effective_map_location = self.device if map_location is None else map_location
        try:
            checkpoint = torch.load(
                path,
                weights_only=True,
                map_location=effective_map_location,
            )
        except Exception:
            # RSL-RL permits arbitrary ``infos`` objects.  Fall back to its
            # deserialization mode, but still inspect metadata before loading.
            checkpoint = torch.load(
                path,
                weights_only=False,
                map_location=effective_map_location,
            )

        if not isinstance(checkpoint, Mapping):
            return super().load(
                str(path),
                load_optimizer=load_optimizer,
                map_location=map_location,
            )

        if "model_state_dict" in checkpoint:
            if isinstance(self.alg.policy, DirectKickingActorCritic):
                _validate_direct_checkpoint_metadata(self.alg.policy, checkpoint)
            return super().load(
                str(path),
                load_optimizer=load_optimizer,
                map_location=map_location,
            )

        if "model" not in checkpoint:
            return super().load(
                str(path),
                load_optimizer=load_optimizer,
                map_location=map_location,
            )
        if not isinstance(self.alg.policy, DirectKickingActorCritic):
            raise ValueError(
                "A source DirectKicking checkpoint requires DirectKickingActorCritic"
            )

        _validate_direct_checkpoint_metadata(self.alg.policy, checkpoint)
        self.alg.policy.load_state_dict(checkpoint["model"], strict=True)

        # A weights-only source checkpoint need not contain optimizer state.
        # When present, loading remains caller-controlled just like RSL-RL.
        optimizer_state = checkpoint.get("optimizer")
        if load_optimizer and optimizer_state is not None:
            self.alg.optimizer.load_state_dict(optimizer_state)
        if checkpoint.get("iteration") is not None:
            self.current_learning_iteration = int(checkpoint["iteration"])
        return checkpoint.get("infos")


# The shorter alias is useful for task configuration while retaining a class
# name that clearly identifies the checkpoint contract in diagnostics.
WalkKickLikelihoodOnPolicyRunner = DirectKickingOnPolicyRunner


__all__ = [
    "DirectKickingOnPolicyRunner",
    "WalkKickLikelihoodOnPolicyRunner",
]
