"""Performance-driven curricula for the walk-kick likelihood task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg

from .events import (
    _BALL_INCOMING_ATTR,
    _BALL_RESET_METADATA_VALID_ATTR,
    _BALL_SPEED_CAP_ATTR,
    _INITIAL_BALL_SPEED_ATTR,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_CURRICULUM_ENV_ATTR = "_walk_kick_likelihood_speed_curriculum"
_STATE_SCHEMA_VERSION = 2


class MovingBallSpeedCurriculum(ManagerTermBase):
    """Raise the moving-ball distance and speed after stable frontier kicks.

    Each stage pairs one spawn-distance cap with one initial-speed cap.  Only
    episodes sampled at the current stage and in its upper speed quartile
    contribute to promotion.  Directions with non-zero reset probability must
    pass independently.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params

        self._stages = tuple(float(value) for value in params["stages_mps"])
        self._distance_max_stages = tuple(
            float(value) for value in params["spawn_distance_max_stages_m"]
        )
        self._distance_min = float(params["spawn_distance_min_m"])
        radius_stages = params.get("closest_approach_radius_max_stages_m")
        self._approach_radius_max_stages = (
            None
            if radius_stages is None
            else tuple(float(value) for value in radius_stages)
        )
        self._success_threshold = float(params["success_threshold"])
        self._frontier_fraction = float(params["frontier_fraction"])
        self._min_episodes_per_direction = int(params["min_episodes_per_direction"])
        self._required_consecutive_windows = int(params["required_consecutive_windows"])
        self._warmup_steps = int(params["warmup_steps"])
        self._reset_event_name = str(params.get("reset_event_name", "reset_ball"))

        reset_event_cfg = env.event_manager.get_term_cfg(self._reset_event_name)
        incoming_probability = (
            1.0
            if self._approach_radius_max_stages is not None
            else float(reset_event_cfg.params["incoming_probability"])
        )
        if not 0.0 <= incoming_probability <= 1.0:
            raise ValueError("reset incoming_probability must be in [0, 1]")
        self._require_incoming = incoming_probability > 0.0
        self._require_outgoing = incoming_probability < 1.0
        self._validate_config()

        self._current_stage = 0
        self._consecutive_passing_windows = 0
        self._incoming_eligible = 0
        self._incoming_successes = 0
        self._outgoing_eligible = 0
        self._outgoing_successes = 0
        self._last_incoming_success_rate = 0.0
        self._last_outgoing_success_rate = 0.0

        setattr(env, _CURRICULUM_ENV_ATTR, self)
        self._apply_stage()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stages_mps: Sequence[float],
        spawn_distance_max_stages_m: Sequence[float],
        spawn_distance_min_m: float,
        success_threshold: float,
        frontier_fraction: float,
        min_episodes_per_direction: int,
        required_consecutive_windows: int,
        warmup_steps: int,
        closest_approach_radius_max_stages_m: Sequence[float] | None = None,
        reset_event_name: str = "reset_ball",
    ) -> dict[str, float]:
        del (
            stages_mps,
            spawn_distance_max_stages_m,
            spawn_distance_min_m,
            closest_approach_radius_max_stages_m,
            success_threshold,
            frontier_fraction,
            min_episodes_per_direction,
            required_consecutive_windows,
            warmup_steps,
            reset_event_name,
        )

        warmup_complete = int(env.common_step_counter) >= self._warmup_steps
        promoted = False
        if warmup_complete:
            self._accumulate_completed_episodes(env, env_ids)
            promoted = self._evaluate_window_if_ready()

        return self._log_state(warmup_complete=warmup_complete, promoted=promoted)

    def state_dict(self) -> dict[str, Any]:
        """Return all mutable promotion state in a weights-only-safe payload."""
        state = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "stages_mps": list(self._stages),
            "spawn_distance_max_stages_m": list(self._distance_max_stages),
            "spawn_distance_min_m": self._distance_min,
            "stage_index": self._current_stage,
            "consecutive_passing_windows": self._consecutive_passing_windows,
            "incoming": {
                "eligible": self._incoming_eligible,
                "successes": self._incoming_successes,
            },
            "outgoing": {
                "eligible": self._outgoing_eligible,
                "successes": self._outgoing_successes,
            },
            "last_incoming_success_rate": self._last_incoming_success_rate,
            "last_outgoing_success_rate": self._last_outgoing_success_rate,
        }
        if self._approach_radius_max_stages is not None:
            state["closest_approach_radius_max_stages_m"] = list(
                self._approach_radius_max_stages
            )
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a checkpointed curriculum and immediately apply its stage."""
        if int(state.get("schema_version", -1)) != _STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported moving-ball speed curriculum state schema")

        saved_stages = tuple(float(value) for value in state.get("stages_mps", ()))
        if saved_stages != self._stages:
            raise ValueError(
                "Moving-ball speed curriculum stages do not match the checkpoint: "
                f"configured={self._stages}, checkpoint={saved_stages}"
            )
        saved_distance_stages = tuple(
            float(value)
            for value in state.get("spawn_distance_max_stages_m", ())
        )
        saved_distance_min = float(state.get("spawn_distance_min_m", -1.0))
        if (
            saved_distance_stages != self._distance_max_stages
            or saved_distance_min != self._distance_min
        ):
            raise ValueError(
                "Moving-ball distance curriculum does not match the checkpoint: "
                f"configured=({self._distance_min}, {self._distance_max_stages}), "
                f"checkpoint=({saved_distance_min}, {saved_distance_stages})"
            )
        saved_radius_stages = state.get("closest_approach_radius_max_stages_m")
        configured_radius_stages = self._approach_radius_max_stages
        if saved_radius_stages is None:
            if configured_radius_stages is not None:
                raise ValueError(
                    "Moving-ball closest-approach curriculum is missing from checkpoint"
                )
        elif tuple(float(value) for value in saved_radius_stages) != configured_radius_stages:
            raise ValueError(
                "Moving-ball closest-approach curriculum does not match checkpoint"
            )

        stage_index = int(state["stage_index"])
        if not 0 <= stage_index < len(self._stages):
            raise ValueError(f"Invalid moving-ball speed curriculum stage: {stage_index}")

        incoming_eligible, incoming_successes = self._load_counts(state, "incoming")
        outgoing_eligible, outgoing_successes = self._load_counts(state, "outgoing")
        consecutive_windows = int(state["consecutive_passing_windows"])
        if consecutive_windows < 0:
            raise ValueError("consecutive_passing_windows must be non-negative")

        self._current_stage = stage_index
        self._consecutive_passing_windows = consecutive_windows
        self._incoming_eligible = incoming_eligible
        self._incoming_successes = incoming_successes
        self._outgoing_eligible = outgoing_eligible
        self._outgoing_successes = outgoing_successes
        self._last_incoming_success_rate = float(state["last_incoming_success_rate"])
        self._last_outgoing_success_rate = float(state["last_outgoing_success_rate"])
        self._apply_stage()

    def _validate_config(self) -> None:
        if not self._stages or self._stages[0] != 0.0:
            raise ValueError("stages_mps must start at 0.0")
        if any(value < 0.0 for value in self._stages):
            raise ValueError("stages_mps must be non-negative")
        if any(left >= right for left, right in zip(self._stages, self._stages[1:])):
            raise ValueError("stages_mps must be strictly increasing")
        if len(self._distance_max_stages) != len(self._stages):
            raise ValueError(
                "spawn_distance_max_stages_m must match the number of speed stages"
            )
        if self._distance_min <= 0.0:
            raise ValueError("spawn_distance_min_m must be positive")
        if any(value <= self._distance_min for value in self._distance_max_stages):
            raise ValueError(
                "each spawn-distance maximum must exceed spawn_distance_min_m"
            )
        if any(
            left > right
            for left, right in zip(
                self._distance_max_stages,
                self._distance_max_stages[1:],
            )
        ):
            raise ValueError("spawn_distance_max_stages_m must be non-decreasing")
        if self._approach_radius_max_stages is not None:
            if len(self._approach_radius_max_stages) != len(self._stages):
                raise ValueError(
                    "closest_approach_radius_max_stages_m must match speed stages"
                )
            if any(value < 0.0 for value in self._approach_radius_max_stages):
                raise ValueError(
                    "closest_approach_radius_max_stages_m must be non-negative"
                )
            if any(
                left > right
                for left, right in zip(
                    self._approach_radius_max_stages,
                    self._approach_radius_max_stages[1:],
                )
            ):
                raise ValueError(
                    "closest_approach_radius_max_stages_m must be non-decreasing"
                )
        if not 0.0 <= self._success_threshold <= 1.0:
            raise ValueError("success_threshold must be in [0, 1]")
        if not 0.0 <= self._frontier_fraction <= 1.0:
            raise ValueError("frontier_fraction must be in [0, 1]")
        if self._min_episodes_per_direction <= 0:
            raise ValueError("min_episodes_per_direction must be positive")
        if self._required_consecutive_windows <= 0:
            raise ValueError("required_consecutive_windows must be positive")
        if self._warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

    def _accumulate_completed_episodes(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
    ) -> None:
        if isinstance(env_ids, slice):
            ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        if ids.numel() == 0:
            return

        metadata = (
            getattr(env, _INITIAL_BALL_SPEED_ATTR, None),
            getattr(env, _BALL_INCOMING_ATTR, None),
            getattr(env, _BALL_SPEED_CAP_ATTR, None),
            getattr(env, _BALL_RESET_METADATA_VALID_ATTR, None),
        )
        if any(value is None for value in metadata):
            return
        initial_speed, incoming, sampled_cap, metadata_valid = metadata

        episode_ended = env.termination_manager.dones[ids]
        current_cap = self._stages[self._current_stage]
        sampled_at_current_stage = torch.isclose(
            sampled_cap[ids],
            torch.full_like(sampled_cap[ids], current_cap),
            atol=1.0e-6,
            rtol=0.0,
        )
        frontier_min = self._frontier_fraction * current_cap
        at_frontier = initial_speed[ids] >= frontier_min - 1.0e-6
        eligible = episode_ended & metadata_valid[ids] & sampled_at_current_stage & at_frontier
        if not eligible.any():
            return

        kick_finished = env.termination_manager.get_term("kick_finished")[ids]
        fell = (
            env.termination_manager.get_term("base_contact")[ids]
            | env.termination_manager.get_term("base_height")[ids]
        )
        successful = kick_finished & (~fell)

        incoming_mask = eligible & incoming[ids]
        outgoing_mask = eligible & (~incoming[ids])
        self._incoming_eligible += int(incoming_mask.sum().item())
        self._incoming_successes += int((successful & incoming_mask).sum().item())
        self._outgoing_eligible += int(outgoing_mask.sum().item())
        self._outgoing_successes += int((successful & outgoing_mask).sum().item())

    def _evaluate_window_if_ready(self) -> bool:
        incoming_ready = (
            not self._require_incoming
            or self._incoming_eligible >= self._min_episodes_per_direction
        )
        outgoing_ready = (
            not self._require_outgoing
            or self._outgoing_eligible >= self._min_episodes_per_direction
        )
        if not incoming_ready or not outgoing_ready:
            return False

        if self._require_incoming:
            self._last_incoming_success_rate = (
                self._incoming_successes / self._incoming_eligible
            )
        if self._require_outgoing:
            self._last_outgoing_success_rate = (
                self._outgoing_successes / self._outgoing_eligible
            )
        passed = (
            (
                not self._require_incoming
                or self._last_incoming_success_rate >= self._success_threshold
            )
            and (
                not self._require_outgoing
                or self._last_outgoing_success_rate >= self._success_threshold
            )
        )
        if passed:
            self._consecutive_passing_windows = min(
                self._consecutive_passing_windows + 1,
                self._required_consecutive_windows,
            )
        else:
            self._consecutive_passing_windows = 0

        self._incoming_eligible = 0
        self._incoming_successes = 0
        self._outgoing_eligible = 0
        self._outgoing_successes = 0

        if (
            self._current_stage < len(self._stages) - 1
            and self._consecutive_passing_windows >= self._required_consecutive_windows
        ):
            self._current_stage += 1
            self._consecutive_passing_windows = 0
            self._apply_stage()
            return True
        return False

    def _apply_stage(self) -> None:
        speed_cap = self._stages[self._current_stage]
        distance_cap = self._distance_max_stages[self._current_stage]
        event_cfg = self._env.event_manager.get_term_cfg(self._reset_event_name)
        event_cfg.params["speed_range_mps"] = (0.0, speed_cap)
        distance_range = (self._distance_min, distance_cap)
        if self._approach_radius_max_stages is not None:
            radius_cap = self._approach_radius_max_stages[self._current_stage]
            event_cfg.params["path_length_range_m"] = distance_range
            event_cfg.params["closest_approach_radius_range_m"] = (0.0, radius_cap)
        else:
            event_cfg.params["incoming_spawn_distance_range_m"] = distance_range
            event_cfg.params["outgoing_spawn_distance_range_m"] = distance_range

    def _log_state(self, *, warmup_complete: bool, promoted: bool) -> dict[str, float]:
        current_cap = self._stages[self._current_stage]
        state = {
            "stage": float(self._current_stage),
            "speed_cap_mps": current_cap,
            "spawn_distance_max_m": self._distance_max_stages[self._current_stage],
            "frontier_min_speed_mps": self._frontier_fraction * current_cap,
            "incoming_success_rate": self._current_rate(
                self._incoming_successes,
                self._incoming_eligible,
                self._last_incoming_success_rate,
            ),
            "outgoing_success_rate": self._current_rate(
                self._outgoing_successes,
                self._outgoing_eligible,
                self._last_outgoing_success_rate,
            ),
            "incoming_frontier_episodes": float(self._incoming_eligible),
            "outgoing_frontier_episodes": float(self._outgoing_eligible),
            "consecutive_passing_windows": float(self._consecutive_passing_windows),
            "warmup_complete": float(warmup_complete),
            "promoted": float(promoted),
        }
        if self._approach_radius_max_stages is not None:
            state["closest_approach_radius_max_m"] = self._approach_radius_max_stages[
                self._current_stage
            ]
        return state

    @staticmethod
    def _current_rate(successes: int, eligible: int, fallback: float) -> float:
        if eligible == 0:
            return fallback
        return successes / eligible

    @staticmethod
    def _load_counts(state: Mapping[str, Any], direction: str) -> tuple[int, int]:
        counts = state[direction]
        if not isinstance(counts, Mapping):
            raise ValueError(f"Invalid {direction} curriculum counts")
        eligible = int(counts["eligible"])
        successes = int(counts["successes"])
        if eligible < 0 or successes < 0 or successes > eligible:
            raise ValueError(f"Invalid {direction} curriculum counts")
        return eligible, successes
