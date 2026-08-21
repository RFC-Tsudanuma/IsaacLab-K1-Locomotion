"""Global map-target command for the walk-kick likelihood task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from ...walk_kick.mdp.commands import KickDirectionCommand, KickDirectionCommandCfg
from ...walk_kick.mdp.kick_state import request_kick_state_reset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GlobalTargetKickCommand(KickDirectionCommand):
    """Fixed global target position plus an independent target kick speed.

    The public command layout is ``[target_x_w, target_y_w, target_speed]``.
    At each episode reset, a heading offset, target distance, and target speed
    are sampled independently.  Converting the heading and distance into a
    global target is deferred until the post-reset robot and ball state is
    first consumed.  The resulting target position then stays fixed for the
    episode; only the unit direction from the current ball to that target is
    recomputed.

    ``direction_from_ball`` is the protocol used by the shared WalkKick
    kick-state code.  Keeping the protocol on the command term lets legacy
    ``KickDirectionCommand`` terms retain their ``[sin, cos, speed]`` layout.
    """

    cfg: "GlobalTargetKickCommandCfg"

    def __init__(self, cfg: "GlobalTargetKickCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ensure_target_buffers()

    def _ensure_target_buffers(self) -> None:
        """Create buffers lazily in case the base constructor resamples."""
        if hasattr(self, "_target_pending"):
            return
        self._pending_heading_offset = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=self.vel_command_b.dtype,
        )
        self._pending_target_distance = torch.zeros_like(self._pending_heading_offset)
        self._target_pending = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )

    def reset(self, env_ids=None):
        """Resample the episode command and invalidate cached rows after reset."""
        metrics = super().reset(env_ids)
        reset_ids = slice(None) if env_ids is None else env_ids
        request_kick_state_reset(self._env, reset_ids)
        return metrics

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Sample episode parameters, deferring their pose-dependent projection."""
        self._ensure_target_buffers()
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        count = env_ids.numel()
        if count == 0:
            return

        heading_low, heading_high = self.cfg.ranges.heading
        distance_low, distance_high = self.cfg.target_distance_range
        speed_low, speed_high = self.cfg.target_speed_range

        self._pending_heading_offset[env_ids] = torch.empty(
            count,
            device=self.device,
            dtype=self.vel_command_b.dtype,
        ).uniform_(heading_low, heading_high)
        self._pending_target_distance[env_ids] = torch.empty(
            count,
            device=self.device,
            dtype=self.vel_command_b.dtype,
        ).uniform_(distance_low, distance_high)
        self.vel_command_b[env_ids, 2] = torch.empty(
            count,
            device=self.device,
            dtype=self.vel_command_b.dtype,
        ).uniform_(speed_low, speed_high)

        # Do not derive target XY here.  CommandManager.reset can run while
        # reset state is still being propagated to asset data buffers.  The
        # first resolver/property/update call observes the post-reset state.
        self.vel_command_b[env_ids, :2] = 0.0
        self._target_pending[env_ids] = True

    def _finalize_pending_targets(self) -> None:
        self._ensure_target_buffers()
        pending = self._target_pending
        if not bool(pending.any()):
            return

        robot = self._env.scene[self.cfg.asset_name]
        ball = self._env.scene[self.cfg.ball_asset_name]
        robot_quat = robot.data.root_quat_w[pending]
        w, x, y, z = robot_quat.unbind(dim=-1)
        robot_yaw = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y.square() + z.square()),
        )
        target_yaw = robot_yaw + self._pending_heading_offset[pending]
        target_direction = torch.stack(
            (torch.cos(target_yaw), torch.sin(target_yaw)),
            dim=-1,
        )
        distance = self._pending_target_distance[pending].unsqueeze(-1)
        self.vel_command_b[pending, :2] = (
            ball.data.root_pos_w[pending, :2] + distance * target_direction
        )
        self._target_pending[pending] = False

    @property
    def command(self) -> torch.Tensor:
        """Return ``[target_x_w, target_y_w, target_speed]`` for every env."""
        if hasattr(self, "_target_pending"):
            self._finalize_pending_targets()
        return self.vel_command_b

    @property
    def target_position_w(self) -> torch.Tensor:
        """Return the fixed episode target XY in world/map coordinates."""
        self._finalize_pending_targets()
        return self.vel_command_b[:, :2]

    def direction_from_ball(self, ball_position_w: torch.Tensor) -> torch.Tensor:
        """Return current true ball-to-fixed-target unit directions in world XY."""
        target_delta = self.target_position_w - ball_position_w[:, :2]
        return target_delta / target_delta.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)

    def _update_command(self) -> None:
        # Target finalization is intentionally demand-driven through
        # target_position_w/direction_from_ball.  CommandManager may invoke this
        # hook before reset asset state has propagated to its data buffers.
        pass

    def _debug_vis_callback(self, _event) -> None:
        if not self.robot.is_initialized:
            return

        ball = self._env.scene[self.cfg.ball_asset_name]
        marker_position = ball.data.root_pos_w.clone()
        marker_position[:, 2] += 0.5
        direction = self.direction_from_ball(ball.data.root_pos_w[:, :2])
        theta = torch.atan2(direction[:, 1], direction[:, 0])
        zeros = torch.zeros_like(theta)
        marker_orientation = math_utils.quat_from_euler_xyz(zeros, zeros, theta)

        default_scale = self.kick_dir_visualizer.cfg.markers["arrow"].scale
        marker_scale = torch.tensor(
            default_scale,
            device=self.device,
            dtype=marker_position.dtype,
        ).repeat(self.num_envs, 1)
        self.kick_dir_visualizer.visualize(
            marker_position,
            marker_orientation,
            marker_scale,
        )


@configclass
class GlobalTargetKickCommandCfg(KickDirectionCommandCfg):
    """Configuration for :class:`GlobalTargetKickCommand`."""

    class_type: type = GlobalTargetKickCommand

    ball_asset_name: str = "soccer_ball"
    """Scene asset that defines the origin of the sampled global target."""

    target_distance_range: tuple[float, float] = (5.0, 12.0)
    """Distance from the post-reset ball to the fixed global target [m]."""
