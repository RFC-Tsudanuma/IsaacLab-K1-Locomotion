"""Reuse the deployed-observer contract; no kicking reward or policy is run."""
import torch

from ..direct_kick.direct_kicking_logic import DirectKickingLogic
from ..direct_kick.math_xyzw import quat_rotate_inverse
from ..direct_kick.utils import apply_randomization
from ..direct_kick.direct_kicking_observation import expected_direct_kicking_observation_size


class BallBelief(DirectKickingLogic):
    """Use only DirectKickingLogic's existing perception methods and buffers.

    State inputs are environment-local SI units with xyzw quaternions. No
    simulator or ROS is required, which also permits CPU parity tests.
    """

    def __init__(self, cfg, count, dt, device):
        self.task_cfg = cfg
        self.direct_cfg = cfg["direct_kicking"]
        self.num_envs = count
        # The reused initializer validates its source task's outer shape. This
        # wrapper exports only its 276 belief features, not that task's policy.
        self.num_obs = expected_direct_kicking_observation_size(
            len(self.direct_cfg["observation"]["prediction_horizons_s"]))
        self.device = torch.device(device)
        self.dt = dt
        self.common_step_counter = 0
        self.root_states = torch.zeros(count, 2, 13, device=self.device)
        self.root_states[..., 6] = 1.
        self._init_direct_perception_buffers()
        self.visible = torch.zeros(count, device=self.device, dtype=torch.bool)

    def reset(self, ids, root_states):
        self.root_states.copy_(root_states)
        self._reset_direct_perception(ids)
        self.visible[ids] = False

    def observe(self, root_states, step, advance=True):
        self.root_states.copy_(root_states)
        self.common_step_counter = int(step)
        if advance:
            old_stamps = self.ball_filter.stamp_ns.clone()
            self._update_ego_motion_sensor_noise()
            self._update_observed_base_pose()
            self._advance_direct_perception()
            camera_frames = self.ball_filter.stamp_ns != old_stamps
            # Hold the accepted-detection flag between camera frames. A policy
            # step with no camera frame does not itself mean a missed detection.
            self.visible[camera_frames] = self.measurement_updated[camera_frames]
        else:
            self.perception_just_reset.zero_()
        angular = quat_rotate_inverse(root_states[:, 0, 3:7], root_states[:, 0, 10:13])
        angular = apply_randomization(angular, self.task_cfg["noise"].get("ang_vel"))
        angular[:, 2] += self.ego_yaw_rate_bias + self.ego_yaw_rate_drift
        velocity_xy = (self._world_to_local_rotation(self.observed_base_yaw)
                       @ root_states[:, 0, 7:9, None]).squeeze(-1)
        velocity = torch.cat((velocity_xy, root_states[:, 0, 9:10]), dim=-1)
        velocity = apply_randomization(velocity, self.task_cfg["noise"].get("lin_vel"))
        velocity[:, :2] += self.ego_velocity_bias + self.ego_velocity_drift
        return self._belief_observation(velocity, angular), angular
