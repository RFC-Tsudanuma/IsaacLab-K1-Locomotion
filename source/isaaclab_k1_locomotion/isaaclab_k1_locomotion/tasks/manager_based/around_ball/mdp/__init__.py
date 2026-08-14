# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール回り込み (around_ball) タスク専用の MDP 項。"""

from .curriculums import modify_kick_angle_range
from .events import (
    reset_ball_in_front_cone,
    reset_ball_last_seen,
    reset_ball_perception,
    reset_base_forward_velocity,
)
from .terminations import ball_kicked
from .observations import (
    ball_in_fov,
    ball_offset_and_bearing,
    ball_pos_rel_fov,
    ball_pos_rel_perceived,
    high_action_phase_obs,
    kick_direction_b_perceived,
)
from .rewards import (
    aligned_pose_hold,
    ball_disturbance_when_misaligned,
    ball_out_of_fov,
    charge_to_ball_when_aligned,
    misaligned_ball_proximity,
    standoff_point_progress,
)
