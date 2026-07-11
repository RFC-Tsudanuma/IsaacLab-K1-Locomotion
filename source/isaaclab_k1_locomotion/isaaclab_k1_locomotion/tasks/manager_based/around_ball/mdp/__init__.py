# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール回り込み (around_ball) タスク専用の MDP 項。"""

from .curriculums import modify_kick_angle_range
from .events import relocate_ball_after_kick, reset_ball_in_front_cone, reset_ball_last_seen
from .observations import (
    ball_in_fov,
    ball_offset_and_bearing,
    ball_pos_rel_fov,
    high_action_phase_obs,
)
from .rewards import (
    aligned_pose_hold,
    ball_disturbance_when_misaligned,
    ball_out_of_fov,
    charge_to_ball_when_aligned,
    misaligned_ball_proximity,
    standoff_point_progress,
)
