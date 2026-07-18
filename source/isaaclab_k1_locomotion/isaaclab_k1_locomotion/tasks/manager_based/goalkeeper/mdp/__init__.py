# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (goalkeeper) タスク専用の MDP 項。

frozen 歩行ポリシー連携 (high_action_phase_obs / last_high_action /
high_action_*_l2 ペナルティ) は around_ball / locomotion から再利用する。
"""

# around_ball / locomotion と共通の項 (再エクスポート)
from ...around_ball.mdp.observations import high_action_phase_obs
from ...locomotion.mdp.events import reset_prev_high_action
from ...locomotion.mdp.observations import last_high_action
from ...locomotion.mdp.rewards import (
    com_jerk_l2,
    high_action_rate_l2,
    high_action_smoothness_l2,
)

# goalkeeper 固有の項
from .curriculums import adaptive_ball_speed
from .events import (
    reset_ball_shot,
    reset_gk_buffers,
    reset_stage1_target_and_park,
    stage1_target_tick,
)
from .observations import (
    compute_target_y,
    gk_ball_active,
    gk_ball_pos_rel,
    gk_ball_vel,
    gk_buffers,
    gk_self_state,
    gk_target_y,
)
from .rewards import (
    face_field,
    hold_at_target,
    return_to_center_after_save,
    save_touch_bonus,
    stay_on_goal_line,
    target_reach_velocity,
    track_target_y,
)
from .terminations import (
    goal_conceded,
    robot_out_of_bounds,
    save_success,
    update_save_state,
)
