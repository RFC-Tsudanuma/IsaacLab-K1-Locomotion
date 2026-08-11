# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_long_pass_flag の mdp。

walk_kick の mdp をそのまま再輸出したうえで、このタスク固有のものを重ねる。
env cfg 側は ``from . import mdp`` だけで両方参照できる。
"""

from ...walk_kick.mdp import *  # noqa: F401,F403
from ...walk_kick.mdp import kick_state  # noqa: F401
from ...walk_kick.mdp.commands import (  # noqa: F401
    BallFollowVelocityCommand,
    BallFollowVelocityCommandCfg,
    KickDirectionCommand,
    KickDirectionCommandCfg,
)

from .actions import KickFlagAction, KickFlagActionCfg  # noqa: F401
from .commands import FLAG_ACTION_NAME, KickFlagDirectionCommand  # noqa: F401
from .observations import (  # noqa: F401
    kick_frozen_values,
    kick_latch_state,
    prev_action_of,
)
from .rewards import kick_flag  # noqa: F401
