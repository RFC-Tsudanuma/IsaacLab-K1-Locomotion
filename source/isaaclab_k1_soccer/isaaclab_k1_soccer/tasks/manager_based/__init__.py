# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym  # noqa: F401

# Import task packages to trigger gym.register() calls.
from . import locomotion  # noqa: F401
from . import navigation  # noqa: F401
from . import soccer_dribble  # noqa: F401
