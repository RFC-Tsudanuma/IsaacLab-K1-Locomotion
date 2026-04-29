# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym  # noqa: F401

# Import robot-specific configs to trigger gym.register() calls.
from .config import k1  # noqa: F401
