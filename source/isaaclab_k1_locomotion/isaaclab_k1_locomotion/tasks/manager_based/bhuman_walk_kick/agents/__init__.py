# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import rsl_rl.runners.on_policy_runner as _on_policy_runner

from .ppo_actor_freeze import PPOActorFreeze

# OnPolicyRunner は algorithm の class_name を on_policy_runner モジュールの名前空間で eval する
_on_policy_runner.PPOActorFreeze = PPOActorFreeze
