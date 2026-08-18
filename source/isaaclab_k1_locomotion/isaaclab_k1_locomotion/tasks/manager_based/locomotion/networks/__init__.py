# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""このリポジトリ独自の rsl_rl 用ネットワーク。"""

import rsl_rl.runners.on_policy_runner as _on_policy_runner

from .actor_critic_history_cnn import (  # noqa: F401
    ActorCriticHistoryCNN,
    HistoryCnnActor,
    ObsHistoryEncoder,
    RslRlPpoActorCriticHistoryCnnCfg,
    export_history_policy_as_jit,
    export_history_policy_as_onnx,
    remap_single_frame_actor,
    remap_widened_obs_actor,
)

# OnPolicyRunner._construct_algorithm は policy の class_name を
# `eval(class_name)` で解決する (= on_policy_runner モジュールの名前空間を引く)。
# 自作クラスはそこに載っていないので、import 時に登録しておく。
_on_policy_runner.ActorCriticHistoryCNN = ActorCriticHistoryCNN
