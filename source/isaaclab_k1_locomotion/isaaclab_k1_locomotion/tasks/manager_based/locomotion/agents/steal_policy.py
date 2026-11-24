# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール奪取方策 (Isaac-Steal-K1) とその PPO (2026-11-03)。

Block (agents/block_policy.py) からボール encoder・latent 蒸留を外した構成:

* 歩行 expert: `WaypointActorCritic` と同じく frozen_policies/best_walk を凍結して内包 (事前分布)。
* 方策の出力 (15 次元) = ``[歩行コマンド (3), 関節残差 (12)]``。環境へは
  ``[expert の関節指令 + residual_scale × 残差 (12), コマンド (3), 残差 (12)]`` の 27 次元を渡す。
  残差 0 なら expert の歩行そのもの。
* actor / critic は MLP。入力は ``steal_policy`` / ``steal_critic`` 観測グループ (内界 + ボール + 障害物)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from rsl_rl.networks import MLP
from rsl_rl.storage import RolloutStorage

from ..mdp.block import JOINT_DIM
from .block_policy import BlockActorCritic, RslRlBlockActorCriticCfg, RslRlBlockPpoAlgorithmCfg
from .waypoint_policy import RslRlWaypointActorCriticCfg, WaypointActorCritic, WaypointPPO


class StealActorCritic(WaypointActorCritic):
    """歩行コマンド (3) + 関節残差 (12) を出す MLP 方策 + 凍結歩行 expert。"""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,  # 環境のアクション次元 (12 + 3 + 12)。方策の出力次元ではない。
        residual_scale: float = 0.5,
        lateral_x_knee: float = 1.0,
        lateral_x_min: float = 0.3,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        self.num_policy_actions = 3 + JOINT_DIM
        # 基底は「num_actions − 3」を expert の出力次元にするので 15 を渡す (expert は 12 出力)。
        # normalizer / 凍結 expert / command() / expert_actions() を流用し、actor・critic・std は置き換える。
        super().__init__(
            obs,
            obs_groups,
            self.num_policy_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        self.num_env_actions = int(num_actions)
        self.residual_scale = float(residual_scale)
        self.lateral_x_knee, self.lateral_x_min = float(lateral_x_knee), float(lateral_x_min)
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        self.actor = MLP(num_actor_obs, self.num_policy_actions, list(actor_hidden_dims), activation)
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(self.num_policy_actions))
        else:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_policy_actions)))
        print(f"Actor (steal): {self.actor}")
        print(f"Critic (steal): {self.critic}")

    def command(self, raw: torch.Tensor) -> torch.Tensor:
        """クランプ + デッドゾーン (基底) に加え、歩行 expert の指令領域に合わせて |vy| > lateral_x_knee では
        |vx| の上限を |vy| = 上限で lateral_x_min まで線形に絞る (best_walk model_56444: 高速の横移動はほぼ純横)。
        BlockActorCritic.command と同じ。vy の上限が膝以下 (旧 expert の 0.9) なら何もしない。"""
        cmd = super().command(raw)
        if float(self.command_limits[1]) <= self.lateral_x_knee:
            return cmd
        t = ((cmd[:, 1:2].abs() - self.lateral_x_knee) / (self.command_limits[1] - self.lateral_x_knee)).clamp(0.0, 1.0)
        x_lim = self.command_limits[0] - t * (self.command_limits[0] - self.lateral_x_min)
        return torch.cat([torch.maximum(torch.minimum(cmd[:, :1], x_lim), -x_lim), cmd[:, 1:]], dim=-1)

    def env_actions(self, obs: TensorDict, raw: torch.Tensor) -> torch.Tensor:
        """環境に渡す 27 次元 ``[expert + 残差 (12), コマンド (3), 残差 (12)]``。

        観測に ``steal_ctrl`` = [w, 歩行コマンド (3)] があれば、ノードの steal_step と同じく歩行と混合する:
        コマンド = (1 − w) × 歩行コマンド + w × 方策、残差 = w × 方策 (試合条件の歩行中開始・途中解除)。"""
        cmd = self.command(raw[:, :3])
        res = self.residual_scale * raw[:, 3:]
        if "steal_ctrl" in obs.keys():
            ctrl = obs["steal_ctrl"]
            w = ctrl[:, :1]
            cmd = (1.0 - w) * ctrl[:, 1:4] + w * cmd
            res = w * res
        return torch.cat([self.expert_actions(obs, cmd) + res, cmd, res], dim=-1)

    # 凍結 expert は checkpoint に含めない (block_policy.py / gating_policy.py と同じ対処)
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        for k in [k for k in sd if k.startswith("expert.")]:
            del sd[k]
        return sd

    def load_state_dict(self, state_dict, strict=True):
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("expert.")}
        missing = [k for k in self.state_dict() if k not in filtered]
        if missing:
            raise KeyError(f"checkpoint に方策側のキーが無い: {missing[:4]} ...")
        return super().load_state_dict(filtered, strict=False)


class StealHistActorCritic(BlockActorCritic):
    """Block 型の奪取方策 (Isaac-StealHist-K1, 2026-11-22)。BlockActorCritic と同じ構造で、
    actor は ``[steal_policy (69), 観測履歴 (T × 8)]``、critic は ``[steal_critic (79), 真値履歴 (T × 6)]``。
    履歴は causal transformer encoder で latent にし、BlockPPO が actor 側 encoder を真値側へ蒸留する。
    歩行との混合は StealActorCritic と同じく観測 ``steal_ctrl`` から。"""

    def env_actions(self, obs: TensorDict, raw: torch.Tensor) -> torch.Tensor:
        if "steal_ctrl" in obs.keys():
            ctrl = obs["steal_ctrl"]
            return super().env_actions(obs, raw, blend=ctrl[:, 0], cmd_ext=ctrl[:, 1:4])
        return super().env_actions(obs, raw)


def steal_mirror_policy(pol: StealHistActorCritic, raw_env, mode: str = "full", hist: int = 30):
    """目標方向がロボット→ボールの線の左にあるとき左右反転して動かす推論関数 (obs → env action) を返す (eval_steal.py / play.py)。
    step = 毎 tick 判定、latch = 作動直後の判定を作動が切れるまで保つ (ノードが目標方向を固定するのと同じ)、
    full = latch に加えて歩行 expert も反転する (制御全体を鏡像にする。expert は左右対称でないため。ノードの steal_mirror と同じ)。
    full_opp = full と同じだが、判定を「目標方向がボール→敵の線の左にあるとき」に変える (立ち位置が必ず敵の反対側になる。評価用)。
    戻り値の ``.left`` は今反転している env (N,)、``.age`` は作動してからの tick 数 (0 にすると判定をやり直す)。"""
    from ..history_layout import POLICY_TERM_SPECS
    from ..mdp.symmetry import _mirror_history_group, _mirror_joints
    from .block_nets import steal_mirror, steal_symmetric, steal_target_left

    mir = [t.to(raw_env.device) for t in steal_mirror(hist)]
    mir_left = torch.zeros(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)
    mir_age = torch.zeros(raw_env.num_envs, dtype=torch.long, device=raw_env.device)  # 奪取が作動してからの tick 数
    expert_actions = pol.expert_actions

    def expert_mirrored(o, cmd):
        om = {k: v for k, v in o.items()}
        om["policy"] = _mirror_history_group(o["policy"], "policy", POLICY_TERM_SPECS)
        am = _mirror_joints(expert_actions(TensorDict(om, batch_size=o.batch_size), cmd * torch.tensor([1.0, -1.0, -1.0], device=cmd.device)))
        return torch.where(mir_left.unsqueeze(1), am, expert_actions(o, cmd))

    if mode in ("full", "full_opp"):
        pol.expert_actions = expert_mirrored
    opp_left = lambda x: (x[:, 65] - x[:, 61]) * x[:, 68] - (x[:, 66] - x[:, 62]) * x[:, 67] > 0.0  # 目標がボール→敵の線の左
    target_left = opp_left if mode == "full_opp" else steal_target_left

    def policy(o):
        x = pol.get_actor_obs(o)
        w_on = o["steal_ctrl"][:, 0] > 0.0
        # 作動していない間は反転しない。--node_dir の目標方向は 1 tick 遅れて観測に入るので、作動 2 tick 目までは判定を更新する
        mir_age[:] = torch.where(w_on & (raw_env.episode_length_buf > 0), mir_age + 1, 0)
        hold = (mir_age > 2) & (mode != "step")
        mir_left[:] = torch.where(hold, mir_left, target_left(x) & w_on)
        return pol.env_actions(o, steal_symmetric(lambda v: pol.actor(pol.actor_obs_normalizer(v)), x, mir, mir_left))

    policy.left, policy.age = mir_left, mir_age
    return policy


class StealPPO(WaypointPPO):
    """15 次元 (コマンド + 残差) を PPO で学習する (storage の action 次元だけ WaypointPPO と異なる)。"""

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.storage = RolloutStorage(
            training_type, num_envs, num_transitions_per_env, self._slim(obs), [self.policy.num_policy_actions], self.device
        )


@configclass
class RslRlStealActorCriticCfg(RslRlWaypointActorCriticCfg):
    """`StealActorCritic` 用の設定 (expert の checkpoint / アーキテクチャ / command_limits は waypoint と共通)。"""

    class_name: str = "StealActorCritic"

    init_noise_std: float = 0.5
    actor_hidden_dims: list[int] = [256, 128]
    critic_hidden_dims: list[int] = [256, 128]

    residual_scale: float = 0.5
    """方策の残差出力 → 関節指令 (raw action 単位) の倍率。Block で 1.0 は転倒が増えて逆効果だった。"""

    command_limits: tuple[float, float, float] = (2.0, 1.9, 2.0)
    """歩行コマンドの上限 (vx, vy, ωz)。2026-11-18 に best_walk model_56444 (y ±1.9) に合わせて vy 0.9 → 1.9
    (Block と同じ)。waypoint 側の既定が変わっても Steal はここで固定する。"""
    lateral_x_knee: float = 1.0
    lateral_x_min: float = 0.3
    """|vy| > lateral_x_knee で |vx| 上限を |vy| = command_limits[1] のとき lateral_x_min まで線形に絞る
    (歩行 expert の指令領域、flat_env_cfg の K1FlatStancePlaneWideCfg / Block と同じ)。"""


@configclass
class RslRlStealPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "StealPPO"


@configclass
class RslRlStealHistActorCriticCfg(RslRlBlockActorCriticCfg):
    class_name: str = "StealHistActorCritic"
    ball_history: int = 30
    """環境側 StealCommandCfg.obs_history と一致させる。"""
    command_limits: tuple[float, float, float] = (2.0, 1.9, 2.0)


# OnPolicyRunner は eval(class_name) を on_policy_runner モジュールの名前空間で評価する。
import rsl_rl.runners.on_policy_runner as _rsl_rl_on_policy_runner  # noqa: E402

_rsl_rl_on_policy_runner.StealActorCritic = StealActorCritic
_rsl_rl_on_policy_runner.StealPPO = StealPPO
_rsl_rl_on_policy_runner.StealHistActorCritic = StealHistActorCritic
