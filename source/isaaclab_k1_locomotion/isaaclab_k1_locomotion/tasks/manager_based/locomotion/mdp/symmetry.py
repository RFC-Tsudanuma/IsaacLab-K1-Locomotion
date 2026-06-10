# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ロボットの左右対称性 (sagittal plane mirror) を表現する関数。

rsl-rl の symmetry 機能 (``RslRlSymmetryCfg``) から呼ばれる
``data_augmentation_func`` を提供する。FlatEnvCfg では mirror loss のみを使い、
data augmentation は使わない設定を想定している (ただし関数自体は両用途で動作する)。

mirror loss が有効な場合、PPO は以下を最小化する::

    || policy(mirror(obs)) - mirror(policy(obs)) ||^2

すなわち「観測を左右反転して与えたら、行動も左右反転して返す」対称な方策を促す。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict

    from isaaclab.envs import ManagerBasedRLEnv

# import 可能な関数を明示
__all__ = ["compute_symmetric_states"]


# ---------------------------------------------------------------------------
# 関節の対応関係
# ---------------------------------------------------------------------------
# JOINT_NAMES_K1 の並び (12関節):
#    0 Left_Hip_Pitch     6 Right_Hip_Pitch
#    1 Left_Hip_Roll      7 Right_Hip_Roll
#    2 Left_Hip_Yaw       8 Right_Hip_Yaw
#    3 Left_Knee_Pitch    9 Right_Knee_Pitch
#    4 Left_Ankle_Pitch  10 Right_Ankle_Pitch
#    5 Left_Ankle_Roll   11 Right_Ankle_Roll
_LEFT_JOINT_IDX = [0, 1, 2, 3, 4, 5]
_RIGHT_JOINT_IDX = [6, 7, 8, 9, 10, 11]
# 左右を入れ替えた後に掛ける符号。
# 矢状面 (前後・上下) 内で動く pitch / knee 関節はそのまま、
# 面外成分を持つ roll / yaw 関節は符号を反転する。
# (既存の rewards.joint_mirror_symmetry と同じ規約: Hip_Roll/Hip_Yaw は逆符号)
_JOINT_MIRROR_SIGN = [
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0,  # 左: pitch, roll, yaw, knee, ankle_pitch, ankle_roll
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0,  # 右: 同上
]

# 観測ベクトル内の各項の区間 (K1PolicyCfg と一致させること)。
#   base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
#   + joint_pos(12) + joint_vel(12) + last_action(12) + gait_phase(4) = 49
_ANG_VEL_SLICE = slice(0, 3)
_PROJ_GRAVITY_SLICE = slice(3, 6)
_VEL_CMD_SLICE = slice(6, 9)
_JOINT_POS_SLICE = slice(9, 21)
_JOINT_VEL_SLICE = slice(21, 33)
_LAST_ACTION_SLICE = slice(33, 45)
_GAIT_PHASE_SLICE = slice(45, 49)
_POLICY_OBS_DIM = 49

# gait_phase = [sin(left), cos(left), sin(right), cos(right)] なので
# 左右入れ替えはこの並べ替えで実現できる (45+[2,3,0,1])。
_GAIT_PHASE_SWAP = [47, 48, 45, 46]


def _mirror_joints(joint_data: torch.Tensor) -> torch.Tensor:
    """関節ごとの量 (..., 12) を左右反転する。

    左脚と右脚を入れ替えたうえで、roll / yaw 関節の符号を反転する。
    joint_pos / joint_vel / action のいずれにも使える。
    """
    out = torch.empty_like(joint_data)
    out[..., _LEFT_JOINT_IDX] = joint_data[..., _RIGHT_JOINT_IDX]
    out[..., _RIGHT_JOINT_IDX] = joint_data[..., _LEFT_JOINT_IDX]
    sign = torch.tensor(_JOINT_MIRROR_SIGN, device=joint_data.device, dtype=joint_data.dtype)
    return out * sign


def _mirror_policy_obs(obs: torch.Tensor) -> torch.Tensor:
    """ポリシー観測 (N, 49) を矢状面に対して左右反転する。"""
    if obs.shape[-1] != _POLICY_OBS_DIM:
        raise ValueError(
            f"symmetry: ポリシー観測の次元が想定 ({_POLICY_OBS_DIM}) と異なります: {obs.shape[-1]}。"
            " mdp/symmetry.py のスライス定義を K1PolicyCfg に合わせて更新してください。"
        )

    out = obs.clone()
    device, dtype = out.device, out.dtype

    # base_ang_vel: roll(x), yaw(z) を反転
    out[:, _ANG_VEL_SLICE] *= torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype)
    # projected_gravity: 横方向(y) を反転
    out[:, _PROJ_GRAVITY_SLICE] *= torch.tensor([1.0, -1.0, 1.0], device=device, dtype=dtype)
    # velocity_commands: lin_vel_y, ang_vel_z を反転
    out[:, _VEL_CMD_SLICE] *= torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)
    # 関節量
    out[:, _JOINT_POS_SLICE] = _mirror_joints(obs[:, _JOINT_POS_SLICE])
    out[:, _JOINT_VEL_SLICE] = _mirror_joints(obs[:, _JOINT_VEL_SLICE])
    out[:, _LAST_ACTION_SLICE] = _mirror_joints(obs[:, _LAST_ACTION_SLICE])
    # gait_phase: 左右の位相を入れ替え
    out[:, _GAIT_PHASE_SLICE] = obs[:, _GAIT_PHASE_SWAP]

    return out


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """観測・行動に左右対称変換を適用して拡張する (rsl-rl 用)。

    返すバッチは ``[元のサンプル, 左右反転したサンプル]`` の順に連結され、
    バッチサイズが 2 倍になる。rsl-rl の PPO は

    * mirror loss: ``policy(mirror(obs)) ≈ mirror(policy(obs))`` を促す MSE 損失
    * data augmentation: 反転サンプルを学習バッチに追加

    のいずれ (または両方) にこの関数を使う。本リポジトリの FlatEnvCfg では
    mirror loss のみを有効化している。

    Args:
        env: 環境インスタンス (本関数では未使用だが rsl-rl の規約に合わせて受け取る)。
        obs: 観測の TensorDict。None の場合は観測を変換しない。
        actions: 行動テンソル (N, 12)。None の場合は行動を変換しない。

    Returns:
        ``(obs_aug, actions_aug)``。入力が None だった側は None を返す。
    """
    # -- 観測
    if obs is not None:
        batch_size = obs.batch_size[0]
        # 左右対称は1種類なのでバッチを2倍に拡張する
        obs_aug = obs.repeat(2)
        # policy グループのみ反転する。critic グループ (特権情報) は mirror loss の
        # 計算経路 (act_inference) では参照されないため、複製したまま据え置く。
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _mirror_policy_obs(obs["policy"])
    else:
        obs_aug = None

    # -- 行動
    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _mirror_joints(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
