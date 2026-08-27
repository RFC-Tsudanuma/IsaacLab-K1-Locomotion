# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``walk_long_pass_history`` の左右対称写像。

PPO の mirror loss が促す条件は次の同変性である::

    policy(mirror(observation)) == mirror(policy(observation))

actor 観測は 5 フレームの履歴を項ごとに flatten した 223 次元。
それぞれの過去フレームに同じ写像を適用する。critic は 61 次元、
action は脚 12 関節で、どちらも同時に左右反転する。

``sole_pos`` という観測属性名は継承元の順序を保つために残しているが、
実際の値はボール 3D 位置。そのため通常の位置ベクトルとして
``(x, y, z) -> (x, -y, z)`` を適用できる。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..locomotion.mdp.symmetry import (
    _JOINT_MIRROR_SIGN as _LOCO_JOINT_MIRROR_SIGN,
    _LEFT_JOINT_IDX as _LOCO_LEFT_JOINT_IDX,
    _RIGHT_JOINT_IDX as _LOCO_RIGHT_JOINT_IDX,
)

if TYPE_CHECKING:
    from tensordict import TensorDict

    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "ACTION_DIM",
    "CRITIC_OBS_DIM",
    "HISTORY_LEN",
    "POLICY_OBS_DIM",
    "compute_symmetric_states",
    "mirror_last_dim",
]


def _joint_mirror_map() -> tuple[list[int], list[float]]:
    """locomotion と同じ脚関節の左右交換と符号を組み立てる。"""
    num_joints = len(_LOCO_JOINT_MIRROR_SIGN)
    perm = [-1] * num_joints
    for left, right in zip(_LOCO_LEFT_JOINT_IDX, _LOCO_RIGHT_JOINT_IDX):
        perm[left] = right
        perm[right] = left
    if -1 in perm:
        raise ValueError("locomotion の関節 mirror permutation が全関節を覆っていません。")
    return perm, [float(value) for value in _LOCO_JOINT_MIRROR_SIGN]


_JOINT_PERM, _JOINT_SIGN = _joint_mirror_map()

HISTORY_LEN = 5
"""mirror 写像が前提とする actor の履歴フレーム数。"""

# 局所写像は ``out[i] = input[perm[i]] * sign[i]`` で表す。
_SLOT_MAPS: dict[str, tuple[list[int], list[float]]] = {
    "vec3": ([0, 1, 2], [1.0, -1.0, 1.0]),
    "vec2": ([0, 1], [1.0, -1.0]),
    "ang_vel3": ([0, 1, 2], [-1.0, 1.0, -1.0]),
    "gait_phase2": ([0, 1], [-1.0, -1.0]),
    "scalar": ([0], [1.0]),
    "joints12": (_JOINT_PERM, _JOINT_SIGN),
}

# (ObservationTerm 名, 1 フレームの次元, 写像種別, 履歴数)
# ``sole_pos`` の値は walk_long_pass_history_env_cfg でボール位置に置換済み。
_POLICY_SLOTS: tuple[tuple[str, int, str, int], ...] = (
    ("projected_gravity", 3, "vec3", HISTORY_LEN),
    ("base_ang_vel", 3, "ang_vel3", HISTORY_LEN),
    ("sole_pos", 3, "vec3", 1),
    ("gait_phase", 2, "gait_phase2", 1),
    ("joint_pos", 12, "joints12", HISTORY_LEN),
    ("joint_vel", 12, "joints12", HISTORY_LEN),
    ("prev_joint_request", 12, "joints12", HISTORY_LEN),
    ("gait_phase_factor_offset", 1, "scalar", 1),
    ("kick_direction", 2, "vec2", 1),
    ("target_kick_velocity", 1, "scalar", 1),
    ("ball_vel", 2, "vec2", 1),
    ("prev_ball_pos", 2, "vec2", 1),
)

_CRITIC_SLOTS: tuple[tuple[str, int, str, int], ...] = (
    ("projected_gravity", 3, "vec3", 1),
    ("base_ang_vel", 3, "ang_vel3", 1),
    ("sole_pos", 3, "vec3", 1),
    ("gait_phase", 2, "gait_phase2", 1),
    ("joint_pos", 12, "joints12", 1),
    ("joint_vel", 12, "joints12", 1),
    ("prev_joint_request", 12, "joints12", 1),
    ("gait_phase_factor_offset", 1, "scalar", 1),
    ("kick_direction", 2, "vec2", 1),
    ("target_kick_velocity", 1, "scalar", 1),
    ("ball_vel", 2, "vec2", 1),
    ("prev_ball_pos", 2, "vec2", 1),
    ("base_lin_vel", 3, "vec3", 1),
    ("ball_pos_rel", 3, "vec3", 1),
)

_ACTION_SLOTS: tuple[tuple[str, int, str, int], ...] = (
    ("joint_pos", 12, "joints12", 1),
)


def _build_mirror_map(
    slots: tuple[tuple[str, int, str, int], ...],
) -> tuple[dict[str, int], list[int], list[float]]:
    """項テーブルから全体の offset / permutation / sign を作る。"""
    offsets: dict[str, int] = {}
    perm: list[int] = []
    sign: list[float] = []
    for name, dim, kind, repeats in slots:
        local_perm, local_sign = _SLOT_MAPS[kind]
        if len(local_perm) != dim or len(local_sign) != dim:
            raise ValueError(f"{name} ({dim}) と mirror 写像 {kind} の次元が一致しません。")
        offsets[name] = len(perm)
        for _ in range(repeats):
            base = len(perm)
            perm.extend(base + index for index in local_perm)
            sign.extend(local_sign)
    return offsets, perm, sign


_POLICY_OFFSETS, _POLICY_PERM, _POLICY_SIGN = _build_mirror_map(_POLICY_SLOTS)
_CRITIC_OFFSETS, _CRITIC_PERM, _CRITIC_SIGN = _build_mirror_map(_CRITIC_SLOTS)
_ACTION_OFFSETS, _ACTION_PERM, _ACTION_SIGN = _build_mirror_map(_ACTION_SLOTS)

POLICY_OBS_DIM = len(_POLICY_PERM)
CRITIC_OBS_DIM = len(_CRITIC_PERM)
ACTION_DIM = len(_ACTION_PERM)

_EXPECTED_POLICY_OFFSETS = {
    "projected_gravity": 0,
    "base_ang_vel": 15,
    "sole_pos": 30,
    "gait_phase": 33,
    "joint_pos": 35,
    "joint_vel": 95,
    "prev_joint_request": 155,
    "gait_phase_factor_offset": 215,
    "kick_direction": 216,
    "target_kick_velocity": 218,
    "ball_vel": 219,
    "prev_ball_pos": 221,
}
_EXPECTED_CRITIC_OFFSETS = {
    "projected_gravity": 0,
    "base_ang_vel": 3,
    "sole_pos": 6,
    "gait_phase": 9,
    "joint_pos": 11,
    "joint_vel": 23,
    "prev_joint_request": 35,
    "gait_phase_factor_offset": 47,
    "kick_direction": 48,
    "target_kick_velocity": 50,
    "ball_vel": 51,
    "prev_ball_pos": 53,
    "base_lin_vel": 55,
    "ball_pos_rel": 58,
}


def _check_layout() -> None:
    """観測順の手書き契約と機械計算結果を突き合わせる。"""
    if _POLICY_OFFSETS != _EXPECTED_POLICY_OFFSETS:
        raise ValueError(
            "walk_long_pass_history policy の観測順が想定と異なります: "
            f"{_POLICY_OFFSETS} != {_EXPECTED_POLICY_OFFSETS}"
        )
    if _CRITIC_OFFSETS != _EXPECTED_CRITIC_OFFSETS:
        raise ValueError(
            "walk_long_pass_history critic の観測順が想定と異なります: "
            f"{_CRITIC_OFFSETS} != {_EXPECTED_CRITIC_OFFSETS}"
        )
    for label, actual, expected in (
        ("policy", POLICY_OBS_DIM, 223),
        ("critic", CRITIC_OBS_DIM, 61),
        ("action", ACTION_DIM, 12),
    ):
        if actual != expected:
            raise ValueError(f"{label} 次元が {actual} です ({expected} を期待)。")


def _check_involution(perm: list[int], sign: list[float], label: str) -> None:
    """同じ mirror を 2 回適用すると元に戻ることを検査する。"""
    if sorted(perm) != list(range(len(perm))):
        raise ValueError(f"{label} mirror permutation が全要素の置換ではありません。")
    for index, source in enumerate(perm):
        if perm[source] != index or sign[index] * sign[source] != 1.0:
            raise ValueError(f"{label} mirror が対合ではありません (index={index})。")


_check_layout()
_check_involution(_POLICY_PERM, _POLICY_SIGN, "policy")
_check_involution(_CRITIC_PERM, _CRITIC_SIGN, "critic")
_check_involution(_ACTION_PERM, _ACTION_SIGN, "action")

_MIRROR_MAPS: dict[int, tuple[list[int], list[float]]] = {
    POLICY_OBS_DIM: (_POLICY_PERM, _POLICY_SIGN),
    CRITIC_OBS_DIM: (_CRITIC_PERM, _CRITIC_SIGN),
    ACTION_DIM: (_ACTION_PERM, _ACTION_SIGN),
}
if len(_MIRROR_MAPS) != 3:
    raise ValueError("policy / critic / action の次元が重複し、写像を選択できません。")

_MIRRORED_OBS_GROUPS = {"policy", "critic"}
_CONST_CACHE: dict = {}


def _mirror_consts(
    dim: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (dim, device, dtype)
    consts = _CONST_CACHE.get(key)
    if consts is None:
        perm, sign = _MIRROR_MAPS[dim]
        consts = (
            torch.tensor(perm, device=device, dtype=torch.long),
            torch.tensor(sign, device=device, dtype=dtype),
        )
        _CONST_CACHE[key] = consts
    return consts


def mirror_last_dim(value: torch.Tensor) -> torch.Tensor:
    """最終軸が 223 / 61 / 12 のテンソルを左右反転する。"""
    dim = int(value.shape[-1])
    if dim not in _MIRROR_MAPS:
        raise ValueError(
            f"walk_long_pass_history.symmetry: 最終軸 {dim} の mirror 写像はありません"
            f" (対応: {sorted(_MIRROR_MAPS)})。"
        )
    perm, sign = _mirror_consts(dim, value.device, value.dtype)
    return value[..., perm] * sign


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """rsl-rl 用に ``[元バッチ, mirror バッチ]`` を返す。"""
    del env

    if obs is not None:
        groups = set(obs.keys())
        unknown = groups - _MIRRORED_OBS_GROUPS
        if unknown:
            raise ValueError(
                "walk_long_pass_history.symmetry: mirror 規約のない観測グループが"
                f" あります: {sorted(unknown)}"
            )
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for group in groups:
            obs_aug[group][:batch_size] = obs[group]
            obs_aug[group][batch_size:] = mirror_last_dim(obs[group])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.empty(
            (batch_size * 2, actions.shape[1]),
            device=actions.device,
            dtype=actions.dtype,
        )
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = mirror_last_dim(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
