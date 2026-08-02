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

from ..history_layout import CRITIC_TERM_SPECS, HISTORY_LENGTH, LATEST_FRAME_GROUP, POLICY_TERM_SPECS

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


# 反転に使う符号定数を (device, dtype) ごとに一度だけ生成してキャッシュする。
# 旧実装は呼び出しのたびに torch.tensor([...], device=...) で Python リストから
# テンソルを生成し host→device コピーを発生させていた (mirror loss は学習の
# epoch×mini_batch 回呼ばれるため無視できない)。プロファイルでも
# internal_new_from_data → to(device) → copy_kernel_cuda として観測された。
_CONST_CACHE: dict = {}


def _mirror_consts(device: torch.device, dtype: torch.dtype) -> dict:
    key = (device, dtype)
    consts = _CONST_CACHE.get(key)
    if consts is None:
        consts = {
            "joint_sign": torch.tensor(_JOINT_MIRROR_SIGN, device=device, dtype=dtype),
            "ang_vel": torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype),
            "proj_gravity": torch.tensor([1.0, -1.0, 1.0], device=device, dtype=dtype),
            "vel_cmd": torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype),
        }
        _CONST_CACHE[key] = consts
    return consts


def _mirror_joints(joint_data: torch.Tensor) -> torch.Tensor:
    """関節ごとの量 (..., 12) を左右反転する。

    左脚と右脚を入れ替えたうえで、roll / yaw 関節の符号を反転する。
    joint_pos / joint_vel / action のいずれにも使える。
    """
    out = torch.empty_like(joint_data)
    out[..., _LEFT_JOINT_IDX] = joint_data[..., _RIGHT_JOINT_IDX]
    out[..., _RIGHT_JOINT_IDX] = joint_data[..., _LEFT_JOINT_IDX]
    sign = _mirror_consts(joint_data.device, joint_data.dtype)["joint_sign"]
    return out * sign


def _mirror_policy_obs(obs: torch.Tensor) -> torch.Tensor:
    """ポリシー観測 (N, 49) を矢状面に対して左右反転する。"""
    if obs.shape[-1] != _POLICY_OBS_DIM:
        raise ValueError(
            f"symmetry: ポリシー観測の次元が想定 ({_POLICY_OBS_DIM}) と異なります: {obs.shape[-1]}。"
            " mdp/symmetry.py のスライス定義を K1PolicyCfg に合わせて更新してください。"
        )

    out = obs.clone()
    consts = _mirror_consts(out.device, out.dtype)

    # base_ang_vel: roll(x), yaw(z) を反転
    out[:, _ANG_VEL_SLICE] *= consts["ang_vel"]
    # projected_gravity: 横方向(y) を反転
    out[:, _PROJ_GRAVITY_SLICE] *= consts["proj_gravity"]
    # velocity_commands: lin_vel_y, ang_vel_z を反転
    out[:, _VEL_CMD_SLICE] *= consts["vel_cmd"]
    # 関節量
    out[:, _JOINT_POS_SLICE] = _mirror_joints(obs[:, _JOINT_POS_SLICE])
    out[:, _JOINT_VEL_SLICE] = _mirror_joints(obs[:, _JOINT_VEL_SLICE])
    out[:, _LAST_ACTION_SLICE] = _mirror_joints(obs[:, _LAST_ACTION_SLICE])
    # gait_phase: 左右の位相を入れ替え
    out[:, _GAIT_PHASE_SLICE] = obs[:, _GAIT_PHASE_SWAP]

    return out


# ---------------------------------------------------------------------------
# 履歴バッファ構成 (K1FlatEnvCfg: command + policy/critic 履歴グループ) 用の反転
# ---------------------------------------------------------------------------
# 各観測グループは「項ごとに (履歴長 × 項次元) のブロック、ブロック内は古い→新しい」
# という flatten レイアウト (history_layout.py 参照)。左右反転は 1 ステップ内の
# 置換 + 符号反転を全ステップに繰り返し適用すればよいので、グループ全体に対する
# 添字置換テンソルと符号テンソルを一度だけ構築してキャッシュする。

# 反転種別 → (項内の添字置換 (None は恒等), 符号 (None は全て +1))
_KIND_TRANSFORMS = {
    "ang_vel": (None, (-1.0, 1.0, -1.0)),       # roll(x), yaw(z) を反転
    "gravity": (None, (1.0, -1.0, 1.0)),        # 横方向(y) を反転
    "lin_vel": (None, (1.0, -1.0, 1.0)),        # 横方向(y) を反転
    "vel_cmd": (None, (1.0, -1.0, -1.0)),       # lin_vel_y, ang_vel_z を反転
    "joint": (tuple(_RIGHT_JOINT_IDX + _LEFT_JOINT_IDX), tuple(_JOINT_MIRROR_SIGN)),
    "phase": ((2, 3, 0, 1), None),              # [sin_L, cos_L, sin_R, cos_R] の左右入れ替え
    "zmp": (None, (1.0, -1.0)),                 # 横方向(y) を反転
}

# (specs_key, device, dtype) → (perm, sign) のキャッシュ
_HISTORY_MIRROR_CACHE: dict = {}


def _history_mirror_consts(
    specs_key: str,
    term_specs,
    device: torch.device,
    dtype: torch.dtype,
    history_length: int = HISTORY_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (specs_key, history_length, device, dtype)
    consts = _HISTORY_MIRROR_CACHE.get(key)
    if consts is None:
        perm: list[int] = []
        sign: list[float] = []
        offset = 0
        for _, dim, kind in term_specs:
            term_perm, term_sign = _KIND_TRANSFORMS[kind]
            term_perm = tuple(range(dim)) if term_perm is None else term_perm
            term_sign = (1.0,) * dim if term_sign is None else term_sign
            for step in range(history_length):
                base = offset + step * dim
                perm.extend(base + p for p in term_perm)
                sign.extend(term_sign)
            offset += history_length * dim
        consts = (
            torch.tensor(perm, dtype=torch.long, device=device),
            torch.tensor(sign, dtype=dtype, device=device),
        )
        _HISTORY_MIRROR_CACHE[key] = consts
    return consts


# 履歴グループから「各項の最新 1 ステップ分」を取り出す添字のキャッシュ ((specs, device) → idx)
_LATEST_FRAME_IDX_CACHE: dict = {}


def _latest_frame_indices(term_specs, device: torch.device) -> torch.Tensor:
    key = (term_specs, device)
    idx = _LATEST_FRAME_IDX_CACHE.get(key)
    if idx is None:
        indices: list[int] = []
        offset = 0
        for _, dim, _ in term_specs:
            start = offset + (HISTORY_LENGTH - 1) * dim
            indices.extend(range(start, start + dim))
            offset += HISTORY_LENGTH * dim
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        _LATEST_FRAME_IDX_CACHE[key] = idx
    return idx


def _mirror_history_group(obs: torch.Tensor, specs_key: str, term_specs) -> torch.Tensor:
    """履歴グループの観測 (N, 履歴長 × 合計次元) を左右反転する。"""
    expected = sum(dim for _, dim, _ in term_specs) * HISTORY_LENGTH
    if obs.shape[-1] != expected:
        raise ValueError(
            f"symmetry: 観測グループ '{specs_key}' の次元が想定 ({expected}) と異なります: {obs.shape[-1]}。"
            " history_layout.py の項リストと mdp/symmetry.py を観測定義に合わせて更新してください。"
        )
    perm, sign = _history_mirror_consts(specs_key, term_specs, obs.device, obs.dtype)
    return obs.index_select(-1, perm) * sign


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
        if "command" in obs.keys():
            # 履歴バッファ構成 (K1FlatEnvCfg): command + policy/critic 履歴グループ。
            # 履歴観測は巨大 (数千次元 × ミニバッチ) なので、TensorDict.repeat(2) で
            # 全グループを複製するとメモリを大量に消費する。
            from tensordict import TensorDict

            consts = _mirror_consts(obs["command"].device, obs["command"].dtype)
            aug_groups: dict = {}

            cmd = obs["command"]
            cmd_aug = torch.empty(2 * batch_size, cmd.shape[-1], device=cmd.device, dtype=cmd.dtype)
            cmd_aug[:batch_size] = cmd
            torch.mul(cmd, consts["vel_cmd"], out=cmd_aug[batch_size:])
            aug_groups["command"] = cmd_aug

            policy = obs["policy"]
            expected = sum(dim for _, dim, _ in POLICY_TERM_SPECS) * HISTORY_LENGTH
            if policy.shape[-1] != expected:
                raise ValueError(
                    f"symmetry: 観測グループ 'policy' の次元が想定 ({expected}) と異なります:"
                    f" {policy.shape[-1]}。history_layout.py の項リストと mdp/symmetry.py を"
                    " 観測定義に合わせて更新してください。"
                )

            if actions is None:
                # mirror loss 用途: 対称性の評価は直近 1 ステップの生の観測のみで行う。
                # 履歴全体 (100 ステップ × 数千次元) を反転・複製する代わりに、最新
                # フレーム (1 ステップ分, 46 次元) だけを取り出して反転し、
                # LATEST_FRAME_GROUP として返す。HistoryActorCritic 側がこのフレームを
                # 履歴長ぶんタイル展開して方策入力を構築する (エピソード開始直後の
                # CircularBuffer バックフィル状態と同じ形の実在しうる観測)。
                frame_idx = _latest_frame_indices(POLICY_TERM_SPECS, policy.device)
                frame = policy.index_select(-1, frame_idx)
                perm, sign = _history_mirror_consts(
                    "policy_frame", POLICY_TERM_SPECS, policy.device, policy.dtype, history_length=1
                )
                frame_aug = torch.empty(2 * batch_size, frame.shape[-1], device=frame.device, dtype=frame.dtype)
                frame_aug[:batch_size] = frame
                torch.index_select(frame, -1, perm, out=frame_aug[batch_size:])
                frame_aug[batch_size:].mul_(sign)
                aug_groups[LATEST_FRAME_GROUP] = frame_aug
            else:
                # data augmentation 用途: 拡張バッチが学習にそのまま使われるため、
                # policy / critic の履歴全体を反転する。
                mirror_targets = [("policy", POLICY_TERM_SPECS)]
                if "critic" in obs.keys():
                    mirror_targets.append(("critic", CRITIC_TERM_SPECS))
                for group_name, term_specs in mirror_targets:
                    group = obs[group_name]
                    perm, sign = _history_mirror_consts(group_name, term_specs, group.device, group.dtype)
                    # 一時テンソルを作らず、拡張バッファの後半に直接書き込む
                    group_aug = torch.empty(
                        2 * batch_size, group.shape[-1], device=group.device, dtype=group.dtype
                    )
                    group_aug[:batch_size] = group
                    torch.index_select(group, -1, perm, out=group_aug[batch_size:])
                    group_aug[batch_size:].mul_(sign)
                    aug_groups[group_name] = group_aug

            obs_aug = TensorDict(aug_groups, batch_size=[2 * batch_size])
        else:
            # 旧レイアウト (単一 policy グループにコマンド込み・履歴なし)。critic グループ
            # (特権情報) は mirror loss の計算経路 (act_inference) では参照されないため、
            # 複製したまま据え置く。
            obs_aug = obs.repeat(2)
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
