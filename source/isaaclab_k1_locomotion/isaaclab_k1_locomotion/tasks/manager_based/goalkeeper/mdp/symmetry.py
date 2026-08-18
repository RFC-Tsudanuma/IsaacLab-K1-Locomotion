# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (直接制御版) の左右対称性 (sagittal plane mirror)。

locomotion の ``mdp/symmetry.py`` と同じ規約・同じ用途 (rsl-rl の mirror loss) だが、
観測が **59 次元** (歩行 49 + タスク 10) に拡張されているため、追加スロットぶんの
反転規則をここで足している。locomotion 側の関数は 49 次元固定で例外を投げる作りなので、
そちらは変更せず本ファイルで対応する (歩行部分の反転は locomotion の実装に委譲)。

mirror loss が有効な場合、PPO は以下を最小化する::

    || policy(mirror(obs)) - mirror(policy(obs)) ||^2

すなわち「観測を左右反転して与えたら、行動も左右反転して返す」対称な方策を促す。

これが無いと左右で別々の歩容に収束しうる。実際、Stage 1 の初回学習 (mirror loss なし)
では「右への横移動は良好だが、左へ動くときだけ左膝を曲げる」非対称な歩容になった。
歩行タスク (locomotion) は同じ mirror loss で対称性を担保しているので、有効化する
ことでむしろ locomotion と条件が揃う。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

# 歩行部分 (先頭 49 次元) の反転は locomotion の実装をそのまま使う。
from ...locomotion.mdp.symmetry import _mirror_joints, _mirror_policy_obs

if TYPE_CHECKING:
    from tensordict import TensorDict

    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states", "compute_symmetric_states_high_level"]

# 観測レイアウト (goalkeeper_direct_env_cfg.py の K1GKDirectStage1PolicyCfg と一致させること)
#   [0:49]  歩行 K1PolicyCfg と同一 (locomotion の _mirror_policy_obs が担当)
#   [49:51] ball_pos_rel  (x, y)  : base yaw frame のボール相対位置
#   [51:53] ball_vel      (x, y)  : base yaw frame のボール速度
#   [53:54] ball_active           : 左右に無関係なフラグ
#   [54:55] target_y              : ゴール座標系の目標 y
#   [55:59] self_state (x, y, sin(yaw), cos(yaw))
_WALK_OBS_DIM = 49
_POLICY_OBS_DIM = 59

# 追加 10 次元に掛ける符号。左右反転なので「横方向 (y) 成分と yaw だけ符号反転」。
#   ball_pos_rel : x そのまま, y 反転
#   ball_vel     : x そのまま, y 反転
#   ball_active  : そのまま (左右に無関係)
#   target_y     : 反転 (左右の目標なので)
#   self_state   : x そのまま, y 反転, sin(yaw) 反転, cos(yaw) そのまま
_TASK_MIRROR_SIGN = [
    1.0, -1.0,             # ball_pos_rel
    1.0, -1.0,             # ball_vel
    1.0,                   # ball_active
    -1.0,                  # target_y
    1.0, -1.0, -1.0, 1.0,  # self_state (x, y, sin(yaw), cos(yaw))
]

# 反転に使う符号定数を (device, dtype) ごとに一度だけ生成してキャッシュする
# (mirror loss は epoch × mini_batch 回呼ばれるため、毎回の host→device コピーは無視できない)。
_CONST_CACHE: dict = {}


# ★ 2026-08-18: ボール履歴版 (goalkeeper/ballhist/) はボール相対位置の履歴を末尾に足すため、
#   観測が 59 + frames*3 次元になる。1 フレームは [x, y, valid] なので、反転符号は
#   ball_pos_rel と同じ考え方で [x そのまま, y 反転, valid そのまま] の繰り返し。
#   critic 側 (真値の履歴) も同じ形なので同一の規則で扱える。
_HIST_FRAME_MIRROR_SIGN = [1.0, -1.0, 1.0]
_HIST_FRAME_DIM = len(_HIST_FRAME_MIRROR_SIGN)


def _task_sign(device: torch.device, dtype: torch.dtype, extra: int = 0) -> torch.Tensor:
    """タスクスロット (+ ボール履歴版の履歴) に掛ける反転符号を返す。

    Args:
        extra: 末尾に足された履歴の次元数 (``frames * 3``)。0 なら直接版と同じ。
    """
    key = (device, dtype, extra)
    sign = _CONST_CACHE.get(key)
    if sign is None:
        values = list(_TASK_MIRROR_SIGN)
        if extra:
            if extra % _HIST_FRAME_DIM != 0:
                raise ValueError(
                    f"symmetry: 追加次元 {extra} が 1 フレーム {_HIST_FRAME_DIM} の倍数では"
                    " ありません。goalkeeper/ballhist/observations.py の FRAME_DIM と揃えてください。"
                )
            values += _HIST_FRAME_MIRROR_SIGN * (extra // _HIST_FRAME_DIM)
        sign = torch.tensor(values, device=device, dtype=dtype)
        _CONST_CACHE[key] = sign
    return sign


# ---------------------------------------------------------------------------
# critic 観測 (64 次元) のレイアウト
#   K1CriticCfg (rough_env_cfg.py) の定義順 = スロット順:
#     [ 0: 3] base_lin_vel        (vx, vy, vz)      → y 反転
#     [ 3: 6] base_ang_vel        (wx, wy, wz)      → roll(x)/yaw(z) 反転
#     [ 6: 9] projected_gravity   (gx, gy, gz)      → y 反転
#     [ 9:12] velocity_commands   (vx, vy, wz)      → y/yaw 反転
#     [12:24] joint_pos                             → 左右入れ替え + roll/yaw 符号
#     [24:36] joint_vel                             → 同上
#     [36:48] actions                               → 同上
#     [48:52] gait_phase (sinL, cosL, sinR, cosR)   → 左右入れ替え
#     [52:54] zmp_position (base yaw frame の x, y) → y 反転 (zmp_xy_base が前提)
#   これに goalkeeper のタスク 10 次元 (policy と同じ順序・同じ符号) が続いて 64。
_CRITIC_OBS_DIM = 64
_CRITIC_TASK_START = 54
_CRITIC_GAIT_PHASE_SLICE = slice(48, 52)
# gait_phase の左右入れ替え (48+[2,3,0,1])。位相が π ずれるので mirror と等価。
_CRITIC_GAIT_PHASE_SWAP = [50, 51, 48, 49]
# joint_pos / joint_vel / actions の開始位置
_CRITIC_JOINT_SLICES = (slice(12, 24), slice(24, 36), slice(36, 48))
# 先頭 12 次元 (base_lin_vel, base_ang_vel, projected_gravity, velocity_commands) と
# zmp_position (2) に掛ける符号。
_CRITIC_HEAD_SIGN = [
    1.0, -1.0, 1.0,    # base_lin_vel
    -1.0, 1.0, -1.0,   # base_ang_vel
    1.0, -1.0, 1.0,    # projected_gravity
    1.0, -1.0, -1.0,   # velocity_commands (vx, vy, wz)
]
_CRITIC_ZMP_SIGN = [1.0, -1.0]


def _critic_sign(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = ("critic", device, dtype)
    consts = _CONST_CACHE.get(key)
    if consts is None:
        consts = (
            torch.tensor(_CRITIC_HEAD_SIGN, device=device, dtype=dtype),
            torch.tensor(_CRITIC_ZMP_SIGN, device=device, dtype=dtype),
        )
        _CONST_CACHE[key] = consts
    return consts


def _mirror_gk_critic_obs(obs: torch.Tensor) -> torch.Tensor:
    """ゴールキーパーの critic 観測 (N, 64) を矢状面に対して左右反転する。

    data augmentation を使う場合、critic は「反転した観測」と「反転した行動・
    リターン」の組で学習される必要がある。policy グループだけ反転して critic を
    複製したままにすると、critic が食い違った組を学習して価値推定が壊れる。
    """
    # ★ 2026-08-18: ボール履歴版は末尾にボール履歴が付くので 64 + frames*3 になる。
    extra = obs.shape[-1] - _CRITIC_OBS_DIM
    if extra < 0 or extra % _HIST_FRAME_DIM != 0:
        raise ValueError(
            f"symmetry: critic 観測の次元が想定 ({_CRITIC_OBS_DIM} または"
            f" {_CRITIC_OBS_DIM}+frames*{_HIST_FRAME_DIM}) と異なります: {obs.shape[-1]}。"
            " goalkeeper/mdp/symmetry.py のスライス定義を K1GKDirectCriticCfg に合わせて"
            " 更新してください。"
        )

    head_sign, zmp_sign = _critic_sign(obs.device, obs.dtype)
    out = obs.clone()
    out[:, :12] = obs[:, :12] * head_sign
    for sl in _CRITIC_JOINT_SLICES:
        out[:, sl] = _mirror_joints(obs[:, sl])
    out[:, _CRITIC_GAIT_PHASE_SLICE] = obs[:, _CRITIC_GAIT_PHASE_SWAP]
    out[:, 52:54] = obs[:, 52:54] * zmp_sign
    out[:, _CRITIC_TASK_START:] = obs[:, _CRITIC_TASK_START:] * _task_sign(obs.device, obs.dtype, extra)
    return out


def _mirror_gk_policy_obs(obs: torch.Tensor) -> torch.Tensor:
    """ゴールキーパーのポリシー観測を矢状面に対して左右反転する。

    次元は直接版が 59、ボール履歴版 (末尾にボール履歴) が ``59 + frames*3``。
    """
    dim = obs.shape[-1]
    extra = dim - _POLICY_OBS_DIM
    if extra < 0 or extra % _HIST_FRAME_DIM != 0:
        raise ValueError(
            f"symmetry: ポリシー観測の次元が想定 ({_POLICY_OBS_DIM} または"
            f" {_POLICY_OBS_DIM}+frames*{_HIST_FRAME_DIM}) と異なります: {dim}。"
            " goalkeeper/mdp/symmetry.py のスライス定義を"
            " K1GKDirectStage1PolicyCfg に合わせて更新してください。"
        )

    out = obs.clone()
    # 先頭 49 次元 = 歩行と同一構造なので locomotion の実装に委譲
    out[:, :_WALK_OBS_DIM] = _mirror_policy_obs(obs[:, :_WALK_OBS_DIM])
    # 残り = タスクスロット 10 次元 (+ ボール履歴版の履歴) の横方向成分を符号反転
    out[:, _WALK_OBS_DIM:] = obs[:, _WALK_OBS_DIM:] * _task_sign(out.device, out.dtype, extra)
    return out


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv",
    obs: "TensorDict | None" = None,
    actions: torch.Tensor | None = None,
):
    """観測・行動に左右対称変換を適用して拡張する (rsl-rl 用)。

    locomotion の同名関数と同じ規約: 返すバッチは ``[元のサンプル, 左右反転したサンプル]``
    の順に連結され、バッチサイズが 2 倍になる。policy グループのみ反転し、critic グループ
    (特権情報) は mirror loss の計算経路 (act_inference) で参照されないため据え置く。

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
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _mirror_gk_policy_obs(obs["policy"])
        # ★ 2026-08-09: critic グループも反転する。
        #   mirror loss だけなら critic は参照されないので複製のままでよかったが、
        #   data augmentation を有効にすると critic は拡張バッチ全体で学習される。
        #   ここを反転しないと、後半のサンプルが「反転していない観測」と
        #   「反転した行動・リターン」の食い違った組になり価値推定が壊れる。
        if "critic" in obs_aug.keys():
            obs_aug["critic"][:batch_size] = obs["critic"][:]
            obs_aug["critic"][batch_size:] = _mirror_gk_critic_obs(obs["critic"])
    else:
        obs_aug = None

    # -- 行動 (12 関節。左右の脚を入れ替えて roll/yaw の符号を反転)
    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(
            batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype
        )
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _mirror_joints(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


# ---------------------------------------------------------------------------
# 階層版 (goalkeeper_hier_env_cfg.py) 用
# ---------------------------------------------------------------------------

# 上位 action = 歩行コマンド (vx, vy, wz)。矢状面の鏡像では横速度と旋回だけ符号反転。
_HIGH_ACTION_DIM = 3
_HIGH_ACTION_MIRROR_SIGN = [1.0, -1.0, -1.0]


def _high_action_sign(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = ("high_action", device, dtype)
    sign = _CONST_CACHE.get(key)
    if sign is None:
        sign = torch.tensor(_HIGH_ACTION_MIRROR_SIGN, device=device, dtype=dtype)
        _CONST_CACHE[key] = sign
    return sign


@torch.no_grad()
def compute_symmetric_states_high_level(
    env: "ManagerBasedRLEnv",
    obs: "TensorDict | None" = None,
    actions: torch.Tensor | None = None,
):
    """階層版ゴールキーパーの上位ポリシー用の左右対称変換 (rsl-rl 用)。

    観測のレイアウトは直接制御版と**完全に同一** (policy 59 / critic 64) なので反転規則も
    そのまま使える。差し替わっているのは中身だけで、対称性の観点では等価:

        * ``velocity_commands`` スロット: 直接版は ``task_drive_vector`` (dx, dy, dyaw)、
          階層版は前回の上位 action (vx, vy, wz)。どちらも (前後, 左右, 旋回) の 3 成分で
          反転符号は (+, -, -) で同じ。critic 側の ``_CRITIC_HEAD_SIGN`` もこの並び。
        * ``gait_phase``: どちらも (sinL, cosL, sinR, cosR) で左右入れ替え。

    違うのは**行動だけ**で、12 関節ではなく 3 次元の歩行コマンドになる。

    上位にも mirror loss を掛ける理由: 下位 (07-28) は data augmentation 無し・
    mirror_loss_coeff=0.5 の世代で学習されており、横移動時に約 10°/s の系統的な yaw
    ドリフト (= 左右非対称歩容の典型症状) が残っている。上位までそれに引きずられて
    左右で別の戦略に収束すると、片側のセーブだけ弱いキーパーになる。
    """
    # -- 観測 (直接制御版と同一レイアウトなので既存の反転をそのまま使う)
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _mirror_gk_policy_obs(obs["policy"])
        if "critic" in obs_aug.keys():
            obs_aug["critic"][:batch_size] = obs["critic"][:]
            obs_aug["critic"][batch_size:] = _mirror_gk_critic_obs(obs["critic"])
    else:
        obs_aug = None

    # -- 行動 (歩行コマンド 3 次元。vy と wz の符号を反転)
    if actions is not None:
        if actions.shape[-1] != _HIGH_ACTION_DIM:
            raise ValueError(
                f"symmetry: 上位 action の次元が想定 ({_HIGH_ACTION_DIM}) と異なります:"
                f" {actions.shape[-1]}。高レベル action を (vx, vy, wz) 以外にしたなら"
                " _HIGH_ACTION_MIRROR_SIGN を更新してください。"
            )
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(
            batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype
        )
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = actions * _high_action_sign(actions.device, actions.dtype)
    else:
        actions_aug = None

    return obs_aug, actions_aug
