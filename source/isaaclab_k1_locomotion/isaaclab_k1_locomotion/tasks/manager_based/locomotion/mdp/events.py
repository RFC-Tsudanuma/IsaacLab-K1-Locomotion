# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 専用のイベント関数。"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_PHASE_FREQ_ATTR = "_phase_freq_per_env"


def randomize_phase_freq(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    base_phase_freq: float,
    offset_range: tuple[float, float] = (-0.1, 0.1),
):
    """環境毎の歩行周波数を ``base_phase_freq + uniform(offset_range)`` でランダム化する。

    結果は ``env._phase_freq_per_env`` (shape ``[num_envs]``) に保持し、
    位相を扱う観測/報酬関数 (``phase_obs``, ``feet_phase``, ``foot_clearance_ji_pen`` 等)
    から :func:`get_phase_freq` 経由で参照する。
    """
    base = float(base_phase_freq)

    buf: torch.Tensor | None = getattr(env, _PHASE_FREQ_ATTR, None)
    if buf is None:
        buf = torch.full((env.num_envs,), base, device=env.device)
        setattr(env, _PHASE_FREQ_ATTR, buf)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    low, high = float(offset_range[0]), float(offset_range[1])
    offsets = torch.empty(env_ids.numel(), device=env.device).uniform_(low, high)
    buf[env_ids] = base + offsets


def get_phase_freq(env: "ManagerBasedEnv", default: float) -> "float | torch.Tensor":
    """環境毎にランダム化された位相周波数があればそれを、無ければスカラー ``default`` を返す。"""
    val = getattr(env, _PHASE_FREQ_ATTR, None)
    if val is None:
        return default
    return val


_PHASE_OFFSET_ATTR = "_phase_offset_per_env"


def randomize_phase_offset(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    prob_flip: float = 0.5,
):
    """エピソード毎に歩行位相を 0 か π のどちらかから始める (= 一歩目の足の左右)。

    位相は全箇所で ``2π * f * (episode_length_buf * dt)`` から作っているので、
    リセット直後は ``episode_length_buf = 0`` により **全 env が必ず同じ位相 0** で
    始まる。周波数だけランダム化しても t=0 では位相が 0 なので、
    **一歩目に出る足が毎エピソード同じ**になる。

    その結果、ボールが前方のどこに湧いても「その足で蹴るのが有利」という状況が
    学習中ずっと続き、ポリシーが片足でしか蹴らなくなる (実測: 右足のみ)。
    キック判定側は左右の足裏のうち近い方を見ている (``kick_state`` の
    ``d_foot_to_ball.min``) ので報酬は両足対等で、偏りは経験の側にしかない。

    π を足すと左右の脚の役割がそのまま入れ替わるので、これが
    「一歩目を右足にするか左足にするか」のランダム化そのものになる。
    ``[0, 2π)`` の一様乱数にしないのは、必ずどちらかの足がきれいに一歩目に
    なる方が歩容が崩れにくいため。

    結果は ``env._phase_offset_per_env`` (shape ``[num_envs]``) に持ち、位相を扱う
    観測/報酬から :func:`get_phase_offset` 経由で参照する。**位相を使う箇所すべてに
    同じオフセットを通すこと**。観測と報酬で位相の定義がずれると、報酬が実際の脚の
    動きと合わなくなって歩行が壊れる。

    Args:
        prob_flip: π を引く確率。0.5 で左右対等。
    """
    buf: torch.Tensor | None = getattr(env, _PHASE_OFFSET_ATTR, None)
    if buf is None:
        buf = torch.zeros((env.num_envs,), device=env.device)
        setattr(env, _PHASE_OFFSET_ATTR, buf)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    flip = torch.rand(env_ids.numel(), device=env.device) < float(prob_flip)
    buf[env_ids] = torch.where(
        flip,
        torch.full_like(flip, math.pi, dtype=buf.dtype),
        torch.zeros_like(flip, dtype=buf.dtype),
    )


def get_phase_offset(env: "ManagerBasedEnv", default: float = 0.0) -> "float | torch.Tensor":
    """env 毎の位相オフセットがあればそれを、無ければスカラー ``default`` を返す。

    :func:`get_phase_freq` と同じ約束。イベントを登録していない環境でも
    位相計算がそのまま動くように、無ければ 0 を返す。
    """
    val = getattr(env, _PHASE_OFFSET_ATTR, None)
    if val is None:
        return default
    return val


def set_kick_foot(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    foot: str = "right",
):
    """蹴り足を env に記録する (startup で 1 度だけ)。

    :func:`~...walk_kick.mdp.kick_state.kick_state` がこれを読んで、理想立ち位置
    ``P_kick`` を **その足がボール手前に来る** ように横へずらす。未設定なら従来
    どおり胴体中心をキック線上に置く。

    Args:
        foot: ``"left"`` または ``"right"``。
    """
    if foot not in ("left", "right"):
        raise ValueError(f'kick_foot は "left" か "right"。受け取った値: {foot!r}')
    env._kick_foot = foot


def set_p_style_exponent(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    exponent: float = 1.0,
):
    """キック報酬に掛かる正対度 p_style の鋭さを設定する (startup で 1 度だけ)。

    ``kick_state`` が ``p_style = clamp(cos(向きのズレ), 0, 1) ** exponent`` として使う。
    1.0 で従来どおり素の cos。

    素の cos は正対付近が平坦で、30° ずれても 0.87 しか下がらない。p_style はキック報酬の
    係数なので、これは「あと一歩回り込めば正対できるのに手前で蹴っても大して損しない」を
    意味し、大きく回り込む必要がある場面で蹴り急ぎ = 方向精度の悪化を招く。

    Args:
        exponent: 3.0 で 30° → 0.65 / 45° → 0.35。大きくするほど正対を厳しく要求する。
    """
    if exponent <= 0.0:
        raise ValueError(f"p_style の指数は正の値にすること。受け取った値: {exponent}")
    env._p_style_exponent = float(exponent)


def reset_prev_high_action(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
):
    """リセットされた env の ``_prev_high_action`` バッファを 0 にする。

    バッファ実体は ``HierarchicalVecEnvWrapper`` が用意するので、本関数は無ければ
    no-op で返す。Observation 計算は ``_reset_idx`` の後に走るので、ここで 0 化
    しておけば新エピソード最初の観測 ``last_high_action`` も 0 になる。
    """
    buf = getattr(env, "_prev_high_action", None)
    if buf is None:
        return
    if env_ids is None:
        buf.zero_()
        return
    buf[env_ids] = 0.0


__all__ = [
    "randomize_phase_freq",
    "get_phase_freq",
    "randomize_phase_offset",
    "get_phase_offset",
    "set_kick_foot",
    "set_p_style_exponent",
    "reset_prev_high_action",
]
