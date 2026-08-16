# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""最終分布での学習用: 難易度を動かさず、成功率だけを記録する項。

なぜ適応カリキュラムをやめるのか (2026-08-16 の実測):

    昇格には ``success_ema > adaptive_success_threshold`` (0.80) が要るが、実測した
    方策の実力は 0.635 で頭打ち。しかもその差は方策の欠陥ではなく **知覚コスト 17pt**
    (クリーン知覚なら同条件で 87.9%) なので、閾値やクールダウンをどう調整しても
    昇格しない。速度 2.074 で 1,100 iter 完全に横ばいだった。

    一方、到達可能性モデルで測ると **速い球は運動学的に難しくない**:

        速度上限   到達不能球   到達時間の中央値   スポーン距離 p90
          2.07       27.4%          0.69s            2.08m
          3.00       28.6%          0.69s            2.94m
          6.00       27.0%          0.75s            5.77m

    スポーン距離が「時間」で決まる設計 (``d = v × spawn_time_*``) なので、速い球ほど
    遠くから来る。結果として **到達時間の分布は速度によらず一定** で、到達不能球の
    割合も変わらない。変わるのは距離だけで、位置ノイズ σ(d)=0.124d+0.149 が
    0.41m → 0.86m に倍増する。つまり速い球の難しさは「速さ」ではなく「遠距離の知覚」。

    したがって **段階的に登る必要が無い**。最初から最終分布 (速度 U(0.5, cap)) で
    学習すればよい。易しい球が分布の半分を占めるので学習信号も途切れない。
    カリキュラムはブートストラップの役目を終えた、という判断。

この項は難易度を一切変えず、``success_ema`` の記録だけを担当する
(学習の進み具合を見る主要な指標なので、ログは残したい)。
``_gk_speed_hi`` / ``_gk_aim_y`` を **作らない** ので、``reset_ball_shot`` は
``GoalkeeperParamsCfg.ball_speed_max`` / ``aim_y_range`` を直接使う。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from ..mdp.curriculums import _update_success_ema
from ..mdp.events import _gk_params

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fixed_difficulty_log(env: "ManagerBasedRLEnv", env_ids) -> dict:
    """難易度は固定したまま、セーブ成功率 EMA だけを更新してログに出す。

    集計ロジックは適応カリキュラムと同一 (:func:`~..mdp.curriculums._update_success_ema`)
    なので、``Curriculum/difficulty/success_ema`` と同じ意味の数字が
    ``Curriculum/fixed/success_ema`` に出る。過去のランと直接比較できる。

    Returns:
        ログ用の dict (Curriculum/<term名>/<key> として TensorBoard に出る)。
    """
    p = _gk_params(env)

    if getattr(env, "_gk_success_ema", None) is None:
        env._gk_success_ema = torch.tensor(0.5, device=env.device)
        env._gk_episode_count = 0

    if env_ids is not None and len(env_ids) > 0:
        _update_success_ema(env, env_ids, p)

    return {
        "success_ema": float(env._gk_success_ema.item()),
        "ball_speed_hi": float(p.ball_speed_max),   # 固定値。動かない
        "aim_y_range": float(p.aim_y_range),        # 同上
        "episode_count": float(env._gk_episode_count),
    }
