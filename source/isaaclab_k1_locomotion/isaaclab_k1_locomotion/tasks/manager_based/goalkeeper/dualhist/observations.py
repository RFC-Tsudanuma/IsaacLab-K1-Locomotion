# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""デュアルヒストリー版ゴールキーパーの履歴観測。

元ネタ: Zhongyu Li et al., *Reinforcement Learning for Versatile, Dynamic, and Robust
Bipedal Locomotion Control* (arXiv:2401.16889)。Cassie の方策は入力を

    * 長期 I/O 履歴 (2.0s = 66 step @33Hz) → 1D CNN で圧縮 = 暗黙のシステム同定
    * 短期 I/O 履歴 (約 0.1s = 4 step)     → 生のまま MLP へ = 接触イベントへの即応

の 2 系統に分けている。ablation では「観測履歴だけ (state feedback only) では効かない、
**I/O ペア (観測 + 自分が出した入力) でなければ意味が無い**」「短期のみは sim では同等でも
実機で劣る」と報告されている。

本タスクへの転用方針 (論文そのままではない):
    論文の長期履歴は「時不変のダイナミクスずれ (摩擦・質量・PDゲイン) の同定」が狙いだが、
    ゴールキーパーで支配的な時変ノイズ源は **ボール知覚** の方。ビジョンは 20〜25Hz に対し
    制御は 50Hz、さらにレイテンシ・検出漏れ・距離依存ノイズ・自己位置推定の跳びが乗る。
    単一フレームの観測では「今の値が新しいのか 2 周期古い据え置きなのか」が原理的に
    判別できず、現行実装はボール速度を α-β フィルタで**手組みして**渡している
    (= 論文が退行すると報告した RMA / teacher-student と同じ「潜在量を人手で推定して渡す」形)。
    ここでは生の位置履歴を直接渡し、フィルタ・レイテンシ補償・到達点外挿を方策に学ばせる。

    論文の「I/O ペア」に対応するのは **自機の pose 履歴**。ボールの相対位置には自分の
    並進・旋回が混ざっているので、自己運動を同じフレームで渡さないと「ボールが動いた」と
    「自分が動いた/自己位置が跳んだ」を分離できない。

既存タスクとは完全に独立したファイルにしてある (共有の ``mdp/observations.py`` は触らない)。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from ..mdp.observations import (
    _gk_perceived_goal_state,
    _gk_perception,
    robot_pose_est,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


GK_HIST_FRAME_DIM = 7
"""履歴 1 フレームの次元。**変えたら :mod:`.symmetry` の反転符号も必ず更新すること。**

    [0] ボール x (フィールド座標系・知覚DR後・見えていなければ 0)
    [1] ボール y (同上)
    [2] 検出マスク (1 = 今見えている / 0 = 見えていない)
    [3] 自機 x (推定値。実機は MCL)
    [4] 自機 y (同上)
    [5] sin(推定 heading)
    [6] cos(推定 heading)

なぜこの 7 つか:
    * **フィールド座標系で持つ**のが要点。単一フレームの観測 (``ball_pos_rel``) は
      base yaw frame なので、そのまま時系列にすると「ボールが動いた」と「自分が動いた /
      回った」が混ざり、履歴から軌道を読むことが原理的にできない。フィールド座標系なら
      シュートの軌道はほぼ直線になり、素直な外挿がそのまま到達点予測になる。
    * ただし実機で使えるのは **推定した自己位置** だけなので、変換には
      :func:`~..mdp.observations.robot_pose_est` (MCL 誤差込み) を使う。したがって履歴には
      自己位置推定のバイアス・ドリフト・跳びがそのまま乗る = 実機と同じ条件。自機 pose も
      同じフレームに入れてあるので、方策は「ボールが動いた」と「自己位置が跳んだ」を
      切り分けられる (論文の I/O ペアに対応する部分)。
    * 検出マスクを入れるのは、位置チャンネルが 0 のとき「ゴール中央に見えた」のか
      「見えていない」のかを区別するため。位置は mask=0 のとき 0 埋めにしてある
      (:func:`~..mdp.observations._gk_perceived_goal_state` は mask=0 でも自機位置を
      返してしまうので、そのまま入れると「足元にボールがある」と読める列になる)。
    * ボール速度 (α-β 出力) は履歴に入れない。それを学習で置き換えるのがこの試験の主眼で、
      単一フレームの観測スロットには従来どおり残してある (併存させて方策に選ばせる)。
"""


def _gk_history_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """履歴リングに書き込む 1 フレーム (N, :data:`GK_HIST_FRAME_DIM`)。"""
    pos, _ = _gk_perceived_goal_state(env)   # 知覚DR後・フィールド座標系
    mask = _gk_perception(env).ball_mask     # (N,) float32
    xy, heading = robot_pose_est(env)        # 推定自己位置 (実機は MCL)
    return torch.stack(
        [
            pos[:, 0] * mask,
            pos[:, 1] * mask,
            mask,
            xy[:, 0],
            xy[:, 1],
            torch.sin(heading),
            torch.cos(heading),
        ],
        dim=1,
    )


def _ordered(buf: torch.Tensor, ptr: int) -> torch.Tensor:
    """リングバッファを「古い → 新しい」順に並べ替えて返す。

    ``ptr`` は次に書き込む位置 = 現時点で最も古いフレームの位置。
    """
    if ptr == 0:
        return buf
    return torch.cat([buf[:, ptr:], buf[:, :ptr]], dim=1)


def _gk_history_tick(env: ManagerBasedRLEnv, capacity: int) -> None:
    """履歴リングを 1 制御ステップ進める (冪等)。

    書き込みポインタは全 env 共通のスカラ。全 env が毎ステップ同時に 1 フレーム書くので
    共有で正しい。リセットされた env の履歴を消すのは :func:`reset_gk_history`
    (mode="reset") の役目 — イベントは ``_reset_idx`` の中、観測計算より前に走る。

    容量は「その時点で要求された最大」に合わせて必要なら伸ばす。短期・長期の 2 項が別々の
    長さを要求するので、初回だけ 2 度目の呼び出しで拡張が走る (中身は保持する)。
    """
    n = env.num_envs
    buf = getattr(env, "_gk_hist_buf", None)
    if buf is None or buf.shape[0] != n:
        env._gk_hist_buf = torch.zeros(n, capacity, GK_HIST_FRAME_DIM, device=env.device)
        env._gk_hist_ptr = 0
        env._gk_hist_step = -1
    elif buf.shape[1] < capacity:
        # 容量拡張。既存の内容を「新しい側が末尾」になるよう詰め直し、ポインタは 0 に戻す
        # (= 先頭が最古)。``_gk_hist_step`` は**保持する**。ここでリセットすると同じ
        # ステップで 2 回書き込み、最古のスロットが最新フレームで潰れる。
        old = _ordered(buf, int(env._gk_hist_ptr))
        new_buf = torch.zeros(n, capacity, GK_HIST_FRAME_DIM, device=env.device)
        new_buf[:, capacity - old.shape[1]:] = old
        env._gk_hist_buf = new_buf
        env._gk_hist_ptr = 0

    step = int(env.common_step_counter)
    if env._gk_hist_step == step:
        return
    env._gk_hist_step = step

    buf = env._gk_hist_buf
    buf[:, int(env._gk_hist_ptr)] = _gk_history_frame(env)
    env._gk_hist_ptr = (int(env._gk_hist_ptr) + 1) % buf.shape[1]


def gk_io_history(
    env: ManagerBasedRLEnv,
    num_frames: int = 5,
    stride: int = 1,
) -> torch.Tensor:
    """知覚・自己位置の履歴 (N, ``num_frames`` × :data:`GK_HIST_FRAME_DIM`)。

    並びは **[古い … 新しい] × [フレーム内 7 チャンネル]** の row-major。ネットワーク側
    (:class:`~.networks.ActorCriticDualHistory`) はこれを ``(N, num_frames, 7)`` に
    reshape して 1D CNN に食わせるので、**順序を変えないこと**。

    Args:
        num_frames: 何フレーム返すか。
        stride: 何ステップおきに取るか (1 = 制御周期そのまま = 50Hz)。
            論文の長期履歴は 33Hz × 66 frame = 2.0s。本タスクは 50Hz なので
            50 frame × stride 1 で 1.0s、25 frame × stride 4 で 2.0s になる。
            ★ stride を上げるとビジョン更新 (20〜25Hz) に対してエイリアスするので、
              「見失いの頻度」の情報が落ちる。長期を伸ばしたいときはまず frame 数で。

    エピソード開始直後は履歴が埋まっていないぶんがゼロで返る (= 「まだ何も見ていない」)。
    """
    num_frames = int(num_frames)
    stride = int(stride)
    capacity = (num_frames - 1) * stride + 1
    _gk_history_tick(env, capacity)

    ordered = _ordered(env._gk_hist_buf, int(env._gk_hist_ptr))
    start = ordered.shape[1] - 1 - (num_frames - 1) * stride
    return ordered[:, start::stride].reshape(env.num_envs, -1)


def reset_gk_history(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """リセットされた env の履歴をゼロに戻す (EventTerm, mode="reset")。

    これが無いと前エピソードのボール軌道が新エピソードの先頭に残り、方策は
    「開始直後に存在しない球が見えている」状態を学習する。
    """
    buf = getattr(env, "_gk_hist_buf", None)
    if buf is not None:
        buf[env_ids] = 0.0
