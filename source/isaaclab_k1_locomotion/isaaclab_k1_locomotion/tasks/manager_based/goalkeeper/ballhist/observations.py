# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール履歴版ゴールキーパーの観測 (ボール相対位置の履歴)。

★ 設計の要点

    直接版 (goalkeeper_direct_env_cfg.py) では「どこへどれだけ速く動くか」を
    手書きの式 (:func:`~..mdp.observations.task_drive_vector`) が決めていた。
    実機で出た不具合3件すべてがこの式に起因していたため、**その判断を方策へ渡す**。

    変更は 2 点だけ:

      1. 方策の ``velocity_commands`` スロットを **ゼロ埋め** する
         (方策は指令の中身を見なくなる)
      2. **ボール相対位置の履歴** を観測の末尾に足す
         (方策はここから方向と速さを自分で決める)

    報酬・歩行位相のゲート・critic は直接版のまま。指令は **特権情報** として
    学習側に残す (実機では動かないので問題ない。critic に真値を渡すのと同じ考え方)。

★ なぜ履歴か

    単フレーム観測では「本物のボールの動き」と「知覚ノイズ・自己位置の跳び」を
    区別できない。両者は時間パターンが違う (跳びは 1 フレームの不連続 + 減衰) ので、
    履歴があれば方策が自分で識別・平均化できる。手書きの α-β フィルタとしきい値で
    やっていた仕事を、学習に置き換えるのが狙い。

★ なぜロボット座標系か

    ``gk_ball_pos_rel_perceived`` は **base yaw frame** を返すので、自己位置推定
    (MCL) を経由しない。ゴール座標系に変換すると MCL の跳びが混入する
    (実機のボール速度が汚染される既知の経路)。履歴は素のまま持たせる。

★ 速度は入れない

    速度は位置の微分でノイズが乗る。位置履歴を渡せば必要な微分は方策が学習する。
    実機の CVKF 由来の速度汚染をそもそも入力しない、という判断。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from ..mdp.observations import gk_ball_active, gk_ball_pos_rel_perceived

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_HIST_BUF_ATTR = "_gk_ballhist_buf"    # (N, capacity, 3) = [x, y, valid]
_HIST_PTR_ATTR = "_gk_ballhist_ptr"
_HIST_STEP_ATTR = "_gk_ballhist_step"  # 同一ステップ内の多重更新を防ぐ
_HIST_PREV_ATTR = "_gk_ballhist_prev_active"

FRAME_DIM = 3  # [x, y, valid] / フレーム


def _tick(env: "ManagerBasedRLEnv", capacity: int) -> None:
    """履歴リングバッファを 1 制御ステップ進める (冪等)。

    ★ 新しい球が発射された瞬間 (``ball_active`` の立ち上がり) に履歴を捨てる。
      前の球の軌道が残っていると、方策が存在しない動きを見ることになる。
      直接版の ``_gk_fit_tick`` と同じ規約。
    """
    n = env.num_envs
    buf = getattr(env, _HIST_BUF_ATTR, None)
    if buf is None or buf.shape[0] != n or buf.shape[1] != capacity:
        buf = torch.zeros(n, capacity, FRAME_DIM, device=env.device)
        setattr(env, _HIST_BUF_ATTR, buf)
        setattr(env, _HIST_PTR_ATTR, 0)
        setattr(env, _HIST_STEP_ATTR, -1)
        setattr(env, _HIST_PREV_ATTR, torch.zeros(n, dtype=torch.bool, device=env.device))

    step = int(env.common_step_counter)
    if getattr(env, _HIST_STEP_ATTR, -1) == step:
        return
    setattr(env, _HIST_STEP_ATTR, step)

    active = gk_ball_active(env).squeeze(-1) > 0.5
    prev = getattr(env, _HIST_PREV_ATTR)
    launched = active & (~prev)
    setattr(env, _HIST_PREV_ATTR, active.clone())
    if bool(launched.any()):
        buf[launched] = 0.0

    pos_b = gk_ball_pos_rel_perceived(env)  # (N, 2) base yaw frame。見失い時は 0
    ptr = int(getattr(env, _HIST_PTR_ATTR))
    slot = buf[:, ptr]
    slot[:, 0] = pos_b[:, 0]
    slot[:, 1] = pos_b[:, 1]
    # 見失い (mask=0) のフレームは pos が 0 になるので、valid で区別できるようにする。
    # ゼロ埋めと「本当に原点にある」を方策が混同しないための 1 ビット。
    slot[:, 2] = (active & (torch.norm(pos_b, dim=1) > 1e-6)).float()
    setattr(env, _HIST_PTR_ATTR, (ptr + 1) % capacity)


def ballhist_ball_history(
    env: "ManagerBasedRLEnv",
    frames: int = 10,
    stride: int = 2,
) -> torch.Tensor:
    """ボール相対位置の履歴 (N, frames * 3)。**新しい順**に並べる。

    1 フレームは ``[x, y, valid]`` (base yaw frame [m], valid は検出できたか)。

    Args:
        frames: 返すフレーム数。
        stride: 何制御ステップおきに採るか。既定の 10 x 2 は
            0.02s x 2 x 10 = **0.4 秒** ぶんを覆う。ボールの到達時間が
            0.55〜1.4 秒なので、接近の判断には十分な窓。

    ★ 並び順は「最新が先頭」。方策から見て最新フレームの位置が固定なので、
      窓長を変えても先頭の意味が変わらない (学習の引き継ぎがしやすい)。
    """
    capacity = max(2, int(frames) * int(stride))
    _tick(env, capacity)

    buf = getattr(env, _HIST_BUF_ATTR)
    ptr = int(getattr(env, _HIST_PTR_ATTR))
    # ptr は「次に書く位置」なので、最新は ptr-1。そこから stride おきに遡る。
    idx = [(ptr - 1 - k * int(stride)) % capacity for k in range(int(frames))]
    return buf[:, idx].reshape(env.num_envs, -1)


def ballhist_ball_history_true(
    env: "ManagerBasedRLEnv",
    frames: int = 10,
    stride: int = 2,
) -> torch.Tensor:
    """critic 用: 真値のボール相対位置を履歴と同じ形で返す (N, frames * 3)。

    critic は真値を見てよいので履歴を持つ必要が無い。次元だけ揃えるため、
    **現在の真値を全フレームに複製**する (非対称 actor-critic の規約に従う)。
    こうすると actor/critic で層の形を揃えられ、実装が単純になる。
    """
    from ..mdp.observations import gk_ball_pos_rel

    pos = gk_ball_pos_rel(env)                                   # (N, 2) 真値
    valid = gk_ball_active(env).squeeze(-1).unsqueeze(1)         # (N, 1)
    frame = torch.cat([pos, valid], dim=1)                       # (N, 3)
    return frame.repeat(1, int(frames))


# ---------------------------------------------------------------------------
# is_engaged: 「出動すべきか」の状態判定
#
# ★ 2026-08-18: 手書きの **制御則** を実機の経路から完全に外すための最後の部品。
#
#   直接版では歩行位相のゲートを task_drive_vector (手書きの制御則) の大きさで
#   決めていた。あれは実機の C++ が計算する必要があり、しかも
#     * t = (ball_x − guard_x) / (−vx)  → vx→0 で発散
#     * dy = ずれ / 0.15                → ゲイン 6.67 のバンバン制御
#   という構造で、実機の不具合3件すべての原因になっていた。
#
#   ここで置き換える is_engaged は **割り算も外挿もクランプも無い真偽値** で、
#   ボール相対観測 (base yaw frame) だけから決まる。発散もゲイン増幅も起きず、
#   自己位置推定 (MCL) も使わない。実機の C++ 側もこれだけ実装すればよい。
#
#     is_engaged = 検出できている AND ( 近づいている OR 守備面での横ずれが大きい )
#
#   「どこへどれだけ速く動くか」は方策がボール履歴から決める。手書きに残すのは
#   「出動するかどうか」の 1 ビットだけ。
# ---------------------------------------------------------------------------

_ENGAGED_ATTR = "_gk_ballhist_engaged"
_ENGAGED_STEP_ATTR = "_gk_ballhist_engaged_step"


def ballhist_is_engaged(
    env: "ManagerBasedRLEnv",
    lateral_enter_m: float = 0.25,
    lateral_exit_m: float = 0.30,
    closing_min_m: float = 0.15,
    frames: int = 10,
    stride: int = 2,
) -> torch.Tensor:
    """出動すべきかの真偽値 (N,)。**自己位置を使わない**。

    Args:
        lateral_enter_m / lateral_exit_m: 守備面へ投影した横ずれのしきい値 [m]。
            入る/出るで変える (ヒステリシス)。同値だと境界でトグルする。
        closing_min_m: 履歴の窓 (frames x stride) の間にボールが縮めた距離 [m]。
            これを超えていれば「近づいている」とみなす。**速度ではなく位置差**で
            見るのが要点で、微分によるノイズ増幅が無い。
    """
    p = env.cfg.goalkeeper
    guard_x = float(p.guard_x)

    hist = ballhist_ball_history(env, frames=frames, stride=stride)
    h = hist.view(env.num_envs, frames, FRAME_DIM)
    valid = h[:, :, 2] > 0.5

    # ★ 2026-08-18: 「今このフレームで見えているか」ではなく
    #   **「直近のどこかで見えたか」** で判定する。
    #
    #   知覚モデルは距離依存のベルヌーイ検出なので毎フレーム落ちる。実測 (汚れた知覚)
    #   では、今フレーム判定だと **1 球あたり 86 回** 出動が落ち、最長 2.32 秒
    #   非出動になった。これで歩行位相をゲートすると、セーブの最中に歩行が
    #   細かく止まり続ける (実機で見ている振動の作り方そのもの)。
    #
    #   人間の GK は一瞬見失っても直前まで見えていた方向へ動き続ける。同じ扱いにする。
    detected = valid.any(dim=1)

    # 位置は **最後に見えたフレーム** を使う。今フレームが miss だと pos が 0 になり、
    # 「ボールが原点にある」と誤解するため。
    ar = torch.arange(env.num_envs, device=env.device)
    idx_last = torch.argmax(valid.float(), dim=1)      # 先頭が最新なので最初の valid
    now = h[ar, idx_last]
    # 窓の古い側も同様に「最も古い valid フレーム」を使う。最古スロット 1 枚の
    # 検出有無で判定すると、そこが miss のたびに approaching が落ちる。
    idx_old = valid.shape[1] - 1 - torch.argmax(valid.flip(1).float(), dim=1)
    old = h[ar, idx_old]
    bx, by = now[:, 0], now[:, 1]

    # --- 横ずれ: ボールとゴール中央を結ぶ線を守備面で切った点までの距離 ---
    #   ロボットは守備面付近に立っているので、ロボット座標系の by をそのまま
    #   距離で減衰させた量が「これから横に動くべき量」の良い近似になる。
    #   遠いボールほど小さくなるので、遠方の知覚ノイズで出動しない。
    # ★ 2026-08-18: 瞬時値ではなく **履歴の平均** を使う。
    #
    #   by はロボット座標系の量なので、**自分のヨー揺れがそのまま横ずれに化ける**。
    #   3m 先のボールなら 0.1 rad の姿勢変化で 30cm 動く。しきい値 0.25/0.30 と
    #   同じスケールなので、ヒステリシスがあっても跨ぎ続ける。
    #   実測 (汚れた知覚・待機中の種ポリシー) では 1 球あたり 86 回も出動が落ちた。
    #   ヨー揺れは高周波なので、窓 0.4 秒で平均すればほぼ消える。
    lat_all = (h[:, :, 1].abs() * guard_x / h[:, :, 0].clamp(min=guard_x))
    w = valid.float()
    lateral = (lat_all * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)

    # --- 近づいているか: 窓の間に縮んだ距離 (両端とも検出できている場合のみ) ---
    # 窓内に 2 枚以上 valid があれば接近判定ができる (最新と最古が別フレームであること)
    both_seen = detected & (idx_old > idx_last)
    closed = (old[:, 0] - bx).clamp(min=0.0)          # x が減っていれば正
    approaching = both_seen & (closed > closing_min_m)

    prev = getattr(env, _ENGAGED_ATTR, None)
    if prev is None or prev.shape[0] != env.num_envs:
        prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, _ENGAGED_ATTR, prev)
    if getattr(env, _ENGAGED_STEP_ATTR, -1) == int(env.common_step_counter):
        return prev

    tol = torch.where(prev, lateral_exit_m, lateral_enter_m)
    out = detected & (approaching | (lateral > tol))

    setattr(env, _ENGAGED_ATTR, out)
    setattr(env, _ENGAGED_STEP_ATTR, int(env.common_step_counter))
    return out


def ballhist_gait_phase(
    env: "ManagerBasedRLEnv",
    phase_freq: float = 1.6,
    step_jitter: float = 0.1,
    frames: int = 10,
    stride: int = 2,
) -> torch.Tensor:
    """歩行位相 (N, 4)。``is_engaged`` のときだけ位相を進め、それ以外はゼロ埋め。

    フォーマットは locomotion の ``phase_obs`` と同一 (左右の sin/cos)。
    位相は **積算** (``_task_gait_phase_accum``) なので、停止→再開で位相が飛ばない。
    """
    import math

    from ..mdp.observations import _task_gait_phase_accum

    walking = ballhist_is_engaged(env, frames=frames, stride=stride)
    phase_left = _task_gait_phase_accum(env, phase_freq, walking, step_jitter=step_jitter)
    phase_right = phase_left + math.pi
    phase = torch.stack([
        torch.sin(phase_left), torch.cos(phase_left),
        torch.sin(phase_right), torch.cos(phase_right),
    ], dim=1)
    return torch.where(walking.unsqueeze(1), phase, torch.zeros_like(phase))
