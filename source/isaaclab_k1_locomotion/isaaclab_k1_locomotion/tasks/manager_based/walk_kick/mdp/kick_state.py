# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""キック報酬が共有する latch 状態。

値 latch (凍結) と状態 latch (フラグ) を厳密に分けて保持する。

* 値 latch: トリガー L (``v_ball > v_thresh``) の発火時に τ_direction / v_ball / v_ball_3d /
  φ (仰角) / p_style を **同時に**スナップショットして固定する。以降はその凍結値で dense に払う。
* 状態 latch: ``kick_done`` (L 発火) と ``overshoot_fired`` (後方レイ R の左右跨ぎ)。
  いずれもエピソード内で一度立ったら解除しない。

状態はステップごとに一度だけ更新する (``common_step_counter`` でステップ境界を検出)。
どの項から先に呼ばれても同じ結果になるので、報酬項の評価順に依存しない。

NOTE: RewardManager は weight==0 の項をスキップするので、カリキュラムで weight を 0 から
      立ち上げる項だけに更新を任せると Phase 1 の間ずっと状態が更新されない。そのため
      **常に有効な termination 項** (:func:`..terminations.kick_finished`) からも
      :func:`kick_state` を呼び、毎ステップの更新を保証している。TerminationManager は
      RewardManager より先に走るので、報酬項が読む時点で状態は最新になっている。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ATTR = "_kick_latch_state"

# --------------------------------------------------------------------------- #
# 接触回数カウントのパラメータ
#
# 「足がボールに何回触ったか」を、接触センサーではなく **ボール速度の跳ね上がり**
# で数える。接触センサー (contact_balls_left/right) は history_length=1 かつ
# decimation=4 なので、衝突フレームを取りこぼすと数え落とす。一方ボールの速度変化は
# 接触後も残るため、50Hz のサンプリングでも確実に捕捉できる。
#
# _TOUCH_DV_THRESH: ステップ間の水平速度の増分がこの値を超えたら「触った」とみなす。
#     ボールは転がりながら減速する (dv < 0) ので、正方向の跳ね上がりだけが接触を意味する。
# _TOUCH_REFRACTORY_STEPS: 1 回の接触が複数ステップにまたがって二重カウントされるのを
#     防ぐ不応期。0.1 秒。これより短い間隔の再接触は 1 回として数える。
# --------------------------------------------------------------------------- #
_TOUCH_DV_THRESH = 0.15
_TOUCH_REFRACTORY_STEPS = 5

# --------------------------------------------------------------------------- #
# init_side (overshoot 判定の基準側) を確定させる横距離の閾値 [m]
#
# init_side はエピソード開始時の sign(s) では決めない。ψ≈180° (ボール正面スタート) や
# ψ≈0° (整列済みスタート) では s≈0 で符号が数値ノイズになり、正しい回り込み・整列済みの
# 接近がコイントスで overshoot 罰 (-1.0) されてしまうため。代わりに 0 (未確定) から始め、
# ロボットが |s| > この閾値 までどちらかの側へ「寄った」時点の符号で確定する。
#
# 0.1 は歩行時の base 横揺れ (±5cm 程度) より上、スタンス幅の半分 (~0.1) 程度。
# 未確定 (init_side==0) の間は s*0=0 < 0 が偽なので crossed は発火しない。
# --------------------------------------------------------------------------- #
_INIT_SIDE_COMMIT_DIST = 0.1

# --------------------------------------------------------------------------- #
# 足リンク原点から足裏までの距離 [m]
#
# 足コライダー (MJCF の box: pos=(0.026, 0, -0.02), size=(0.09, 0.035, 0.018)) の
# 底面は足リンク原点から見て z = -0.02 - 0.018 = -0.038 にある。上面は z = -0.002 なので
# 箱の厚みは 0.036。
#
# 射出仰角を決めるのは「足先の上エッジ高さ」で、これは (足裏高さ + 0.036) に等しい。
# 球とボックスの接触では法線がエッジ→球中心を向くため、ボール半径 R とエッジ高さ e から
#     仰角 = atan((R − e) / sqrt(R² − (R − e)²))
# で決まる。R=0.11 のとき e=3.6cm (足裏接地) で 42°、e=9.1cm (足裏 5.5cm) で 10°。
# --------------------------------------------------------------------------- #
_SOLE_OFFSET = 0.038


def kick_state(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    command_name: str = "kick_direction",
    ball_name: str = "soccer_ball",
) -> dict:
    """キック関連の共有状態を返す。同一ステップ内では一度しか更新しない。"""
    step = int(env.common_step_counter)
    state = getattr(env, _ATTR, None)
    if state is not None and state["step"] == step:
        return state

    robot = env.scene["robot"]
    ball = env.scene[ball_name]
    device = env.device

    ball_pos = ball.data.root_pos_w[:, :2]
    ball_vel = ball.data.root_lin_vel_w[:, :2]
    ball_vel_z = ball.data.root_lin_vel_w[:, 2]
    ball_z = ball.data.root_pos_w[:, 2]
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_vel = robot.data.root_lin_vel_w[:, :2]

    # kick_direction コマンドは [sin θ, cos θ, v_target] (θ は world frame)
    cmd = env.command_manager.get_command(command_name)
    kick_dir = torch.stack([cmd[:, 1], cmd[:, 0]], dim=-1)  # (cos θ, sin θ), 単位ベクトル
    v_target = cmd[:, 2]
    # kick_dir を水平面で -90° 回した右向き単位ベクトル: R(-90)·(x, y) = (y, -x)
    right_vec = torch.stack([kick_dir[:, 1], -kick_dir[:, 0]], dim=-1)

    # ロボット胴体のヨー方向
    quat = robot.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    forward = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)

    if state is None:
        state = {
            "step": -1,
            "P_kick": torch.zeros(env.num_envs, 2, device=device),
            "init_side": torch.zeros(env.num_envs, device=device),  # 0 = 未確定
            "kick_done": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "overshoot_fired": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "overshoot_event": torch.zeros(env.num_envs, device=device),
            "tau_direction_frozen": torch.zeros(env.num_envs, device=device),
            "v_ball_frozen": torch.zeros(env.num_envs, device=device),
            "v_ball_3d_frozen": torch.zeros(env.num_envs, device=device),
            "phi_frozen": torch.zeros(env.num_envs, device=device),
            "p_style_frozen": torch.zeros(env.num_envs, device=device),
            "apex_height": torch.zeros(env.num_envs, device=device),
            "prev_v_ball": torch.zeros(env.num_envs, device=device),
            "touch_count": torch.zeros(env.num_envs, device=device),
            "touch_refractory": torch.zeros(env.num_envs, dtype=torch.int32, device=device),
            "extra_touch_event": torch.zeros(env.num_envs, device=device),
            "sole_height_last_touch": torch.zeros(env.num_envs, device=device),
            "sole_height_at_kick": torch.zeros(env.num_envs, device=device),
            # 軸足 (蹴っていない方の足) のボール相対位置。値 latch で凍結する。
            # plant_lon: キック方向成分 (+ = ボールより前)。plant_lat: 横方向の **絶対値**。
            "plant_lon_frozen": torch.zeros(env.num_envs, device=device),
            "plant_lat_frozen": torch.zeros(env.num_envs, device=device),
            "G": torch.zeros(env.num_envs, 2, device=device),
            "p_walk": torch.zeros(env.num_envs, device=device),
            "tau_walk": torch.zeros(env.num_envs, device=device),
            "d_sole_to_ball": torch.zeros(env.num_envs, device=device),
            "p_kick_pose": torch.zeros(env.num_envs, device=device),
            "v_target": torch.zeros(env.num_envs, device=device),
        }
        setattr(env, _ATTR, state)
        # 初回はまだエピソードが始まっていないので全 env を初期化対象にする
        just_reset = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    else:
        # reward / termination は episode_length_buf の加算後、_reset_idx の前に走るので、
        # 新エピソードの 1 歩目は episode_length_buf == 1 になる。
        just_reset = env.episode_length_buf == 1

    state["step"] = step
    state["v_target"] = v_target

    # ------------------------------------------------------------------ #
    # エピソード開始時のリセット: 凍結値・フラグ・P_kick・初期側符号
    # ------------------------------------------------------------------ #
    # 符号付き横距離 s: 後方レイ R からロボットがどちら側にいるか
    s = ((robot_pos - ball_pos) * right_vec).sum(dim=-1)

    if just_reset.any():
        # P_kick: R 上、ボールから後方 r_stance の点。エピソード終了まで固定。
        state["P_kick"][just_reset] = (ball_pos - r_stance * kick_dir)[just_reset]
        # init_side は未確定 (0) に戻す。確定は下の commit ブロックで行う。
        state["init_side"][just_reset] = 0.0
        state["kick_done"][just_reset] = False
        state["overshoot_fired"][just_reset] = False
        state["tau_direction_frozen"][just_reset] = 0.0
        state["v_ball_frozen"][just_reset] = 0.0
        state["v_ball_3d_frozen"][just_reset] = 0.0
        state["phi_frozen"][just_reset] = 0.0
        state["p_style_frozen"][just_reset] = 0.0
        state["apex_height"][just_reset] = 0.0
        state["prev_v_ball"][just_reset] = 0.0
        state["touch_count"][just_reset] = 0.0
        state["touch_refractory"][just_reset] = 0
        state["sole_height_last_touch"][just_reset] = 0.0
        state["sole_height_at_kick"][just_reset] = 0.0
        state["plant_lon_frozen"][just_reset] = 0.0
        state["plant_lat_frozen"][just_reset] = 0.0

    # ------------------------------------------------------------------ #
    # init_side の確定: ロボットが |s| > 閾値 までどちらかの側へ寄った時点の符号で確定
    # (0 = 未確定)。開始時の sign(s) で決めない理由は _INIT_SIDE_COMMIT_DIST 参照。
    # 従来エピソード (|s| > 0.1 で開始するほぼ全部) は 1 ステップ目で確定するので
    # 挙動は実質変わらない。
    # ------------------------------------------------------------------ #
    commit = (state["init_side"] == 0.0) & (s.abs() > _INIT_SIDE_COMMIT_DIST)
    if commit.any():
        state["init_side"] = torch.where(commit, torch.sign(s), state["init_side"])

    # ------------------------------------------------------------------ #
    # p_style: 胴体の向きが蹴り方向にどれだけ正対しているか (1 = 正対)
    # ------------------------------------------------------------------ #
    p_style = torch.clamp((forward * kick_dir).sum(dim=-1), min=0.0, max=1.0)

    # ------------------------------------------------------------------ #
    # 値 latch: L = (v_ball > v_thresh) の立ち上がりで
    # τ_direction, v_ball, v_ball_3d, φ, p_style を同時凍結
    #
    # NOTE: トリガーは意図的に **水平成分 v_ball のみ** で判定している。ループシュート
    #       (walk_loop) でもこのままで、閾値を 3D ノルムに変えてはいけない。3D にすると
    #       「ボールを踏んで真上に跳ね上げる」だけで latch が成立してしまい、φ 報酬の
    #       抜け道になる。水平トリガーのままなら「前に飛んでいること」が latch の前提条件
    #       として無料で手に入る。仰角 30° / v=3m/s のループでも v_xy = 2.6 m/s あるので
    #       閾値 0.8 は余裕で超える（見逃すのは 70° 超の、そもそも狙っていない打ち上げだけ）。
    # ------------------------------------------------------------------ #
    v_ball = ball_vel.norm(dim=-1)

    # ------------------------------------------------------------------ #
    # 接触回数: ボール速度の跳ね上がり (dv > 閾値) の立ち上がりを数える。
    #
    # エピソード開始直後 (just_reset) は prev_v_ball が 0 にリセットされた直後なので、
    # ボールが既に転がっていると偽の dv が出る。reset_ball は静止状態で置くので実害は
    # 無いが、念のため just_reset のステップはカウントしない。
    # ------------------------------------------------------------------ #
    dv = v_ball - state["prev_v_ball"]
    touched = (dv > _TOUCH_DV_THRESH) & (state["touch_refractory"] == 0) & (~just_reset)

    # 足リンクの位置。d_sole_to_ball と接触時足高さの両方で使う。
    foot_ids = _foot_body_ids(env, robot)
    foot_pos = robot.data.body_pos_w[:, foot_ids, :]  # (N, 2, 3)

    # ------------------------------------------------------------------ #
    # 接触瞬間の足裏高さ [m]。射出仰角を決めている唯一の量なので、真値で記録する。
    #
    # ボールに近い方の足を「蹴った足」とみなし、その足裏高さ (リンク原点 − _SOLE_OFFSET)
    # を接触のたびに更新する。凍結は値 latch (下の trigger) のタイミングで行う。
    #
    # NOTE: 「最初の接触」で凍結してはいけない。多重接触 (ball_touch_count ≈ 1.6) が
    #       ある状態では、最初の接触は蹴る前の偶発的な接触であることが多く、
    #       キック本体の足高さを測れない (φ=27° なのに 7.4cm という矛盾した値になった)。
    #       latch を起こした接触 = キック本体なので、そのときの値を採る。
    # ------------------------------------------------------------------ #
    d_foot_to_ball = (foot_pos - ball.data.root_pos_w[:, :3].unsqueeze(1)).norm(dim=-1)  # (N, 2)
    kicking_foot = d_foot_to_ball.argmin(dim=1)  # (N,)
    sole_z = foot_pos[torch.arange(env.num_envs, device=device), kicking_foot, 2] - _SOLE_OFFSET

    state["sole_height_last_touch"] = torch.where(
        touched, sole_z, state["sole_height_last_touch"]
    )

    # ------------------------------------------------------------------ #
    # 軸足 (蹴っていない方の足) のボール相対位置。値 latch で凍結する。
    #
    # 「蹴った足」を d_foot_to_ball の argmin で決めているので、軸足はその反対側。
    # キック方向フレームで測る:
    #   plant_lon = (p_sup − ball) · kick_dir   前後 (+ = ボールより前)
    #   plant_lat = |(p_sup − ball) · right_vec| 左右の **絶対値**
    #
    # NOTE: 左右は必ず絶対値で持つこと。符号付きにすると「左足キック (軸足は右)」と
    #       「右足キック (軸足は左)」の鏡像解のうち片方だけが正解になり、報酬側で
    #       探索空間を半分潰してしまう。どちらの足で蹴るかはポリシーの自由にしておく。
    # ------------------------------------------------------------------ #
    support_foot = 1 - kicking_foot
    p_sup = foot_pos[torch.arange(env.num_envs, device=device), support_foot, :2]
    d_sup = p_sup - ball_pos
    plant_lon = (d_sup * kick_dir).sum(dim=-1)
    plant_lat = torch.abs((d_sup * right_vec).sum(dim=-1))

    state["touch_count"] = state["touch_count"] + touched.float()
    # 2 回目以降の接触が起きたステップだけ 1。1 回目 (touch_count == 1) は無料。
    state["extra_touch_event"] = (touched & (state["touch_count"] >= 2.0)).float()

    state["touch_refractory"] = torch.where(
        touched,
        torch.full_like(state["touch_refractory"], _TOUCH_REFRACTORY_STEPS),
        torch.clamp(state["touch_refractory"] - 1, min=0),
    )
    state["prev_v_ball"] = v_ball

    trigger = (v_ball > v_thresh) & (~state["kick_done"])

    if trigger.any():
        # τ_direction: ボールの飛翔方向と蹴り方向の角度誤差 [rad]
        # 水平投影で測るので、ボールが浮いていてもそのまま「狙った方位か」を意味する。
        ball_dir = ball_vel / (v_ball.unsqueeze(-1) + 1e-6)
        cos_err = torch.clamp((ball_dir * kick_dir).sum(dim=-1), min=-1.0, max=1.0)
        sin_err = ball_dir[:, 0] * kick_dir[:, 1] - ball_dir[:, 1] * kick_dir[:, 0]
        tau_direction = torch.abs(torch.atan2(sin_err, cos_err))

        # φ: 仰角 [rad]。水平 = 0、真上 = π/2。負値 (打ち下ろし) もそのまま持つ。
        phi = torch.atan2(ball_vel_z, v_ball + 1e-6)
        v_ball_3d = torch.sqrt(v_ball**2 + ball_vel_z**2)

        state["tau_direction_frozen"] = torch.where(
            trigger, tau_direction, state["tau_direction_frozen"]
        )
        state["v_ball_frozen"] = torch.where(trigger, v_ball, state["v_ball_frozen"])
        state["v_ball_3d_frozen"] = torch.where(trigger, v_ball_3d, state["v_ball_3d_frozen"])
        state["phi_frozen"] = torch.where(trigger, phi, state["phi_frozen"])
        state["p_style_frozen"] = torch.where(trigger, p_style, state["p_style_frozen"])
        # latch を起こした接触 = キック本体。その足裏高さを凍結する。
        # 接触検出 (touched) は同じ関数内でこの上に走っているので、キックと同じ
        # ステップなら sole_height_last_touch は既に今回の値に更新されている。
        state["sole_height_at_kick"] = torch.where(
            trigger, state["sole_height_last_touch"], state["sole_height_at_kick"]
        )
        # 軸足の配置も latch と同時に凍結する。sole_height_at_kick と違って
        # 「最後の接触時」ではなく **latch したステップの現在値** を採る。軸足は接触の
        # 瞬間に接地しているので、キック本体のステップの値がそのまま構えを表す。
        state["plant_lon_frozen"] = torch.where(trigger, plant_lon, state["plant_lon_frozen"])
        state["plant_lat_frozen"] = torch.where(trigger, plant_lat, state["plant_lat_frozen"])
        state["kick_done"] = state["kick_done"] | trigger

    kick_done = state["kick_done"]

    # ------------------------------------------------------------------ #
    # 到達最高高度: latch 後のボール z の running max。報酬には使わず、メトリクス専用。
    # kick_finished の猶予窓 (2.0 秒) の間に実際の弾道の頂点を捕まえる。
    # ------------------------------------------------------------------ #
    state["apex_height"] = torch.where(
        kick_done, torch.maximum(state["apex_height"], ball_z), state["apex_height"]
    )

    # ------------------------------------------------------------------ #
    # 目標終端 G: R 上をボール側へ滑る点。latch 後は P_kick に固定して飛翔ボールを追わせない。
    # ------------------------------------------------------------------ #
    dist_robot_ball = (robot_pos - ball_pos).norm(dim=-1)
    reach = torch.clamp(alpha * dist_robot_ball, min=r_stance, max=0.5)
    G = ball_pos - reach.unsqueeze(-1) * kick_dir
    G = torch.where(kick_done.unsqueeze(-1), state["P_kick"], G)
    state["G"] = G

    # τ_walk: ロボット速度の G 方向成分 (符号付き)
    to_G = G - robot_pos
    d_to_G = to_G.norm(dim=-1)
    dir_to_G = to_G / (d_to_G.unsqueeze(-1) + 1e-6)
    state["tau_walk"] = (robot_vel * dir_to_G).sum(dim=-1)
    state["d_to_G"] = d_to_G

    # ------------------------------------------------------------------ #
    # 状態 latch: overshoot。キック線 R を跨いで確定側 (init_side) と反対側へ入ったら発火。
    # 前後位置・0.5m・G とは無関係。base_link の水平位置のみで判定する。
    # init_side が未確定 (0) の間は s*0=0 なので発火しない。
    # ------------------------------------------------------------------ #
    crossed = (s * state["init_side"]) < 0.0
    newly_fired = crossed & (~state["overshoot_fired"])
    state["overshoot_event"] = newly_fired.float()  # 発火したステップだけ 1 (1エピソード最大1回)
    state["overshoot_fired"] = state["overshoot_fired"] | crossed

    # ------------------------------------------------------------------ #
    # d_soleToBall: 左右の足裏のうちボールに近い方の距離
    # (foot_pos / d_foot_to_ball は接触時足高さの計算で既に求めてある)
    # ------------------------------------------------------------------ #
    state["d_sole_to_ball"] = d_foot_to_ball.min(dim=1).values

    state["p_style"] = p_style
    state["d_to_P_kick"] = (robot_pos - state["P_kick"]).norm(dim=-1)

    return state


_FOOT_IDS_ATTR = "_kick_foot_body_ids"


def _foot_body_ids(env: ManagerBasedRLEnv, robot) -> list[int]:
    """左右の足リンクの body index を一度だけ解決してキャッシュする。"""
    ids = getattr(env, _FOOT_IDS_ATTR, None)
    if ids is None:
        left = robot.find_bodies("left_foot_link")[0][0]
        right = robot.find_bodies("right_foot_link")[0][0]
        ids = [left, right]
        setattr(env, _FOOT_IDS_ATTR, ids)
    return ids
