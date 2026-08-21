# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""キック報酬が共有する latch 状態。

値 latch (凍結) と状態 latch (フラグ) を厳密に分けて保持する。

* 値 latch: トリガー L (既定は ``v_ball > v_thresh``、opt-in では足とボールの物理的な
  速度変化) の発火時に τ_direction / v_ball / v_ball_3d / φ (仰角) / p_style を
  **同時に**スナップショットして固定する。以降はその凍結値で dense に払う。
* 状態 latch: ``kick_done`` (L 発火) と ``overshoot_fired`` (後方レイ R の左右跨ぎ)。
  いずれもエピソード内で一度立ったら解除しない。

状態はステップごとに一度だけ更新する (``common_step_counter`` でステップ境界を検出)。
同じステップ内で部分 reset された行だけは新エピソードの基準値へ再初期化するが、
他の行は再更新しない。どの項から先に呼ばれても同じ結果になるので、報酬項の評価順に
依存しない。

NOTE: RewardManager は weight==0 の項をスキップするので、カリキュラムで weight を 0 から
      立ち上げる項だけに更新を任せると Phase 1 の間ずっと状態が更新されない。そのため
      **常に有効な termination 項** (:func:`..terminations.kick_finished`) からも
      :func:`kick_state` を呼び、毎ステップの更新を保証している。TerminationManager は
      RewardManager より先に走るので、報酬項が読む時点で状態は最新になっている。
"""

from __future__ import annotations

import math
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


def resolve_kick_direction_w(
    env: ManagerBasedRLEnv,
    command_name: str,
    ball_position_w: torch.Tensor,
) -> torch.Tensor:
    """Resolve a command term to a current world-frame kick unit direction.

    Likelihood map-target commands implement ``direction_from_ball`` and
    derive the direction from the current ball to their fixed global target.
    Legacy WalkKick commands have no such method and keep the original
    ``[sin(theta), cos(theta), speed]`` interpretation unchanged.
    """
    get_term = getattr(env.command_manager, "get_term", None)
    if callable(get_term):
        command_term = get_term(command_name)
        resolver = getattr(command_term, "direction_from_ball", None)
        if callable(resolver):
            return resolver(ball_position_w)

    command = env.command_manager.get_command(command_name)
    return torch.stack((command[:, 1], command[:, 0]), dim=-1)


def request_kick_state_reset(
    env: ManagerBasedRLEnv,
    env_ids,
) -> None:
    """Mark rows whose cached kick state must be rebuilt after an env reset.

    Isaac Lab can reset a subset of environments after rewards have populated
    the step cache, then ask the command manager to update again without
    incrementing ``common_step_counter``.  Recording the reset explicitly keeps
    those rows from reusing the preceding episode's geometry while retaining
    the one-update-per-step guarantee for every other row.
    """
    state = getattr(env, _ATTR, None)
    if state is None:
        return
    state["pending_reset"][env_ids] = True


def kick_state(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    command_name: str = "kick_direction",
    ball_name: str = "soccer_ball",
    track_ball: bool = False,
    v_thresh_target_frac: float = 0.0,
    v_thresh_floor: float = 0.0,
    physical_kick_detection: bool = False,
    kick_detection_foot_distance_threshold: float = 0.23,
    kick_detection_min_foot_speed_towards_ball: float = 0.2,
    kick_detection_velocity_change_threshold: float = 0.5,
    kick_detection_warmup_steps: int = 5,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> dict:
    """キック関連の共有状態を返す。同一ステップ内では一度しか更新しない。

    Args:
        v_thresh_target_frac: >0 にすると値 latch のトリガー速度を **指令速度 v_target に
            比例させ、env ごとに下げる**::

                v_thresh_eff = clamp(frac * v_target, min=v_thresh_floor, max=v_thresh)

            0.0 (既定) ではスカラーの ``v_thresh`` をそのまま使う = 従来挙動。

            **なぜ必要か**: 既定の閾値 0.8 m/s に対し walk_kick 系の指令帯は (0.25, 2.0)
            なので、**指令の下半分は「指令どおり正しく蹴ると latch が発火しない」**。
            latch しなければキック報酬は 1 つも払われず (項1-3 は kick_done ゲート)、
            kick_finished も発火しないので、弱いキックは学習上まったく報われない。
            結果として「指令を無視して強く蹴る」が構造的な最適解になる。閾値を指令に
            追従させると、弱い指令のときだけ低い閾値で latch できるようになる。

            **必ず比例にすること (切片つきの一次式にしないこと)**。``scale*v + offset``
            の形にすると帯の下端で閾値が指令を追い越す (0.5×0.25+0.2 = 0.325 > 0.25) ため、
            いちばん救いたい最弱の指令だけが救われないという逆転が起きる。比例なら
            frac < 1 である限り閾値は必ず指令より下に来る。

            frac の値は「指令の何割出れば蹴ったと認めるか」。0.6 は
            ``kick_direction`` の片側速度ゲート (``v_gate_frac``) の既定と同じ考え方で、
            指令の 6 割に届かないへなちょこ接触を latch させないための足切りになる。

        v_thresh_floor: 実効閾値の下限 [m/s]。歩行中に足がボールをかすっただけで
            誤 latch するのを防ぐ床。

            NOTE: 接触カウント (:data:`_TOUCH_DV_THRESH` = 0.15 m/s) より **上**に取ること。
                  0.2 なら「触ったが latch はしない」領域が残り、``ball_touch_count`` と
                  ``kick_rate`` の差として観測できる。0.15 以下にすると両者が区別できない。
            NOTE: 帯の下端の指令より **下**であることを必ず確認すること。床が指令を
                  上回ると、その指令では正しく蹴っても latch しない (上の逆転が復活する)。
                  指令帯 (0.25, 2.0) に対して床 0.2 は 0.05 m/s の余裕しかないので、
                  帯の下端を下げるときは床も一緒に下げること。

        track_ball: True にすると、値 latch 前は **P_kick を毎ステップ現在のボール位置から
            引き直す**。既定 (False) はエピソード開始時のボール位置に固定する従来挙動。

            ボールが動くタスク (リセット時に初速を与える ablation など) では固定できない。
            P_kick は「理想キック立ち位置」なので、ボールが転がったのに開始位置に残ると、
            approach_penalty / ball_avoidance が実際のボールとは違う点への整列を要求し、
            latch 後に G を固定する先も的外れな点になる。

            NOTE: このフラグは **kick_state を最初に呼ぶ項** が渡した値でその step の
                  状態が確定する (r_stance / alpha / v_thresh と同じ)。ただし P_kick を
                  計算するのは kick_state 自身だけで、報酬項はどれも出来上がった
                  ``P_kick`` / ``d_to_P_kick`` / ``G`` を読むだけなので、報酬項の全てに
                  配る必要はない。毎ステップ最初に走ることが保証されている
                  :func:`..terminations.kick_finished` (と、同じ値を持つ
                  :class:`..commands.BallFollowVelocityCommandCfg`) にだけ渡せばよい。

        physical_kick_detection: True のとき、絶対ボール速度ではなく、足がボールへ向かう
            接触候補とボール XY 速度ベクトルのステップ間変化の両方で latch する。
            初速を持つボールが触れていないのに latch するのを防ぐ moving-ball 用 opt-in。
            False (既定) は従来の ``v_ball > v_thresh`` を一切変更しない。

        kick_detection_foot_distance_threshold: 現在または前ステップの足―ボール 3D 距離の
            小さい方に対する接触候補の上限 [m]。

        kick_detection_min_foot_speed_towards_ball: 前ステップの足からボールへの方向へ進む
            足速度の下限 [m/s]。

        kick_detection_velocity_change_threshold: ボール XY 速度ベクトルのステップ間変化量の
            下限 [m/s]。

        kick_detection_warmup_steps: エピソード開始後に物理判定を遮断するステップ数。
            既定 5 は標準の 50 Hz 制御で 0.1 秒。

        r_max: None (既定) 以外を入れると、目標終端 G の作り方を **回り込み型** に
            切り替える。既定 (None) では従来どおり「ボールの真後ろ (キック線 R 上) を
            ボール側へ滑る点」。詳細は下の G の計算のコメント参照。

        orbit_beta: 回り込み型 G の先読み係数 (``r_max`` を入れたときだけ使う)。
            0 < beta < 1。小さいほど G がキック線の真後ろへ強く引き寄せる。

        overshoot_margin: overshoot 判定 (キック線 R の左右跨ぎ) の遊び [m]。
            0.0 (既定) では従来どおり「符号が反転したら即発火」。正の値を入れると、
            確定側と反対側へ **この距離まで** 入り込んでも発火しない。

        lateral_band: 終端の構え位置 (P_kick と、回り込み型 G の真後ろ極限) に
            **横方向のあそび (帯)** を持たせる。``(下端, 上端)`` [m]、符号は
            ``right_vec`` と同じで **正 = ロボットから見て右**。None (既定) では
            横成分ゼロ = 従来どおり「ボールを両足の真ん中に置く一点」を指令する。
            詳細は下の帯のブロックのコメント参照。

            NOTE: この 4 つも ``track_ball`` / ``v_thresh_target_*`` とは違い、
                  **kick_state を呼ぶ全ての項に配る必要がある**。G と overshoot は
                  その step の最初の呼び出しで確定するので、項ごとに違う値を渡すと
                  結果が評価順に依存する。

    NOTE: ``track_ball`` / ``v_thresh_target_*`` / ``physical_kick_detection`` と
          その閾値群はトリガーと latch だけを設定するため、``kick_finished`` と
          ``base_velocity`` の 2 か所に同じ値を配ればよい。一方、``r_max`` /
          ``orbit_beta`` / ``overshoot_margin`` / ``lateral_band`` は G や overshoot を
          変えるため、kick_state を呼ぶ全項に同じ値を配ること。
    """
    step = int(env.common_step_counter)
    state = getattr(env, _ATTR, None)
    pending_reset = None if state is None else state["pending_reset"].clone()
    has_pending_reset = pending_reset is not None and bool(pending_reset.any())
    if state is not None and state["step"] == step:
        if has_pending_reset:
            _reset_cached_episode_rows(
                env,
                state,
                pending_reset,
                r_stance=r_stance,
                alpha=alpha,
                v_thresh=v_thresh,
                command_name=command_name,
                ball_name=ball_name,
                v_thresh_target_frac=v_thresh_target_frac,
                v_thresh_floor=v_thresh_floor,
                r_max=r_max,
                orbit_beta=orbit_beta,
                lateral_band=lateral_band,
            )
        return state

    robot = env.scene["robot"]
    ball = env.scene[ball_name]
    device = env.device

    ball_pos_3d = ball.data.root_pos_w[:, :3]
    ball_pos = ball_pos_3d[:, :2]
    ball_vel = ball.data.root_lin_vel_w[:, :2]
    ball_vel_z = ball.data.root_lin_vel_w[:, 2]
    ball_z = ball.data.root_pos_w[:, 2]
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_vel = robot.data.root_lin_vel_w[:, :2]

    # Legacy command: [sin θ, cos θ, v_target] (θ is world frame).
    # Map-target command: [target_x_w, target_y_w, v_target], resolved through
    # the command-term protocol against the current true ball position.
    resolved_kick_dir = resolve_kick_direction_w(env, command_name, ball_pos)
    cmd = env.command_manager.get_command(command_name)
    v_target = cmd[:, 2]

    # ロボット胴体のヨー方向
    quat = robot.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    forward = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)

    if state is None:
        state = {
            "step": -1,
            # P_kick_base: 横のあそび (lateral_band) を **足す前** の終端点。
            # エピソード開始時のスナップショット (と track_ball の引き直し) はこちらが持ち、
            # 報酬項が読む "P_kick" は帯の横成分を足した後の点になる。
            # lateral_band=None のときは両者は常に同一。
            "P_kick_base": torch.zeros(env.num_envs, 2, device=device),
            "P_kick": torch.zeros(env.num_envs, 2, device=device),
            "init_side": torch.zeros(env.num_envs, device=device),  # 0 = 未確定
            "kick_done": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "overshoot_fired": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "overshoot_event": torch.zeros(env.num_envs, device=device),
            "tau_direction_frozen": torch.zeros(env.num_envs, device=device),
            # 符号付きの方向誤差 [rad]。**正 = ボールが指令方向より右**。
            # tau_direction_frozen は abs なので系統的な左右バイアスが見えない。
            # 報酬には使わず、メトリクス専用 (walk_kick.mdp.commands 参照)。
            "tau_signed_frozen": torch.zeros(env.num_envs, device=device),
            # True ball-to-target direction at the kick latch.  Map-target
            # directions change as the ball moves, so post-latch geometry must
            # use this snapshot rather than chase the flying ball.
            "kick_dir_frozen": torch.zeros(env.num_envs, 2, device=device),
            "kick_dir": torch.zeros(env.num_envs, 2, device=device),
            # latch を起こした蹴りの足 (0.0 = 左, 1.0 = 右)。メトリクス専用。
            "kick_foot_frozen": torch.zeros(env.num_envs, device=device),
            "v_ball_frozen": torch.zeros(env.num_envs, device=device),
            "v_ball_3d_frozen": torch.zeros(env.num_envs, device=device),
            "phi_frozen": torch.zeros(env.num_envs, device=device),
            "p_style_frozen": torch.zeros(env.num_envs, device=device),
            "apex_height": torch.zeros(env.num_envs, device=device),
            "prev_v_ball": torch.zeros(env.num_envs, device=device),
            "prev_ball_pos_w": torch.zeros(env.num_envs, 3, device=device),
            "prev_ball_vel_xy": torch.zeros(env.num_envs, 2, device=device),
            "prev_foot_pos_w": torch.zeros(env.num_envs, 2, 3, device=device),
            "touch_count": torch.zeros(env.num_envs, device=device),
            "touch_refractory": torch.zeros(env.num_envs, dtype=torch.int32, device=device),
            "extra_touch_event": torch.zeros(env.num_envs, device=device),
            "sole_height_last_touch": torch.zeros(env.num_envs, device=device),
            "sole_height_at_kick": torch.zeros(env.num_envs, device=device),
            # 軸足 (蹴っていない方の足) のボール相対位置。値 latch で凍結する。
            # plant_lon: キック方向成分 (+ = ボールより前)。plant_lat: 横方向の **絶対値**。
            "plant_lon_frozen": torch.zeros(env.num_envs, device=device),
            "plant_lat_frozen": torch.zeros(env.num_envs, device=device),
            # 蹴り足のワールド鉛直速度 [m/s]。値 latch で凍結する。+ = すくい上げ。
            "foot_vz_frozen": torch.zeros(env.num_envs, device=device),
            "G": torch.zeros(env.num_envs, 2, device=device),
            "p_walk": torch.zeros(env.num_envs, device=device),
            "tau_walk": torch.zeros(env.num_envs, device=device),
            "d_sole_to_ball": torch.zeros(env.num_envs, device=device),
            "d_sole_to_ball_mean": torch.zeros(env.num_envs, device=device),
            "p_kick_pose": torch.zeros(env.num_envs, device=device),
            "v_target": torch.zeros(env.num_envs, device=device),
            "pending_reset": torch.zeros(
                env.num_envs,
                dtype=torch.bool,
                device=device,
            ),
        }
        setattr(env, _ATTR, state)
        # 初回はまだエピソードが始まっていないので全 env を初期化対象にする
        just_reset = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    else:
        # reward / termination は episode_length_buf の加算後、_reset_idx の前に走るので、
        # 新エピソードの 1 歩目は episode_length_buf == 1 になる。
        just_reset = (env.episode_length_buf == 1) | state["pending_reset"]

    state["step"] = step
    state["v_target"] = v_target

    # Before the kick, map-target direction follows the current true ball.
    # From the first step after latch onward, all geometry/scoring shares the
    # direction captured on the latch step.  A newly reset row must not reuse
    # the preceding episode's latch before its flags are cleared below.
    use_frozen_direction = state["kick_done"] & (~just_reset)
    kick_dir = torch.where(
        use_frozen_direction.unsqueeze(-1),
        state["kick_dir_frozen"],
        resolved_kick_dir,
    )
    state["kick_dir"] = kick_dir
    # kick_dir rotated -90 degrees: R(-90)·(x, y) = (y, -x).
    right_vec = torch.stack([kick_dir[:, 1], -kick_dir[:, 0]], dim=-1)

    # ------------------------------------------------------------------ #
    # エピソード開始時のリセット: 凍結値・フラグ・P_kick・初期側符号
    # ------------------------------------------------------------------ #
    # 符号付き横距離 s: 後方レイ R からロボットがどちら側にいるか
    s = ((robot_pos - ball_pos) * right_vec).sum(dim=-1)

    if just_reset.any():
        # P_kick_base: R 上、ボールから後方 r_stance の点。既定ではエピソード終了まで固定
        # (track_ball=True のときだけ latch まで下のブロックが引き直す)。
        state["P_kick_base"][just_reset] = (ball_pos - r_stance * kick_dir)[just_reset]
        # init_side は未確定 (0) に戻す。確定は下の commit ブロックで行う。
        state["init_side"][just_reset] = 0.0
        state["kick_done"][just_reset] = False
        state["overshoot_fired"][just_reset] = False
        state["tau_direction_frozen"][just_reset] = 0.0
        state["tau_signed_frozen"][just_reset] = 0.0
        state["kick_dir_frozen"][just_reset] = 0.0
        state["kick_foot_frozen"][just_reset] = 0.0
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
        state["foot_vz_frozen"][just_reset] = 0.0

    # ------------------------------------------------------------------ #
    # 転がるボール用: latch 前は P_kick を現在のボール位置から毎ステップ引き直す。
    #
    # kick_done は下の値 latch ブロックでこの step ぶんが立つので、ここで読むのは
    # 前 step までの値。つまり latch したその step も更新対象に入り、P_kick は
    # 「蹴った瞬間のボール位置 − r_stance·kick_dir」で凍結される。飛翔後に G を戻す先
    # としてはそれが正しい (開始位置でも飛んでいった先でもない)。
    # ------------------------------------------------------------------ #
    if track_ball:
        state["P_kick_base"] = torch.where(
            state["kick_done"].unsqueeze(-1), state["P_kick_base"], ball_pos - r_stance * kick_dir
        )

    # ------------------------------------------------------------------ #
    # 横方向の帯 (lateral_band): 終端の構えに「横のあそび」を持たせる
    #
    # 蹴り足をキック線 R (ボールの真後ろに伸びる直線) の上に乗せるには、base を
    # 蹴り足と反対側へスタンス半幅 (股関節の横オフセット 0.096 m) だけずらして
    # 立つ必要がある。ただし最適なずらし量は振り足の内転ぶんだけ小さくなるので、
    # いくつが正解かは事前にはわからない。
    #
    # そこで一点 (横ずれ 0 = ボールを両足の真ん中に置く姿勢) を指令するのをやめ、
    # **帯**で指令する。帯の中では目標がロボットの今の横位置にそのまま追従するので、
    # 横へ引く力が消える。帯の外へ出たときだけ、いちばん近い帯の端へ引き戻す。
    # 帯の中のどこに落ち着くかを決めるのは kick_direction (キックの正確さ) だけに
    # なるので、学習中に方策が自分で最適な位置を見つける。
    #
    # 帯の端 0.096 は股関節の横オフセットそのもの = 幾何的な上限であって、
    # チューニングで選んだ数字ではない。ログ実績で weak / long_pass 系は 9/9 右足で
    # 蹴っているので、帯は左片側にする ((-0.096, 0.0)。right_vec は正が右なので
    # 負が左 = 右足をキック線に乗せる側)。
    #
    # 凍結: 横成分も kick_done で他と一緒に凍らせる。凍らせないと蹴った後も目標が
    # ロボットを追い続け、latch 後に固定したはずの G が漂ってしまう。
    # kick_done はこの時点ではまだ前 step ぶんなので、track_ball と同じく
    # 「蹴ったその step の値」で凍る。
    # ------------------------------------------------------------------ #
    if lateral_band is None:
        # 既定: 横成分なし。P_kick は P_kick_base そのもの = 従来と完全に同じ点。
        lateral = None
        state["P_kick"] = state["P_kick_base"]
    else:
        s_clamped = torch.clamp(s, min=lateral_band[0], max=lateral_band[1])
        lateral = s_clamped.unsqueeze(-1) * right_vec
        state["P_kick"] = torch.where(
            state["kick_done"].unsqueeze(-1),
            state["P_kick"],
            state["P_kick_base"] + lateral,
        )

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
    # ボールが既に転がっていると偽の dv が出る。just_reset のステップをカウントしない
    # ことでこれを塞いでいる (reset_ball が初速を与えるタスクではこのガードが必須。
    # 静止配置のタスクでも保険として効く)。
    # ------------------------------------------------------------------ #
    dv = v_ball - state["prev_v_ball"]
    speed_touch_candidate = (
        (dv > _TOUCH_DV_THRESH) & (state["touch_refractory"] == 0) & (~just_reset)
    )

    # 足リンクの位置。d_sole_to_ball と接触時足高さの両方で使う。
    foot_ids = _foot_body_ids(env, robot)
    foot_pos = robot.data.body_pos_w[:, foot_ids, :]  # (N, 2, 3)

    # reset 行はその時点の状態を履歴の基準にする。全 env を一括初期化せず、
    # episode_length_buf == 1 の行だけを更新するので partial reset と干渉しない。
    if just_reset.any():
        state["prev_ball_pos_w"][just_reset] = ball_pos_3d[just_reset]
        state["prev_ball_vel_xy"][just_reset] = ball_vel[just_reset]
        state["prev_foot_pos_w"][just_reset] = foot_pos[just_reset]

    if physical_kick_detection:
        per_foot_candidate = _physical_kick_candidates(
            foot_pos,
            state["prev_foot_pos_w"],
            ball_pos_3d,
            state["prev_ball_pos_w"],
            float(env.step_dt),
            kick_detection_foot_distance_threshold,
            kick_detection_min_foot_speed_towards_ball,
        )
        ball_velocity_change = (ball_vel - state["prev_ball_vel_xy"]).norm(dim=-1)
        warmup_finished = env.episode_length_buf >= kick_detection_warmup_steps
        physical_trigger = (
            per_foot_candidate.any(dim=-1)
            & (ball_velocity_change >= kick_detection_velocity_change_threshold)
            & warmup_finished
            & (~just_reset)
            & (~state["kick_done"])
        )
        touched = speed_touch_candidate | physical_trigger
    else:
        per_foot_candidate = None
        physical_trigger = None
        touched = speed_touch_candidate

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
    d_foot_to_ball = (foot_pos - ball_pos_3d.unsqueeze(1)).norm(dim=-1)  # (N, 2)
    kicking_foot = d_foot_to_ball.argmin(dim=1)  # (N,)
    if per_foot_candidate is not None:
        previous_distance = (
            state["prev_foot_pos_w"] - state["prev_ball_pos_w"].unsqueeze(1)
        ).norm(dim=-1)
        candidate_distance = torch.minimum(d_foot_to_ball, previous_distance)
        masked_distance = torch.where(
            per_foot_candidate,
            candidate_distance,
            torch.full_like(candidate_distance, torch.inf),
        )
        candidate_foot = masked_distance.argmin(dim=1)
        kicking_foot = torch.where(per_foot_candidate.any(dim=-1), candidate_foot, kicking_foot)
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

    # ------------------------------------------------------------------ #
    # 蹴り足のワールド鉛直速度 v_z [m/s]。値 latch で凍結する (foot_vz_frozen)。
    #
    # + = 接触の瞬間に足が上へ動いている = 「すくい上げ」。ボールを浮かせる運動量が
    # 反発 (ボールの restitution) ではなく **足の運動** から来ていることの直接の指標。
    # Isaac (e≈0.6) と MuJoCo/実機 (e≈0) で挙動が割れるのは、反発に頼った解が
    # 反発係数の消える環境で浮かなくなるため。この量を報酬に使うことで、浮かせる
    # メカニズムを反発非依存 (運動学依存) の側へ寄せる (:func:`..rewards.kick_foot_lift`)。
    #
    # NOTE: 足リンク原点の速度をそのまま採る。足裏中心は原点から z = −_SOLE_OFFSET に
    #       あるので厳密には足の角速度ぶん (ω × r) だけずれるが、sole_z (足裏高さ) を
    #       「リンク原点 − _SOLE_OFFSET」で近似しているのと同じ扱いに揃えてある。
    # ------------------------------------------------------------------ #
    foot_vel = robot.data.body_lin_vel_w[:, foot_ids, :]  # (N, 2, 3)
    foot_vz = foot_vel[torch.arange(env.num_envs, device=device), kicking_foot, 2]

    state["touch_count"] = state["touch_count"] + touched.float()
    # 2 回目以降の接触が起きたステップだけ 1。1 回目 (touch_count == 1) は無料。
    state["extra_touch_event"] = (touched & (state["touch_count"] >= 2.0)).float()

    state["touch_refractory"] = torch.where(
        touched,
        torch.full_like(state["touch_refractory"], _TOUCH_REFRACTORY_STEPS),
        torch.clamp(state["touch_refractory"] - 1, min=0),
    )
    state["prev_v_ball"] = v_ball

    # ------------------------------------------------------------------ #
    # トリガー閾値。既定はスカラー (従来挙動) だが、v_thresh_target_frac > 0 なら
    # 指令速度に比例する **per-env テンソル** になる。
    #   v_thresh_eff = clamp(frac * v_target, min=v_thresh_floor, max=v_thresh)
    # v_target は既にこの関数内で読んである (cmd[:, 2]) ので追加コストはほぼ無い。
    # 比較 (v_ball > v_thr) はスカラー・テンソルどちらでもそのまま通る。
    # ------------------------------------------------------------------ #
    if v_thresh_target_frac > 0.0:
        v_thr = torch.clamp(
            v_thresh_target_frac * v_target, min=v_thresh_floor, max=v_thresh
        )
    else:
        v_thr = v_thresh
    state["v_thresh_eff"] = v_thr if torch.is_tensor(v_thr) else torch.full_like(v_ball, v_thr)

    if physical_kick_detection:
        trigger = physical_trigger
    else:
        # 既定/static 経路は従来の式をそのまま維持する。
        trigger = (v_ball > v_thr) & (~state["kick_done"])

    if trigger.any():
        # τ_direction: ボールの飛翔方向と蹴り方向の角度誤差 [rad]
        # 水平投影で測るので、ボールが浮いていてもそのまま「狙った方位か」を意味する。
        ball_dir = ball_vel / (v_ball.unsqueeze(-1) + 1e-6)
        cos_err = torch.clamp((ball_dir * kick_dir).sum(dim=-1), min=-1.0, max=1.0)
        sin_err = ball_dir[:, 0] * kick_dir[:, 1] - ball_dir[:, 1] * kick_dir[:, 0]
        # 符号付き: **正 = ボールが指令方向より右へ出た**。
        # sin_err = cross_z(ball_dir, kick_dir) なので、ball_dir を反時計回りに
        # tau_signed だけ回すと kick_dir に重なる = ボールは kick_dir の時計回り側 = 右。
        tau_signed = torch.atan2(sin_err, cos_err)
        tau_direction = torch.abs(tau_signed)

        # φ: 仰角 [rad]。水平 = 0、真上 = π/2。負値 (打ち下ろし) もそのまま持つ。
        phi = torch.atan2(ball_vel_z, v_ball + 1e-6)
        v_ball_3d = torch.sqrt(v_ball**2 + ball_vel_z**2)

        state["tau_direction_frozen"] = torch.where(
            trigger, tau_direction, state["tau_direction_frozen"]
        )
        state["tau_signed_frozen"] = torch.where(trigger, tau_signed, state["tau_signed_frozen"])
        state["kick_dir_frozen"] = torch.where(
            trigger.unsqueeze(-1),
            kick_dir,
            state["kick_dir_frozen"],
        )
        # 蹴った足は d_foot_to_ball の argmin (foot_ids = [left, right]) なので 1 = 右。
        state["kick_foot_frozen"] = torch.where(
            trigger, kicking_foot.float(), state["kick_foot_frozen"]
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
        # 蹴り足の鉛直速度も plant_* と同じく **latch したステップの現在値** を採る。
        # latch = ボールが動き出した瞬間なので、そのステップの足速度がすくい上げの
        # 有無をそのまま表す。
        state["foot_vz_frozen"] = torch.where(trigger, foot_vz, state["foot_vz_frozen"])
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
    if r_max is None:
        # 従来の G: キック線 R (ボールの真後ろに伸びる直線) 上だけを動く点。
        reach = torch.clamp(alpha * dist_robot_ball, min=r_stance, max=0.5)
        G = ball_pos - reach.unsqueeze(-1) * kick_dir
    else:
        # ------------------------------------------------------------ #
        # 回り込み型の G (r_max を入れたときだけ)
        #
        # 従来の G はボールの真後ろにしか置けないので、ボールの正面にいるロボットへは
        # 「ボールを突き抜けて向こう側へ行け」という指令になる。回り込みは
        # ball_avoidance (近づくと罰) が遠回りに追い込むことで初めて成立していた。
        # ここでは **罰ではなく指令側** で回り込みを作る: G をボールを中心とする円弧の
        # 上に置き、ロボットの現在位置から少しだけキック線側へ寄せた点にする。
        # ロボットが G を追えば、そのままボールの周りを回ってキック線の後ろに着く。
        #
        # 記号:
        #   u     : ボール → ロボット の単位ベクトル (今ロボットがボールのどっち側にいるか)
        #   back  : −kick_dir。ボールから見て「キック線の後ろ」を指す単位ベクトル
        #   φ     : back から u への符号付き角度 (−π, π]。φ=0 でロボットは真後ろに居る
        #   φ_G   : orbit_beta * φ。beta < 1 なので G は必ずロボットより真後ろ寄りに出る
        #   ρ     : G のボールからの距離。真後ろ (φ=0) で r_stance、真正面 (|φ|=π) で r_max
        #
        # 半径は r_max で **直接** 指定できる。従来の 0.5 のハードコード上限や
        # alpha による距離連動ではなく、「正面から近づくときはボールから何 m 離れて
        # 回るか」をそのまま数字で書ける。
        #
        # φ=±π (ボールの真正面) では左右どちらへ回るかが φ の符号で決まり、
        # そこだけ G が不連続に飛ぶ。ただし真正面ちょうどはコイントスと同じで、
        # どちらに回っても等価なので実害は小さい (少しでもどちらかへ寄れば以降は連続)。
        # 逆に φ=0 (真後ろ、いちばん大事な仕上げの領域) では連続で、
        # ρ → r_stance・G → 従来の P_kick に一致する。
        # ------------------------------------------------------------ #
        to_robot = robot_pos - ball_pos
        u = to_robot / (to_robot.norm(dim=-1, keepdim=True) + 1e-6)
        back = -kick_dir
        phi = torch.atan2(
            back[:, 0] * u[:, 1] - back[:, 1] * u[:, 0],
            back[:, 0] * u[:, 0] + back[:, 1] * u[:, 1],
        )
        phi_G = orbit_beta * phi
        rho = r_stance + (r_max - r_stance) * phi.abs() / math.pi
        cos_g = torch.cos(phi_G)
        sin_g = torch.sin(phi_G)
        # back を φ_G だけ回した単位ベクトル (2D の標準的な回転)
        back_rot = torch.stack(
            [
                back[:, 0] * cos_g - back[:, 1] * sin_g,
                back[:, 0] * sin_g + back[:, 1] * cos_g,
            ],
            dim=-1,
        )
        G = ball_pos + rho.unsqueeze(-1) * back_rot
        if lateral is not None:
            # 帯の横成分を、真後ろ (φ=0) で 1、真正面 (|φ|=π) で 0 になるよう
            # フェードさせて足す。円弧を大きく回っている間は指令の形をまったく
            # 変えず、仕上げの真後ろでだけ帯つきの点 (= P_kick と同じ点) に
            # 一致させるため。φ=0 では ρ=r_stance かつ back_rot=−kick_dir なので
            # G = ball − r_stance·kick_dir + lateral となり、P_kick に厳密に一致する。
            fade = 1.0 - phi.abs() / math.pi
            G = G + fade.unsqueeze(-1) * lateral
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
    # overshoot_margin > 0 なら、確定側と反対側へこの距離まで入っても発火しない。
    # 0.0 (既定) では従来どおり符号反転で即発火。
    crossed = (s * state["init_side"]) < -overshoot_margin
    newly_fired = crossed & (~state["overshoot_fired"])
    state["overshoot_event"] = newly_fired.float()  # 発火したステップだけ 1 (1エピソード最大1回)
    state["overshoot_fired"] = state["overshoot_fired"] | crossed

    # ------------------------------------------------------------------ #
    # d_soleToBall: 左右の足裏のうちボールに近い方の距離
    # (foot_pos / d_foot_to_ball は接触時足高さの計算で既に求めてある)
    # ------------------------------------------------------------------ #
    state["d_sole_to_ball"] = d_foot_to_ball.min(dim=1).values

    # ------------------------------------------------------------------ #
    # d_soleToBall (両足平均)。min 版とは **別のキー** として持つ (既存項は触らない)。
    #
    # min 版は「どちらか片方の足がボールの近くにあるか」しか見ないので、
    #   * 綺麗なインサイドキックの構え (両足ともボール近傍)
    #   * 軸足を後ろに残して蹴り足だけ突き出した退行解
    # が同じ値になり区別できない。平均なら後者だけが大きくなる (実測見積もりで
    # 前者 ≈ 0.17-0.20 m、後者 ≈ 0.32 m)。「構えができているのに足がボールから
    # 遠い」ことを罰する :func:`..rewards.ball_avoidance_exec` 専用の量。
    # ------------------------------------------------------------------ #
    state["d_sole_to_ball_mean"] = d_foot_to_ball.mean(dim=1)

    state["p_style"] = p_style
    state["d_to_P_kick"] = (robot_pos - state["P_kick"]).norm(dim=-1)

    # 次ステップの物理判定用履歴。reset 行を含め、現在値で環境ごとに更新する。
    state["prev_ball_pos_w"].copy_(ball_pos_3d)
    state["prev_ball_vel_xy"].copy_(ball_vel)
    state["prev_foot_pos_w"].copy_(foot_pos)

    state["pending_reset"][just_reset] = False

    return state


def _reset_cached_episode_rows(
    env: ManagerBasedRLEnv,
    state: dict,
    reset_mask: torch.Tensor,
    *,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    command_name: str,
    ball_name: str,
    v_thresh_target_frac: float,
    v_thresh_floor: float,
    r_max: float | None,
    orbit_beta: float,
    lateral_band: tuple[float, float] | None,
) -> None:
    """Rebase reset rows without advancing cached state for other envs.

    This path is used only when Isaac Lab resets rows after the step cache was
    already populated. It constructs the same new-episode baseline as the
    regular ``just_reset`` path, but does not run latch/touch transitions a
    second time for non-reset rows.
    """
    env_ids = reset_mask.nonzero(as_tuple=False).flatten()
    if env_ids.numel() == 0:
        return

    robot = env.scene["robot"]
    ball = env.scene[ball_name]
    ball_pos_3d = ball.data.root_pos_w[env_ids, :3]
    ball_pos = ball_pos_3d[:, :2]
    ball_vel = ball.data.root_lin_vel_w[env_ids, :2]
    robot_pos = robot.data.root_pos_w[env_ids, :2]
    robot_vel = robot.data.root_lin_vel_w[env_ids, :2]

    all_ball_pos = ball.data.root_pos_w[:, :2]
    kick_dir = resolve_kick_direction_w(env, command_name, all_ball_pos)[env_ids]
    command = env.command_manager.get_command(command_name)
    v_target = command[env_ids, 2]
    right_vec = torch.stack((kick_dir[:, 1], -kick_dir[:, 0]), dim=-1)
    side = ((robot_pos - ball_pos) * right_vec).sum(dim=-1)

    p_kick_base = ball_pos - r_stance * kick_dir
    if lateral_band is None:
        lateral = None
        p_kick = p_kick_base
    else:
        lateral = torch.clamp(
            side,
            min=lateral_band[0],
            max=lateral_band[1],
        ).unsqueeze(-1) * right_vec
        p_kick = p_kick_base + lateral

    dist_robot_ball = (robot_pos - ball_pos).norm(dim=-1)
    if r_max is None:
        reach = torch.clamp(alpha * dist_robot_ball, min=r_stance, max=0.5)
        approach_position = ball_pos - reach.unsqueeze(-1) * kick_dir
    else:
        to_robot = robot_pos - ball_pos
        unit_to_robot = to_robot / (to_robot.norm(dim=-1, keepdim=True) + 1.0e-6)
        back = -kick_dir
        phi = torch.atan2(
            back[:, 0] * unit_to_robot[:, 1] - back[:, 1] * unit_to_robot[:, 0],
            back[:, 0] * unit_to_robot[:, 0] + back[:, 1] * unit_to_robot[:, 1],
        )
        phi_goal = orbit_beta * phi
        radius = r_stance + (r_max - r_stance) * phi.abs() / math.pi
        cos_goal = torch.cos(phi_goal)
        sin_goal = torch.sin(phi_goal)
        rotated_back = torch.stack(
            (
                back[:, 0] * cos_goal - back[:, 1] * sin_goal,
                back[:, 0] * sin_goal + back[:, 1] * cos_goal,
            ),
            dim=-1,
        )
        approach_position = ball_pos + radius.unsqueeze(-1) * rotated_back
        if lateral is not None:
            fade = 1.0 - phi.abs() / math.pi
            approach_position = approach_position + fade.unsqueeze(-1) * lateral

    zero_keys = (
        "kick_done",
        "overshoot_fired",
        "overshoot_event",
        "tau_direction_frozen",
        "tau_signed_frozen",
        "kick_dir_frozen",
        "kick_foot_frozen",
        "v_ball_frozen",
        "v_ball_3d_frozen",
        "phi_frozen",
        "p_style_frozen",
        "apex_height",
        "touch_count",
        "touch_refractory",
        "extra_touch_event",
        "sole_height_last_touch",
        "sole_height_at_kick",
        "plant_lon_frozen",
        "plant_lat_frozen",
        "foot_vz_frozen",
        "p_walk",
        "p_kick_pose",
    )
    for key in zero_keys:
        state[key][env_ids] = 0

    state["init_side"][env_ids] = torch.where(
        side.abs() > _INIT_SIDE_COMMIT_DIST,
        torch.sign(side),
        torch.zeros_like(side),
    )
    state["P_kick_base"][env_ids] = p_kick_base
    state["P_kick"][env_ids] = p_kick
    state["kick_dir"][env_ids] = kick_dir
    state["G"][env_ids] = approach_position
    state["v_target"][env_ids] = v_target

    robot_quat = robot.data.root_quat_w[env_ids]
    qw, qx, qy, qz = robot_quat.unbind(dim=-1)
    robot_yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy.square() + qz.square()),
    )
    forward = torch.stack((torch.cos(robot_yaw), torch.sin(robot_yaw)), dim=-1)
    state["p_style"][env_ids] = torch.clamp(
        (forward * kick_dir).sum(dim=-1),
        min=0.0,
        max=1.0,
    )
    to_goal = approach_position - robot_pos
    distance_to_goal = to_goal.norm(dim=-1)
    direction_to_goal = to_goal / (distance_to_goal.unsqueeze(-1) + 1.0e-6)
    state["tau_walk"][env_ids] = (robot_vel * direction_to_goal).sum(dim=-1)
    state["d_to_G"][env_ids] = distance_to_goal
    state["d_to_P_kick"][env_ids] = (robot_pos - p_kick).norm(dim=-1)

    foot_ids = _foot_body_ids(env, robot)
    foot_pos = robot.data.body_pos_w[env_ids][:, foot_ids, :]
    foot_distances = (foot_pos - ball_pos_3d.unsqueeze(1)).norm(dim=-1)
    state["d_sole_to_ball"][env_ids] = foot_distances.min(dim=1).values
    state["d_sole_to_ball_mean"][env_ids] = foot_distances.mean(dim=1)
    state["prev_v_ball"][env_ids] = ball_vel.norm(dim=-1)
    state["prev_ball_pos_w"][env_ids] = ball_pos_3d
    state["prev_ball_vel_xy"][env_ids] = ball_vel
    state["prev_foot_pos_w"][env_ids] = foot_pos

    if v_thresh_target_frac > 0.0:
        v_thresh_eff = torch.clamp(
            v_thresh_target_frac * v_target,
            min=v_thresh_floor,
            max=v_thresh,
        )
    else:
        v_thresh_eff = torch.full_like(v_target, v_thresh)
    state["v_thresh_eff"][env_ids] = v_thresh_eff
    state["pending_reset"][env_ids] = False


def _physical_kick_candidates(
    current_feet_pos: torch.Tensor,
    previous_feet_pos: torch.Tensor,
    current_ball_pos: torch.Tensor,
    previous_ball_pos: torch.Tensor,
    dt: float,
    max_foot_ball_distance: float,
    min_foot_speed_towards_ball: float,
) -> torch.Tensor:
    """足ごとの接触候補を source 3af2acc と同じテンソル式で返す。"""

    current_ball_pos = current_ball_pos.unsqueeze(1)
    previous_ball_pos = previous_ball_pos.unsqueeze(1)

    current_distance = (current_feet_pos - current_ball_pos).norm(dim=-1)
    previous_distance = (previous_feet_pos - previous_ball_pos).norm(dim=-1)
    foot_ball_distance = torch.minimum(current_distance, previous_distance)

    previous_foot_to_ball = previous_ball_pos - previous_feet_pos
    previous_foot_to_ball_direction = previous_foot_to_ball / (
        previous_foot_to_ball.norm(dim=-1, keepdim=True) + 1.0e-6
    )
    foot_velocity = (current_feet_pos - previous_feet_pos) / dt
    foot_speed_towards_ball = (foot_velocity * previous_foot_to_ball_direction).sum(dim=-1)

    return (foot_ball_distance <= max_foot_ball_distance) & (
        foot_speed_towards_ball >= min_foot_speed_towards_ball
    )


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
