# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_long_pass_orbit の 3 つの改造 (回り込み G / 跨ぎの遊び / ボール物性 DR)。

いずれも共有コード側は **既定値で従来どおり** になるよう作ってあり、ここで明示的に
値を入れたタスクだけ挙動が変わる。既存タスクの cfg には一切触れていない。
"""

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

# --------------------------------------------------------------------------- #
# 改造 1+2: 回り込み型の目標終端 G と、キック線を跨ぐときの遊び
#
# r_max = 0.5:
#   ボールの真正面から近づくとき、G をボール中心・半径 0.5 m の円の上に置く。
#   従来の G は「ボールの真後ろ」にしか置けないので、正面にいるロボットへは
#   「ボールを突き抜けて向こう側へ行け」という指令になっていて、回り込みは
#   ball_avoidance (近づくと罰) が遠回りに追い込むことでしか成立しなかった。
#   半径を数字で直接書けるようになったので、罰ではなく指令で回り込みを作る。
#   真後ろに着けば ρ は r_stance (0.25) まで縮み、従来の P_kick と一致する。
#
# orbit_beta = 0.6:
#   G を「ロボットの今いる向きから 0.6 倍だけキック線側へ寄せた向き」に置く。
#   1 未満なので G は必ずロボットより真後ろ寄りにあり、追い続けると円弧に沿って
#   キック線の後ろへ収束する。1.0 にすると G がロボットの真横に張り付いて進まない。
#
# overshoot_margin = 0.25:
#   overshoot 罰 (キック線 R を跨いだら 1 回だけ罰) の遊び [m]。
#   **反対側の足でボールを蹴るには base がキック線を約 0.096 m 越えて立つ必要がある**
#   (スタンス幅 0.192 m の半分)。従来は跨いだ瞬間に無条件で罰していたので、
#   正しい蹴り方の一部まで罰していた。体幅ぶん + マージンまでは跨ぎを許し、
#   それを超えた「回り直し」だけを罰する。
# --------------------------------------------------------------------------- #
_ORBIT_PARAMS = {
    "r_max": 0.5,
    "orbit_beta": 0.6,
    "overshoot_margin": 0.25,
}

# kick_state を参照しうる報酬項の名前。cfg に無い項は getattr の None ガードで飛ばす
# (親の構成が変わっても取りこぼさないよう、使っていない項も並べてある)。
_KICK_STATE_REWARD_TERMS = (
    "kick_direction",
    "kick_velocity_scaled",
    "kick_velocity_strong",
    "kick_velocity_overshoot",
    "kick_elevation",
    "kick_loft",
    "kick_plant_foot",
    "kick_foot_lift",
    "kick_contact_height",
    "walk_speed",
    "approach_penalty",
    "ball_avoidance",
    "ball_avoidance_exec",
    "kick_pose_overshoot",
    "kick_latch_bonus",
    "extra_ball_touch",
)


def apply_orbit_params(cfg) -> None:
    """kick_state を共有する全ての項に :data:`_ORBIT_PARAMS` を配る。

    G と overshoot 判定は **その step で最初に kick_state を呼んだ項の値で確定する**
    ので、1 つでも取りこぼすと結果が報酬項の評価順に依存してしまう
    (walk_long_pass の ``_apply_v_thresh`` とまったく同じ理由)。
    **報酬項を追加し終えた後に呼ぶこと。**

    キック項が 1 つも無い段 (walk phase) でも呼んで構わない。None ガードで
    全部飛ばされるので何も起きない。
    """
    base_velocity = getattr(cfg.commands, "base_velocity", None)
    if base_velocity is not None:
        for _key, _val in _ORBIT_PARAMS.items():
            setattr(base_velocity, _key, _val)

    kick_finished = getattr(cfg.terminations, "kick_finished", None)
    if kick_finished is not None:
        kick_finished.params.update(_ORBIT_PARAMS)

    for _name in _KICK_STATE_REWARD_TERMS:
        _term = getattr(cfg.rewards, _name, None)
        if _term is not None:
            _term.params.update(_ORBIT_PARAMS)


# --------------------------------------------------------------------------- #
# 改造 3: ボールまわりの 4 つのドメインランダマイゼーション
# --------------------------------------------------------------------------- #

# (1) 足の反発係数。
#
# 地形マテリアルは restitution_combine_mode="multiply" かつ反発 0 なので、
# 足↔地面の反発は足側に何を入れても 0 のまま = 歩容には影響しない。
# 一方 **足↔ボールは average** で効くので、実効反発は (e_ball + e_foot)/2 になる。
# つまりこの範囲はボールの飛び方だけを振る。
_FOOT_RESTITUTION_RANGE = (0.3, 0.7)

# (2) ボールのマテリアル。
#
# 実球は空気圧・表面の摩耗・床材でこのくらい振れる。反発の下限 0.2 は
# 「ほとんど跳ねない体育館の床 + 空気の抜けた球」、上限 0.7 は新品の 5 号球。
_BALL_STATIC_FRICTION_RANGE = (0.3, 0.7)
_BALL_DYNAMIC_FRICTION_RANGE = (0.2, 0.5)
_BALL_RESTITUTION_RANGE = (0.2, 0.7)
_BALL_MATERIAL_BUCKETS = 64

# ボール質量のスケール。5 号球の公称 410-450 g のばらつき + 吸水ぶん。
_BALL_MASS_SCALE_RANGE = (0.9, 1.15)

# (3) 初期回転。
#
# 初速を持つボールには転がりと辻褄の合う角速度 ω = (ẑ × v)/R を入れ、
# さらに 3D のランダム軸まわりに 0-5 rad/s を足す。実機のボールは前の蹴りや
# 床の凹凸で必ず何かしら回っており、回転は接触時の接線インパルスを通じて
# 跳ね返り方向を変える。回転ゼロだけで学習すると接触モデルに寄りかかる。
_RAND_SPIN_RANGE = (0.0, 5.0)

# (4) 転がりの減速。
#
# PhysX の球には転がり摩擦が無いので、放っておくと減速せずに転がり続ける。
# angular_damping で近似する: v = 2 m/s の転がりで減速はおよそ 1 m/s² 相当。
# 指数減衰なので高速域では実際の転がり摩擦より強めに効く。
#
# NOTE: long_pass の docstring にある「速度 → 距離」の対応 (d = v²/2a, a ≈ 1.0) は
#       実機の 1 点計測を sim の等減速に当てはめたもの。この変更で sim 側の減速が
#       変わるので、**対応は実測し直しが必要**。帯 (3.2, 5.0) の根拠もそこに乗っている。
_BALL_ANGULAR_DAMPING = 0.5


def apply_ball_param_dr(cfg) -> None:
    """ボールまわりの 4 点セットを適用する (足の反発 / 物性 / 初期回転 / 転がり減速)。

    ボールが居ない段 (walk phase) から呼んでも安全なように、``soccer_ball`` と
    ``reset_ball`` の有無を見てから触る。
    """
    # -- (1) 足の反発係数 ------------------------------------------------ #
    cfg.events.physics_material.params["restitution_range"] = _FOOT_RESTITUTION_RANGE

    if getattr(cfg.scene, "soccer_ball", None) is None:
        return

    # -- (2) ボールのマテリアルと質量 ------------------------------------ #
    #
    # mode="startup" なので env ごとに固定値が 1 回だけ割り当てられる (4096 env あれば
    # 分布としては十分)。半径だけは spawn 後に env ごとへ変えられないので固定のまま。
    cfg.events.ball_physics_material = EventTerm(
        func=loco_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("soccer_ball"),
            "static_friction_range": _BALL_STATIC_FRICTION_RANGE,
            "dynamic_friction_range": _BALL_DYNAMIC_FRICTION_RANGE,
            "restitution_range": _BALL_RESTITUTION_RANGE,
            "num_buckets": _BALL_MATERIAL_BUCKETS,
        },
    )
    cfg.events.ball_mass = EventTerm(
        func=loco_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("soccer_ball"),
            "mass_distribution_params": _BALL_MASS_SCALE_RANGE,
            "operation": "scale",
        },
    )

    # -- (3) 初期回転 ---------------------------------------------------- #
    reset_ball = getattr(cfg.events, "reset_ball", None)
    if reset_ball is not None:
        reset_ball.params["spin_from_speed"] = True
        reset_ball.params["rand_spin_range"] = _RAND_SPIN_RANGE

    # -- (4) 転がりの減速 ------------------------------------------------ #
    cfg.scene.soccer_ball.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        angular_damping=_BALL_ANGULAR_DAMPING,
    )
