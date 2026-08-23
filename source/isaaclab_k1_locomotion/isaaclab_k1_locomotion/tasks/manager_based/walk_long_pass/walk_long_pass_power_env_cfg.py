# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_long_pass の **飛距離重視** 版 (Stage 4 のみ)。

:class:`~.walk_long_pass_env_cfg.K1WalkLongPassEnvCfg` を継承し、「指令どおりの速度で
蹴る」から「とにかく遠くへ転がす」へ寄せたもの。Stage 1-3 は共通 (同じ観測・同じ
checkpoint から始める) なので、``STAGE=4`` で本タスクだけ回せばよい。

飛距離を伸ばす 3 つのてこ
-------------------------
転がしパスの飛距離は ``d = v_h² / 2a`` (a ≈ 1.0 m/s²、実機較正 1 点) なので、
**水平速度 v_h をいかに稼ぐか** がすべて。継承元は次の理由で v_h を抑えている:

1. ``kick_velocity_scaled`` は「指令速度への一致度」を測る Gaussian なので、
   **指令より速く蹴っても得にならない**。可変強度を学習するには正しい設計だが、
   飛距離を伸ばす圧にはならない。
2. ``kick_elevation`` (target 30°) は浮き球報酬。3D 速度が同じなら仰角のぶん
   水平成分が削られる (30° で 0.87 倍)。モジュール docstring が
   「浮き球報酬による意図的な威力劣化」と明記しているとおり。
3. 速度帯の上限が 5.0 m/s。それ以上を指令されないので、それ以上速く蹴る動機が無い。

そこで:

* ``kick_velocity_strong`` (= r_direction × v_ball、**生の水平速度に比例**) を有効化する。
  上限が無いので「速いほど得」がそのまま働く。継承元では None。
* ``kick_elevation`` の weight を下げる。転がしパスなら浮かせる必要は無く、
  浮かせるほど水平成分と転がり距離を失う。ゼロにはしない (足裏の入り方が変わって
  蹴り自体が崩れるのを避けるため)。
* 速度帯の終点を (3.2, 5.0) → (4.0, 6.0) へ。ハード壁は関節速度上限から出る足先速度
  ≈ 6.5 m/s (loop_shoot 実測) なので 6.0 は射程内。帯は
  :func:`~..walk_kick.mdp.curriculums.kick_rate_gated_speed_range` が kick_rate で
  開閉するので、届かなければ自動で止まる (一段で張り替えたときの
  「蹴らずに歩く」への収束は起きない)。

距離換算 (a = 1.0):

    v_h [m/s]   4.0    4.5    5.0    5.5    6.0
    d   [m]     8.0   10.1   12.5   15.1   18.0

NOTE: 継承元は「すくい型スイングの sim2real 特性まで含めて検証済み」の動作ファミリー。
      本タスクはそこから意図的に外れるので、実機での挙動は別途確認すること。
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import _KICK_STATE_PARAMS, _KICK_W_SCALE, _SIGMA_DIRECTION
from .walk_long_pass_env_cfg import K1WalkLongPassEnvCfg

# --------------------------------------------------------------------------- #
# 飛距離重視の速度帯 [m/s]。継承元は (3.2, 5.0)。
#
# 上限 5.5 はハード壁 (足先速度 ≈ 6.5) に対して余裕を取った値 (距離換算 15 m)。
# 初版は 6.0 にしたが、kick_velocity_strong と組むと帯が上がりすぎて崩れた
# (_POWER_STRONG_W のコメント参照)。帯はゲート付きカリキュラムが kick_rate を
# 見ながら進めるので、届かない場合はそこで止まるだけで破綻はしない。
# --------------------------------------------------------------------------- #
_POWER_SPEED_RANGE = (4.0, 5.5)

# kick_velocity_strong の weight (× _KICK_W_SCALE)。
#
# この項は r_direction × v_ball で、v_ball が 4-6 m/s なので素の値が他項 (0-1) より
# 一桁大きい。継承元の kick_direction が 6.0、kick_velocity_scaled が 4.0 なので、
# 同じ払い出し規模に収めるには 1/5 程度が目安。
#
# 0.8 + 帯上限 6.0 で回した 1 本目は 2000 iteration で崩れかけた:
#   iteration     1000    1400    1800    2000
#   mean_reward   2.51    3.16    2.02    0.14
#   転倒率        2.7%    3.8%    4.4%    6.3%
#   std           0.13    0.14    0.15    0.17
#   speed_max     3.60    4.08    4.56    4.81
# 「速いほど得」が無制限なので kick_rate ゲートが緩いうちに帯が上がり切り、
# 届かない速度を指令され続けて無理に振って転ぶ悪循環になっていた
# (std の上昇 = PPO が「今のやり方では報われない」と探索を広げているサイン)。
# 圧を半分にし、帯の上限も 6.0 -> 5.5 に下げる。
_POWER_STRONG_W = 0.4

# kick_elevation の weight (× _KICK_W_SCALE)。継承元は 5.0。
#
# 転がしパスでは浮きは不要だが、0 にすると足裏の入り方が変わって蹴り自体が
# 崩れうるので半分に留める。実機では sim より仰角が目減りするので、
# ここを下げると実機はほぼ転がしになる想定。
_POWER_ELEVATION_W = 2.5


@configclass
class K1WalkLongPassPowerEnvCfg(K1WalkLongPassEnvCfg):
    """飛距離重視版 Stage 4。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. 生の速度に比例する項を有効化する (継承元は None)
        #
        # この項にはカリキュラムが無いので weight 直指定でよい (他のキック項は
        # curriculum 側が weight を持っているので、そちらを書き換える必要がある)。
        self.rewards.kick_velocity_strong = RewTerm(
            func=mdp.kick_velocity_strong,
            weight=_POWER_STRONG_W * _KICK_W_SCALE,
            params={**_KICK_STATE_PARAMS, "sigma_direction": _SIGMA_DIRECTION},
        )

        # -- 2. 浮き球報酬を弱める
        #
        # weight の実体は curriculum (linear_reward_weight) 側が毎ステップ書き戻すので、
        # RewTerm.weight を直接いじっても潰される。**カリキュラムの start/end を書き換える**
        # のが正しい (Stage 4 では _freeze_fade_in_curricula が start=end に揃えてある)。
        _elev_w = _POWER_ELEVATION_W * _KICK_W_SCALE
        curr_elev = getattr(self.curriculum, "kick_elevation_weight", None)
        if curr_elev is not None:
            curr_elev.params["start_weight"] = _elev_w
            curr_elev.params["end_weight"] = _elev_w
        if self.rewards.kick_elevation is not None:
            self.rewards.kick_elevation.weight = _elev_w

        # -- 3. 速度帯の終点を上げる
        #
        # 開始点 (_LONG_PASS_SPEED_RANGE_START) は継承元のまま。ゲート付き
        # カリキュラムの end_range だけ差し替える。
        curr = getattr(self.curriculum, "kick_speed_range", None)
        if curr is not None:
            curr.params["end_range"] = _POWER_SPEED_RANGE


@configclass
class K1WalkLongPassPowerEnvCfg_PLAY(K1WalkLongPassPowerEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
