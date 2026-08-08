# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""トルク-速度カーブ (T-N カーブ) 付きアクチュエータ。

**出典 / 引用元**

本ファイルの :class:`BoosterDelayedPDActuator` および
:class:`BoosterDelayedPDActuatorCfg` は Booster Robotics 公式の学習リポジトリ
``github.com/BoosterRobotics/booster_train`` からの移植である。

- 対象ファイル: ``source/booster_train/booster_train/assets/robots/actuator.py``
- commit: ``651b7a5`` "update robot joint config for K1"
- author: wufeng <wufeng@boosterobotics.com>

**なぜこれが必要か**

isaaclab 標準の :class:`~isaaclab.actuators.DelayedPDActuator` は
``effort_limit`` (トルク上限) と ``velocity_limit`` (速度上限) を *独立に* クリップする。
つまり sim 上では「関節が最高速で回りながら同時に最大トルクを出す」ことが許されてしまう。

実機の BLDC モータはそんな挙動をしない。速度が上がるほど逆起電力でトルクが落ちる
(いわゆる T-N カーブ)。このカーブが無いまま学習すると、方策は「高速 × 最大トルク」という
sim にしか存在しない幻の性能領域に最適化される。特にキックのような最大出力動作は
その領域を全力で使いに行くため、sim では綺麗に蹴れるのに実機では全く同じ動作が出ない、
という sim-to-real ギャップの主要因になる。

そこで区分線形の T-N カーブを入れる:

- ``|v| <= knee_point_velocity``  : ``effort_limit`` まで出せる (定トルク領域)
- ``knee_point_velocity < |v| < velocity_limit`` : 線形にトルク上限が減衰
- ``|v| >= velocity_limit``       : トルク上限 0 (無負荷回転速度)

**K1 の関節ごとのモータ機種と定数** (booster_train の ``BoosterJoint*`` クラス由来)

============  ======  ============  ==============  ====================
関節          機種    effort_limit  velocity_limit  knee_point_velocity
============  ======  ============  ==============  ====================
Hip_Pitch     E6408   68.0          14.66           1.88
Hip_Roll      E4315   76.0          12.57           2.62
Hip_Yaw       E4310   38.3          17.59           7.85
Knee_Pitch    E6416   112.0         12.57           2.09
Ankle_Pitch   E4310   38.3          17.59           7.85
Ankle_Roll    E4310   38.3          17.59           7.85
============  ======  ============  ==============  ====================

(参考: 上半身は Arm=R14 knee 5.24 / Neck=HT4438 knee 10.47、
K1 では未使用だが T1 系は E8116 knee 6.28 / E8112 knee 7.54 / DM4310 knee 41.89)

本リポジトリでは既存の歩行が壊れないよう、``effort_limit`` / ``velocity_limit`` /
``stiffness`` / ``damping`` / ``armature`` は既存値をそのまま使い、
**追加するのは knee_point_velocity だけ** としている (rough_env_cfg.py 参照)。
"""

from __future__ import annotations

import torch
from collections.abc import Sequence  # noqa: F401  (型注釈の互換のため)

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

##
# K1 の機種別 knee_point_velocity 定数 [rad/s]
# 値の出典は booster_train の BoosterJoint* クラス (上表参照)。
##

KNEE_POINT_VELOCITY_E8116: float = 6.28   # T1 Knee (K1 では未使用、参考値)
KNEE_POINT_VELOCITY_E6408: float = 1.88   # K1 Hip_Pitch
KNEE_POINT_VELOCITY_E4315: float = 2.62   # K1 Hip_Roll
KNEE_POINT_VELOCITY_E4310: float = 7.85   # K1 Hip_Yaw / Ankle_Pitch / Ankle_Roll
KNEE_POINT_VELOCITY_E6416: float = 2.09   # K1 Knee_Pitch
KNEE_POINT_VELOCITY_R14: float = 5.24     # K1 Arm (現状アクチュエータ未定義)
KNEE_POINT_VELOCITY_HT4438: float = 10.47  # K1 Neck (現状アクチュエータ未定義)

# 脚 (Hip x3 + Knee) の関節名正規表現 → knee_point_velocity
K1_LEG_KNEE_POINT_VELOCITY: dict[str, float] = {
    ".*_Hip_Pitch": KNEE_POINT_VELOCITY_E6408,   # E6408
    ".*_Hip_Roll": KNEE_POINT_VELOCITY_E4315,    # E4315
    ".*_Hip_Yaw": KNEE_POINT_VELOCITY_E4310,     # E4310
    ".*_Knee_Pitch": KNEE_POINT_VELOCITY_E6416,  # E6416
}

# 足首 (Ankle Pitch/Roll)。K1 は両方とも E4310。
# 注: 公式は平行リンク (BoosterK1AnkleParaWrapperCfg) で armature を 2 倍にしているが、
#     本リポジトリでは既存の armature 値を優先するため knee_point_velocity のみ使う。
K1_ANKLE_KNEE_POINT_VELOCITY: float = KNEE_POINT_VELOCITY_E4310


class BoosterDelayedPDActuator(DelayedPDActuator):
    """速度依存のトルククリップ (区分線形 T-N カーブ) を持つ遅延付き PD アクチュエータ。

    :class:`~isaaclab.actuators.DelayedPDActuator` の遅延処理はそのまま継承し、
    :meth:`_clip_effort` だけを差し替えて速度依存のトルク上限を適用する。

    トルク上限 (絶対値) は測定された関節速度 ``|v|`` の関数として:

    .. math::

        \\tau_{max}(|v|) = clip\\left(
            \\tau_{limit} \\cdot \\frac{v_{max} - |v|}{v_{max} - v_{knee}},
            0, \\tau_{limit}\\right)

    ここで :math:`\\tau_{limit}` = ``effort_limit``、:math:`v_{max}` = ``velocity_limit``、
    :math:`v_{knee}` = ``knee_point_velocity``。

    ``knee_point_velocity`` を指定しない場合は ``velocity_limit`` と同値になり、
    速度依存の減衰は実質無効 (= 従来の箱型クリップ) になる。

    移植元: booster_train ``assets/robots/actuator.py`` (commit 651b7a5)
    """

    cfg: BoosterDelayedPDActuatorCfg
    """アクチュエータモデルの設定。"""

    def __init__(self, cfg: BoosterDelayedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # T-N カーブのニー速度。未指定なら velocity_limit と同値 = 減衰なし。
        self.knee_point_velocity = self._parse_joint_parameter(cfg.knee_point_velocity, self.velocity_limit)
        self.knee_point_velocity = torch.clamp(self.knee_point_velocity, min=0.0)
        # ニー速度が velocity_limit を超えるとカーブが破綻するので必ず内側に収める。
        self.knee_point_velocity = torch.minimum(self.knee_point_velocity, self.velocity_limit)
        # 速度依存クリップ用に、直近の実測関節速度を保持するバッファ。
        self._joint_vel = torch.zeros_like(self.computed_effort)
        # 定トルク領域から無負荷速度までの幅。0 除算回避のため下限を入れる。
        v_knee = self.knee_point_velocity
        v_max = self.velocity_limit
        self._denom = (v_max - v_knee).clamp(min=1e-6)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # _clip_effort は基底の compute の中から呼ばれ、関節速度を受け取れないので
        # ここで実測速度をキャッシュしておく (DCMotor と同じ手口)。
        self._joint_vel[:] = joint_vel
        # 遅延処理 + PD 計算は基底実装に任せる。
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # 区分線形の T-N カーブ:
        # - |v| <= knee_point_velocity        -> effort_limit までフルトルク
        # - knee_point_velocity < |v| < v_max -> 線形減衰
        # - |v| >= velocity_limit             -> 0
        joint_vel_abs = self._joint_vel.abs()
        v_max = self.velocity_limit
        tau_max = self.effort_limit

        # velocity_limit が非有限 / 非正の関節はカーブが定義できないので退避処理する。
        non_positive_vmax = v_max <= 0.0
        non_finite_vmax = ~torch.isfinite(v_max)

        # v_max == v_knee の関節でも 0 除算しないよう _denom は下限クランプ済み。
        tau_linear = tau_max * (v_max - joint_vel_abs) / self._denom
        max_effort = tau_linear.clamp(min=0.0).clamp(max=tau_max)

        # velocity_limit = inf (速度制限なし) の関節は従来通りの箱型クリップに戻す。
        max_effort = torch.where(non_finite_vmax, tau_max, max_effort)
        # velocity_limit <= 0 は「動かせない」ので出力トルク 0。
        max_effort = torch.where(non_positive_vmax, torch.zeros_like(max_effort), max_effort)

        return torch.clip(effort, min=-max_effort, max=max_effort)


@configclass
class BoosterDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """:class:`BoosterDelayedPDActuator` の設定。

    ``DelayedPDActuatorCfg`` に ``knee_point_velocity`` を足しただけなので、
    既存の ``effort_limit`` / ``velocity_limit`` / ``stiffness`` / ``damping`` /
    ``armature`` / ``min_delay`` / ``max_delay`` はそのまま使える。

    移植元: booster_train ``assets/robots/actuator.py`` (commit 651b7a5) の
    ``BoosterDelayedActuatorCfg`` / ``BoosterDelayedPDActuatorCfg``。
    ただし本リポジトリでは既存の limit 値を維持したいので、
    機種定義から limit を丸ごと差し込む ``booster_joint_cfgs`` の仕組みは移植していない
    (機種ごとの定数は本ファイル冒頭の表と定数群に残してある)。
    """

    class_type: type = BoosterDelayedPDActuator

    knee_point_velocity: dict[str, float] | float | None = None
    """T-N カーブのニー速度 [rad/s]。

    ``|joint_vel|`` がこの値以下なら ``effort_limit`` までフルトルクを出せる。
    これを超えると ``velocity_limit`` で 0 になるまで線形にトルク上限が下がる。
    None の場合は ``velocity_limit`` と同値 = 速度依存の減衰なし。
    """
