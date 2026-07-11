# Booster公式 (booster_train) の BoosterDelayedPDActuator の移植。
# https://github.com/BoosterRobotics (booster_train/source/booster_train/booster_train/assets/robots/actuator.py)
#
# DelayedPDActuator (遅延付きPD) に、実機モータの区分線形 T-N カーブ
# (トルク-速度曲線) によるトルク制限を追加したもの:
#   - |v| <= knee_point_velocity : effort_limit までフルトルク
#   - knee_point_velocity < |v| < velocity_limit : 線形に減衰
#   - |v| >= velocity_limit : トルク 0

from __future__ import annotations

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class BoosterDelayedPDActuator(DelayedPDActuator):
    """速度依存トルククリップ (T-Nカーブ) 付きの遅延PDアクチュエータ。"""

    cfg: BoosterDelayedPDActuatorCfg

    def __init__(self, cfg: "BoosterDelayedPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # T-Nカーブの折れ点速度。未指定なら velocity_limit (= 減衰なし) にフォールバック。
        self.knee_point_velocity = self._parse_joint_parameter(cfg.knee_point_velocity, self.velocity_limit)
        self.knee_point_velocity = torch.clamp(self.knee_point_velocity, min=0.0)
        self.knee_point_velocity = torch.minimum(self.knee_point_velocity, self.velocity_limit)
        # 速度ベースのクリップに使うバッファ
        self._joint_vel = torch.zeros_like(self.computed_effort)
        v_knee = self.knee_point_velocity
        v_max = self.velocity_limit
        self._denom = (v_max - v_knee).clamp(min=1e-6)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # 速度依存クリップ用に現在の関節速度をキャッシュしてから、遅延PDの計算に渡す
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # 区分線形 T-N カーブ:
        # - effort_limit: |v| <= knee_point_velocity での最大トルク
        # - velocity_limit: |v| == velocity_limit でトルクが 0 になる速度
        joint_vel_abs = self._joint_vel.abs()
        v_max = self.velocity_limit
        tau_max = self.effort_limit

        # velocity_limit が非有限/非正の関節は従来の箱型クリップ (またはゼロ) にフォールバック
        non_positive_vmax = v_max <= 0.0
        non_finite_vmax = ~torch.isfinite(v_max)

        tau_linear = tau_max * (v_max - joint_vel_abs) / self._denom
        max_effort = tau_linear.clamp(min=0.0).clamp(max=tau_max)

        max_effort = torch.where(non_finite_vmax, tau_max, max_effort)
        max_effort = torch.where(non_positive_vmax, torch.zeros_like(max_effort), max_effort)

        return torch.clip(effort, min=-max_effort, max=max_effort)


@configclass
class BoosterDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """:class:`BoosterDelayedPDActuator` 用の設定。"""

    class_type: type = BoosterDelayedPDActuator

    knee_point_velocity: dict[str, float] | float | None = None
    """T-Nカーブの折れ点速度 [rad/s]。None なら velocity_limit (速度依存の減衰なし)。"""
