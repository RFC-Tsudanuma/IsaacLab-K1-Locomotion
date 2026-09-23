# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""起き上がり (get-up) ポリシー用の報酬関数群。

各関数は :class:`isaaclab.managers.RewardTermCfg` に渡して使う。歩行用の rewards.py とは
独立に置き、起き上がり課題 (寝た姿勢 → 立位) 向けの項をまとめている。

含まれる報酬:
  - :func:`base_height_increase`    : base 高さが前ステップより高くなった分への報酬 (進捗報酬)
  - :func:`base_height`             : base 高さそのものへの報酬 (立つほど高い)
  - :func:`head_height`             : 頭の高さへの報酬
  - :func:`feet_ground_contact`     : 足裏が接地していることへの報酬
  - :func:`upright_posture`         : 上体 (Trunk) がまっすぐ (鉛直) であることへの報酬
  - :func:`body_symmetry`           : 全身の姿勢が左右対称であることへの報酬 (mirror loss 不要)
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# 内部ヘルパ
# ---------------------------------------------------------------------------
def _ground_height(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg | None) -> float | torch.Tensor:
    """地面の高さ (z)。raycaster センサがあればその平均高さ、無ければ 0 を返す。

    rough 地形では地面が z=0 とは限らないため、height_scanner (RayCaster) を渡すと
    各 env 直下の地面高さで補正できる。flat では ``sensor_cfg=None`` で 0 を使う。
    """
    if sensor_cfg is None:
        return 0.0
    sensor = env.scene[sensor_cfg.name]
    return torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)


def _upright_factor(asset) -> torch.Tensor:
    """直立度 [0,1]。``projected_gravity_b`` の z 成分が -1 (完全直立) で 1、
    0 (真横) で 0、+1 (上下反転/handstand) で 0 を返す。

    高さ系報酬 (base/head/increase) にこの係数を掛けることで、「反転して体を高く
    持ち上げる (逆立ち) と高さ報酬が稼げてしまう」exploit を防ぐ。上体が上向きに
    まっすぐなときだけ高さが報酬になる。
    """
    g_z = asset.data.projected_gravity_b[:, 2]
    return torch.clamp(-g_z, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 1. base 高さの増加 (進捗報酬)
# ---------------------------------------------------------------------------
def base_height_increase(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    only_increase: bool = True,
    require_upright: bool = False,
) -> torch.Tensor:
    """base (Trunk) の高さが前ステップより高くなった分を報酬にする進捗報酬。

    寝た姿勢から起き上がる途中の「今より少しでも高くなる」動きを継続的に評価するための項。
    値は 1 ステップあたりの高さ変化 (m) なので小さい。weight は他項と桁を合わせて調整すること。

    Args:
        asset_cfg: 対象アセット (root 高さを使う)。
        sensor_cfg: 地面高さ補正用の RayCaster センサ。None なら地面 z=0 とみなす。
        only_increase: True なら「高くなった (正)」分のみ報酬にし、下がった場合は 0。
                       False なら下降を負の報酬として与える。
        require_upright: True なら直立度 (_upright_factor) を掛け、反転して持ち上げても
                         報酬にならないようにする (handstand exploit 対策)。
    """
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - _ground_height(env, sensor_cfg)

    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    key = "getup_base_height_prev"
    if key not in env._custom_buffers:
        env._custom_buffers[key] = height.clone()

    prev_height = env._custom_buffers[key]
    delta = height - prev_height
    if only_increase:
        delta = torch.clamp(delta, min=0.0)

    # reset した env は前値が不連続になるので当ステップの報酬を 0 にする。
    reset_mask = env.reset_buf > 0
    delta = torch.where(reset_mask, torch.zeros_like(delta), delta)

    if require_upright:
        delta = delta * _upright_factor(asset)

    env._custom_buffers[key] = height.clone()
    return delta


# ---------------------------------------------------------------------------
# 2. base 高さそのもの
# ---------------------------------------------------------------------------
def base_height(
    env: ManagerBasedRLEnv,
    target_height: float = 0.6,
    min_height: float = 0.2,
    exponent: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    require_upright: bool = False,
) -> torch.Tensor:
    """base (Trunk) の高さそのものへの報酬。立位に近いほど高い。

    ``min_height`` から ``target_height`` への正規化高さを ``exponent`` 乗し、
    ``target_height`` 以上で 1 に飽和する (跳ね上がりを過剰に報酬しない)。

    Args:
        target_height: 立位時の目標 base 高さ [m]。ここで報酬が 1 に飽和する。
        min_height: 報酬が 0 になる下限高さ [m] (寝た姿勢相当)。
        exponent: 正規化高さに適用する指数。1.0 は線形、2.0 は高い領域を重視する二次曲線。
        sensor_cfg: 地面高さ補正用の RayCaster センサ。None なら地面 z=0。
        require_upright: True なら直立度 (_upright_factor) を掛け、反転して trunk を
                         高く上げても報酬にならないようにする (handstand exploit 対策)。
    """
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - _ground_height(env, sensor_cfg)
    normalized_height = torch.clamp((height - min_height) / (target_height - min_height), 0.0, 1.0)
    reward = normalized_height.pow(exponent)
    if require_upright:
        reward = reward * _upright_factor(asset)
    return reward


# ---------------------------------------------------------------------------
# 3. 頭の高さ
# ---------------------------------------------------------------------------
def head_height(
    env: ManagerBasedRLEnv,
    target_height: float = 0.9,
    min_height: float = 0.2,
    exponent: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Head.*"),
    sensor_cfg: SceneEntityCfg | None = None,
    require_upright: bool = False,
) -> torch.Tensor:
    """頭リンクの高さへの報酬。頭が高く持ち上がるほど高い。

    ``asset_cfg.body_names`` に複数マッチする場合は最も高いリンクの高さを使う。
    ``min_height`` → ``target_height`` の正規化高さを ``exponent`` 乗し、以上で 1 に飽和する。

    Args:
        target_height: 立位時の目標頭高さ [m]。ここで報酬が 1 に飽和する。
        min_height: 報酬が 0 になる下限高さ [m]。
        exponent: 正規化高さに適用する指数。1.0 は線形、2.0 は高い領域を重視する二次曲線。
        asset_cfg: 頭リンクを指す body_names を持つアセット設定。
        sensor_cfg: 地面高さ補正用の RayCaster センサ。None なら地面 z=0。
        require_upright: True なら直立度 (_upright_factor) を掛け、反転して頭を高く
                         上げても報酬にならないようにする (handstand exploit 対策)。
    """
    asset = env.scene[asset_cfg.name]
    # body_ids で指定された (1個以上の) リンクのうち最も高い z を頭高さとする。
    head_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].max(dim=1).values
    height = head_z - _ground_height(env, sensor_cfg)
    normalized_height = torch.clamp((height - min_height) / (target_height - min_height), 0.0, 1.0)
    reward = normalized_height.pow(exponent)
    if require_upright:
        reward = reward * _upright_factor(asset)
    return reward


def head_height_exp_reward(
    env: ManagerBasedRLEnv,
    min_height: float = 0.0,
    target_height: float = 0.9,
    sharpness: float = 4.0,
    double_scale_height: float = 0.35,
    triple_scale_height: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Head.*"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward absolute head height with a bounded, staged exponential curve.

    Unlike the upright-gated head reward, this term supplies a gradient while
    rising from a bridge. The height component saturates at ``target_height``;
    its scale doubles and triples after the two height thresholds to make
    continued upper-body rise more valuable than remaining in a low bridge.
    """
    if target_height <= min_height:
        raise ValueError("target_height must be greater than min_height")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive")
    if not min_height <= double_scale_height < triple_scale_height <= target_height:
        raise ValueError("scale heights must satisfy min <= double < triple <= target")

    asset = env.scene[asset_cfg.name]
    head_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].max(dim=1).values
    height = head_z - _ground_height(env, sensor_cfg)
    progress = torch.clamp((height - min_height) / (target_height - min_height), 0.0, 1.0)
    height_reward = torch.expm1(sharpness * progress) / math.expm1(sharpness)
    scale = torch.where(
        height >= triple_scale_height,
        3.0,
        torch.where(height >= double_scale_height, 2.0, 1.0),
    )
    return scale * height_reward


def low_head_height_penalty(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Head.*"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize ground-relative head height below the first get-up phase threshold."""
    if minimum_height <= 0.0:
        raise ValueError("minimum_height must be positive")
    asset = env.scene[asset_cfg.name]
    head_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].max(dim=1).values
    height = head_z - _ground_height(env, sensor_cfg)
    return torch.clamp((minimum_height - height) / minimum_height, 0.0, 1.0)


def timeout_below_head_height(
    env: ManagerBasedRLEnv,
    head_cfg: SceneEntityCfg,
    max_head_height: float | None = 0.6,
    max_trunk_height: float | None = None,
    timeout_term_name: str = "time_out",
    success_term_name: str = "success",
) -> torch.Tensor:
    """Return one when an unsuccessful episode times out.

    When height limits are set, an unsuccessful timeout is selected if either
    the head or trunk remains at or below its limit. Passing both as ``None``
    treats every timeout without success as failure.
    """
    asset = env.scene[head_cfg.name]
    head_height = asset.data.body_pos_w[:, head_cfg.body_ids, 2].max(dim=1).values
    timed_out = env.termination_manager.get_term(timeout_term_name)
    succeeded = env.termination_manager.get_term(success_term_name)
    failed_timeout = timed_out & ~succeeded
    if max_head_height is not None or max_trunk_height is not None:
        low_pose = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if max_head_height is not None:
            low_pose |= head_height <= max_head_height
        if max_trunk_height is not None:
            low_pose |= asset.data.root_pos_w[:, 2] <= max_trunk_height
        failed_timeout &= low_pose
    return failed_timeout.float()


def supported_head_height_exp(
    env: ManagerBasedRLEnv,
    foot_sensor_names: list[str],
    head_cfg: SceneEntityCfg,
    min_height: float = 0.5,
    target_height: float = 0.9,
    sharpness: float = 4.0,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward standing-height head lift only when both feet carry body weight.

    The normalized exponential is zero below ``min_height`` and one at
    ``target_height``. It is multiplied by the weaker foot's load fraction and
    trunk uprightness, so sitting on the pelvis or raising only the head cannot
    earn the high-value portion of the reward.
    """
    if target_height <= min_height:
        raise ValueError("target_height must be greater than min_height")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive")

    asset = env.scene[head_cfg.name]
    head_height = asset.data.body_pos_w[:, head_cfg.body_ids, 2].max(dim=1).values
    height_progress = torch.clamp(
        (head_height - min_height) / (target_height - min_height),
        0.0,
        1.0,
    )
    height_reward = torch.expm1(sharpness * height_progress) / math.expm1(sharpness)

    body_weight = asset.data.default_mass.to(asset.device).sum(dim=1) * 9.81
    per_foot_load = []
    for sensor_name in foot_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        vertical_force = sensor.data.force_matrix_w[..., 2].clamp(min=0.0).sum(dim=(1, 2))
        is_contacting = vertical_force > contact_threshold
        load_fraction = torch.clamp(2.0 * vertical_force / body_weight.clamp(min=1e-6), 0.0, 1.0)
        per_foot_load.append(load_fraction * is_contacting.float())
    balanced_foot_load = torch.stack(per_foot_load, dim=1).min(dim=1).values
    return height_reward * balanced_foot_load * _upright_factor(asset)


def center_of_mass_height_reward(
    env: ManagerBasedRLEnv,
    min_height: float = 0.12,
    target_height: float = 0.5,
    exponent: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward mass-weighted whole-body CoM height with bounded progress."""
    if target_height <= min_height:
        raise ValueError("target_height must be greater than min_height")
    asset = env.scene[asset_cfg.name]
    masses = asset.data.default_mass.to(asset.device)
    com_z = asset.data.body_com_pos_w[:, :, 2]
    height = (masses * com_z).sum(dim=1) / masses.sum(dim=1).clamp(min=1e-6)
    progress = torch.clamp((height - min_height) / (target_height - min_height), 0.0, 1.0)
    return progress.pow(exponent)


def lowest_hip_height_reward(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    min_height: float = 0.04,
    target_height: float = 0.5,
    exponent: float = 1.0,
) -> torch.Tensor:
    """Reward raising the lowest hip, independent of foot contact state."""
    if target_height <= min_height:
        raise ValueError("target_height must be greater than min_height")
    if exponent <= 0.0:
        raise ValueError("exponent must be positive")
    asset = env.scene[hip_cfg.name]
    lowest_hip = asset.data.body_pos_w[:, hip_cfg.body_ids, 2].min(dim=1).values
    progress = torch.clamp(
        (lowest_hip - min_height) / (target_height - min_height),
        0.0,
        1.0,
    )
    return progress.pow(exponent)


# ---------------------------------------------------------------------------
# 4. 足裏の接地
# ---------------------------------------------------------------------------
def feet_ground_contact(
    env: ManagerBasedRLEnv,
    foot_sensor_names: list[str],
    threshold: float = 1.0,
    require_upright: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """足裏 (foot link) が地面に接触していることへの報酬。

    ``sensor_cfg`` で指定した足リンクのうち、接触力が ``threshold`` [N] を超えているものの
    割合 (0〜1) を返す。両足接地で 1、片足で 0.5、両足浮きで 0。

    Args:
        foot_sensor_names: 左右各足を地面だけにフィルターした ContactSensor 名。
        threshold: 接地とみなす接触力の下限 [N]。
        require_upright: True なら直立度 (_upright_factor) を掛ける。これがないと
                         「寝たまま足裏だけ接地」で満点が取れてしまい、寝姿勢の局所最適に
                         はまる。上体が立っているときの「足で立つ」ことだけを報酬にする。
        asset_cfg: 直立度を測るアセット (require_upright=True のとき使用)。
    """
    contacts = []
    for sensor_name in foot_sensor_names:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
        if contact_sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = contact_sensor.data.force_matrix_w.norm(dim=-1)
        contacts.append((ground_force > threshold).any(dim=(1, 2)))
    reward = torch.stack(contacts, dim=1).float().mean(dim=1)
    if require_upright:
        reward = reward * _upright_factor(env.scene[asset_cfg.name])
    return reward


def one_foot_airborne_above_head_height(
    env: ManagerBasedRLEnv,
    foot_sensor_names: list[str],
    head_cfg: SceneEntityCfg,
    min_head_height: float = 0.35,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize exactly one airborne foot above a configurable head height."""
    asset = env.scene[head_cfg.name]
    head_height = asset.data.body_pos_w[:, head_cfg.body_ids, 2].max(dim=1).values
    contacts = []
    for sensor_name in foot_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = sensor.data.force_matrix_w.norm(dim=-1)
        contacts.append((ground_force > contact_threshold).any(dim=(1, 2)))
    contact_count = torch.stack(contacts, dim=1).sum(dim=1)
    return ((head_height >= min_head_height) & (contact_count == 1)).float()


def airborne_feet_count(
    env: ManagerBasedRLEnv,
    foot_sensor_names: list[str],
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Return the fraction of feet not touching the ground, independent of pose."""
    contacts = []
    for sensor_name in foot_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = sensor.data.force_matrix_w.norm(dim=-1)
        contacts.append((ground_force > contact_threshold).any(dim=(1, 2)))
    contact_fraction = torch.stack(contacts, dim=1).float().mean(dim=1)
    return 1.0 - contact_fraction


def feet_above_head_penalty(
    env: ManagerBasedRLEnv,
    head_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    allowed_margin: float = 0.1,
    scale: float = 0.3,
) -> torch.Tensor:
    """Penalize each foot in proportion to how far it rises above the head."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    asset = env.scene[head_cfg.name]
    head_height = asset.data.body_pos_w[:, head_cfg.body_ids, 2].max(dim=1).values
    foot_heights = asset.data.body_pos_w[:, foot_cfg.body_ids, 2]
    excess = torch.clamp(foot_heights - head_height.unsqueeze(1) - allowed_margin, min=0.0)
    return torch.clamp(excess / scale, max=1.0).mean(dim=1)


def supine_bridge_progress(
    env: ManagerBasedRLEnv,
    foot_sensor_names: list[str],
    hip_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    hip_target_height: float = 0.45,
    foot_height_sigma: float = 0.08,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Guide feet to the floor, then reward lifting both hips into a bridge.

    Foot proximity supplies a dense gradient before contact. Hip lift is only
    rewarded while both feet contact the ground, preventing elbow/hip-supported
    poses with airborne feet from exploiting the bridge term.
    """
    asset = env.scene[hip_cfg.name]
    foot_height = asset.data.body_pos_w[:, foot_cfg.body_ids, 2].clamp(min=0.0)
    foot_proximity = torch.exp(-foot_height / foot_height_sigma).mean(dim=1)

    contacts = []
    for sensor_name in foot_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = sensor.data.force_matrix_w.norm(dim=-1)
        contacts.append((ground_force > contact_threshold).any(dim=(1, 2)))
    both_feet = torch.stack(contacts, dim=1).all(dim=1)

    lowest_hip = asset.data.body_pos_w[:, hip_cfg.body_ids, 2].min(dim=1).values
    hip_progress = torch.clamp(lowest_hip / hip_target_height, 0.0, 1.0)
    return 0.25 * foot_proximity + 0.75 * both_feet.float() * hip_progress


def hip_clearance_with_foot_support_reward(
    env: ManagerBasedRLEnv,
    foot_sensor_names: list[str],
    hip_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    min_clearance: float = 0.02,
    target_clearance: float = 0.35,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward lifting the lowest hip above the highest foot while both feet support."""
    if target_clearance <= min_clearance:
        raise ValueError("target_clearance must be greater than min_clearance")
    asset = env.scene[hip_cfg.name]
    contacts = []
    for sensor_name in foot_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = sensor.data.force_matrix_w.norm(dim=-1)
        contacts.append((ground_force > contact_threshold).any(dim=(1, 2)))
    both_feet = torch.stack(contacts, dim=1).all(dim=1)

    lowest_hip = asset.data.body_pos_w[:, hip_cfg.body_ids, 2].min(dim=1).values
    highest_foot = asset.data.body_pos_w[:, foot_cfg.body_ids, 2].max(dim=1).values
    clearance = lowest_hip - highest_foot
    progress = torch.clamp(
        (clearance - min_clearance) / (target_clearance - min_clearance),
        0.0,
        1.0,
    )
    return progress * both_feet.float()


def hip_below_feet_exponential_penalty(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    scale: float = 0.25,
    sharpness: float = 4.0,
) -> torch.Tensor:
    """Exponentially penalize the lowest hip falling below the highest foot."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive")
    asset = env.scene[hip_cfg.name]
    lowest_hip = asset.data.body_pos_w[:, hip_cfg.body_ids, 2].min(dim=1).values
    highest_foot = asset.data.body_pos_w[:, foot_cfg.body_ids, 2].max(dim=1).values
    normalized_deficit = torch.clamp((highest_foot - lowest_hip) / scale, 0.0, 1.0)
    return torch.expm1(sharpness * normalized_deficit) / math.expm1(sharpness)


def feet_ground_reaction_increase(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """足が地面から受ける「垂直反力」が前ステップより増加した分への報酬。

    摩擦 (水平方向) ではなく法線方向 (world +z, 平地では純粋な垂直反力) の押し込み力の
    増加を促す。脚で地面を鉛直に押して体を持ち上げる、摩擦に依存しない起き上がりを学習
    させるのが狙い。両足の垂直反力合計が前ステップより大きくなった分だけ (増加分のみ)
    報酬を与え、体重 (m·g) で正規化して質量スケールに依存しない無次元量にする。

    Note: ``net_forces_w_history`` は roll 実装により index 0 = 現ステップ, index 1 =
    前ステップ (feet_ground_contact が使う [:, -1] は履歴最古なので、ここでは明示的に
    0/1 を使う)。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 垂直 (+z) 成分。地面反力は上向きなので通常 >= 0。念のため負値は 0 にクランプ。
    fz = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].clamp(min=0.0)  # [N, hist, F]
    cur = fz[:, 0, :].sum(dim=1)   # 現ステップの両足合計垂直反力 [N]
    prev = fz[:, 1, :].sum(dim=1)  # 前ステップ
    asset = env.scene[asset_cfg.name]
    weight = asset.data.default_mass.to(asset.device).sum(dim=1) * 9.81  # 体重 [N]
    return torch.clamp(cur - prev, min=0.0) / weight.clamp(min=1e-6)


def balanced_foot_load_progress(
    env: ManagerBasedRLEnv,
    foot_sensor_names: list[str],
    hip_cfg: SceneEntityCfg,
    min_hip_height: float = 0.3,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward signed progress in the weaker foot's normalized vertical load.

    Each foot reaches load 1 when it carries half the robot weight. Taking the
    minimum requires balanced bilateral support. The signed temporal difference
    makes unload/reload cycles sum to zero instead of yielding repeatable reward.
    """
    asset = env.scene[hip_cfg.name]
    body_weight = asset.data.default_mass.to(asset.device).sum(dim=1) * 9.81
    per_foot_load = []
    for sensor_name in foot_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        vertical_force = sensor.data.force_matrix_w[..., 2].clamp(min=0.0).sum(dim=(1, 2))
        contacting = vertical_force > contact_threshold
        load = torch.clamp(2.0 * vertical_force / body_weight.clamp(min=1e-6), 0.0, 1.0)
        per_foot_load.append(load * contacting.float())
    balanced_load = torch.stack(per_foot_load, dim=1).min(dim=1).values

    lowest_hip = asset.data.body_pos_w[:, hip_cfg.body_ids, 2].min(dim=1).values
    potential = balanced_load * (lowest_hip >= min_hip_height).float()
    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    key = "getup_balanced_foot_load_prev"
    previous = env._custom_buffers.get(key)
    if previous is None or previous.shape != potential.shape:
        previous = potential.clone()

    progress = potential - previous
    progress = torch.where(env.reset_buf > 0, torch.zeros_like(progress), progress)
    env._custom_buffers[key] = potential.clone()
    return progress


def feet_vertical_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
    max_fraction: float = 1.0,
    flatness_sigma: float = 0.25,
    require_upright: bool = False,
    hip_cfg: SceneEntityCfg | None = None,
    min_hip_clearance: float = 0.02,
    target_hip_clearance: float = 0.3,
) -> torch.Tensor:
    """両足が地面を鉛直に押す力 (法線反力) の「絶対値」を体重比 (0〜max_fraction) で報酬。

    ``feet_ground_reaction_increase`` が「増分」を報酬にするのに対し、こちらは現在の垂直反力
    そのものを報酬にする。足で体重を支える状態 (報酬 ~1.0) まで上向きの勾配があるので、摩擦に
    頼らず「足裏で地面を押して立つ」ことを促す。スラムを稼ぐ抜け道を防ぐため ``max_fraction``
    で頭打ちにする。

    Args:
        require_upright: True なら直立度 (_upright_factor) を掛け、「上体が起きた後」だけ
                         足裏押しを要求する。これがないと寝たまま足を押し付けて満点を farm
                         できてしまう (スクラッチで prone 局所最適の原因になる)。上体を起こす
                         段階は妨げず、起きた後に足裏で押して立つ動きを誘導する。

    Note: net_forces_w_history は index 0 = 現ステップ (roll 実装)。world +z が法線方向。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fz = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2].clamp(min=0.0)  # [N, F]
    asset = env.scene[asset_cfg.name]
    foot_quat = asset.data.body_quat_w[:, foot_cfg.body_ids, :]
    num_envs, num_feet, _ = foot_quat.shape
    roll, pitch, _ = euler_xyz_from_quat(foot_quat.reshape(-1, 4))
    foot_tilt_sq = torch.square(wrap_to_pi(roll)) + torch.square(wrap_to_pi(pitch))
    flatness = torch.exp(-foot_tilt_sq.reshape(num_envs, num_feet) / flatness_sigma)
    total = (fz * flatness).sum(dim=1)
    weight = asset.data.default_mass.to(asset.device).sum(dim=1) * 9.81  # 体重 [N]
    reward = (total / weight.clamp(min=1e-6)).clamp(max=max_fraction)
    if hip_cfg is not None:
        if target_hip_clearance <= min_hip_clearance:
            raise ValueError("target_hip_clearance must be greater than min_hip_clearance")
        lowest_hip = asset.data.body_pos_w[:, hip_cfg.body_ids, 2].min(dim=1).values
        highest_foot = asset.data.body_pos_w[:, foot_cfg.body_ids, 2].max(dim=1).values
        hip_clearance = lowest_hip - highest_foot
        support_progress = torch.clamp(
            (hip_clearance - min_hip_clearance)
            / (target_hip_clearance - min_hip_clearance),
            0.0,
            1.0,
        )
        reward = reward * support_progress
    if require_upright:
        reward = reward * _upright_factor(asset)
    return reward


def knee_above_trunk_and_hips_penalty(
    env: ManagerBasedRLEnv,
    foot_sensor_names: list[str],
    knee_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    allowed_margin: float = 0.02,
    scale: float = 0.25,
    allow_below_hip_height: float = 0.3,
    full_above_hip_height: float = 0.45,
) -> torch.Tensor:
    """Penalize high knees only after the low bridge phase is complete."""
    if full_above_hip_height <= allow_below_hip_height:
        raise ValueError("full_above_hip_height must exceed allow_below_hip_height")
    asset = env.scene[knee_cfg.name]
    contacts = []
    for sensor_name in foot_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = sensor.data.force_matrix_w.norm(dim=-1)
        contacts.append((ground_force > contact_threshold).any(dim=(1, 2)))
    both_feet = torch.stack(contacts, dim=1).all(dim=1)

    highest_knee = asset.data.body_pos_w[:, knee_cfg.body_ids, 2].max(dim=1).values
    lowest_hip = asset.data.body_pos_w[:, hip_cfg.body_ids, 2].min(dim=1).values
    trunk_height = asset.data.root_pos_w[:, 2]
    trunk_violation = torch.clamp(highest_knee - trunk_height - allowed_margin, min=0.0)
    hip_violation = torch.clamp(highest_knee - lowest_hip - allowed_margin, min=0.0)
    phase = torch.clamp(
        (lowest_hip - allow_below_hip_height)
        / (full_above_hip_height - allow_below_hip_height),
        0.0,
        1.0,
    )
    return (
        both_feet.float()
        * phase
        * torch.clamp((trunk_violation + hip_violation) / scale, max=2.0)
    )


def trunk_contact_under_foot_load_penalty(
    env: ManagerBasedRLEnv,
    trunk_sensor_names: list[str],
    foot_sensor_names: list[str],
    contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize keeping the trunk grounded while the feet carry body weight."""
    asset = env.scene[asset_cfg.name]
    body_weight = asset.data.default_mass.to(asset.device).sum(dim=1) * 9.81
    trunk_force = torch.zeros(env.num_envs, device=env.device)
    for sensor_name in trunk_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        force = sensor.data.force_matrix_w.norm(dim=-1).sum(dim=(1, 2))
        trunk_force += torch.where(force > contact_threshold, force, torch.zeros_like(force))

    foot_vertical_force = torch.zeros(env.num_envs, device=env.device)
    for sensor_name in foot_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        foot_vertical_force += sensor.data.force_matrix_w[..., 2].clamp(min=0.0).sum(dim=(1, 2))
    trunk_fraction = torch.clamp(trunk_force / body_weight.clamp(min=1e-6), max=2.0)
    foot_load_fraction = torch.clamp(foot_vertical_force / body_weight.clamp(min=1e-6), 0.0, 1.0)
    return trunk_fraction * foot_load_fraction


def non_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=[".*_hand_link", ".*_Shank"]
    ),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """足 (foot_link) 以外の body が地面から受ける接触力の合計 (体重比) へのペナルティ。

    手 (hand_link)・膝 (Shank) で地面を押してレバレッジ起き上がりするのを抑え、「足だけで
    起き上がる」動きを誘導する。sim2real で接触モデル差が出やすい非足部の接地依存を減らすのが
    狙い。体重 (m·g) で正規化した無次元量。負の weight で使う。

    Note: net_forces_w_history は index 0 = 現ステップ (roll 実装)。胴 (Trunk) は寝姿勢で
    不可避に接地するのでここには含めない (手・膝の「押し」だけを罰する)。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :].norm(dim=-1)  # [N, B]
    total = forces.sum(dim=1)  # 非足部の接触力合計 [N]
    asset = env.scene[asset_cfg.name]
    weight = asset.data.default_mass.to(asset.device).sum(dim=1) * 9.81  # 体重 [N]
    return total / weight.clamp(min=1e-6)


def staged_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    allow_below_height: float,
    penalize_above_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Contact-force penalty that is disabled while the trunk is low.

    Hands and shanks may be used during the initial rise. The penalty ramps
    linearly from zero to full strength as the root height moves through the
    configured phase-transition band.
    """
    if penalize_above_height <= allow_below_height:
        raise ValueError("penalize_above_height must be greater than allow_below_height")

    asset = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1).sum(dim=1)
    body_weight = asset.data.default_mass.to(asset.device).sum(dim=1) * 9.81
    phase = torch.clamp(
        (asset.data.root_pos_w[:, 2] - allow_below_height)
        / (penalize_above_height - allow_below_height),
        0.0,
        1.0,
    )
    return phase * forces / body_weight.clamp(min=1e-6)


def staged_ground_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    head_cfg: SceneEntityCfg,
    allow_below_head_height: float = 0.25,
    full_above_head_height: float = 0.5,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize ground-filtered body contact after the head starts rising."""
    if full_above_head_height <= allow_below_head_height:
        raise ValueError("full_above_head_height must be greater than allow_below_head_height")

    asset = env.scene[head_cfg.name]
    head_height = asset.data.body_pos_w[:, head_cfg.body_ids, 2].max(dim=1).values
    phase = torch.clamp(
        (head_height - allow_below_head_height)
        / (full_above_head_height - allow_below_head_height),
        0.0,
        1.0,
    )
    body_weight = asset.data.default_mass.to(asset.device).sum(dim=1) * 9.81
    total_ground_force = torch.zeros(env.num_envs, device=env.device)
    for sensor_name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = sensor.data.force_matrix_w.norm(dim=-1).sum(dim=(1, 2))
        total_ground_force += torch.where(
            ground_force > contact_threshold,
            ground_force,
            torch.zeros_like(ground_force),
        )
    return phase * torch.clamp(total_ground_force / body_weight.clamp(min=1e-6), max=2.0)


def proximal_arm_contact_penalty(
    env: ManagerBasedRLEnv,
    proximal_sensor_names: list[str],
    hand_sensor_names: list[str],
    contact_threshold: float = 1.0,
    contact_presence_scale: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize Arm_3 contact presence and load, especially without hand contact."""
    asset = env.scene[asset_cfg.name]
    body_weight = asset.data.default_mass.to(asset.device).sum(dim=1) * 9.81
    proximal_force = torch.zeros(env.num_envs, device=env.device)
    proximal_contacts = []
    for sensor_name in proximal_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        force_norm = sensor.data.force_matrix_w.norm(dim=-1)
        force = force_norm.sum(dim=(1, 2))
        proximal_force += torch.where(force > contact_threshold, force, torch.zeros_like(force))
        proximal_contacts.append((force_norm > contact_threshold).any(dim=(1, 2)))

    hand_contacts = []
    for sensor_name in hand_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        force = sensor.data.force_matrix_w.norm(dim=-1)
        hand_contacts.append((force > contact_threshold).any(dim=(1, 2)))
    hand_contact_fraction = torch.stack(hand_contacts, dim=1).float().mean(dim=1)
    proximal_contact_fraction = torch.stack(proximal_contacts, dim=1).float().mean(dim=1)
    missing_hand_scale = 2.0 - hand_contact_fraction
    force_penalty = torch.clamp(proximal_force / body_weight.clamp(min=1e-6), max=2.0)
    return missing_hand_scale * (
        force_penalty + contact_presence_scale * proximal_contact_fraction
    )


def hand_support_by_head_and_hip_height(
    env: ManagerBasedRLEnv,
    hand_sensor_names: list[str],
    foot_sensor_names: list[str],
    hand_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    head_height_threshold: float = 0.35,
    hip_height_threshold: float = 0.3,
    target_head_height: float = 0.9,
    target_hip_height: float = 0.5,
    distal_tip_offset: float = 0.17,
    distal_height_sigma: float = 0.04,
    contact_threshold: float = 1.0,
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Head.*"),
) -> torch.Tensor:
    """Shape hand use and body lift while requiring support from the feet.

    Before release, hand contact becomes more valuable as the head and lowest hip
    approach their thresholds. After release, free hands become more valuable as
    both heights continue toward standing targets.
    """
    if len(hand_sensor_names) != 2 or len(hand_cfg.body_ids) != 2:
        raise ValueError("hand support reward requires ordered left and right hand links")
    if distal_height_sigma <= 0.0:
        raise ValueError("distal_height_sigma must be positive")
    if target_head_height <= head_height_threshold:
        raise ValueError("target_head_height must exceed head_height_threshold")
    if target_hip_height <= hip_height_threshold:
        raise ValueError("target_hip_height must exceed hip_height_threshold")
    asset = env.scene[head_cfg.name]
    hand_contacts = []
    for sensor_name in hand_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = sensor.data.force_matrix_w.norm(dim=-1)
        hand_contacts.append((ground_force > contact_threshold).any(dim=(1, 2)))
    hand_contacts = torch.stack(hand_contacts, dim=1)
    contact_count = hand_contacts.sum(dim=1)
    contact_fraction = hand_contacts.float().mean(dim=1)
    hand_pos = asset.data.body_pos_w[:, hand_cfg.body_ids, :]
    hand_quat = asset.data.body_quat_w[:, hand_cfg.body_ids, :]
    local_tip_offsets = torch.zeros_like(hand_pos)
    local_tip_offsets[:, 0, 1] = distal_tip_offset
    local_tip_offsets[:, 1, 1] = -distal_tip_offset
    distal_tip_pos = hand_pos + quat_apply(
        hand_quat.reshape(-1, 4), local_tip_offsets.reshape(-1, 3)
    ).reshape_as(hand_pos)
    distal_proximity = torch.exp(-torch.clamp(distal_tip_pos[:, :, 2], min=0.0) / distal_height_sigma)
    distal_contact_fraction = (hand_contacts.float() * distal_proximity).mean(dim=1)
    head_height = asset.data.body_pos_w[:, head_cfg.body_ids, 2].max(dim=1).values

    foot_contacts = []
    for sensor_name in foot_sensor_names:
        foot_sensor: ContactSensor = env.scene.sensors[sensor_name]
        if foot_sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = foot_sensor.data.force_matrix_w.norm(dim=-1)
        foot_contacts.append((ground_force > contact_threshold).any(dim=(1, 2)))
    foot_contact_count = torch.stack(foot_contacts, dim=1).sum(dim=1)
    both_feet = foot_contact_count == len(foot_sensor_names)

    lowest_hip = asset.data.body_pos_w[:, hip_cfg.body_ids, 2].min(dim=1).values
    support_height_progress = torch.minimum(
        torch.clamp(head_height / head_height_threshold, 0.0, 1.0),
        torch.clamp(lowest_hip / hip_height_threshold, 0.0, 1.0),
    )
    release_height_progress = torch.minimum(
        torch.clamp(
            (head_height - head_height_threshold)
            / (target_head_height - head_height_threshold),
            0.0,
            1.0,
        ),
        torch.clamp(
            (lowest_hip - hip_height_threshold)
            / (target_hip_height - hip_height_threshold),
            0.0,
            1.0,
        ),
    )

    hands_free = contact_count == 0
    foot_support_fraction = foot_contact_count.float() / float(len(foot_sensor_names))
    low_phase_reward = (
        foot_support_fraction
        * distal_contact_fraction
        * (0.5 + 0.5 * support_height_progress)
    )
    high_phase_reward = (
        hands_free.float()
        * both_feet.float()
        * (1.0 + release_height_progress)
        - contact_fraction
        - (2 - foot_contact_count).float() * 0.5
    )
    release_phase = (head_height >= head_height_threshold) & (
        lowest_hip >= hip_height_threshold
    )
    reward = torch.where(release_phase, high_phase_reward, low_phase_reward)
    return torch.clamp(reward, min=-1.0, max=2.0)


def staged_knee_flexion_extension_reward(
    env: ManagerBasedRLEnv,
    knee_cfg: SceneEntityCfg,
    head_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    flexion_target: float = math.pi / 2.0,
    extension_target: float = 0.52,
    flexion_tolerance: float = 1.0,
    extension_tolerance: float = 0.7,
    head_release_height: float = 0.35,
    hip_release_height: float = 0.3,
) -> torch.Tensor:
    """Reward 90-degree knee flexion before bridge release and extension afterwards."""
    if flexion_tolerance <= 0.0 or extension_tolerance <= 0.0:
        raise ValueError("knee target tolerances must be positive")
    asset = env.scene[knee_cfg.name]
    knee_angles = asset.data.joint_pos[:, knee_cfg.joint_ids]
    if knee_angles.shape[1] != 2:
        raise ValueError("staged knee reward requires exactly two knee joints")

    head_height = asset.data.body_pos_w[:, head_cfg.body_ids, 2].max(dim=1).values
    lowest_hip = asset.data.body_pos_w[:, hip_cfg.body_ids, 2].min(dim=1).values
    release_phase = (head_height >= head_release_height) & (
        lowest_hip >= hip_release_height
    )

    target = torch.where(
        release_phase.unsqueeze(1),
        torch.full_like(knee_angles, extension_target),
        torch.full_like(knee_angles, flexion_target),
    )
    tolerance = torch.where(
        release_phase.unsqueeze(1),
        torch.full_like(knee_angles, extension_tolerance),
        torch.full_like(knee_angles, flexion_tolerance),
    )
    per_knee_reward = torch.clamp(1.0 - torch.abs(knee_angles - target) / tolerance, 0.0, 1.0)
    return per_knee_reward.min(dim=1).values


def staged_hip_pitch_squat_to_stand_reward(
    env: ManagerBasedRLEnv,
    hip_pitch_cfg: SceneEntityCfg,
    head_cfg: SceneEntityCfg,
    hip_body_cfg: SceneEntityCfg,
    minimum_hip_height: float = 0.3,
    transition_head_height: float = 0.35,
    standing_head_height: float = 0.7,
    squat_target: float = -1.0,
    standing_target: float = -0.26,
    tolerance: float = 0.8,
) -> torch.Tensor:
    """Move hip pitch from a forward squat target toward standing as the head rises."""
    if standing_head_height <= transition_head_height:
        raise ValueError("standing_head_height must exceed transition_head_height")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    asset = env.scene[hip_pitch_cfg.name]
    hip_pitch = asset.data.joint_pos[:, hip_pitch_cfg.joint_ids]
    if hip_pitch.shape[1] != 2:
        raise ValueError("staged hip-pitch reward requires exactly two joints")

    head_height = asset.data.body_pos_w[:, head_cfg.body_ids, 2].max(dim=1).values
    lowest_hip = asset.data.body_pos_w[:, hip_body_cfg.body_ids, 2].min(dim=1).values
    stand_progress = torch.clamp(
        (head_height - transition_head_height)
        / (standing_head_height - transition_head_height),
        0.0,
        1.0,
    )
    target = squat_target + stand_progress.unsqueeze(1) * (standing_target - squat_target)
    normalized_error = (hip_pitch - target) / tolerance
    per_hip_reward = torch.exp(-torch.square(normalized_error))
    active = lowest_hip >= minimum_hip_height
    return active.float() * per_hip_reward.min(dim=1).values


def backward_hip_pitch_fold_penalty(
    env: ManagerBasedRLEnv,
    hip_pitch_cfg: SceneEntityCfg,
    head_cfg: SceneEntityCfg,
    hip_body_cfg: SceneEntityCfg,
    minimum_head_height: float = 0.35,
    minimum_hip_height: float = 0.3,
    allowed_positive_pitch: float = 0.0,
    scale: float = 1.2,
) -> torch.Tensor:
    """Penalize positive hip pitch that folds the trunk backward over upright legs."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    asset = env.scene[hip_pitch_cfg.name]
    hip_pitch = asset.data.joint_pos[:, hip_pitch_cfg.joint_ids]
    if hip_pitch.shape[1] != 2:
        raise ValueError("backward hip-pitch penalty requires exactly two joints")

    head_height = asset.data.body_pos_w[:, head_cfg.body_ids, 2].max(dim=1).values
    lowest_hip = asset.data.body_pos_w[:, hip_body_cfg.body_ids, 2].min(dim=1).values
    active = (head_height >= minimum_head_height) & (lowest_hip >= minimum_hip_height)
    per_hip_penalty = torch.clamp(
        (hip_pitch - allowed_positive_pitch) / scale,
        min=0.0,
        max=1.0,
    )
    return active.float() * per_hip_penalty.max(dim=1).values


def body_height_order(
    env: ManagerBasedRLEnv,
    head_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    foot_sensor_names: list[str] | None = None,
    contact_threshold: float = 1.0,
    hip_foot_target: float = 0.45,
    double_scale_head_hip_gap: float = 0.15,
    triple_scale_head_hip_gap: float = 0.3,
) -> torch.Tensor:
    """Reward the continuous height order ``head > both hips > both feet``.

    Hip-to-foot progress is always the base reward, preserving exploration that
    raises the pelvis above both feet. The reward scale doubles and triples as
    the head rises above the highest hip, favoring upper-body rise after bridge.
    """
    if hip_foot_target <= 0.0:
        raise ValueError("hip_foot_target must be positive")
    if not 0.0 <= double_scale_head_hip_gap < triple_scale_head_hip_gap:
        raise ValueError("head-hip scale gaps must satisfy 0 <= double < triple")

    asset = env.scene[head_cfg.name]
    head_height = asset.data.body_pos_w[:, head_cfg.body_ids, 2].max(dim=1).values
    hip_height = asset.data.body_pos_w[:, hip_cfg.body_ids, 2]
    highest_hip = hip_height.max(dim=1).values
    lowest_hip = hip_height.min(dim=1).values
    highest_foot = asset.data.body_pos_w[:, foot_cfg.body_ids, 2].max(dim=1).values

    hips_above_feet = torch.clamp((lowest_hip - highest_foot) / hip_foot_target, 0.0, 1.0)
    head_hip_gap = head_height - highest_hip
    scale = torch.where(
        head_hip_gap >= triple_scale_head_hip_gap,
        3.0,
        torch.where(head_hip_gap >= double_scale_head_hip_gap, 2.0, 1.0),
    )
    reward = scale * hips_above_feet
    if foot_sensor_names is not None:
        contacts = []
        for sensor_name in foot_sensor_names:
            sensor: ContactSensor = env.scene.sensors[sensor_name]
            if sensor.data.force_matrix_w is None:
                raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
            ground_force = sensor.data.force_matrix_w.norm(dim=-1)
            contacts.append((ground_force > contact_threshold).any(dim=(1, 2)))
        reward *= torch.stack(contacts, dim=1).all(dim=1).float()
    return reward


def inverted_posture_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the trunk turning upside down without constraining side-lying motion."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.projected_gravity_b[:, 2], min=0.0, max=1.0)


def lateral_roll_penalty(
    env: ManagerBasedRLEnv,
    tolerance: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize side-lying roll while allowing the initial supine pitch motion."""
    asset = env.scene[asset_cfg.name]
    lateral_gravity = torch.abs(asset.data.projected_gravity_b[:, 1])
    return torch.clamp((lateral_gravity - tolerance) / (1.0 - tolerance), 0.0, 1.0)


def bilateral_limb_height_asymmetry_penalty(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    tolerance: float = 0.05,
    scale: float = 0.35,
) -> torch.Tensor:
    """Penalize raising only one hand or one foot above its opposite-side limb."""
    if scale <= tolerance:
        raise ValueError("scale must be greater than tolerance")
    asset = env.scene[hand_cfg.name]
    if len(hand_cfg.body_ids) != 2 or len(foot_cfg.body_ids) != 2:
        raise ValueError("limb height asymmetry requires ordered left and right hand/foot links")
    hand_height = asset.data.body_pos_w[:, hand_cfg.body_ids, 2]
    foot_height = asset.data.body_pos_w[:, foot_cfg.body_ids, 2]
    hand_difference = torch.abs(hand_height[:, 0] - hand_height[:, 1])
    foot_difference = torch.abs(foot_height[:, 0] - foot_height[:, 1])
    total_difference = hand_difference + foot_difference
    return torch.clamp((total_difference - tolerance) / (scale - tolerance), 0.0, 2.0)


def hip_roll_excess_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_abs_roll: float = 0.5,
    symmetry_tolerance: float = 0.15,
) -> torch.Tensor:
    """Penalize excessive hip abduction and left-right mirror mismatch.

    K1's outward hip-roll directions have opposite signs, so mirror symmetry is
    measured by ``abs(left + right)`` rather than ``abs(left - right)``.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if joint_pos.shape[1] != 2:
        raise ValueError("hip_roll_excess_penalty requires exactly two hip-roll joints")

    left_roll = joint_pos[:, 0]
    right_roll = joint_pos[:, 1]
    excessive_opening = (
        torch.clamp(torch.abs(left_roll) - max_abs_roll, min=0.0)
        + torch.clamp(torch.abs(right_roll) - max_abs_roll, min=0.0)
    )
    mirror_mismatch = torch.clamp(
        torch.abs(left_roll + right_roll) - symmetry_tolerance,
        min=0.0,
    )
    return excessive_opening + mirror_mismatch


def vertical_alignment_to_support_reward(
    env: ManagerBasedRLEnv,
    foot_sensor_names: list[str],
    head_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    min_head_height: float = 0.35,
    max_radius: float = 0.25,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Align the head, trunk, and hip center vertically above the feet midpoint."""
    if max_radius <= 0.0:
        raise ValueError("max_radius must be positive")

    asset = env.scene[head_cfg.name]
    contacts = []
    for sensor_name in foot_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = sensor.data.force_matrix_w.norm(dim=-1)
        contacts.append((ground_force > contact_threshold).any(dim=(1, 2)))
    both_feet = torch.stack(contacts, dim=1).all(dim=1)

    body_pos = asset.data.body_pos_w
    feet_mid_xy = body_pos[:, foot_cfg.body_ids, :2].mean(dim=1)
    hip_mid_xy = body_pos[:, hip_cfg.body_ids, :2].mean(dim=1)
    head_pos = body_pos[:, head_cfg.body_ids, :]
    highest_head_idx = head_pos[:, :, 2].argmax(dim=1)
    env_indices = torch.arange(env.num_envs, device=env.device)
    head_xy = head_pos[env_indices, highest_head_idx, :2]
    trunk_xy = asset.data.root_pos_w[:, :2]

    alignment_error = torch.stack(
        (
            torch.linalg.vector_norm(head_xy - feet_mid_xy, dim=1),
            torch.linalg.vector_norm(trunk_xy - feet_mid_xy, dim=1),
            torch.linalg.vector_norm(hip_mid_xy - feet_mid_xy, dim=1),
        ),
        dim=1,
    ).mean(dim=1)
    alignment_reward = torch.clamp(1.0 - alignment_error / max_radius, min=-1.0, max=1.0)
    head_height = head_pos[:, :, 2].max(dim=1).values
    active = (head_height >= min_head_height) & both_feet
    return alignment_reward * active.float()


def standing_support_geometry(
    env: ManagerBasedRLEnv,
    foot_sensor_names: list[str],
    head_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    min_head_height: float = 0.35,
    contact_threshold: float = 1.0,
    target_foot_spacing: float = 0.2,
    alignment_sigma: float = 0.15,
    spacing_sigma: float = 0.08,
) -> torch.Tensor:
    """Reward a vertically stacked body and stable foot spacing after both feet land."""
    asset = env.scene[head_cfg.name]
    contacts = []
    for sensor_name in foot_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = sensor.data.force_matrix_w.norm(dim=-1)
        contacts.append((ground_force > contact_threshold).any(dim=(1, 2)))
    both_feet = torch.stack(contacts, dim=1).all(dim=1)

    foot_pos = asset.data.body_pos_w[:, foot_cfg.body_ids, :]
    feet_mid_xy = foot_pos[:, :, :2].mean(dim=1)
    foot_spacing = torch.linalg.vector_norm(foot_pos[:, 0, :2] - foot_pos[:, 1, :2], dim=1)
    head_pos = asset.data.body_pos_w[:, head_cfg.body_ids, :]
    highest_head_idx = head_pos[:, :, 2].argmax(dim=1)
    env_indices = torch.arange(env.num_envs, device=env.device)
    head_xy = head_pos[env_indices, highest_head_idx, :2]
    trunk_xy = asset.data.root_pos_w[:, :2]

    alignment_error = 0.5 * (
        torch.linalg.vector_norm(head_xy - feet_mid_xy, dim=1)
        + torch.linalg.vector_norm(trunk_xy - feet_mid_xy, dim=1)
    )
    alignment_reward = torch.exp(-torch.square(alignment_error / alignment_sigma))
    spacing_reward = torch.exp(-torch.square((foot_spacing - target_foot_spacing) / spacing_sigma))
    head_height = head_pos[:, :, 2].max(dim=1).values
    active = (head_height >= min_head_height) & both_feet
    return 0.5 * (alignment_reward + spacing_reward) * active.float()


def hands_free_when_upright(
    env: ManagerBasedRLEnv,
    hand_sensor_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    min_height: float = 0.55,
    max_tilt_deg: float = 25.0,
    contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward both hands being off the ground once a supported upright pose is reached."""
    asset = env.scene[asset_cfg.name]
    hand_sensor: ContactSensor = env.scene.sensors[hand_sensor_cfg.name]
    foot_sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]

    hand_force = hand_sensor.data.net_forces_w[:, hand_sensor_cfg.body_ids, :].norm(dim=-1)
    foot_force = foot_sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :].norm(dim=-1)
    hands_free = ~(hand_force > contact_threshold).any(dim=1)
    both_feet = (foot_force > contact_threshold).all(dim=1)

    max_tilt = math.radians(max_tilt_deg)
    gravity_xy = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1)
    upright = gravity_xy < math.sin(max_tilt)
    high_enough = asset.data.root_pos_w[:, 2] >= min_height
    return (hands_free & both_feet & upright & high_enough).float()


def standing_success_reward(
    env: ManagerBasedRLEnv,
    foot_sensor_cfg: SceneEntityCfg,
    hand_sensor_cfg: SceneEntityCfg,
    min_height: float = 0.6,
    max_tilt_deg: float = 20.0,
    max_lin_speed: float = 0.2,
    max_ang_speed: float = 0.4,
    contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the complete standing-success condition on every maintained step."""
    from .getup_terminations import standing_success_mask

    return standing_success_mask(
        env,
        foot_sensor_cfg=foot_sensor_cfg,
        hand_sensor_cfg=hand_sensor_cfg,
        min_height=min_height,
        max_tilt_deg=max_tilt_deg,
        max_lin_speed=max_lin_speed,
        max_ang_speed=max_ang_speed,
        contact_threshold=contact_threshold,
        asset_cfg=asset_cfg,
    ).float()


def feet_height_low(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
    sensor_cfg: SceneEntityCfg | None = None,
    scale: float = 10.0,
) -> torch.Tensor:
    """足が低い (地面に近い) ほど大きい報酬 ``Σ exp(-scale · h_foot)``。

    ``h_foot`` = 足リンク z 高さ − 地面高さ (flat では地面 0)。足が接地 (h≈0) で最大 (≈1)、
    高く上げるほど 0 に近づく。足を地面近くに保ち、無駄に高く上げない planted な動きを促す。
    両足で合計するので値域は概ね [0, 2]。小さめの weight で使う。
    """
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [N, F]
    ground = _ground_height(env, sensor_cfg)
    if isinstance(ground, torch.Tensor):
        ground = ground.unsqueeze(-1)  # [N, 1] へブロードキャスト
    h = (foot_z - ground).clamp(min=0.0)  # [N, F]
    return torch.sum(torch.exp(-scale * h), dim=1)


# ---------------------------------------------------------------------------
# 4b. 足裏が地面と平行 (水平) であることへのペナルティ
# ---------------------------------------------------------------------------
def feet_flat_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
    sensor_cfg: SceneEntityCfg | None = None,
    require_upright: bool = False,
    require_contact: bool = False,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """足リンクが地面と平行 (水平) からずれている度合いをペナルティとして返す。

    各足の world 座標での roll・pitch (足裏が水平なら 0) の二乗和を足ごとに計算する。
    足裏が地面と平行なほど 0 に近づく。負の weight で使う。

    Args:
        asset_cfg: 足リンクを指す body_names を持つアセット設定。
        sensor_cfg: 足リンクを指す ContactSensor 設定 (require_contact=True のとき必須)。
                    ``asset_cfg`` と同じ足リンク集合・同じ順序を指すこと。
        require_upright: True なら直立度 (_upright_factor) を全体に掛ける。
        require_contact: True なら「接地している足だけ」その水平を要求する (per-foot ゲート)。
                         接地中の足を確実に平らに踏ませたいときに使う。空中の足の向きは罰さない。
        contact_threshold: 接地とみなす接触力 [N] の下限。
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # [N, F, 4]
    n_env, n_feet, _ = quat.shape
    roll, pitch, _ = euler_xyz_from_quat(quat.reshape(-1, 4))
    roll = wrap_to_pi(roll).reshape(n_env, n_feet)
    pitch = wrap_to_pi(pitch).reshape(n_env, n_feet)
    per_foot = torch.square(roll) + torch.square(pitch)  # [N, F]
    if require_contact:
        if sensor_cfg is None:
            raise ValueError("feet_flat_penalty: require_contact=True には sensor_cfg が必要です。")
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        # 履歴内の最大接触力で接地判定 (単フレームのちらつきに頑健)。
        forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)  # [N,hist,F]
        in_contact = (forces.amax(dim=1) > contact_threshold).float()  # [N, F]
        per_foot = per_foot * in_contact
    penalty = torch.sum(per_foot, dim=1)
    if require_upright:
        penalty = penalty * _upright_factor(asset)
    return penalty


# ---------------------------------------------------------------------------
# 5. 上体がまっすぐ (鉛直)
# ---------------------------------------------------------------------------
def upright_posture(
    env: ManagerBasedRLEnv,
    sigma: float = 0.25,
    min_trunk_height: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Trunk が所定の高さまで上がった後、鉛直に近いほど高い報酬。

    body 座標の重力 z 成分 ``g_z = projected_gravity_b[:,2]`` は直立で -1、真横で 0、
    上下反転で +1。直立度を ``(1 - g_z)/2`` で [0,1] に写す:
        直立 (g_z=-1) → 1.0、 真横/寝 (g_z≈0) → 0.5、 反転 (g_z=+1) → 0.0。
    ``min_trunk_height`` 未満では 0 とし、ブリッジ形成中の股関節運動を妨げない。
    閾値を超えた後は直立方向に勾配を与え、反転 (handstand) exploit も防ぐ。

    NOTE: 旧版は水平成分の指数カーネル exp(-(gx²+gy²)/sigma) だったが、(1) 反転でも満点
    (逆立ち exploit)、(2) それを潰すと tilt≥90° が一律 0 になり寝姿勢から勾配が消える、
    という二つの問題があったため単調写像に変更した。``sigma`` は後方互換のため残すが未使用。
    """
    asset = env.scene[asset_cfg.name]
    g_z = asset.data.projected_gravity_b[:, 2]  # -1 直立, 0 真横, +1 反転
    posture = (1.0 - g_z) * 0.5
    above_bridge = asset.data.root_pos_w[:, 2] >= min_trunk_height
    return posture * above_bridge.float()


# ---------------------------------------------------------------------------
# 4a2. 行動の滑らかさ (二階差分 = ジャーク) ペナルティ
# ---------------------------------------------------------------------------
def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """行動の二階差分 (a_t - 2 a_{t-1} + a_{t-2}) の二乗和ペナルティ (ジャーク)。

    action_rate (一階差分) とは独立に「行動の加速度」を罰し、急な指令変化を抑えて
    滑らかな動作にする (sim2real 向け)。負の weight で使う。
    """
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    if not hasattr(env, "_getup_prev_prev_action") or env._getup_prev_prev_action.shape != a.shape:
        env._getup_prev_prev_action = torch.zeros_like(a)
    diff2 = torch.sum(torch.square(a - 2.0 * a_prev + env._getup_prev_prev_action), dim=1)
    env._getup_prev_prev_action = a_prev.clone()
    return diff2


# ---------------------------------------------------------------------------
# 4c. ジャンプ (両足が地面から離れる) ペナルティ
# ---------------------------------------------------------------------------
def jump_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    threshold: float = 1.0,
    com_height_threshold: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """立ち上がった後に両足が同時に地面から離れる (=ジャンプ/ホップ) ことへのペナルティ。

    両足の接触力がともに ``threshold`` [N] 未満 (=両足浮き)、**かつ** すでに起き上がっている
    (CoM > ``com_height_threshold``) ときだけ 1 を返す (負の weight で強く罰する)。

    重要: 起き上がり途中は CoM が低いので発火しない。以前は直立度ゲートだったが、起き上がり中は
    「胴体が立ってきても CoM はまだ低い」段階で両足が浮くことがあり、その rise を -10 で潰して
    しまい起き上がれなくなっていた。CoM ゲートにすることで「立った後のジャンプ」だけを罰する。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)  # [N, hist, F]
    in_contact = forces[:, -1, :] > threshold  # [N, F]
    both_off = ~in_contact.any(dim=1)  # [N] 両足とも浮いている
    asset = env.scene[asset_cfg.name]
    masses = asset.data.default_mass.to(asset.device)
    com_z = (masses * asset.data.body_com_pos_w[:, :, 2]).sum(dim=1) / masses.sum(dim=1)
    stood_up = com_z > com_height_threshold
    return (both_off & stood_up).float()


# ---------------------------------------------------------------------------
# 4d. トルクが「設定最大トルクの一定割合」を超えた分のペナルティ (下半身用)
# ---------------------------------------------------------------------------
def joint_torque_over_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit_ratio: float = 0.7,
) -> torch.Tensor:
    """指定関節の applied torque が「設定 effort_limit × limit_ratio」を超えた分を合計してペナルティ。

    ``limit_ratio=0.7`` なら各関節の最大トルクの 7 割を超えたトルクだけを L1 で罰する。
    effort_limit は explicit アクチュエータ (DelayedPD) の設定値を actuator から直接読む
    (joint_effort_limits は explicit だと sim 用の大きな既定値なので使わない)。初回に per-joint の
    effort_limit ベクトルを構築してキャッシュする。

    Args:
        asset_cfg: 対象アセット。``joint_names`` で下半身関節を指定する。
        limit_ratio: 最大トルクに対する閾値の割合 (0.7 = 7割)。
    """
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    key = "getup_effort_limits"
    if key not in env._custom_buffers:
        n_env, n_joints = asset.data.applied_torque.shape
        limits = torch.zeros((n_env, n_joints), device=asset.device)
        for act in asset.actuators.values():
            limits[:, act.joint_indices] = act.effort_limit.to(asset.device)
        env._custom_buffers[key] = limits
    limits = env._custom_buffers[key][:, asset_cfg.joint_ids]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    over = torch.clamp(torch.abs(torque) - limit_ratio * limits, min=0.0)
    return torch.sum(over, dim=1)


def joint_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """各関節の機械的パワー |applied_torque × joint_vel| の総和へのペナルティ。

    トルクと関節速度の積 (= 瞬時パワー [W]) の絶対値を全関節で合計する。トルクだけ・
    速度だけの penalty と違い「大トルクを高速で出す」勢いの良い動きをまとめて罰するので、
    エネルギー消費が小さく実機に優しい (sim2real 向け) 動きを促す。値のスケールが大きい
    ので重みは小さく (-1e-4 程度) 使う。
    """
    asset = env.scene[asset_cfg.name]
    power = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids]
        * asset.data.joint_vel[:, asset_cfg.joint_ids]
    )
    return torch.sum(power, dim=1)


# ---------------------------------------------------------------------------
# 5b. 起き上がった後、震えずに静止する
# ---------------------------------------------------------------------------
def stand_still_when_up(
    env: ManagerBasedRLEnv,
    com_height_threshold: float = 0.4,
    min_trunk_height: float = 0.0,
    std: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """起き上がり判定 (全身 CoM 高さ > 閾値 かつ 直立) が成立したときだけ、関節速度が
    小さい (=震えず静止している) ほど高い報酬を返す。立ち上がった後に「ピタッと止まる」
    (高周波の震え・小刻みな動きをしない) ことを促す。

    起き上がり途中 (CoM が低い/直立していない) では 0 なので、起き上がりの動作自体は妨げない。

    Args:
        com_height_threshold: 「起き上がった」とみなす CoM 高さ [m]。
        std: 関節速度二乗和に対する指数カーネル幅。小さいほど厳しく静止を要求する。
        asset_cfg: 対象アセット。
    """
    asset = env.scene[asset_cfg.name]
    masses = asset.data.default_mass.to(asset.device)
    com_z = (masses * asset.data.body_com_pos_w[:, :, 2]).sum(dim=1) / masses.sum(dim=1)
    upright = asset.data.projected_gravity_b[:, 2] < 0.0
    trunk_high_enough = asset.data.root_pos_w[:, 2] >= min_trunk_height
    stood_up = (com_z > com_height_threshold) & trunk_high_enough & upright
    motion = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    return torch.exp(-motion / std) * stood_up.float()


# ---------------------------------------------------------------------------
# 6. 全身の左右対称性
# ---------------------------------------------------------------------------
# 左右の対称関節ペアと、右関節に掛ける符号。
#   sign = +1 : 矢状面内で動く関節 (pitch / knee)。左右で同符号なら対称 → 差を罰する。
#   sign = -1 : 面外成分を持つ関節 (roll / yaw)。左右で逆符号なら対称 → 和を罰する。
# (rewards.joint_mirror_symmetry と同じ規約を全身 (腕含む) に拡張したもの)
_SYMMETRY_PAIRS: list[tuple[str, str, float]] = [
    ("Left_Hip_Pitch",        "Right_Hip_Pitch",        1.0),
    ("Left_Hip_Roll",         "Right_Hip_Roll",        -1.0),
    ("Left_Hip_Yaw",          "Right_Hip_Yaw",         -1.0),
    ("Left_Knee_Pitch",       "Right_Knee_Pitch",       1.0),
    ("Left_Ankle_Pitch",      "Right_Ankle_Pitch",      1.0),
    ("Left_Ankle_Roll",       "Right_Ankle_Roll",      -1.0),
    ("ALeft_Shoulder_Pitch",  "ARight_Shoulder_Pitch",  1.0),
    ("Left_Shoulder_Roll",    "Right_Shoulder_Roll",   -1.0),
    ("Left_Elbow_Pitch",      "Right_Elbow_Pitch",      1.0),
    ("Left_Elbow_Yaw",        "Right_Elbow_Yaw",       -1.0),
]


def _symmetry_indices(env: ManagerBasedRLEnv, asset) -> dict:
    """対称ペアの関節インデックス・符号テンソルを一度だけ構築してキャッシュする。"""
    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    key = "getup_symmetry_idx"
    cache = env._custom_buffers.get(key)
    if cache is None:
        left_ids, right_ids, signs = [], [], []
        for l_name, r_name, sign in _SYMMETRY_PAIRS:
            left_ids.append(asset.find_joints(l_name)[0][0])
            right_ids.append(asset.find_joints(r_name)[0][0])
            signs.append(sign)
        device = asset.data.joint_pos.device
        cache = {
            "left": torch.tensor(left_ids, device=device, dtype=torch.long),
            "right": torch.tensor(right_ids, device=device, dtype=torch.long),
            "sign": torch.tensor(signs, device=device, dtype=asset.data.joint_pos.dtype),
        }
        env._custom_buffers[key] = cache
    return cache


def body_symmetry(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """全身 (脚・腕) の関節角が左右対称であるほど高い報酬。

    起き上がり時に左右バランスの取れた動きを促す。mirror loss を使わない代わりに、この報酬で
    対称性を担保する。各対称ペアで pitch/knee は差を、roll/yaw は和を誤差とし、その総和を
    指数カーネルで [0,1] 報酬にする。頭関節 (中央) は左右ペアが無いので対象外。

    Args:
        std: 指数カーネル幅。小さいほど厳しく対称性を要求する。
    """
    asset = env.scene[asset_cfg.name]
    idx = _symmetry_indices(env, asset)
    joint_pos = asset.data.joint_pos

    left_pos = joint_pos[:, idx["left"]]              # [N, num_pairs]
    right_pos = joint_pos[:, idx["right"]] * idx["sign"]
    error = torch.sum(torch.square(left_pos - right_pos), dim=1)
    return torch.exp(-error / std)


def body_symmetry_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """左右対称関節ペアの角度差 ``|q_left - sign·q_right|`` の総和 (L1) ペナルティ。

    左右の動作が同一でない (非対称な) ほど大きくなる。負の weight で使う。body_symmetry
    (exp カーネル報酬) と同じ左右ペア・ミラー符号 (_SYMMETRY_PAIRS) を再利用する:
      - pitch / knee (sign=+1): 対称なら q_left = q_right → |q_left - q_right| を罰する。
      - roll / yaw (sign=-1):  対称なら q_left = -q_right → |q_left + q_right| を罰する。
    (符号を掛けるので、真に左右対称な姿勢では 0 になる。素朴な |q_left - q_right| だと
     roll/yaw を誤って罰してしまうためミラー符号で補正している。) 頭関節は中央で対象外。
    """
    asset = env.scene[asset_cfg.name]
    idx = _symmetry_indices(env, asset)
    joint_pos = asset.data.joint_pos
    left_pos = joint_pos[:, idx["left"]]                    # [N, num_pairs]
    right_pos = joint_pos[:, idx["right"]] * idx["sign"]
    return torch.sum(torch.abs(left_pos - right_pos), dim=1)


def elbow_deviation_exponential_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    tolerance: float = 0.35,
    scale: float = 1.5,
    sharpness: float = 3.0,
) -> torch.Tensor:
    """Exponentially penalize elbow angles outside a band around their default pose."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive")
    asset = env.scene[asset_cfg.name]
    deviation = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    normalized_excess = torch.clamp((deviation - tolerance) / scale, 0.0, 1.0)
    per_joint_penalty = torch.expm1(sharpness * normalized_excess) / math.expm1(sharpness)
    return per_joint_penalty.mean(dim=1)


def hip_yaw_deviation_exponential_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    tolerance: float = 0.15,
    scale: float = 0.85,
    sharpness: float = 3.0,
) -> torch.Tensor:
    """Exponentially penalize each hip-yaw joint moving away from its default angle."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive")
    asset = env.scene[asset_cfg.name]
    deviation = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    normalized_excess = torch.clamp((deviation - tolerance) / scale, 0.0, 1.0)
    per_joint_penalty = torch.expm1(sharpness * normalized_excess) / math.expm1(sharpness)
    return per_joint_penalty.mean(dim=1)


def joint_deviation_l1_above_head_height(
    env: ManagerBasedRLEnv,
    head_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    minimum_head_height: float = 0.6,
) -> torch.Tensor:
    """Penalize joint deviation from default only after the head reaches standing height."""
    asset = env.scene[asset_cfg.name]
    head_height = asset.data.body_pos_w[:, head_cfg.body_ids, 2].max(dim=1).values
    deviation = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ).sum(dim=1)
    return deviation * (head_height >= minimum_head_height).float()


# ---------------------------------------------------------------------------
# 上体が垂直なときのみ効く関節姿勢誤差ペナルティ
# ---------------------------------------------------------------------------
def joint_deviation_l1_when_upright(
    env: ManagerBasedRLEnv,
    max_tilt_deg: float = 30.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """関節の default 姿勢からの L1 偏差。ただし上体が概ね垂直なときだけ値を返す。

    ``asset_cfg.joint_ids`` で指定した関節の ``|q - q_default|`` の総和 (= ターゲット姿勢誤差)
    を返すが、上体 (Trunk) の roll・pitch が **ともに** ``max_tilt_deg`` 以内のときのみ適用し、
    それ以外 (寝ている / 大きく傾いている間) は 0 にする。

    狙い: 寝姿勢から起き上がる途中の大きな関節運動をこのペナルティで妨げないようにし、
    ある程度立ち上がってからターゲット (rough と同じ立位) 姿勢へ収束させる。

    Args:
        max_tilt_deg: 適用条件とする roll/pitch の上限 [deg]。
        asset_cfg: 対象アセット。``joint_names`` で誤差を測る関節を指定する。
    """
    asset = env.scene[asset_cfg.name]

    # 関節偏差 (指定関節の |q - q_default| 総和) — upstream joint_deviation_l1 と同じ計算。
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(angle), dim=1)

    # Trunk の roll/pitch を root quaternion から取得し、ともに閾値内かを判定。
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    roll = wrap_to_pi(roll)
    pitch = wrap_to_pi(pitch)
    max_tilt = math.radians(max_tilt_deg)
    upright = (torch.abs(roll) < max_tilt) & (torch.abs(pitch) < max_tilt)

    return deviation * upright.float()
