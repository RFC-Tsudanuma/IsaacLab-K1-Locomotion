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
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    require_upright: bool = False,
) -> torch.Tensor:
    """base (Trunk) の高さそのものへの報酬。立位に近いほど高い。

    ``min_height`` (寝た姿勢相当) から ``target_height`` (立位相当) へ 0→1 で線形に増加し、
    ``target_height`` 以上で 1 に飽和する (跳ね上がりを過剰に報酬しない)。

    Args:
        target_height: 立位時の目標 base 高さ [m]。ここで報酬が 1 に飽和する。
        min_height: 報酬が 0 になる下限高さ [m] (寝た姿勢相当)。
        sensor_cfg: 地面高さ補正用の RayCaster センサ。None なら地面 z=0。
        require_upright: True なら直立度 (_upright_factor) を掛け、反転して trunk を
                         高く上げても報酬にならないようにする (handstand exploit 対策)。
    """
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - _ground_height(env, sensor_cfg)
    reward = torch.clamp((height - min_height) / (target_height - min_height), 0.0, 1.0)
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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Head.*"),
    sensor_cfg: SceneEntityCfg | None = None,
    require_upright: bool = False,
) -> torch.Tensor:
    """頭リンクの高さへの報酬。頭が高く持ち上がるほど高い。

    ``asset_cfg.body_names`` に複数マッチする場合は最も高いリンクの高さを使う。
    ``min_height`` → ``target_height`` を 0→1 で線形に増加し、以上で 1 に飽和する。

    Args:
        target_height: 立位時の目標頭高さ [m]。ここで報酬が 1 に飽和する。
        min_height: 報酬が 0 になる下限高さ [m]。
        asset_cfg: 頭リンクを指す body_names を持つアセット設定。
        sensor_cfg: 地面高さ補正用の RayCaster センサ。None なら地面 z=0。
        require_upright: True なら直立度 (_upright_factor) を掛け、反転して頭を高く
                         上げても報酬にならないようにする (handstand exploit 対策)。
    """
    asset = env.scene[asset_cfg.name]
    # body_ids で指定された (1個以上の) リンクのうち最も高い z を頭高さとする。
    head_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].max(dim=1).values
    height = head_z - _ground_height(env, sensor_cfg)
    reward = torch.clamp((height - min_height) / (target_height - min_height), 0.0, 1.0)
    if require_upright:
        reward = reward * _upright_factor(asset)
    return reward


# ---------------------------------------------------------------------------
# 4. 足裏の接地
# ---------------------------------------------------------------------------
def feet_ground_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    threshold: float = 1.0,
    require_upright: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """足裏 (foot link) が地面に接触していることへの報酬。

    ``sensor_cfg`` で指定した足リンクのうち、接触力が ``threshold`` [N] を超えているものの
    割合 (0〜1) を返す。両足接地で 1、片足で 0.5、両足浮きで 0。

    Args:
        sensor_cfg: 足リンクを指す ContactSensor 設定。
        threshold: 接地とみなす接触力の下限 [N]。
        require_upright: True なら直立度 (_upright_factor) を掛ける。これがないと
                         「寝たまま足裏だけ接地」で満点が取れてしまい、寝姿勢の局所最適に
                         はまる。上体が立っているときの「足で立つ」ことだけを報酬にする。
        asset_cfg: 直立度を測るアセット (require_upright=True のとき使用)。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w_history: [N, history, num_bodies, 3] → 最新ステップの各足の力ノルム
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)  # [N, hist, num_feet]
    in_contact = forces[:, -1, :] > threshold  # [N, num_feet]
    reward = in_contact.float().mean(dim=1)
    if require_upright:
        reward = reward * _upright_factor(env.scene[asset_cfg.name])
    return reward


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


def feet_vertical_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_fraction: float = 1.0,
    require_upright: bool = False,
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
    total = fz.sum(dim=1)  # 両足合計の垂直反力 [N]
    asset = env.scene[asset_cfg.name]
    weight = asset.data.default_mass.to(asset.device).sum(dim=1) * 9.81  # 体重 [N]
    reward = (total / weight.clamp(min=1e-6)).clamp(max=max_fraction)
    if require_upright:
        reward = reward * _upright_factor(asset)
    return reward


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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """上体 (Trunk) が鉛直にまっすぐ立っているほど高い、寝姿勢からの連続勾配つき報酬。

    body 座標の重力 z 成分 ``g_z = projected_gravity_b[:,2]`` は直立で -1、真横で 0、
    上下反転で +1。直立度を ``(1 - g_z)/2`` で [0,1] に写す:
        直立 (g_z=-1) → 1.0、 真横/寝 (g_z≈0) → 0.5、 反転 (g_z=+1) → 0.0。
    全域で単調な勾配があるので「寝た状態から上体を起こす」方向に常に勾配が出る。
    反転 (handstand) では 0 になるので逆立ち exploit も同時に防げる。

    NOTE: 旧版は水平成分の指数カーネル exp(-(gx²+gy²)/sigma) だったが、(1) 反転でも満点
    (逆立ち exploit)、(2) それを潰すと tilt≥90° が一律 0 になり寝姿勢から勾配が消える、
    という二つの問題があったため単調写像に変更した。``sigma`` は後方互換のため残すが未使用。
    """
    asset = env.scene[asset_cfg.name]
    g_z = asset.data.projected_gravity_b[:, 2]  # -1 直立, 0 真横, +1 反転
    return (1.0 - g_z) * 0.5


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
    stood_up = (com_z > com_height_threshold) & upright
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
