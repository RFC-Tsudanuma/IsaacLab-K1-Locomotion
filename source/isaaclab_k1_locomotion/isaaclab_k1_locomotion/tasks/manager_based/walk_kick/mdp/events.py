from __future__ import annotations

import math
import os
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_ball_in_front_of_robot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    dist_range: tuple[float, float] = (0.5, 0.8),
    half_angle: float = math.pi / 3,
    ball_radius: float = 0.11,
    spawn_clearance: float = 0.0,
    speed_range: tuple[float, float] = (0.0, 0.0),
    spin_from_speed: bool = False,
    rand_spin_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """ボールをロボット前方コーン内にリセットする。

    ロボットの実際のリセット後の姿勢を基準に、±half_angle のコーン内・dist_range の距離で
    ランダム配置する。``speed_range`` を与えると水平方向のランダム初速も乗せる。

    NOTE: 基準は ``default_root_state`` ではなく ``root_pos_w`` / ``root_quat_w``。
          reset_base (reset_root_state_uniform) がロボットを default から x,y ±0.5m・
          yaw ±π だけランダムにずらすため、default を基準にするとボールがロボットから見て
          まったく別の場所 (最悪は足の間) に湧く。このイベントは reset_base より後に走り
          (``__post_init__`` で後から追加されるので ``__dict__`` の末尾に入る)、
          ``write_root_link_pose_to_sim`` が内部バッファを即時更新するので、ここでは
          リセット済みの姿勢が読める。

    NOTE: dist_range の下限はロボットの足がボールに初期接触しない距離にすること。
          距離は base 中心からなので、ball_radius (0.11m) と足の張り出し (~0.15m) を
          考えると 0.3m では正面に湧いたときに接触する。

    Args:
        spawn_clearance: 接地高さからさらに浮かせる量 [m]。凹凸地形 (terrain generator) では
            env origin の z が「サブ地形中央付近の最大高さ」なので、そこから離れた xy では
            地面がそれより数 cm 低いことも高いこともある。地面より下に湧くとボールが
            めり込んだまま弾かれるため、地形の起伏ぶんだけ浮かせて落下させる。
            **平坦地形では 0.0 のままにすること** (既定値。既存タスクの挙動を変えない)。
        speed_range: 水平初速の大きさ [m/s] のサンプリング範囲。方向は 0-2π の一様乱数。
            既定 ``(0.0, 0.0)`` は静止配置 = 既存タスクの挙動。

            NOTE: 上限は必ず kick_state の ``v_thresh`` (既定 0.8 m/s) より **十分下**に
                  すること。初速が閾値を超えると、ロボットが触る前に値 latch が成立して
                  「蹴っていないのに τ_direction が凍結される」ことになる。
        spin_from_speed: True にすると、初速を与えたボールに **転がりと辻褄の合う回転**
            を乗せる。滑らずに転がるときの角速度は ω = (ẑ × v) / ball_radius なので、
            それをそのまま入れる。False (既定) では角速度 0 = 従来どおり
            「速度だけ持っているのに回っていない」ボールになる。

            初速が 0 のボール (``speed_range`` 既定) では転がりぶんは 0 なので、
            ``rand_spin_range`` のランダム回転だけが乗る。
        rand_spin_range: 追加でかけるランダム回転の大きさ [rad/s] のサンプリング範囲。
            軸は 3D の一様乱数方向。既定 ``(0.0, 0.0)`` では何も足さない。

            実機のボールは前の蹴りや床の凹凸で必ず何かしら回っている。回転はマグヌス
            効果ではなく **接触時の接線方向のインパルス** を通じて跳ね方を変えるので、
            回転ゼロだけで学習すると足↔ボールの接触モデルに寄りかかったキックになる。

            NOTE: 既定 ``(0.0, 0.0)`` は既存タスクの挙動そのまま (角速度 0)。
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]
    n = len(env_ids)

    # リセット後の実際のロボット位置・向き
    robot_pos_w = robot.data.root_pos_w[env_ids]
    quat = robot.data.root_quat_w[env_ids]
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    dist = torch.empty(n, device=env.device).uniform_(*dist_range)
    angle_offset = torch.empty(n, device=env.device).uniform_(-half_angle, half_angle)
    target_angle = yaw + angle_offset

    state = ball.data.default_root_state[env_ids].clone()
    state[:, 0] = robot_pos_w[:, 0] + dist * torch.cos(target_angle)
    state[:, 1] = robot_pos_w[:, 1] + dist * torch.sin(target_angle)
    # z の基準は env origin (= 地形原点)。terrain_type="plane" では 0.0 なので、
    # 平坦タスクではこれまでどおり絶対高さ ball_radius に置かれる。
    state[:, 2] = env.scene.env_origins[env_ids, 2] + ball_radius + spawn_clearance
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    state[:, 7:] = 0.0

    # 水平初速 (転がるボール用)。既定 (0.0, 0.0) では静止のまま。
    if speed_range[1] > 0.0:
        speed = torch.empty(n, device=env.device).uniform_(*speed_range)
        vel_angle = torch.empty(n, device=env.device).uniform_(0.0, 2.0 * math.pi)
        state[:, 7] = speed * torch.cos(vel_angle)
        state[:, 8] = speed * torch.sin(vel_angle)

        # 転がりと辻褄の合う角速度 ω = (ẑ × v) / R。
        # ẑ × (vx, vy, 0) = (−vy, vx, 0) なので、x 軸まわりに −vy/R、y 軸まわりに +vx/R。
        if spin_from_speed:
            state[:, 10] = -state[:, 8] / ball_radius
            state[:, 11] = state[:, 7] / ball_radius
            state[:, 12] = 0.0

    # ランダム回転 (3D の一様な軸まわり)。転がりぶんの上に足す。
    if rand_spin_range[1] > 0.0:
        spin_mag = torch.empty(n, device=env.device).uniform_(*rand_spin_range)
        # 3D の一様ランダム方向: 正規分布のベクトルを正規化する
        axis = torch.randn(n, 3, device=env.device)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-6)
        state[:, 10:13] += spin_mag.unsqueeze(-1) * axis

    ball.write_root_state_to_sim(state, env_ids=env_ids)


def perturb_ball_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    distance_range: tuple[float, float] = (0.01, 0.05),
) -> None:
    """ボールをランダムな方向に distance_range [m] だけずらす。"""
    ball = env.scene[ball_cfg.name]
    n = len(env_ids)

    state = ball.data.root_state_w[env_ids].clone()

    dist = torch.empty(n, device=env.device).uniform_(*distance_range)
    angle = torch.empty(n, device=env.device).uniform_(0.0, 2.0 * torch.pi)
    state[:, 0] += dist * torch.cos(angle)
    state[:, 1] += dist * torch.sin(angle)

    ball.write_root_state_to_sim(state, env_ids=env_ids)


# --------------------------------------------------------------------------- #
# Reference State Initialization (RSI): walk ポリシーの歩行状態からリセットする
# --------------------------------------------------------------------------- #
# 実機では walk (model_34995.pt) で歩いてから walk_kick に切り替える。walk_kick を
# 「立ち姿勢から」学習すると、切り替え時に walk が作った歩行状態 (脚の振り出し途中・
# 上下動・位相) を一度も見たことがない状態から始まるので、一瞬がくっとなる。
#
# scripts/rsl_rl/record_walk_states.py が walk ポリシーの歩行中スナップショットを
# npz に集めるので、それを reset 時に流し込んで学習分布を実機の切り替え時と揃える。

_WALK_STATE_CACHE: dict[str, dict] = {}
_RSI_LAST_ACTION_ATTR = "_rsi_last_action"
_PHASE_OFFSET_ATTR = "_phase_offset_per_env"


def _load_walk_states(env: ManagerBasedRLEnv, path: str, asset_cfg: SceneEntityCfg) -> dict:
    """npz を読み、関節を名前で現在の articulation / action 並びに対応付けて返す。

    プロセス内でキャッシュするので、reset のたびにディスクを触ることはない。
    """
    import numpy as np

    key = f"{path}|{env.device}"
    cached = _WALK_STATE_CACHE.get(key)
    if cached is not None:
        return cached

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"walk state プールが見つからない: {path}\n"
            "scripts/rsl_rl/record_walk_states.py で先に収集すること。"
        )
    raw = np.load(path, allow_pickle=False)

    def _t(name: str) -> torch.Tensor:
        return torch.as_tensor(raw[name], dtype=torch.float32, device=env.device)

    robot = env.scene[asset_cfg.name]
    robot_names = list(robot.joint_names)
    pool_names = [str(n) for n in raw["joint_names"]]
    missing = [n for n in pool_names if n not in robot_names]
    if missing:
        raise ValueError(f"walk state の関節 {missing} が現在のロボットに無い: {robot_names}")
    # プールの列 i を articulation の関節 dst_ids[i] に書き込む (write_joint_state_to_sim
    # の joint_ids がこの対応表)。
    dst_ids = [robot_names.index(n) for n in pool_names]

    # last_action は action 項の並び。action 項の関節名が取れる構成なら名前で並べ替える。
    act_pool_names = [str(n) for n in raw["action_joint_names"]]
    act_dst_cols = None
    try:
        term = env.action_manager.get_term("joint_pos")
        env_act_names = list(getattr(term, "_joint_names", None) or getattr(term, "joint_names"))
    except Exception:
        env_act_names = None
    if env_act_names is not None and env_act_names != act_pool_names:
        if sorted(env_act_names) != sorted(act_pool_names):
            raise ValueError(
                f"last_action の関節集合が一致しない:\n  pool={act_pool_names}\n  env ={env_act_names}"
            )
        act_dst_cols = torch.tensor([act_pool_names.index(n) for n in env_act_names], device=env.device)

    phase = _t("gait_phase")
    data = {
        "root_height": _t("root_height"),
        "root_quat_wxyz": _t("root_quat_wxyz"),
        "root_lin_vel_b": _t("root_lin_vel_b"),
        "root_ang_vel_b": _t("root_ang_vel_b"),
        "joint_pos": _t("joint_pos"),
        "joint_vel": _t("joint_vel"),
        "last_action": _t("last_action") if act_dst_cols is None else _t("last_action")[:, act_dst_cols],
        "command": _t("command"),
        "gait_phase": phase,
        "has_phase": bool(torch.isfinite(phase).all()),
        "dst_ids": dst_ids,
        "num_samples": int(raw["root_height"].shape[0]),
    }
    _WALK_STATE_CACHE[key] = data
    print(f"[INFO] walk state プールを読み込んだ: {path} ({data['num_samples']} サンプル, "
          f"phase={'あり' if data['has_phase'] else 'なし'})")
    return data


def reset_from_walk_states(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    states_path: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    probability: float = 1.0,
    set_gait_phase: bool = True,
    set_last_action: bool = True,
) -> None:
    """walk ポリシーの歩行中スナップショットからロボットをリセットする。

    上書きするのは「歩容そのもの」に相当する量だけで、**x, y, yaw は既存の
    ``reset_base`` が決めた値をそのまま残す**。こうしておくと ``reset_ball_in_front_of_robot``
    (リセット後のロボット姿勢を基準にボールを置く) と順序に依存せず両立する。

      * base 高さ / roll / pitch
      * base の線速度・角速度 (サンプルの base フレーム量を現在の yaw で世界系に戻す)
      * 脚 12 関節の角度・角速度
      * 歩行位相オフセット (``gait_phase_sincos`` が t=0 で読む値)
      * 前ステップの関節指令 (``prev_joint_request_rsi`` 観測が 1 ステップ目だけ返す)

    Args:
        states_path: ``record_walk_states.py`` が吐いた npz のパス。
        probability: RSI を適用する env の割合。1.0 未満なら残りは従来どおりの
            立ち姿勢リセットになるので、両方の初期条件を混ぜて学習できる。
        set_gait_phase: 位相オフセットも合わせるか。walk_kick の位相は
            ``2π * pf * t + offset`` で t はリセットからの経過時間なので、offset に
            サンプルの位相を入れると歩容の途中から始められる。
        set_last_action: ``prev_joint_request`` 用のバッファを埋めるか。
            効かせるには観測項を :func:`~.observations.prev_joint_request_rsi` に
            差し替えること (walk_kick_env_cfg の ``_apply_walk_state_init`` が行う)。
    """
    from isaaclab.utils.math import quat_apply, quat_mul, yaw_quat

    data = _load_walk_states(env, states_path, asset_cfg)
    robot = env.scene[asset_cfg.name]

    if probability < 1.0:
        pick = torch.rand(len(env_ids), device=env.device) < probability
        env_ids = env_ids[pick]
    if len(env_ids) == 0:
        return

    n = len(env_ids)
    sel = torch.randint(0, data["num_samples"], (n,), device=env.device)

    # -- root: x, y, yaw は温存し、高さ・roll/pitch・速度だけ差し替える
    state = robot.data.root_state_w[env_ids].clone()
    yaw_q = yaw_quat(state[:, 3:7])
    quat = quat_mul(yaw_q, data["root_quat_wxyz"][sel])
    state[:, 2] = env.scene.env_origins[env_ids, 2] + data["root_height"][sel]
    state[:, 3:7] = quat
    state[:, 7:10] = quat_apply(quat, data["root_lin_vel_b"][sel])
    state[:, 10:13] = quat_apply(quat, data["root_ang_vel_b"][sel])
    robot.write_root_state_to_sim(state, env_ids=env_ids)

    # -- 脚関節
    robot.write_joint_state_to_sim(
        data["joint_pos"][sel],
        data["joint_vel"][sel],
        joint_ids=data["dst_ids"],
        env_ids=env_ids,
    )

    # -- 歩行位相: リセット直後は t=0 なので、位相 = オフセットになる
    if set_gait_phase and data["has_phase"]:
        buf = getattr(env, _PHASE_OFFSET_ATTR, None)
        if buf is None:
            buf = torch.zeros(env.num_envs, device=env.device)
            setattr(env, _PHASE_OFFSET_ATTR, buf)
        buf[env_ids] = data["gait_phase"][sel]

    # -- 前ステップの関節指令
    if set_last_action:
        act = data["last_action"]
        buf = getattr(env, _RSI_LAST_ACTION_ATTR, None)
        if buf is None:
            buf = torch.zeros(env.num_envs, act.shape[1], device=env.device)
            setattr(env, _RSI_LAST_ACTION_ATTR, buf)
        buf[env_ids] = act[sel]
