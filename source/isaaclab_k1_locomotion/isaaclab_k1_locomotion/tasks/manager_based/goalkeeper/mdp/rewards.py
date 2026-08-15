# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (goalkeeper) タスク専用の報酬関数。

exp 型の距離報酬は σ を 1 本にすると「遠い目標で勾配が消えて足踏み均衡に落ちる」
既知の問題があるため、σ の異なる 2 項 (粗い/細かい) を cfg 側で重ねて使う
(multi-scale。exp 報酬の諦め問題対策)。
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from ...around_ball.mdp.observations import _high_action_cmd
from ...locomotion.mdp.events import get_phase_freq
from .observations import compute_target_y, gk_buffers, robot_pos_goal
from .terminations import update_save_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stance_foot_flat(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
) -> torch.Tensor:
    """**実際に接地している足だけ** の足裏の傾き (pitch² + roll²) を返すペナルティ (weight<0)。

    つま先立ち = 足首が底屈して足裏が前傾した状態。既存の
    ``feet_parallel_to_ground`` はポテンシャル形式 (差分報酬) なので定常的なつま先立ちに
    勾配が出ず効かなかった。本項は **状態ペナルティ** なので傾いている限り毎ステップ罰す。

    接地判定は **接触センサ (接触力 > force_threshold)** で行う。旧実装は位相ベースの
    接地推定を使っていたが、位相モデルは前進歩行前提で、速い横移動では実際の足の動きと
    合わず「移動方向と反対の足がずっとつま先立ち」なのに位相上は swing 扱いされて
    罰を逃れていた (実測で判明)。接触力ベースなら、つま先だけで接地している足も
    「接地中」と正しく判定でき、その傾きを直接罰せる。遊脚 (完全に浮いている足) は
    接触力が無いので自動的に対象外になり、足上げ (foot_clearance) と干渉しない。

    Returns:
        接地脚の (pitch² + roll²) の和 [(rad)²], shape (N,)。weight を負にして使う。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # 傾き (robot 側) と接触 (sensor 側) は別コンテナから body_ids を解決するため、
    if getattr(env, "_gk_stance_flat_order_ok", None) is None:
        asset_names = [asset.body_names[i] for i in asset_cfg.body_ids]
        sensor_names = [contact_sensor.body_names[i] for i in sensor_cfg.body_ids]
        if asset_names != sensor_names:
            raise RuntimeError(
                f"stance_foot_flat: 足の並び順が robot={asset_names} / sensor={sensor_names} "
                "で一致しません。body_names の指定を見直してください。"
            )
        env._gk_stance_flat_order_ok = True

    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # [N, F, 4]
    n_env, n_foot = foot_quat.shape[0], foot_quat.shape[1]
    roll, pitch, _ = euler_xyz_from_quat(foot_quat.reshape(-1, 4))
    err = (torch.square(wrap_to_pi(pitch)) + torch.square(wrap_to_pi(roll))).reshape(n_env, n_foot)

    # 接触判定は履歴の最大値で取る (touchdown 直後のチャタリングに強い)。
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)  # [N, hist, F]
    in_contact = (forces.max(dim=1)[0] > float(force_threshold)).float()  # [N, F]

    return (err * in_contact).sum(dim=1)


def flight_phase(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """**両足とも接地していない** ステップを 1 とするペナルティ [0, 1] (weight<0 で使う)。

    跳躍 (hopping) の定義そのもの。歩行は常にどちらかの足が接地しているので、
    両足が同時に浮くのは「跳んだ」ときだけ。

    なぜこれが要るか:
        足上げ報酬 ``foot_clearance_ji`` が見るのは足リンクの **ワールド z (絶対高さ)**
        なので、「遊脚を股関節・膝で上げる」以外に「体ごと持ち上げる」でも達成できる。
        目標 7cm + 高速 (カリキュラム stage 3) の組み合わせで、後者 = 跳躍が
        最安の解として選ばれた (2026-07-29 実測: 上下動 raw 0.028 → 0.052)。

        従来は ``lin_vel_z_l2`` (上下動全般のペナルティ) で抑えようとしたが、
        あれは歩行に必要な重心の上下動まで巻き添えにするため、強めると横速度が落ちる。
        -2.5 まで上げても目標 7cm + 高速では跳躍を止められなかった。
        本項は「両足浮き」だけを罰するので、**片足を高く上げること自体は無罰**であり、
        足上げにも横速度にも干渉しない。

    ★ locomotion 側に同名の ``both_feet_not_in_contact`` があるが使わない:
        1. あちらは -1.0 を返すので weight を **正** にしないとペナルティにならない
           (負にすると跳躍を報酬してしまう)。本関数は他の項と揃えて正値を返す。
        2. あちらは ``[:, -1, ...]`` = 履歴の **最古** サンプルを見ている
           (net_forces_w_history は index 0 が最新)。本関数は最新を見る。
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # index 0 が最新。跳躍は数十 ms 続くので履歴の最大値ではなく最新値で判定する
    # (最大値だと「直前まで接地していた」だけで接地扱いになり、浮きを取りこぼす)。
    forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :].norm(dim=-1)  # [N, F]
    in_contact = forces > float(force_threshold)
    return (~in_contact.any(dim=1)).float()


def _target_y_error(env: "ManagerBasedRLEnv", max_y: float) -> torch.Tensor:
    """robot_y − target_y (ゴール座標系) の符号付き誤差 (N,)。"""
    return robot_pos_goal(env)[:, 1] - compute_target_y(env, max_y=max_y)


def track_target_y(
    env: "ManagerBasedRLEnv",
    std: float = 0.5,
    max_y: float = 1.3,  # = GOAL_HALF_WIDTH (ゴール幅 2.6m)
) -> torch.Tensor:
    """目標 y への距離のガウス報酬 exp(-err²/σ²) ∈ [0, 1]。

    cfg 側で std の違う 2 項 (例: 0.5 と 0.15) を重ねてマルチスケールにする。
    目標はステージ1 = ランダム点、ステージ2 以降 = ボール到達予測点 (compute_target_y)。
    """
    err = _target_y_error(env, max_y)
    return torch.exp(-torch.square(err) / std**2)


def target_reach_velocity_direct(
    env: "ManagerBasedRLEnv",
    deadband: float = 0.12,
    v_cap: float = 1.3,
    stop_speed: float = 0.5,
    max_y: float = 1.3,  # = GOAL_HALF_WIDTH (ゴール幅 2.6m)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """目標 y 方向への横移動速度に比例する密報酬 ∈ [-1, 1] (**直接制御版**)。

    :func:`target_reach_velocity` (階層版) の単一ポリシー版。直接制御版の
    ステージ2/3 には y 方向の dense 報酬が一つも無く、``save_touch_bonus`` (+100) と
    ``termination_penalty`` (-200) というスパース報酬しか無かった。しかもそれらは
    ボール到達まで 3〜5 秒先で、gamma=0.99 @ 50Hz では 0.99^250 ≈ 0.08 まで減衰する。
    一方で動けば energy / feet_slide / dof_vel の減点が即座に確定するため、
    「必要最小限しか動かない」が最適解になっていた。本項がその穴を埋める。

    exp 型 (:func:`track_target_y`) ではなく速度の線形報酬にしてあるのは、目標が
    遠いと exp が数値的にゼロになり勾配が消えるため (諦め問題)。線形ならどれだけ
    離れていても「目標方向へ速く動けば得」という勾配が残る。

    ★ 階層版との違い: 階層版は deadband 内の停止判定に ``_high_action_cmd``
      (上位ポリシーが frozen に渡すコマンド) を使う。直接制御版にそのバッファは
      存在せず **常にゼロが返る** ため、そのまま流用すると「deadband 内なら何を
      していても満額 1.0」という抜け道になる (足踏みしたまま満額取り = 階層版
      Stage1 の既知の失敗そのもの)。本関数は実ベース速度で停止を判定する。

    Args:
        deadband: この誤差 [m] 以内を「到達済み」とみなし、停止報酬に切り替える。
        v_cap: 満額とみなす横移動速度 [m/s]。LATERAL_TARGET_SPEED と揃える。
        stop_speed: 停止報酬がゼロになるベース速度 [m/s]。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    err = _target_y_error(env, max_y)

    # 誤差を減らす向きの横速度 (world y = ゴールライン方向)。逆方向なら負。
    v_y = robot.data.root_lin_vel_w[:, 1]
    toward = -torch.sign(err) * v_y
    r_move = (toward / v_cap).clamp(-1.0, 1.0)

    # 到達済みの区間は「止まっているほど高い」に切り替える。無条件で満額にすると
    # 目標付近で足踏みするだけで報酬を取り切れ、停止方向の勾配が消える。
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    r_stop = (1.0 - speed / stop_speed).clamp(0.0, 1.0)

    return torch.where(err.abs() <= deadband, r_stop, r_move)


def target_reach_velocity(
    env: "ManagerBasedRLEnv",
    deadband: float = 0.12,
    v_cap: float = 0.8,
    cmd_scale: float = 0.5,
    max_y: float = 1.3,  # = GOAL_HALF_WIDTH (ゴール幅 2.6m)
) -> torch.Tensor:
    """目標方向への横移動速度に比例する密報酬 ∈ [-1, 1]。

    目標が遠くても勾配が一定に出る (exp 型の勾配消失の補完)。逆方向への移動は負。

    deadband 内 (到達済み) では「上位コマンドを 0 へ落とすほど高い」線形報酬
    ``1 - |cmd|/cmd_scale`` に切り替える。旧実装は deadband 内を無条件で満額 1 に
    しており、**足踏みしたまま (コマンドを残したまま) 目標付近に立つだけで
    主報酬が取り切れてしまい、停止方向への勾配が一切出なかった**
    (Stage1 の学習で hold_at_target が終始 0.000 だった直接の原因)。
    線形なのでコマンドがどんなに大きくても勾配が消えない (exp 型の欠点を回避)。
    """
    err = _target_y_error(env, max_y)
    robot = env.scene["robot"]
    v_y = robot.data.root_lin_vel_w[:, 1]
    toward = -torch.sign(err) * v_y  # 誤差を減らす向きの速度
    r_move = (toward / v_cap).clamp(-1.0, 1.0)

    cmd_norm = torch.norm(_high_action_cmd(env)[:, :3], dim=1)
    r_stop = (1.0 - cmd_norm / cmd_scale).clamp(0.0, 1.0)
    return torch.where(err.abs() <= deadband, r_stop, r_move)


def hold_at_target(
    env: "ManagerBasedRLEnv",
    pos_std: float = 0.25,
    cmd_std_coarse: float = 0.35,
    cmd_std_fine: float = 0.1,
    lin_vel_std: float = 0.4,
    yaw_rate_weight: float = 0.25,
    max_y: float = 1.3,  # = GOAL_HALF_WIDTH (ゴール幅 2.6m)
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """目標に到達した状態で「コマンドを 0 に落として静止」するほど高い報酬 [0, 1]。

    around_ball_ready の stop_when_ready と同じ構造 (gate × stop_cmd × stop_body)。
    stop_cmd (上位コマンドノルムのガウス) が足踏み局所最適対策の本命:
    frozen が初期姿勢で立つのはコマンドノルム < 0.05 (gait_phase ゼロ埋め閾値) の
    ときだけなので、実ベース速度だけを見る停止報酬では「コマンド 0.1〜0.3 を残した
    その場足踏み」がほぼ満額を取ってしまい、0.05 の壁を越える勾配が出ない。

    stop_cmd は σ の異なる 2 つのガウスの平均 (マルチスケール)。旧実装は σ=0.1 の
    単一ガウスで、歩行コマンド ~0.5 の地点では exp(-25)≈1e-11 と数値的にゼロになり
    「報酬の存在自体がポリシーから観測できない」勾配消失を起こしていた
    (Stage1 で hold_at_target が終始 0.000 だった要因のひとつ)。粗い σ=0.35 が
    遠くから 0.05 の壁際まで誘導し、細かい σ=0.1 が最後の押し込みを担当する。
    """
    robot = env.scene[robot_cfg.name]
    err = _target_y_error(env, max_y)
    gate = torch.exp(-torch.square(err) / pos_std**2)

    cmd_norm = torch.norm(_high_action_cmd(env)[:, :3], dim=1)
    stop_cmd = 0.5 * (
        torch.exp(-torch.square(cmd_norm) / cmd_std_coarse**2)
        + torch.exp(-torch.square(cmd_norm) / cmd_std_fine**2)
    )

    lin_speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    yaw_rate = robot.data.root_ang_vel_w[:, 2].abs()
    motion = lin_speed + yaw_rate_weight * yaw_rate
    stop_body = torch.exp(-torch.square(motion) / lin_vel_std**2)

    return gate * stop_cmd * stop_body


def track_lin_vel_y_exp(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """横方向 (base yaw frame の y) の速度コマンド追従だけを見るガウス報酬 [0, 1]。

    直接制御版ステージ1 の「横移動特化」を担う項。locomotion の
    ``track_lin_vel_xy_yaw_frame_exp`` は前後と左右を合算して評価するため、
    前進で誤差を稼いでも横で損してもトータルが同じになり、横方向へ学習圧が
    集中しない。本項を上乗せして、横の追従だけを追加で報酬する。

    y 成分のみを評価する以外は locomotion の追従報酬と同じ規約
    (base の yaw frame で評価、σ はガウス幅)。
    """
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vel_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    err = torch.square(cmd[:, 1] - vel_b[:, 1])
    return torch.exp(-err / std**2)


def lateral_speed_bonus(
    env: "ManagerBasedRLEnv",
    v_ref: float = 1.3,
    command_name: str = "base_velocity",
    min_cmd: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """「速い横移動コマンドが出ているとき、実際に何 m/s 出せたか」に比例する報酬 [0, 1]。

    追従報酬 (ガウス) だけだと、コマンドが実現不可能に速い領域では誤差が大きすぎて
    勾配がほぼ消え、「どうせ届かないから諦める」局所解に落ちやすい (exp 報酬の
    諦め問題)。本項は指令と実速度の一致ではなく **実速度そのもの** を線形に報酬する
    ので、上限付近でも「1cm/s でも速く」の勾配が残る。

    ``min_cmd`` 以上の横コマンドが出ている env でのみ有効 (低速時に暴れないため)。
    ``v_ref`` は正規化の基準速度で、目標とする横移動速度を入れる。
    """
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vel_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    # コマンド方向に沿った横速度 (逆方向に動いていれば負)
    toward = torch.sign(cmd[:, 1]) * vel_b[:, 1]
    gate = cmd[:, 1].abs() > min_cmd
    return (toward / v_ref).clamp(0.0, 1.0) * gate.float()


def save_touch_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """ボールに触れて弾いた瞬間の一回限りのボーナス (ボール 1 球につき 1 回)。"""
    newly = update_save_state(env)
    bufs = gk_buffers(env)
    fire = newly & (~bufs["touch_rewarded"])
    bufs["touch_rewarded"][fire] = True
    return fire.float()


def save_clearance_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """セーブ確定時に、危険圏からの遠ざけ具合に応じて払う一回限りの報酬 [0, 1]。

    「止めた」と「危険を除去した」を区別する。従来はゴール正面 0.3m に転がったまま
    止めても ``save_touch_bonus`` (+100) が満額で、実戦なら即座に押し込まれる結末と
    完全に弾き出した結末が同じ扱いだった。

    値は :func:`~..events._save_clearance_quality` がセーブ確定時にバッファへ書き、
    本関数が読み取ってゼロに戻す (読み捨て)。イベントと報酬の実行順に依存しないよう
    バッファ越しにしてあるので、最大 1 ステップ (20ms) 遅れて払われることがある。

    明示的なキック方向は教えない。「なるべく外へ弾け」という圧力だけを与え、
    どう当てるかはポリシーに任せる (キック技術は ball_kick タスクの担当)。
    """
    bufs = gk_buffers(env)
    q = bufs["save_quality"].clone()
    bufs["save_quality"].zero_()
    return q


def return_to_center_after_save(
    env: "ManagerBasedRLEnv",
    std: float = 0.5,
) -> torch.Tensor:
    """ボールを弾いた後、ゴール中央 (y=0) へ戻るほど高い報酬 (タッチ後のみ有効)。

    ★ ゲートは ``touched`` のみで判定する (2026-07-24)。以前は
      ``ball_active & touched`` だったが、エピソード継続モードでは
      :func:`~..events.relaunch_ball_after_save` がセーブ確定時に ``ball_active``
      を False にするため、**まさに中央へ戻るべき「次の球までの空き時間」に
      ゲートが閉じてしまう** という取り違えになっていた。
      ``touched`` は次の球の発射時にリセットされるので、
      「弾いた後〜次の球が来るまで」を正しく覆う。
    """
    bufs = gk_buffers(env)
    y = robot_pos_goal(env)[:, 1]
    r = torch.exp(-torch.square(y) / std**2)
    gate = bufs["touched"]
    return r * gate.float()


def hold_default_pose_after_save(
    env: "ManagerBasedRLEnv",
    std: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """★ 2026-08-11 追加: セーブ後の保持区間だけ、初期姿勢に近いほど高い報酬 [0, 1]。

    ユーザー指示により :func:`return_to_center_after_save` を廃止し、代わりに
    「止めた地点で数秒間、初期姿勢のまま立つ」を学習させる。転倒しないかを
    切り分けて確認するのが目的。

    区間は :func:`~.observations.post_save_hold` (タッチ〜次の球、約3.0秒)。
    その間 :func:`~.observations.task_drive_vector` が指令をゼロにするので歩容は
    停止し、この報酬が関節を既定姿勢へ引き戻す。
    """
    from .observations import post_save_hold

    robot: Articulation = env.scene[asset_cfg.name]
    dev = robot.data.joint_pos - robot.data.default_joint_pos
    # 関節数で正規化して std の意味を関節あたりの平均ずれ [rad] に揃える
    err2 = torch.square(dev).mean(dim=1)
    return torch.exp(-err2 / std**2) * post_save_hold(env).float()


def stay_on_goal_line(
    env: "ManagerBasedRLEnv",
    std: float = 0.3,
    x_offset: float | None = None,
) -> torch.Tensor:
    """守備面 (x ≈ guard_x、ゴールラインのフィールド側) に留まるほど高い報酬 [0, 1]。

    ロボットはライン上ではなく **ラインの guard_x [m] 前** で守る (ボールを
    ゴールの外側で止めるため + 前に出るほどシュートコースが狭まる)。
    ``x_offset`` を省略すると GoalkeeperParamsCfg.guard_x を使う (JSON で変更可能)。
    ガウス (σ=0.3) の緩い誘導なので、ボールに合わせた前後の微調整は妨げない。

    ★ 2026-08-11: セーブ後の保持区間 (:func:`~.observations.post_save_hold`) では
      ゲートを閉じる。指令はゼロにして「止めた地点で立つ」ようにしたのに、この報酬が
      守備面へ引き戻すと指令と報酬が食い違うため。向き (face_field) は初期姿勢の一部
      なので保持区間でも有効なまま残す。
    """
    from .observations import post_save_hold

    if x_offset is None:
        x_offset = float(env.cfg.goalkeeper.guard_x)
    x = robot_pos_goal(env)[:, 0]
    r = torch.exp(-torch.square(x - x_offset) / std**2)
    return r * (~post_save_hold(env)).float()


def face_field(
    env: "ManagerBasedRLEnv",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """フィールド側 (+x, yaw=0) を向き続けるほど高い報酬 [0, 1]。

    セーブは vy の横ステップで行う想定なので、体の向きを固定して
    「横歩き」のコマンド意味論を保つ shaping。
    """
    robot = env.scene[asset_cfg.name]
    heading = robot.data.heading_w  # env はワールド軸に沿って配置されるので yaw=0 が +x
    return torch.exp(-torch.square(heading) / std**2)


# ---------------------------------------------------------------------------
# 横移動特化タスク (goalkeeper_lateral_env_cfg.py) 用の報酬
#
# 07-28 (k1_gk_direct_stage1/2026-07-28_17-13-15) の実測を出発点にしている:
#   横追従は誤差 2% で極めて良好 / 立ち上がり 0.6s / yaw ドリフト 10°/s /
#   足上げ 4cm 台 / 後退ドリフト -0.10 m/s。
# 定常速度はもう伸ばす余地がないので、**過渡 (立ち上がり) と姿勢** を直接評価する。
# ---------------------------------------------------------------------------


def onset_speed_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    v_ref: float = 1.3,
    min_cmd: float = 0.6,
    onset_s: float = 0.8,
    change_tol: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """**速度コマンドが変わった直後 ``onset_s`` 秒だけ** 効く、実速度の線形報酬 [0, 1]。

    なぜ要るか:
        セーブに必要な横移動は 0.3〜0.8m が中心で、その帯域はまるごと加速区間に入る。
        2.6m 横断は 1.278 m/s で 2.35s、1.6 m/s に上げても 1.99s しか縮まらない
        (定常 25% 増で 15% 短縮) 一方、立ち上がり 0.6s → 0.4s は全域に効く。
        既存の :func:`lateral_speed_bonus` は定常速度しか見ておらず、
        「ゆっくり立ち上がってから定常で稼ぐ」解と「即座に立ち上がる」解を区別できない。

    設計:
        * **線形** (ガウスではない)。過渡の序盤は誤差が必ず大きく、exp 型だと
          そこで勾配が消えて「諦め」に落ちる (exp 報酬の諦め問題)。
        * 指令方向への射影速度なので、逆方向に飛び出すと 0 になる。
        * コマンドが大きく変わった (``change_tol``) 直後だけ有効。定常区間で
          二重取りさせない。窓の長さ ``onset_s`` は目標立ち上がり時間より少し長く取る。

    Returns:
        [0, 1] の報酬。窓の外・低速コマンドでは 0。
    """
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    from .events import update_lateral_buffers

    bufs = update_lateral_buffers(env, command_name=command_name, change_tol=change_tol)

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vel_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])

    cmd_xy = cmd[:, :2]
    cmd_norm = torch.norm(cmd_xy, dim=1)
    direction = cmd_xy / cmd_norm.clamp(min=1e-6).unsqueeze(1)
    toward = (vel_b[:, :2] * direction).sum(dim=1)

    onset_steps = max(1, int(onset_s / env.step_dt))
    in_window = bufs["since_change"] < onset_steps
    gate = (cmd_norm > min_cmd) & in_window & (bufs["since_change"] >= 0)
    return (toward / v_ref).clamp(0.0, 1.0) * gate.float()


def onset_action_rate_l2(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
    stand_still_scale: float = 3.0,
    onset_s: float = 0.8,
    onset_scale: float = 0.4,
    change_tol: float = 0.4,
) -> torch.Tensor:
    """``action_rate_l2`` の、**立ち上がり区間だけ倍率を下げる**版 (weight < 0 で使う)。

    加速は本質的に「アクションを速く動かすこと」なので、平滑性ペナルティと正面衝突する。
    07-28 は ``action_rate_l2 = -0.4`` / ``action_smoothness_l2 = -0.12`` で、定常歩容には
    適切だが立ち上がりを鈍らせる。過渡の ``onset_s`` 秒だけ ``onset_scale`` 倍に緩める。

    ★ 実機で動きがガタつくようなら **最初にここを 1.0 (= 緩和なし) に戻す**こと。
      定常区間の倍率は 07-28 と同一なので、緩和は過渡にしか効かない。
    """
    from ...locomotion.mdp.rewards import _stand_still_boost
    from .events import update_lateral_buffers

    bufs = update_lateral_buffers(env, command_name=command_name, change_tol=change_tol)

    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    penalty = torch.sum(torch.square(a - a_prev), dim=1)
    if stand_still_scale != 1.0:
        penalty = penalty * _stand_still_boost(
            env, command_name, cmd_threshold, 0.2, 0.2, stand_still_scale
        )

    onset_steps = max(1, int(onset_s / env.step_dt))
    in_window = (bufs["since_change"] >= 0) & (bufs["since_change"] < onset_steps)
    scale = torch.where(in_window, torch.full_like(penalty, onset_scale), torch.ones_like(penalty))
    return penalty * scale


def heading_hold(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    max_err: float = 0.6,
    change_tol: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """指令角速度を積分した **基準ヘディングからのズレ** のペナルティ (weight < 0 で使う)。

    なぜ要るか:
        ``track_ang_vel_z_exp`` が見ているのは **角速度** であって向きそのものではない。
        わずかな定常角速度誤差は積分されて向きのドリフトになり、角速度追従の報酬では
        原理的に止められない (07-28 実測: 横移動中に約 10°/s、円を描く)。
        重みを上げる対処は 2026-07-29 に試して失敗している
        (7.0 で yaw は改善したが横速度 1.182 → 0.628、外股 3 倍)。
        本項は「溜まったズレ」だけを罰するので、旋回性能そのものは削らない。

    基準ヘディングは :func:`~.events.update_lateral_buffers` が
    ``ref_yaw += wz_cmd * dt`` で積分し、線速度コマンドの変化時とリセット時に
    実ヨーへ取り直す。したがって wz=0 の指令では「向きを保て」、wz≠0 の指令では
    「指令レートで回れ (溜め込むな)」を同時に意味する。

    Returns:
        ``|wrap(yaw - ref_yaw)|`` を ``max_err`` [rad] 付近で飽和させた値 [0, max_err)。

        飽和に ``clamp`` ではなく ``tanh`` を使うのは、**飽和域で勾配を残すため**。
        10°/s のドリフトは再サンプル周期 (最大 4s) で 0.7rad に達し、上限 0.6 を
        普通に超える。clamp だと一番直したい領域で勾配がゼロになる。
    """
    from .events import update_lateral_buffers

    bufs = update_lateral_buffers(env, command_name=command_name, change_tol=change_tol)
    asset: Articulation = env.scene[asset_cfg.name]
    err = wrap_to_pi(asset.data.heading_w - bufs["ref_yaw"]).abs()
    return max_err * torch.tanh(err / max_err)


def track_lin_vel_x_exp(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """前後方向 (base yaw frame の x) の速度コマンド追従だけを見るガウス報酬 [0, 1]。

    :func:`track_lin_vel_y_exp` の前後版。07-28 は横移動中に約 -0.10 m/s の後退
    ドリフトが出る。合算型の ``track_lin_vel_xy_exp`` では、横で稼げていると
    前後の小さなバイアスに勾配がほとんど残らないため、専用項で圧をかける。
    """
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vel_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    err = torch.square(cmd[:, 0] - vel_b[:, 0])
    return torch.exp(-err / std**2)


def foot_clearance_relative(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    target_lift: float = 0.07,
    phase_freq: float = 1.6,
    stance_ratio: float = 0.5,
    cmd_threshold: float = 0.05,
    sigma: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """遊脚の **支持脚に対する相対高さ** を目標値に追従させる報酬 [0, 1]。

    locomotion の :func:`~...locomotion.mdp.rewards.foot_clearance_ji` を 2 点直した版。

    1. **測り方**: あちらは足リンクの *ワールド z (絶対高さ)* を見るため、
       「遊脚を股関節・膝で上げる」以外に「体ごと持ち上げる (跳ぶ)」でも達成できる。
       実際 2026-07-27〜29 に目標を 6cm→7cm に上げた実験では跳躍に退行し、
       跳躍を封じると足上げが 2.6〜3.3cm まで落ちた。
       本項は ``遊脚 z − 支持脚 z`` を見るので、両足が同時に上がる跳躍では値が増えず、
       **抜け道が原理的に塞がる**。支持脚の足リンクは接地しているので、実質
       「地面からの高さ」を測っているのと同じ (地形は平面固定)。
       跳躍を間接的に抑えるための ``lin_vel_z_l2`` の強い重み (-2.5) を緩められる
       ぶん、重心の上下動が要る加速にも効く。

    2. **位相**: あちらは ``phase_freq`` を固定値で使うが、``feet_phase`` /
       ``phase_obs`` は ``get_phase_freq`` 経由で ``randomize_phase_freq``
       (±0.05Hz, env ごと固定) に追従する。エピソード 20s では δ=0.05 の個体が
       t=10s で **逆位相** になり、報酬が接地脚の高さを要求する時間帯が生まれる
       (そしてそれを満たす唯一の手段が跳躍)。本項は他の位相項と同じ周波数を使う。

    ``target_lift`` は **持ち上げ量** [m] (絶対高さではない)。接地時の足リンク原点は
    地面から 0.035m にあるので、絶対高さ表記に直すと ``0.035 + target_lift``。
    """
    from ...locomotion.mdp.events import get_phase_freq

    asset: Articulation = env.scene[asset_cfg.name]
    left_idx = asset.find_bodies("left_foot_link")[0][0]
    right_idx = asset.find_bodies("right_foot_link")[0][0]
    z_left = asset.data.body_pos_w[:, left_idx, 2]
    z_right = asset.data.body_pos_w[:, right_idx, 2]

    # feet_phase と同一の desired-stance 判定 (位相周波数も同じソースから取る)
    t = env.episode_length_buf * env.step_dt
    pf = get_phase_freq(env, phase_freq)
    phase_left = (2.0 * math.pi * pf * t) % (2.0 * math.pi)
    phase_right = (phase_left + math.pi) % (2.0 * math.pi)
    stance_threshold = 2.0 * math.pi * stance_ratio
    stance_left = phase_left < stance_threshold
    stance_right = phase_right < stance_threshold

    cmd_speed = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    is_stopped = cmd_speed < cmd_threshold
    stance_left = stance_left | is_stopped
    stance_right = stance_right | is_stopped

    # 支持脚基準の持ち上げ量 (負にもなりうる)
    lift_left = z_left - z_right
    lift_right = z_right - z_left
    rew_left = torch.exp(-torch.square(target_lift - lift_left) / sigma**2)
    rew_right = torch.exp(-torch.square(target_lift - lift_right) / sigma**2)

    swing_left = (~stance_left).float()
    swing_right = (~stance_right).float()
    return rew_left * swing_left + rew_right * swing_right


def onset_reach_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    reach_frac: float = 0.9,
    min_cmd: float = 0.6,
    onset_s: float = 0.8,
    change_tol: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """指令速度の ``reach_frac`` 倍に**初めて到達した瞬間**、早いほど大きい一回限りの報酬。

    なぜ線形の :func:`onset_speed_bonus` では足りなかったか (2026-08-15 の実測):
        1 本目 (11200 iter) の t90 は指令 1.3 で 0.719s と、07-28 の 0.565s より遅かった。
        立ち上がりの内訳を見ると t50 = 0.28s (ほぼ 1 歩) は速く、**残り 40% に丸 1 歩ぶん
        余計にかかる**。線形報酬は「速度が上がるほど得」なので序盤 (誤差が大きい領域) に
        勾配が集中し、定常の 90% 付近では傾きがほぼ無い。つまり **一番遅い「最後の詰め」に
        圧がかかっていなかった**。

    本項は t90 そのものを目的関数にする:
        到達時刻 t で ``1 - t / onset_s`` を一回だけ払う (早いほど大)。窓を過ぎたら 0。
        1 コマンドにつき 1 回だけなので「到達後に速度を維持する」動機は既存の追従報酬に任せ、
        こちらは純粋に **到達までの時間** だけを最適化する。

    ★ 歩行位相は 1.6Hz 固定 (ユーザー指定の制約) なので、1 歩 = 0.31s の量子化から逃げられない。
      本項で縮められるのは「1 歩あたりの押しの強さ」であって歩数そのものではない。
      0.4s 台に届かない場合、残る手は位相周波数を上げることだけになる。
    """
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    from .events import update_lateral_buffers

    bufs = update_lateral_buffers(env, command_name=command_name, change_tol=change_tol)

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vel_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])

    cmd_xy = cmd[:, :2]
    cmd_norm = torch.norm(cmd_xy, dim=1)
    direction = cmd_xy / cmd_norm.clamp(min=1e-6).unsqueeze(1)
    toward = (vel_b[:, :2] * direction).sum(dim=1)

    onset_steps = max(1, int(onset_s / env.step_dt))
    elapsed = bufs["since_change"]
    in_window = (elapsed >= 0) & (elapsed < onset_steps)
    newly = (
        in_window
        & (~bufs["reach_paid"])
        & (cmd_norm > min_cmd)
        & (toward >= reach_frac * cmd_norm)
    )
    bufs["reach_paid"] |= newly

    speed_ratio = 1.0 - elapsed.float() / float(onset_steps)
    return newly.float() * speed_ratio.clamp(min=0.0)


def _sole_min_z(asset: Articulation, body_idx: int, corners: torch.Tensor) -> torch.Tensor:
    """足裏 4 隅のうち **最も低い点** のワールド z を返す (N,)。"""
    from isaaclab.utils.math import quat_apply

    n = asset.data.body_pos_w.shape[0]
    pos = asset.data.body_pos_w[:, body_idx, :]           # (N, 3)
    quat = asset.data.body_quat_w[:, body_idx, :]         # (N, 4) wxyz
    q = quat.unsqueeze(1).expand(n, 4, 4).reshape(-1, 4)
    c = corners.unsqueeze(0).expand(n, 4, 3).reshape(-1, 3)
    world_z = (quat_apply(q, c).reshape(n, 4, 3) + pos.unsqueeze(1))[:, :, 2]
    return world_z.min(dim=1).values


def foot_clearance_sole(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    target_clearance: float = 0.03,
    phase_freq: float = 1.6,
    stance_ratio: float = 0.5,
    cmd_threshold: float = 0.05,
    sigma: float = 0.02,
    foot_box: tuple[float, float, float, float] = (0.1195, -0.0659, 0.040, -0.0382),
    speed_gate_frac: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """遊脚の **足裏最下点** のクリアランスを目標に追従させる報酬 [0, 1]。

    :func:`foot_clearance_relative` (足リンク原点で測る版) の後継。2026-08-15 の実測で、
    **上げた高さの 40〜70% が足の傾きで失われている** ことが分かったのが理由:

        | | 原点 lift | 足裏最下点 p50 | p05 |
        |---|---|---|---|
        | 07-28 @cmd1.3 | 3.2〜3.9cm | 1.9〜2.3cm | **0.7cm** |
        | 2 本目 @cmd1.3 | 6.6〜7.3cm | 3.4〜4.7cm | 1.6〜2.2cm |

    足長は 0.185m あるので、遊脚が 12° 底屈しているだけでつま先は 2.5cm 下がる。
    つまずくのはつま先なので、**原点の高さを目標にするのは指標として間違っていた**。

    測定点を足裏にすると、ポリシーは同じ報酬を 2 通りの手段で達成できる:

        1. 脚を高く上げる … 長く浮く必要があり、刻めなくなって横速度が落ちる (高コスト)
        2. 遊脚の足首を水平に保つ … 空中の足首は地面反力に関与しないので速度に無影響 (低コスト)

    最適化は安い方を選ぶので、**「控えめな足上げ + 水平な足」** に収束することを狙う。
    2 本目は原点基準 6.6〜7.3cm を要求した結果、横速度が 1.278 → 0.710 m/s と半減した。
    本項の目標 3cm は 07-28 の脚上げ量 (3.2〜3.9cm) より低いので、速度を削る圧にならない。

    跳躍の抜け道は :func:`foot_clearance_relative` と同じく **支持脚基準** (遊脚の足裏 −
    支持脚の足裏) で塞ぐ。位相も同じく ``get_phase_freq`` 経由で DR に追従する。

    Args:
        target_clearance: 足裏最下点の目標クリアランス [m]。人工芝のパイル 20〜30mm を
            上回る 0.03 を既定にしている。
        speed_gate_frac: 指令速度のこの割合に達していれば満額。**「遅く歩いて足を上げる」
            解を報酬上ありえなくするためのゲート**で、追従率に比例して報酬を減らす。
            2 本目はこのゲートが無く、速度を半分捨てて足上げ報酬を取り切っていた。
        foot_box: 足裏 4 隅のオフセット (toe_x, heel_x, half_y, sole_z) [m]。
            既定値は K1_locomotion.urdf の Left_Foot.STL バウンディングボックス実測。
    """
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    from ...locomotion.mdp.events import get_phase_freq

    asset: Articulation = env.scene[asset_cfg.name]
    left_idx = asset.find_bodies("left_foot_link")[0][0]
    right_idx = asset.find_bodies("right_foot_link")[0][0]

    toe_x, heel_x, half_y, sole_z = [float(v) for v in foot_box]
    corners = torch.tensor(
        [[toe_x, half_y, sole_z], [toe_x, -half_y, sole_z],
         [heel_x, half_y, sole_z], [heel_x, -half_y, sole_z]],
        device=env.device,
    )
    z_left = _sole_min_z(asset, left_idx, corners)
    z_right = _sole_min_z(asset, right_idx, corners)

    # feet_phase と同一の desired-stance 判定 (位相周波数も同じソースから取る)
    t = env.episode_length_buf * env.step_dt
    pf = get_phase_freq(env, phase_freq)
    phase_left = (2.0 * math.pi * pf * t) % (2.0 * math.pi)
    phase_right = (phase_left + math.pi) % (2.0 * math.pi)
    stance_threshold = 2.0 * math.pi * stance_ratio
    stance_left = phase_left < stance_threshold
    stance_right = phase_right < stance_threshold

    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :3], dim=1)
    is_stopped = cmd_speed < cmd_threshold
    stance_left = stance_left | is_stopped
    stance_right = stance_right | is_stopped

    # 支持脚の足裏を基準にした相対クリアランス (跳躍で稼げない)
    clr_left = z_left - z_right
    clr_right = z_right - z_left
    rew_left = torch.exp(-torch.square(target_clearance - clr_left) / sigma**2)
    rew_right = torch.exp(-torch.square(target_clearance - clr_right) / sigma**2)

    # --- 速度ゲート: 指令に追従できている割合だけ支払う ---
    vel_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    cmd_xy = cmd[:, :2]
    cmd_norm = torch.norm(cmd_xy, dim=1)
    direction = cmd_xy / cmd_norm.clamp(min=1e-6).unsqueeze(1)
    toward = (vel_b[:, :2] * direction).sum(dim=1)
    gate = (toward / (speed_gate_frac * cmd_norm).clamp(min=1e-3)).clamp(0.0, 1.0)
    gate = torch.where(cmd_norm < cmd_threshold, torch.ones_like(gate), gate)

    swing_left = (~stance_left).float()
    swing_right = (~stance_right).float()
    return (rew_left * swing_left + rew_right * swing_right) * gate
