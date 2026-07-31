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
    # 左右の並び順が一致している保証が構造的には無い。初回だけ名前で突き合わせて
    # 検証し、不一致なら即座に落とす (黙って左右を取り違えると症状の悪化に気づけない)。
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
    # NOTE: net_forces_w_history は index 0 が最新・末尾が最古。[:, -1] は最古サンプル
    #       なので使わない (feet_slide がそう書いているが、あれは 10ms 古い値を見ている)。
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
    max_y: float = 1.25,
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
    max_y: float = 1.25,
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
    max_y: float = 1.25,
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
    max_y: float = 1.25,
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
    """
    if x_offset is None:
        x_offset = float(env.cfg.goalkeeper.guard_x)
    x = robot_pos_goal(env)[:, 0]
    return torch.exp(-torch.square(x - x_offset) / std**2)


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
