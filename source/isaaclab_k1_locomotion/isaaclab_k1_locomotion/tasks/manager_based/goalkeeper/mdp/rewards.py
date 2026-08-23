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

    ★ 2026-08-17: 区間を :func:`~.observations.post_save_hold` (タッチ〜次の球、
      約3.0秒) から :func:`~.observations.is_idle_hold` へ広げた。**脅威が無く
      定位置の近くに居る待機全般**が対象になる。

      理由: 待機中の姿勢が報酬でほとんど規定されておらず、MuJoCo で小刻みに
      震える状態が学習中に一度も罰されていなかった (静止ブーストが開いていた
      割合は実測 1.1%)。指令ゼロ化と対にして、待機を「既定姿勢で立つ」1 つの
      状態に固定する。

    保持区間では :func:`~.observations.task_drive_vector` が指令をゼロにするので
    歩容は停止し、この報酬が関節を既定姿勢へ引き戻す。
    """
    from .observations import is_idle_hold

    robot: Articulation = env.scene[asset_cfg.name]
    dev = robot.data.joint_pos - robot.data.default_joint_pos
    # 関節数で正規化して std の意味を関節あたりの平均ずれ [rad] に揃える
    err2 = torch.square(dev).mean(dim=1)
    return torch.exp(-err2 / std**2) * is_idle_hold(env).float()


def _idle_boost(
    env: "ManagerBasedRLEnv",
    scale: float,
    lin_vel_max: float = 0.5,
) -> torch.Tensor:
    """待機保持中に action ペナルティへ掛ける倍率 (N,)。

    ★ 2026-08-17 追加。locomotion の :func:`~...locomotion.mdp.rewards._stand_still_boost`
      を GK 用に置き換えたもの。**判定から「ベースが実際に静止しているか」を外した**
      のが違いで、これが本質。

      旧ゲートは ``|lin_vel| < 0.2 かつ |ang_vel_z| < 0.2`` を要求していた。ところが
      抑えたい対象そのもの (待機中の震え) がこの条件を満たさない。実測 (39500,
      静止ボール条件) では待機中の ``|ang_vel_z|`` が **平均 0.955 rad/s** あり、
      ``ang_vel < 0.2`` の成立率はわずか **7.9%** だった。
      → **震えているという理由で、震えを抑える罰が無効化される**自己矛盾。
      weight や scale をいくら上げても効かなかったのはこれが理由。

      待機保持 (:func:`~.observations.is_idle_hold`) は「指令が厳密ゼロ」を意味する
      ので、その区間では体が動いていること自体が抑制対象であり、静止判定は不要。

    ``lin_vel_max`` だけは残してある。push イベントで大きく突き飛ばされた直後まで
    倍率を掛けると復帰動作を罰してしまうため。閾値は旧 0.2 より緩い 0.5 で、
    震え (実測 0.024 m/s) は確実に対象内、押された直後は対象外になる。
    """
    from .observations import is_idle_hold

    robot: Articulation = env.scene["robot"]
    lin = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
    boost = is_idle_hold(env) & (lin < lin_vel_max)
    return torch.where(boost, torch.full_like(lin, scale), torch.ones_like(lin))


def gk_action_smoothness_l2(
    env: "ManagerBasedRLEnv",
    stand_still_scale: float = 5.0,
    lin_vel_max: float = 0.5,
) -> torch.Tensor:
    """locomotion の ``action_smoothness_l2`` の GK 版 (待機ゲートを差し替え)。

    ペナルティの式は locomotion 版と完全に同じ。違いは倍率のゲートだけで、
    :func:`_idle_boost` を使う (理由は同関数の docstring)。
    """
    from ...locomotion.mdp.rewards import action_smoothness_l2

    base = action_smoothness_l2(env, stand_still_scale=1.0)
    return base * _idle_boost(env, stand_still_scale, lin_vel_max)


def gk_action_rate_l2(
    env: "ManagerBasedRLEnv",
    stand_still_scale: float = 5.0,
    lin_vel_max: float = 0.5,
) -> torch.Tensor:
    """locomotion の ``action_rate_l2`` の GK 版 (待機ゲートを差し替え)。"""
    from ...locomotion.mdp.rewards import action_rate_l2

    base = action_rate_l2(env, stand_still_scale=1.0)
    return base * _idle_boost(env, stand_still_scale, lin_vel_max)


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


def swing_ground_exposure(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    h_target: float = 0.010,
    edge_frac: float = 0.3,
    cmd_threshold: float = 0.05,
    move_threshold: float = 0.002,
    force_threshold: float = 1.0,
    foot_box: tuple[float, float, float, float] = (0.1195, -0.0659, 0.040, -0.0382),
    sensor_name: str = "contact_forces",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """遊脚が **地面すれすれのまま水平に流れた割合** [0, 1]。ペナルティ (負の重み) 用。

    2026-08-17 の実測 (:file:`scripts/rsl_rl/eval_gk_trip_margin.py`) に基づく設計。
    転倒の実因解析では「シム平地では足上げ不足は転倒原因ではない」と出たが、平地には
    引っかかる対象が無いので当然で、足上げの価値は **凹凸への耐性** として測る必要がある。

    そこで **露出率 f(h)** を定義した::

        f(h) = (遊脚の水平移動のうちクリアランスが h 未満だった距離) / (遊脚の総水平移動)
             = 高さ h の突起が通り道にあるとき足が当たる確率

    実測 (cmd 1.3、足裏最下点・接地 p10 基準):

        | h | 07-28 | 2 本目 (高足上げ) | 07-28 @cmd0.6 (速度対照) |
        |---|---|---|---|
        | 2mm  | 0.055 | 0.055 | 0.031 |
        | 5mm  | 0.161 | 0.079 | 0.128 |
        | 10mm | 0.321 | 0.120 | 0.286 |
        | 20mm | 0.558 | 0.194 | 0.499 |

    足上げは凹凸耐性を 2.0〜2.9 倍改善し、**速度を落としただけでは再現しない**
    (07-28 を 2 本目と同程度の速度まで落としても 1.6〜2.6 倍悪いまま)。

    **なぜ「目標高さの追従」ではなく「低い高さで流れた距離」なのか**:
    07-28 の上がりきった区間の足裏 p50 は既に 13.3mm あるのに f(10mm)=0.32 で、
    スイング水平移動の 1/3 が 10mm 以下にある。つまり **頂点は足りていて、上がり際と
    下り際で低いまま長く流れている** のが問題。本項は「同じ高さまで速く上げて遅く下ろせ」
    としか要求しないので、:func:`foot_clearance_sole` のように名目スイング窓いっぱい
    浮くことを強制しない。2 本目が横速度を 1.278 → 0.710 m/s と半減させた原因
    (頂点を 6〜7cm に引き上げた = 長く浮く必要が生じた) を構造的に避けるのが狙い。

    **速度を落とす抜け道が無い**: 分母も分子も距離なので、ゆっくり歩いても比は変わらない。
    :func:`foot_clearance_sole` が必要とした ``speed_gate_frac`` のような保険が要らない。

    クリアランスの基準面は **接地している足の足裏** (支持脚基準)。足リンクの接触オフセット
    (実測で足裏 z の接地時中央値が −4mm 程度) と地形高さの両方を自動的に相殺できる。
    両足とも浮いている (跳躍) 間は基準が取れないので env 原点の z にフォールバックする。
    跳べば露出は下がるが、それは物理的に正しい (跳べば越えられる)。**跳躍そのものの
    抑制は :func:`flight_phase` と ``lin_vel_z_l2`` の担当**で、本項は関与しない。

    Args:
        h_target: この高さ未満を「露出」とみなす [m]。閾値ではなく線形ランプの上端で、
            クリアランス c に対する重みは ``clamp((h_target - c) / h_target, 0, 1)``。
            ``edge_frac`` で決まる上端の幅だけで 0 に落ちる急峻なランプなので、
            実質「h_target 未満か否か」= 指標 f(h) の指示関数とほぼ同じ。
            既定 0.010 は 07-28 の足裏 p50 (13.3mm) のすぐ下 = 頂点を上げずに軌道を
            作り直すだけで届く水準。会場の人工芝が長い場合は 0.015 へ上げる。
        edge_frac: ランプが 0 に落ちる幅を h_target に対する割合で指定する。
            0.3 なら「7mm 以下は満額 1.0、7→10mm で 0」。**1.0 にすると 2026-08-17 に
            失敗した線形ランプに戻るので上げないこと**。
        move_threshold: 遊脚の水平移動がこれ未満のステップは比が不安定なので 0 を返す [m]。

    Returns:
        (N,) の [0, 1]。1 = そのステップの遊脚移動が全部 h_target 未満だった。
    """
    from .events import update_lateral_buffers

    bufs = update_lateral_buffers(env, command_name=command_name)

    asset: Articulation = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_name]
    left_idx = asset.find_bodies("left_foot_link")[0][0]
    right_idx = asset.find_bodies("right_foot_link")[0][0]
    # ☠ 接触センサの body 順は **名前で引き直す**。SceneEntityCfg を params 経由で
    #   渡さない限り body_ids は解決されないうえ、正規表現で引くと左右の順が
    #   アーティキュレーションの body 順に依存し、z_sole の [左, 右] とズレうる。
    sensor_ids = [sensor.body_names.index("left_foot_link"), sensor.body_names.index("right_foot_link")]

    toe_x, heel_x, half_y, sole_z = [float(v) for v in foot_box]
    corners = torch.tensor(
        [[toe_x, half_y, sole_z], [toe_x, -half_y, sole_z],
         [heel_x, half_y, sole_z], [heel_x, -half_y, sole_z]],
        device=env.device,
    )
    z_sole = torch.stack(
        [_sole_min_z(asset, left_idx, corners), _sole_min_z(asset, right_idx, corners)], dim=1
    )                                                                   # (N, 2)

    # 接触は位相ではなく **実際の接地** で判定する (実スイング時間が名目の半分以下という
    # 高デューティ歩容なので、位相ベースだと接地中の足を遊脚扱いしてしまう)。
    forces = sensor.data.net_forces_w[:, sensor_ids, :]                 # (N, 2, 3)
    on_ground = torch.norm(forces, dim=-1) > force_threshold            # (N, 2)
    airborne = ~on_ground

    # 基準面 = 接地している足の足裏 z (両足接地なら低い方)。無ければ env 原点 z。
    big = torch.full_like(z_sole, float("inf"))
    ref = torch.where(on_ground, z_sole, big).min(dim=1).values         # (N,)
    ref = torch.where(torch.isinf(ref), env.scene.env_origins[:, 2], ref)

    clearance = z_sole - ref.unsqueeze(1)                               # (N, 2)
    # ☠ 2026-08-17: ここを **h_target で割る線形ランプ** にしていたのが失敗だった。
    #    線形だと 0mm → 5mm に上げるだけでペナルティが半分になり、10mm を越えるより
    #    はるかに安い。実測 (iter 1000 → 4000、cmd 1.0) はその通りに動いた:
    #        f(2mm)  0.123 → 0.048  (報酬が見ている低い側だけ 2.6 倍改善)
    #        f(10mm) 0.416 → 0.515  (**測りたかった指標は悪化**)
    #    = 「低いところを避けろ」と書いたつもりが「一番低いところだけ少し上げろ」に
    #    なっていた。地面すれすれ数 mm を舐めるように滑る歩容へ収束する。
    #    → **上端 edge_frac だけで 0 に落ちる急峻なランプ**にして、指標 f(h) の
    #    指示関数にほぼ一致させる。これで「5mm に逃げる」に利得が無くなり、
    #    h_target を越える以外にペナルティを下げる手段が無くなる。
    #    (RL の報酬は微分可能である必要が無いので階段関数でも良いが、途中経過に
    #     まったく credit が出ないと学習が難しいので上端 30% だけ勾配を残す)
    edge = max(h_target * edge_frac, 1e-6)
    ramp = torch.clamp((h_target - clearance) / edge, 0.0, 1.0)

    ds = bufs["foot_ds"] * airborne.float()                             # (N, 2)
    denom = ds.sum(dim=1)
    exposure = (ds * ramp).sum(dim=1) / denom.clamp(min=1e-6)

    # 移動が小さいステップは比が暴れるので無効化する
    exposure = torch.where(denom > move_threshold, exposure, torch.zeros_like(exposure))
    # 停止指令のときは足を上げる必要がない
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(cmd[:, :3], dim=1)
    return torch.where(cmd_norm < cmd_threshold, torch.zeros_like(exposure), exposure)


# ---------------------------------------------------------------------------
# 振動対策 (2026-08-21)
# ---------------------------------------------------------------------------
#
# ★ 実機で 07-28 が振動する件の対策。**停止指令時** に diag_walk_jitter.py で測ると:
#
#       ||Δaction||  mean 0.0332      ← 1階差分 (動きの大きさ)
#       2階差分       mean 0.0404      ← 高周波成分
#       比            1.217
#
#   正弦波なら 2階差分/1階差分 = 2·sin(πf/fs) なので、fs=50Hz で逆算すると **f ≈ 10.4 Hz**。
#   きれいな 1.6Hz の歩容なら比は 0.20 なので、6 倍の高周波成分が乗っている。
#   停止指令なのに ``ベース水平速度 max = 0.684 m/s`` も出ており、待機中に暴れている。
#
# ☠ ``action_smoothness_l2`` の weight を 100 倍 (-0.12 → -12.0) にすると振動は消えたが、
#   **横速度が 1/3 になった** (2026-08-20 実機デプロイ)。あの項は 1階差分と2階差分の
#   **和**、つまり **振幅** を罰するので、高周波と一緒に歩行そのものを潰す。
#   → 狙うべきは振幅ではなく **周波数**。以下の 2 つはその 2 通りの実装。

_JITTER_HIST_ATTR = "_gk_jitter_hist"        # (2, N, A) [a_{t-1}, a_{t-2}]
_JITTER_STEP_ATTR = "_gk_jitter_step"
_JITTER_VAL_ATTR = "_gk_jitter_val"          # (d1, d2) のキャッシュ


def _action_diffs(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """``(||a-a₋₁||, ||a-2a₋₁+a₋₂||)`` を返す (どちらも (N,))。

    ☠ 履歴の更新は **1 ステップにつき 1 回だけ**。locomotion の ``action_smoothness_l2``
      が使う ``env._prev_prev_action`` とは **別のバッファ**にしてある。あちらは呼ばれる
      たびに書き換える実装なので、同じステップで両方を使うと a₋₂ が壊れる。

    ☠ 指標は ``scripts/rsl_rl/diag_walk_jitter.py`` と同じ **L2 ノルム**で揃えてある
      (あちらは二乗和ではなくノルムを出力する)。報酬と評価指標の形を一致させること。
    """
    a = env.action_manager.action
    cur = int(getattr(env, "common_step_counter", 0))
    if getattr(env, _JITTER_STEP_ATTR, -1) == cur:
        return getattr(env, _JITTER_VAL_ATTR)

    hist: torch.Tensor | None = getattr(env, _JITTER_HIST_ATTR, None)
    if hist is None or hist.shape[1:] != a.shape:
        hist = a.detach().unsqueeze(0).repeat(2, 1, 1)
        setattr(env, _JITTER_HIST_ATTR, hist)

    d1 = torch.norm(a - hist[0], dim=1)
    d2 = torch.norm(a - 2.0 * hist[0] + hist[1], dim=1)
    hist[1] = hist[0]
    hist[0] = a.detach()
    setattr(env, _JITTER_STEP_ATTR, cur)
    setattr(env, _JITTER_VAL_ATTR, (d1, d2))
    return d1, d2


def _stopped_boost(
    env: "ManagerBasedRLEnv",
    scale: float,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
    lin_vel_max: float = 0.5,
) -> torch.Tensor:
    """**停止指令中** だけペナルティ倍率を ``scale`` にするゲート (N,)。

    ☠☠ locomotion の :func:`~...locomotion.mdp.rewards._stand_still_boost` は
      「停止指令 **かつ** base が実際に静止 (|lin_vel|<0.2 かつ |ang_vel_z|<0.2)」を
      要求する。ところが **抑えたい対象そのもの (待機中の震え) がこの条件を満たさない**。
      GK Stage2 側の実測では待機中の |ang_vel_z| が平均 0.955 rad/s あり、
      ``ang_vel < 0.2`` の成立率はわずか **7.9%** だった。
      → **震えているという理由で、震えを抑える罰が無効化される**自己矛盾。
      weight や scale をいくら上げても効かなかったのはこれが理由。
      本関数は **実速度の条件を外し**、指令ノルムだけで判定する。

    ``lin_vel_max`` だけは残す。push イベントで突き飛ばされた直後まで倍率を掛けると
    復帰動作を罰してしまうため。閾値 0.5 は震え (実測 0.02 m/s 級) を確実に含み、
    押された直後は外れる値。
    """
    cmd = env.command_manager.get_command(command_name)[:, :3]
    is_stopped = torch.norm(cmd, dim=1) < float(cmd_threshold)
    robot: Articulation = env.scene["robot"]
    lin = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
    boost = is_stopped & (lin < float(lin_vel_max))
    return torch.where(boost, torch.full_like(lin, float(scale)), torch.ones_like(lin))


def lateral_action_smoothness_l2(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
    stand_still_scale: float = 8.0,
    lin_vel_max: float = 0.5,
) -> torch.Tensor:
    """locomotion の ``action_smoothness_l2`` の横移動版 (待機ゲートを差し替え)。

    ペナルティの式は locomotion 版と完全に同じ。違いはゲートだけで、
    :func:`_stopped_boost` を使う (理由は同関数の docstring)。
    ★ **weight は 07-28 のまま (-0.12) にすること**。移動中にも効く weight を上げるのは
      振幅を罰する道で、速度を 1/3 にした失敗の再現になる。調整は stand_still_scale で。
    """
    from ...locomotion.mdp.rewards import action_smoothness_l2

    base = action_smoothness_l2(env, stand_still_scale=1.0)
    return base * _stopped_boost(env, stand_still_scale, command_name, cmd_threshold, lin_vel_max)


def lateral_action_rate_l2(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
    stand_still_scale: float = 8.0,
    lin_vel_max: float = 0.5,
) -> torch.Tensor:
    """locomotion の ``action_rate_l2`` の横移動版 (待機ゲートを差し替え)。"""
    from ...locomotion.mdp.rewards import action_rate_l2

    base = action_rate_l2(env, stand_still_scale=1.0)
    return base * _stopped_boost(env, stand_still_scale, command_name, cmd_threshold, lin_vel_max)


def action_jitter_ratio(
    env: "ManagerBasedRLEnv",
    move_threshold: float = 0.005,
    eps: float = 1e-3,
) -> torch.Tensor:
    """**動きあたりのガタつき** = ``||a-2a₋₁+a₋₂|| / (||a-a₋₁|| + eps)`` (N,)。

    ☆ 分子・分母が同じ次元なので **振幅が約分され、周波数だけが残る**。
      正弦波なら値は ``2·sin(πf/fs)`` に等しく、fs=50Hz では:

          1.6Hz (歩行の基本波) → 0.20
          10Hz  (実機の振動)   → 1.17
          25Hz  (Nyquist)      → 2.00

      **大きく速く動くことにはコストがかからず、ガタつきだけにコストがかかる。**
      ``action_smoothness_l2`` の weight を上げる道は振幅を罰するので歩行そのものを
      潰した (速度 1/3)。本項にはその代償が原理的に無い。

    ☠ 動きが小さいと 0/0 で暴れるので、``||Δa|| < move_threshold`` のステップは 0 を返す。
      **待機中はこのゲートで落ちる**ので、停止時の振動は本項ではなく
      :func:`lateral_action_smoothness_l2` の停止時ゲートが担当する (両方要る)。

    ☠ 指標は ``diag_walk_jitter.py`` と同じ L2 ノルム比。**報酬と評価指標の形が一致**
      しているので、学習後にそのまま同じ数字で合否が判定できる。
    """
    d1, d2 = _action_diffs(env)
    ratio = d2 / (d1 + float(eps))
    return torch.where(d1 > float(move_threshold), ratio, torch.zeros_like(ratio))


def standstill_jitter(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
    d2_ref: float = 0.05,
    lin_vel_max: float = 0.5,
) -> torch.Tensor:
    """**停止指令中の高周波の振れ幅**を有界に罰する (N,)。weight < 0 で使う。

    ``d2 / (d2 + d2_ref)`` を返す。値域は ``[0, 1)`` で有界。

    ☠☠ **倍率 (stand_still_scale) で実現してはいけない**、というのが 6 本目
      (2026-08-21) の失敗から得た教訓。停止 env だけ 100 倍の圧を掛けたところ、
      **PPO が壊れた** (iter 500 をピークに mean_reward 69 → 35、転倒増、
      カリキュラムは stage 0 のまま)。同一バッチ内で報酬スケールが env 間で
      100 倍違うと、少数の巨大な負の advantage が更新を支配し価値関数も壊れる。
      1 本目が ``action_smoothness_l2 = -12.0`` で学習できたのは **全 env 一様**
      だったから。→ 圧は倍率でなく **有界な独立項** で与える。

    なぜこの量か (2026-08-21 の実測、diag_walk_jitter.py の停止指令):

        ポリシー        ‖Δa‖     2階差分   実機
        1 本目          0.0089   **0.0055**  **振動しない**
        3 本目 m4999    0.0133   0.0124    ?
        07-28           0.0332   **0.0407**  **振動する**
        4 本目          0.0394   0.0417    (07-28 と同値)

      比 (周波数) の差は 2 倍だが **2階差分の絶対値は 7.4 倍**。振動は物理的な
      振れ幅なので、周波数より **振幅** が効く。目標は **≤ 0.01**。

    ☆ 移動中はゲートで 0 になるので **速度コストは構造的にゼロ**。
      移動中の高周波は :func:`action_jitter_ratio` (振幅非依存の比) が担当する。
      1 本目は停止時・移動中の **両方** が滑らかだったので、両方要る。

    Args:
        d2_ref: 圧縮の基準 [action 単位]。0.05 なら 07-28 (0.0407) で 0.45、
            目標 (0.0055) で 0.10 と、**運転範囲の全域で勾配が残る**。
            ☠ tanh でなく ``x/(x+ref)`` を使うのは、学習初期に d2 が大きい
            (0.5 級) ときに tanh だと完全に飽和して勾配が消えるため。
    """
    _, d2 = _action_diffs(env)
    pen = d2 / (d2 + float(d2_ref))
    return pen * _stop_mask(env, command_name, cmd_threshold, lin_vel_max)


def _stop_mask(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
    lin_vel_max: float = 0.5,
) -> torch.Tensor:
    """停止指令中なら 1.0、それ以外 0.0 (N,)。:func:`_stopped_boost` と同じ判定。"""
    cmd = env.command_manager.get_command(command_name)[:, :3]
    is_stopped = torch.norm(cmd, dim=1) < float(cmd_threshold)
    robot: Articulation = env.scene["robot"]
    lin = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
    return (is_stopped & (lin < float(lin_vel_max))).float()


_BODY_JITTER_HIST_ATTR = "_gk_body_jitter_hist"    # (2, N, 3) [w_{t-1}, w_{t-2}]
_BODY_JITTER_STEP_ATTR = "_gk_body_jitter_step"
_BODY_JITTER_VAL_ATTR = "_gk_body_jitter_val"


def body_jitter(
    env: "ManagerBasedRLEnv",
    w_ref: float = 0.15,
    stop_only: bool = False,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
    lin_vel_max: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """**胴体角速度の2階差分**を有界に罰する (N,)。weight < 0 で使う。

    ``d2w / (d2w + w_ref)`` を返す。値域 ``[0, 1)`` で有界 (倍率は PPO を壊すので使わない。
    :func:`standstill_jitter` の docstring 参照)。

    ☆☆ なぜ指令ではなく **機体側** を見るか (2026-08-22 の実測):

        指令 (action) の比を罰する :func:`action_jitter_ratio` は **移動中に効いていない**。
        速度が 1.5 倍違う 3 本を測っても比は横並びだった:
            15-48-46 (0.90 m/s) 0.838 / min_height (0.96) 0.835 / 08-41-39 (1.32) 0.841
        移動中の高周波は歩容そのものの性質で決まっていて、指令の比では動かせない。

        一方 **胴体角速度の2階差分は実機の振動と 9.4 倍の分離**を持つ:
            0.0207  -12.0版 (実機で振動しない)
            0.0820  08-41-39
            0.1956  07-28   (実機で振動する)
        指令ベース (7.7 倍) より分離が良く、しかも物理量なので実機との対応が直接的。

    ☆ シムでロボットが実際に震えていることは確認済み。PD と armature が指令の高周波を
      一部吸収する (吸収率 07-28 0.348 / -12.0版 1.038) が、それでも胴体は 9.4 倍震える。

    Args:
        w_ref: 圧縮の基準 [rad/s]。0.15 なら 07-28 (0.196) で 0.57、
            目標 (0.021) で 0.12 と、**運転範囲の全域で勾配が残る**。
            ☠ 停止時と移動中では絶対値が桁で違う (移動中は 0.57 級)。
        stop_only: True で **停止指令中だけ** 罰する (:func:`_stopped_boost` と同じゲート)。

            ★★ 2026-08-23: **既定は False だが、横移動タスクでは True にすること。**
              実機フィードバックで「動いているときは振動しない / **止まるときに振動する**」
              と確認された。ところがゲート無しだと移動中 (raw 0.57 級) が支配的になり、
              iter 1844 の実測で **-1.245 と報酬セット最大の負項**、しかも raw 0.83 と
              ほぼ飽和していた。**実機で問題になっていない挙動に、報酬セット最大の圧を
              かけている**状態で、移動性能を無駄に削る。
              w_ref=0.15 はそもそも停止時のレンジ (0.02〜0.2) に合わせた値なので、
              停止時に絞る方が設計とも整合する。
        command_name / cmd_threshold / lin_vel_max: ``stop_only=True`` のときのゲート条件。
            :func:`_stopped_boost` を参照 (実速度の条件を外し指令ノルムで判定する理由も同じ)。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    w = robot.data.root_ang_vel_w
    cur = int(getattr(env, "common_step_counter", 0))
    if getattr(env, _BODY_JITTER_STEP_ATTR, -1) == cur:
        d2w = getattr(env, _BODY_JITTER_VAL_ATTR)
    else:
        hist: torch.Tensor | None = getattr(env, _BODY_JITTER_HIST_ATTR, None)
        if hist is None or hist.shape[1:] != w.shape:
            hist = w.detach().unsqueeze(0).repeat(2, 1, 1)
            setattr(env, _BODY_JITTER_HIST_ATTR, hist)
        d2w = torch.norm(w - 2.0 * hist[0] + hist[1], dim=1)
        hist[1] = hist[0]
        hist[0] = w.detach()
        setattr(env, _BODY_JITTER_STEP_ATTR, cur)
        setattr(env, _BODY_JITTER_VAL_ATTR, d2w)
    penalty = d2w / (d2w + float(w_ref))
    if stop_only:
        # _stopped_boost は scale 倍率を返すゲートなので、scale=1.0 で「停止中=1 / それ以外=0」
        # のマスクにはならない。ここは 0/1 のマスクが要るので直接組む。
        cmd = env.command_manager.get_command(command_name)[:, :3]
        is_stopped = torch.norm(cmd, dim=1) < float(cmd_threshold)
        lin = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
        # ☠ push で突き飛ばされた直後は外す (復帰動作を罰しないため)。閾値の根拠は
        #   _stopped_boost の docstring と同じ。
        penalty = penalty * (is_stopped & (lin < float(lin_vel_max))).float()
    return penalty
