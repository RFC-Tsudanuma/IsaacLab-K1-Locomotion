# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (goalkeeper) タスク専用のイベント関数。

ボールのスポーン・初速・狙い先などの可変パラメータは、EventTerm の params ではなく
env cfg の ``goalkeeper`` フィールド (GoalkeeperParamsCfg) から実行時に読む。
これにより ``--override_json`` の ``{"env": {"goalkeeper.ball_speed_max": 1.5}}``
のようなドットパス上書きで、ステージ遷移条件・初速レンジを設定ファイルから制御できる。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from ...around_ball.mdp.observations import _high_action_cmd
from .observations import ball_pos_goal, gk_buffers, robot_pos_goal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _gk_params(env: "ManagerBasedEnv"):
    """env cfg に取り付けられた GoalkeeperParamsCfg を返す。"""
    return env.cfg.goalkeeper


def reset_gk_buffers(env: "ManagerBasedEnv", env_ids: torch.Tensor):
    """リセットされた env の goalkeeper 状態バッファを初期化する。"""
    bufs = gk_buffers(env)
    bufs["target_y"][env_ids] = 0.0
    bufs["ball_active"][env_ids] = False
    bufs["touched"][env_ids] = False
    bufs["touch_rewarded"][env_ids] = False
    bufs["save_cd"][env_ids] = -1
    bufs["hold_ctr"][env_ids] = 0
    bufs["respawn_cd"][env_ids] = -1
    bufs["save_count"][env_ids] = 0
    bufs["save_quality"][env_ids] = 0.0


def _sample_stage1_targets(env: "ManagerBasedEnv", robot_y: torch.Tensor) -> torch.Tensor:
    """ステージ1 の目標 y をサンプルする (robot_y と同形状)。

    速度学習の圧を保つための分布制御 (パラメータは GoalkeeperParamsCfg):

    * 確率 ``stage1_far_prob``: **反対側のポスト際ゾーン** (|y| ∈ stage1_far_zone) から
      採る。ロボットがポスト際にいるときは逆ポストまでの最大距離 (~2.6m) の
      スプリントになる — 「ゴール幅全域をなるべく速く横移動」のワーストケースを
      確実に学習分布へ入れるための枝。
    * それ以外: ゴール幅内の一様ランダム。ただし現在位置 ±``stage1_min_move`` の
      近距離帯を **除外した区間から直接サンプル** する (棄却ループなし)。
      近すぎる目標は移動なしで達成できてしまい、速度学習に寄与しないため。
    """
    p = _gk_params(env)
    n = robot_y.shape[0]
    device = env.device
    r = float(p.stage1_target_range)
    m = float(p.stage1_min_move)
    y = robot_y.clamp(-r, r)

    # --- 一様ランダム枝: [-r, r] から近距離帯 (y ± m) を除いた区間を直接サンプル ---
    a = (y - m).clamp(min=-r)
    b = (y + m).clamp(max=r)
    left_len = (a + r).clamp(min=0.0)
    right_len = (r - b).clamp(min=0.0)
    total = (left_len + right_len).clamp(min=1e-6)
    u = torch.rand(n, device=device) * total
    uniform_far = torch.where(u < left_len, -r + u, b + (u - left_len))

    # --- ポスト際ゾーン枝: 現在位置の反対側の |y| ∈ far_zone ---
    z_lo, z_hi = float(p.stage1_far_zone[0]), float(p.stage1_far_zone[1])
    side = -torch.sign(y)
    rand_side = torch.sign(torch.rand(n, device=device) - 0.5)
    side = torch.where(y.abs() < 0.1, rand_side, side)          # 中央付近なら左右ランダム
    side = torch.where(side == 0, torch.ones_like(side), side)  # sign(0)=0 の保険
    post_zone = side * (z_lo + torch.rand(n, device=device) * (z_hi - z_lo))

    use_far = torch.rand(n, device=device) < float(p.stage1_far_prob)
    return torch.where(use_far, post_zone, uniform_far)


def reset_stage1_target_and_park(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
):
    """ステージ1: ボールを遠方にパーク (非アクティブ) し、ランダム目標 y を採番する。

    ボール観測は ``_gk_ball_active`` = False によりダミー 0 になるが、物理体としては
    フィールド奥 (park_pos) に静止させておき、観測次元・シーン構成をステージ2 以降と
    完全に一致させる。★ reset_gk_buffers より後に登録すること (目標を上書きするため)。
    目標のサンプリング分布は :func:`_sample_stage1_targets` 参照。
    """
    p = _gk_params(env)
    ball = env.scene[ball_cfg.name]
    bufs = gk_buffers(env)
    n = len(env_ids)

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = env.scene.env_origins[env_ids, 0] + float(p.park_pos[0])
    pose[:, 1] = env.scene.env_origins[env_ids, 1] + float(p.park_pos[1])
    pose[:, 2] = float(p.ball_radius)
    pose[:, 3] = 1.0  # 単位クォータニオン (w, x, y, z)
    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)

    bufs["ball_active"][env_ids] = False
    bufs["target_y"][env_ids] = _sample_stage1_targets(env, robot_pos_goal(env)[env_ids, 1])
    bufs["hold_ctr"][env_ids] = 0


def stage1_target_tick(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None = None,
):
    """ステージ1: 目標に到達して静止を保てた env の目標 y を再サンプルする毎ステップイベント。

    ``EventTerm(mode="interval", interval_range_s=(dt, dt), is_global_time=True)`` で
    毎制御ステップ呼ぶ。到達判定は

        * |robot_y − target_y| < reach_tol
        * 上位コマンドノルム < stage1_cmd_tol (足踏み対策: around_ball_ready の教訓。
          frozen が「初期姿勢で立つ」のはコマンドノルム < 0.05 のときだけなので、
          実ベース速度だけで判定するとコマンドを残した足踏みで達成できてしまう)
        * ベース並進速度 < stage1_speed_tol

    を ``stage1_hold_steps`` 連続で満たすこと。達成した env は新しい目標を採番し、
    カウンタを 0 に戻す (エピソードは切らない)。
    """
    p = _gk_params(env)
    bufs = gk_buffers(env)
    robot = env.scene["robot"]

    robot_y = robot_pos_goal(env)[:, 1]
    y_err = (robot_y - bufs["target_y"]).abs()
    cmd_norm = torch.norm(_high_action_cmd(env)[:, :3], dim=1)
    lin_speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)

    reached = (
        (y_err < float(p.stage1_reach_tol))
        & (cmd_norm < float(p.stage1_cmd_tol))
        & (lin_speed < float(p.stage1_speed_tol))
    )
    # リセット直後の過渡では判定しない
    reached = reached & (env.episode_length_buf >= 2)

    ctr = bufs["hold_ctr"]
    ctr[reached] += 1
    ctr[~reached] = 0

    done = ctr >= int(p.stage1_hold_steps)
    if int(done.sum()) > 0:
        bufs["target_y"][done] = _sample_stage1_targets(env, robot_y[done])
        ctr[done] = 0


def reset_ball_shot(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
):
    """ステージ2/3: ボールをフィールド側にスポーンし、ゴール内の狙い先へ初速を与える。

    ランダム化 (すべて GoalkeeperParamsCfg から):
        * スポーン位置: ゴール中央からの距離 d ∈ spawn_dist_range、
          方位 θ ∈ ±spawn_half_angle (+x 正面基準)。ロボットに近すぎる位置
          (< 0.6m) はロボットから見て radial に押し出して重なりを防ぐ。
        * 狙い先: ゴールライン上の y_aim ∈ ±aim_y_range (ポスト内側)
        * 初速: v ∈ [ball_speed_min, hi]。hi は通常 ball_speed_max だが、
          適応カリキュラム (adaptive_difficulty) が ``_gk_speed_hi``
          バッファを持っている場合はそちらを使う (成功率に応じて連続的に引き上げ)。
        * 実現可能性クランプ: ゴールライン到達までの時間が min_time_to_line [s] を
          下回らないよう初速に上限を掛ける (近距離×高速の「原理的にセーブ不可能な
          球」を作らない。距離と速度に自然な相関がつく: 近い球は遅く、遠い球は速い)。

    ボールには転がり整合の角速度 ω = (-vy, vx, 0)/r も与える (滑り減速の過渡を消す)。
    ★ reset_gk_buffers より後、reset_base (ロボット配置) より後に登録すること。
    """
    p = _gk_params(env)
    ball = env.scene[ball_cfg.name]
    bufs = gk_buffers(env)
    n = len(env_ids)
    r = float(p.ball_radius)

    dist = torch.empty(n, device=env.device).uniform_(*[float(v) for v in p.spawn_dist_range])
    ang = torch.empty(n, device=env.device).uniform_(-float(p.spawn_half_angle), float(p.spawn_half_angle))
    spawn_x = dist * torch.cos(ang)
    spawn_y = dist * torch.sin(ang)

    # ロボットとの重なり防止: 近すぎるスポーンはロボットから radial に 0.6m まで押し出す
    robot_xy = robot_pos_goal(env)[env_ids, :2]
    d_rel_x = spawn_x - robot_xy[:, 0]
    d_rel_y = spawn_y - robot_xy[:, 1]
    d_rel = torch.sqrt(d_rel_x**2 + d_rel_y**2).clamp(min=1e-6)
    too_close = d_rel < 0.6
    scale = torch.where(too_close, 0.6 / d_rel, torch.ones_like(d_rel))
    spawn_x = robot_xy[:, 0] + d_rel_x * scale
    spawn_y = robot_xy[:, 1] + d_rel_y * scale

    # 狙い先 y の範囲。適応カリキュラム (adaptive_difficulty) が段階的に広げるので、
    aim_buf = getattr(env, "_gk_aim_y", None)
    aim_range = float(aim_buf.item()) if aim_buf is not None else float(p.aim_y_range)
    aim_y = torch.empty(n, device=env.device).uniform_(-aim_range, aim_range)

    hi_buf = getattr(env, "_gk_speed_hi", None)
    hi = float(hi_buf.item()) if hi_buf is not None else float(p.ball_speed_max)
    speed = torch.empty(n, device=env.device).uniform_(float(p.ball_speed_min), hi)

    # スポーン点 → 狙い先 (x=0, y=aim_y) の方向単位ベクトル
    dir_x = -spawn_x
    dir_y = aim_y - spawn_y
    norm = torch.sqrt(dir_x**2 + dir_y**2).clamp(min=1e-6)
    # 実現可能性クランプ: 到達時間 = 距離/速度 ≥ min_time_to_line
    v_feasible = norm / max(float(p.min_time_to_line), 1e-3)
    speed = torch.minimum(speed, v_feasible)
    vx = speed * dir_x / norm
    vy = speed * dir_y / norm

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = env.scene.env_origins[env_ids, 0] + spawn_x
    pose[:, 1] = env.scene.env_origins[env_ids, 1] + spawn_y
    pose[:, 2] = r
    pose[:, 3] = 1.0
    ball.write_root_pose_to_sim(pose, env_ids=env_ids)

    vel = torch.zeros(n, 6, device=env.device)
    vel[:, 0] = vx
    vel[:, 1] = vy
    # 転がり整合角速度: v_contact = v + ω × (-r ẑ) = 0 ⇔ ω = (-vy, vx, 0)/r
    vel[:, 3] = -vy / r
    vel[:, 4] = vx / r
    ball.write_root_velocity_to_sim(vel, env_ids=env_ids)

    bufs["ball_active"][env_ids] = True


def reset_ball_perception(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
):
    """VirtualPerception のリングバッファ/per-env パラメータをリセットし、速度バイアスを
    採番する。

    位置・遅延・ノイズ・検出率・occlusion は VirtualPerception (実機カメラ準拠) が担当。
    速度バイアス (0.5〜1.0 m/s のエピソード固定系統誤差) だけは goalkeeper 側で持つので
    ここで採番する。自己位置推定 (実機は MCL) の誤差もここで採番する。
    ★ reset_ball (ボール配置イベント) より後に登録すること。
    """
    import math

    from .observations import _gk_loc_buffers, _gk_perception

    p = _gk_params(env)
    vp = _gk_perception(env)
    vp.reset(env_ids)
    n = len(env_ids)

    # 速度バイアス: x, y 各軸独立。大きさを perc_vel_bias_range から一様サンプルし、
    if bool(getattr(p, "perception_clean", False)):
        env._gkp_vel_bias[env_ids] = 0.0
    else:
        vlo, vhi = float(p.perc_vel_bias_range[0]), float(p.perc_vel_bias_range[1])
        vmag = torch.empty(n, 2, device=env.device).uniform_(vlo, vhi)
        vsign = torch.where(
            torch.rand(n, 2, device=env.device) < 0.5,
            torch.ones(n, 2, device=env.device),
            -torch.ones(n, 2, device=env.device),
        )
        env._gkp_vel_bias[env_ids] = vmag * vsign
    env._gkp_out_vel[env_ids] = 0.0

    # 自己位置推定の誤差: エピソード固定バイアス + ドリフト速度 + 跳びの頻度。
    _gk_loc_buffers(env)
    env._gk_loc_err[env_ids] = 0.0
    if bool(getattr(p, "perception_clean", False)):
        env._gk_loc_bias[env_ids] = 0.0
        env._gk_loc_drift[env_ids] = 0.0
        env._gk_loc_jump_p[env_ids] = 0.0
        return

    deg = math.pi / 180.0
    bias = torch.empty(n, 3, device=env.device).uniform_(-1.0, 1.0)
    bias[:, :2] *= float(p.loc_bias_xy_m)
    bias[:, 2] *= float(p.loc_bias_yaw_deg) * deg
    env._gk_loc_bias[env_ids] = bias

    drift = torch.empty(n, 3, device=env.device).uniform_(-1.0, 1.0)
    drift[:, :2] *= float(p.loc_drift_xy_mps)
    drift[:, 2] *= float(p.loc_drift_yaw_dps) * deg
    env._gk_loc_drift[env_ids] = drift

    jlo, jhi = float(p.loc_jump_hz_range[0]), float(p.loc_jump_hz_range[1])
    env._gk_loc_jump_p[env_ids] = (
        torch.empty(n, device=env.device).uniform_(jlo, jhi) * float(env.step_dt)
    )


def _save_clearance_quality(
    env: "ManagerBasedEnv",
    safe_dist: float = 1.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """セーブ品質 [0, 1]: ボールを「ゴール枠」からどれだけ遠ざけたか (N,)。

    「止めた」と「危険を除去した」を区別するための指標。ゴール正面 0.3m に
    ボールを止めただけでも従来はセーブ成功扱いだったが、実戦ではそのまま
    押し込まれる。ゴール枠 (x=0, |y| <= goal_half_width) の矩形からの
    ユークリッド距離で測り、``safe_dist`` [m] 離れれば満点とする。

        * 前方へ押し出した  → dx が伸びる
        * ポストの外へ弾いた → dy が伸びる (枠外に出た球もここで加点される)

    明示的なキック方向は与えない。「なるべく外へ弾け」という圧力だけを与え、
    体の当て方はポリシーに任せる (キック技術は ball_kick タスクの担当)。
    """
    p = _gk_params(env)
    pos = ball_pos_goal(env, ball_cfg)
    dx = pos[:, 0].clamp(min=0.0)                                       # ライン前方への距離
    dy = (pos[:, 1].abs() - float(p.goal_half_width)).clamp(min=0.0)    # ポスト外へのはみ出し
    dist = torch.sqrt(dx * dx + dy * dy)
    return (dist / max(float(safe_dist), 1e-3)).clamp(0.0, 1.0)


def sync_task_command(
    env: "ManagerBasedEnv",
    # ★ env_ids にデフォルト値を付けないこと (理由は relaunch_ball_after_save 参照)。
    env_ids: torch.Tensor | None,
    command_name: str = "base_velocity",
):
    """``base_velocity`` コマンドをタスク由来の移動要求で上書きする毎ステップイベント。

    ステージ2/3 では観測スロットだけを :func:`~.observations.task_drive_vector` に
    差し替えていたため、コマンドを参照する報酬 (``feet_phase`` / ``foot_clearance``)
    がランダムなコマンドを見て常に「歩け」と要求し、足踏みが止まらなかった。
    コマンド自体を差し替えて観測・報酬・位相を同じ信号で駆動する。

    ★ 併せて cfg 側で ``heading_command`` / ``rel_standing_envs`` /
      ``resampling_time_range`` を無効化すること。
    """
    from .observations import task_drive_vector

    cmd_term = env.command_manager.get_term(command_name)
    buf = getattr(cmd_term, "vel_command_b", None)
    if buf is None:
        return
    # ★ 推定値を使うこと。ここは feet_phase / foot_clearance の停止判定に使われる。
    #   真値にすると「ポリシーが観測できない情報で足上げ報酬が ON/OFF する」状態になり、
    #   ポリシーは報酬が付くかを予測できなくなる。実測で 28% 食い違い、足上げの期待値が
    #   下がって foot_clearance が 0.70 → 0.25 に落ちた。自己位置のバイアスは設計上
    #   エピソード内で一定なので、真値のままだとロボットが正しく定位置へ行くほど
    #   食い違いが恒久化する (平均回帰では解消できない)。このスイッチは実機に存在しない
    #   学習時だけのものなので、推定値にしても sim-to-real のリアルさは損なわれない。
    drive = task_drive_vector(env, use_perceived=True)
    buf[:, :2] = drive[:, :2]
    # ★ 向き成分は 0 にする。feet_phase / foot_clearance の停止判定は
    buf[:, 2] = 0.0


def relaunch_ball_after_save(
    env: "ManagerBasedEnv",
    # ★ env_ids にデフォルト値を付けないこと (理由は下記)。EventManager の引数検証
    env_ids: torch.Tensor | None,
    respawn_delay_steps: int = 50,
    clearance_safe_dist: float = 1.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
):
    """**エピソード継続モード**: セーブが確定した env に、次のボールを撃ち直す。

    ``EventTerm(mode="interval", interval_range_s=(dt, dt), is_global_time=True)`` で
    毎制御ステップ呼ぶ。ステージ1 の :func:`resample_stage1_target` と同じ思想で、
    「成功したらエピソードを切らずに次の課題を出す」。

    なぜ継続にするか:
        従来は ``save_success`` が ``DoneTerm`` で、**セーブ成功＝エピソード終了**
        だった。そのため
          * 1 エピソード (10s) に報酬イベントが 1 回しか無く、ただでさえスパースな
            セーブ報酬がさらに希薄になっていた。
          * ``return_to_center_after_save`` が発火する間もなく終了するため、
            「弾いた後に定位置へ戻る」がほぼ学習されていなかった。
        継続にすると 1 エピソードに複数回のセーブが入り、単位時間あたりの学習信号が
        増える。さらに「セーブ後すぐ中央へ戻らないと次が取れない」という圧力が
        自然に生まれる (実機の連続シュート対応に近い)。

    失敗 (失点・転倒・場外) は従来通り ``DoneTerm`` でエピソードを切る。
    ロボットの位置はリセットせず、**セーブした場所からそのまま次の球に備える**。

    Args:
        respawn_delay_steps: セーブ確定から次の発射までの待ちステップ数。
            弾いたボールが転がりきる前に撃つと接触判定が混線するので間を置く。
            50 step = 1.0s @ 50Hz。
    """
    bufs = gk_buffers(env)
    cd = bufs["respawn_cd"]

    # --- 1. セーブ確定を検知して待ちカウンタを開始する ---
    # update_save_state は save_cd を進め、確定時に 0 になる (冪等なので多重呼び出し可)。
    from .terminations import update_save_state

    update_save_state(env)
    newly_saved = (bufs["save_cd"] == 0) & (cd < 0)
    if bool(newly_saved.any()):
        cd[newly_saved] = int(respawn_delay_steps)
        # 適応カリキュラムが「1 球あたりの成功率」を出すための実績カウント
        bufs["save_count"][newly_saved] += 1
        # セーブ品質 (どれだけ危険圏から遠ざけたか) を記録する。
        q = _save_clearance_quality(env, safe_dist=float(clearance_safe_dist))
        bufs["save_quality"][newly_saved] = q[newly_saved]
        # 確定済みとして save_cd を無効化 (save_success と同じ後始末)
        bufs["save_cd"][newly_saved] = -1
        # ボールを非アクティブにして観測をダミーに戻す (次の発射までは「脅威なし」)
        bufs["ball_active"][newly_saved] = False

    # --- 2. 待ちカウンタを進め、0 になった env に次の球を撃つ ---
    waiting = cd > 0
    cd[waiting] -= 1
    fire = cd == 0
    if not bool(fire.any()):
        return

    fire_ids = torch.nonzero(fire).flatten()
    # ボール 1 球ぶんの状態をリセットしてから再発射する (touched/touch_rewarded を
    # 残すと 2 球目以降の save_touch_bonus が払われない)。
    bufs["touched"][fire_ids] = False
    bufs["touch_rewarded"][fire_ids] = False
    bufs["save_cd"][fire_ids] = -1
    cd[fire_ids] = -1

    reset_ball_shot(env, fire_ids, ball_cfg=ball_cfg)
    reset_ball_perception(env, fire_ids)
