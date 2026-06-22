# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for the K1 locomotion task."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_command_resampling_time_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    resampling_time_range: tuple[float, float],
    num_steps: int,
):
    """指定ステップ数を超えたら、コマンドのリサンプリング時間範囲を変更する。"""
    if env.common_step_counter > num_steps:
        term = env.command_manager.get_term(command_name)
        term.cfg.resampling_time_range = resampling_time_range


def modify_push_robot(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
    interval_range_s: tuple[float, float] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
):
    """指定ステップ数を超えたら、push_robot イベントの interval_range_s と velocity_range を更新する。

    EventManager は次回のインターバルサンプリング時に ``term_cfg.interval_range_s`` を
    再読込するため、cfg の書き換えだけで反映される。``velocity_range`` は ``params`` 経由で
    イベント関数に渡されるので、こちらも cfg.params を上書きすれば次回呼び出しで反映される。
    """
    if env.common_step_counter > num_steps:
        term_cfg = env.event_manager.get_term_cfg(term_name)
        if interval_range_s is not None:
            term_cfg.interval_range_s = interval_range_s
        if velocity_range is not None:
            term_cfg.params["velocity_range"] = velocity_range


def randomize_ball_init_velocity(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
    velocity_range: dict[str, tuple[float, float]],
):
    """指定ステップ数を超えたら、reset イベントの ``velocity_range`` を差し替えてボールに初速を与える。

    ``term_name`` は ``mdp.reset_root_state_uniform`` を使ったイベント (例: ``reset_ball``)
    を指す。EventManager は reset 呼び出し時に ``term_cfg.params`` を再読込するため、
    ``params["velocity_range"]`` を上書きするだけで次回 reset から反映される。

    Note:
        ``num_steps`` は ``env.common_step_counter`` (env.step 呼び出し回数 ≒ num_steps_per_env *
        iteration 数) と比較するので、「総学習ステップ数の半分」を狙うなら
        ``num_steps_per_env * max_iterations // 2`` を渡せばよい。
    """
    if env.common_step_counter > num_steps:
        term_cfg = env.event_manager.get_term_cfg(term_name)
        term_cfg.params["velocity_range"] = velocity_range

def modify_reward_weight_linear(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int,
):
    """報酬項の weight を ``start_step`` → ``end_step`` で ``start_weight`` → ``end_weight`` に線形変化させる。

    ``start_step`` 以前は ``start_weight``、``end_step`` 以降は ``end_weight`` で据え置く。
    ペナルティを学習初期は弱く(または 0)、後半で強めたい場合に使う。

    Note:
        ``start_step`` / ``end_step`` は ``env.common_step_counter`` (env.step 呼び出し回数
        ≒ num_steps_per_env * iteration 数) と比較する。
    """
    s = env.common_step_counter
    if s <= start_step:
        weight = start_weight
    elif s >= end_step:
        weight = end_weight
    else:
        alpha = (s - start_step) / float(end_step - start_step)
        weight = start_weight + alpha * (end_weight - start_weight)
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)


def randomize_ball_init_pose(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
    pose_range: dict[str, tuple[float, float]],
):
    """指定ステップ数を超えたら、reset イベントの ``pose_range`` を差し替えてボールに初期姿勢を与える。

    ``term_name`` は ``mdp.reset_root_state_uniform`` を使ったイベント (例: ``reset_ball``)
    を指す。EventManager は reset 呼び出し時に ``term_cfg.params`` を再読込するため、
    ``params["pose_range"]`` を上書きするだけで次回 reset から反映される。

    Note:
        ``num_steps`` は ``env.common_step_counter`` (env.step 呼び出し回数 ≒ num_steps_per_env *
        iteration 数) と比較するので、「総学習ステップ数の半分」を狙うなら
        ``num_steps_per_env * max_iterations // 2`` を渡せばよい。
    """
    if env.common_step_counter > num_steps:
        term_cfg = env.event_manager.get_term_cfg(term_name)
        term_cfg.params["pose_range"] = pose_range

class lin_vel_command_curriculum(ManagerTermBase):
    """線速度コマンド範囲(lin_vel_x / lin_vel_y)を段階的に拡げるカリキュラム。

    全環境にわたるトラッキング誤差 ``||cmd_xy - root_lin_vel_b_xy||_2`` の指数移動平均(EMA)が
    ``error_threshold`` を下回ったら次のステージに進む。最終ステージに到達したら以降は据え置く。

    Args:
        stages_x: 各ステージで ``lin_vel_x`` に適用する ``(min, max)`` のリスト。
        stages_y: 各ステージで ``lin_vel_y`` に適用する ``(min, max)`` のリスト。 ``stages_x`` と同じ長さでなければならない。
        error_threshold: ステージを進めるためのEMA誤差(m/s)の上限。
            単一の float を渡すと全ステージ共通の閾値になる。
            ``stages_x`` と同じ長さのリストを渡すとステージごとに閾値を設定でき、
            広い速度範囲のステージほど緩い(大きい)閾値にして難易度を均せる。
        command_name: 対象コマンド名(例: ``"base_velocity"``)。
        asset_name: ロボットのアセット名。
        ema_alpha: EMA の更新係数 (0,1]。大きいほど直近の誤差を強く反映する。
        min_updates: ステージ進行を許可する前に必要な呼び出し回数(EMAを温めるため)。
        stage_cooldown_resamples: ステージを進めた直後、誤差計測(EMA)を再開するまで待つ
            「コマンド再サンプリング周期」の倍数。ステージを進めると ``cfg.ranges`` は即座に
            広がるが、各 env が新しい範囲のコマンドを実際に引くのは次回の再サンプリング時
            (最大 ``resampling_time_range[1]`` 秒後)である。この待機を入れないと、ステージ変更
            直後の EMA は依然として古い狭い範囲の誤差を映しており、緩い次ステージ閾値を即座に
            満たして 0→1→2 と一気に遷移してしまう。再サンプリング周期 (= ``resampling_time_range``
            の最大値) と env.step_dt から待機ステップ数を算出する。1.0 で「全 env が最低 1 回は
            新範囲を引く」のを保証し、EMA 収束の余裕を見て既定 1.5。
        post_switch_hold_steps: ステージを進めた直後に EMA を高い値で固定し、計測・更新・判定を
            止める最小ステップ数 (既定 500)。実際の hold は ``stage_cooldown_resamples`` から
            算出した再サンプリング待ちステップ数とこの値の大きい方になる。固定値で下限を
            設けることで、再サンプリング周期が短い設定でも切替直後の一気な遷移を確実に防ぐ。
        post_switch_ema_scale: 切替直後に EMA を固定する初期値の、新ステージ閾値に対する倍率
            (既定 2.0)。hold 明けはこの高い値から実測値へ向けて減衰するため、運良く低い誤差を
            引いただけで即遷移するのを防ぐ。大きいほど次遷移までの猶予が長くなる。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._stages_x: list[tuple[float, float]] = [tuple(s) for s in params["stages_x"]]
        self._stages_y: list[tuple[float, float]] = [tuple(s) for s in params["stages_y"]]
        if len(self._stages_x) != len(self._stages_y):
            raise ValueError(
                f"stages_x と stages_y は同じ長さでなければなりません: "
                f"len(stages_x)={len(self._stages_x)}, len(stages_y)={len(self._stages_y)}"
            )
        # error_threshold は float(全ステージ共通)または stages と同じ長さのリスト(ステージ毎)
        raw_threshold = params["error_threshold"]
        if isinstance(raw_threshold, (list, tuple)):
            self._error_thresholds: list[float] = [float(t) for t in raw_threshold]
            if len(self._error_thresholds) != len(self._stages_x):
                raise ValueError(
                    f"error_threshold をリストで渡す場合は stages と同じ長さでなければなりません: "
                    f"len(error_threshold)={len(self._error_thresholds)}, len(stages_x)={len(self._stages_x)}"
                )
        else:
            self._error_thresholds = [float(raw_threshold)] * len(self._stages_x)
        self._command_name: str = params["command_name"]
        self._asset_name: str = params.get("asset_name", "robot")

        self._current_stage: int = 0
        # EMA は GPU 上のスカラーテンソルのまま保持し、毎ステップの .item() 同期を避ける。
        # 閾値判定・ログ用に CPU 値が必要なときだけ同期する (下記 __call__ 参照)。
        self._error_ema: torch.Tensor | None = None
        self._cached_ema: float = 0.0  # ログ表示用にキャッシュした EMA(同期時のみ更新)
        self._update_count: int = 0
        # ステージ変更直後の hold ステップ数。残っている間は EMA 計測・更新・判定を
        # 完全に停止し、EMA を高い値に固定したまま待つ。全 env が新しい範囲のコマンドを
        # 引いてポリシーがある程度適応するのを待ってから計測を再開する (チェーン遷移を防ぐ)。
        self._hold_remaining: int = 0

        # 初期ステージの範囲を即時適用
        self._apply_stage(self._current_stage)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stages_x: Sequence[Sequence[float]],
        stages_y: Sequence[Sequence[float]],
        error_threshold: float | Sequence[float],
        command_name: str,
        asset_name: str = "robot",
        ema_alpha: float = 0.02,
        min_updates: int = 50,
        stage_cooldown_resamples: float = 1.5,
        post_switch_hold_steps: int = 500,
        post_switch_ema_scale: float = 2.0,
    ) -> dict[str, float]:
        asset: Articulation = env.scene[asset_name]
        cmd_term = env.command_manager.get_term(command_name)

        current_threshold = self._error_thresholds[self._current_stage]

        # --- ステージ変更直後の hold ---
        # ステージを進めると cfg.ranges は即座に広がるが、各 env が新しい範囲のコマンドを
        # 実際に引くのは次回の再サンプリング時 (最大 resampling_time_range[1] 秒後)。
        # さらにポリシーが新範囲に適応するにも時間がかかる。この間に EMA を更新・判定すると、
        # 依然として古い狭い範囲の誤差を見て次の (より緩い) 閾値を即満たし、0→1→2 と
        # 一気に遷移してしまう。hold 中は EMA を高い値に固定したまま計測・更新・判定を
        # 完全に止め、全 env が新範囲を引いて適応するのを待つ。hold 明けも EMA は高い値の
        # ままなので、実測値へ向けて減衰しきる (= 本当に追従できる) まで次遷移は起きない。
        if self._hold_remaining > 0:
            self._hold_remaining -= 1
            return {
                "stage": float(self._current_stage),
                "error_ema": float(self._cached_ema),
                "error_threshold": float(current_threshold),
                "lin_vel_x_max": float(self._stages_x[self._current_stage][1]),
                "lin_vel_y_max": float(self._stages_y[self._current_stage][1]),
            }

        cmd_lin_xy = cmd_term.command[:, :2]
        actual_lin_xy = asset.data.root_lin_vel_b[:, :2]
        # 誤差(全env平均)は GPU 上のスカラーテンソルのまま計算する。ここで .item() しない
        # ことで、毎ステップ走る curriculum 更新での GPU→CPU 同期(collection の律速要因)を排除する。
        err = torch.norm(cmd_lin_xy - actual_lin_xy, dim=-1).mean()

        if self._error_ema is None:
            self._error_ema = err
        else:
            self._error_ema = (1.0 - ema_alpha) * self._error_ema + ema_alpha * err
        self._update_count += 1

        # 閾値判定とログ値の更新は min_updates ステップごとにまとめて行う。
        # CPU への同期 (.item()) はこの分岐の中だけで起き、毎ステップではなくなる。
        # NOTE: 旧実装は warm-up 後は毎ステップ判定していたが、本実装では判定が周期的になる
        #       (ステージ進行が最大 min_updates ステップ遅れる)。EMA 値自体は同一に更新され続け、
        #       カリキュラムの挙動への実質的な影響はない。
        if self._update_count >= min_updates:
            ema_val = float(self._error_ema)  # 同期はここだけ (min_updates 回に1回)
            self._cached_ema = ema_val
            if self._current_stage < len(self._stages_x) - 1 and ema_val < current_threshold:
                self._current_stage += 1
                self._apply_stage(self._current_stage)
                current_threshold = self._error_thresholds[self._current_stage]
                # 切り替え直後は EMA を「新ステージ閾値より十分高い値」で固定する。
                # こうすると hold 明けに実測値で seed し直す代わりに、この高い値から
                # 実測値へ向けて徐々に減衰していくため、運良く低い誤差を 1 回引いただけで
                # 即次ステージへ進む (一気に遷移する) ことを防げる。
                high_ema = float(current_threshold) * post_switch_ema_scale
                self._error_ema = self._error_ema.new_full((), high_ema)
                self._cached_ema = high_ema
                # hold は「新範囲が全 env に行き渡るまで」と「固定 500 ステップ」の
                # 大きい方。前者は resampling_time_range の最大値 (秒) ÷ step_dt (秒)。
                max_resample_s = float(cmd_term.cfg.resampling_time_range[1])
                resample_steps = int(
                    math.ceil(stage_cooldown_resamples * max_resample_s / env.step_dt)
                )
                self._hold_remaining = max(int(post_switch_hold_steps), resample_steps)
            self._update_count = 0

        return {
            "stage": float(self._current_stage),
            "error_ema": float(self._cached_ema),
            "error_threshold": float(current_threshold),
            "lin_vel_x_max": float(self._stages_x[self._current_stage][1]),
            "lin_vel_y_max": float(self._stages_y[self._current_stage][1]),
        }

    def _apply_stage(self, stage_idx: int) -> None:
        cmd_term = self._env.command_manager.get_term(self._command_name)
        cmd_term.cfg.ranges.lin_vel_x = tuple(self._stages_x[stage_idx])
        cmd_term.cfg.ranges.lin_vel_y = tuple(self._stages_y[stage_idx])


class ball_max_speed_curriculum(ManagerTermBase):
    """ball_velocity_toward_target の max_speed を段階的に上げるカリキュラム。

    全環境にわたる「目標方向へのボール速度成分」 v_along の EMA が
    ``threshold_ratio * 現在 max_speed`` を超えたら次のステージに進む。
    最終ステージに到達したら据え置く。

    Args:
        stages: 各ステージで使用する ``max_speed`` の値のリスト (昇順)。
        threshold_ratio: ステージを進めるための ``v_along_ema / max_speed`` の下限。
        reward_term_name: 更新対象の reward term 名 (params["max_speed"] を書き換える)。
        command_name: ボールの目標位置を持つコマンド名。
        ball_asset: ボールのシーンエンティティ名。
        ema_alpha: EMA の更新係数 (0,1]。
        min_updates: ステージ進行を許可する前に必要な呼び出し回数。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._stages: list[float] = [float(s) for s in params["stages"]]
        self._threshold_ratio: float = float(params["threshold_ratio"])
        self._reward_term_name: str = params["reward_term_name"]
        self._command_name: str = params.get("command_name", "target_pos")
        self._ball_asset: str = params.get("ball_asset", "soccer_ball")

        self._current_stage: int = 0
        self._v_along_ema: float | None = None
        self._update_count: int = 0

        self._apply_stage(self._stages[self._current_stage])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stages: Sequence[float],
        threshold_ratio: float,
        reward_term_name: str,
        command_name: str = "target_pos",
        ball_asset: str = "soccer_ball",
        ema_alpha: float = 0.02,
        min_updates: int = 50,
    ) -> dict[str, float]:
        ball = env.scene[ball_asset]
        ball_pos_w = ball.data.root_pos_w[:, :2]
        ball_vel_w = ball.data.root_com_vel_w[:, :2]
        target_pos_w = env.command_manager.get_term(command_name).pos_command_w[:, :2]
        desired_dir = target_pos_w - ball_pos_w
        desired_unit = desired_dir / (torch.norm(desired_dir, dim=1, keepdim=True) + 1e-6)
        v_along = (ball_vel_w * desired_unit).sum(dim=1).clamp(min=0.0)
        v_along_mean = v_along.mean().item()

        if self._v_along_ema is None:
            self._v_along_ema = v_along_mean
        else:
            self._v_along_ema = (1.0 - ema_alpha) * self._v_along_ema + ema_alpha * v_along_mean
        self._update_count += 1

        current_max_speed = self._stages[self._current_stage]
        if (
            self._current_stage < len(self._stages) - 1
            and self._update_count >= min_updates
            and self._v_along_ema > threshold_ratio * current_max_speed
        ):
            self._current_stage += 1
            self._apply_stage(self._stages[self._current_stage])
            self._update_count = 0
            self._v_along_ema = None

        return {
            "stage": float(self._current_stage),
            "v_along_ema": float(self._v_along_ema if self._v_along_ema is not None else 0.0),
            "max_speed": float(self._stages[self._current_stage]),
        }

    def _apply_stage(self, max_speed: float) -> None:
        term_cfg = self._env.reward_manager.get_term_cfg(self._reward_term_name)
        term_cfg.params["max_speed"] = max_speed


class kick_direction_command_curriculum(ManagerTermBase):
    """蹴り方向 (target_pos の pos_y 範囲) を段階的に拡げるカリキュラム。

    動いているボールに対して ``1 - cos(ball_vel, desired_dir)`` の EMA が
    ``error_threshold`` を下回ったら次のステージへ。
    最終ステージに到達したら据え置く。

    Args:
        stages: 各ステージで使用する ``(pos_y_min, pos_y_max)`` のリスト。
            最初のステージを ``(0.0, 0.0)`` にすれば「正面のみ」になる。
        error_threshold: ステージを進めるための ``1 - cos_sim`` の上限。
            例: 0.2 なら cos_sim > 0.8 (≒ 37° 以内) になったら進む。
        command_name: 対象コマンド名 (UniformPose2dCommandCfg)。
        ball_asset: ボールのシーンエンティティ名。
        min_speed: 角度誤差を集計する際のボール速度下限 [m/s]。
            停止中はノイズになるので除外する。
        ema_alpha: EMA の更新係数 (0,1]。
        min_updates: ステージ進行を許可する前に必要な呼び出し回数。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._stages: list[tuple[float, float]] = [tuple(s) for s in params["stages"]]
        self._error_threshold: float = float(params["error_threshold"])
        self._command_name: str = params["command_name"]
        self._ball_asset: str = params.get("ball_asset", "soccer_ball")
        self._min_speed: float = float(params.get("min_speed", 0.5))

        self._current_stage: int = 0
        self._error_ema: float | None = None
        self._update_count: int = 0

        self._apply_stage(self._stages[self._current_stage])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stages: Sequence[Sequence[float]],
        error_threshold: float,
        command_name: str,
        ball_asset: str = "soccer_ball",
        min_speed: float = 0.5,
        ema_alpha: float = 0.02,
        min_updates: int = 50,
    ) -> dict[str, float]:
        ball = env.scene[ball_asset]
        ball_pos_w = ball.data.root_pos_w[:, :2]
        ball_vel_w = ball.data.root_com_vel_w[:, :2]
        target_pos_w = env.command_manager.get_term(command_name).pos_command_w[:, :2]
        desired_dir = target_pos_w - ball_pos_w

        ball_speed = torch.norm(ball_vel_w, dim=1)
        mask = ball_speed > min_speed
        if mask.any():
            cos_sim = torch.nn.functional.cosine_similarity(
                ball_vel_w[mask], desired_dir[mask], dim=1, eps=1e-6
            )
            err = (1.0 - cos_sim).mean().item()
            if self._error_ema is None:
                self._error_ema = err
            else:
                self._error_ema = (1.0 - ema_alpha) * self._error_ema + ema_alpha * err

        self._update_count += 1

        if (
            self._current_stage < len(self._stages) - 1
            and self._update_count >= min_updates
            and self._error_ema is not None
            and self._error_ema < error_threshold
        ):
            self._current_stage += 1
            self._apply_stage(self._stages[self._current_stage])
            self._update_count = 0
            self._error_ema = None

        return {
            "stage": float(self._current_stage),
            "error_ema": float(self._error_ema if self._error_ema is not None else -1.0),
            "pos_y_max": float(self._stages[self._current_stage][1]),
        }

    def _apply_stage(self, y_range: tuple[float, float]) -> None:
        cmd_term = self._env.command_manager.get_term(self._command_name)
        cmd_term.cfg.ranges.pos_y = tuple(y_range)


class kick_angle_range_curriculum(ManagerTermBase):
    """``KickDirectionCommand`` の ``angle_range`` を性能ベースで段階的に拡げるカリキュラム。

    最初は正面付近の狭い角度レンジ (例: ``(-0.5, 0.5)``) のみを出題する。ロボットの初期
    yaw はほぼ正面・ボールも正面なので、狭いレンジでは「直線接近 → そのまま前へ蹴る」が
    そのまま正解になり、回り込みを必要としない。これにより「とにかく蹴る」局所解ではなく
    「正しい方向に蹴る」挙動を先に獲得させ、その後レンジを ``(-π, π)`` まで広げて
    回り込みが必要なケースを徐々に導入する。

    動いているボールに対する方向誤差 ``1 - cos(ball_vel, kick_dir)`` の EMA が
    ``error_threshold`` を下回ったら次のステージへ進む。最終ステージで据え置く。

    Args:
        stages: 各ステージで使用する ``(angle_min, angle_max)`` [rad] のリスト。
            0 を中心に対称に広げるのが基本 (例: ``[(-0.5,0.5), (-1.2,1.2), (-2.0,2.0), (-π,π)]``)。
        error_threshold: ステージを進めるための ``1 - cos_sim`` の上限。
            例: 0.3 なら cos_sim > 0.7 (≒ 45° 以内) で進む。
            float を渡すと全ステージ共通。``stages`` と同じ長さのリストを渡すとステージ毎に
            設定でき、広い角度レンジのステージほど緩い (大きい) 閾値にして「広い範囲は本質的に
            精度が落ちる」分を均し、永久停滞を防げる。
        command_name: 対象コマンド名 (``KickDirectionCommandCfg``)。
        ball_asset: ボールのシーンエンティティ名。
        min_speed: 角度誤差を集計するボール速度下限 [m/s]。停止中はノイズなので除外する。
        ema_alpha: EMA の更新係数 (0,1]。
        min_updates: 閾値判定を行う呼び出し回数の周期 (EMA を温める)。
        stage_cooldown_resamples: ステージを進めた直後、計測を再開するまで待つ「コマンド
            再サンプリング周期」の倍数。レンジを広げても各 env が新レンジのキック方向を
            実際に引くのは次の再サンプリング時 (最大 ``resampling_time_range[1]`` 秒後) で、
            それまでに蹴られて飛んでいるボールは **前の狭いレンジ** のものなので、待たずに
            計測すると古い (揃った) 誤差を見て即連鎖遷移する。再サンプリング周期 ÷ step_dt で
            待機ステップ数を算出する。
        post_switch_hold_steps: 切替直後に計測・更新・判定を止める最小ステップ数。実際の
            hold は再サンプリング待ちとこの値の大きい方。固定下限で確実に連鎖遷移を防ぐ。
        post_switch_ema_scale: 切替直後に EMA を固定する初期値の、閾値に対する倍率。hold 明けは
            この高い値から実測値へ減衰するため、運良く低い誤差を 1 回引いただけでは進まない。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._stages: list[tuple[float, float]] = [tuple(s) for s in params["stages"]]
        # error_threshold は float (全ステージ共通) か stages と同じ長さのリスト (ステージ毎)。
        raw_threshold = params["error_threshold"]
        if isinstance(raw_threshold, (list, tuple)):
            self._error_thresholds: list[float] = [float(t) for t in raw_threshold]
            if len(self._error_thresholds) != len(self._stages):
                raise ValueError(
                    f"error_threshold をリストで渡す場合は stages と同じ長さにすること: "
                    f"len(error_threshold)={len(self._error_thresholds)}, len(stages)={len(self._stages)}"
                )
        else:
            self._error_thresholds = [float(raw_threshold)] * len(self._stages)
        self._command_name: str = params["command_name"]
        self._ball_asset: str = params.get("ball_asset", "soccer_ball")
        self._min_speed: float = float(params.get("min_speed", 0.5))

        self._current_stage: int = 0
        self._error_ema: float | None = None
        self._cached_ema: float = -1.0
        self._update_count: int = 0
        self._hold_remaining: int = 0

        self._apply_stage(self._stages[self._current_stage])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stages: Sequence[Sequence[float]],
        error_threshold: float | Sequence[float],
        command_name: str,
        ball_asset: str = "soccer_ball",
        min_speed: float = 0.5,
        ema_alpha: float = 0.02,
        min_updates: int = 100,
        stage_cooldown_resamples: float = 1.5,
        post_switch_hold_steps: int = 1500,
        post_switch_ema_scale: float = 2.0,
    ) -> dict[str, float]:
        current_threshold = self._error_thresholds[self._current_stage]

        # --- ステージ変更直後の hold ---
        # 新レンジのキック方向が全 env に行き渡り、前レンジで飛んでいたボールが消えて
        # ポリシーが適応するまで、計測・更新・判定を完全に止める (連鎖遷移を防ぐ)。
        if self._hold_remaining > 0:
            self._hold_remaining -= 1
            return {
                "stage": float(self._current_stage),
                "error_ema": float(self._cached_ema),
                "error_threshold": float(current_threshold),
                "angle_max": float(self._stages[self._current_stage][1]),
            }

        ball = env.scene[ball_asset]
        ball_vel_w = ball.data.root_com_vel_w[:, :2]
        kick_dir_w = env.command_manager.get_term(command_name).command  # (N, 2) 単位ベクトル

        ball_speed = torch.norm(ball_vel_w, dim=1)
        mask = ball_speed > min_speed
        if mask.any():
            cos_sim = torch.nn.functional.cosine_similarity(
                ball_vel_w[mask], kick_dir_w[mask], dim=1, eps=1e-6
            )
            err = (1.0 - cos_sim).mean()
            if self._error_ema is None:
                self._error_ema = err
            else:
                self._error_ema = (1.0 - ema_alpha) * self._error_ema + ema_alpha * err

        self._update_count += 1

        # 閾値判定とログ用 .item() 同期は min_updates 周期でまとめて行う。
        if self._update_count >= min_updates and self._error_ema is not None:
            ema_val = float(self._error_ema)
            self._cached_ema = ema_val
            if self._current_stage < len(self._stages) - 1 and ema_val < current_threshold:
                self._current_stage += 1
                self._apply_stage(self._stages[self._current_stage])
                current_threshold = self._error_thresholds[self._current_stage]
                # EMA を「新ステージ閾値」より十分高い値で固定し、hold 明けに実測値へ減衰させる。
                high_ema = current_threshold * post_switch_ema_scale
                self._error_ema = self._error_ema.new_full((), high_ema)
                self._cached_ema = high_ema
                # hold = 「新レンジが全 env に行き渡るまで」と固定下限の大きい方。
                cmd_term = env.command_manager.get_term(command_name)
                max_resample_s = float(cmd_term.cfg.resampling_time_range[1])
                resample_steps = int(math.ceil(stage_cooldown_resamples * max_resample_s / env.step_dt))
                self._hold_remaining = max(int(post_switch_hold_steps), resample_steps)
            self._update_count = 0

        return {
            "stage": float(self._current_stage),
            "error_ema": float(self._cached_ema),
            "error_threshold": float(current_threshold),
            "angle_max": float(self._stages[self._current_stage][1]),
        }

    def _apply_stage(self, angle_range: tuple[float, float]) -> None:
        cmd_term = self._env.command_manager.get_term(self._command_name)
        cmd_term.cfg.angle_range = tuple(angle_range)
