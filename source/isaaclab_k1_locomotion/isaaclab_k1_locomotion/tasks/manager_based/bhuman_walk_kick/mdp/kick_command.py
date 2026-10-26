# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""B-Human の walk-kick 環境 (``envs/t1_ball.py``) の状態管理を 1 つの CommandTerm に移植したもの。

参照元: https://github.com/bhuman/MachineLearning (IsaacGymRL/envs/t1_ball.py, K1_Ball.yaml)

B-Human の環境は「物理ステップ → ``_update_kick_tensors`` (キック検出・キック姿勢・
ボール相対位置) → ボール再配置要求 → 押し → オドメトリ → 終了判定 → 報酬 →
リセット → 観測」を 1 本の ``step()`` で回している。IsaacLab の manager 構成では
この順序を次のように対応させる:

* 物理ステップ直後の更新 (``_update_kick_tensors`` 〜 ``_check_termination``) は
  :meth:`BHumanKickCommand.update` にまとめ、終了判定・報酬・観測の各関数が最初に
  呼んだ時点で 1 ステップに 1 回だけ実行する (``env.common_step_counter`` で判定)。
  IsaacLab の step は「終了判定 → 報酬 → リセット → コマンド → 観測」の順なので、
  最初に呼ぶのは必ず終了判定になり、B-Human と同じ順序になる。
* エピソードのリセット (``_reset_idx``) は :meth:`reset` (CommandManager.reset)。
  ロボットの初期姿勢は reset event が書いた後なので、そこからボールを置ける。
* ボールだけのリセット (``reset_ball``) は終了判定で立てたフラグを :meth:`compute`
  (リセット後・観測前に呼ばれる) で処理する。B-Human もリセット後・観測前に処理している。

B-Human はボールのリセットのたびに ``env.episode_length_buf`` を 0 に戻すが、ここでは戻さない。
歩行 (bhuman_walk_kick の土台) の位相は ``episode_length_buf == 0`` で 0 に巻き戻るので、
ボールごとに戻すと歩いている最中に位相が飛ぶ。代わりにボール 1 個あたりの経過ステップ
``ball_steps`` を持ち、B-Human が ``episode_length_buf`` で見ていた所 (キック判定の猶予、
行き過ぎ判定、30 秒のタイムアウト) はこれで見る。エピソードは ``episode_number_of_kicks`` 回
蹴り終わった時点 (``kicks_done``) か、1 個のボールに 30 秒かかった時点 (``ball_timeout``) で終わる。

移植していないもの (理由はいずれも「うちの walk 側の仕組みをそのまま使う」):

* 関節 PD を python で回す部分 (遅延・フィルタ付き速度・摩擦・トルク clip) →
  土台の walk のアクチュエータ設定。
* ロボットへの速度インパルス (``_kick_robots``)、初期姿勢・関節・PD ゲイン・質量の DR →
  土台の walk の events。胴体を押し続ける ``_push_robots`` も既定では切ってある
  (``push_force_std = push_torque_std = 0``)。
* 足上げ報酬 (``feet_swing``) → 土台の walk の位相報酬 (``feet_phase``) が受け持つ。
* 関節キャリブレーションずれ (``dof_pos_offset``)、ボール半径の DR、heavy_ball、
  場外テレポート。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --------------------------------------------------------------------------- #
# 小物
# --------------------------------------------------------------------------- #
def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """wxyz クォータニオンの yaw (B-Human の get_euler_xyz の 3 番目と同じ値)。"""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def roll_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def rotate_2d(v: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]), dim=-1)


def uniform_log(shape, lo: float, hi: float, device) -> torch.Tensor:
    """B-Human の ``uniform_log`` 分布 (下限寄りに偏った一様分布)。"""
    u = torch.rand(shape, device=device)
    s = -torch.log(torch.rand(shape, device=device).clamp(min=0.0067, max=0.37))
    return lo + (hi - lo) * u / s


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return quat_from_euler_xyz(z, z, yaw)


# --------------------------------------------------------------------------- #
# Command term
# --------------------------------------------------------------------------- #
class BHumanKickCommand(CommandTerm):
    """キック方向・距離の指令と、B-Human 環境のボール/キック状態をまとめて持つ。

    ``command`` は (キック方向 [rad/π], 期待ボール初速 [m/s], 強キックフラグ)。
    観測・報酬は ``command`` ではなく、このクラスの属性を直接読む。
    """

    cfg: BHumanKickCommandCfg

    def __init__(self, cfg: BHumanKickCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.ball: RigidObject = env.scene[cfg.ball_name]
        self.feet_ids = self.robot.find_bodies(list(cfg.foot_body_names), preserve_order=True)[0]
        self.trunk_id = self.robot.find_bodies(cfg.base_body_name)[0][0]
        self.joint_ids = self.robot.find_joints(list(cfg.joint_names), preserve_order=True)[0]
        self.hip_yaw_ids = self.robot.find_joints(["Left_Hip_Yaw", "Right_Hip_Yaw"], preserve_order=True)[0]

        n, dev = self.num_envs, self.device
        f = dict(device=dev, dtype=torch.float)
        self.dt = env.step_dt
        self.ball_r = cfg.ball_radius
        # キック判定・再配置判定に使う「ボールに触れている」距離 (B-Human: 半径 + 半径 DR 上限 + 0.04)
        self.touch_dist = cfg.ball_radius + cfg.ball_radius_margin + 0.04

        self.feet_edge_rel = torch.tensor(cfg.feet_edge_pos, **f)  # (4, 3)
        self.sole_rel = torch.tensor(cfg.feet_sole_pos, **f)  # (3,)

        # 半分の env はボールが転がった状態から始める (B-Human は地形の右半分の env)。
        self.moving_ball_env = (torch.arange(n, device=dev) % 2) == 1

        # ---- ロボット状態 (update で毎ステップ更新) ----
        self.base_pos = torch.zeros(n, 3, **f)
        self.base_quat = torch.zeros(n, 4, **f)
        self.base_quat[:, 0] = 1.0
        self.yaw = torch.zeros(n, **f)
        self.base_quat_z = self.base_quat.clone()
        self.base_lin_vel = torch.zeros(n, 3, **f)
        self.base_ang_vel = torch.zeros(n, 3, **f)
        self.filtered_lin_vel = torch.zeros(n, 3, **f)
        self.filtered_ang_vel = torch.zeros(n, 3, **f)
        self.root_vel_w = torch.zeros(n, 6, **f)
        self.last_root_vel_w = torch.zeros(n, 6, **f)
        self.dof_vel = torch.zeros(n, len(self.joint_ids), **f)
        self.hip_yaw = torch.zeros(n, 2, **f)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.feet_pos = torch.zeros(n, 2, 3, **f)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_quat = torch.zeros(n, 2, 4, **f)
        self.feet_quat[..., 0] = 1.0
        self.feet_vel = torch.zeros(n, 2, 3, **f)
        self.feet_vel_filtered = torch.zeros(n, 2, 3, **f)
        self.feet_roll = torch.zeros(n, 2, **f)
        self.feet_yaw = torch.zeros(n, 2, **f)
        self.feet_contact = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.last_feet_contact = torch.zeros_like(self.feet_contact)
        self.feet_height_over_ground = torch.zeros(n, 2, **f)
        self.last_feet_height_over_ground = torch.zeros(n, 2, **f)

        # ---- ボール状態 ----
        self.ball_pos_w = torch.zeros(n, 3, **f)
        self.ball_vel_w = torch.zeros(n, 6, **f)  # lin(3) + ang(3)
        self.last_ball_vel_w = torch.zeros(n, 3, **f)
        self.buffered_global_ball_pos = torch.zeros(n, cfg.ball_pos_buffer_size, 3, **f)
        self.delay_ball_pos = torch.zeros(n, dtype=torch.long, device=dev)
        self.ball_pos = torch.zeros(n, 3, **f)  # 両足裏中点・yaw フレームでのボール位置 (観測用)
        self.last_relative_ball_pos = torch.zeros(n, 2, **f)
        self.ball_vel = torch.zeros(n, 3, **f)  # yaw フレームでのボール速度
        self.ball_velocity_norm = torch.zeros(n, **f)
        self.ball_velocity_norm_at_kick = torch.zeros(n, **f)
        self.ball_vel_before_kick = torch.zeros(n, **f)
        self.ball_vel_w_before_kick = torch.zeros(n, 3, **f)  # ball_progress (B-Human に無い項) 用
        self.ball_still_counter = torch.zeros(n, **f)
        self.ball_kicked_counter = torch.zeros(n, **f)
        self.ball_pos_at_kick = torch.zeros(n, 3, **f)
        self.ball_freeze_at_kick = torch.zeros(n, dtype=torch.bool, device=dev)
        self.changed_ball_vel = torch.zeros(n, dtype=torch.bool, device=dev)
        self.allow_ball_kick = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ball_rel_root = torch.zeros(n, 2, **f)  # Stage 0 (kick teacher) 用: 胴体 yaw フレーム
        self.ball_rel_root_prev = torch.zeros(n, 2, **f)
        self.l_dis = torch.ones(n, **f)
        self.r_dis = torch.ones(n, **f)

        # ---- キック指令 ----
        self.direction_angle = torch.zeros(n, **f)  # [-1, 1] (×π でロボットの基準向きからの角度)
        self.direction_angle_current = torch.zeros(n, **f)  # 現在の胴体向きから見たキック方向 [rad]
        self.robot_direction_ref = torch.zeros(n, **f)
        self.global_direction = torch.zeros(n, **f)
        self.range = torch.zeros(n, **f)
        self.ball_end_target = torch.zeros(n, 2, **f)
        self.ball_kick_pose = torch.zeros(n, 3, **f)
        self.flag_is_strong_kick = torch.zeros(n, dtype=torch.bool, device=dev)
        self.flag_is_inaccurate_kick = torch.zeros(n, **f)
        self.flag_allow_deviation = torch.zeros(n, dtype=torch.bool, device=dev)

        # ---- キック姿勢 ----
        self.origin_pos = torch.zeros(n, 3, **f)
        self.ball_close_to_kick_pose = torch.ones(n, **f)
        self.inside_kick_pose_counter = torch.zeros(n, **f)
        self.kick_pose_translation_ratio = torch.zeros(n, **f)
        self.filtered_ball_pos = torch.zeros(n, 3, **f)
        self.kick_leg_was_obvious = torch.zeros(n, **f)
        self.ball_walk_target_overshoot = torch.zeros(n, **f)
        self.kick_sole_yaw_ratio = torch.zeros(n, **f)
        self.request_changed = torch.zeros(n, **f)

        # ---- オドメトリ (ボール観測ノイズ用) ----
        self.current_world_pose = torch.zeros(n, 3, **f)
        self.last_world_pose = torch.zeros(n, 3, **f)
        self.odo_buffer = torch.zeros(n, 10, 3, **f)
        self.odo_noise_factor = torch.zeros(n, 2, **f)

        # ---- 押し (privileged obs にも入る) ----
        self.push_force = torch.zeros(n, 3, **f)
        self.push_torque = torch.zeros(n, 3, **f)

        # ---- 終了・ボール再配置 ----
        self.ball_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.ball_timeout = torch.zeros(n, dtype=torch.bool, device=dev)
        self.terminate_counter = torch.zeros(n, **f)
        self.terminated = torch.zeros(n, dtype=torch.bool, device=dev)
        self.kicks_done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.reset_ball = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ball_reset_counter = torch.zeros(n, **f)

        # ---- 報酬キャッシュ ----
        self.ball_kick_direction_reward = torch.zeros(n, **f)

        # ---- 観測ノイズのキャッシュ (policy と critic で同じノイズを使う) ----
        self._obs_cache_step = -1
        self.obs_ball_pos = torch.zeros(n, 2, **f)
        self.obs_last_ball_pos = torch.zeros(n, 2, **f)
        self.obs_direction_angle = torch.zeros(n, **f)
        self.obs_ball_vel = torch.zeros(n, 2, **f)
        self.obs_expected_vel = torch.zeros(n, **f)

        self._last_update_step = -1

        self.metrics["kicks"] = torch.zeros(n, **f)

        # ---- 学習の確認用の集計 (B-Human には無い。報酬には使わない) ----
        # キック 1 回ごとに「ボールの最高速度」「目標方向とのずれ」を測り、直近のキックの
        # 指数移動平均でログに出す (reset() の戻り値 → Metrics/kick/*)。
        self.kick_target_dir = torch.zeros(n, **f)  # キックした瞬間の「ボール → 目標点」の向き (ワールド)
        self.kick_strike_height = torch.zeros(n, **f)  # キックした瞬間の蹴り足の足裏の一番低い点の高さ [m]
        self.kick_strike_vz = torch.zeros(n, **f)  # キックした瞬間の蹴り足の上下の速さ [m/s] (確認用)
        self.kick_stats = {
            "kick_speed": 0.0,             # 全キックのボール最高速度 [m/s]
            "kick_speed_strong": 0.0,      # 強キックのボール最高速度 [m/s]
            "kick_speed_err_normal": 0.0,  # 距離指定キック: |最高速度 - 必要な初速| [m/s]
            "kick_dir_err_deg": 0.0,       # 目標方向とボールの進んだ向きのずれ [deg]
            "kick_dir_ok_rate": 0.0,       # ずれ 15° 以内の割合
            "kick_strike_height": 0.0,     # 当てた瞬間の蹴り足の足裏の一番低い点の高さ [m]
            # 仕様 (docs/bhuman_walk_kick_spec.md) の項目。値は当たる直前 (キック判定の 1 制御ステップ前) のもの。
            "kick_foot_yaw_in_deg": 0.0,   # 蹴り足の向き - キック方向 (左足 +、右足 - を掛ける。+90 でインサイド)
            "kick_yaw_ok_rate": 0.0,       # |足の向き - キック方向| が 90±25° の割合
            "kick_inside_rate": 0.0,       # ボール中心が蹴り足の足首から見て内側 (±45° の扇) にある割合
            "kick_stop_rate": 0.0,         # 当たる前 1.5 秒に胴体の水平速度が 0.6 m/s 以下まで下がった割合
            "kick_plant_rate": 0.0,        # 当たった瞬間、蹴っていない足が 0.1 秒以上前から接地している割合
            "kick_swing_rate": 0.0,        # 蹴り足が浮いていて水平速度 1.0 m/s 以上の割合
            "kick_walkoff_rate": 0.0,      # 蹴って 3 秒以内に胴体の水平速度 (0.2 秒平均) が 0.3 m/s を超えた割合
            "kick_trunk_yaw_in_deg": 0.0,  # 胴体の向き - キック方向 (蹴り足の側へ回っていれば +。B-Human は約 30°)
            "kick_hip_yaw_out_deg": 0.0,   # 蹴り足の股関節 yaw (足先を外へ開く向きが +。B-Human は約 45°)
            "kick_strike_height_L": 0.0,   # 左足で蹴ったときの当たる直前の足裏の一番低い点 [m] (左右差を見る)
            "kick_strike_height_R": 0.0,   # 右足で蹴ったとき
        }
        # 仕様の項目を測るための記録 (報酬には使わない)
        self.speed_hist = torch.zeros(n, int(round(1.5 / self.dt)), **f)  # 胴体の水平速度 (古い → 新しい)
        self.contact_streak = torch.zeros(n, 2, **f)  # 足が続けて接地している制御ステップ数
        self.km_foot_yaw_abs = torch.zeros(n, **f)
        self.km_foot_yaw_in = torch.zeros(n, **f)
        self.km_ball_x = torch.zeros(n, **f)
        self.km_ball_yin = torch.zeros(n, **f)
        self.km_stop_s = torch.zeros(n, **f)
        self.km_min_speed = torch.zeros(n, **f)
        self.km_plant = torch.zeros(n, dtype=torch.bool, device=dev)
        self.km_foot_air = torch.zeros(n, dtype=torch.bool, device=dev)
        self.km_foot_hvel = torch.zeros(n, **f)
        self.km_trunk_yaw_in = torch.zeros(n, **f)
        self.km_kick_left = torch.zeros(n, dtype=torch.bool, device=dev)
        self.km_foot_roll_in = torch.zeros(n, **f)
        self.km_foot_pitch = torch.zeros(n, **f)
        self.km_hip_yaw_out = torch.zeros(n, **f)
        self.since_kick = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.walkoff_max = torch.zeros(n, **f)
        self.eval_walkoff: list | None = None
        self.kick_total = 0.0  # 学習開始からのキック回数 (全 env)
        self.kick_stats_alpha = 0.002
        self.eval_log: list | None = None

    def __str__(self) -> str:
        return "BHumanKickCommand: B-Human walk-kick の状態 (キック方向・距離、ボール、キック姿勢)"

    # ------------------------------------------------------------------ #
    # CommandTerm interface
    # ------------------------------------------------------------------ #
    @property
    def command(self) -> torch.Tensor:
        return torch.stack(
            (self.direction_angle_current / math.pi, self.expected_vel(), self.flag_is_strong_kick.float()), dim=-1
        )

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """エピソードのリセット (B-Human ``_reset_idx`` のボール/状態部分)。"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        extras = {}
        if len(env_ids) == 0:
            return extras
        extras["kicks"] = torch.mean(self.ball_reset_counter[env_ids]).item()
        extras.update(self.kick_stats)
        extras["kick_total"] = self.kick_total

        self._refresh_robot_state(env_ids)
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.last_root_vel_w[env_ids] = self.root_vel_w[env_ids]
        self.last_dof_vel[env_ids] = self.dof_vel[env_ids]
        self.last_feet_pos[env_ids] = self.feet_pos[env_ids]
        self.last_feet_contact[env_ids] = False
        self.feet_vel_filtered[env_ids] = 0.0
        self.terminate_counter[env_ids] = 0.0
        self.ball_reset_counter[env_ids] = 0.0
        self.delay_ball_pos[env_ids] = torch.randint(0, 3, (len(env_ids),), device=self.device)
        self.push_force[env_ids] = 0.0
        self.push_torque[env_ids] = 0.0
        if self.cfg.push_force_std != 0.0 or self.cfg.push_torque_std != 0.0:
            self._apply_push(env_ids)

        # 直前の位置のまま (足は前エピソードの位置) なので、ボール相対位置の原点はルートで代用する
        self.speed_hist[env_ids] = 9.0  # エピソード開始直後 (立っているだけ) を「減速」に数えない
        self.contact_streak[env_ids] = 0.0
        self.since_kick[env_ids] = -1
        self._reset_ball_root_states(env_ids)
        self._reset_ball_idx(env_ids, episode_reset=True)
        self.reset_ball[env_ids] = False
        self.terminated[env_ids] = False
        self.kicks_done[env_ids] = False
        self.ball_timeout[env_ids] = False
        return extras

    def compute(self, dt: float):
        """リセット後・観測前。終了判定で立てたボール再配置フラグを処理する。"""
        ids = self.reset_ball.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            self._reset_ball_idx(ids, episode_reset=False)
        self.reset_ball[:] = False

    # ------------------------------------------------------------------ #
    # 1 ステップ 1 回の更新
    # ------------------------------------------------------------------ #
    def ensure_updated(self):
        step = self._env.common_step_counter
        if self._last_update_step == step:
            return
        self._last_update_step = step
        self._update()

    def _refresh_robot_state(self, env_ids=None):
        r = self.robot
        ids = slice(None) if env_ids is None else env_ids
        self.base_pos[ids] = r.data.root_pos_w[ids]
        self.base_quat[ids] = r.data.root_quat_w[ids]
        self.yaw[ids] = yaw_from_quat(self.base_quat[ids])
        self.base_quat_z[ids] = _yaw_quat(self.yaw[ids])
        self.base_lin_vel[ids] = r.data.root_lin_vel_b[ids]
        self.base_ang_vel[ids] = r.data.root_ang_vel_b[ids]
        self.root_vel_w[ids] = torch.cat((r.data.root_lin_vel_w[ids], r.data.root_ang_vel_w[ids]), dim=-1)
        self.dof_vel[ids] = r.data.joint_vel[ids][:, self.joint_ids]
        self.feet_pos[ids] = r.data.body_pos_w[ids][:, self.feet_ids]
        self.feet_quat[ids] = r.data.body_quat_w[ids][:, self.feet_ids]
        self.feet_vel[ids] = r.data.body_lin_vel_w[ids][:, self.feet_ids]
        self.hip_yaw[ids] = r.data.joint_pos[ids][:, self.hip_yaw_ids]

    def _feet_edges(self) -> torch.Tensor:
        """両足裏の 4 隅のワールド位置 (N, 2, 4, 3)。"""
        n = self.num_envs
        q = self.feet_quat.unsqueeze(2).expand(n, 2, 4, 4).reshape(-1, 4)
        rel = self.feet_edge_rel.view(1, 1, 4, 3).expand(n, 2, 4, 3).reshape(-1, 3)
        return (self.feet_pos.unsqueeze(2).expand(n, 2, 4, 3).reshape(-1, 3) + quat_apply(q, rel)).view(n, 2, 4, 3)

    def _refresh_feet_state(self):
        self.feet_roll[:] = wrap_to_pi(roll_from_quat(self.feet_quat))
        self.feet_yaw[:] = wrap_to_pi(yaw_from_quat(self.feet_quat))
        edges = self._feet_edges()
        # 平面地形なので地面高さは 0
        self.feet_contact[:] = torch.any(edges[..., 2] < 0.01, dim=-1)
        self.feet_height_over_ground[:] = edges[..., 2].min(dim=-1).values
        d = torch.norm(edges[..., :2] - self.ball_pos_w[:, None, None, :2], dim=-1)
        self.l_dis[:] = d[:, 0].min(dim=-1).values
        self.r_dis[:] = d[:, 1].min(dim=-1).values

    def _compute_origin(self) -> torch.Tensor:
        """両足裏 (足首原点から sole_rel だけ下) の中点。B-Human のボール座標の原点。"""
        sole = self.feet_pos + quat_apply(self.feet_quat.reshape(-1, 4), self.sole_rel.expand(self.num_envs * 2, 3)).view(
            self.num_envs, 2, 3
        )
        return sole.mean(dim=1)

    def expected_vel(self) -> torch.Tensor:
        """B-Human BallPhysics: 距離 range を転がすのに必要な初速。"""
        return math.sqrt(2.0) * torch.sqrt(self.cfg.ball_friction * self.range)

    def _update(self):
        cfg, dt, env = self.cfg, self.dt, self._env
        n = self.num_envs
        self.ball_steps += 1
        ep_len = self.ball_steps

        # ---- 前ステップの値を退避 (B-Human では step 末尾で last_* に入れている値) ----
        self.last_feet_pos[:] = self.feet_pos
        self.last_feet_contact[:] = self.feet_contact
        self.last_feet_height_over_ground[:] = self.feet_height_over_ground
        self.last_world_pose[:] = self.current_world_pose
        self.last_relative_ball_pos[:] = self.ball_pos[:, :2]
        self.last_ball_vel_w[:] = self.ball_vel_w[:, :3]
        self.last_root_vel_w[:] = self.root_vel_w
        self.last_dof_vel[:] = self.dof_vel
        # 仕様の測定用: 当たる直前 (前の制御ステップ) の足とボール
        prev = dict(
            yaw=self.feet_yaw.clone(), pos=self.feet_pos[..., :2].clone(), quat=self.feet_quat.clone(),
            hvel=torch.norm(self.feet_vel[..., :2], dim=-1), contact=self.feet_contact.clone(),
            streak=self.contact_streak.clone(), ball=self.ball_pos_w[:, :2].clone(),
            l_dis=self.l_dis.clone(), r_dis=self.r_dis.clone(), trunk_yaw=self.yaw.clone(),
            hip_yaw=self.hip_yaw.clone(),
        )

        # ---- 物理ステップ後の状態を読む ----
        self._refresh_robot_state()
        w = cfg.filter_weight
        self.filtered_lin_vel[:] = self.base_lin_vel * w + self.filtered_lin_vel * (1.0 - w)
        self.filtered_ang_vel[:] = self.base_ang_vel * w + self.filtered_ang_vel * (1.0 - w)
        self.feet_vel_filtered[:] = self.feet_vel * 0.3 + self.feet_vel_filtered * 0.7
        self.ball_pos_w[:] = self.ball.data.root_pos_w
        self.ball_vel_w[:] = torch.cat((self.ball.data.root_lin_vel_w, self.ball.data.root_ang_vel_w), dim=-1)
        self._refresh_feet_state()

        self.ball_rel_root_prev[:] = self.ball_rel_root
        self.ball_rel_root[:] = quat_apply_inverse(self.base_quat_z, self.ball_pos_w - self.base_pos)[:, :2]
        self.buffered_global_ball_pos[:] = self.buffered_global_ball_pos.roll(shifts=-1, dims=1)
        self.buffered_global_ball_pos[:, -1] = self.ball_pos_w
        self.request_changed += 1

        # ================= _update_kick_tensors ================= #
        yaw = self.yaw
        l_dis, r_dis = self.l_dis, self.r_dis
        min_distance = torch.minimum(l_dis, r_dis)

        # 1) キックの検出
        ball_acc = torch.norm(self.ball_vel_w[:, :2] - self.last_ball_vel_w[:, :2], dim=-1)
        is_right = l_dis >= r_dis
        foot_idx = is_right.long()
        ar = torch.arange(n, device=self.device)
        sole_movement = self.feet_pos[ar, foot_idx, :2] - self.last_feet_pos[ar, foot_idx, :2]
        sole_to_ball = self.ball_pos_w[:, :2] - self.feet_pos[ar, foot_idx, :2]
        kick_sole_direction = torch.atan2(sole_movement[:, 1], sole_movement[:, 0])
        sole_movement_speed = torch.norm(sole_movement, dim=-1) / dt
        sole_to_ball_direction = torch.atan2(sole_to_ball[:, 1], sole_to_ball[:, 0])
        self.allow_ball_kick[:] = (
            (torch.abs(wrap_to_pi(kick_sole_direction - sole_to_ball_direction)) < math.pi / 2.0)
            & (ep_len > 5)
            & (sole_movement_speed > 0.1)
            & (self.feet_height_over_ground[ar, foot_idx] > 0.003)
        ) | (ep_len > 50)
        ball_still_allow_kick = (self.ball_still_counter > 10) & (ball_acc > 0.01)
        ball_moved_kick = (ball_acc > 0.1) & self.allow_ball_kick
        ball_kicked_mask = (ball_still_allow_kick | ball_moved_kick) & (min_distance < self.touch_dist)
        self.ball_kicked_counter += ((self.ball_kicked_counter > 0) | ball_kicked_mask).float()

        # キックした瞬間の蹴り足 (ボールに近い方) の足裏の一番低い点の高さ (B-Human に無い。strike_height_range 参照)
        # キックの検出は当たった次の制御ステップになるので、1 ステップ前 (当たる直前) の高さを使う
        self.kick_strike_height[:] = torch.where(
            self.ball_kicked_counter == 1, self.last_feet_height_over_ground[ar, foot_idx], self.kick_strike_height
        )
        self.kick_strike_vz[:] = torch.where(
            self.ball_kicked_counter == 1, self.feet_vel[ar, foot_idx, 2], self.kick_strike_vz
        )

        # 2) キック開始時のボール位置
        self.ball_pos_at_kick[:] = torch.where(
            (self.ball_kicked_counter == 0).unsqueeze(-1), self.ball_pos_w, self.ball_pos_at_kick
        )

        # 3.1) 両足裏の中点を原点にとる
        self.origin_pos[:] = self._compute_origin()
        self.current_world_pose[:, :2] = self.origin_pos[:, :2]
        self.current_world_pose[:, 2] = yaw
        self.ball_velocity_norm[:] = torch.norm(self.ball_vel_w[:, :3], dim=-1)

        # 3.2) キック中はボールが見えないことが多いので、一定確率でキック開始位置に凍結する
        cand = self.ball_kicked_counter == 1
        self.ball_freeze_at_kick |= cand & (torch.rand(n, device=self.device) < cfg.ball_freeze_at_kick)

        # 3.3) 観測用のボール位置 (1-3 ステップ遅れ) と速度
        src = torch.where(
            self.ball_freeze_at_kick.unsqueeze(-1),
            self.ball_pos_at_kick,
            self.buffered_global_ball_pos[ar, self.delay_ball_pos],
        )
        self.ball_pos[:] = quat_apply_inverse(self.base_quat_z, src - self.origin_pos)
        self.ball_pos[:, 2] = 0.0
        self.ball_vel[:] = quat_apply_inverse(self.base_quat_z, self.ball_vel_w[:, :3])
        self.ball_vel[:, 2] = 0.0

        # 4) キック方向の更新 (転がっているボールでも目標点を指し続ける)
        ball_to_target = self.ball_end_target - self.ball_pos_w[:, :2]
        global_direction = torch.atan2(ball_to_target[:, 1], ball_to_target[:, 0])
        self.global_direction[:] = global_direction
        self.kick_target_dir[:] = torch.where(self.ball_kicked_counter == 1, global_direction, self.kick_target_dir)
        self.direction_angle[:] = wrap_to_pi(global_direction - self.robot_direction_ref) / math.pi
        self.range[:] = torch.norm(ball_to_target, dim=-1).clip(max=cfg.kick_strong_range)
        self.direction_angle_current[:] = wrap_to_pi(
            self.direction_angle * math.pi - (yaw - self.robot_direction_ref)
        )
        self._measure_contact(prev, global_direction)

        # 5) 情報の記録とボールの減速
        ball_distance = torch.norm(self.ball_pos[:, :2], dim=-1)
        kicked = self.ball_kicked_counter > 0
        self.ball_still_counter[:] = torch.where(
            kicked,
            self.ball_still_counter,
            torch.where(self.ball_velocity_norm < 0.0001, self.ball_still_counter + 1, torch.zeros_like(ball_distance)),
        )
        self.ball_velocity_norm_at_kick[:] = torch.where(
            self.ball_kicked_counter <= 1,
            self.ball_velocity_norm,
            torch.max(self.ball_velocity_norm_at_kick, self.ball_velocity_norm),
        )
        self.ball_vel_before_kick[:] = torch.where(kicked, self.ball_vel_before_kick, self.ball_velocity_norm)
        self.ball_vel_w_before_kick[:] = torch.where(
            kicked.unsqueeze(-1), self.ball_vel_w_before_kick, self.ball_vel_w[:, :3]
        )
        slow = torch.where(
            (~kicked) & (self.ball_velocity_norm > 1.5), 0.97 - 0.2 * (ball_distance - 2).clip(min=0.0), 1.0
        )
        self.ball_vel_w[:, :3] *= slow.unsqueeze(-1)

        # 6) キック方向に回したボール位置
        sin_dir, cos_dir = torch.sin(global_direction), torch.cos(global_direction)
        dac = wrap_to_pi(self.direction_angle_current)
        kick_pose_rotation_ratio = ((torch.abs(dac) - cfg.ball_rotation_offset) / 0.4).clip(0.0, 1.0)
        ball_in_kick_angle = self.ball_pos.clone()
        ball_in_kick_angle[:, :2] = rotate_2d(self.ball_pos[:, :2], -dac)

        # 7) 左右どちらの足で蹴るかの「行き過ぎ」検出
        old_obvious = self.kick_leg_was_obvious.clone()
        by = ball_in_kick_angle[:, 1]
        self.kick_leg_was_obvious[:] = torch.where(
            (self.kick_leg_was_obvious * by > 0) | (torch.abs(by) > 0.05), torch.sign(by), torch.zeros_like(by)
        )
        self.ball_walk_target_overshoot[:] = torch.where(
            (old_obvious != self.kick_leg_was_obvious) & (self.kick_leg_was_obvious == 0) & (ep_len > 25),
            torch.full_like(by, 10.0),
            self.ball_walk_target_overshoot - 1,
        )

        # 8) キック姿勢 (キック報酬のスケールにも使う)
        ball_y_dis_ratio = (
            (torch.abs(by) - cfg.ball_y_dis_ratio[0] * cfg.kick_pose_y_off) / (cfg.ball_y_dis_ratio[1] * self.ball_r)
        ).clip(0.0, 1.0)
        self.kick_pose_translation_ratio[:] = (
            (torch.abs(torch.atan2(by, ball_in_kick_angle[:, 0])) - 0.3) / 1.2
        ).clip(0.0, 1.0)
        kick_pose_distance_ratio = ((ball_distance - cfg.ball_distance_offset) / 0.4).clip(0.0, 1.0)
        min_kick_pose_ratio = torch.max(
            ball_y_dis_ratio,
            torch.max(torch.max(kick_pose_distance_ratio, kick_pose_rotation_ratio), self.kick_pose_translation_ratio),
        )
        self.ball_close_to_kick_pose[:] = torch.clamp(
            -1.0 + 2.0 * min_kick_pose_ratio,
            min=self.ball_close_to_kick_pose - dt * 2,
            max=self.ball_close_to_kick_pose + dt * 2,
        )
        inc = torch.where(kicked, 0.0, cfg.kick_pose_increase_value)
        self.inside_kick_pose_counter[:] = torch.where(
            (self.ball_close_to_kick_pose < 1) | (self.ball_kicked_counter > 1),
            (self.inside_kick_pose_counter + inc).clip(max=1.0),
            torch.zeros_like(inc),
        )
        lowpass = torch.where(min_kick_pose_ratio == 1.0, 0.9, 0.2).unsqueeze(-1)
        self.filtered_ball_pos[:] = (1.0 - lowpass) * self.filtered_ball_pos + lowpass * ball_in_kick_angle
        kp0, kp1 = cfg.kick_pose_x
        kp = kp0 + (kp1 - kp0) * self.ball_close_to_kick_pose.clip(min=0.0)
        self.ball_kick_pose[:, 0] = self.ball_pos_w[:, 0] + cos_dir * kp
        self.ball_kick_pose[:, 1] = self.ball_pos_w[:, 1] + sin_dir * kp

        # 9.1) ランダムにボールへ速度を与える
        cand = (~kicked) & (~self.changed_ball_vel) & (min_distance > 0.35) & (min_distance < 0.55)
        ids = (cand & (torch.rand(n, device=self.device) < cfg.add_ball_vel)).nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            v = torch.zeros(len(ids), 3, device=self.device)
            v[:, 0] = torch.rand(len(ids), device=self.device) * cfg.add_ball_vel_max_velocity
            v[:, 1] = (torch.rand(len(ids), device=self.device) * cfg.add_ball_vel_max_velocity - 0.5) * 2.0
            self.ball_vel[ids] = v
            self.ball_vel_w[ids, :2] = quat_apply(self.base_quat_z[ids], v)[:, :2]
            self.changed_ball_vel[ids] = True
        # 9.2) キック方向へ速度を与える
        cand = (
            (~kicked)
            & (~self.changed_ball_vel)
            & (min_distance > 0.35)
            & (min_distance < 0.55)
            & (self.kick_pose_translation_ratio < 1)
            & (self.ball_pos[:, 0] > 0)
        )
        ids = (cand & (torch.rand(n, device=self.device) < cfg.add_ball_vel_in_direction)).nonzero(as_tuple=False)
        ids = ids.flatten()
        if len(ids) > 0:
            strength = (torch.rand(len(ids), device=self.device) + 1.0) / 2.0
            vdir = self.direction_angle_current[ids] + torch.randn(len(ids), device=self.device) * cfg.ball_vel_in_direction_std
            v = torch.zeros(len(ids), 3, device=self.device)
            v[:, 0] = torch.cos(vdir) * cfg.add_ball_vel_in_direction_max_velocity * strength
            v[:, 1] = torch.sin(vdir) * cfg.add_ball_vel_in_direction_max_velocity * strength
            self.ball_vel[ids] = v
            self.ball_vel_w[ids, :2] = quat_apply(self.base_quat_z[ids], v)[:, :2]
            self.changed_ball_vel[ids] = True
        # 9.3) ボールの速度を書き戻す
        self.ball.write_root_velocity_to_sim(self.ball_vel_w)

        # 10) キック足の yaw
        target_yaw = torch.where(l_dis < r_dis, 1.0, -1.0) * cfg.target_sole_yaw_angle
        kick_foot_yaw = torch.where(l_dis < r_dis, self.feet_yaw[:, 0], self.feet_yaw[:, 1])
        kick_sole_yaw_diff = wrap_to_pi(target_yaw - (kick_foot_yaw - global_direction))
        ratio = torch.exp(-torch.square(kick_sole_yaw_diff) / cfg.target_sole_yaw_sigma)
        self.kick_sole_yaw_ratio[:] = torch.where(
            self.ball_kicked_counter <= 1, ratio, torch.min(ratio, self.kick_sole_yaw_ratio)
        )
        # ================= end _update_kick_tensors ================= #

        # ボールの再配置要求 (キック姿勢に入りかけた所で目標だけ変える)
        cand = (self.ball_close_to_kick_pose < 1.0) & (min_distance < 0.5) & (min_distance > 0.25) & (~kicked)
        ids = (cand & (torch.rand(n, device=self.device) < cfg.reset_ball_request)).nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            self._resample_kick_ball(ids)
            self.request_changed[ids] = 0

        self._push_robots()
        self._update_odometry()
        self._check_termination(min_distance)
        self.ball_kick_direction_reward[:] = self._ball_kick_direction()
        self._update_history_and_walkoff()

    # ------------------------------------------------------------------ #
    # 仕様 (docs/bhuman_walk_kick_spec.md) の項目の測定。報酬には使わない (ログと評価スクリプト用)。
    # ------------------------------------------------------------------ #
    def _measure_contact(self, prev: dict, global_direction: torch.Tensor):
        """キックを検出したステップで、当たる直前 (1 制御ステップ前) の蹴り足の向き・ボールの位置などを記録する。"""
        ids = (self.ball_kicked_counter == 1).nonzero(as_tuple=False).flatten()
        if len(ids) == 0:
            return
        left = prev["l_dis"][ids] <= prev["r_dis"][ids]
        k = (~left).long()
        fy = prev["yaw"][ids].gather(1, k[:, None]).squeeze(1)
        rel = wrap_to_pi(fy - global_direction[ids])
        self.km_foot_yaw_abs[ids] = torch.rad2deg(rel.abs())
        self.km_foot_yaw_in[ids] = torch.rad2deg(rel) * torch.where(left, 1.0, -1.0)
        fp = prev["pos"][ids].gather(1, k[:, None, None].expand(-1, 1, 2)).squeeze(1)
        b = rotate_2d(prev["ball"][ids] - fp, -fy)
        self.km_ball_x[ids] = b[:, 0]
        self.km_ball_yin[ids] = torch.where(left, -b[:, 1], b[:, 1])
        # 止まる: 当たる前 1.5 秒の胴体の水平速度で 0.3 m/s 以下が続いた一番長い区間
        low = (self.speed_hist[ids] <= 0.3).float()
        c = torch.cumsum(low, dim=1)
        base = torch.cummax(torch.where(low > 0, torch.zeros_like(c), c), dim=1).values
        self.km_stop_s[ids] = (c - base).max(dim=1).values * self.dt
        self.km_min_speed[ids] = self.speed_hist[ids].min(dim=1).values
        sup = 1 - k
        # 軸足: 0.1 秒以上前から接地 (B-Human K1_kick_inner の MuJoCo 測定で 0.10〜0.18 秒、仕様の較正値)
        self.km_plant[ids] = prev["streak"][ids].gather(1, sup[:, None]).squeeze(1) >= round(0.1 / self.dt)
        self.km_foot_air[ids] = ~prev["contact"][ids].gather(1, k[:, None]).squeeze(1)
        self.km_foot_hvel[ids] = prev["hvel"][ids].gather(1, k[:, None]).squeeze(1)
        # 胴体の向き - キック方向、蹴り足の股関節 yaw (蹴り足が左なら +、右なら - を掛ける。足先を外へ開く向きが +)
        sgn = torch.where(left, 1.0, -1.0)
        self.km_kick_left[ids] = left
        # 蹴り足の傾き (MuJoCo の確認と同じ定義): 横 (+ = 足裏の内側が上がる、左足は +、右足は - を掛ける)、前後 (+ = つま先が下がる)
        q = prev["quat"][ids].gather(1, k[:, None, None].expand(-1, 1, 4)).squeeze(1)
        w_, x_, y_, z_ = q.unbind(-1)
        r20 = 2.0 * (x_ * z_ - w_ * y_)
        r21 = 2.0 * (y_ * z_ + w_ * x_)
        r22 = 1.0 - 2.0 * (x_ * x_ + y_ * y_)
        self.km_foot_roll_in[ids] = torch.rad2deg(torch.atan2(r21, r22)) * sgn
        self.km_foot_pitch[ids] = torch.rad2deg(-torch.asin(r20.clamp(-1.0, 1.0)))
        self.km_trunk_yaw_in[ids] = torch.rad2deg(wrap_to_pi(prev["trunk_yaw"][ids] - global_direction[ids])) * sgn
        self.km_hip_yaw_out[ids] = torch.rad2deg(prev["hip_yaw"][ids].gather(1, k[:, None]).squeeze(1)) * sgn
        self.since_kick[ids] = 0
        self.walkoff_max[ids] = 0.0

    def _update_history_and_walkoff(self):
        speed = torch.norm(self.root_vel_w[:, :2], dim=-1)
        self.speed_hist[:] = self.speed_hist.roll(shifts=-1, dims=1)
        self.speed_hist[:, -1] = speed
        self.contact_streak[:] = torch.where(self.feet_contact, self.contact_streak + 1, torch.zeros_like(self.contact_streak))
        active = self.since_kick >= 0
        self.since_kick[active] += 1
        n_avg = max(1, int(round(0.2 / self.dt)))
        after_reset = active & (self.since_kick > self.cfg.time_after_kick)
        self.walkoff_max[:] = torch.where(
            after_reset, torch.maximum(self.walkoff_max, self.speed_hist[:, -n_avg:].mean(dim=1)), self.walkoff_max
        )
        done = active & (self.since_kick >= int(round(3.0 / self.dt)))
        ids = done.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            ok = (self.walkoff_max[ids] >= 0.3).float()
            a = 1.0 - (1.0 - self.kick_stats_alpha) ** len(ids)
            self.kick_stats["kick_walkoff_rate"] += a * (ok.mean().item() - self.kick_stats["kick_walkoff_rate"])
            if self.eval_walkoff is not None:
                self.eval_walkoff.append(self.walkoff_max[ids].cpu())
            self.since_kick[ids] = -1

    # ------------------------------------------------------------------ #
    def _push_robots(self):
        """B-Human ``_push_robots``: 5 秒ごとに 2 秒間、胴体へ横方向の力とトルクを掛け続ける。"""
        if self.cfg.push_force_std == 0.0 and self.cfg.push_torque_std == 0.0:
            return
        step = self._env.common_step_counter
        interval = math.ceil(self.cfg.push_interval_s / self.dt)
        if step % interval == 0:
            self.push_force.zero_()
            self.push_force[:, 1] = torch.randn(self.num_envs, device=self.device) * self.cfg.push_force_std
            self.push_torque[:] = torch.randn(self.num_envs, 3, device=self.device) * self.cfg.push_torque_std
            self._apply_push(None)
        elif step % interval == math.ceil(self.cfg.push_duration_s / self.dt):
            self.push_force.zero_()
            self.push_torque.zero_()
            self._apply_push(None)

    def _apply_push(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        self.robot.set_external_force_and_torque(
            self.push_force[ids].unsqueeze(1),
            self.push_torque[ids].unsqueeze(1),
            body_ids=[self.trunk_id],
            env_ids=env_ids,
        )

    def _update_odometry(self):
        def local(p, pose):
            return rotate_2d(p - pose[:, :2], -pose[:, 2])

        last_l = local(self.last_feet_pos[:, 0, :2], self.last_world_pose)
        last_r = local(self.last_feet_pos[:, 1, :2], self.last_world_pose)
        cur_l = local(self.feet_pos[:, 0, :2], self.current_world_pose)
        cur_r = local(self.feet_pos[:, 1, :2], self.current_world_pose)
        odo_trans = (
            torch.where(
                (self.feet_height_over_ground[:, 0] > self.feet_height_over_ground[:, 1]).unsqueeze(-1),
                (last_l - last_r) - (cur_l - cur_r),
                (cur_r - cur_l) - (last_r - last_l),
            )
            * 0.5
        )
        odo_rot = self.current_world_pose[:, 2] - self.last_world_pose[:, 2]
        self.odo_buffer[:] = self.odo_buffer.roll(shifts=-1, dims=1)
        self.odo_buffer[:, -1] = torch.cat((torch.abs(odo_trans), torch.abs(odo_rot).unsqueeze(-1)), dim=-1)

    def _check_termination(self, min_distance: torch.Tensor):
        cfg, n = self.cfg, self.num_envs
        self.reset_ball[:] = False
        self.terminated[:] = self.root_vel_w.square().sum(dim=-1) > cfg.terminate_vel
        self.terminate_counter += ((self.terminate_counter > 0) | (self.base_pos[:, 2] < cfg.terminate_height)).float()
        self.terminated |= self.terminate_counter > cfg.terminate_time / self.dt

        self.reset_ball |= self.ball_kicked_counter >= cfg.time_after_kick

        rbr = cfg.random_ball_reset
        thr = torch.full((n,), rbr[0], device=self.device)
        thr[self.flag_allow_deviation] = rbr[1]
        thr[self.flag_allow_deviation & self.flag_is_strong_kick] = rbr[2]
        cand = torch.rand(n, device=self.device) < thr
        mask = (
            cand
            & (self.ball_reset_counter < cfg.episode_number_of_kicks)
            & (
                ((self.ball_close_to_kick_pose < 1.0) & (min_distance < cfg.feet_edge_pos[0][0] + self.ball_r))
                | (self.ball_kicked_counter > 1)
            )
        )
        self.reset_ball |= mask
        dcand = (
            (self.ball_reset_counter < cfg.episode_number_of_kicks)
            & (min_distance < cfg.distance_reset[1])
            & (min_distance > cfg.distance_reset[0])
        )
        self.reset_ball |= dcand & (torch.rand(n, device=self.device) < cfg.random_ball_distance_reset)

        self.ball_reset_counter += (self.reset_ball & (self.ball_kicked_counter > 0)).float()
        self._record_kick_stats(self.reset_ball & (self.ball_kicked_counter > 0))
        self.kicks_done[:] = (self.ball_reset_counter > cfg.episode_number_of_kicks) & self.reset_ball
        self.ball_timeout[:] = self.ball_steps > cfg.ball_timeout_s / self.dt
        self.metrics["kicks"][:] = self.ball_reset_counter

    def _record_kick_stats(self, done: torch.Tensor):
        """ボールを片付ける (キックから time_after_kick 経過) 時点で、そのキックの出来を集計する。"""
        ids = done.nonzero(as_tuple=False).flatten()
        k = len(ids)
        if k == 0:
            return
        self.kick_total += k
        speed = self.ball_velocity_norm_at_kick[ids]
        disp = self.ball_pos_w[ids, :2] - self.ball_pos_at_kick[ids, :2]
        err = torch.abs(wrap_to_pi(torch.atan2(disp[:, 1], disp[:, 0]) - self.kick_target_dir[ids]))
        strong = self.flag_is_strong_kick[ids]
        vals = {
            "kick_speed": speed,
            "kick_speed_strong": speed[strong],
            "kick_speed_err_normal": torch.abs(speed - self.expected_vel()[ids])[~strong],
            "kick_dir_err_deg": torch.rad2deg(err),
            "kick_dir_ok_rate": (err < math.radians(15.0)).float(),
            "kick_strike_height": self.kick_strike_height[ids],
            "kick_foot_yaw_in_deg": self.km_foot_yaw_in[ids],
            "kick_yaw_ok_rate": ((self.km_foot_yaw_abs[ids] - 90.0).abs() <= 25.0).float(),
            "kick_inside_rate": (self.km_ball_yin[ids] > self.km_ball_x[ids].abs()).float(),
            "kick_stop_rate": (self.km_min_speed[ids] <= 0.6).float(),
            "kick_plant_rate": self.km_plant[ids].float(),
            "kick_swing_rate": (self.km_foot_air[ids] & (self.km_foot_hvel[ids] >= 1.0)).float(),
            "kick_trunk_yaw_in_deg": self.km_trunk_yaw_in[ids],
            "kick_hip_yaw_out_deg": self.km_hip_yaw_out[ids],
            "kick_strike_height_L": self.kick_strike_height[ids][self.km_kick_left[ids]],
            "kick_strike_height_R": self.kick_strike_height[ids][~self.km_kick_left[ids]],
        }
        if self.eval_log is not None:  # 評価スクリプト (eval_bhuman_walk_kick.py) 用の生データ
            self.eval_log.append({
                "speed": speed.cpu(), "target_speed": self.expected_vel()[ids].cpu(), "strong": strong.cpu(),
                "dir_err_deg": torch.rad2deg(err).cpu(),
                "kick_dir_rel_deg": torch.rad2deg(wrap_to_pi(self.kick_target_dir[ids] - self.robot_direction_ref[ids])).cpu(),
                "ball_steps": self.ball_steps[ids].cpu(),
                "disp_m": torch.norm(disp, dim=-1).cpu(),
                "kick_counter": self.ball_kicked_counter[ids].cpu(),
                "hspeed_now": torch.norm(self.ball_vel_w[ids, :2], dim=-1).cpu(),
                "strike_h": self.kick_strike_height[ids].cpu(),
                "strike_vz": self.kick_strike_vz[ids].cpu(),
                "foot_yaw_abs": self.km_foot_yaw_abs[ids].cpu(),
                "foot_yaw_in": self.km_foot_yaw_in[ids].cpu(),
                "ball_x_foot": self.km_ball_x[ids].cpu(),
                "ball_yin_foot": self.km_ball_yin[ids].cpu(),
                "stop_s": self.km_stop_s[ids].cpu(),
                "min_speed": self.km_min_speed[ids].cpu(),
                "plant": self.km_plant[ids].cpu(),
                "foot_air": self.km_foot_air[ids].cpu(),
                "foot_hvel": self.km_foot_hvel[ids].cpu(),
                "trunk_yaw_in": self.km_trunk_yaw_in[ids].cpu(),
                "kick_left": self.km_kick_left[ids].cpu(),
                "foot_roll_in": self.km_foot_roll_in[ids].cpu(),
                "foot_pitch": self.km_foot_pitch[ids].cpu(),
                "hip_yaw_out": self.km_hip_yaw_out[ids].cpu(),
            })
        for key, v in vals.items():
            if v.numel() == 0:
                continue
            a = 1.0 - (1.0 - self.kick_stats_alpha) ** v.numel()
            self.kick_stats[key] += a * (v.mean().item() - self.kick_stats[key])

    def _ball_kick_direction(self) -> torch.Tensor:
        """B-Human ``_helper_ball_kick_direction``。他のボール報酬もこの値を使う。"""
        cfg = self.cfg
        kc = self.ball_kicked_counter
        clip_factor = torch.where(kc - cfg.ball_kicked_start_counter > cfg.ball_kicked_clipped_counter, 0.0, 1.0)
        time_ratio = 1.0 - ((kc - cfg.ball_kicked_start_counter) / cfg.ball_kicked_max_counter).clip(0.0, 1.0)
        ratio = (kc > 0).float()
        angle_diff = wrap_to_pi(self.direction_angle_current - torch.atan2(self.ball_vel[:, 1], self.ball_vel[:, 0]))

        sigma = torch.full_like(kc, cfg.kick_direction_sigma_range)
        sigma[self.flag_is_inaccurate_kick > 0.5] = cfg.kick_direction_sigma_just_hit

        kick_pose_counter_factor = torch.square(self.inside_kick_pose_counter)
        sole_yaw_factor = self.kick_sole_yaw_ratio * 0.5 + 0.5
        kick_pose_counter_factor[self.flag_allow_deviation] = 1.0
        sole_yaw_factor[self.flag_allow_deviation] = 1.0
        if cfg.disable_sole_yaw_direction_penalty:
            sole_yaw_factor[:] = 1.0

        reward = (torch.exp(-torch.square(angle_diff) / sigma) - 0.5) * 2.0 * ratio * time_ratio * clip_factor
        height_ratio = (
            (self.base_pos[:, 2] - cfg.base_height_scaling[0]) / (cfg.base_height_scaling[1] - cfg.base_height_scaling[0])
        ).clip(0.0, 1.0)
        ball_vel_negative_scale = 1.0 - 0.9 * self.ball_vel_before_kick.clip(max=2.0) / 2.0
        negative_scale = 0.1 + 0.9 * ((self.request_changed - 0.25 / self.dt) / (0.5 / self.dt)).clip(0.0, 1.0)
        positive = reward * height_ratio * kick_pose_counter_factor * sole_yaw_factor
        if cfg.strike_height_range is not None:
            # B-Human に無い (s4d から): 足裏の一番低い点がボールの中心より上でボールに当たると (上から踏みつける
            # ように当たると)、PhysX では強く飛ぶが MuJoCo ではボールを床に押し付けて引きずる・倒れる。
            # 低い位置 (ボールの中心かそれより下) で当てたキックだけを褒める。
            lo, hi = cfg.strike_height_range
            positive = positive * (1.0 - ((self.kick_strike_height - lo) / (hi - lo)).clip(0.0, 1.0))
        return torch.where(
            reward < 0,
            reward * cfg.kick_direction_negative_factor * ball_vel_negative_scale * negative_scale,
            positive,
        )

    # ------------------------------------------------------------------ #
    # ボールのリセット
    # ------------------------------------------------------------------ #
    def _resample_input_flags(self, ids: torch.Tensor):
        k = len(ids)
        self.flag_is_strong_kick[ids] = torch.rand(k, device=self.device) < self.cfg.strong_kick_prob
        self.flag_is_inaccurate_kick[ids] = (torch.rand(k, device=self.device) < 0.3).float()
        self.flag_allow_deviation[ids] = torch.rand(k, device=self.device) < 0.5
        if self.cfg.disable_deviation_flag:
            self.flag_allow_deviation[ids] = False
        if self.cfg.disable_extra_deviation_flag:
            self.flag_is_inaccurate_kick[ids] = 0.0

    def _sample_range(self, ids: torch.Tensor) -> torch.Tensor:
        lo, hi = self.cfg.kick_normal_range
        r = uniform_log((len(ids),), lo, hi, self.device)
        return torch.where(self.flag_is_strong_kick[ids], torch.full_like(r, self.cfg.kick_strong_range), r)

    def _reset_ball_idx(self, ids: torch.Tensor, episode_reset: bool):
        """B-Human ``_reset_ball_idx``: 新しいキック課題を出す。"""
        cfg = self.cfg
        self._resample_input_flags(ids)
        self.ball_close_to_kick_pose[ids] = 1.0
        self.ball_kicked_counter[ids] = 0.0
        self.ball_vel_before_kick[ids] = 0.0
        self.ball_vel_w_before_kick[ids] = 0.0
        self.ball_still_counter[ids] = 0.0
        self.inside_kick_pose_counter[ids] = 0.0
        self.changed_ball_vel[ids] = False
        if not episode_reset:
            replace = ids[torch.rand(len(ids), device=self.device) > cfg.keep_old_ball_when_reset]
            if len(replace) > 0:
                self._reset_ball_root_states(replace)
        self._resample_kick_ball(ids, use_root_origin=episode_reset)
        # B-Human はここで episode_length_buf を 0 に戻すが、歩行の位相が飛ぶので代わりに ball_steps を戻す
        self.ball_steps[ids] = 0
        self.allow_ball_kick[ids] = False
        self.ball_walk_target_overshoot[ids] = 0.0
        self.kick_leg_was_obvious[ids] = 0.0
        self.kick_sole_yaw_ratio[ids] = 0.0
        self.odo_buffer[ids] = 0.0
        self.ball_freeze_at_kick[ids] = False
        self.odo_noise_factor[ids, 0] = torch.randn(len(ids), device=self.device) * cfg.ball_pos_odo_noise_std[0]
        self.odo_noise_factor[ids, 1] = torch.randn(len(ids), device=self.device) * cfg.ball_pos_odo_noise_std[1]

    def _reset_ball_root_states(self, ids: torch.Tensor):
        """B-Human ``_reset_ball_root_states``: ロボットの周囲にボールを置く。"""
        cfg, k, dev = self.cfg, len(ids), self.device
        robot_xy = self.robot.data.root_pos_w[ids, :2]
        robot_yaw = yaw_from_quat(self.robot.data.root_quat_w[ids])
        moving = self.moving_ball_env[ids]
        x = torch.where(
            moving, uniform_log((k,), *cfg.init_ball_pos_far_x, dev), uniform_log((k,), *cfg.init_ball_pos_x, dev)
        )
        rot = torch.randint(-180, 180, (k,), device=dev).float() / 180.0 * math.pi
        rel = torch.stack((x * torch.cos(rot), x * torch.sin(rot)), dim=-1)
        pos = torch.zeros(k, 3, device=dev)
        pos[:, :2] = robot_xy + rotate_2d(rel, robot_yaw)
        pos[:, 2] = cfg.ball_radius + cfg.ball_radius_margin + 0.01

        vel = torch.zeros(k, 6, device=dev)
        vel[:, 0] = uniform_log((k,), *cfg.init_ball_vel, dev)
        vel[:, 1] = uniform_log((k,), *cfg.init_ball_vel, dev)
        vel[~moving, :2] = 0.0
        # 転がる側の 5% はロボットへまっすぐ向かわせる
        towards = moving & (torch.rand(k, device=dev) < 0.05)
        to_robot = robot_xy - pos[:, :2]
        norm = torch.norm(to_robot, dim=-1, keepdim=True)
        speed = torch.norm(vel[:, :2], dim=-1, keepdim=True)
        vel[:, :2] = torch.where(towards.unsqueeze(-1) & (norm > 0), to_robot / norm.clamp(min=1e-6) * speed, vel[:, :2])

        # 一部は止まったボールをすぐ近くに置き、ボールの周りを回り込む歩き方を覚えさせる
        close = torch.rand(k, device=dev) < cfg.force_ball_close
        close_dist = torch.where(moving, cfg.init_ball_pos_far_x[0], cfg.init_ball_pos_x[0])
        vec = pos[:, :2] - robot_xy
        vec = vec / torch.norm(vec, dim=-1, keepdim=True).clamp(min=1e-6)
        pos[:, :2] = torch.where(close.unsqueeze(-1), robot_xy + vec * close_dist.unsqueeze(-1), pos[:, :2])
        vel[close] = 0.0

        quat = torch.zeros(k, 4, device=dev)
        quat[:, 0] = 1.0
        self.ball.write_root_pose_to_sim(torch.cat((pos, quat), dim=-1), env_ids=ids)
        self.ball.write_root_velocity_to_sim(vel, env_ids=ids)
        self.ball_pos_w[ids] = pos
        self.ball_vel_w[ids] = vel
        rel = rotate_2d(pos[:, :2] - robot_xy, -robot_yaw)
        self.ball_rel_root[ids] = rel
        self.ball_rel_root_prev[ids] = rel
        self.last_ball_vel_w[ids] = vel[:, :3]
        self.buffered_global_ball_pos[ids] = pos.unsqueeze(1).expand(-1, self.buffered_global_ball_pos.shape[1], -1)

    def _resample_kick_ball(self, ids: torch.Tensor, use_root_origin: bool = False):
        """B-Human ``_resample_kick_ball``: キック方向と距離を引き直す。"""
        cfg = self.cfg
        yaw = yaw_from_quat(self.robot.data.root_quat_w[ids])
        quat_z = _yaw_quat(yaw)
        if use_root_origin:
            origin = self.robot.data.root_pos_w[ids].clone()
        else:
            origin = self.origin_pos[ids].clone()
            if (origin == 0).all():
                origin = self.robot.data.root_pos_w[ids].clone()
        self.origin_pos[ids] = origin
        self.current_world_pose[ids, :2] = origin[:, :2]
        self.current_world_pose[ids, 2] = yaw
        self.last_world_pose[ids] = self.current_world_pose[ids]

        ball_w = self.ball_pos_w[ids]
        bp = quat_apply_inverse(quat_z, ball_w - origin)
        bp[:, 2] = 0.0
        self.ball_pos[ids] = bp
        self.last_relative_ball_pos[ids] = bp[:, :2]
        bv = quat_apply_inverse(quat_z, self.ball_vel_w[ids, :3])
        bv[:, 2] = 0.0
        self.ball_vel[ids] = bv

        self.direction_angle[ids] = torch.rand(len(ids), device=self.device) * 2.0 - 1.0
        self.robot_direction_ref[ids] = yaw
        dac = wrap_to_pi(self.direction_angle[ids] * math.pi)
        self.direction_angle_current[ids] = dac
        self.range[ids] = self._sample_range(ids)

        gdir = wrap_to_pi(yaw + self.direction_angle[ids] * math.pi)
        self.global_direction[ids] = gdir
        self.ball_end_target[ids, 0] = ball_w[:, 0] + torch.cos(gdir) * self.range[ids]
        self.ball_end_target[ids, 1] = ball_w[:, 1] + torch.sin(gdir) * self.range[ids]
        self.ball_kick_pose[ids, 0] = ball_w[:, 0] + torch.cos(gdir) * cfg.kick_pose_x[1]
        self.ball_kick_pose[ids, 1] = ball_w[:, 1] + torch.sin(gdir) * cfg.kick_pose_x[1]
        self.ball_kick_pose[ids, 2] = 0.0

        self.ball_close_to_kick_pose[ids] = 1.0
        self.ball_kicked_counter[ids] = 0.0
        self.inside_kick_pose_counter[ids] = 0.0
        self.ball_still_counter[ids] = 0.0
        self.ball_vel_before_kick[ids] = 0.0
        fb = bp.clone()
        fb[:, :2] = rotate_2d(bp[:, :2], -dac)
        self.filtered_ball_pos[ids] = fb
        self.last_ball_vel_w[ids] = self.ball_vel_w[ids, :3]
        self._obs_cache_step = -1

    # ------------------------------------------------------------------ #
    # 観測ノイズ (B-Human ``_compute_observations``)
    # ------------------------------------------------------------------ #
    def obs_cache(self):
        """ボール観測のノイズを 1 ステップ 1 回だけ引く (policy と critic で共有する)。"""
        self.ensure_updated()
        step = self._env.common_step_counter
        if self._obs_cache_step == step:
            return
        self._obs_cache_step = step
        cfg, n, dev = self.cfg, self.num_envs, self.device
        if not cfg.obs_noise:
            self.obs_ball_pos[:] = self.ball_pos[:, :2]
            self.obs_last_ball_pos[:] = self.last_relative_ball_pos
            self.obs_direction_angle[:] = self.direction_angle_current
            self.obs_ball_vel[:] = self.ball_vel[:, :2]
            self.obs_expected_vel[:] = self.expected_vel()
            return
        lo, hi = cfg.ball_pos_noise_scale_range
        rx = ((torch.abs(self.ball_pos[:, 0]) - lo) / (hi - lo)).clip(0.0, 1.0)
        ry = ((torch.abs(self.ball_pos[:, 1]) - lo) / (hi - lo)).clip(0.0, 1.0)
        nx, ny = cfg.ball_pos_noise_x, cfg.ball_pos_noise_y
        std = torch.stack(((nx[1] - nx[0]) * rx + nx[0], (ny[1] - ny[0]) * ry + ny[0]), dim=-1)
        noise = torch.randn(n, 2, device=dev) * std
        odo = self.odo_buffer.sum(dim=1)
        noise += self.odo_noise_factor * odo[:, :2]
        noise += 0.2 * (self.odo_noise_factor[:, 1] * odo[:, 2]).unsqueeze(-1)
        self.obs_ball_pos[:] = self.ball_pos[:, :2] + noise
        self.obs_last_ball_pos[:] = self.last_relative_ball_pos + noise
        self.obs_direction_angle[:] = self.direction_angle_current + torch.randn(n, device=dev) * cfg.ball_direction_noise_std
        bv = self.ball_vel[:, :2] + torch.randn(n, 2, device=dev) * cfg.ball_vel_noise_std
        s_lo, s_hi = cfg.ball_vel_scale_noise
        self.obs_ball_vel[:] = bv * (s_lo + (s_hi - s_lo) * torch.rand(n, 2, device=dev))
        self.obs_expected_vel[:] = self.expected_vel() + torch.randn(n, device=dev) * cfg.ball_range_noise_std


@configclass
class BHumanKickCommandCfg(CommandTermCfg):
    """B-Human ``K1_Ball.yaml`` のうち、ボール・キック・リセットに関わるパラメータ。

    既定値は README の「1) Train with default parameters」(学習の最初の段) のもの。
    後段の値は :mod:`..bhuman_kick_env_cfg` の Stage クラスで上書きする。
    """

    class_type: type = BHumanKickCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    asset_name: str = "robot"
    ball_name: str = "soccer_ball"
    base_body_name: str = "Trunk"
    foot_body_names: tuple[str, ...] = ("left_foot_link", "right_foot_link")
    joint_names: tuple[str, ...] = MISSING

    # --- asset ---
    ball_radius: float = 0.11
    ball_radius_margin: float = 0.01  # B-Human の ball_radius DR の上限。判定距離にだけ使う
    feet_edge_pos: tuple = (
        (0.1185, 0.035, -0.038),
        (0.1185, -0.035, -0.038),
        (-0.0665, 0.035, -0.038),
        (-0.0665, -0.035, -0.038),
    )
    feet_sole_pos: tuple = (0.0, 0.0, -0.038)
    ball_pos_buffer_size: int = 4
    filter_weight: float = 0.1

    # --- kick target (ball:) ---
    kick_strong_range: float = 20.0
    kick_normal_range: tuple[float, float] = (0.5, 20.0)
    strong_kick_prob: float = 0.5

    # --- commands: ---
    random_ball_reset: tuple[float, float, float] = (0.0001, 0.0001, 0.0001)
    random_ball_distance_reset: float = 0.00005
    ball_freeze_at_kick: float = 0.2
    distance_reset: tuple[float, float] = (0.3, 0.5)
    force_ball_close: float = 0.1
    keep_old_ball_when_reset: float = 0.2
    reset_ball_request: float = 0.0001
    add_ball_vel: float = 0.00001
    add_ball_vel_max_velocity: float = 0.5
    add_ball_vel_in_direction: float = 0.00001
    add_ball_vel_in_direction_max_velocity: float = 0.5
    ball_vel_in_direction_std: float = 0.5

    # --- randomization: (ball placement) ---
    init_ball_pos_x: tuple[float, float] = (0.4, 1.5)
    init_ball_pos_far_x: tuple[float, float] = (0.6, 1.5)
    init_ball_vel: tuple[float, float] = (0.0, 7.5)

    # --- push (randomization: push_*) ---
    push_interval_s: float = 5.0
    push_duration_s: float = 2.0
    push_force_std: float = 0.0
    push_torque_std: float = 0.0

    # --- termination (rewards:) ---
    episode_number_of_kicks: int = 3
    ball_timeout_s: float = 30.0
    """1 個のボールにこれ以上かかったらエピソードを終える (B-Human: episode_length_s)。"""
    terminate_height: float = 0.30
    terminate_vel: float = 400.0
    terminate_time: float = 0.1
    swing_period: float = 0.2

    # --- rewards: ball_parameters ---
    time_after_kick: int = 20
    kick_pose_y_off: float = 0.06
    ball_y_dis_ratio: tuple[float, float] = (2.0, 2.0)
    ball_kicked_start_counter: int = 1
    ball_kicked_max_counter: int = 50
    ball_kicked_clipped_counter: int = 10
    kick_direction_sigma_range: float = 1.0
    kick_direction_sigma_just_hit: float = 1.0
    ball_friction: float = 0.3
    kick_pose_x: tuple[float, float] = (0.0, -0.35)
    ball_distance_offset: float = 0.3
    ball_rotation_offset: float = 0.1
    base_height_scaling: tuple[float, float] = (0.45, 0.49)
    ball_avoidance_range: float = 0.55
    kick_pose_increase_value: float = 0.025
    kick_direction_negative_factor: float = 0.1
    target_sole_yaw_angle: float = 0.0
    target_sole_yaw_sigma: float = 1.0
    disable_sole_yaw_direction_penalty: bool = False
    disable_deviation_flag: bool = True
    disable_extra_deviation_flag: bool = True
    strike_height_range: tuple[float, float] | None = None
    """(B-Human に無い) キックした瞬間の蹴り足の足裏の一番低い点の高さが lo 以下なら 1、hi 以上なら 0 を
    キック方向の報酬 (正のとき) に掛ける。None で無効。"""

    # --- noise: (ball observation) ---
    obs_noise: bool = True
    ball_pos_noise_scale_range: tuple[float, float] = (0.5, 10.0)
    ball_pos_noise_x: tuple[float, float] = (0.03, 2.0)
    ball_pos_noise_y: tuple[float, float] = (0.03, 0.35)
    ball_pos_odo_noise_std: tuple[float, float] = (0.01, 0.01)
    ball_vel_noise_std: float = 0.01
    ball_vel_scale_noise: tuple[float, float] = (0.0, 2.0)
    ball_range_noise_std: float = 0.05
    ball_direction_noise_std: float = 0.01
