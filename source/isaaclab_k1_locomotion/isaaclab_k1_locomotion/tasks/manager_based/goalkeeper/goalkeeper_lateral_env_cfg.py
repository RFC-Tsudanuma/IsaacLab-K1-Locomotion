# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""横移動特化の歩行ポリシー (階層型ゴールキーパーの **下位**) の環境定義。

``goalkeeper_direct_env_cfg.py`` の :class:`K1GKDirectStage1EnvCfg` を継承した
**別タスク**。現行の下位 (``k1_gk_direct_stage1/2026-07-28_17-13-15``、以下 07-28) は
実機にデプロイ済みで良好に動いているため、そちらは一切変更せずに残す。

07-28 の実測 (eval_gk_direct_lateral.py):
    指令 vy  0.5 / 0.9 / 1.2 / 1.3 / 1.5
    実測     0.460 / 0.878 / 1.182 / 1.278 / 1.474 m/s (誤差 2%、極めて良好)
    足上げ   0.048 / 0.042 / 0.038〜0.043 / 0.041 / 0.041 m
    yaw ドリフト 9.7〜12.4°/s、前後ドリフト -0.10 m/s、立ち上がり 0.6s、転倒 0.16%

**定常速度の追従精度に改善余地はほぼ無い**ので、本タスクが狙うのは過渡と姿勢:

1. **立ち上がり (最優先)**: セーブに必要な横移動は 0.3〜0.8m が中心で、その帯域は
   まるごと加速区間に入る。2.6m 横断は 1.278 m/s で 2.35s、1.6 m/s に上げても 1.99s
   (定常 25% 増で 15% 短縮) にしかならない。立ち上がり 0.6s → 0.4s 台の方が効く。
   → :func:`~.mdp.rewards.onset_speed_bonus` (過渡だけを線形評価) と
     :func:`~.mdp.rewards.onset_action_rate_l2` (過渡だけ平滑ペナルティを緩める)。
2. **yaw ドリフト**: 約 10°/s で円を描く。原因は ①対称性の学習残差
   (07-28 は data augmentation 無効 / mirror 係数 0.5 の世代) と ②角速度しか見ておらず
   heading を保持する項が無いこと。→ 対称性は現行設定 (aug 有効 / 係数 2.0) を継承し、
   :func:`~.mdp.rewards.heading_hold` で積分 yaw 誤差を直接罰する。
3. **足上げ**: 速度域で 4cm 台。人工芝のパイル (20〜30mm) を考えると薄い。
   → :func:`~.mdp.rewards.foot_clearance_relative` (支持脚基準 + 位相整合) で
     跳躍の抜け道を塞いだうえで目標を 7cm へ。
4. **後退ドリフト**: -0.10 m/s。→ :func:`~.mdp.rewards.track_lin_vel_x_exp` を薄く追加。

**観測レイアウトは 59 次元のまま一切変更していない** (継承のみ)。階層型の上位
ポリシーがそのまま読める。アクション・ネットワーク形状・速度指令レンジも同じ。

学習: ``scripts/rsl_rl/train_gk_lateral.sh`` (既定で 07-28 から ``--resume``)。
評価: ``scripts/rsl_rl/eval_gk_direct_lateral.py`` を 07-28 と同じコマンドで。
ベースライン記録: ``docs/baselines/gk_direct_stage1_2026-07-28.md``。
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..locomotion.rough_env_cfg import (
    _COMMAND_THRESHOLD,
    _PHASE_FREQ,
    _STANCE_RATIO,
)

from .goalkeeper_direct_env_cfg import LATERAL_TARGET_SPEED, K1GKDirectStage1EnvCfg
from .mdp.events import reset_lateral_buffers
from .mdp.rewards import (
    flight_phase,
    foot_clearance_relative,
    heading_hold,
    onset_action_rate_l2,
    onset_reach_bonus,
    onset_speed_bonus,
    track_lin_vel_x_exp,
)

# 立ち上がりを評価する窓 [s]。目標 (0.4s 台) より少し長く取り、
# 「窓の中でどれだけ速く定常に届いたか」に勾配が残るようにする。
ONSET_WINDOW_S: float = 0.8
# 「コマンドが変わった」とみなす線速度指令の変化量 [m/s]。
ONSET_CHANGE_TOL: float = 0.4
# 遊脚の目標持ち上げ量 [m] (**絶対高さではない**)。接地時の足リンク原点は地面から
# 0.035m なので、絶対高さ表記なら 0.095m。
# ★ 2026-08-15: 0.07 → 0.06 に引き下げ。1 本目 (11200 iter) は実測 7.6cm と要件 5cm を
#   大きく超えた一方、**その要求が歩容を跳躍に変え、立ち上がり (0.565→0.719s) と
#   横速度 (1.278→1.151) を同時に壊した**。足上げは優先度 3 位なので、1 位の立ち上がりと
#   「速度を落とさない」制約を優先して要求を下げる。07-28 と同じ 6cm 相当だが、
#   測り方 (支持脚基準) と位相 (get_phase_freq 追従) を直してあるぶん実測値は上に出るはず。
TARGET_FOOT_LIFT: float = 0.06


@configclass
class K1GKLateralEnvCfg(K1GKDirectStage1EnvCfg):
    """横移動特化の下位ポリシー (07-28 の後継候補)。"""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------
        # 1. コマンド: 立ち上がりを学習させるための分布
        # ------------------------------------------------------------------
        # ★ heading_command を切る (07-28 は True)。
        #   True だと wz 指令が「heading 誤差 × 0.5」の **フィードバック** になるため、
        #   学習中は向きのドリフトが常に閉ループで潰されて表に出ない。ところが実機・
        #   eval・階層型の上位はいずれも wz を **開ループ** で与えるので、そこで初めて
        #   ドリフトが積分されて円を描く。デプロイ時の使われ方に合わせて開ループで学習し、
        #   向きの維持は heading_hold で直接評価する。
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        # heading レンジは heading_command=True のときしか使われない。残しておくと
        # コマンド項が起動時に警告を出すので明示的に外す。
        self.commands.base_velocity.ranges.heading = None
        # 再サンプルを短く (07-28 は 10s → カリキュラムで 0.5〜7.0s)。
        # 1 エピソード 20s あたりの「立ち上がり」回数がそのまま過渡の学習量になる。
        self.commands.base_velocity.resampling_time_range = (1.5, 4.0)
        # 停止指令の割合を上げる (07-28 は 0.02)。停止 → 全開の遷移が
        # セーブ開始時の状況そのもので、最も鍛えたい過渡。
        self.commands.base_velocity.rel_standing_envs = 0.15
        # カリキュラム側の再サンプル時間上書きも短めに揃える (既定は 14000 step で
        # (0.5, 7.0) に変更。上限 7s は過渡の頻度を下げてしまう)。
        self.curriculum.command_resampling_time_range.params["resampling_time_range"] = (0.8, 4.0)

        # 速度レンジ・カリキュラムは 07-28 のまま (vx ±1.0 / vy ±1.3)。
        # 実機で検証済みのインターフェースなので広げない。

        # ------------------------------------------------------------------
        # 2. 立ち上がり (最優先)
        # ------------------------------------------------------------------
        self.rewards.onset_speed = RewTerm(
            func=onset_speed_bonus,
            weight=4.0,
            params={
                "command_name": "base_velocity",
                "v_ref": LATERAL_TARGET_SPEED,
                "min_cmd": 0.6,
                "onset_s": ONSET_WINDOW_S,
                "change_tol": ONSET_CHANGE_TOL,
            },
        )
        # ★ 2026-08-15 追加。上の線形報酬だけでは **立ち上がりが 07-28 より遅くなった**
        #   (t90 @cmd1.3 = 0.565s → 0.719s)。線形は序盤に勾配が集中し、定常の 90% 付近
        #   (= 一番時間を食っている「最後の詰め」) にほとんど圧がかからないため。
        #   本項は到達時刻そのものを目的関数にする (早く届くほど大きい一回限りの報酬)。
        self.rewards.onset_reach = RewTerm(
            func=onset_reach_bonus,
            weight=8.0,
            params={
                "command_name": "base_velocity",
                "reach_frac": 0.9,
                "min_cmd": 0.6,
                "onset_s": ONSET_WINDOW_S,
                "change_tol": ONSET_CHANGE_TOL,
            },
        )
        # 過渡だけ平滑ペナルティを緩める。定常区間の倍率は 07-28 と同一。
        # ★ 実機で動きがガタついたら最初にここを onset_scale=1.0 に戻す。
        self.rewards.action_rate_l2 = RewTerm(
            func=onset_action_rate_l2,
            weight=-0.4,
            params={
                "command_name": "base_velocity",
                "cmd_threshold": _COMMAND_THRESHOLD,
                "stand_still_scale": 3.0,
                "onset_s": ONSET_WINDOW_S,
                "onset_scale": 0.4,
                "change_tol": ONSET_CHANGE_TOL,
            },
        )

        # ------------------------------------------------------------------
        # 3. yaw ドリフト
        # ------------------------------------------------------------------
        self.rewards.heading_hold = RewTerm(
            func=heading_hold,
            weight=-3.0,
            params={
                "command_name": "base_velocity",
                "max_err": 0.6,
                "change_tol": ONSET_CHANGE_TOL,
            },
        )
        # ★ 2026-08-15: 3.5 に下げたのは失敗。heading_hold が肩代わりする想定だったが、
        #   1 本目の実測で yaw ドリフトは平均 13.3 → 17.1°/s と **悪化**した。
        #   07-28 と同じ 5.0 に戻し、heading_hold と併用する
        #   (角速度追従 = 指令への即応、heading_hold = 溜まったズレ、で役割が違う)。
        #   ※ 7.0 まで上げるのは 2026-07-29 に失敗済み (横速度 1.182 → 0.628)。
        self.rewards.track_ang_vel_z_exp.weight = 5.0

        # ------------------------------------------------------------------
        # 4. 足上げ
        # ------------------------------------------------------------------
        # 支持脚基準 + 位相整合の版に差し替える (絶対高さ版は跳躍で達成できてしまう)。
        self.rewards.foot_clearance = RewTerm(
            func=foot_clearance_relative,
            weight=2.5,
            params={
                "command_name": "base_velocity",
                "target_lift": TARGET_FOOT_LIFT,
                "phase_freq": _PHASE_FREQ,
                "stance_ratio": _STANCE_RATIO,
                "cmd_threshold": _COMMAND_THRESHOLD,
            },
        )
        # ★ 2026-08-15 (1 本目の実測で修正): -1.5 に緩めたが **跳躍が出た** ので
        #   07-28 と同じ -2.5 に戻す。
        #   緩めた根拠は「支持脚基準にしたので跳んでも値が増えない」だったが、これは誤り。
        #   塞げるのは「両足が同じだけ上がる (体ごと持ち上げ)」だけで、**空中で片脚を畳んで
        #   もう片脚を垂らせば相対高さは稼げる**。接地の拘束が無いぶん空中の方が楽なので、
        #   抜け道は狭まっただけで塞がっていなかった。
        self.rewards.lin_vel_z_l2.weight = -2.5
        # 両足同時浮き = 跳躍そのものを直接罰する。lin_vel_z_l2 は上下動全般を罰するので
        # 強めると歩行に必要な重心の上下動や着地の衝撃吸収まで巻き添えにするが、本項は
        # 「両足浮き」だけを見るので **片脚を高く上げること自体は無罰**。
        # ★ 2026-07-29 に絶対高さ版 + 位相ズレ状態で -2.0 を試したときは、跳躍は完全に
        #   止まった代わりに足上げが 2.6〜3.3cm まで落ちた。今回は測り方 (支持脚基準) と
        #   位相 (get_phase_freq 追従) を直してあるので同じ結果になるとは限らないが、
        #   **足上げが 07-28 の 4cm 台を割ったら真っ先にここを疑うこと**。
        #   その場合の次の一手は本項を下げるのではなく、足上げを縛っている
        #   ankle_deviation / joint_deviation_hip を緩める方向。
        self.rewards.flight_phase = RewTerm(func=flight_phase, weight=-2.0)

        # ------------------------------------------------------------------
        # 5. 後退ドリフト (優先度低)
        # ------------------------------------------------------------------
        self.rewards.track_lin_vel_x = RewTerm(
            func=track_lin_vel_x_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.25},
        )

        # ------------------------------------------------------------------
        # 6. バッファのリセット
        # ------------------------------------------------------------------
        # 実体の更新は報酬側から毎ステップ 1 回だけ呼ばれる (EventTerm の実行順に
        # 依存させないため)。ここではリセットされた env を無効化するだけ。
        self.events.reset_lateral_buffers = EventTerm(func=reset_lateral_buffers, mode="reset")


@configclass
class K1GKLateralEnvCfg_PLAY(K1GKLateralEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
