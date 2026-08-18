# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール履歴版ゴールキーパーの環境定義 (試験実装)。

直接版 (:mod:`..goalkeeper_direct_env_cfg`) の派生で、**既存タスクの挙動には
一切影響しない** (dualhist と同じ方針)。

直接版との差は 2 点だけ:

    1. 方策の ``velocity_commands`` スロットを **ゼロ埋め**
       → 方策は「どこへどれだけ速く動け」という手書きの指令を見なくなる
    2. **ボール相対位置の履歴** を観測の末尾に追加
       → 方策はここから方向と速さを自分で決める

報酬・カリキュラム・終了条件・critic は直接版のまま。指令 (task_drive_vector) は
歩行位相のゲートと報酬の停止判定に **特権情報として残す**。実機では動かない
学習側の量なので、critic に真値を渡すのと同じ扱いでよい。

★ 観測の先頭 59 次元は直接版と同じ並び・同じ次元。末尾に履歴を足すだけなので、
  第1層の重みは **既存 59 列をコピーし、新規列をゼロ初期化** すれば、初期状態が
  直接版のポリシーと数学的に完全に同一になる。Stage1 からの作り直しは不要で、
  ``k1_gk_direct_stage2`` の ckpt から追加学習として始められる
  (変換は scripts/rsl_rl/expand_ckpt_for_ballhist.py)。

★ ゼロ埋めスロットに勾配は流れないので、``velocity_commands`` に対応する重みは
  初期値のまま残る (ball_kick / Stage1 のダミースロットと同じ方式)。
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..goalkeeper_direct_env_cfg import (
    GOAL_HALF_WIDTH,
    TASK_DRIVE_VY_SCALE,
    K1GKDirectCriticCfg,
    K1GKDirectEnvCfg,
    K1GKDirectObservationsCfg,
    K1GKDirectPolicyCfg,
    _make_play_clean,
)
from ..mdp.curriculums import adaptive_difficulty
from ..mdp.observations import zeros_obs
from ...locomotion.rough_env_cfg import _PHASE_FREQ
from .events import sync_engaged_command
from .rewards import ball_lateral_progress
from .observations import ballhist_velocity_commands
from .observations import ballhist_ball_history, ballhist_ball_history_true, ballhist_gait_phase

from isaaclab.managers import CurriculumTermCfg as CurrTerm

# 履歴の窓。10 フレーム x stride 2 = 0.4 秒ぶん (制御 50Hz)。
# ボールの到達時間が 0.55〜1.4 秒なので接近の判断には十分。
# 伸ばすと入力次元が増えるので、まずこの最小構成から始める。
BALLHIST_FRAMES = 10
BALLHIST_STRIDE = 2


@configclass
class K1GKBallHistPolicyCfg(K1GKDirectPolicyCfg):
    """ボール履歴版の actor 観測。**項の定義順 = スロット順**。

    先頭 59 は直接版と同一。``velocity_commands`` だけ中身をゼロに差し替え、
    末尾に履歴を足す。
    """

    # ★ 手書きの指令を方策から隠す。次元は保つ (先頭 49 の並びを壊さないため)。
    #   確率 cmd_dropout_p で隠す形にしてあり、既定 1.0 = 常に隠す。
    #   0 から上げていくと分布外を通らずに移行できる (立ち尽くす局所解の対策)。
    velocity_commands = ObsTerm(
        func=ballhist_velocity_commands,
        params={"max_y": GOAL_HALF_WIDTH, "vy_scale": TASK_DRIVE_VY_SCALE},
    )

    # ★ 歩行位相のゲートを手書きの制御則から is_engaged へ差し替える。
    #   直接版は task_drive_vector (外挿と除算を含む制御則) の大きさで位相を
    #   ゲートしていた。あれは実機の C++ が計算する必要があり、不具合3件の原因
    #   だった。is_engaged はボール相対観測だけの真偽値で、発散もゲイン増幅も
    #   自己位置依存も無い (詳細は observations.ballhist_is_engaged)。
    gait_phase = ObsTerm(
        func=ballhist_gait_phase,
        params={"phase_freq": _PHASE_FREQ, "frames": BALLHIST_FRAMES, "stride": BALLHIST_STRIDE},
    )

    # ★ 末尾に追加。ここが「方策が判断するための材料」。
    ball_history = ObsTerm(
        func=ballhist_ball_history,
        params={"frames": BALLHIST_FRAMES, "stride": BALLHIST_STRIDE},
    )


@configclass
class K1GKBallHistCriticCfg(K1GKDirectCriticCfg):
    """ボール履歴版の critic 観測 (真値・ノイズなし)。actor と次元を揃える。"""

    # ★ 位相は actor と同じソースにする。critic だけ別の位相を見ると、
    #   「同じ状態なのに評価が食い違う」ことになり価値推定が壊れる。
    gait_phase = ObsTerm(
        func=ballhist_gait_phase,
        params={"phase_freq": _PHASE_FREQ, "frames": BALLHIST_FRAMES, "stride": BALLHIST_STRIDE},
    )

    ball_history = ObsTerm(
        func=ballhist_ball_history_true,
        params={"frames": BALLHIST_FRAMES, "stride": BALLHIST_STRIDE},
    )


@configclass
class K1GKBallHistObservationsCfg(K1GKDirectObservationsCfg):
    policy: K1GKBallHistPolicyCfg = K1GKBallHistPolicyCfg()
    critic: K1GKBallHistCriticCfg = K1GKBallHistCriticCfg()


@configclass
class K1GKBallHistEnvCfg(K1GKDirectEnvCfg):
    """ボール履歴版 (難易度固定)。Play 用の土台も兼ねる。"""

    observations: K1GKBallHistObservationsCfg = K1GKBallHistObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # ★ コマンドバッファへの書き込みを、手書きの制御則から出動フラグへ差し替える。
        #   方策はこのコマンドを見ない (velocity_commands はゼロ埋め)。書く目的は
        #   feet_phase / foot_clearance / 待機ペナルティの判定を動かすことだけ。
        #   ノルムが「0 か nominal」の二値になるので、直接版で問題になった
        #   「小さい非ゼロ指令」の帯域が構造的に消える。
        _dt = self.sim.dt * self.decimation
        self.events.sync_task_command = EventTerm(
            func=sync_engaged_command,
            mode="interval",
            interval_range_s=(_dt, _dt),
            is_global_time=True,
        )



@configclass
class K1GKBallHistStage2EnvCfg(K1GKBallHistEnvCfg):
    """ボール履歴版の本体。直接版 Stage2 と同じ適応カリキュラムを使う。"""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.difficulty = CurrTerm(func=adaptive_difficulty)


@configclass
class K1GKBallHistEnvCfg_PLAY(K1GKBallHistEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _make_play_clean(self)


# ---------------------------------------------------------------------------
# Pure 版: **手書きの式を一切使わない** (is_engaged を除く)
#
# 基準版 (上) は critic と報酬に手書きの外挿式を **特権情報として残している**。
# 実機では動かないので害は無いが、「外挿点へ向かうのが良い状態」という評価を
# critic が学び、報酬もその戦略を誘導するため、**線形外挿が表現できない振る舞い
# (バウンド球・回転球・遅い球を引きつける判断) は学習されない**。式が実機から
# 消えても、学習の天井としては残る。
#
# Pure 版はそこも外す:
#   * critic の target_y / velocity_commands をゼロ埋め
#   * 密な報酬を予測なしの潜在関数ベース (ball_lateral_progress) に差し替え
#
# 収束は基準版より難しくなる見込み。両方を回して比較できるよう別タスクにしてある
# (Pure が収束しなかったとき「履歴で判断するのが難しいのか」「報酬の誘導を
#  外したのが効いたのか」を切り分けるため)。
# ---------------------------------------------------------------------------


@configclass
class K1GKBallHistPureCriticCfg(K1GKBallHistCriticCfg):
    """Pure 版の critic 観測。手書きの外挿式をゼロ埋めする。

    critic は真値のボール履歴と自機状態を持っているので、価値関数は自分で
    組み立てられる。次元は基準版と同じ (ゼロ埋めなので勾配が流れないだけ)。
    """

    velocity_commands = ObsTerm(func=zeros_obs, params={"dim": 3})
    target_y = ObsTerm(func=zeros_obs, params={"dim": 1})


@configclass
class K1GKBallHistPureObservationsCfg(K1GKBallHistObservationsCfg):
    critic: K1GKBallHistPureCriticCfg = K1GKBallHistPureCriticCfg()


@configclass
class K1GKBallHistPureEnvCfg(K1GKBallHistEnvCfg):
    """Pure 版 (難易度固定)。Play 用の土台も兼ねる。"""

    observations: K1GKBallHistPureObservationsCfg = K1GKBallHistPureObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 密な報酬を予測なしの形へ差し替える。
        # weight は元の target_reach_velocity (4.0) と寄与の桁を合わせた暫定値。
        # 差分は 1 ステップあたり最大 0.1m なので、4.0 のままだと 2 桁小さくなる。
        self.rewards.target_reach_velocity = None
        self.rewards.ball_lateral_progress = RewTerm(
            func=ball_lateral_progress,
            weight=1500.0,  # 実測 0.035 を基準版の target_reach_velocity (1.2) と同じ桁へ
            params={"max_step_m": 0.03},
        )


@configclass
class K1GKBallHistPureStage2EnvCfg(K1GKBallHistPureEnvCfg):
    """Pure 版の本体 (適応カリキュラム付き)。"""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.difficulty = CurrTerm(func=adaptive_difficulty)


@configclass
class K1GKBallHistPureEnvCfg_PLAY(K1GKBallHistPureEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _make_play_clean(self)
