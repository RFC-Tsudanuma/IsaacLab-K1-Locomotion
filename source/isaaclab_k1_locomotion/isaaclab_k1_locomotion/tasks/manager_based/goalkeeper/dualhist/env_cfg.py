# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""デュアルヒストリー版ゴールキーパー (階層版 v2 の上位ポリシーの試験実装)。

既存の階層版 v2 (``goalkeeper_hier_env_cfg.py``) との差は **上位 policy の観測に履歴
ブロックを 2 本足した点だけ**。シーン・ボール発射・知覚モデル・報酬・終了条件・
凍結下位はすべて既存タスクをそのまま継承する。

    policy   59 → 59 + 5×7 (短期 0.1s) + 50×7 (長期 1.0s) = 444 次元
    critic   64 (変更なし。特権情報の真値を単一フレームで持っているため)
    low_level 59 (変更なし。凍結下位 07-28 が読める形でなければならない)

なぜ上位だけか (2026-08-15 のユーザーとの議論):
    論文の dual history が効くのは「実機転移」であって sim の return ではない、という
    ablation がある。下位 (07-28) は既に実機デプロイ済みなので、構造を変えると再デプロイ
    と実機再検証が発生する。一方このタスクで支配的な時変ノイズはボール知覚であり、それを
    見ているのは上位。したがって費用対効果は上位 > 下位。

学習手順:
    **既存の階層版 Stage1 ckpt からは resume できない** (actor の構造が違う)。
    Stage1 から学習し直すこと。observation の先頭 59 は既存と同じ並びなので、
    履歴なしの ckpt から先頭列だけ引き継ぐ warmstart は原理的には書けるが、
    今回は素直に Stage1 からやり直す (Stage1 は 5000 iter で頭打ちになる想定)。
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ..goalkeeper_hier_env_cfg import (
    K1GKHierObservationsCfg,
    K1GKHierPolicyCfg,
    K1GKHierStage1EnvCfg,
    K1GKHierStage2EnvCfg,
    _make_play_clean,
)
from .curriculums import fixed_difficulty_log
from .observations import gk_io_history, reset_gk_history

# --- 履歴の長さ (agents 側の hist_short_frames / hist_long_frames と必ず一致させること) ---
# 制御は 50Hz なので 5 frame = 0.1s、50 frame = 1.0s。
#
# 短期 (0.1s): 論文と同じ長さ。接触・急変への即応担当。
# 長期 (1.0s): 論文は 2.0s だが、こちらの狙いは「時不変ダイナミクスの同定」ではなく
#   「ボール軌道のフィルタリングと外挿」なので、飛翔時間 (1〜2s) の直近 1 秒あれば足りる。
#   2.0s にすると大半が「まだ球が無かった時間」のゼロ埋めになる。
HIST_SHORT_FRAMES = 5
HIST_LONG_FRAMES = 50


@configclass
class K1GKHierDHPolicyCfg(K1GKHierPolicyCfg):
    """上位 policy 観測 = 既存 59 次元 + 履歴 2 本。

    **項の定義順 = スロット順。短期 → 長期の順序をネットワーク側が前提にしている**
    (:class:`~.networks.DualHistoryActor`)。dataclass は既存フィールドの位置を保つので、
    ここで足した 2 項は必ず末尾 (59 の後ろ) に付く。
    """

    hist_short = ObsTerm(
        func=gk_io_history, params={"num_frames": HIST_SHORT_FRAMES, "stride": 1}
    )
    hist_long = ObsTerm(
        func=gk_io_history, params={"num_frames": HIST_LONG_FRAMES, "stride": 1}
    )


@configclass
class K1GKHierDHObservationsCfg(K1GKHierObservationsCfg):
    policy: K1GKHierDHPolicyCfg = K1GKHierDHPolicyCfg()


def _add_history_reset(cfg) -> None:
    """エピソード開始時に履歴を消すイベントを足す。

    これが無いと前エピソードのボール軌道が新エピソードの先頭に残る。
    """
    cfg.events.reset_gk_history = EventTerm(func=reset_gk_history, mode="reset")


@configclass
class K1GKHierDHStage1EnvCfg(K1GKHierStage1EnvCfg):
    """Stage 1 (ボールなし)。履歴のボール系チャンネルは常にゼロ、自機 pose 側だけが動く。

    ボールが無くても自機 pose の履歴は「自分がどう動いているか」の情報を持つので、
    CNN は無駄にならない (停止判断に効く)。
    """

    observations: K1GKHierDHObservationsCfg = K1GKHierDHObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _add_history_reset(self)


@configclass
class K1GKHierDHStage2EnvCfg(K1GKHierStage2EnvCfg):
    """Stage 2 (ゴール + ボール + 適応カリキュラム)。Stage 1 の ckpt から --resume。"""

    observations: K1GKHierDHObservationsCfg = K1GKHierDHObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _add_history_reset(self)

        # --- セーブ後に中央へ戻す (2026-08-15) ---
        # 継続モードでは 2 球目以降が必ず「前の球を止めた場所」から始まる。到達可能性
        # モデルで測ると、球が来た瞬間のキーパー y と「物理的に取れない球」の割合は
        #     y=0.0 → 33.0% / y=0.4 → 37.3% / y=0.6 → 42.1% / y=0.8 → 47.9%
        # で、中央から 0.6m ずれるだけで +9pt。下位を横 2.0 m/s に作り直しても -2.8pt
        # しか改善しない (猶予時間に対して立ち上がり 0.6s + 遅延 0.156s が支配的で、
        # 定常速度に乗る前に決着がつくため) ので、**立ち位置の方が桁で効く**。
        #
        # 従来はセーブ後 3.0s のあいだ指令が全成分ゼロで、復帰時間が構造的にゼロだった。
        # 保持をセーブ確定 (2.0s) までに縮め、respawn 待ちを 1.0s → 2.0s に伸ばして
        # 復帰に充てる。2.0s あれば横 1.278 m/s・立ち上がり 0.6s でも約 1.3m 戻れるので、
        # 0.8m のずれは解消できる。目標 0 (中央) は compute_target_y が自動で返すため
        # 報酬の変更は不要。
        self.goalkeeper.post_save_hold_until_relaunch = False
        if getattr(self.events, "relaunch_ball", None) is not None:
            self.events.relaunch_ball.params["respawn_delay_steps"] = 100

        # --- 守備面を 0.8m → 0.6m (ユーザー指示 2026-08-15) ---
        # 到達可能性モデルでの「物理的に取れない球」の割合 (速度上限 1.728, キーパー中央):
        #     guard_x 0.9 → 40.6% / 0.8 → 35.4% / **0.6 → 26.1%** / 0.4 → 13.8%
        # 0.8 → 0.6 で **-9.3pt**。これまで見つけた中で中央復帰と並ぶ最大の効き幅で、
        # 下位を横 2.0 m/s に作り直した場合 (-2.8pt) の 3 倍以上ある。
        #
        # 効く理由は反応時間。守備面を後ろに下げるとボールがそこへ届くまでの距離が
        # 伸びる (t_guard = (spawn_x - guard_x) / vx) ので、猶予時間が直接増える。
        # 本タスクの律速は最高速ではなく「立ち上がり 0.6s + 知覚遅延 0.156s」なので、
        # 猶予が増えることの価値が大きい。
        #
        # トレードオフ: 前に出るほどシュートコースを狭められる、という利点は減る。
        # このモデルはその効果 (角度による有効ゴール幅の縮小) を含んでいないので、
        # 0.4 まで下げれば 13.8% という数字を額面どおりには受け取らないこと。
        # 実測で 0.6 の結果を見てから次を判断する。
        self.goalkeeper.guard_x = 0.6

        # --- ボール位置の時系列平均化: 試したが **効果なし**、無効のまま (2026-08-16) ---
        # 到達点予測に渡す位置が生の 1 フレーム観測だったので、直近 0.5s の平均に変えて
        # みた。予測誤差は狙いどおり 0.41m → 0.14m (3.6 倍) に下がったが、**同一 ckpt での
        # 対照実験でセーブ率が動かなかった**:
        #     model_30200 / 汚い知覚 / 800球
        #       平均化 ON  到達可能球 76.2%
        #       平均化 OFF 到達可能球 76.7%     ← 標準誤差 1.65pt の範囲内で同じ
        #
        # つまりキーパーの失敗は「どこに来るか分からない」ことが原因ではない。位置ノイズは
        # 白色なので平均化で消せる誤差だったが、消せない **系統誤差** (速度バイアス
        # 0.05〜0.15 m/s × 到達時間、検出の欠落、自己位置バイアス 0.20m) の方が効いている、
        # というのが現時点の解釈。
        #
        # 機構は mdp/observations.py に残してある (既定 0.0 = 無効)。実機のノイズが
        # シムより大きければ効く可能性はあるので、消さずに否定的知見ごと残す。
        # ★ 実機側に実装する必要はない (シムで中立なので、入れると負担が増えるだけ)。

        # 初期配置は guard_x から導出されるので、親が 0.8 で書いた値を上書きし直す
        # (cfg 構築時の 1 回きりの導出なので、guard_x を変えただけでは追従しない)。
        _gx = float(self.goalkeeper.guard_x)
        self.events.reset_base.params["pose_range"]["x"] = (_gx, _gx)


@configclass
class K1GKHierDHFinalEnvCfg(K1GKHierDHStage2EnvCfg):
    """最終分布で直接学習する版。**適応カリキュラムを使わない。**

    なぜ止めるのか (2026-08-16 の実測):

    1. **崩壊が 2 回とも昇格の直後に起きた。** 速度 2.488 と 2.509 で、刻みを 1.2 → 1.1 に、
       ``hard_ball_speed_mult`` を 1.6 → 1.3 に下げ、健全と検証した ckpt から始めても再現した。
       内訳を見ると ``Loss/symmetry`` が 0.04 → 1.7 (40倍) に爆発し、損失全体の 53% を
       占める状態になっていた (surrogate の 400 倍)。昇格イベントそのものが引き金なので、
       **無くせば消える**。

    2. **段階的に登る必要がない。** スポーン距離が時間で決まる設計 (``d = v × spawn_time_*``)
       なので、速い球ほど遠くから来て **到達時間の分布は速度によらず一定**。到達不能球の
       割合も 27〜29% で変わらない。変わるのはスポーン距離、つまり知覚ノイズだけ:

           速度上限   到達不能   到達時間中央   スポーン距離p90   σ(d)
             2.07      27.4%       0.69s          2.08m        0.41m
             6.00      27.0%       0.75s          5.77m        0.86m

    3. **固定重みなら高速球でも転ばない。** model_32000 を速度別に走らせた実測:

           速度上限 2.074  セーブ 67.5%  転倒 0.2%
           速度上限 3.0    セーブ 55.7%  転倒 0.0%
           速度上限 4.0    セーブ 45.8%  転倒 0.2%

       つまり 3 m/s は **能力としては既に満たしている**。登れないのは学習の問題。

    難易度は ``ball_speed_max`` (= 速度上限、下限は ``ball_speed_min`` = 0.5 固定) と
    ``aim_y_range`` で直接指定する。速度は U(0.5, ball_speed_max) の一様分布なので、
    易しい球が分布の半分を占め、学習信号は途切れない。
    """

    def __post_init__(self):
        super().__post_init__()

        # 適応カリキュラムを外し、成功率の記録だけを残す (学習の進み具合を見る主指標)
        self.curriculum.difficulty = None
        self.curriculum.hard_ball = None
        self.curriculum.fixed = CurrTerm(func=fixed_difficulty_log)

        # 難易度を直接指定する。カリキュラムが _gk_speed_hi / _gk_aim_y を作らないので、
        # reset_ball_shot はこの 2 つを直接読む。
        # ★ 2026-08-16 (ユーザー判断): **2 段階でやる。まず 3.0、固まったら 6.0。**
        #
        #   要件は「最低 3 m/s、最大 6 m/s」。3.0 を先にやる理由:
        #     1. 締め切りリスク。時間切れでも必須要件を満たした方策が残る。6.0 から
        #        始めて途中で切れると 3 も 6 も中途半端になる。
        #     2. 分布の飛びが小さい。現在の方策は 2.07 で学習済み。2.07 → 3.0 は小さいが
        #        2.07 → 6.0 は大きい。崩壊が 2 回とも 2.5 付近で起きているので、
        #        大きく飛ばすのは博打になる。
        #     3. 拡張が無料。観測の次元は変わらないので、この値を 6.0 に書き換えて
        #        resume するだけで済む。3.0 で学習した ckpt をそのまま使える。
        #
        #   3.0 → 6.0 へ進む判定基準:
        #     * base_contact が 0.01 未満を維持している (崩壊していない)
        #     * Curriculum/fixed/success_ema の上昇が 2000 iter 以上止まった (頭打ち)
        #     * eval_gk_hier_envelope.py で転倒 1% 未満を確認
        #
        #   期待値の目安 (model_32000 を速度別に固定して測った実測値):
        #       速度上限 3.0 → 全球 55.7%   4.0 → 45.8%   6.0 → 40% 前後 (外挿)
        #   転倒は 4.0 でも 0.2% なので、速度を上げても物理的に崩れる心配は無い。
        self.goalkeeper.ball_speed_max = 3.0
        self.goalkeeper.aim_y_range = 1.1       # 適応カリキュラムの最終段と同じ

        # 到達不能球は適応で増やさず、最終値 (0.1) を最初から固定で混ぜる
        self.goalkeeper.hard_ball_auto = False
        self.goalkeeper.hard_ball_prob = 0.1


@configclass
class K1GKHierDHStage1EnvCfg_PLAY(K1GKHierDHStage1EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _make_play_clean(self)


@configclass
class K1GKHierDHStage2EnvCfg_PLAY(K1GKHierDHStage2EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _make_play_clean(self)


@configclass
class K1GKHierDHFinalEnvCfg_PLAY(K1GKHierDHFinalEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _make_play_clean(self)
