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
