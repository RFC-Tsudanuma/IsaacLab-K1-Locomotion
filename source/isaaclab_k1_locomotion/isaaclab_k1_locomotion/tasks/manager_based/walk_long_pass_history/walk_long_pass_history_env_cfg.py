# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ロングパス + 短期 I/O 履歴 (short history) 環境。

:class:`~..walk_long_pass.walk_long_pass_env_cfg.K1WalkLongPassEnvCfg` の
**短期履歴 + 左右対称学習用** バリアント。報酬・コマンド分布・カリキュラム・
行動空間は変えず、policy 観測の本体状態 5 項に 0.1 秒ぶんの履歴を付ける。
さらに、左足裏だけを見る 3 次元スロットをミラー可能なボール 3D 位置へ
差し替える。観測フィールド名と順序は継承元のままなので、policy / critic の
次元は 223 / 61 を維持する。

arXiv:2401.16889 (Locomotion policy に短期の観測+行動履歴を与える) 相当。
ネットワーク構造 (MLP / 隠れ層) は変更しない。増えるのは入力層の幅だけ。

なぜ履歴か
----------
policy は feedforward MLP なので 1 ステップの観測しか見ていない。実機で効く情報の
多くは「単フレームでは観測できないが数フレームの差分には現れる」もの:

* アクチュエータの遅れ … 目標角 (``prev_joint_request``) と実現角 (``joint_pos``) の
  ずれの **時間発展** が、そのステップの実効ゲイン/遅延を語る。sim2real で最も
  ずれる部分なので、履歴があると「今の機体がどれだけ鈍いか」を暗黙同定できる。
* 接地/離地の遷移 … ``projected_gravity`` と ``base_ang_vel`` の 0.1 秒の軌跡には、
  接地衝撃・スリップ・押されの区別が出る。単フレームでは同じ値になる。
* 速度ノイズの平滑 … ``joint_vel`` は Unoise(±1.5) と大きなノイズを載せているので、
  5 フレームあれば実質的なローパスを policy 側で学習できる。

履歴長 = 5 ステップ (0.1 秒)
---------------------------
制御周期は ``sim.dt (0.005) * decimation (4) = 0.02 s`` (50 Hz) なので::

    0.1 s / 0.02 s = 5 ステップ

``_HISTORY_LEN`` は下限 4 を掛けて算出している (制御周期を上げたときに履歴が
2-3 フレームまで潰れると、差分が取れず履歴の意味が無くなるため)。

履歴を付ける項 / 付けない項
---------------------------
付ける (本体状態と自分の出力 = 「機体の I/O」):

======================  ====  =========================================
term                    dim   意味
======================  ====  =========================================
``projected_gravity``   3     胴体姿勢
``base_ang_vel``        3     胴体角速度
``joint_pos``           12    実現関節角
``joint_vel``           12    実現関節速度
``prev_joint_request``  12    **actions** (前ステップの目標関節角)
======================  ====  =========================================

付けない (タスク条件付け項 = 履歴化しても情報が増えない):

* ``kick_direction`` / ``target_kick_velocity`` … エピソード中ずっと定数。
  履歴を取ると同じ値が 5 回並ぶだけ。
* ``gait_phase`` / ``gait_phase_factor_offset`` … 位相は時刻の決定的関数なので、
  履歴は現在値から復元できる。
* ``ball_vel`` / ``prev_ball_pos`` … **既に履歴になっている**。``prev_ball_pos`` は
  遅延させたボール位置そのもの、``ball_vel`` はその差分。ここに history_length を
  重ねると同じ情報を二重に持つことになる (指示の「既存の ball history と重複しない」)。
* ``sole_pos`` … 順序保持のため属性名だけを残した、1 ステップ遅延の
  **ボール 3D 位置**。既存の ``prev_ball_pos`` / ``ball_vel`` と同様に履歴化しない。

次元
----
========  ======  =======  ==============================================
group     before  after    内訳
========  ======  =======  ==============================================
policy    55      **223**  履歴 5 × 42 = 210 + 非履歴 13
critic    61      61       変更なし
action    12      12       変更なし
========  ======  =======  ==============================================

critic に履歴は付けない。左足裏スロットだけは遅延なしボール位置へ
置換するが、既存の特権情報 ``ball_pos_rel`` も残し、61 次元と checkpoint の
形状契約を変えない。どちらもミラー写像では (x, y, z) → (x, −y, z) とする。

既存 checkpoint からの引き継ぎ (重要)
--------------------------------------
**``--load_pretrained`` に生の long_pass checkpoint を渡してはいけない。**
[train.py] は「形の合わないテンソルを捨てる」ので、``actor.0.weight`` (入力層) と
``actor_obs_normalizer.*`` が捨てられ、入力層がランダム初期化された死んだポリシーになる。

さらに **``expand_checkpoint_kick_flag.py`` も使えない**。あちらは「末尾にゼロを足す」
だけだが、履歴化は各項を **その場で 5 倍に展開** するので、55 次元の並びが 223 次元の
中に散らばる (``joint_pos`` は index 11-22 → 35-94 へ移動する)。専用の
:mod:`scripts.rsl_rl.expand_checkpoint_history` を使うこと::

    python scripts/rsl_rl/expand_checkpoint_history.py \\
        logs/rsl_rl/k1_walk_long_pass/<run>/model_<N>.pt \\
        -o /tmp/long_pass_history_init.pt

このスクリプトは履歴ブロックの列を拡張して **形状上は** 読み込めるようにする。
ただし、元 checkpoint の左足裏スロットの重みと正規化統計は、新タスクでは
ボール位置に対して適用される。したがって **意味的に互換ではなく、挙動も一致しない**。
利用する場合は近似的な初期化として扱うこと。通し実行は
``./scripts/rsl_rl/train_walk_long_pass_history.sh`` だが、これも同じ近似初期化を使う。

カリキュラムは :func:`~..walk_long_pass.walk_long_pass_env_cfg._freeze_curricula_at_final`
で全部終値に固定してあるので、``common_step_counter`` が 0 でも iter 0 から親タスクの
収束状態で始まる。``--reset_noise_std`` は **付けないこと** (蹴り方を壊す)。

見るべきもの
------------
* ``Metrics/kick_direction/kick_rate`` … 観測の意味変更と mirror loss 導入時の過渡を
  監視する。旧 checkpoint を使う場合も iter 0 で親と同じ値になる保証はない。
* ``Metrics/kick_direction/kick_vel_ratio`` / ``kick_dir_error_deg`` … 履歴で改善するか。
* ``Train/mean_episode_length`` … 履歴は転倒直前の兆候 (角速度の発散) を見せるので、
  転倒が減れば伸びる。
* ``Policy/mean_noise_std`` … 入力層だけが広がった状態からの再学習なので、std が
  暴れないか。暴れるなら learning_rate を下げる。
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..walk_long_pass.walk_long_pass_env_cfg import (
    _LONG_PASS_SPEED_RANGE,
    K1WalkLongPassEnvCfg,
    _freeze_curricula_at_final,
)
from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import (
    K1WalkKickCriticCfg,
    K1WalkKickObservationsCfg,
    K1WalkKickPolicyCfg,
)
from .symmetry import HISTORY_LEN as _SYMMETRY_HISTORY_LEN

# --------------------------------------------------------------------------- #
# 短期履歴の長さ
#
# 制御周期 = sim.dt * decimation。velocity_env_cfg.py が sim.dt=0.005 / decimation=4
# を設定しているので 0.02 s (50 Hz)。0.1 秒相当は 5 ステップ。
#
# NOTE: 下限 4 は「差分を取るのに最低限必要なフレーム数」。制御周期を 0.04 s 以上に
#       上げると 0.1 秒が 2-3 フレームになり、履歴が実質ただのノイズ平均になる。
# NOTE: この値を変えたら scripts/rsl_rl/expand_checkpoint_history.py の
#       --history-len も揃えること (既定値は 5)。
# --------------------------------------------------------------------------- #
_HISTORY_S = 0.1
_SIM_DT = 0.005
_DECIMATION = 4
_CTRL_DT = _SIM_DT * _DECIMATION  # 0.02 s = 50 Hz
_HISTORY_LEN = max(4, round(_HISTORY_S / _CTRL_DT))  # 5
if _HISTORY_LEN != _SYMMETRY_HISTORY_LEN:
    raise ValueError(
        "walk_long_pass_history の履歴長と mirror 写像が一致しません: "
        f"{_HISTORY_LEN} != {_SYMMETRY_HISTORY_LEN}"
    )

# --------------------------------------------------------------------------- #
# 履歴を付ける policy 観測項 (K1WalkKickPolicyCfg の項名)
#
# 「本体状態 + 自分の出力」だけ。タスク条件付け項 (kick_direction, gait_phase, ball 系)
# は入れない。理由はモジュール docstring の表を参照。
#
# NOTE: 順番は ObservationManager の連結順とは無関係 (連結順は PolicyCfg の宣言順で
#       決まる)。ここは「どの項に付けるか」の集合でしかない。
# NOTE: この集合を変えたら expand_checkpoint_history.py の _POLICY_TERMS も直すこと。
#       あちらは 55 → 223 の列写像を持っているので、片方だけ変えると checkpoint が
#       黙って壊れた並びで読まれる。
# --------------------------------------------------------------------------- #
_HISTORY_TERMS = (
    "projected_gravity",   # 3
    "base_ang_vel",        # 3
    "joint_pos",           # 12
    "joint_vel",           # 12
    "prev_joint_request",  # 12  ← actions (前ステップの目標関節角)
)


@configclass
class K1WalkLongPassHistoryPolicyCfg(K1WalkKickPolicyCfg):
    """223 次元 actor 観測の 1 フレーム分の定義。

    ``sole_pos`` は継承元の項順を保つための属性名。値は左足裏ではなく、
    実機 vision の遅延を表す 1 制御ステップ前のボール 3D 位置である。
    """

    sole_pos = ObsTerm(
        func=mdp.delayed_ball_pos_b,
        noise=Unoise(n_min=-0.02, n_max=0.02),
        params={"delay_steps": 1, "dim": 3},
    )


@configclass
class K1WalkLongPassHistoryCriticCfg(K1WalkKickCriticCfg):
    """61 次元 critic 観測。同じスロットを遅延なしボール位置に置換する。"""

    sole_pos = ObsTerm(
        func=mdp.delayed_ball_pos_b,
        params={"delay_steps": 0, "dim": 3},
    )


@configclass
class K1WalkLongPassHistoryObservationsCfg(K1WalkKickObservationsCfg):
    policy: K1WalkLongPassHistoryPolicyCfg = K1WalkLongPassHistoryPolicyCfg()
    critic: K1WalkLongPassHistoryCriticCfg = K1WalkLongPassHistoryCriticCfg()


@configclass
class K1WalkLongPassHistoryEnvCfg(K1WalkLongPassEnvCfg):
    """ロングパス + 短期 I/O 履歴 + ミラー可能な観測。"""

    observations: K1WalkLongPassHistoryObservationsCfg = K1WalkLongPassHistoryObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. カリキュラムを終値で固定する（継続学習タスクなので必須）
        #
        # --load_pretrained は common_step_counter を 0 のままにするので、放置すると
        # キック報酬の weight も速度帯も親のランプをやり直してしまう。理由の詳細は
        # _freeze_curricula_at_final の docstring 参照。
        _freeze_curricula_at_final(self)

        # -- 2. policy 観測の本体状態 5 項に履歴を付ける
        #
        # ObsGroup 側の history_length は **使わない**。グループに設定すると
        # ObservationManager が全項に一括で配ってしまい (observation_manager.py の
        # 「check group history params and override terms」)、kick_direction や
        # ball_vel まで 5 倍になる。項ごとに設定すること。
        #
        # flatten_history_dim=True で (num_envs, H, dim) → (num_envs, H*dim) に潰す。
        # 並びは CircularBuffer.buffer の仕様どおり **古い順 → 最新が末尾**。
        # checkpoint 拡張スクリプトはこの並びに依存している。
        #
        # NOTE: 履歴に積まれるのは **ノイズを載せた後** の値 (ObservationManager は
        #       modifier → noise → clip → scale の後に append する)。フレームごとに
        #       独立なノイズが乗るので、policy は平滑化も学習できる = 狙いどおり。
        # NOTE: エピソード開始直後は CircularBuffer が 0 埋めではなく
        #       **最初の 1 フレームで全スロットを埋める** ので、リセット直後に
        #       「過去 = 0」という嘘の遷移を見せることはない。
        _policy = self.observations.policy
        for _name in _HISTORY_TERMS:
            _term = getattr(_policy, _name)
            _term.history_length = _HISTORY_LEN
            _term.flatten_history_dim = True


@configclass
class K1WalkLongPassHistoryEnvCfg_PLAY(K1WalkLongPassHistoryEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # 帯カリキュラムを外して終点に固定する。CurriculumManager は PLAY でも
        # 毎リセットで走り、common_step_counter=0 から始まるので、残すと
        # --kick_speed の指定が上書きされる (K1WalkLongPassEnvCfg_PLAY と同じ理由)。
        # 親が K1WalkLongPassEnvCfg_PLAY ではないので、ここで同じ処理を行う。
        self.curriculum.kick_speed_range = None
        self.commands.kick_direction.target_speed_range = _LONG_PASS_SPEED_RANGE
