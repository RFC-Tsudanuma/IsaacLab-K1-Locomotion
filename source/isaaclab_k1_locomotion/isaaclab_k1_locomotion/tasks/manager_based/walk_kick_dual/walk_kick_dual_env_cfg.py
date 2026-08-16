# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_kick 系列の dual 版 (dual encoder + 両足キック用の観測 2 変更)。

``fewa/walk_kick_dual_encoder_tune`` が walk_long_pass に入れた「dual encoder」化を、
walk_kick ファミリー (walk_kick / walk_weak_kick / walk_middle_kick) へ移植し、
さらに :mod:`..walk_kick_both_feet` の 2 変更を **畳み込んだ** もの。
このファイルは **3 ファミリー共通のヘルパー層** で、
:mod:`..walk_weak_kick_dual` と :mod:`..walk_middle_kick_dual` からも import される。
履歴長・遅延量をここ 1 箇所で管理しないと段の間で checkpoint が繋がらない。

dual に入っているもの
---------------------
1. **dual encoder** (fewa 由来): actor の入力を 100 フレームの観測履歴にする。
2. **観測スロット 3 = ボール 3D 位置** (both_feet 由来): 元は左足裏 ``sole_pos``。
   これは原典 B-Human の観測表 ("Current Ball 3D Position") の読み違えでもあった。
3. **歩行位相の初期オフセット {0, π}** (both_feet 由来): 位相 0 固定だと歩き出しの
   遊脚が常に右足になり、蹴り足が右に固定される。
4. **着地 shaping 3 項の無効化** (fewa 47b8863 由来): 全 stage で
   :func:`disable_landing_shaping`。蹴るほど損をする圧を消す。
5. **feet_phase の weight 2.0 → 0.8** (fewa 47b8863 由来): 蹴りのある段だけ
   :func:`rebalance_gait_vs_kick`。歩行報酬とキックの綱引きを釣り合わせる。
6. **ボール観測の DR** (fewa 47b8863 由来): 視覚レイテンシ (0.02-0.10 s) と
   一様ノイズの拡大 (位置 ±0.07 m / 速度 ±0.5 m/s)。最終 stage (DR 段) のみ
   (:func:`enable_obs_delay`)。

2 と 3 を別 variant にせず直接畳み込むのは、**dual 系がまだ未学習で失う互換性が
無い**ため。both_feet の効果は stage 2 で実測済み:
``kick_foot_right_frac`` 1.0 → 0.39 (両足で蹴れている)、``kick_dir_error`` 4.5°、
``kick_rate`` 0.998 (方向精度・成功率を落とさずに左右が割れた)。

段構成 (3 ファミリーとも **4 段**、fewa 47b8863 と同じ配置)
-----------------------------------------------------------
* **walk_kick_dual**       : walk phase → kick → 360 (クリーン) → **360 DR** (最終)
* **walk_weak_kick_dual**  : (walk phase は walk_kick_dual と共用) → weak
  → 360Weak (クリーン) → **360Weak DR** (最終)
* **walk_middle_kick_dual**: (walk phase は walk_kick_dual と共用) → middle
  → 360Middle (クリーン) → **360Middle DR** (最終)

段ごとに載るもの (fewa 47b8863 の配置と同一):

===========================  =========  =======  ==========  ==========
機能                         walk phase  蹴り段   クリーン360  DR 段 (最終)
===========================  =========  =======  ==========  ==========
enable_obs_history           ✓          ✓        ✓           ✓
disable_landing_shaping      ✓          ✓        ✓           ✓
rebalance_gait_vs_kick       —          ✓        ✓           ✓
apply_sigma_direction_anneal —          —        ✓ (本体)     hold で 0.15 固定
enable_obs_delay             —          —        —           ✓
_freeze_fade_in_curricula    —          —        —           ✓ (全項)
位相オフセット / both_feet 観測  ✓          ✓        ✓           ✓
===========================  =========  =======  ==========  ==========

**4 段構成と DR の配置は fewa 47b8863 に合わせた。** 3 段構成 (360 が最終) から
DR 段を切り出したのは、fewa が実機で成果を出している段の切り方に揃えるため。
副作用として ``ball_avoidance`` のランプがクリーン 360 で完走するので、DR 段の
:func:`_freeze_fade_in_curricula` は **除外なしの全凍結**でよくなった
(3 段構成では 360 が最終段だったので ``ball_avoidance_weight`` の除外が必須だった)。

**最終段のボール観測 DR は一様ノイズ + 連続遅延** (fewa 47b8863 準拠)。ガウスの
認識パイプライン (``noisy_ball_pos_b``) 方式は fewa の実機実績に合わせて不採用
(ユーザー判断 2026-08-17)。関数 :func:`apply_noisy_ball_pos_obs` /
:func:`disable_noisy_ball_pos_jitter` は戻せるように残してあるが、呼び出し元は無い。

副作用として critic は 58 次元になる (both_feet の critic は特権スロット
``ball_pos_rel`` を持たない。スロット 3 を ``delay_steps=0`` にすれば同じ値が
その場で得られるため。詳細はあちらの定数ブロックの NOTE)。

dual encoder とは
-----------------
actor の入力を 1 フレーム (55 次元) から **H = 100 フレームの履歴** (N, H, 55) に変え、
ネットワーク側 (:class:`~..locomotion.networks.ActorCriticHistoryCNN`) が

* 直近 K = 5 フレームをそのまま平坦化したもの (5 × 55 = 275 次元)
* 全 H フレームを Conv1d × 2 で符号化した潜在 (16 × 15 = 240 次元)

を concat して actor MLP に入れる (= 「そのまま」と「符号化」の 2 本の encoder)。
critic は **1 フレームのまま**。環境側の差分は :func:`enable_obs_history` の 2 行だけで、
報酬・コマンド・シーン・カリキュラムは継承元とまったく同じ。挙動の違いは観測の
見え方だけに閉じている。

``walk_long_pass_history`` と混同しないこと
-------------------------------------------
:mod:`..walk_long_pass_history` も「履歴」を名乗るが **別方式**。あちらは
``flatten_history_dim = True`` で **項ごとに** (H, d_i) を平坦化してから連結するので、
並びが [gravity の H フレーム][ang_vel の H フレーム]... になり、フレーム単位の
(N, H, 55) には戻せない (CNN のチャンネル = 観測次元にできない)。
dual encoder 版は ``flatten_history_dim = False`` が必須で、ネットワークも
``ActorCriticHistoryCNN`` に差し替わる。両者の checkpoint は互換性が無い。

既存 1 フレームタスクとの checkpoint 互換性
--------------------------------------------
actor MLP の 1 層目の形と重みの名前 (``actor.0.*`` vs ``actor.mlp.0.*``) が変わるので、
1 フレーム観測の checkpoint をそのまま ``--load_pretrained`` すると **actor の重みが
1 本も引き継がれない** (train.py が形の合わないテンソルを黙って捨てる。
現在は actor が 0 本なら止まるようにしてある)。

1 フレーム観測の checkpoint から始めるときは ``--warm_start_from_single_frame`` を
付ける。旧 actor を履歴 actor の「最新フレームの列」へ移植するので、学習開始時点の
出力が旧ポリシーと一致する (仕組みは
:func:`~..locomotion.networks.remap_single_frame_actor` の docstring)。
dual 系どうしを繋ぐときは不要 (形が同じなのでそのまま載る)。

**引き継ぎ元は both_feet 系の checkpoint に限る。** 同梱の
``k1_walk_kick_walk_phase/2026-08-03_11-22-52`` のような **旧 sole_pos 系** は使えない:
policy は同じ 55 次元なので ``--load_pretrained`` は形の上では通ってしまうが、
スロット 3 の意味が違う (左足裏 vs ボール 3D 位置) ので入力の解釈がずれる。
critic も 61 → 58 次元で形が合わない。使えるのは
``logs/rsl_rl/k1_walk_kick_both_feet_walk_phase`` / ``k1_walk_kick_both_feet`` の run。

学習の通し実行は ``scripts/rsl_rl/train_walk_kick_dual.sh``。
"""

import inspect

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..locomotion.flat_env_cfg import NOISY_FLAT_TERRAIN_CFG
from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import (
    _BALL_OBS_CAMERA_HZ,
    _BALL_OBS_DELAY_STEP_RANGE,
    _BALL_OBS_JITTER_CLIP,
    _BALL_OBS_JITTER_STD,
    K1WalkKick360EnvCfg,
)
from ..walk_kick_both_feet.walk_kick_both_feet_env_cfg import (
    _apply_phase_offset,
    K1WalkKickBothFeetEnvCfg,
    K1WalkKickBothFeetEnvCfg_PLAY,
    K1WalkKickBothFeetObservationsCfg,
    K1WalkKickBothFeetWalkPhaseEnvCfg,
    K1WalkKickBothFeetWalkPhaseEnvCfg_PLAY,
)

# --------------------------------------------------------------------------- #
# 観測履歴の長さ H
#
#   H = 50  → CNN 出力長 15 → 6  → 潜在  96 → actor 入力 371
#   H = 100 → CNN 出力長 32 → 15 → 潜在 240 → actor 入力 515
#
# 0.02 s/step なので H = 100 は 2 秒分。キックは接近から latch まで 1 秒以上かかるので、
# 1 秒 (H = 50) だと踏み込みの序盤がバッファから落ちる。
#
# **H を変えた checkpoint 同士は繋がらない。** actor MLP の 1 層目の形が変わるので、
# --load_pretrained では入力層が黙って捨てられる (train.py が actor を 1 本も
# 引き継げなければ止める)。1 フレーム観測の checkpoint からなら H に関係なく
# --warm_start_from_single_frame で移植できる (最新フレームの列に置いて残りをゼロに
# するだけなので、潜在の次元が変わっても成立する)。
#
# NOTE: この定数は walk_kick_dual / walk_weak_kick_dual / walk_middle_kick_dual の
#       **全 stage** から読まれる。1 段でも違う値にすると段の間で checkpoint が
#       繋がらず、しかも起動時には気づけないので、必ずここ 1 箇所で管理すること。
# NOTE: ONNX の入力形状も (1, H, 55) になる。H を変えたら実機側のリングバッファ長も
#       合わせること。
# --------------------------------------------------------------------------- #
_OBS_HISTORY_LENGTH = 100


def enable_obs_history(cfg) -> None:
    """policy 観測グループを 100 フレームの履歴にする。

    ``flatten_history_dim = False`` が必須。True にすると ObservationManager は
    **項ごとに** (H, d_i) を平坦化してから連結するので、並びが
    [gravity の H フレーム][ang_vel の H フレーム]... になり、フレーム単位の
    (N, H, 55) には戻せない (CNN のチャンネル = 観測次元にできなくなる)。
    False なら各項が (N, H, d_i) のまま最終軸で連結され、(N, H, 55) が得られる。

    履歴は環境リセットで巻き戻り、直後は現在フレームで H 個すべてが埋まる
    (IsaacLab の CircularBuffer の仕様)。ゼロ詰めの履歴は入らない。

    critic はこれまでどおり 1 フレーム観測 (特権情報付き) のまま。critic 側にも
    履歴を付ける場合は ``ActorCriticHistoryCNN`` の critic 側も直す必要がある。
    """
    cfg.observations.policy.history_length = _OBS_HISTORY_LENGTH
    cfg.observations.policy.flatten_history_dim = False


# --------------------------------------------------------------------------- #
# センサ遅延 DR の上限 [s]
#
# 制御周期 dt = 0.02 s に対して 1 ステップぶん。実機は IMU も関節エンコーダも
# 「測ってから policy に届くまで」に遅れがあり、sim で遅延ゼロのまま学習すると
# 遅れた観測に対して過剰に反応する (特に base_ang_vel は歩行の安定化に直結する)。
#
# 遅延量は env ごと・エピソードごとに [0, この値] から一様サンプリングし、
# 過去フレームの線形補間で連続値として与える (整数ステップだと 0/1 の 2 値に
# なってしまう)。詳細は :func:`~..walk_kick.mdp.observations._delayed_signal`。
#
# NOTE: ボール知覚の遅延 (``_apply_noisy_ball_obs`` / ``delayed_ball_pos_b``) とは
#       別物。あちらは vision パイプライン、こちらは自己受容感覚。
# --------------------------------------------------------------------------- #
_OBS_DELAY_MAX_S = 0.02

# --------------------------------------------------------------------------- #
# ボール観測 (視覚由来) の遅延上限 [s] と固定遅延
#
# 移植元: fewa/walk_kick_dual_encoder_tune の 47b8863。
#
# 本体センサ (IMU / エンコーダ) と分けてあるのは、実機での出所が違うため。
# ボールはカメラ → 検出 → 座標変換のパイプラインを通るので、実機のレイテンシは
# 内界センサより **明確に大きい** (一般的なヒューマノイドで 30-60 ms = 1.5-3 ステップ)。
#
# both_feet 観測は同じボール位置履歴から 2 スロットを取る (スロット 3 = 1 step 遅延の
# 3D、スロット 12 = 2 step 遅延の 2D)。この整数ステップの設計遅延の上に、vision group
# 共有の連続遅延が乗る。実効遅延:
#
#   ball_pos (スロット 3)  : 0.02 (設計 1 step) + [0, 0.06] = 0.02-0.08 s
#   prev_ball_pos (スロ 12): 0.04 (設計 2 step) + [0, 0.06] = 0.04-0.10 s
#   ball_vel               : 0.02 (_BALL_OBS_BASE_DELAY_S) + [0, 0.06] = 0.02-0.08 s
#
# ball_vel に固定ぶんを足すのは、位置スロットが設計上すでに遅れているため。同じカメラ
# フレームから出る量のレイテンシがずれているのは実機ではあり得ないので、いちばん新しい
# 位置スロット (スロット 3) と揃える。vision group で乱数を共有するので、ランダム成分は
# 3 項とも同じ値になる (= スロット 12 が常にスロット 3 の 1 step 前、という関係は不変)。
#
# 0.08 s = 4 ステップ。実機のカメラ遅延を測った結果で調整すること。
# --------------------------------------------------------------------------- #
_BALL_OBS_DELAY_MAX_S = 0.06
_BALL_OBS_BASE_DELAY_S = 0.02

# --------------------------------------------------------------------------- #
# ボール観測のノイズ (一様、±この値)
#
# 継承値は位置 ±0.02 m / 速度 ±0.4 m/s。実機のカメラ由来の量としては楽観的すぎる
# (とくに速度はフレーム間差分から求めるので、位置ノイズを dt = 0.02 s で微分した
#  ぶんのばらつきが原理的に乗る)。
#
# 位置 ±0.07 m はボール半径 (0.11 m) の 6 割ほど。fewa の実測では ±0.1 m
# (半径とほぼ同じ) まで上げると critic が繰り返し発散した (10000 iteration 中
# value loss > 50 が 230 回、最大 1.26e12、kick_rate < 0.5 への崩壊が 102 回)。
# ±0.07 では発散 0 回・崩壊 3 回。**±0.07 を超えないこと。**
#
# 速度 ±0.5 m/s はボール速度観測に頼った蹴り分けをほぼ不可能にする量だが、そもそも
# 実機で信用できない量なので頼らせない方が正しい。
#
# NOTE: 位置スロットに認識パイプライン (noisy_ball_pos_b、σ = 6.7 cm のジッタ) が
#       載っている cfg では位置ノイズを **触らない** (二重掛けになる)。現在そういう
#       cfg は無い (noisy 段は廃止) が、ガードは防御として残してある
#       (:data:`_PIPELINE_BALL_POS_FUNCS`)。
# --------------------------------------------------------------------------- #
_BALL_POS_NOISE = 0.07
_BALL_VEL_NOISE = 0.5

# 遅延を掛ける policy 観測項と、遅延量を共有するセンサ (group)。
#
# group が同じ項は同じ遅延量を引く。projected_gravity と base_ang_vel は同じ IMU、
# joint_pos と joint_vel は同じエンコーダ読み出しから来るので、独立に遅れることは
# 物理的にあり得ない (重力は最新なのに角速度だけ 1 ステップ古い、など)。
# ボール 3 項も同じカメラフレームから出るので "vision" で共有する。
#
# 要素は (項名, group, mdp の関数名, source, 追加 params)。
# source は max_delay_s を引く元 ("body" = 内界センサ, "ball" = 視覚)。
#
# 項名は :class:`~..walk_kick_both_feet.walk_kick_both_feet_env_cfg.K1WalkKickBothFeetPolicyCfg`
# のフィールド名と一致していること (存在しなければ enable_obs_delay が例外を投げる)。
_DELAYED_OBS_TERMS = (
    ("projected_gravity", "imu", "delayed_projected_gravity", "body", {}),
    ("base_ang_vel", "imu", "delayed_base_ang_vel", "body", {}),
    ("joint_pos", "encoder", "delayed_joint_pos_rel", "body", {}),
    ("joint_vel", "encoder", "delayed_joint_vel_rel", "body", {}),
    ("ball_vel", "vision", "delayed_ball_vel_b", "ball", {"base_delay_s": _BALL_OBS_BASE_DELAY_S}),
    ("ball_pos", "vision", "delayed_ball_pos_vision_b", "ball", {"delay_steps": 1, "dim": 3}),
    ("prev_ball_pos", "vision", "delayed_ball_pos_vision_b", "ball", {"delay_steps": 2, "dim": 2}),
)

# 認識パイプラインが既に載っている位置スロットは遅延 DR の対象外にする
# (noisy_ball_pos_b がエピソードごとランダム遅延 2-6 step + 30Hz サンプル&ホールド +
#  ジッタを既に掛けているので、その上に連続遅延と一様ノイズを重ねない)。
#
# NOTE: **現在このガードに掛かる cfg は無い。** noisy-ball の最終 stage は廃止した
#       (ユーザー判断 2026-08-17。:func:`apply_noisy_ball_pos_obs` の NOTE 参照)。
#       戻したときに二重掛けを踏まないよう、防御としてそのまま残してある。
_PIPELINE_BALL_POS_FUNCS = (mdp.noisy_ball_pos_b,)


def enable_obs_delay(
    cfg,
    max_delay_s: float = _OBS_DELAY_MAX_S,
    ball_max_delay_s: float | None = None,
) -> list[str]:
    """policy 観測の IMU / エンコーダ / ボール由来の項にセンサ遅延 DR を掛ける。

    既存の ObsTerm の ``func`` と ``params`` だけを差し替える。項を作り直さないのは

    * ノイズ設定 (Unoise) と asset_cfg をそのまま引き継ぐため
    * 観測の **並び順を絶対に変えない** ため (policy 観測 55 次元は順序が
      実機側の詰め方と 1 対 1 で対応している。configclass のフィールド順で
      連結されるので代入では順序は変わらないが、項を消して足し直すと変わる)

    観測次元は変わらないので、遅延を入れる前後で checkpoint はそのまま繋がる。

    critic には掛けない。特権情報から価値を推定させる側なので、遅延させても
    学習が難しくなるだけで得るものが無い。

    中間 stage から呼ぶこともできるが、その場合は全 stage で同じ ``max_delay_s``
    にすること (遅延の有無で歩き方が変わるので、段の間で条件が変わると引き継ぎの
    意味が薄れる)。

    **認識パイプラインが載っている位置スロットはスキップする。** ``ball_pos`` /
    ``prev_ball_pos`` が既に :func:`~..walk_kick.mdp.observations.noisy_ball_pos_b`
    (ランダム遅延 + サンプル&ホールド + ジッタ) になっている cfg では、その上に
    連続遅延とノイズを重ねない。**現在そういう cfg は無い** (noisy 段は廃止) ので、
    このガードは防御。掛かった場合、その段で実際に差し替わるボール項は
    ``ball_vel`` だけになる (認識パイプラインは速度を扱わない既知の穴で、
    そこだけは塞ぐのが正しいため)。

    Args:
        max_delay_s: 内界センサ (IMU / エンコーダ) の遅延上限 [s]。
        ball_max_delay_s: ボール観測 (視覚) の遅延上限 [s]。None なら
            :data:`_BALL_OBS_DELAY_MAX_S`。実機のカメラ遅延は内界センサより
            大きいので別パラメータにしてある。

    Returns:
        実際に遅延を掛けた観測項の名前 (スキップした項は含まない)。
    """
    if ball_max_delay_s is None:
        ball_max_delay_s = _BALL_OBS_DELAY_MAX_S

    applied: list[str] = []
    for term_name, group, func_name, source, extra in _DELAYED_OBS_TERMS:
        term = getattr(cfg.observations.policy, term_name, None)
        if term is None:
            raise AttributeError(
                f"policy 観測に '{term_name}' がありません。遅延 DR の対象項名が"
                " 継承元の観測構成と食い違っています。"
            )
        # 認識パイプラインが載っている位置スロットは触らない (二重掛けを避ける)。
        if term_name in ("ball_pos", "prev_ball_pos") and term.func in _PIPELINE_BALL_POS_FUNCS:
            continue
        delay = ball_max_delay_s if source == "ball" else max_delay_s
        term.func = getattr(mdp, func_name)
        term.params = {**term.params, "max_delay_s": delay, "group": group, **extra}
        applied.append(term_name)

    # ボール観測のノイズを実機のカメラ由来の量に合わせて広げる
    # (_BALL_POS_NOISE / _BALL_VEL_NOISE のコメント参照)。
    # 速度は常に。位置は差し替えた (= パイプラインが載っていない) 場合だけ。
    cfg.observations.policy.ball_vel.noise = Unoise(n_min=-_BALL_VEL_NOISE, n_max=_BALL_VEL_NOISE)
    for term_name in ("ball_pos", "prev_ball_pos"):
        if term_name in applied:
            getattr(cfg.observations.policy, term_name).noise = Unoise(
                n_min=-_BALL_POS_NOISE, n_max=_BALL_POS_NOISE
            )
    return applied


# --------------------------------------------------------------------------- #
# 着地の作り込み系の shaping 項。dual 3 ファミリーでは **全 stage** で無効化する。
#
# 移植元: fewa/walk_kick_dual_encoder_tune の 47b8863。
#
#   feet_landing_impact : 接地瞬間の接地力ノルムを罰する    (weight -0.5e-2)
#   feet_landing_vel    : 接地瞬間の足速度を罰する          (weight -1.5)
#   feet_heel_strike    : 接地瞬間のかかと下がり姿勢を賞する (weight +5.0)
#
# いずれも「接地イベントの瞬間」にだけ効くスパースな項で、蹴りとは相性が悪い。
# 蹴り足のスイングは接地を強く速くする方向にしか作れないので、蹴るほど
# feet_landing_impact / feet_landing_vel の罰が増える。feet_heel_strike も
# 蹴り足の踏み込みがかかと下がりになりにくいぶん、蹴ると取り逃す報酬になる。
# 結果として「蹴らずに丁寧に歩く」方が得になる圧が常時かかる。
#
# NOTE: 消すのは dual 系列だけ。これらは共用の K1FlatEnvCfg で定義されており、
#       元を消すと walk_kick / walk_pass / walk_lob / walk_mid_kick / loop_shoot と
#       コミット済みの基準 checkpoint まで道連れになる。
# --------------------------------------------------------------------------- #
_LANDING_SHAPING_TERMS = (
    "feet_landing_impact",
    "feet_landing_vel",
    "feet_heel_strike",
)


def disable_landing_shaping(cfg) -> None:
    """着地 shaping 3 項 (:data:`_LANDING_SHAPING_TERMS`) を無効化する。

    dual 3 ファミリーの **全 stage** (walk phase を含む) から呼ぶ。段によって有無が
    変わると、前段が獲得した歩容が次の段で罰せられる (あるいはその逆) ことになるので、
    必ず全段で揃えること。

    報酬項の増減は観測にも行動にも影響しないので、checkpoint の引き継ぎには
    影響しない (actor / critic の形は変わらない)。
    """
    for name in _LANDING_SHAPING_TERMS:
        # 親の構成が変わって項が無くなっていても落ちないように getattr でガードする。
        if getattr(cfg.rewards, name, None) is not None:
            setattr(cfg.rewards, name, None)


# --------------------------------------------------------------------------- #
# 蹴りのある段での feet_phase の weight。継承元 (rough_env_cfg) は 2.0。
#
# 移植元: fewa/walk_kick_dual_encoder_tune の 47b8863。
#
# 経緯 (fewa の long_pass 系列での実測):
#   着地 shaping 3 項を消したことで Stage 1 の歩容が大きく良くなり、位相追従がほぼ
#   満点に張り付くようになった。その結果、蹴り段に入った時点で
#   Episode_Reward/feet_phase が 0.62 → 1.71 と 2.7 倍になり、キックが払うコスト
#   (kick_finished が latch の 2 秒後にエピソードを終わらせるので、キックは常に
#    「残りの歩行報酬を捨てる」コストを払う) がそのぶん膨らんだ。
#
#   実測: 未対処だと kick_rate が 0.19-0.28 のまま 4000 iteration 停滞し、
#   kick_elevation_deg は 0.97°。それでいて mean_reward は 12.5 (旧 run 7.7) と高く、
#   「蹴らずに歩く」が正しく最適解になっていた。weight を下げると iteration 1400 で
#   kick_rate 0.99 に立ち上がった。
#
#   0.8 は割り戻し: 2.0 * (0.62 / 1.71) ≒ 0.73。
#
# NOTE: これは **fewa の long_pass での実測由来**の値。うちのファミリー
#       (walk_kick / weak / middle) で歩容が崩れるようなら見直すこと。
# NOTE: 監視は kick_rate。停滞したら歩行報酬過多のサイン (fewa は 0.19-0.28 で
#       4000 iteration 停滞 → 0.8 に下げて iteration 1400 で 0.99)。逆に歩容が
#       崩れる (feet_phase の実測値が落ちる・転倒終了が増える) なら上げ戻す。
# NOTE: walk phase (stage 1) では下げない。歩容そのものをこの項で作っており、
#       ここで下げると土台が崩れる。下げるのは蹴りと綱引きになる段だけ。
# --------------------------------------------------------------------------- #
_KICK_STAGE_FEET_PHASE_WEIGHT = 0.8


def rebalance_gait_vs_kick(cfg) -> None:
    """蹴りのある段で feet_phase の weight を下げる (:data:`_KICK_STAGE_FEET_PHASE_WEIGHT`)。

    蹴りのある全段 (stage 2 と 360、およびその PLAY) から呼ぶ。
    **walk phase からは呼ばない。**
    """
    term = getattr(cfg.rewards, "feet_phase", None)
    if term is not None:
        term.weight = _KICK_STAGE_FEET_PHASE_WEIGHT


# --------------------------------------------------------------------------- #
# カリキュラムが 1 iteration とみなす env step 数 (新設カリキュラム専用)
#
# curriculums.py の start_step / end_step は
# ``env.common_step_counter // steps_per_iteration`` で iteration に換算される。
# common_step_counter は env.step() 1 回で 1 増えるので、この値は PPO の
# ``num_steps_per_env`` と一致していなければならない
# (locomotion/agents/rsl_rl_ppo_cfg.py: num_steps_per_env = 48)。
#
# NOTE: 継承元 (walk_kick_env_cfg の _spi / weak の _SPI / middle の _SPI) は 24 の
#       ままで、コメントには「steps_per_iteration = num_steps_per_env」と書いてあるのに
#       実際の num_steps_per_env は 48。つまり walk_kick 系のランプは全部、書いてある
#       iteration 数の **半分** で終わっている。既に収束している既存タスクの挙動を
#       変えないよう継承元は触らず、**dual 系で新しく足すカリキュラムだけ** この値を使う。
#
# 使っているのは :func:`apply_sigma_direction_anneal` (σ_direction のアニール)。
# 帯を動かすカリキュラム (fewa の ``kick_rate_gated_speed_range`` /
# ``linear_reward_weight_after_speed_gate``) を後から足すときも、この値を使うこと。
#
# NOTE: 継承元のランプ (weak の strong 折れ線 / σ_velocity アニール / overshoot、
#       middle の σ_lon アニール) は _SPI = 24 のままなので、**同じ「1500 → 3000」でも
#       実際の iteration は半分 (750 → 1500) で終わる**。つまり σ_direction の
#       アニール (真の 1500 → 3000) は、weak/middle の「強さの移行」が終わった後に
#       始まる。方向の締め上げを強さの収束後に置くのは順序として妥当なので、
#       これは意図どおり。数値が同じでも時刻は違う点に注意すること。
# --------------------------------------------------------------------------- #
_STEPS_PER_ITERATION = 48

# --------------------------------------------------------------------------- #
# キック方向のシェイピング係数 σ_direction のアニール [rad]
#
# 報酬は f(τ) = exp(−τ² / 2σ²) (τ = 指令方向と実ボール速度方向のなす角)。
# 継承元の σ = 0.35 rad (20°) に対して both_feet の実測誤差は **4.5°** で、
# f(4.5°) = 0.975。**ほぼ飽和していて、これ以上精度を上げる勾配圧力が無い**。
# σ = 0.15 なら f(4.5°) ≈ 0.87 になり、圧力が復活する。
#
# 下限を 0.15 で止める理由:
#
# * ``kick_direction`` は (f − 0.5) · 2 · p_style なので、**f < 0.5 で蹴りが負報酬**に
#   なり「蹴らない方が得」が復活する。現状の誤差 4.5° だと σ = 0.06 がその崖。
#   0.15 は崖から十分手前。
# * noisy 段はボール位置ジッタ σ = 6.7cm があり、1 m 先のボールで約 **3.8° の知覚床**を
#   作る。それ以下の精度は原理的に追えないので、深追いしても報酬が下がるだけ。
#
# 窓 1500 → 3000 iteration。開始を 1500 にするのは、weak/middle の「強さの移行」
# (strong フェードアウト / σ_velocity アニール) と同じ数値だが、あちらは _SPI = 24 の
# ため実時刻では先に終わっている (上の _STEPS_PER_ITERATION の NOTE 参照)。
#
# NOTE: **kick_rate が 1.0 から落ちたら σ が深すぎるサイン**。次に深くする
#       (0.15 → 0.10) のは ``Metrics/kick_direction/kick_dir_error_deg`` が
#       下がりきってから。順序を逆にすると、精度が付く前に報酬が負に振れて
#       「蹴らない」へ落ちる。
# NOTE: weak/middle が既に持っている ``sigma_velocity`` のアニール (1.0 → 0.35 / 0.5)
#       とは **別物**。あちらは ``kick_velocity_scaled`` の「速さの採点の厳しさ」、
#       こちらは全キック項が共有する「方向の採点の厳しさ」。
# --------------------------------------------------------------------------- #
_SIGMA_DIRECTION_ANNEAL_START = 0.35
_SIGMA_DIRECTION_ANNEAL_END = 0.15
_SIGMA_DIRECTION_ANNEAL_START_ITER = 1500
_SIGMA_DIRECTION_ANNEAL_END_ITER = 3000


def _sigma_direction_terms(cfg) -> list[str]:
    """``sigma_direction`` を取る報酬項の名前を集める。

    「関数のシグネチャに ``sigma_direction`` があるか」で判定し、**その項の params に
    ``sigma_direction`` が入っていなければ例外**にする。params に無い項は関数の
    デフォルト (0.35) で動くので、アニールの対象から漏れて **項ごとに σ が食い違う**
    (``kick_foot_lift`` の docstring にある「σ_direction は同じタスクの他のキック項と
    必ず揃えること」という不変条件が壊れる)。壊れたことが数値でしか見えないので、
    構造的に検出して止める。
    """
    targets: list[str] = []
    for name in dir(cfg.rewards):
        if name.startswith("_"):
            continue
        term = getattr(cfg.rewards, name, None)
        func = getattr(term, "func", None)
        if func is None:
            continue
        try:
            takes_sigma = "sigma_direction" in inspect.signature(func).parameters
        except (TypeError, ValueError):  # C 実装など内省できないもの
            continue
        if not takes_sigma:
            continue
        if "sigma_direction" not in term.params:
            raise ValueError(
                f"報酬項 '{name}' は sigma_direction を取るのに params に入っていません"
                " (関数デフォルトの 0.35 で動きます)。アニールの対象から漏れて他の"
                " キック項と σ が食い違うので、params に明示してください。"
            )
        targets.append(name)
    if not targets:
        raise ValueError(
            "sigma_direction を持つ報酬項が 1 つも見つかりませんでした。"
            " 継承元の報酬構成と食い違っています (配線ミスの可能性)。"
        )
    return targets


def apply_sigma_direction_anneal(cfg) -> list[str]:
    """全キック項の ``sigma_direction`` を 0.35 → 0.15 へ **同時に** 絞る。

    対象は :func:`_sigma_direction_terms` が自動で見つける。名前を列挙しないのは、
    「σ_direction を使う全項を同時に同じ値で動かす」という不変条件を構造的に
    保証するため (項を足したときに片方だけ絞る事故が起きない)。項ごとに
    ``<term_name>_sigma_direction`` という curriculum 項を、**全て同じスケジュール**で
    追加する。

    値と窓の根拠は :data:`_SIGMA_DIRECTION_ANNEAL_END` のコメント参照。

    NOTE: :func:`_freeze_fade_in_curricula` は ``linear_reward_weight`` の項しか
          見ないので、ここで足す ``linear_reward_param`` の項とは干渉しない
          (呼ぶ順序も問わない)。

    Returns:
        アニール対象になった報酬項の名前。
    """
    targets = _sigma_direction_terms(cfg)
    for term_name in targets:
        setattr(
            cfg.curriculum,
            f"{term_name}_sigma_direction",
            CurrTerm(
                func=mdp.linear_reward_param,
                params={
                    "term_name": term_name,
                    "param_name": "sigma_direction",
                    "start_value": _SIGMA_DIRECTION_ANNEAL_START,
                    "end_value": _SIGMA_DIRECTION_ANNEAL_END,
                    "start_step": _SIGMA_DIRECTION_ANNEAL_START_ITER,
                    "end_step": _SIGMA_DIRECTION_ANNEAL_END_ITER,
                    "steps_per_iteration": _STEPS_PER_ITERATION,
                },
            ),
        )
    return targets


def hold_sigma_direction(cfg, value: float = _SIGMA_DIRECTION_ANNEAL_END) -> list[str]:
    """``sigma_direction`` を終値で **固定**する (アニールのカリキュラムは外す)。

    **現在の用途は 2 だけ** (アニールが入っている段の PLAY)。1 は noisy 段の廃止で
    呼び出し元が消えたが、アニール済みの段の後ろに段を足すときの作法として残す。

    **1. アニールを済ませた段の後段** (現在は該当なし)。もう一度アニールし直さない
    のは、段の境界で σ が 0.35 に戻ると方向への圧力が一度緩み、せっかく詰めた方向が
    流れるため。fine-tune の出発点は「σ = 0.15 で収束したポリシー」なので、その値を
    そのまま 1 iteration 目から掛けるのが正しい
    (:func:`_freeze_fade_in_curricula` がフェードインに対してやるのと同じ理屈)。

    **2. アニールが入っている段の PLAY**。CurriculumManager は PLAY でも走るが
    ``common_step_counter`` は 0 から始まるので、放っておくと σ が **開始値 0.35 に
    巻き戻る** (walk_long_pass の PLAY が帯カリキュラムで踏んだのと同じ機序)。
    σ_direction は報酬のシェイピング係数なので観測にも行動にも影響しないが、
    そのままだと PLAY で出る報酬・メトリクスが学習終盤の値と比較できない。
    項ごと外して終値を直接入れる。

    Returns:
        固定した報酬項の名前。
    """
    targets = _sigma_direction_terms(cfg)
    for term_name in targets:
        getattr(cfg.rewards, term_name).params["sigma_direction"] = value
        # アニールのカリキュラムが載っていれば外す (無ければ何もしない)。
        curr_name = f"{term_name}_sigma_direction"
        if getattr(cfg.curriculum, curr_name, None) is not None:
            setattr(cfg.curriculum, curr_name, None)
    return targets


def _freeze_fade_in_curricula(cfg, before_iter: int, exclude: tuple[str, ...] = ()) -> list[str]:
    """``before_iter`` までに立ち上がりきる報酬重みのランプを、終値の定数に潰す。

    継承元 (walk_kick / 360) では、キック報酬の weight を 0 → 500 iteration で
    フェードインさせている。ゼロから学習する段では「まず歩けるようになってから
    蹴りを乗せる」ための正しい順序だが、**既に蹴れるポリシーからの fine-tune では
    逆に働く**:

    * 歩行報酬 (feet_phase / track_lin_vel など) は 1 iteration 目から満額。
    * キック報酬は weight ≈ 0 から始まる。
    * ``kick_finished`` は latch の 2 秒後にエピソードを終わらせるので、キックは
      「残りの歩行報酬を捨てる」コストだけを払っている。

    つまり最初の 500 iteration は「蹴らない方が得」が明示的に成立していて、PPO は
    その間ずっと継承したキックを消す方向に更新される。500 iteration 後に weight が
    戻ってきても、そのときには蹴らなくなった後なので報酬が payout されない。

    最終 stage の出発点は「その weight で既に収束したポリシー」なので、終値をそのまま
    1 iteration 目から掛けるのが正しい。``start_weight`` を ``end_weight`` に揃える
    ことで、ランプ項を消さずに定数化する (どの項がいくつなのかがログに出たままになる)。

    ``before_iter`` より後にランプする項 (weak/middle の ``kick_velocity_overshoot``:
    1500 → 3000) は意図的に後段へ置かれているので触らない。
    ``func`` が :func:`~..walk_kick.mdp.curriculums.linear_reward_weight` の項だけを
    見るので、weak/middle の ``piecewise_reward_weight`` (strong の折れ線) や
    ``linear_reward_param`` (σ アニール) も対象外になる。

    ``exclude`` — **前段で完走していないランプは凍結しない**
    ------------------------------------------------------
    この関数の前提は「出発点が、そのランプを完走したポリシーであること」。前段で
    一度も掛かっていない罰を凍結すると、fine-tune の 1 iteration 目からいきなり
    満額で降ってくる。``exclude`` はその例外を名前で外すための引数で、既定は空
    (= 全て凍結。挙動は従来どおり)。

    実例が ``ball_avoidance_weight`` (0 → -3, 0-500 iteration)。この項は
    **360 で初登場** で、限定レンジの段から来たポリシーは一度も受けていない。
    凍結すると「回り込みを覚える段」の初回から満額の -3 が掛かり、実績のある既存
    レシピ (walk_kick → 360 をランプで漸進導入) から逸脱する。

    NOTE: **現在 ``exclude`` を渡す呼び出し元は無い** (既定の空 = 全凍結)。旧 3 段
          構成では 360 が最終段だったので ``ball_avoidance_weight`` の除外が必須
          だったが、4 段構成ではクリーン 360 (stage 3) でランプが完走するため、
          DR 段 (stage 4) では除外なしの全凍結が正しい (fewa 47b8863 の Stage 4 と
          同じ)。引数は「前段で完走していないランプを足したとき」のために残してある。

    Returns:
        定数化した curriculum term の名前 (呼び出し側で表示・検証に使う)。
    """
    frozen: list[str] = []
    for name in dir(cfg.curriculum):
        if name.startswith("_") or name in exclude:
            continue
        term = getattr(cfg.curriculum, name, None)
        if term is None or getattr(term, "func", None) is not mdp.linear_reward_weight:
            continue
        params = term.params
        # end_step を持たない項 (帯ゲートに紐付いたランプなど) は壁時計で動いて
        # いないので対象外。get(..., 0) で拾うと「0 → before_iter に収まる」と
        # 誤判定して潰してしまう。
        if "end_step" not in params or params["end_step"] > before_iter:
            continue
        if params["start_weight"] == params["end_weight"]:
            continue
        params["start_weight"] = params["end_weight"]
        frozen.append(name)
    return frozen


# --------------------------------------------------------------------------- #
# 最終 stage のキック報酬フェードイン終点 [iteration]
#
# walk_kick 系のキック報酬ランプは全て 0 → 500 iteration (walk_kick_env_cfg の
# _phase2、360 の ball_avoidance_weight も同じ窓)。fine-tune 前提の最終 stage では
# これを定数化する。
# --------------------------------------------------------------------------- #
_FADE_IN_END_ITER = 500

# ボールを地面から浮かせて落とす量 [m]。凹凸の振幅 (最大 4cm) より上に取る。
# terrain generator の env origin z は「サブ地形中央付近の最大高さ」なので、そこから
# 離れた位置では地面がそれより低い。浮かせずに置くとめり込んだ状態から弾かれてしまう。
# 値と理由は :data:`~..walk_kick.walk_kick_env_cfg._BALL_SPAWN_CLEARANCE` と同じ
# (fewa 側には無いが、feat の凹凸地形タスクの流儀に合わせる)。
_BALL_SPAWN_CLEARANCE = 0.05


def apply_noisy_flat_terrain(cfg) -> None:
    """地面を「段差・坂なし・±1-4 cm のノイズだけ」の軽い凹凸に差し替える。

    NOTE: **現在は未使用。** 学習時間が伸びすぎるため、最終 stage のレシピ
          (:func:`apply_dr_stage_recipe`) から外した (ユーザー判断 2026-08-17)。
          戻すときは :func:`apply_dr_stage_recipe` に呼び出しを足すだけでよい。
          その際は 3 つの最終 stage の PLAY cfg に ``_apply_play_viewer`` も
          足すこと (generator 地形では既定の world 固定カメラだと ``--video`` に
          地形しか映らない。feat の rough 系 PLAY と同じ流儀)。

    継承元 (K1FlatEnvCfg) は ``terrain_type="plane"`` の完全平面。実機の床は
    わずかに波打っていて、平面だけで学習したポリシーは足裏が想定どおり接地する
    前提に寄りかかる。段差・坂は入れず、ノイズだけを乗せる
    (:data:`~..locomotion.flat_env_cfg.NOISY_FLAT_TERRAIN_CFG`)。

    height_scanner は継承元で None のまま = 地形は観測できない (blind)。
    凹凸は「観測できない外乱」として効くので、観測 55 次元・並びは不変。

    地形カリキュラム (``curriculum.terrain_levels``) は継承元で None、
    TerrainGeneratorCfg 側も ``curriculum=False`` なので、5x5 の patch は
    難易度順ではなく単なるバリエーション。``max_init_terrain_level=None`` で
    全 patch に env が散る。

    NOTE: 高さ系のしきい値は絶対 z のまま (base_height_penalty 0.45 /
          base_height 終了 0.35)。地形ノイズは +0.01-0.04 m と上向きだけなので
          判定は緩む方向に動き、誤終了は増えない。
    NOTE: kick_apex_height / sole_height_at_kick も絶対 z なので、地形ぶん
          (最大 4 cm) 嵩上げされて見える。報酬には使っていないが、平地の run と
          数値を直接比べるときは注意すること。
    """
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = NOISY_FLAT_TERRAIN_CFG
    cfg.scene.terrain.max_init_terrain_level = None

    # ボールを置くタスクでは、地形の起伏ぶんだけ浮かせてから落とす。
    # walk phase は reset_ball ごと None なので触らない。
    if getattr(cfg.events, "reset_ball", None) is not None:
        cfg.events.reset_ball.params["spawn_clearance"] = _BALL_SPAWN_CLEARANCE


def apply_dr_stage_recipe(
    cfg,
    fade_in_end_iter: int = _FADE_IN_END_ITER,
    freeze_exclude: tuple[str, ...] = (),
) -> None:
    """dual 系の **最終段 (DR 段 = stage 4)** 用のセット。3 ファミリー共通。

    クリーン 360 (stage 3) との差はここに並ぶ 3 点だけで、いずれも「実機に出す
    1 歩手前」の締め。報酬の構成・コマンド・ボール配置・地形は stage 3 と同一:

    1. :func:`enable_obs_delay` — IMU / エンコーダの遅延 DR (≤ 0.02 s) に加えて、
       **ボール観測 (視覚) の遅延 DR とノイズ拡大** (位置 ±0.07 m / 速度 ±0.5 m/s)。
       観測の次元も並びも変わらないので stage 3 の checkpoint がそのまま載る。
    2. :func:`_freeze_fade_in_curricula` — 収束済みポリシーからの fine-tune なので
       キック報酬のフェードインを定数化する。
    3. :func:`hold_sigma_direction` — σ_direction を stage 3 の終値 0.15 で固定する。
       アニールは stage 3 で完走しているので、ここでやり直すと段の境界で 0.35 に
       戻り、詰めた方向精度が流れる。

    ``freeze_exclude`` はそのまま :func:`_freeze_fade_in_curricula` の ``exclude`` へ
    渡す。**4 段構成では 3 ファミリーとも既定の空 (= 全凍結) でよい。** DR 段より前に
    クリーン 360 が入り、そこで ``ball_avoidance_weight`` のランプが完走するため
    (fewa 47b8863 の Stage 4 と同じ)。引数は「その段で初登場のランプ」を足したときの
    ために残してある (理由は :func:`_freeze_fade_in_curricula` の docstring)。

    ここに含めないもの: :func:`disable_landing_shaping` と
    :func:`rebalance_gait_vs_kick` は **中間 stage でも掛ける** (段によって報酬の
    構成が変わると前段の歩容が次段で罰せられる) ので、各 stage が個別に呼ぶ。

    地形は継承元のまま (``terrain_type="plane"`` の完全平面)。軽い凹凸を入れる
    :func:`apply_noisy_flat_terrain` も用意してあるが、学習時間が伸びすぎるため
    このレシピからは外してある (ユーザー判断 2026-08-17)。

    履歴入力 (:func:`enable_obs_history`) は前段から入っているので、ここでは
    改めて呼ばない前提 (呼び出し側で先に呼ぶこと)。
    """
    enable_obs_delay(cfg, _OBS_DELAY_MAX_S)
    _freeze_fade_in_curricula(cfg, before_iter=fade_in_end_iter, exclude=freeze_exclude)
    hold_sigma_direction(cfg)


def apply_noisy_ball_pos_obs(cfg) -> None:
    """観測スロット 3 (ball_pos) も実機の認識パイプライン寄りに差し替える。

    NOTE: **現在は未使用。** ガウスパイプライン方式は fewa の実機実績に合わせて
          不採用 (ユーザー判断 2026-08-17)。DR 段のボール観測は一様ノイズ + 連続遅延
          (fewa 47b8863 準拠) にしてある。戻す場合は **DR 段の継承元を NoisyBall 系
          (``K1WalkKick360WeakNoisyBallEnvCfg`` / ``K1WalkMiddleKick360NoisyBallEnvCfg``)
          に戻し、この関数と :func:`disable_noisy_ball_pos_jitter` (PLAY 側) の 2 つを
          呼ぶ**。:func:`enable_obs_delay` はパイプラインが載った位置スロットを自動で
          スキップするので、二重掛けにはならない (:data:`_PIPELINE_BALL_POS_FUNCS`)。

    基底の :func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs` は
    ``prev_ball_pos`` (スロット 12) しか差し替えない。walk_kick 系では
    ボール位置を持つ policy スロットがそれ 1 つだったからで、**both_feet の観測を
    使う dual ではスロット 3 にもボール 3D 位置が入っている**。そのままだと
    「知覚ノイズ ablation なのにスロット 3 だけ綺麗なボール位置が見える」抜け穴に
    なり、ポリシーはノイズを避けてスロット 3 だけを読むようになる。

    そこでスロット 3 も :func:`~..walk_kick.mdp.observations.noisy_ball_pos_b`
    (``dim=3``) に差し替える。z も同じ遅延・サンプル&ホールド・ジッタを通る。

    2 つのスロットは **同じ 1 台のカメラ** から取る。``noisy_ball_pos_b`` の状態は
    env 単位で共有されるので、スロット 3 が最新フレーム (``frame_lag=0``)、
    スロット 12 がその 1 フレーム前 (``frame_lag=1``) になる。両者で
    ``delay_step_range`` / ``camera_hz`` / ``jitter_std`` / ``jitter_clip`` を必ず
    揃えること (揃えないと、そのステップで先に評価された項の値だけが効く。
    あちらの警告参照)。

    critic には掛けない (真値のまま)。``__post_init__`` で基底の
    ``_apply_noisy_ball_obs`` が済んだ **後** に呼ぶこと。
    """
    params = {
        "delay_step_range": _BALL_OBS_DELAY_STEP_RANGE,
        "camera_hz": _BALL_OBS_CAMERA_HZ,
        "jitter_std": _BALL_OBS_JITTER_STD,
        "jitter_clip": _BALL_OBS_JITTER_CLIP,
    }

    ball_pos = cfg.observations.policy.ball_pos
    ball_pos.func = mdp.noisy_ball_pos_b
    ball_pos.params = {**params, "dim": 3, "frame_lag": 0}
    # ジッタは関数内でフレーム同期に付与するので、毎ステップ独立の Unoise は外す
    # (残すと二重にノイズが乗る)。基底が prev_ball_pos に対してやっているのと同じ。
    ball_pos.noise = None

    # スロット 12 は「スロット 3 の 1 フレーム前」に戻す。基底の _apply_noisy_ball_obs は
    # frame_lag を指定しないので、このままだとスロット 3 と同じ値 (の水平成分) になり、
    # 「直前のボール位置」というスロットの意味が消えてしまう。
    cfg.observations.policy.prev_ball_pos.params["frame_lag"] = 1


def disable_noisy_ball_pos_jitter(cfg) -> None:
    """PLAY 用: スロット 3 の関数内ジッタを切る。遅延 + サンプル&ホールドは残す。

    NOTE: **現在は未使用** (ガウスパイプライン方式は不採用。戻し方は
          :func:`apply_noisy_ball_pos_obs` の NOTE 参照)。

    基底の :func:`~..walk_kick.walk_kick_env_cfg._disable_ball_obs_jitter` は
    ``prev_ball_pos`` しか触らない。``noisy_ball_pos_b`` の状態は共有で、しかも
    ジッタ幅は「そのステップで先に評価された項」の値が使われるので、宣言順で先に
    ある ``ball_pos`` を 0 にしないと ``prev_ball_pos`` 側の 0 が効かない。
    **必ず両方呼ぶこと。**
    """
    cfg.observations.policy.ball_pos.params["jitter_std"] = 0.0


# --------------------------------------------------------------------------- #
# Stage 1: 歩行のみ (履歴入力版)
# --------------------------------------------------------------------------- #
@configclass
class K1WalkKickDualWalkPhaseEnvCfg(K1WalkKickBothFeetWalkPhaseEnvCfg):
    """Stage 1 (歩行のみ) の dual 版。

    継承元を both_feet 側にしてあるので、スロット 3 (ball_pos) の walk phase 用
    差し替え (ボールが居ないので歩行コマンド ``walk_command_xyz`` を載せる) と
    位相オフセットの event が丸ごと入る。ここでの追加は 2 行:

    * :func:`enable_obs_history` (100 フレーム履歴)
    * :func:`disable_landing_shaping` (着地 shaping 3 項を外す。全段で揃える)

    :func:`rebalance_gait_vs_kick` は **呼ばない**。この段は歩容そのものを
    ``feet_phase`` で作っているので、ここで weight を下げると土台が崩れる
    (fewa 47b8863 も Stage 1 では下げていない)。

    共用の Walk-Phase タスクに履歴を足すと walk_kick / walk_pass / walk_lob /
    walk_mid_kick / loop_shoot まで道連れになるので、dual 系列専用のタスクを
    別 ID で登録している。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        enable_obs_history(self)
        disable_landing_shaping(self)


@configclass
class K1WalkKickDualWalkPhaseEnvCfg_PLAY(K1WalkKickBothFeetWalkPhaseEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        enable_obs_history(self)
        disable_landing_shaping(self)


# --------------------------------------------------------------------------- #
# Stage 2: 限定レンジのキック (履歴入力版)
# --------------------------------------------------------------------------- #
@configclass
class K1WalkKickDualEnvCfg(K1WalkKickBothFeetEnvCfg):
    """Stage 2 (限定レンジのキック) の dual 版。

    継承元を both_feet 側にしてあるので、観測スロット 3 (ボール 3D 位置) と
    位相オフセットが入った状態になる。ここでの追加は 3 行:

    * :func:`enable_obs_history` (100 フレーム履歴)
    * :func:`disable_landing_shaping` (着地 shaping 3 項を外す)
    * :func:`rebalance_gait_vs_kick` (``feet_phase`` の weight 2.0 → 0.8)

    後ろ 2 つは **蹴りのある段すべて** に掛ける (fewa 47b8863 の Stage 2-4 と同じ)。
    引き継ぎ元は Stage 1 (dual) の checkpoint。

    both_feet 系の **1 フレーム** checkpoint (``k1_walk_kick_both_feet_walk_phase`` /
    ``k1_walk_kick_both_feet``) からなら ``--warm_start_from_single_frame`` で
    直行できる。旧 sole_pos 系は不可 (モジュール docstring 参照)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        enable_obs_history(self)
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)


@configclass
class K1WalkKickDualEnvCfg_PLAY(K1WalkKickBothFeetEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        enable_obs_history(self)
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)


# --------------------------------------------------------------------------- #
# Stage 3: 全方位 (クリーン)。DR はまだ入れない
# --------------------------------------------------------------------------- #
@configclass
class K1WalkKickDual360EnvCfg(K1WalkKick360EnvCfg):
    """Stage 3 (全方位、クリーン) の dual 版。**最終段ではない** (次が DR 段)。

    :class:`~..walk_kick.walk_kick_env_cfg.K1WalkKick360EnvCfg` との差は

    * ``observations`` を :class:`~..walk_kick_both_feet.walk_kick_both_feet_env_cfg.K1WalkKickBothFeetObservationsCfg`
      に差し替え (スロット 3 = ボール 3D 位置、critic 58 次元)
    * :func:`~..walk_kick_both_feet.walk_kick_both_feet_env_cfg._apply_phase_offset`
      (歩行位相の初期オフセット {0, π})
    * :func:`enable_obs_history` (100 フレーム履歴)
    * :func:`disable_landing_shaping` / :func:`rebalance_gait_vs_kick`
    * :func:`apply_sigma_direction_anneal` (方向の採点を σ 0.35 → 0.15 に締める。
      **このアニールの本体はこの段**。1500 → 3000 iteration なので
      ``--max_iterations`` は 3000 以上)

    **観測 DR は入れない。** センサ遅延・ボール観測のノイズ拡大は次の DR 段
    (:class:`K1WalkKickDual360DREnvCfg`) の担当で、この段は「回り込みを覚える」ことと
    「方向を詰める」ことに集中させる。``ball_avoidance_weight`` のランプ (0 → -3,
    0-500 iteration) もこの段で **完走させる** (だから :func:`_freeze_fade_in_curricula`
    をここでは呼ばない)。fewa 47b8863 の Stage 3 と同じ役割分担。

    NOTE: both_feet パッケージには 360 版が無いので、ここだけは継承ではなく
          「共用の 360 に both_feet の 2 変更を載せ直す」形になる。both_feet 側の
          観測を変えたときはこのクラスも自動で追随する (同じ cfg クラスを指すため)
          が、``_apply_phase_offset`` の呼び忘れには注意すること。

    報酬・コマンド・ボール配置・終了条件・地形は継承元のまま。観測 55 次元・並びも
    Stage 2 (dual) と同一なので、その checkpoint がそのまま載る::

        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-v0 \
            --headless --num_envs 4096 --max_iterations 5000 \
            --load_pretrained logs/rsl_rl/k1_walk_kick_dual/<run>/model_<N>.pt

    NOTE: 1 フレーム観測の checkpoint から始めるなら
          ``--warm_start_from_single_frame`` が必要。付け忘れると actor が乱数のまま
          歩行からやり直しになる (train.py が止める)。ただし引き継ぎ元は
          **both_feet 系** に限る (共用の k1_walk_kick_360 はスロット 3 の意味が違う)。
    """

    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_phase_offset(self)
        enable_obs_history(self)
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)
        # σ_direction 0.35 → 0.15 (1500 → 3000 iteration)。
        apply_sigma_direction_anneal(self)


@configclass
class K1WalkKickDual360EnvCfg_PLAY(K1WalkKickDual360EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # σ_direction を学習終盤の値 (0.15) に固定する。カリキュラムは PLAY でも走るが
        # common_step_counter が 0 からなので、放っておくと開始値 0.35 に巻き戻る。
        hold_sigma_direction(self)


# --------------------------------------------------------------------------- #
# Stage 4 (最終): 全方位 + 観測 DR
# --------------------------------------------------------------------------- #
@configclass
class K1WalkKickDual360DREnvCfg(K1WalkKickDual360EnvCfg):
    """Stage 4 (全方位 + 観測 DR) の dual 版。**このファミリーの最終 stage**。

    Stage 3 (:class:`K1WalkKickDual360EnvCfg`) との差は
    :func:`apply_dr_stage_recipe` の 3 点だけ:

    * :func:`enable_obs_delay` — IMU / エンコーダの遅延 DR (≤ 0.02 s) と
      **ボール観測 (視覚) の遅延 DR + ノイズ拡大** (位置 ±0.07 m / 速度 ±0.5 m/s)
    * :func:`_freeze_fade_in_curricula` — 0 → 500 のランプを **除外なしで全凍結**
      (``ball_avoidance_weight`` も含む。Stage 3 で完走しているため)
    * :func:`hold_sigma_direction` — σ_direction を Stage 3 の終値 0.15 で固定

    報酬・コマンド・ボール配置・終了条件・地形は Stage 3 とまったく同じ。観測の
    次元も並びも変わらないので、Stage 3 の checkpoint がそのまま載る::

        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-DR-v0 \
            --headless --num_envs 4096 --max_iterations 10000 \
            --load_pretrained logs/rsl_rl/k1_walk_kick_dual_360/<run>/model_<N>.pt

    **ボール観測 DR は一様ノイズ + 連続遅延** (fewa 47b8863 準拠)。ガウスの認識
    パイプライン (:func:`apply_noisy_ball_pos_obs`) は不採用
    (ユーザー判断 2026-08-17。理由と戻し方はあちらの NOTE)。

    NOTE: 位置ノイズ ±0.07 m は fewa の実測上限。±0.1 m では critic が繰り返し
          発散した (:data:`_BALL_POS_NOISE` のコメント参照)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        apply_dr_stage_recipe(self)


@configclass
class K1WalkKickDual360DREnvCfg_PLAY(K1WalkKickDual360DREnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        # Unoise (ボール位置 ±0.07 / 速度 ±0.5 を含む) を切る。遅延は観測関数の側に
        # 入っているのでここでは残る (実機のレイテンシは PLAY でも起きるため)。
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
