# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウォークロングパス環境 fewa 版（5-10 m の強い転がしパス専用ポリシー）。

移植元
======
**コミット 47b8863 / ブランチ ``fewa/walk_kick_dual_encoder_tune`` の
``walk_long_pass`` をそのまま持ってきたもの。** この構成の checkpoint は
**実機で動作確認済み** (ロングパスが出る) の唯一の系統なので、原則として
1 行も変えない。ablation は Stage 4 を継承した別クラスとして
:mod:`.walk_long_pass_fewa_ablation_env_cfg` 側に置く。

このブランチ (feat/walk_kick) 側の ``walk_long_pass`` は 1 フレーム観測のまま
DR を足す別方向へ進んでいて、**checkpoint も報酬構成も繋がらない別系統**。
既存パッケージ (walk_long_pass_history / _orbit / _dr / _flag) がそちらを
import しているので、あちらには一切触らず別パッケージとして登録する。

移植で **やむを得ず変えた 2 点** (それ以外は逐語コピー):

1. ``mdp.delayed_prev_ball_pos_b`` が HEAD には無い。master 側で both_feet 系の
   2 スロット構成に作り替えられて ``delayed_ball_pos_vision_b`` になっている
   (整数ステップの設計遅延が 1 種類 → 2 種類)。こちらの観測は 55 次元・
   ``prev_ball_pos`` 1 スロットのままなので、共有コードを触らずに移植元の実装を
   このモジュール内へ持ってきた (:func:`delayed_prev_ball_pos_b`)。
   ``walk_long_pass_orbit`` が先に同じ判断をしていて、そちらと同一の実装。
   これに伴い :data:`_DELAYED_OBS_TERMS` は関数名の文字列ではなく関数オブジェクトを
   持つ (``getattr(mdp, name)`` ではローカル関数を引けないため)。挙動は同じ。
2. 共有の ``NOISY_FLAT_TERRAIN_CFG`` が HEAD では random_rough 0.9 / plane 0.1 に
   変わっている (locomotion 系の歩行チューニング由来)。47b8863 の実機実証は
   0.7 / 0.3 で回した run なので、共有定数を書き換えず
   :data:`_FEWA_NOISY_FLAT_TERRAIN_CFG` として 0.7 / 0.3 をこのパッケージに持つ。

以下は移植元の記録。失敗記録も含めてそのまま残す (帯カリキュラムや warm start の
落とし穴は、この系統でもそのまま当てはまる)。


:class:`K1WalkLoopPass360EnvCfg` を継承し、目標ボール速度の帯を (2.0, 3.0) →
(3.2, 5.0) に張り替えたもの。戦略側の要求「5-10 m 飛ぶ・誤差 ±10° のキック」の
飛距離側を担う。

距離⇔速度の換算は実機較正の 1 点 (指令 2.0 m/s → 実測 2 m ⇒ 転がり減速
a ≈ 1.0 m/s²) に基づく::

    d = v² / 2a  →  3.2 m/s → 5 m,  4.0 m/s → 8 m,  4.5 m/s → 10 m

帯の下限 3.2 は要件下限 5 m 相当。2-4.5 m の弱いパスは loop_pass_360 が引き続き
担い、戦略側が距離帯でポリシーを選ぶ 2 本立てとする（帯を重ねない理由は
walk_pass と同じ: それぞれが狭い速度域に集中できる）。

すくい型のスイングは維持する
----------------------------
kick_elevation (φ_target=30°) は sim で φ≈27° を現役で誘導しており
(loop_pass_360 run の実測: kick_elevation_deg ≈ 26.8°, 報酬 ≈ 0.157)、この
「浮き球報酬による意図的な威力劣化」が実機で威力指令が意味を持つ土台になっている。
実機では仰角が sim より目減りしてほぼ浮かない柔らかいパスになる、という sim2real
特性まで含めて検証済みの動作ファミリーなので、**帯を上げるだけで蹴り方には触らない**:

* kick_elevation (Gaussian 30°) … そのまま
* kick_velocity_scaled の use_3d_speed=True … そのまま（elevation との綱引き防止）
* ball_avoidance / 360° サンプリング / episode_length 15 s … そのまま

5 m/s はすくい型の実証済み射程内 (loop_shoot が同型のスイングで v3d 6.3-6.5 m/s)。
学習後に kick_vel_ratio が帯上限側だけ垂れるようなら、そのとき初めて elevation の
weight を下げる等を検討する。

帯は一段で上げてはいけない（失敗記録）
--------------------------------------
初版は ``target_speed_range`` をいきなり (2.0, 3.0) → (3.2, 5.0) に張り替えていたが、
**1200 iteration で「蹴らずに 15 秒歩き回る」に完全収束した**
(run 2026-08-09_09-56-21: kick_rate 0.997 → 0.037、time_out 0.944、
ball_touch_count 1.30 → 0.91、Episode_Reward/kick_direction 0.213 → 0.001)。

機序は報酬の収支逆転:

* ``kick_finished`` は latch の 2 秒後にエピソードを終わらせるので、キックは常に
  **「残りの歩行報酬 (feet_phase など) を捨てる」コスト**を払っている。
* 継承元の実蹴りは約 2.4 m/s。新しい帯で採点すると、v_gate × f_vel は
  指令 3.2 で 0.38、4.1 で 0.009、**5.0 で ~0**。帯の大半でキック報酬が消える。
* 結果、蹴らない方が得になる (mean_reward はむしろ 12.1 → 21.5 に増えた)。
  探索 std も 0.3 → 0.08 に潰れ、iteration を増やしても抜けられない局所最適。

そこで帯は :func:`~..walk_kick.mdp.curriculums.kick_rate_gated_speed_range` で
継承元が満点を取れる (2.0, 3.0) から目標の (3.2, 5.0) へ動かす。各時点で
「今のポリシーがぎりぎり届く速度」が指令されるのでキック報酬が払われ続ける。

帯は壁時計ではなく **kick_rate で開閉するゲート** で進める。壁時計の線形ランプ
(``linear_command_speed_range``) は「ポリシーがこの速さで付いてくる」という賭けで、
外れたときに自己回復しない: 帯が実力を追い越すとキック報酬が消え、上の収支逆転が
起きて「蹴らない」に落ちるが、帯はそのまま上がり続けるので戻る道が無い。
ゲート版は kick_rate が落ちれば帯を止め、崩れ続ければ蹴れていた帯まで戻すので、
報酬信号が復活してポリシーが自力で戻れる。

**この構造は帯を触る限り常について回る。** 帯を再調整するときは、必ず
「継承元の実蹴り速度で採点したときにキック報酬が残るか」を先に検算すること。

fine-tune ではキック報酬をフェードインさせてはいけない
------------------------------------------------------
継承元は「キック報酬 weight を 0 → 500 iteration で立ち上げる」カリキュラムを
持っている。ゼロから学習する段では正しい順序だが、**既に蹴れるポリシーから始める
Stage 4 では最初の 500 iteration がそのまま「蹴らない方が得」の期間になる**
(歩行報酬は満額・キック報酬は ~0・キックは kick_finished で残りの歩行報酬を捨てる
コストを払う)。帯ではなく weight 側で同じ収支逆転が起きる。

そのため Stage 4 では :func:`_freeze_fade_in_curricula` でこれらのランプを終値の
定数に潰し、1 iteration 目から満額のキック報酬を払う。

Stage 4 が「歩くだけ」になったときの切り分け順序 (失敗記録 2026-08-11)
----------------------------------------------------------------------
run 2026-08-11_16-31-27 は 5000 iteration を丸ごと潰した。原因は報酬設計ではなく
**actor が引き継がれていなかった**こと。共用タスク (1 フレーム観測) の
loop_pass_360 checkpoint を ``--warm_start_from_single_frame`` 無しで渡したため、
critic と正規化統計と std だけが載り、actor は乱数のまま学習が始まっていた
(model_0.pt の actor.mlp.0.weight は全列が初期化スケール ±0.053 のまま)。
歩行からやり直しなので当然キックは出ず、その間に帯のランプだけが壁時計で
進み切っていた (iteration 1570 で speed_max が 5.0 に到達)。

したがって切り分けは **報酬より先に「本当に継承できているか」** を見ること:

1. 起動ログの ``Loaded N tensors`` / ``Skipped N tensors``。actor.* が Skipped 側に
   並んでいたら、そこで止めて引数を直す (現在は train.py が例外で止める)。
2. 最初の 20 iteration の ``Episode_Termination/base_height`` と
   ``Train/mean_episode_length``。継承できていれば base_height は 0.03 前後、
   episode length は 200 以上で、kick_rate は 15 iteration ほどで 0.98 に戻る。
   base_height が 0.8 まで上がるなら、それは「歩けていない」のであって
   「蹴らない」のではない。
3. ``Policy/mean_noise_std``。``--reset_noise_std 0.3`` は継承元の収束 std
   (0.06-0.095) の 3-5 倍で、**これだけで歩行が壊れる**。warm start を正しく効かせた
   うえで 0.3 を入れた run は 12 iteration で base_height 終了 0.82・kick_rate 0.19
   止まりだったが、std リセットを外すと 14 iteration で kick_rate 0.99 に戻った。
   Stage 4 の既定は「std リセットなし」。探索を足すなら 0.12 程度から。

観測を 50 フレームの履歴にする
------------------------------
policy 観測 (内界センサ + 前ステップの行動 + 受信コマンドの 55 次元) を毎ステップ
50 フレーム分バッファし、actor には

* 直近 5 フレームをそのまま (5 × 55 = 275)
* 50 フレーム全部を 1D-CNN で符号化した潜在 (16 × 6 = 96)

を連結して入れる (合計 371 次元)。CNN は隠れ層 2 つで
[kernel, filter, stride] = [6, 32, 3], [4, 16, 2]。実装は
:class:`~..locomotion.networks.ActorCriticHistoryCNN`、形の指定は
``agents/rsl_rl_ppo_cfg.py``。critic はこれまでどおり 1 フレームの特権観測を見る。

**共用タスクの checkpoint とは互換性が無い。** actor の MLP の入力次元と重みの名前が
変わるので、共用の loop_pass_360 (``k1_walk_loop_pass_360``) から
``--load_pretrained`` すると引き継がれるのは critic・観測正規化の統計・action noise
std の 17 テンソルだけで、**actor は初期化されたまま**学習が始まる
(実測: actor.{0,2,4,6}.{weight,bias} の 8 本が "not in model" で落ちる)。
帯カリキュラムは **継承元が既にキックできる**ことを前提に組んであるので
(上の失敗記録参照)、actor がゼロからでは意味が変わる。

そのため Stage 1-3 も履歴入力版を別タスクとして用意し、歩行から通しで学習し直す
(:mod:`.walk_long_pass_fewa_stages_env_cfg`)。段が繋がったかは起動ログの
"Skipped N tensors" が 0 本かどうかで確認できる (train.py は形の合わないテンソルを
黙って捨てるので、ログを見ない限り気づけない)。

ONNX の入力も (1, 50, 55) に変わる。実機側は 55 次元の観測をリングバッファに
古い順で積んで渡す (詳細は :mod:`..locomotion.networks.actor_critic_history_cnn`
のモジュール docstring)。

学習手順 (Stage 1-4 を通しで)::

    ./scripts/rsl_rl/train_walk_long_pass_fewa.sh
    # Stage 4 だけやり直す場合 (履歴入力版 Stage 3 の checkpoint から):
    _labpython2 scripts/rsl_rl/train.py \
        --task Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-v0 \
        --headless --num_envs 4096 \
        --load_pretrained logs/rsl_rl/k1_walk_long_pass_fewa_loop_360/<run>/model_<N>.pt
    # 共用タスク (1 フレーム観測) の Stage 3 checkpoint から始める場合:
    #   --warm_start_from_single_frame を必ず付ける (無いと actor が引き継がれない)。

--resume ではなく --load_pretrained を使うこと（理由は walk_pass_env_cfg の
docstring 参照）。

``--reset_noise_std`` は **付けない**。以前は「収束済みポリシーは std が潰れていて
4-5 m/s を探索できないから必須」としていたが、0.3 は継承元の収束 std (0.06-0.095) の
3-5 倍で、実測ではキック以前に歩行が壊れる (上の切り分け 3.)。速度域の探索は帯の
ゲート付きカリキュラムが「今の実力の少し上」を指令し続けることで担保している。
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from ..walk_kick import mdp
from ..walk_kick.mdp.observations import _delayed_signal, prev_ball_pos_b
from ..walk_kick.walk_kick_env_cfg import _KICK_STATE_PARAMS
from ..walk_loop_pass.walk_loop_pass_env_cfg import K1WalkLoopPass360EnvCfg

# --------------------------------------------------------------------------- #
# 段差・坂道なし、ランダムノイズのみの軽く凹凸した地面 (47b8863 の値)
#
# 共有の :data:`~..locomotion.flat_env_cfg.NOISY_FLAT_TERRAIN_CFG` と同じレシピだが、
# サブ地形の比率だけ **移植元の 0.7 / 0.3** で固定する。HEAD 側の共有定数は
# locomotion 系の歩行チューニングで 0.9 / 0.1 に変わっており、実機で動いた run の
# 条件と違う。共有定数を戻すと walk_kick_dual / walk_short_pass / walk_long_pass_orbit
# まで道連れになるので、このパッケージ内に自前で持つ。
#
# 平面の比率が効くのは「凹凸を観測できない (height_scanner は None) 状態で、
# ときどき完全な平地も経験する」割合。0.3 は 47b8863 で実機に載った値。
# --------------------------------------------------------------------------- #
_FEWA_NOISY_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    curriculum=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.7,
            noise_range=(0.01, 0.04),
            noise_step=0.01,
            border_width=0.25,
        ),
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
    },
)


# --------------------------------------------------------------------------- #
# ロングパス用の目標ボール速度レンジ [m/s]（3D ノルム基準）
#
# 下限 3.2 = 要件下限 5 m 相当 (a=1.0)。それ未満は loop_pass_360 の担当なので重ねない。
# 上限 5.0 = 10 m + α。ハード壁 (関節速度上限から出る足先速度 ≈ 6.5 m/s、loop_shoot
# 実測) に対して余裕があるので、帯全域が到達可能 = 威力指令が全域で意味を持つ。
#
# NOTE: a=1.0 は 1 点計測。実機で --kick_speed 3.0 を測って a を確定させたら、
#       この帯を締め直すこと (a=0.7 なら 10 m に 3.74 m/s で足りる)。
# NOTE: これは **カリキュラムの終点**。開始時は _LONG_PASS_SPEED_RANGE_START。
#       一段で張り替えると「蹴らない」に収束する (モジュール docstring の失敗記録参照)。
# --------------------------------------------------------------------------- #
_LONG_PASS_SPEED_RANGE = (3.2, 5.0)

# --------------------------------------------------------------------------- #
# 帯カリキュラムの開始点と区間
#
# 開始点は継承元 (loop_pass_360) の帯そのもの。そこでは実蹴り 2.4 m/s に対して
# kick_vel_ratio 0.866 / kick_rate 0.997 が出ているので、fine-tune の 1 iteration 目から
# キック報酬が満額で払われ、「蹴る」行動が維持される。
#
# 区間 500 → 3000 iteration:
#   * 500 まで待つのは、履歴入力 (CNN 潜在の列がゼロから育つ) に慣れるまでの
#     落ち着き期間。実測では 15 iteration ほどで kick_rate 0.99 に戻るので十分な余裕。
#     (キック報酬 weight のランプと重ねない、という元の理由は _freeze_fade_in_curricula
#      でランプ自体を消したので、もう当てはまらない)
#   * 2500 iteration かけるのは、帯の上限が 3.0 → 5.0 と 1.7 倍に伸びるため。
#     1 iteration あたり 0.0008 m/s ずつで、ポリシーが追従する余裕を取る。
#   * これは **公称** の速さ。実際は kick_rate ゲートが開いている間しか進まないので、
#     3000 で終点に着くとは限らない。終了時に Curriculum/kick_speed_range/alpha が
#     1.0 か必ず確認すること。ITER は 5000 以上。
# --------------------------------------------------------------------------- #
_LONG_PASS_SPEED_RANGE_START = (2.0, 3.0)
_SPEED_RAMP_START_ITER = 500
_SPEED_RAMP_END_ITER = 3000

# --------------------------------------------------------------------------- #
# 帯ゲートのしきい値 (kick_rate の EMA)
#
# 継承元 (loop_pass_360) の収束値は kick_rate ≈ 0.99。帯を進めてよいのは
# 「今の帯でちゃんと蹴れている」間だけなので、0.80 を上回っている間だけ進める。
# 0.50 を下回ったら **帯を戻す**: キック報酬が復活するので、蹴らなくなった状態から
# 自力で戻ってこられる (壁時計の線形ランプにはこの復路が無い)。
# 間の 0.50-0.80 は据え置き (ヒステリシス。しきい値付近での往復を防ぐ)。
#
# retreat_scale = 2.0 は「崩れたら進んだ倍の速さで戻る」。崩れた帯に長く留まるほど
# 「蹴らない」方の解が固まるので、復路は往路より速くしておく。
# --------------------------------------------------------------------------- #
_SPEED_GATE_ADVANCE_ABOVE = 0.80
_SPEED_GATE_RETREAT_BELOW = 0.50
_SPEED_GATE_RETREAT_SCALE = 2.0

# --------------------------------------------------------------------------- #
# カリキュラムが 1 iteration とみなす env step 数
#
# curriculums.py の start_step / end_step は
# ``env.common_step_counter // steps_per_iteration`` で iteration に換算される。
# common_step_counter は env.step() 1 回で 1 増えるので、この値は PPO の
# ``num_steps_per_env`` と一致していなければならない
# (locomotion/agents/rsl_rl_ppo_cfg.py: num_steps_per_env = 48)。
#
# NOTE: 継承元 (walk_kick_env_cfg の _spi) は 24 のままで、コメントには
#       「steps_per_iteration = num_steps_per_env」と書いてあるのに実際の
#       num_steps_per_env は 48。つまり walk_kick 系のランプは全部、書いてある
#       iteration 数の **半分** で終わっている (失敗 run で帯が iteration 3000 では
#       なく 1570 で 5.0 に到達していたのはこれが理由)。既に収束している
#       Stage 1-3 の挙動を変えないよう継承元は触らず、このタスクの中だけ揃える。
# --------------------------------------------------------------------------- #
_STEPS_PER_ITERATION = 48

# --------------------------------------------------------------------------- #
# 値 latch のトリガー速度 [m/s]
#
# 継承値 0.8 は帯 (3.2-5.0) に対して低すぎる。接近中に足がかすっただけ (v_ball ≈ 1)
# で latch が不可逆に成立し、2 秒後に kick_finished がエピソードを終わらせるため、
# 「ちゃんと蹴った経験」がサンプリングされなくなる (walk_pass が 3000 iteration
# 潰した機序と同じ)。1.5 ならかすり当てでは latch せず、蹴り直しのチャンスが残る。
#
# 2.5 まで上げないのは fine-tune の出発点への配慮。開始時のポリシー
# (loop_pass_360) の実蹴りは 2.15-2.6 m/s (指令 2.0-3.0 × ratio 0.86) なので、
# それを上回る閾値にすると初期の正当な蹴りが latch せず、報酬信号が消えて
# fine-tune 自体が立ち上がらない。1.5 < 2.15 で初期の蹴りは全て拾える。
#
# NOTE: kick_state はステップ単位でキャッシュされ、最初に呼んだ項の v_thresh で
#       その step の状態が確定する。全ての項に同じ値を配る必要があるため、
#       :func:`_apply_v_thresh` でまとめて上書きしている (walk_pass と同じ手順)。
# --------------------------------------------------------------------------- #
_LONG_PASS_V_THRESH = 1.5

# --------------------------------------------------------------------------- #
# kick_velocity_scaled の速度シェイピング係数 [m/s]
#
# 継承値 0.7 は帯幅 1.0 (2.0-3.0) 用。帯幅 1.8 に広げるぶん少し緩めて、帯の端の
# 指令でも勾配が届くようにする。0.9 で帯の両端 (3.2 vs 5.0) は 2σ 離れるので
# 指令の識別は保たれる。
# --------------------------------------------------------------------------- #
_LONG_PASS_SIGMA_VELOCITY = 0.9

# --------------------------------------------------------------------------- #
# 項1 (kick_direction) の片側速度ゲート
#
# 項1 は重み最大 (6.0) なのに素の定義では速度非依存なので、v_thresh (1.5) を
# ぎりぎり超えるだけの弱い蹴りでも方向さえ合っていれば満点が出てしまう。
# g(v) = sigmoid((v_ball − 0.6·v_target) / 0.3) を掛けて弱すぎる蹴りを削る。
# 片側ゲートなので蹴りすぎは削らない (それは項2 の仕事)。
#
# 0.6 は walk_pass の 0.8 より緩い。fine-tune 開始時の実蹴り 2.2-2.6 m/s が
# v_target=3.2 のゲート (1.92) を十分通過できるようにするため
# (0.8 だと 2.56 で、初期の蹴りの大半が半減以下になり立ち上がりが遅れる)。
# sigma_gate 0.3 も同じ理由で walk_pass の 0.05 より緩い (速度スケールが 6-10 倍)。
# --------------------------------------------------------------------------- #
_LONG_PASS_V_GATE_FRAC = 0.6
_LONG_PASS_SIGMA_GATE = 0.3

# --------------------------------------------------------------------------- #
# Actor が見る観測履歴の長さ [frames]
#
# policy 観測 (55 次元: 内界センサ + 前ステップの行動 + 受信コマンド) を毎ステップ
# H フレーム分バッファし、actor には
#   * 直近 5 フレームをそのまま (5 × 55 = 275)
#   * H フレーム全部を 1D-CNN で符号化した潜在
# を入れる。ネットワーク側は
# :class:`~..locomotion.networks.ActorCriticHistoryCNN` (agents/rsl_rl_ppo_cfg.py で
# 指定) が担当し、直近フレーム数と CNN の形はそちらの定数で決まる。
#
# H を変えると CNN の出力系列長 = 潜在次元 = actor MLP の入力次元が変わる
# ([kernel, stride] = [6, 3], [4, 2] のとき L1 = (H-6)//3 + 1, L2 = (L1-4)//2 + 1、
#  潜在 = 16 × L2):
#
#   H = 50  → L1 = 15, L2 = 6  → 潜在  96 → actor 入力 371
#   H = 100 → L1 = 32, L2 = 15 → 潜在 240 → actor 入力 515
#
# 0.02 s/step なので H = 100 は 2 秒分の履歴。キックは接近から latch まで 1 秒以上
# かかるので、1 秒 (H=50) だと踏み込みの序盤がバッファから落ちる。
#
# **H を変えた checkpoint 同士は繋がらない。** actor MLP の 1 層目の形が変わるので、
# --load_pretrained では入力層が黙って捨てられる (train.py が actor を 1 本も
# 引き継げなければ止めるようになっている)。1 フレーム観測の checkpoint からなら
# H に関係なく --warm_start_from_single_frame で移植できる (最新フレームの列に
# 置いて残りをゼロにするだけなので、潜在の次元が変わっても成立する)。
#
# flatten_history_dim = False が必須。True にすると ObservationManager は
# **項ごとに** (H, d_i) を平坦化してから連結するので、並びが
# [gravity の H フレーム][ang_vel の H フレーム]... になり、フレーム単位の
# (N, H, 55) には戻せない (CNN のチャンネル = 観測次元にできなくなる)。
# False なら各項が (N, H, d_i) のまま最終軸で連結され、(N, H, 55) が得られる。
#
# 履歴は環境リセットで巻き戻り、直後は現在フレームで H 個すべてが埋まる
# (IsaacLab の CircularBuffer の仕様)。ゼロ詰めの履歴は入らない。
#
# NOTE: ONNX の入力形状も (1, H, 55) になる。H を変えたら実機側のリングバッファ長も
#       合わせること。
# NOTE: critic はこれまでどおり 1 フレーム観測 (特権情報付き) のまま。critic 側にも
#       履歴を付ける場合は ActorCriticHistoryCNN の critic 側も直す必要がある。
# NOTE: この定数は Stage 1-3 の履歴入力版タスク
#       (:mod:`.walk_long_pass_fewa_stages_env_cfg`) からも読まれる。全 stage で同じ値に
#       しないと段の間で checkpoint が繋がらないので、ここ 1 箇所で管理している。
# --------------------------------------------------------------------------- #
_OBS_HISTORY_LENGTH = 100


def enable_obs_history(cfg) -> None:
    """policy 観測グループを :data:`_OBS_HISTORY_LENGTH` フレームの履歴にする。

    Stage 1-3 の履歴入力版 (:mod:`.walk_long_pass_fewa_stages_env_cfg`) からも呼ぶので、
    設定は必ずここに 1 箇所にまとめる。片方だけ履歴長を変えると stage 間で
    checkpoint が繋がらなくなり、しかも起動時には気づけない
    (train.py が形の合わないテンソルを黙って捨てるため)。
    """
    cfg.observations.policy.history_length = _OBS_HISTORY_LENGTH
    cfg.observations.policy.flatten_history_dim = False


# 着地の作り込み系の shaping 項。long_pass 系列 (Stage 1-4) では全段で無効化する。
#
#   feet_landing_impact : 接地瞬間の接地力ノルムを罰する  (weight -0.5e-2)
#   feet_landing_vel    : 接地瞬間の足速度を罰する        (weight -1.5)
#   feet_heel_strike    : 接地瞬間のかかと下がり姿勢を賞する (weight +5.0)
#
# いずれも「接地イベントの瞬間」にだけ効くスパースな項で、蹴りとは相性が悪い。
# 蹴り足のスイングは接地を強く速くする方向にしか作れないので、蹴るほど
# feet_landing_impact / feet_landing_vel の罰が増える。feet_heel_strike も
# 蹴り足の踏み込みがかかと下がりになりにくいぶん、蹴ると取り逃す報酬になる。
# 結果として「蹴らずに丁寧に歩く」方が得になる圧が常時かかる。
#
# NOTE: 消すのは long_pass 系列だけ。これらは共用の K1FlatEnvCfg で定義されており、
#       元を消すと walk_kick / walk_pass / walk_lob / walk_mid_kick / loop_shoot と
#       コミット済みの基準 checkpoint まで道連れになる。
_LANDING_SHAPING_TERMS = (
    "feet_landing_impact",
    "feet_landing_vel",
    "feet_heel_strike",
)


# 蹴りのある段 (Stage 2-4) での feet_phase の weight。継承元は 2.0。
#
# 経緯 (実測):
#   着地 shaping 3 項を消したことで Stage 1 の歩容が大きく良くなり、位相追従が
#   ほぼ満点に張り付くようになった。その結果 Stage 2 に入った時点で
#   Episode_Reward/feet_phase が 0.62 (旧) → 1.71 (新) と 2.7 倍になり、
#   キックが払うコスト (kick_finished で latch の 2 秒後にエピソードが終わるので、
#   キックは常に「残りの歩行報酬を捨てる」コストを払う。冒頭の「帯は一段で
#   上げてはいけない」節と同じ機序) がそのぶん膨らんだ。
#
#   実測: Stage 2 を 4000 iteration 回して kick_rate は 0.19-0.28 で完全に停滞し、
#   kick_elevation_deg は 0.97° (旧 run は同 iteration で kick_rate 0.99 /
#   elevation 15.6°)。それでいて mean_reward は 12.5 対 7.7 で新 run の方が高い。
#   つまり「蹴らずに歩く」が正しく最適解になっていた。
#
#   歩行報酬の総量を、実際に蹴りを獲得できていた頃の水準に戻す。
#   2.0 * (0.62 / 1.71) ≒ 0.73 なので 0.8 を採る。
#
# NOTE: Stage 1 (歩行のみ) では下げない。歩容そのものはこの項で作っており、
#       Stage 1 で下げると土台が崩れる。下げるのは蹴りと綱引きになる段だけ。
_KICK_STAGE_FEET_PHASE_WEIGHT = 0.8


def rebalance_gait_vs_kick(cfg) -> None:
    """蹴りのある段で feet_phase の weight を下げる (:data:`_KICK_STAGE_FEET_PHASE_WEIGHT`)。

    Stage 2-4 から呼ぶ。Stage 1 からは呼ばない。
    """
    term = getattr(cfg.rewards, "feet_phase", None)
    if term is not None:
        term.weight = _KICK_STAGE_FEET_PHASE_WEIGHT


def disable_landing_shaping(cfg) -> None:
    """着地 shaping 3 項 (:data:`_LANDING_SHAPING_TERMS`) を無効化する。

    Stage 1-4 の全段から呼ぶ。段によって有無が変わると、前段が獲得した歩容が
    次の段で罰せられる (あるいはその逆) ことになるので、必ず全段で揃えること。

    報酬項の増減は観測にも行動にも影響しないので、checkpoint の引き継ぎには
    影響しない (actor / critic の形は変わらない)。
    """
    for name in _LANDING_SHAPING_TERMS:
        # 親の構成が変わって項が無くなっていても落ちないように getattr でガードする
        # (kick 系の項を None にしている他 cfg と同じ扱い)。
        if getattr(cfg.rewards, name, None) is not None:
            setattr(cfg.rewards, name, None)


# kick_state を参照する報酬項（v_thresh を配る対象）。
# この cfg で None の項 (kick_velocity_strong / approach_penalty) も名前だけ挙げて
# おき、getattr の None ガードで飛ばす (親の構成が変わっても取りこぼさないように)。
_KICK_STATE_REWARD_TERMS = (
    "kick_direction",
    "kick_velocity_scaled",
    "kick_velocity_strong",
    "kick_elevation",
    "walk_speed",
    "approach_penalty",
    "ball_avoidance",
    "kick_pose_overshoot",
    "extra_ball_touch",
    # ablation D (Grounded) だけが実際に足す項。名簿に載せておくのは
    # _apply_v_thresh の配布先にするためで、他の変種では属性が無いので
    # getattr の None ガードで飛ばされる (kick_velocity_strong と同じ扱い)。
    #
    # NOTE: 名簿に載っていても **_apply_v_thresh を呼ぶ時点で属性が無ければ配れない**。
    #       ablation D は基底の __post_init__ (= -- 5 の配布) が終わった後に項を
    #       足すので、足した直後にもう一度 _apply_v_thresh を呼んでいる。
    "kick_plant_grounded",
)


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
# --------------------------------------------------------------------------- #
_OBS_DELAY_MAX_S = 0.02

# --------------------------------------------------------------------------- #
# ボール観測 (視覚由来) の遅延上限 [s]
#
# 本体センサ (IMU / エンコーダ) と分けてあるのは、実機での出所が違うため。
# ボールはカメラ → 検出 → 座標変換のパイプラインを通るので、実機のレイテンシは
# 内界センサより **明確に大きい** (一般的なヒューマノイドで 30-60 ms = 1.5-3 ステップ)。
#
# 実効遅延は **2 項とも 0.02-0.08 s** になるよう組んである:
#
#   prev_ball_pos : 0.02 (prev_ball_pos_b の固定 1 ステップ) + [0, 0.06] = 0.02-0.08
#   ball_vel      : 0.02 (_BALL_OBS_BASE_DELAY_S)            + [0, 0.06] = 0.02-0.08
#
# ball_vel 側に固定ぶんを足しているのは、prev_ball_pos が設計上すでに 1 ステップ
# 遅れているため。同じカメラフレームから出る 2 つの量のレイテンシが 1 ステップ
# ずれているのは実機ではあり得ないので揃える。vision group で乱数を共有するので、
# ランダム成分も 2 項で同じ値になる。
#
# 0.08 s = 4 ステップ。実機のカメラ遅延を測った結果で調整すること。
# ball_vel だけ従来どおり [0, 0.06] に戻したいなら _BALL_OBS_BASE_DELAY_S = 0.0。
# --------------------------------------------------------------------------- #
_BALL_OBS_DELAY_MAX_S = 0.06
_BALL_OBS_BASE_DELAY_S = 0.02

# --------------------------------------------------------------------------- #
# ボール観測のノイズ (一様、±この値)
#
# 継承値は位置 ±0.02 m / 速度 ±0.1 m/s。実機のカメラ由来の量としては楽観的すぎる
# (とくに速度はフレーム間差分から求めるので、位置ノイズを dt = 0.02 s で微分した
#  ぶんのばらつきが原理的に乗る)。
#
# 位置 ±0.07 m はボール半径 (0.11 m) の 6 割ほど。±0.1 m (半径とほぼ同じ) で回した
# run では帯が 5.0 に乗った直後 (iteration 3000 付近) が性能のピークで、その後は
# critic の value loss が繰り返し発散して崩壊と自己回復を繰り返した。位置ノイズを
# 少し緩めて、その不安定さがボール位置の観測ノイズ由来かどうかを見る。
#
# 速度 ±0.5 m/s は指令帯 (3.2-5.0) の 10-15%。ボール速度観測に頼った蹴り分けは
# ほぼできなくなるが、そもそも実機で信用できない量なので頼らせない方が正しい。
# --------------------------------------------------------------------------- #
_BALL_POS_NOISE = 0.07
_BALL_VEL_NOISE = 0.5


def delayed_prev_ball_pos_b(
    env,
    max_delay_s: float,
    group: str = "vision",
    base_delay_s: float = 0.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
):
    """視覚由来のボール位置を遅延させたもの。

    移植元 (47b8863) では ``walk_kick.mdp.observations`` に同名の関数があったが、
    このブランチ側では both_feet 系の 2 スロット構成に作り替えられて
    :func:`~..walk_kick.mdp.observations.delayed_ball_pos_vision_b` になっている
    (整数ステップの設計遅延が 1 ではなく 1/2 の 2 種類になった)。こちらの系統の観測は
    55 次元・``prev_ball_pos`` 1 スロットのままなので、共有コードを触らずに
    移植元の実装をこのパッケージ内へ持ってきてある
    (``walk_long_pass_orbit`` が先に同じ判断をしていて、実装もそちらと同一)。

    NOTE: :func:`~..walk_kick.mdp.observations.prev_ball_pos_b` は **元から 1 ステップ
          (0.02 s) 遅らせた値** を返す (知覚遅延を模す設計。関数名の "prev" がそれ)。
          ここでの遅延はその上に載るので、この項の実効遅延は
          ``0.02 + [0, max_delay_s]`` 秒。``base_delay_s`` はここでは 0 のまま使うこと
          (二重に足すことになる)。
    """
    value = prev_ball_pos_b(env, ball_cfg=ball_cfg)
    return _delayed_signal(env, "prev_ball_pos_b", group, value, max_delay_s, base_delay_s)


# 遅延を掛ける policy 観測項と、遅延量を共有するセンサ (group)。
#
# group が同じ項は同じ遅延量を引く。projected_gravity と base_ang_vel は同じ IMU、
# joint_pos と joint_vel は同じエンコーダ読み出しから来るので、独立に遅れることは
# 物理的にあり得ない (重力は最新なのに角速度だけ 1 ステップ古い、など)。
# ボール 2 項も同じカメラフレームから出るので "vision" で共有する。
#
# 4 要素目は max_delay_s を引く元の定数名 ("body" = 内界センサ, "ball" = 視覚)。
#
# NOTE: 移植元は関数「名」を持って ``getattr(mdp, name)`` で引いていたが、
#       prev_ball_pos だけこのモジュールのローカル関数になったので (上の
#       :func:`delayed_prev_ball_pos_b`)、関数オブジェクトを直接持つ形に変えてある。
#       挙動は同じ。
_DELAYED_OBS_TERMS = (
    ("projected_gravity", "imu", mdp.delayed_projected_gravity, "body"),
    ("base_ang_vel", "imu", mdp.delayed_base_ang_vel, "body"),
    ("joint_pos", "encoder", mdp.delayed_joint_pos_rel, "body"),
    ("joint_vel", "encoder", mdp.delayed_joint_vel_rel, "body"),
    ("ball_vel", "vision", mdp.delayed_ball_vel_b, "ball"),
    ("prev_ball_pos", "vision", delayed_prev_ball_pos_b, "ball"),
)


def enable_obs_delay(
    cfg,
    max_delay_s: float = _OBS_DELAY_MAX_S,
    ball_max_delay_s: float | None = None,
) -> list[str]:
    """policy 観測の IMU / エンコーダ由来の項にセンサ遅延 DR を掛ける。

    既存の ObsTerm の ``func`` と ``params`` だけを差し替える。項を作り直さないのは

    * ノイズ設定 (Unoise) と asset_cfg をそのまま引き継ぐため
    * 観測の **並び順を絶対に変えない** ため (policy 観測 55 次元は順序が
      実機側の詰め方と 1 対 1 で対応している。configclass のフィールド順で
      連結されるので代入では順序は変わらないが、項を消して足し直すと変わる)

    観測次元は変わらないので、遅延を入れる前後で checkpoint はそのまま繋がる。

    critic には掛けない。特権情報から価値を推定させる側なので、遅延させても
    学習が難しくなるだけで得るものが無い。

    Stage 1-3 (:mod:`.walk_long_pass_fewa_stages_env_cfg`) から呼ぶこともできるが、
    その場合は全 stage で同じ ``max_delay_s`` にすること (遅延の有無で歩き方が
    変わるので、段の間で条件が変わると引き継ぎの意味が薄れる)。

    Args:
        max_delay_s: 内界センサ (IMU / エンコーダ) の遅延上限 [s]。
        ball_max_delay_s: ボール観測 (視覚) の遅延上限 [s]。None なら
            :data:`_BALL_OBS_DELAY_MAX_S`。実機のカメラ遅延は内界センサより
            大きいので別パラメータにしてある。

    Returns:
        遅延を掛けた観測項の名前。
    """
    if ball_max_delay_s is None:
        ball_max_delay_s = _BALL_OBS_DELAY_MAX_S

    applied: list[str] = []
    for term_name, group, func, source in _DELAYED_OBS_TERMS:
        term = getattr(cfg.observations.policy, term_name, None)
        if term is None:
            raise AttributeError(
                f"policy 観測に '{term_name}' がありません。遅延 DR の対象項名が"
                " 継承元の観測構成と食い違っています。"
            )
        delay = ball_max_delay_s if source == "ball" else max_delay_s
        term.func = func
        term.params = {**term.params, "max_delay_s": delay, "group": group}
        # prev_ball_pos は固定 1 ステップを元から持っているので base は足さない。
        # ball_vel だけ同じぶんを足して、同じカメラ由来の 2 項の実効遅延を揃える。
        if term_name == "ball_vel":
            term.params["base_delay_s"] = _BALL_OBS_BASE_DELAY_S
        applied.append(term_name)

    # ボール観測のノイズを実機のカメラ由来の量に合わせて広げる
    # (_BALL_POS_NOISE / _BALL_VEL_NOISE のコメント参照)。
    cfg.observations.policy.ball_vel.noise = Unoise(n_min=-_BALL_VEL_NOISE, n_max=_BALL_VEL_NOISE)
    cfg.observations.policy.prev_ball_pos.noise = Unoise(n_min=-_BALL_POS_NOISE, n_max=_BALL_POS_NOISE)
    return applied


def _freeze_fade_in_curricula(cfg, before_iter: int) -> list[str]:
    """``before_iter`` までに立ち上がりきる報酬重みのランプを、終値の定数に潰す。

    継承元 (walk_kick → walk_loop_pass → 360) では、キック報酬の weight を 0 → 500
    iteration でフェードインさせている。ゼロから学習する段では「まず歩けるように
    なってから蹴りを乗せる」ための正しい順序だが、**既に蹴れるポリシーからの
    fine-tune である Stage 4 では逆に働く**:

    * 歩行報酬 (feet_phase / track_lin_vel など) は 1 iteration 目から満額。
    * キック報酬は weight ≈ 0 から始まる。
    * ``kick_finished`` は latch の 2 秒後にエピソードを終わらせるので、キックは
      「残りの歩行報酬を捨てる」コストだけを払っている。

    つまり最初の 500 iteration は「蹴らない方が得」が明示的に成立していて、PPO は
    その間ずっと継承したキックを消す方向に更新される。500 iteration 後に weight が
    戻ってきても、そのときには蹴らなくなった後なので報酬が payout されない
    (モジュール docstring の失敗記録と同じ収支逆転が、帯ではなく weight 側で起きる)。

    Stage 4 の出発点は「その weight で既に収束したポリシー」なので、終値をそのまま
    1 iteration 目から掛けるのが正しい。``start_weight`` を ``end_weight`` に揃える
    ことで、ランプ項を消さずに定数化する (どの項がいくつなのかがログに出たままになる)。

    ``before_iter`` より後にランプする項 (例: extra_ball_touch) は意図的に後段へ
    置かれているので触らない。

    Returns:
        定数化した curriculum term の名前 (呼び出し側で表示・検証に使う)。
    """
    frozen: list[str] = []
    for name in dir(cfg.curriculum):
        if name.startswith("_"):
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


def _apply_v_thresh(cfg: "K1WalkLongPassFewaEnvCfg", v_thresh: float) -> None:
    """kick_state を共有する全ての項に同じ v_thresh を配る。

    1 つでも取りこぼすと、その step で最初に評価された項の値で latch 状態が確定し、
    結果が報酬項の評価順に依存してしまう (walk_pass の同名ヘルパーと同じ)。
    報酬項を追加し終えた後に呼ぶこと。
    """
    cfg.commands.base_velocity.v_thresh = v_thresh
    cfg.terminations.kick_finished.params["v_thresh"] = v_thresh
    for _name in _KICK_STATE_REWARD_TERMS:
        _term = getattr(cfg.rewards, _name, None)
        if _term is not None:
            _term.params["v_thresh"] = v_thresh


@configclass
class K1WalkLongPassFewaEnvCfg(K1WalkLoopPass360EnvCfg):
    """ロングパス専用。

    観測項・行動空間・蹴り方の報酬は Walk-Loop-Pass-360 と同一だが、actor だけは
    50 フレームの観測履歴を見る (モジュール docstring の「観測を 50 フレームの
    履歴にする」節)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 0. Actor の観測を _OBS_HISTORY_LENGTH フレームの履歴にする
        #
        # 項の中身・順序・ノイズは継承元のまま。1 フレーム 55 次元が
        # (num_envs, H, 55) になるだけで、切り出し方 (直近 5 + CNN 潜在) は
        # ネットワーク側の仕事。定数の意図は _OBS_HISTORY_LENGTH のコメント参照。
        enable_obs_history(self)

        # -- 0a. 着地 shaping 3 項を無効化する
        #
        # Stage 1-3 でも同じヘルパを呼んでいる。理由は disable_landing_shaping の
        # コメント参照 (蹴るほど損をする圧がかかるため)。
        disable_landing_shaping(self)

        # -- 0a'. 歩行報酬 (feet_phase) を蹴りと釣り合う量まで下げる
        #
        # Stage 2/3 (walk_long_pass_fewa_stages_env_cfg) でも同じヘルパを呼ぶ。
        # 理由は _KICK_STAGE_FEET_PHASE_WEIGHT のコメント参照。
        rebalance_gait_vs_kick(self)

        # -- 0b. IMU / エンコーダ由来の観測にセンサ遅延 DR を掛ける
        #
        # 観測の次元も並びも変わらないので checkpoint はそのまま繋がる。
        # 対象と理由は enable_obs_delay / _OBS_DELAY_MAX_S のコメント参照。
        enable_obs_delay(self, _OBS_DELAY_MAX_S)

        # -- 0c. 地面をランダムな軽い凹凸にする
        #
        # 継承元 (K1FlatEnvCfg) は terrain_type="plane" の完全平面。実機の床は
        # わずかに波打っていて、平面だけで学習したポリシーは足裏が想定どおり接地する
        # 前提に寄りかかる。段差・坂は入れず、±1-4 cm のノイズだけを乗せる
        # (_FEWA_NOISY_FLAT_TERRAIN_CFG: random_rough 0.7 / plane 0.3)。
        #
        # height_scanner は継承元で None のまま = 地形は観測できない (blind)。
        # 凹凸は「観測できない外乱」として効く。
        #
        # 地形カリキュラム (curriculum.terrain_levels) は継承元で None、
        # TerrainGeneratorCfg 側も curriculum=False なので、5x5 の patch は
        # 難易度順ではなく単なるバリエーション。max_init_terrain_level=None で
        # 全 patch に env が散る。
        #
        # NOTE: 高さ系のしきい値は絶対 z のまま (base_height_penalty 0.45 /
        #       base_height 終了 0.35)。地形ノイズは +0.01-0.04 m と上向きだけなので
        #       判定は緩む方向に動き、誤終了は増えない。
        # NOTE: kick_apex_height / sole_height_at_kick も絶対 z なので、地形ぶん
        #       (最大 4 cm) 嵩上げされて見える。報酬には使っていないが、平地の run と
        #       数値を直接比べるときは注意すること。
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _FEWA_NOISY_FLAT_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = None

        # -- 1. 目標ボール速度をロングパスの帯へ（このタスクの本体）
        #
        # cfg の初期値は **カリキュラムの開始点** (= 継承元の帯) にしておく。
        # 終点 _LONG_PASS_SPEED_RANGE へはカリキュラム (下の -- 6) が動かす。
        # ここに終点を直接書くと、カリキュラムが最初に走るまでの数ステップだけ
        # 帯が飛んでしまう上、意図が読めなくなる。
        self.commands.kick_direction.target_speed_range = _LONG_PASS_SPEED_RANGE_START

        # -- 2. 速度シェイピングを帯幅に合わせて緩める
        self.rewards.kick_velocity_scaled.params["sigma_velocity"] = _LONG_PASS_SIGMA_VELOCITY

        # -- 3. 項1 に片側速度ゲートを掛けて、弱い蹴りで満点が出る穴を塞ぐ
        self.rewards.kick_direction.params["v_gate_frac"] = _LONG_PASS_V_GATE_FRAC
        self.rewards.kick_direction.params["sigma_gate"] = _LONG_PASS_SIGMA_GATE

        # -- 4. 項8: 2 回目以降のボール接触を罰する
        #
        # v_thresh を 1.5 に上げた副作用で「まず軽く触って、それから蹴る」が
        # 無コストになる (1.5 m/s 未満の接触では latch もエピソード終了も起きない)。
        # walk_pass と同じ対策。1 回目の接触は無料なので、罰を避ける唯一の道は
        # 最初の接触で蹴り切ること。loop_pass_360 の実測 ball_touch_count ≈ 1.29 が
        # 1.0 に寄るかで効果を見る。
        self.rewards.extra_ball_touch = RewTerm(
            func=mdp.extra_ball_touch,
            weight=0.0,
            params={**_KICK_STATE_PARAMS},
        )
        # ランプは帯カリキュラムの **後**。初版は 0 → 500 で他のキック報酬と同時に
        # 立ち上げていたが、「触ると罰される」圧力と「蹴っても報酬が出ない」状況が
        # 重なると、ポリシーはボールを避ける方向へまとめて逃げる
        # (実測 ball_touch_count 1.30 → 0.91)。帯が目標まで動いてキックが安定してから、
        # 最後に多重接触だけを削るのが正しい順序。
        #
        # 開始時刻は壁時計ではなく **帯が目標に届いた時点** に紐付ける。帯はゲート付き
        # (下の -- 7) で「蹴れている間しか進まない」ので、目標に届く iteration が
        # 事前に決まらない。壁時計の 3000 で立ち上げると、帯がまだ途中のポリシーに
        # 追加の罰を掛けることになる。
        self.curriculum.extra_ball_touch_weight = CurrTerm(
            func=mdp.linear_reward_weight_after_speed_gate,
            params={
                "term_name": "extra_ball_touch",
                "start_weight": 0.0,
                # イベント項なので value = weight * dt。-50.0 で 1 回あたり -1.0
                # (キック 1 回の収益 ≈ +5 の 2 割)。kick_rate が落ちるようなら
                # 弱めること (ボールに触ること自体を避け始めるサイン)。
                "end_weight": -50.0,
                "command_name": "kick_direction",
                "ramp_iterations": 500,
                "steps_per_iteration": _STEPS_PER_ITERATION,
            },
        )

        # -- 5. latch のトリガー速度を帯に合わせて上げる
        # (全項に一括配布。extra_ball_touch を追加した後に呼ぶこと)
        _apply_v_thresh(self, _LONG_PASS_V_THRESH)

        # -- 6. キック報酬のフェードインを潰す (fine-tune では有害)
        #
        # 継承した 0 → 500 iteration のランプは「ゼロから学習する段」用の順序で、
        # 既に蹴れるポリシーから始める Stage 4 では最初の 500 iteration に
        # 「蹴らない方が得」を明示的に作ってしまう。理由は
        # :func:`_freeze_fade_in_curricula` の docstring を参照。
        # 後段でランプする extra_ball_touch (3000 → 3500) は対象外。
        _freeze_fade_in_curricula(self, before_iter=_SPEED_RAMP_START_ITER)

        # -- 7. 目標ボール速度の帯を (2.0,3.0) → (3.2,5.0) へ動かす
        #
        # このタスクの成否を決めるカリキュラム。一段で張り替えると
        # 「蹴らずに time_out まで歩く」に収束する (モジュール docstring の失敗記録)。
        #
        # 壁時計の線形ランプ (linear_command_speed_range) ではなく、キック成立率で
        # 開閉するゲート版を使う。壁時計は「ポリシーがこの速さで付いてくる」という
        # 賭けで、外れたときに自己回復しない (帯が実力を追い越す → キック報酬が消える →
        # 蹴らない方が得 → std も潰れて戻れない)。ゲート版なら崩れた時点で帯が止まり、
        # 崩れ続ければ蹴れていた帯まで戻るので報酬信号が復活する。
        #
        # 進捗は Curriculum/kick_speed_range/{speed_min,speed_max,alpha,kick_rate_ema}。
        # alpha が途中で止まったままなら、その speed_max がこのスイングの実質的な上限。
        self.curriculum.kick_speed_range = CurrTerm(
            func=mdp.kick_rate_gated_speed_range,
            params={
                "command_name": "kick_direction",
                "start_range": _LONG_PASS_SPEED_RANGE_START,
                "end_range": _LONG_PASS_SPEED_RANGE,
                "start_step": _SPEED_RAMP_START_ITER,
                "end_step": _SPEED_RAMP_END_ITER,
                "steps_per_iteration": _STEPS_PER_ITERATION,
                "advance_above": _SPEED_GATE_ADVANCE_ABOVE,
                "retreat_below": _SPEED_GATE_RETREAT_BELOW,
                "retreat_scale": _SPEED_GATE_RETREAT_SCALE,
            },
        )


@configclass
class K1WalkLongPassFewaEnvCfg_PLAY(K1WalkLongPassFewaEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # -- 帯カリキュラムを外し、帯を終点に固定する
        #
        # CurriculumManager は PLAY でも毎リセットで走る。残しておくと
        # common_step_counter が 0 から始まるので alpha=0、つまり帯が開始点
        # (2.0, 3.0) に巻き戻され、**--kick_speed の指定も上書きされる**
        # (play.py の _apply_kick_speed は env 生成前に cfg を書くだけなので、
        #  カリキュラムが後から潰してしまう)。学習済みポリシーを見るときに欲しいのは
        # 終点の帯なので、項ごと外して _LONG_PASS_SPEED_RANGE を直接入れる。
        self.curriculum.kick_speed_range = None
        self.commands.kick_direction.target_speed_range = _LONG_PASS_SPEED_RANGE
