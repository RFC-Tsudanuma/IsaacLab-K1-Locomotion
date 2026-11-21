# ボールトラップ (足元に止める, Isaac-Trap-K1) 新歩行 expert R9 + 摩擦 DR 拡大 + 保持 3 s での再学習 引き継ぎ (2026-11-20)

別 PC でトラップ方策を学習し直すエージェント向け。これ 1 ファイルで開始できる。
タスクの設計と経緯の全記録は元の PC の Claude メモリ (ball-block-task.md) にあり別 PC からは読めないので、
必要なことはこのファイルに書いた。阻止 (パスカット) 側の同じ作業は `BLOCK_HANDOFF.md`、歩行 expert 側は
`WALK_HANDOFF.md` (特に L 節と E 節)。実機側の契約は booster_k1_locomotion の README「ボールトラップ」節
(branch `feat/new_fast_side_walk`)。

## 今回の目的

実機で現行 best_trap (t2 model_25798) を試したところ、**止めた直後に少し転倒**があった (2026-11-19 ユーザー報告)。
対策として環境を 3 点変えた (cfg はコミット済み、**この設定での学習はまだ無い**):

1. 足の摩擦 DR を歩行 (0.3〜1.0) から **U(0.3, 3.0)** に広げる (実機は床の摩擦変動が大きい)。
2. 成功判定を「止めてから **3 s** 立ち続けた」にする (以前は 1 s)。
3. 止めた後の保持中に転倒したら **`fell_after_resolve` −400** を `fell` −200 に上乗せ (合計 −600)。

同時に歩行 expert が **摩擦 DR 0.3〜3.0 (床ごとに 1 つ) で再学習した R9 `model_64440.pt`** に替わった
(`frozen_policies/best_walk/README.md`)。t2 は旧 expert (model_56444) 上の方策なので、R9 の上で t2 を起点に
新 cfg で再学習し、best_trap を置き換えるのが今回の仕事。

## 前提 (別 PC で最初に確認すること)

1. IsaacLab-K1-Locomotion を branch `feat/inoue_walk_double_encoder` の最新で pull。この文書と同じコミットに
   Trap の新 cfg (`block_env_cfg.py: _configure_trap` / `K1TrapRewardsCfg.fell_after_resolve`) と起点
   `frozen_policies/best_trap/model_t2_25798.pt` が入っている。
2. `frozen_policies/best_walk/model_64440.pt` (R9) があること。Trap は Block と同じく `best_walk/model_*.pt` の
   **番号最大**を自動で拾う。学習・評価の起動ログに `[WaypointActorCritic] walk expert <- .../model_64440.pt` が
   出ることを必ず確認する (56444 が出たら R9 が置かれていない)。
3. 位相周波数の式・指令領域は model_56444 と同じ (`1.8 + max(||cmd_xy|| − 1.0, 0) × 0.25 + max(|vy| − 0.5, 0) × 0.75` Hz、
   x ±2.0 / y ±1.9、|vy| > 1.0 で |vx| 上限を 0.3 まで線形に絞る膝)。Trap cfg は FastLateral の継承でこの値 →
   **コード変更なしで載る**。

## タスクの中身 (要点)

- **配置**: 立位スタート。ロボット初期姿勢で固定したライン系 (観測のゴールライン 4 次元は Block と同じ)。
- **ボール** (`BallTrapCommandCfg`): 1.5〜8 m 先から **1〜4 m/s**、到達 1〜3 s、進入角 ±45°、狙いはロボットの左右 ±0.5 m。
  バウンド・反発 0.6・線形/角減衰 0.1 は Block と同じ (転がり抵抗の代わり。実物より小さく、弾いた球が止まりにくい)。
- **成功 (resolved、eval では「阻止」の欄)**: 最寄り足から `trap_dist` 0.4 m 以内でボールが `trap_speed` 0.3 m/s 未満に
  なった時点でラッチし、そこから **`resolve_hold_s` = 3.0 s** 転倒もロストもせず立ち続けたら成立。終了。
  ラッチ後に転倒すると `fell` (−200) に `fell_after_resolve` (−400) が上乗せ。
- **ロスト (conceded、eval では「失点」の欄)**: ボールがライン後方 `lost_x` 2.0 m より奥へ抜けた。終了 (−300)。
- **escaped**: 足で触った (`touch_dist` 0.35 m) 後にボールが胴体より `escape_x` 1.5 m 以上前方で遠ざかっている (vx > 0.1)。
  弾いて逃がした失敗。終了 (−100)。止まった球・逃げた球を追うのは別方策の役目なので追わせない。
- **転倒 (fell)**: 胴体接触 / 胴体高さ。**時間切れ**: 10.0 s (到達 ≤ 3 s + トラップ + 保持 3 s)。
- **評価基準 (ユーザー決定 2026-11-15)**: この方策の目的は「動いている球を止める」こと。足に触れずにロボットの手前で
  止まった球は対象外。`eval_block.py` は `対象外` と `動いている球のみ` の行で分けて出す。

## 構成 (コードはコミット済み)

| 項目 | 内容 |
|---|---|
| タスク | **`Isaac-Trap-K1-v0`** / 評価は `Isaac-Trap-K1-Play-v0`。cfg は `block_env_cfg.py: K1TrapEnvCfg` (← `K1BlockFastRawPosEnvCfg`、差分は `_configure_trap`)。runner は `agents/rsl_rl_ppo_cfg.py: K1TrapPPORunnerCfg` (experiment_name `k1_trap`、`latent_loss_coef` 0) |
| 方策・観測 | Block と完全に同じ (`agents/block_policy.py: BlockActorCritic`、`block_policy` 65 + `ball_noisy` 20×4、critic は真値)。ONNX の入出力も同じなので実機ノードは block と同じ C++ (別インスタンス) で動く |
| ボール観測ノイズ | Block b1 以降と同じ (`K1BallNoisyCfg`、継承で自動適用。t2 の学習時はこのノイズ強化前) |
| コマンド領域 | クランプ (2.0, 1.9, 2.0)、デッドゾーン、膝。ONNX に焼き込む |
| 摩擦 DR | `events.physics_material` = `mdp.events.randomize_floor_friction`、static/dynamic **U(0.3, 3.0)** (Block は 0.15〜3.0)、既定 `per_env` False (shape ごと)。**R9 は `per_env` True で学習した** → 下の「学習を始める」 |
| 判定 (`BallTrapCommandCfg` + `_configure_trap`) | `resolve_hold_s` **3.0** (t2 は 1.0)、`trap_dist` 0.4、`trap_speed` 0.3、`lost_x` 2.0、`touch_dist` 0.35、`escape_x` 1.5。episode **10.0 s** (t2 は 8.0) |
| 主な報酬 (`K1TrapRewardsCfg`、t2 の学習設定 + 新規 1 項) | 成功 +100、転倒 −200、**保持中転倒 −400 (新規)**、ロスト −300、escaped −100、足元ゾーン `trap_zone` 3.0 (σ 0.3 m / 0.5 m/s)、広いゾーン `trap_zone_wide` 1.0 (σ 1.5 m)、接触後のボール速度 −0.5、接触中に足を引く `foot_yield_on_contact` +1.0、足を予測到達点へ 0.5、胴体を予測到達点へ 0.3、残差 L2 −0.05、残差変化率 −0.02、コマンド変化率 −0.05、関節加速度 0 (無効)。RewardManager が dt 0.02 を掛ける |
| 重み 0 の項 (試して不採用) | `touch_velocity_kill` (硬い跳ね返しにも加点される) |
| 滑らかさ罰則 | Block b4 は実機の足首振動対策で 残差変化率 −1.0 / コマンド変化率 −0.5 / 関節加速度 −3e-6 を採用済み。**Trap はまだ弱い既定のまま**なので今回 hydra で同じ値を入れて試す (下) |

## 元の PC での直近の結果 (基準値)

評価は `eval_block.py --task Isaac-Trap-K1-Play-v0` (決定的方策)。「成功 / ロスト / 転倒 / 時間切れ / escaped」は %。
2026-11-16 以前の run は旧 expert (model_52700 系)、t2 は model_56444、最後の行だけ R9。

| | expert | 環境 | 成功 | ロスト | 転倒 | 時間切れ | escaped | 備考 |
|---|---|---|---|---|---|---|---|---|
| v4c@22799 | 旧 | 保持 1 s、摩擦 0.3〜1.0 | 82.5 | 4.1 | 1.3 | 10.5 | 1.5 | 2048 env。実機で最初に試したモデル |
| v4c@22799 | 56444 | 同上 | 67.3 | 11.2 | 3.4 | | | expert 差し替えで載せただけ (劣化の例) |
| **t2@25798 (現行 best_trap)** | 56444 | 同上 | **94.2** | **2.1** | **1.0** | 2.2 | | 2048 env。v4c から reset 0.2 で 3000 iter |
| t2@25798 | **R9 64440** | **新 cfg** (保持 3 s、摩擦 0.3〜3.0 shape ごと) | 72.1 | 6.7 | **16.6** | 2.3 | 2.3 | 1024 env、2026-11-20。再学習の基準値。‖Δa‖² 0.38 / 関節加速度² 20.5 万 |

- **新 cfg + R9 で t2 をそのまま使うと転倒 16.6 %**。`lost_after_touch` (= ラッチ後に成功もロストもせず終わった) が 16.0 % なので、
  ほぼ全部が**止めた後の保持 3 s の間の転倒** (実機の報告と同じ形)。失敗の内訳はロスト 6.7、escaped 2.3 で、止める能力自体は残っている。
  保持中転倒罰 −400 と摩擦 DR、R9 の再学習でここをどこまで戻せるかが見どころ。
- t2 の滑らかさ ‖Δa‖² 0.38 / 関節加速度² 20.5 万は Block b4 (0.14 / 8.1 万) の 2.5 倍。実機の足首振動が出るなら滑らかさ罰則の効果も見る。

- t2 の時間切れ 2.2 % のほとんどは「触って前に残した」(弾いた後に 1.5 m 以内で転がっている) ケース。
- 1024 env の成功率・ロストは ±0.5 pt 程度ぶれる。差を言うなら 2048 env (× seed)。

## 学習の回し方

### 1. 起点を置く

```bash
cd scripts/rsl_rl
mkdir -p logs/rsl_rl/k1_trap/init_t2_25798
cp ../../frozen_policies/best_trap/model_t2_25798.pt logs/rsl_rl/k1_trap/init_t2_25798/
```

### 2. 先に基準を測る (R9 + 新 cfg での t2)

```bash
C=$PWD/logs/rsl_rl/k1_trap/init_t2_25798/model_t2_25798.pt
uv run eval_block.py --task Isaac-Trap-K1-Play-v0 --headless --num_envs 1024 --steps 1200 --device cuda:0 --checkpoint $C
```

起動ログで `walk expert <- .../model_64440.pt` を確認。上の表の最終行と同程度の値が出れば環境が揃っている。
1 本約 3〜5 分 (GPU が空いていれば)、4 GB 程度。

### 3. 学習を始める

```bash
cd scripts/rsl_rl
COMMON="--task Isaac-Trap-K1-v0 --headless --num_envs 2048 --resume --load_run init_t2_25798 \
  --checkpoint model_t2_25798.pt --reset_noise_std 0.2 --max_iterations 3000 agent.algorithm.latent_loss_coef=0.0"
SMOOTH="env.rewards.joint_residual_rate_l2.weight=-1.0 env.rewards.walk_command_rate_l2.weight=-0.5 env.rewards.dof_acc_l2.weight=-3.0e-6"
# 本命: 摩擦を床ごとに 1 つ引く (R9 と同じ引き方) + Block b4 と同じ滑らかさ罰則
nohup uv run train.py $COMMON --device cuda:0 --run_name t3_r9_perenv_smooth \
    env.events.physics_material.params.per_env=true $SMOOTH > trap_t3_nohup.log 2>&1 &
sleep 60   # 2 本目は 1 本目の起動完了後 (同時起動は USD 変換の一時ファイルが衝突して落ちる)
# 対照: 滑らかさ罰則は t2 のまま (弱い) で、新 cfg + R9 の効果だけを見る
nohup uv run train.py $COMMON --device cuda:1 --run_name t4_r9_perenv \
    env.events.physics_material.params.per_env=true > trap_t4_nohup.log 2>&1 &
```

- **`--reset_noise_std 0.2` は必須かつ値が重要**。無いと optimizer state ごと読まれ std が負に落ちて
  `normal expects all elements of std >= 0.0` で落ちる。expert を差し替えた再学習では 0.1 は停滞し (t1: 旧基準並みで頭打ち)、
  0.2 で旧基準を超えた (t2)。Block (w1 / w2) でも同じ結果。0.1 は試さなくてよい。
- `--max_iterations` は追加分。t2 は 3000 iter で収束 (model_25798 = 22799 + 3000 の最終)。
- `agent.algorithm.latent_loss_coef=0.0`: latent 蒸留はスクラッチ学習用。resume では常に 0。
- `per_env`: R9 は床ごとに 1 つで学習した (shape ごとだと両足とも滑る床がほぼ出ない。`mdp/events.py` の docstring)。
  Trap も揃えるのが筋。難度が上がって転倒が下がらないなら対照として `per_env` 既定 (shape ごと) も回す。
  **`physics_material` は startup イベントなので cfg を変えたら学習の再起動が必要**。
- 滑らかさ罰則は Block では強い値でも成功率が落ちず ‖Δa‖² が半減した。Trap は接触時に足を引く shaping があるので
  同じ値で成功率が落ちる可能性があり、t3 / t4 の差で判断する。
- 速度: 元の PC (RTX 4000 Ada 20 GB) で 2048 env ≈ 15 s/iter (235 iter/h、3000 iter ≈ 13 h)。評価 1024 env ≈ 4 GB で、
  学習 1 本と評価 1 本を同じ 20 GB GPU に同居させていた。学習 2 本の同居はしていない。
- **止めるとき**: `pkill -9 -f "run_name t3_r9_perenv_smoot[h]"` のように**単独の Bash 呼び出し**で、パターンに `[ ]` を入れる
  (同じコマンド行に run 名を含む自分の shell が巻き添えになる事故が 4 回あった)。SIGTERM では train.py は死なず `-9` が必要。
- **学習ログで見る指標**: `Metrics/ball_launch/resolved` (成功率) / `conceded` (ロスト) / `escaped`。
  `Episode_Termination/*` は全 env 数で割った値で率ではない。学習中は探索ノイズ込みで決定的評価より数 pt 低い。
- resume 直後の約 400 iter は成功率が 55 % 前後まで落ちてから戻る (ノイズ reset)。400 iter 以降で比較する。

## 評価と採用判断

```bash
# 200 iter ごとの ckpt を 1024 env で (学習と別 GPU で同時に回してよい)
uv run eval_block.py --task Isaac-Trap-K1-Play-v0 --headless --num_envs 1024 --steps 1200 \
    --device cuda:1 --checkpoint $PWD/logs/rsl_rl/k1_trap/<run>/model_N.pt
# 最終候補は 2048 env (× --seed 1 でもう 1 本)
uv run eval_block.py ... --num_envs 2048 --steps 1200 --checkpoint ... [--seed 1]
```

出力の読み方 (Block と同じスクリプト。欄の名前は Block のまま):

- `conceded:` = **ロスト**、`resolved:` = **成功**、`fell:`、`timeout:`、`escaped:` が主結果。
- `対象外 (未接触で前方に残った時間切れ)` と `動いている球のみ (N 件)` の行: ユーザー基準の数値はこちら (通常 1 % 未満しか変わらない)。
- `時間切れ内訳 接触あり・前方` = 触って前に残した (弾き)。これが主な失敗の形。
- `logged Metrics/ ...` の `lost_after_touch` = 触ったのに後ろへ抜かれた。
- `滑らかさ (1 step 平均)` の行: ‖Δa‖² と関節加速度² (全体 / ボール 0.6 m 以内)。実機の足首振動の指標。Block b4 で 0.14 / 8.1 万。
- 発射速度帯ごとの表 (1〜2 / 2〜3 / 3〜4 m/s)。速い帯ほど弾きが残る。
- **resume 直後に保存される model_N (開始 iter と同番号) はノイズ reset 直後で低く出るので比較から除外。**

**採用基準** (ユーザーの優先順: ロスト → 転倒 → 成功率 → 滑らかさ): 新 cfg で ロスト ≤ 3 %、転倒 ≤ 1.5 %
(保持 3 s の間も含む)、成功 ≥ 90 % を目標。t2 を新 cfg でそのまま評価した値 (上の表) より転倒が明確に下がっていること。

## 書き出しと配備

```bash
cd scripts/rsl_rl
CUDA_VISIBLE_DEVICES="" uv run python export_block_policy.py \
    --checkpoint $PWD/logs/rsl_rl/k1_trap/<run>/model_N.pt \
    --out ../../frozen_policies/best_trap/trap_t3_N.onnx
cp $PWD/logs/rsl_rl/k1_trap/<run>/model_N.pt ../../frozen_policies/best_trap/model_t3_N.pt
```

- **合格条件**: torch と onnxruntime の差が 1e-4 未満 (最近は 2e-6)。ログに `knee on`、limits (2.0, 1.9, 2.0)、residual_scale 0.5。
- ONNX は Block と同じ名前・形 (`block_obs (1,65)` + `ball_history (1,20,4)` → `command (1,3)` + `residual (1,12)`)。
  **観測を変えたら C++ も変える。今回は変えない。**
- 採用したら `K1TrapRewardsCfg` の docstring の「既定値は best_trap (...)」を更新。hydra で変えた重み (滑らかさなど) を採用した場合は既定値も書き写す。
- **実機リポジトリ** booster_k1_locomotion (branch `feat/new_fast_side_walk`):
  - ONNX を `assets/k1_trap_<日付>_<run>_<iter>.onnx` として追加。
  - `launch/k1_isaaclab_deploy.launch.py` と `launch/k1_isaaclab_sim.launch.py` の `default_trap_model`、README「ボールトラップ」節の
    現行 entry を更新 (dcdcb9a の書き方に倣う)。
  - トラップは C++ の `LinePolicy` / `line_policy_step` (block と共用) で動く。有効化は `/trap_cmd` (std_msgs/Bool)、自動判定なし、
    優先順は奪取 > 阻止 > トラップ。**C++ の変更は不要**。ただし**実機の歩行 ONNX とゲート (gating) も R9 版に差し替わっていること**が前提
    (`WALK_HANDOFF.md` E / F / G 節)。
  - 2026-11-21 時点で booster 側の gating 既定は R9 ベースの v8 (`k1_gating_walk_turn_2026-11-21_v8_2999_walk64440_turn10000.onnx`、b1c5cea)。
    **現行 t2 を v8 の上で動かすと上の表どおり転倒 16.6 %** なので、再学習版が出るまで trap を使う運用は v7 一式
    (`model_path:=<pkg>/assets/k1_gating_walk_turn_2026-11-16_v7_2999_walk56444_turn9000.onnx`) で動かす。再学習版を配備するときは
    gating が v8 であることを確認してから `default_trap_model` を差し替える。
- 旧版 (t2 / v4c) は `best_trap/` に併置したままでよい。

## 既知の結論 (やり直さなくて良いこと)

- **expert 差し替え時の再学習**: reset 0.2 で 600 iter 以内に旧基準を超える。0.1 は停滞。
- **escaped 終了 (−100)**: 弾いた球を追わせないだけで成功 +4〜5 pt。
- **吸収 shaping** (`foot_yield_on_contact` +1.0、接触後ボール速度 −0.5): 成功 +2〜4 pt、弾き出し減。
- **ロスト罰**: −100 → −200 でロスト −1.5〜2.5 pt (成功同等)、→ −300 でさらに −1 pt (成功 −1.7 pt)。ユーザー優先順に従い −300 を採用。
- **`touch_velocity_kill`** (接触時の減速に加点): 硬い跳ね返しにも加点されるので不採用 (0)。
- **球速カリキュラム** (`K1BlockCurriculumEnvCfg` の `BallSpeedCurriculum`): 用意はあるが Trap では未使用。1〜4 m/s なら不要だった。
- **Block 側の結論で Trap にも当てはまるもの**: 観測ノイズ強化は成績を悪化させない、滑らかさ罰則 (b4 の値) は成功率を落とさない、
  摩擦 DR 拡大だけなら方策側で転倒は戻る (0.15〜3.0 で 10.1 → 2.9 %)、保持中転倒罰は −400 と −200 で差なし。
- **学習の型**: スクラッチは遅いので常に resume。1 GPU 1 本、評価は別 GPU。

## 残課題 (今回の対象外、判断はユーザーと)

- 射出は 1〜4 m/s のみ。速いパス (4 m/s 超) は未学習。
- 実機の床の摩擦は未計測。0.3〜3.0 はユーザー指定。
- ボールの転がり抵抗 (線形減衰 0.1) が実物より小さい。弾いた後に止まらない原因の一つ。
- 学習にカメラ視野の制限 (真横・足元のボールが見えない) は入っていない。足元のボールは実機では見失いやすい。
- 止まった球の処理 (寄る・蹴る) は別方策。実機ではボールが止まったら `/trap_cmd` を false にして次へ移る (戦略側)。

## 結果 (2026-11-21 追記): t3 model_26900 を best_trap に採用 (t4 model_28500 も併置)

vast.ai (RTX 5090 × 1) で t2@25798 から reset 0.2 で 3000 iter ずつ、t3 → t4 の順に逐次実行 (約 4.9 s/iter、1 本約 4 時間)。
歩行 expert は R9 `best_walk/model_64440.pt` (起動ログで確認)。摩擦は床 (env) ごとに 1 つ (`per_env` は Block cfg の既定で既に True)。

- t3 = run `2026-11-20_13-56-40_t3_r9_perenv_smooth` (滑らかさ罰則 −1.0 / −0.5 / −3e-6 を hydra で上書き)
- t4 = run `2026-11-20_18-06-37_t4_r9_perenv` (報酬は cfg 既定のまま)

評価 (2048 env × seed 0/1 の平均、動いている球のみ。基準の t2 だけ 1024 env):

| | 成功 | ロスト | 転倒 | 時間切れ | escaped | ‖Δa‖² | 関節加速度² |
|---|---|---|---|---|---|---|---|
| t2@25798 (載せ替えのみ) | 72.2 | 6.6 | 16.7 | 2.1 | 2.4 | 0.375 | 20.5万 |
| **t4@28500** (滑らかさ罰則 弱) | 96.1 | 1.1 | 1.45 | 0.7 | 0.6 | 0.229 | 13.6万 |
| t4@28400 | 96.1 | 1.05 | 1.65 | 0.7 | 0.55 | 0.236 | 14.0万 |
| t4@28700 | 96.1 | 1.25 | 1.65 | 0.7 | 0.4 | 0.226 | 13.2万 |
| **t3@26900** (滑らかさ罰則 強) | 93.3 | 2.5 | 1.55 | 1.6 | 0.95 | 0.138 | 8.3万 |
| t3@27000 | 93.2 | 2.35 | 1.85 | 1.75 | 0.9 | 0.141 | 8.4万 |
| t3@28797 | 92.8 | 3.0 | 1.8 | 1.45 | 1.0 | 0.141 | 8.3万 |

- どちらも転倒 16.7 → 1.5 % 前後で、保持 3 s 中の転倒はほぼ解消。
- 優先順 (ロスト → 転倒 → 成功 → 滑らかさ) では t4@28500。滑らかさ罰則はロスト +1.4 pt / 成功 −2.8 pt の代わりに
  ‖Δa‖² −40 %・関節加速度² −39 % (Block b4 並み)。実機で足首振動が出るなら t3@26900。
- t3 は 27500〜28200 でロストが 3〜5 % に上がる区間がある (1024 env)。t4 は後半ほど良い。
- 書き出し済み: `frozen_policies/best_trap/{model,trap}_t4_28500` (parity 2.1e-6)、`{model,trap}_t3_26900` (2.9e-6)。
  `export_block_policy.py` は checkpoint の隣の `params/agent.yaml` を読むので run ディレクトリ内の ckpt を指定すること。
- **採用は t3@26900** (ユーザー決定 2026-11-21: 滑らかさを優先)。`K1TrapRewardsCfg` の既定値を t3 の滑らかさ罰則に書き換えたので、
  上の「学習を始める」の `$SMOOTH` 上書きは今は不要 (t4 を再現するなら逆に −0.02 / −0.05 / 0 を hydra で渡す)。
- 実機リポジトリには `assets/k1_trap_2026-11-21_t3_26900.onnx` を追加し、両 launch の `default_trap_model` と README を更新済み。
  **実機リポジトリの歩行ゲートの既定はまだ model_56444 版 (gating v7)** なので、R9 版ゲートに差し替えるまでは学習時と expert が食い違う。
- 未実施: R9 版ゲートの配備、実機検証。
