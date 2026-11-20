# 飛来ボール阻止 (パスカット, Isaac-Block-K1) 新歩行 expert (摩擦 DR 版 R9) での再学習 引き継ぎ (2026-11-20)

別 PC で阻止方策を学習し直すエージェント向け。これ 1 ファイルで開始できる。
タスクの設計と経緯の全記録は元の PC の Claude メモリ (ball-block-task.md) にあり別 PC からは読めないので、
必要なことはこのファイルに書いた。実機側の契約は booster_k1_locomotion の README「ボール阻止 / キーパー」節
(branch `feat/new_fast_side_walk`)。歩行 expert 側の経緯は `WALK_HANDOFF.md` (特に L 節と E 節)。

## 今回の目的

歩行 expert を **摩擦 DR 0.3〜3.0 (床ごとに 1 つ) + 足上げ罰で再学習した R9 `model_64440.pt`**
(`WALK_HANDOFF.md` L 節。学習した PC と vast.ai にしか checkpoint が無く、`frozen_policies/best_walk` は未更新) に
差し替え、その上で阻止方策を再学習して現行 best_block (b4 model_19399、実機採用済み) を置き換える。

背景: 実機で b4 がシュートを止めた直後に転倒 (2026-11-19)。元の PC では対策として環境を変えて b5 を学習した
(摩擦 DR 0.15〜3.0、保持 3 s、保持中転倒罰 −400。下の「b5 の結果」)。**歩行 expert 自体の低摩擦耐性が足りない**のが
根本なので、R9 の上で b5 を起点にやり直すのが今回の仕事。

## 前提 (別 PC で最初に確認すること)

1. IsaacLab-K1-Locomotion を branch `feat/inoue_walk_double_encoder` の最新で pull。この文書と同じコミットに
   b5 の環境変更 (`block_env_cfg.py` / `mdp/block.py`)、`eval_block.py` の滑らかさ指標、起点 checkpoint
   `frozen_policies/best_block/model_b5_20800.pt` が入っている。
2. **R9 を `frozen_policies/best_walk/` に置く** (`WALK_HANDOFF.md` E 節): `model_64440.pt` と、できれば
   `policy.onnx` / `policy.pt` / `env.yaml` / README 追記。Block は `best_walk/model_*.pt` の**番号最大**を自動で拾う
   (64440 > 56444 なので旧モデルは残置でよい)。学習・評価の起動ログに
   `[WaypointActorCritic] walk expert <- .../model_64440.pt` が出ることを必ず確認する。
   アーキテクチャ (HistoryActorCritic 512-256-128、入力 command 3 + obs_history 100×49) は model_56444 と同じ。
3. 位相周波数の式・指令領域は model_56444 と同じ (`1.8 + max(||cmd_xy|| − 1.0, 0) × 0.25 + max(|vy| − 0.5, 0) × 0.75` Hz、
   x ±2.0 / y ±1.9、|vy| > 1.0 で |vx| 上限を 0.3 まで線形に絞る膝)。Block cfg 側は既にこの値 → **コード変更なしで載る**。

## タスクの中身 (要点)

- **配置**: 立位スタート。ロボット初期姿勢で固定したゴールライン (後方 `behind_x` 0.3 m、半幅 `goal_half_width` 2.6 m)。
- **ボール**: 最長 12 m 先から 0.4〜8.0 m/s で自分の左右を狙って来る (ライン上の到達 y が分布の中心)。30 % はバウンド (vz ≤ 2.0)。
  反発 0.6、線形/角減衰 0.1 (転がり抵抗の代わり。実物より小さく、弾いた球が止まりにくい)。
- **失点 (conceded)**: ボールがライン後方かつゴール幅内に入った。終了。
- **阻止 (resolved)**: 失点せずに (a) 幅外へ逸れた (out_wide) / (b) 前へ離れた (away、vx > 0.3) / (c) 止まった (stopped、0.1 m/s 未満)
  のいずれかになった時点でラッチし、そこから **`resolve_hold_s` = 3.0 s** 転倒も失点もせず立ち続けたら成立。終了。
  ラッチ後に転倒すると `fell` (−200) に `fell_after_resolve` (−400) が上乗せされる。
- **転倒 (fell)**: 胴体接触 / 胴体高さ。**時間切れ**: 10.5 s。
- **足で触った後に失点** は `lost_after_touch` としてメトリクスに出る (ボールへの対応はしたが抜かれた)。

## 構成 (コードはコミット済み、報酬などの既定値 = b5 の学習設定)

| 項目 | 内容 |
|---|---|
| タスク | **`Isaac-Block-K1-FastRawPos-v0`** / 評価は `Isaac-Block-K1-FastRawPos-Play-v0`。cfg は `block_env_cfg.py: K1BlockFastRawPosEnvCfg` (← FastLateral ← `K1BlockEnvCfg` ← 歩行 `K1FlatStancePlanePolishCfg`)。runner は `agents/rsl_rl_ppo_cfg.py: K1BlockFastLateralPPORunnerCfg` (experiment_name `k1_block`) |
| 方策 | `agents/block_policy.py: BlockActorCritic`。出力 15 = 歩行コマンド (vx, vy, ωz) + 脚 12 関節の残差 (×0.5)。凍結歩行 expert を内包し、env には [expert + 残差, コマンド, 残差] の 27 次元を渡す。ボール履歴は causal transformer encoder (actor はノイズ付き観測、critic は真値) |
| 観測 | `block_policy` 65 次元 (歩行 49 フレーム + 前 tick 残差 12 + ゴールライン 4) + `ball_noisy` 20 step × 4 (相対位置 xy、速度 xy)。critic は `block_critic` (線速度・ZMP 込み) + `ball_true` |
| ボール観測ノイズ (b1 以降) | 速度の遅延 0.16〜0.30 s、位置の遅延 0.03〜0.10 s (env ごと)、30 fps、検出欠け 10 %、σ は距離の 2 乗で (近, 遠) を補間し 10 m で飽和: 位置 0.05 → 0.25 m、速度 0.25 → 1.5 m/s (`K1BallNoisyCfg`、`DelayedBallObs`) |
| コマンド領域 | クランプ (2.0, 1.9, 2.0)、デッドゾーン ‖cmd‖ < 0.1、膝 (`lateral_x_knee` 1.0 / `lateral_x_min` 0.3)。ONNX に焼き込む |
| 歩容位相 | expert と同じ写像 (上記)。policy / critic / block_policy / block_critic の 4 グループで揃えてある (`K1BlockFastLateralEnvCfg`) |
| 摩擦 DR | `events.physics_material` = `mdp.events.randomize_floor_friction`、static/dynamic **U(0.15, 3.0)**、`make_consistent` True、既定 `per_env` False (shape ごと = b5 と同じ)。地形は 1.0 × multiply なので値がそのまま実効摩擦。**R9 は `per_env` True で学習した**ので下の「学習を始める」参照 |
| 主な報酬 (`K1BlockRewardsCfg`) | 阻止 +100、失点 −200、転倒 −200、保持中転倒 −400 (上乗せ)、足を予測到達点へ 0.5 (σ 0.15)、胴体を予測到達点へ 0.5 (σ 0.6)、ライン前方 1.0 m 超 −2.0、阻止後に足元で遅く保持 0.4、接触後のボール速度 −0.2、残差 L2 −0.05、**残差変化率 −1.0、コマンド変化率 −0.5、関節加速度 −3e-6** (実機の足首振動対策、b4)。RewardManager が dt 0.02 を掛ける |
| 重み 0 の項 (試して不採用) | `stopped_bonus`、`touch_velocity_kill`、`foot_yield_on_contact` |
| その他 | episode 10.5 s、押し外乱 4〜8 s ごと ±0.5 m/s、歩行系カリキュラムは全て無効、entropy 0、`latent_loss_coef` は resume では 0 |

## 元の PC での直近の結果 (基準値)

評価は `eval_block.py` (決定的方策)。「失点 / 阻止 / 転倒 / 時間切れ」は %、滑らかさは関節指令の 1 step 変化 ‖Δa‖² と関節加速度²。

| | 環境 | 失点 | 阻止 | 転倒 | ‖Δa‖² | 関節加速度² | 備考 |
|---|---|---|---|---|---|---|---|
| b4@19399 (実機採用中) | 旧 cfg: 摩擦 0.3〜1.0、保持 2.5 s | 10.3 | 83.3 | 2.0 | 0.139 | 8.1万 | 2048 env × seed 0/1 |
| b4@19399 | 新 cfg: 摩擦 0.15〜3.0、保持 3.0 s | 15.4 | 70.9 | **10.1** | 0.147 | 9.1万 | 2048 env。低摩擦帯で現行方策はほぼ転ぶ |
| **b5@20800 (起点)** | 新 cfg | 14.1 | 79.4 | 2.9 | 0.105 | 6.3万 | 1024 env。lost_after_touch 2.0 % |
| b5 終盤 21000〜22200 平均 | 新 cfg | 15.2 | 77.6 | 2.9 | 0.104 | 6.2万 | 1024 env × 7 点 |
| b5b (保持中転倒 −200) 終盤平均 | 新 cfg | 15.1 | 77.6 | 2.7 | 0.103 | 6.2万 | −400 と差なし |

- b5 / b5b は b4@19399 から 3000 iter (19400 → 22399)。400 iter で転倒 10.1 → 3.7、以後 3 前後で安定。失点は基準並み (15 前後)、阻止 +7 pt。
- **2048 env の確認評価は未実施**。b5@20800 は 1024 env の単点で最良だっただけなので、起点としては b5 終盤のどれでも大差ない。
- 歩行 expert 自体は μ0.3 で前進 2.0 指令 → 実速度 1.2〜1.4 m/s、立ち上がり 2.4 s (μ0.8 の 2 倍遅い)。
  低摩擦帯の失点は expert の限界も含むので、R9 でどこまで戻るかが見どころ。

## 学習の回し方

### 1. 起点を置く

```bash
cd scripts/rsl_rl
mkdir -p logs/rsl_rl/k1_block/init_b5_20800
cp ../../frozen_policies/best_block/model_b5_20800.pt logs/rsl_rl/k1_block/init_b5_20800/
```

### 2. 先に基準を測る (新 expert の上での b5@20800)

```bash
C=$PWD/logs/rsl_rl/k1_block/init_b5_20800/model_b5_20800.pt
uv run eval_block.py --task Isaac-Block-K1-FastRawPos-Play-v0 --headless --num_envs 1024 --steps 1200 \
    --device cuda:0 --checkpoint $C
```

- 起動ログで `walk expert <- .../model_64440.pt` を確認。
- 前回 expert を差し替えたとき (model_52700 → 56444) は、載せただけで失点 21.6 → 27.3 に悪化し、
  reset 0.2 の再学習 600 iter で旧基準を超えた。今回も同程度の劣化を想定。
- 1 本約 3 分 (GPU が空いていれば)。4 GB 程度。

### 3. 学習を始める

```bash
cd scripts/rsl_rl
COMMON="--task Isaac-Block-K1-FastRawPos-v0 --headless --num_envs 2048 --resume --load_run init_b5_20800 \
  --checkpoint model_b5_20800.pt --reset_noise_std 0.2 --max_iterations 3000 agent.algorithm.latent_loss_coef=0.0"
# 本命: 摩擦を床ごとに 1 つ引く (R9 と同じ引き方)
nohup uv run train.py $COMMON --device cuda:0 --run_name f1_w64440_perenv \
    env.events.physics_material.params.per_env=true > block_f1_nohup.log 2>&1 &
sleep 60   # 2 本目は 1 本目の起動完了後 (同時起動は USD 変換の一時ファイルが衝突して落ちる)
# 対照: b5 と同じ引き方 (shape ごと)
nohup uv run train.py $COMMON --device cuda:1 --run_name f2_w64440_pershape > block_f2_nohup.log 2>&1 &
```

- **`--reset_noise_std 0.2` は必須かつ値が重要**。無いと optimizer state ごと読まれ std が負に落ちて
  `normal expects all elements of std >= 0.0` で落ちる。expert を差し替えた再学習では 0.1 は停滞し (w1、t1)、
  0.2 で旧基準を超えた (w2、t2)。2 回とも同じ結果なので 0.1 は試さなくてよい。
- `--max_iterations` は追加分。3000 iter で収束 (b5 は 1600 iter 以降横ばい)。
- `agent.algorithm.latent_loss_coef=0.0`: latent 蒸留はスクラッチ学習用。v8 以降の resume は全て 0。
- `per_env`: R9 は「床ごとに 1 つ」で学習した (shape ごとだと両足とも滑る床がほぼ出ない。`mdp/events.py` の docstring)。
  Block も揃えるのが筋だが低摩擦の難度は上がるので、対照と並走して決める。下限 0.15 は R9 の学習範囲 (0.3) の外。
  f1 の転倒が 5 % を超えて下がらないなら `env.events.physics_material.params.static_friction_range=[0.3,3.0]`
  `...dynamic_friction_range=[0.3,3.0]` に戻すのが次の手。**`physics_material` は startup イベントなので cfg を変えたら学習の再起動が必要**。
- 保持中転倒罰 `fell_after_resolve` は −400 と −200 で差が出なかった。既定 −400 のままでよい。
- 速度: 元の PC (RTX 4000 Ada 20 GB) で 2048 env ≈ 16 s/iter (220 iter/h、3000 iter ≈ 14 h)。評価 1024 env ≈ 4 GB で、
  元の PC では学習 1 本と評価 1 本を同じ 20 GB GPU に同居させていた。学習 2 本の同居はしていない。
- **止めるとき**: `pkill -9 -f "run_name f1_w6444[0]"` のように**単独の Bash 呼び出し**で、パターンに `[ ]` を入れる
  (同じコマンド行に run 名を含む自分の shell が巻き添えになる事故が 4 回あった)。SIGTERM では train.py は死なず `-9` が必要。
- **学習ログで見る指標**: `Metrics/ball_launch/conceded` / `resolved` (率)。`Episode_Termination/*` は全 env 数で割った値で率ではない。
  学習中は探索ノイズ込みで決定的評価より失点が数 pt 高い。採用判断は下の評価で行う。

## 評価と採用判断

```bash
# 200 iter ごとの ckpt を 1024 env で (学習と別 GPU で同時に回してよい)
uv run eval_block.py --task Isaac-Block-K1-FastRawPos-Play-v0 --headless --num_envs 1024 --steps 1200 \
    --device cuda:1 --checkpoint $PWD/logs/rsl_rl/k1_block/<run>/model_N.pt
# 最終候補は 2048 env × seed 0 / 1 (失点の標準誤差 ≈ 0.3 pt)
uv run eval_block.py ... --num_envs 2048 --steps 1200 --checkpoint ... [--seed 1]
```

出力の読み方:

- `conceded: / resolved: / fell: / timeout:` の行が主結果。
- `logged Metrics/ ...` の行に阻止の内訳 (out_wide / away / stopped) と **lost_after_touch** (触ったのに抜かれた)。
- `滑らかさ (1 step 平均)` の行: ‖Δa‖² と関節加速度² (全体 / ボール 0.6 m 以内)。実機の足首振動の指標。b5 で 0.10 / 6.2 万。
- 発射速度帯・狙い横位置・距離帯ごとの表も出る (遠距離 > 11.9 m、横 > 1.5 m、速度 > 6 m/s が難しい帯)。
- resume 直後に保存される model_N (開始 iter と同番号) はノイズ reset 直後で低く出るので比較から除外。
- 1024 env の失点は ±0.5〜0.6 pt ぶれる。差を言うなら 2048 env × 2 seed。

**採用基準** (優先順: 失点 → 転倒 → 阻止率 → 滑らかさ): 新 cfg で b5 相当 (失点 ≤ 15、転倒 ≤ 3、‖Δa‖² ≤ 0.11) を保ちつつ
低摩擦帯の転倒・失点が下がっていれば採用。R9 で低摩擦の歩行が良くなった分、失点 12〜13 まで戻る余地があると見ている。

## 書き出しと配備

```bash
cd scripts/rsl_rl
CUDA_VISIBLE_DEVICES="" uv run python export_block_policy.py \
    --checkpoint $PWD/logs/rsl_rl/k1_block/<run>/model_N.pt \
    --out ../../frozen_policies/best_block/block_f1_N.onnx
cp $PWD/logs/rsl_rl/k1_block/<run>/model_N.pt ../../frozen_policies/best_block/model_f1_N.pt
```

- **合格条件**: torch と onnxruntime の差が 1e-4 未満 (最近は 3e-6)。ログに `knee on` と limits (2.0, 1.9, 2.0)、residual_scale 0.5。
- ONNX は上位 actor のみ (観測正規化 + encoder + MLP + クランプ / デッドゾーン / 膝 + 残差スケール)。歩行 expert は含まない。
  入力 `block_obs (1,65)` + `ball_history (1,20,4)`、出力 `command (1,3)` + `residual (1,12)`。観測の並びは `export_block_policy.py` の docstring。
  **観測を変えたら C++ も変える。今回は変えない。**
- 採用したら `K1BlockRewardsCfg` の docstring の「既定値は best_block (...)」を更新。報酬の重みを変えて採用した場合は既定値も書き写す。
- **実機リポジトリ** booster_k1_locomotion (branch `feat/new_fast_side_walk`):
  - ONNX を `assets/k1_block_<日付>_<run>_<iter>.onnx` として追加。
  - `launch/k1_isaaclab_sim.launch.py` の `default_block_model` と README「ボール阻止 / キーパー」節の現行 entry を更新
    (前回 f45d5be の書き方に倣う: 学習設定の差分、評価値、旧版を「旧」に)。`k1_isaaclab_deploy.launch.py` の既定は
    w2 のままになっている (実機では launch 引数で渡している)。
  - ノードの入力 65 次元・位相の式は不変なので **C++ の変更は不要**。ただし**実機の歩行 ONNX とゲート (gating) も R9 版に
    差し替わっていること**が前提 (`WALK_HANDOFF.md` E / F / G 節)。旧歩行 ONNX の上に新 Block ONNX を載せると学習時と expert が食い違う。
- 旧版 (b4 / w2 / v22d ...) は `best_block/` に併置したままでよい。

## 既知の結論 (やり直さなくて良いこと)

- **expert 差し替え時の再学習**: reset 0.2 で 600 iter 以内に旧基準を超える (w2、t2)。0.1 は停滞。
- **観測ノイズ強化** (b1): 失点を悪化させない (w2 12.0 → b1 10.1)。
- **滑らかさ罰則**: 強い重み (b4: −1.0 / −0.5 / −3e-6) でも失点は落ちず、‖Δa‖² −49 %・関節加速度² −47 %。
  弱い b3 (−0.3 / −0.2 / −1e-6) は滑らかさが 16 % 劣り失点は 0.6 pt 良い程度。さらに強くする実験はしていない。
- **摩擦 DR 拡大** (b5): 方策側だけで転倒 10.1 → 2.9 まで戻る。失点は改善しない (expert の低摩擦性能が壁)。
- **保持 2.5 → 3.0 s、保持中転倒罰 −400**: 成立。−200 と差なし。
- **報酬で効かなかったもの** (重み 0 で残置): 足元停止ボーナス (失点 +2〜3 pt)、接触時の減速加点、接触時に足を引く加点 (5〜10 で「よける」になり失点 30〜60 %)。
  接触後のボール速度罰は −0.2 まで (−0.4 は失点率が戻らない)。
- **上体の揺れ罰** (v23、`block_upper_body_regularization.md`): 失点悪化で不採用。
- **ライン前方への罰の margin** 0.2 → 1.0: 12 m・±2 m の分布では斜め前方の迎撃を許す方が失点が減る。
- **学習の型**: スクラッチは遅いので常に resume。1 GPU 1 本、評価は別 GPU。

## 残課題 (今回の対象外、判断はユーザーと)

- **Trap (足元に止める) の再学習**: `Isaac-Trap-K1-v0` は `K1BlockFastRawPosEnvCfg` を継承するので新ノイズ・新摩擦 DR・R9 は自動で載る。
  起点 `frozen_policies/best_trap/model_t2_25798.pt` (旧 expert 56444 上、2048 env: 成功 94.2 / ロスト 2.1 / 転倒 1.0)。
  `K1TrapRewardsCfg` の滑らかさ 3 項はまだ弱い (−0.02 / −0.05 / 0) ので、Block と同じ値を hydra で
  `env.rewards.joint_residual_rate_l2.weight=-1.0 env.rewards.walk_command_rate_l2.weight=-0.5 env.rewards.dof_acc_l2.weight=-3.0e-6`。
  評価は `eval_block.py --task Isaac-Trap-K1-Play-v0` (成功 = resolved、ロスト = conceded)。保持は `BallTrapCommandCfg` の 1.0 s のまま。
- 実機の床の摩擦は未計測。b5 の下限 0.15 はユーザー指定で、実測で見直す余地あり。
- ボールの転がり抵抗 (線形減衰 0.1) が実物より小さい。弾いた後に止まらない原因の一つ。
- 学習にカメラ視野の制限 (真横・足元のボールが見えない) は入っていない。
