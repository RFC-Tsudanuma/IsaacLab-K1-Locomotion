# ボール奪取 (方向指定版, Isaac-StealDir-K1) 新歩行 expert での再学習 引き継ぎ (2026-11-18)

別 PC で奪取方策を学習し直すエージェント向け。これ 1 ファイルで開始できる。
タスクの設計と経緯の全記録は、元の PC の Claude メモリ (ball-steal-task.md) にあり別 PC からは読めないので、
必要なことはこのファイルに書いた。実機側の契約は booster_k1_locomotion の README「ボール奪取」節
(branch `feat/ball_steal`)。

## 今回の目的

歩行 expert が `frozen_policies/best_walk/model_56444.pt` (横 ±1.9 m/s、2026-11-16) に更新された。この上で奪取方策を
再学習し、現行 `frozen_policies/best_steal_dir/model_38200.pt` (旧歩行 model_52700 上で学習) を置き換える。
同じ更新で、パスカット (best_block) とトラップ (best_trap) は再学習済み。

## タスクの中身 (要点)

- **配置**: ロボット (立位) の向きから ±180° の方位・1〜3 m に敵 (半径 0.15 m の kinematic 円柱)。ボールは敵の中心から
  0.31〜0.5 m、全周のどこか。
- **目標方向**: 開始時のロボットの向きを基準に、前・左・後・右のどれか。ボール→敵の向きから 60° 未満の方向は抽選しない。
  方策は base yaw 系の単位ベクトル (2 次元) として観測する。
- **成功 (stolen)**: 足で触れた後、ボールが敵の中心から 0.8 m の外へ出て、その時点で目標方向 ±45° 以内なら成立 (ラッチ)。
  ラッチから 1.5 s、転倒も失敗もしなければ成功。
- **失敗**:
  - lost: ボールが敵に触れた。
  - no_foot: 足以外で 0.8 m の外へ出した。
  - wrong_dir: 方向違いで 0.8 m の外へ出した。
  - fell: 胴体の高さ 0.35 m 未満。胴体と敵の接触 (体当たり) では終了しない。
- **動く敵とボール (カリキュラム)**:
  - 敵は速さ U(0, 上限)・向き一様で直進する。ボールは敵の速度 + 最大 0.2 m/s の差で動く。
  - ロボットが近づく (胴体 0.45 m / 足 0.3 m) とボールの駆動をやめる。足で触れるまでは、敵がボールに一定の offset で付いていく。
  - 30% のエピソードは静止。
  - 上限は 20000 エピソードごとに、学習中の奪取率が 75% 以上なら +0.1 m/s (最大 1.0)。

## 構成 (コードはコミット済み、報酬などの既定値 = model_38200 の採用設定)

| 項目 | 内容 |
|---|---|
| タスク | `Isaac-StealDir-K1-v0` / 評価は `Isaac-StealDir-K1-Play-v0` (`steal_env_cfg.py: K1StealDirEnvCfg`) |
| 方策 | `agents/steal_policy.py: StealActorCritic`。MLP 256-128、出力 15 = 歩行コマンド (vx, vy, ωz) + 脚 12 関節の残差 (×0.5)。凍結歩行 expert を内包する |
| 凍結 expert | `frozen_policies/best_walk/model_*.pt` の**番号最大**を自動で拾う。起動ログの `walk expert <- .../model_56444.pt` で確認する |
| 観測 | policy 69 次元: 歩行フレーム 49 + 前 tick の残差 12 + ボール位置・速度 4 + 敵の位置 2 + 目標方向 2。critic 76 次元: 特権観測込み + 敵の速度 2 |
| コマンド領域 (今回変更) | クランプ (2.0, **1.9**, 2.0)。`|vy| > 1.0` では `|vx|` 上限を 2.0 → 0.3 (|vy| = 1.9) に線形に絞る (歩行 expert の学習分布、Block と同じ)。`RslRlStealActorCriticCfg.command_limits / lateral_x_knee / lateral_x_min` |
| 歩容位相 (今回変更) | expert と同じ写像: 1.8 → 2.0 Hz (‖cmd‖ 1.0 → 1.8 m/s)、横移動時は (|vy| − 0.5) × 0.75 Hz を加算。policy / critic / steal_policy / steal_critic の 4 グループで揃える |
| 主な報酬 | 成功 +100、敵に当てた −200、転倒 −1600、足以外 −60、方向違い −60、ボールへの接近 50、足をボールへ 3、目標方向のボール速度 5、時間 −0.5、残差 L2 −0.2、残差変化 −0.02、コマンド変化 −0.2、胴体と敵の接触 −0.5 |
| 上体の揺れの罰 | 傾き −40、ロール/ピッチ角速度 −0.8、角加速度 −4e-4、並進加速度 −8e-3。実機の頭部カメラでボールを見失わないため |

## 学習の回し方

### 1. 開始点を置く

学習の起点は model_38200。元の run が別 PC に無いので、logs に置く。

```bash
cd scripts/rsl_rl
mkdir -p logs/rsl_rl/k1_steal_dir/init_38200
cp ../../frozen_policies/best_steal_dir/model_38200.pt logs/rsl_rl/k1_steal_dir/init_38200/
```

### 2. 最初にベースラインを測る (未実施)

元の PC では GPU が空かず、未実施。まず、新 expert の上での model_38200 の成績を測る。

```bash
C=$PWD/logs/rsl_rl/k1_steal_dir/init_38200/model_38200.pt
CUDA_VISIBLE_DEVICES=0 uv run eval_steal.py --headless --task Isaac-StealDir-K1-Play-v0 --num_envs 1024 --steps 1500 --checkpoint $C --move_speed 0.0
CUDA_VISIBLE_DEVICES=0 uv run eval_steal.py --headless --task Isaac-StealDir-K1-Play-v0 --num_envs 1024 --steps 1500 --checkpoint $C --move_speed 1.0
```

- `--move_speed 0.0`: 静止の評価。
- `--move_speed 1.0`: 速さ U(0, 1.0) の動く敵の評価。カリキュラムを無効にし、静止エピソードも入れない。
- 評価は seed 固定で決定的。1 本約 4 分。
- 出力: 結果の割合、上体の揺れ (傾き / ロール・ピッチ角速度 / 角加速度 / 並進加速度)、敵の初期方位別・目標方向別・敵速度別の表。

### 3. 学習を始める

```bash
nohup env CUDA_VISIBLE_DEVICES=0 uv run train.py --task Isaac-StealDir-K1-v0 --headless --num_envs 2048 \
    --run_name stealdir_w56444_r02 --max_iterations 4000 \
    --resume --load_run init_38200 --checkpoint model_38200.pt --reset_noise_std 0.2 \
    env.commands.steal.move_speed_max=0.5 > stealdir_w56444_r02_nohup.log 2>&1 &
```

`train_steal.sh` は元の PC のシェル関数 (`~/.bash_functions` の `_labpython2` = `uv run`) に依存するので、
別 PC では上のように `uv run train.py` を直接使う。

- **GPU**: 1 GPU × 2048 env で約 3.2 s/iter。4000 iter で約 3.5 時間。これまで 2 本の A/B を GPU 0 / 1 で並走させてきた。
  2 本目は 30 秒以上ずらして起動する (同時起動は USD 変換の一時ファイルが衝突する)。
- **`--reset_noise_std`**: 必須。省略すると optimizer state ごと読まれ、std が負に落ちて落ちる。
  Block の新 expert 再学習では 0.1 が停滞し、0.2 で改善した。Steal はこれまで 0.1 で回してきた。0.2 を本命にし、0.1 と A/B を推奨。
- **`env.commands.steal.move_speed_max`**: 敵速度の上限の初期値。カリキュラムの現在値は checkpoint に保存されないので、
  再開のたびに指定する。新 expert で動きが変わるので、0.5 から上げ直す案にした。ログの `Curriculum/steal_move_speed` で進行を見る。
- **`--max_iterations`**: 追加分の iter 数。
- **止めるとき**: `ps -eo pid,args | grep "run_name <名前> "` の python を PID で kill する。
  `pkill -f <パターン>` は、同じコマンド行に同じ文字列を含む自分の shell も殺す。
- **学習ログで見る指標**:
  - `Episode_Termination/ball_stolen`: 学習中は探索ノイズ込みで、決定的評価より約 10 pt 低い。
  - `base_height` (転倒)。
  - `Curriculum/steal_move_speed`。
  - 揺れの報酬項 `Episode_Reward/{flat_orientation_l2, ang_vel_xy_l2, base_ang_acc_l2, base_lin_acc_l2}`。
  採用判断は下の評価で行う。

## 評価と採用判断

上の `eval_steal.py` を、500 iter ごとの checkpoint に `--move_speed 0.0` と `1.0` の 2 本ずつ回す。

採用基準。現行 model_38200 の**旧 expert 上**での値。新 expert 上の値は上の手順 2 で測る。

| | 奪取 | 転倒 | 傾き | ロール/ピッチ角速度 | 角加速度 | 並進加速度 |
|---|---|---|---|---|---|---|
| 静止 (`--move_speed 0.0`) | 94.3 % | 0.6 % | 0.058 | 0.53 rad/s | 18.7 | 6.2 |
| 動く敵 (`--move_speed 1.0`) | 77.6 % (0.75〜1.0 m/s は 65.5 %) | 2.4 % | 0.058 | 0.54 | 19.1 | 6.1 |

- 静止で奪取 ≥ 93 %・転倒 ≤ 1 %、動く敵で奪取 ≥ 75 % を保ち、揺れが同等以下の checkpoint を選ぶ。
- 新 expert は横に速いので、動く敵で伸びる余地がある。
- 1.0 m/s の失敗は、時間切れ (追いつけない) 約 8 % と方向違い約 9 % が主。
- 歩行 expert 自体が実機未検証。横 1.9 m/s 付近の指令が多く出るなら、実機で先に歩行単体を確認してもらう。
  MuJoCo の試合 sim では、上限に張り付いた指令での転倒が問題になっている (下の「未解決」)。

## 書き出しと配備

```bash
cd scripts/rsl_rl
CUDA_VISIBLE_DEVICES="" uv run python export_steal_policy.py \
    --checkpoint /abs/path/logs/rsl_rl/k1_steal_dir/<run>/model_N.pt \
    --out ../../frozen_policies/best_steal_dir/steal_dir_w56444_N.onnx
cp /abs/path/logs/rsl_rl/k1_steal_dir/<run>/model_N.pt ../../frozen_policies/best_steal_dir/model_w56444_N.pt
```

- **合格条件**: torch と onnxruntime の差が 1e-4 未満。起動ログに `knee on` が出ること (横の膝を ONNX に焼き込んだ印)。
- **ONNX の中身**: 上位 actor のみ。観測正規化 + MLP + クランプ / デッドゾーン / 膝 + 残差スケール。
  入力 `steal_obs (1,69)`、出力 `command (1,3)` と `residual (1,12)`。
- **観測の並び**: 元の 67 次元は `export_steal_policy.py` の docstring にある。方向指定版はその後ろに目標方向 2 次元が付く。
  **観測を変えたら C++ も変える。**
- **実機リポジトリ** booster_k1_locomotion (branch `feat/ball_steal`):
  - ONNX を `assets/k1_steal_dir_<日付>_<run>_<iter>.onnx` として追加する。
  - `launch/k1_isaaclab_sim.launch.py` の `default_steal_model` と README の表を更新する。
  - 入力 69 次元は不変なので、ノードのコードは変更不要。
- **位相周波数**: 新歩行は位相周波数に横加算がある。C++ の `cmd_phase_freq` (k1_constants_isaaclab.hpp) を合わせないと位相がずれる
  (`frozen_policies/best_walk/README.md`)。歩行・Block 側の配備作業と共通なので、済んでいるか確認すること。
- **旧版**: 旧版 (model_38200 / steal_dir_38200.onnx) は `best_steal_dir/` に併置したままで良い。

## 既知の結論 (やり直さなくて良いこと)

- **揺れの罰**: 段階的に 1 → 2 → 4 → 8 倍へ上げ、奪取率を落とさず傾き −69% / 角速度 −70%。8 倍が現在の既定。
  揺れの大きい方策にいきなり 8 倍を掛けると、揺れの罰が奪取報酬を大きく上回る。今回は揺れの小さい 38200 から始めるので問題ない。
- **転倒の罰**: 転倒の罰を −200 → −1600 と段階的に強めるたびに、奪取率も転倒も改善した。足以外・方向違いを −60 にして上積み。
- **カリキュラムの閾値**: 85% では 0.2〜0.3 m/s で停滞した。学習中の奪取率は決定的評価より約 10 pt 低いため。今は 75%。
- **静止エピソード**: 動く敵のカリキュラムで速さを一様に抽選すると、静止の奪取率が 94 → 90.6% に落ちた。静止 30% で戻った。
- **効かなかったこと**:
  - エピソード長 10 s (既定 8 s)。
  - ωz 上限 1.0 (背後スタートの転倒対策のつもり。原因は立位からの後退・横・旋回 + 大きな残差の同時出力だった)。
  - 目標方向の速度報酬 10 (5 より悪い)。
- **学習の型**:
  - スクラッチ学習は遅いので、常に既存 checkpoint からの resume。
  - 観測を足すときは `expand_steal_checkpoint.py` で入力列を 0 埋めで拡張する。`--nets critic` で critic だけも可。

## 未解決 (今回の再学習の対象外、判断はユーザーと)

MuJoCo の 5v5 試合 sim で、奪取の起動あたり約 14% が転倒した (学習時 1%)。調査結果の要点は次のとおり。

- **ノード側の修正**: キック後に混合を保持したまま再開していた。修正済み (booster `2e7b948`、Walk 以外に移ったら混合をリセット)。
- **単体 sim での切り分け**: 学習と同じ配置の単体 MuJoCo sim で、転倒 24%。ボールを全周で検出させると 16.7%。
  - 開始直後の転倒と時間切れは、カメラ視野外のボール / 敵 (ノードが古い位置や仮想の敵を使う) が主因。
  - 残りはボール接触後の転倒。敵の観測が視野外で仮想になること / 敵の形状 (学習は円柱、sim はロボット) / 物理差のどれかで、未分離。
- **学習側の候補** (未着手):
  - 観測の視野制限・遅延を学習に入れる。
  - 歩行中スタート・途中解除を学習に入れる。
  - 出力 (指令の上限・残差) を抑える。

## 結果: 歩行 expert R9 (model_64440) + 摩擦 DR での再学習 (2026-11-21、vast.ai)

run `2026-11-21_06-03-35_stealdir_w64440_fricfeet_r02`: model_38200 から reset 0.2・敵速度上限 0.5 開始で 4000 iter
(約 2.1 s/iter)。カリキュラムは +2400 iter で 1.0 m/s に到達。成果物は `best_steal_dir/model_w64440_42199.pt` /
`steal_dir_w64440_42199.onnx` (parity 2.1e-6、knee on)。**実機未検証・採用判断はユーザー。**

- **摩擦 DR**: 歩行 R9 と同じ U(0.3, 3.0)・env ごとに 1 つ。ただし**足 (`.*_foot_link`) だけ**に掛ける (`K1StealEnvCfg` の既定値)。
  全 body に掛けると μ > 1 の env で物理が発散し (揺れの罰が −1e7 台、value loss 2e8)、120 iter で方策が崩壊した。
  敵の円柱を無摩擦にしても発散は残った (足以外の body 同士か対ボールの接触。未特定)。足だけなら 4000 iter で発散 0 回。
- **評価** (1024 env × 1500 step、新しい摩擦 DR の下。上の採用基準は旧 expert・旧摩擦 U(0.6, 1.0) での値):

| | 奪取 | 転倒 | 傾き | ロール/ピッチ角速度 | 角加速度 | 並進加速度 |
|---|---|---|---|---|---|---|
| 38200 載せ替えのみ・静止 (摩擦は全 body) | 22.4 % | 25.3 % | 0.079 | 0.78 | 25.6 | 7.4 |
| 38200 載せ替えのみ・静止・旧摩擦 | 26.4 % | 17.5 % | 0.075 | 0.71 | 23.4 | 7.2 |
| 42199 静止 | 91.2 % | 1.3 % | 0.050 | 0.54 | 19.8 | 5.9 |
| 42199 動く敵 1.0 (0.75〜1.0 m/s は 62.4 %) | 75.4 % | 3.6 % | 0.050 | 0.54 | 20.0 | 5.9 |

- 39700 以降の 6 checkpoint は静止 89.8〜91.8 % / 動く敵 72.7〜75.4 % で横ばい (評価ログは run の `eval/`)。
  静止の基準 (奪取 ≥ 93 %・転倒 ≤ 1 %) には 2 pt / 0.3 pt 届いていない。
