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

## 次の再学習: 試合での使われ方から足す状況 (2026-11-22)

MuJoCo の 3v3 試合 sim (`3v3_vision_game.sh --bhuman --mcl`、25 分 × 15 試合、model_w64440_42199) で転倒を測った。
- 奪取方策の作動中は 1.09 回/分 で、歩行中 (0.10 回/分) の約 10 倍倒れる。
- 混合 (切替) の途中で倒れたものはほぼ無い。w = 1 になってから 0.3〜2.8 s 後に倒れている。
- 71 % は他機と接触していない。

原因は、学習で見ていない入力と開始状態にある (詳細は futbol_main `src/ros2_ws/experiment_switch_falls/REPORT_3v3_vision.md`)。
以下は、試合での起動・終了条件から見て学習に入れるべき状況で、根拠の強い順に並べた。

**観測 69 次元の並びは変えない。** 変えなければ C++ ノードも ONNX の入出力も変更不要。見失いなどは、下の「ノードの代替則」を学習 env で再現して入れる。

### 試合での使われ方 (strategy `activate_ball_steal_mode` と locomotion ノード)

- **起動**: 次の 3 つが揃ったとき。
  - 敵がボールから 2.0 m 以内に居る (`contest_enemy_ball_distance_cm`)。
  - 自分がボールから 2.0 m 以内に居る (`max_ball_distance_cm`)。
  - 推定ボール速度が 1.5 m/s 以下 (`max_ball_speed_cm_s`)。
  - ただし、回り込まずに空いた方向へ蹴れるときは起動しない。
  - 起動の直前までは、歩行方策でボールへ向かって歩いている。
- **方向**: 起動時に 1 回だけ送る。起動時の向きを基準に前/左/後/右へ丸めた値で、学習と同じ。
- **終了**: 次のどれかで歩行へ戻る (混合 0.3 s)。学習のような成功・失敗の終端は無い。
  - ボールが自分から 2.0 m より遠くなる (蹴り出した・取られた)。
  - 敵が全員ボールから 0.8 m 以上離れ、キックで勝てる。
  - 起動から 4 s 経つ (`max_hold_s`)。
- **ノードの代替則** (`rl_policy_isaaclab_node.cpp` の steal_step / steal_opponent):
  - **ボールを見失った tick**: 最後に見た位置を odom で今の body 系へ移して使う。速度は 0 にする。
  - **敵**: 検出ロボットのうちボールに最も近いもの。ボールから 1.0 m 以内のものだけを敵とみなす (`OPPONENT_MAX_M`)。
    - 該当が無ければ「ロボット→ボールの延長線上、ボールの 0.4 m 先」に仮想の敵を置く (`steal_virtual_opponent_m`)。
    - 検出は味方・敵を区別しない。
- **観測の元**: ボールと敵は vision_filter の推定値で、頭部カメラの水平視野は 105°。首は作動中ボールを追う。

### 足す状況 (優先順)

1. **歩行中からの開始**: 根拠が最も強い。
   - 試合では、起動直前のコマンドが中央値で |vx|+|vy| ≈ 2.3 m/s、旋回は 0.5〜3 rad/s。学習は立位スタート (base 速度 ±0.1)。
   - 開始時のコマンドが 1.5 m/s 未満だった起動では、転倒は 0/25 回だった。
   - 案: エピソードの最初の 0.5〜2 s は、凍結した歩行 expert にボールへ向かうコマンド (上限付近、旋回込み) を与えて歩かせる。
     その後ノードと同じ 0.3 s の線形混合で奪取方策へ移す。
     - 混合中も、方策の出力 × w + 歩行コマンド × (1 − w) で実行する。
     - 立位スタートは 20〜30 % 残す。

2. **配置を試合の起動条件に合わせる**:
   - ボールはロボットから 0.1〜2.0 m。
   - 敵はボールから 0.31〜2.0 m で、ボールへ向かって歩いてくる。現行は 0.31〜0.5 m で敵にボールが付いている配置だけ。
   - 敵がボールから 1.0 m より遠い間は、観測を仮想の敵にする (5 と同じ規則)。

3. **途中で蹴られて速く動くボール**:
   - 真のボール速度別の転倒率は、< 0.5 m/s で 0.61 回/分、0.5〜3 m/s で約 1.6、≥ 3 m/s で 3.1。
   - strategy は速さを起動時にしか見ていないので、蹴られた後も方策は動き続ける。終了判定 (ボール 2 m 超) までに 0.5〜1 s かかる。
   - 案: 敵がボールに届いたエピソードの一部 (30 % 程度) で、ボールに 2〜6 m/s・任意の方向の速度を与える。開始時の速度は 0〜1.5 m/s。
     - ボールが 2 m より遠くなったら打ち切る。転倒しなければ中立 (罰なし)。
     - 「ボールを追って倒れない」ことを学ばせる。

4. **ボール観測の劣化**:
   - 試合の作動中の観測誤差は中央値 0.15〜0.18 m、90 % 点 1.2 m、> 0.5 m が 22 % (学習は ±5 cm)。
   - 誤差 0.2 m 以上で転倒率は 3〜4 倍。0.2 m 以上の跳びの約半数は、倒れ始めの 0.3〜1.0 s 前に起きている。
   - 跳びの原因は、自己位置の補正で body 系の位置が飛ぶことと誤検出。
   - 案:
     - 遅延: 0.05〜0.2 s。
     - 視野外の見失い: 頭の向き = ボール方向として ±52° の外、および足元の近距離。見失った間は上の「最後に見た位置を odom で移す・速度 0」。
     - 跳び: 確率的に 0.2〜1.0 m のオフセットを 0.1〜0.5 s 足す。
     - 速度観測: 遅れと、0 になる区間 (vel_valid = false)。

5. **敵が見えない・仮想の敵**:
   - ボールから 1 m 以内に本当に敵が居るのに仮想の敵になっていた時間が 34〜36 %。逆に、実検出なのに近くに敵が居ない時間が 18〜20 %。
   - 実/仮想の切替で、観測は 1 tick に 1 m 以上跳ぶ。
   - 転倒率は、仮想の間 3.4 回/分、実検出の間 2.1 回/分。
   - 案: 敵の観測をノードの規則で作る。
     - 視野外、またはボールから 1 m 超のときは仮想の敵。
     - それとは別に、確率的に見落とす (仮想になる) 区間と、別のロボット (味方役) を敵とみなす区間を入れる。

6. **途中解除と保持の打ち切り**:
   - 試合では最長 4 s で打ち切られ、0.3 s の混合で歩行へ戻る。成功ラッチ後の 1.5 s を待たないこともある。
   - 解除の混合中の転倒は少なかったが (steal_off 1 回)、7 の出力抑制と合わせて入れておくと安全。
   - 案: 1〜4 s のランダムな時点で歩行 (コマンド 0 または次の移動先) へ混合で戻し、その後 1 s 倒れなければ中立。

7. **出力の大きさ** (任意、上の 1〜6 で減らなければ):
   - 転倒率は、コマンドが上限付近のとき 3.0 回/分、残差ノルム ≥ 2 で 41 回/分。横コマンドの ±1.9 の急反転もある。
   - 歩行 expert 自体は、上限付近のコマンドでもほぼ倒れない (78 分で 3 回)。
   - ωz 上限 1.0 は以前効かなかった (上の「効かなかったこと」)。罰を足すなら、コマンド変化の罰を強める方から試す。

### 評価に足すもの

`eval_steal.py` に「試合条件」モードを足し、1〜5 を全て有効にした評価を、現行の静止 / 動く敵の 2 本に加える。
- 現行 42199 の値をこのモードでまず測り、ベースラインにする。
- 採用は、静止・動く敵の基準を保ったまま、試合条件の転倒が下がった checkpoint から選ぶ。
- 最終確認は futbol_main の MuJoCo 3v3 sim で、steal 作動 1 分あたりの転倒 (現行 1.1〜1.8 回/分)。
  測り方は `experiment_switch_falls/summarize.py` を使う。

### 注意

- 上の数値は MuJoCo の試合 sim の値で、実機ではない。ただし、vision 機と真値検出 (ノイズ・遅延付き) 機で、観測の質も転倒率もほぼ同じだった。
- strategy の起動・終了の閾値を変えたら、ここの分布 (2・3・6) も合わせる。

## 再学習 (2026-11-22): 試合条件 + Block 型の履歴 encoder (Isaac-StealHist-K1-v0)

コミット 90bf0a9。上の「次の再学習」の 1〜6 は `StealCommand` (mdp/steal.py) に実装済み。
StealDir でも既定で有効 (`match_enable`)。

### 分かっていること (このサーバーで測定、決定的評価 1024 env × 1500 step)

| 方策 | 静止: 奪取 | 静止: 転倒 | 動く敵 1.0: 奪取 | 試合条件: 作動 1 分あたりの転倒 |
|---|---|---|---|---|
| 現行 w64440_42199 | 90.6% | 1.6% | 76.0% | **1.35** (MuJoCo 試合 sim の 1.1〜1.8 を再現) |
| MLP のまま試合条件で +1000 iter (A@45200) | 81.3% | 1.4% | | 0.34 |
| MLP のまま試合条件で +2000 iter (A@46198) | 78.0% | 1.7% | | 0.42 |

* 試合条件で学習すると、転倒は 1/4 に減る。一方で静止の奪取率が 90% → 80% に落ち、戻らない。
* ユーザーの判断: 落ちる原因は観測劣化 (1 フレームの観測では見失い・遅延と真値を区別できない)。
  そこで Block と同じ履歴 encoder を入れる。原因の切り分け run は不要。
* コマンド変化の罰を強めても (−0.2 → −0.5) 効果は無かった。

### 構成

* actor = `steal_policy` (69、従来と同じ並び) + `steal_hist`。
  * `steal_hist` は 30 step × 8 の履歴: 方策が見たボール位置・速度、敵位置、ボールが見えているか、敵が仮想か。
  * world 系で持ち、毎 step 現在の base yaw 系に直す。
* critic = `steal_critic` (79) + `steal_hist_true` (30 × 6 の真値)。
* 履歴は causal transformer encoder で latent にする。`BlockPPO` が actor 側の encoder を真値側へ latent 蒸留する。
  方策クラスは `StealHistActorCritic` (BlockActorCritic を継承)。
* 敵・ボールの速度上限は 1.6 m/s (`move_speed_max` = `move_speed_limit`)。
  試合条件では学習中の奪取率が昇格条件 0.75 に届かず、カリキュラムが上がらないため固定にしている。
* `num_mini_batches` は 8 にしている。4 だと 2048 env でも 20 GB の GPU で OOM になる。
* 速度: 1 iter 約 17 s (RTX 20 GB 級、2048 env、収集 4.2 s + 学習 12.5 s)。MLP 版は 3.6 s。

### 開始点を作る

変換済みの開始点を `frozen_policies/steal_hist_init/` に置いてある。encoder の乱数初期化も含め、このサーバーの run と同じもの:

| ファイル | 元 | 備考 |
|---|---|---|
| `model_hist_42199.pt` | 現行 w64440_42199 | 静止 90.6%、試合条件 1.35 回/分 |
| `model_hist_A45200.pt` | MLP 版を試合条件で +1000 iter した A@45200 | 静止 81.3%、試合条件 0.34 回/分 |

logs に置いて resume する (`--checkpoint` はファイル名、`--load_run` はディレクトリ名):

```bash
cd scripts/rsl_rl
mkdir -p logs/rsl_rl/k1_steal_hist/2026-11-22_00-00-00_init42199
cp ../../frozen_policies/steal_hist_init/model_hist_42199.pt logs/rsl_rl/k1_steal_hist/2026-11-22_00-00-00_init42199/model_42199.pt
mkdir -p logs/rsl_rl/k1_steal_hist/2026-11-22_00-00-01_initA45200
cp ../../frozen_policies/steal_hist_init/model_hist_A45200.pt logs/rsl_rl/k1_steal_hist/2026-11-22_00-00-01_initA45200/model_45200.pt
```

A@45200 起点なら `--load_run 2026-11-22_00-00-01_initA45200 --checkpoint model_45200.pt` にする。
このサーバーでは両方を回している (下の「このサーバーの run」)。

自分で作り直す場合は、`frozen_policies/best_steal_dir/model_w64440_42199.pt` から次の手順で作る:
`frozen_policies/best_steal_dir/model_w64440_42199.pt` から作る:
1. critic を +3 拡張 (79 次元化)。
2. 履歴版に変換。latent 入力の重みは 0 なので、変換直後の出力は元と同一。自己チェック付き。

```bash
cd scripts/rsl_rl
uv run python expand_steal_checkpoint.py --src ../../frozen_policies/best_steal_dir/model_w64440_42199.pt \
    --dst logs/rsl_rl/k1_steal_dir/2026-11-22_00-00-00_c79/model_42199.pt --extra 3 --nets critic
uv run python convert_steal_to_hist.py --src logs/rsl_rl/k1_steal_dir/2026-11-22_00-00-00_c79/model_42199.pt \
    --dst logs/rsl_rl/k1_steal_hist/2026-11-22_00-00-00_init42199/model_42199.pt
```

### 学習

```bash
# 1 GPU
CUDA_VISIBLE_DEVICES=0 uv run train.py --task Isaac-StealHist-K1-v0 --headless --num_envs 2048 \
    --run_name hist_from42199 --max_iterations 3000 \
    --resume --load_run 2026-11-22_00-00-00_init42199 --checkpoint model_42199.pt --reset_noise_std 0.2
# 複数 GPU (BlockPPO は multi-GPU 対応、num_envs は GPU あたり)
uv run torchrun --standalone --nproc_per_node=<N> train.py --task Isaac-StealHist-K1-v0 --headless --distributed \
    --num_envs 2048 --run_name hist_from42199 --max_iterations 3000 \
    --resume --load_run 2026-11-22_00-00-00_init42199 --checkpoint model_42199.pt --reset_noise_std 0.2
```

* `--reset_noise_std` は必須。変換で optimizer を捨てているため。
* `--max_iterations` は追加分。
* 開始点のディレクトリ名は、run 名で終わらせないこと。`rm -rf *_<run名>` で一緒に消える (一度やった)。
* 学習中に見る値: `Mean latent loss` (初期 0.4 → 0.16 程度に下がる)、`Episode_Termination/base_height`。

### 評価 (500 iter ごと、4 条件)

```bash
E="uv run python eval_steal.py --headless --task Isaac-StealHist-K1-Play-v0 --num_envs 1024 --steps 1500 --device cuda:0 --checkpoint <ckpt>"
$E --move_speed 0.0            # 静止 (観測は真値)
$E --move_speed 1.0            # 動く敵 (観測は真値)
$E --match --move_speed 1.0    # 試合条件。最後の [match] 行の「回/分」が作動 1 分あたりの転倒
$E --match --move_speed 1.6    # 試合条件・速い敵
```

* 試合条件の奪取率は 20% 前後と低い。35% 前後が罰なし打ち切り (neutral: 途中解除・遠くへ蹴られた) になる設計のため。
  見るのは転倒の回/分。
* 評価は学習と同じ GPU でも動く。ただしその間、学習が 1.3 倍ほど遅くなる。

### 採用基準

* 静止 93% 以上・転倒 1% 以下 (少なくとも 42199 の 90.6% / 1.6% を下回らない)。
* 動く敵 1.0 で 75% 以上。
* 試合条件の転倒が 42199 (1.35 回/分) より大きく下がる (MLP 版 A@45200 の 0.34 が目安)。

### 未対応

* ONNX 書き出し (`export_steal_policy.py` は MLP 版のみ)。
* booster ノード: 履歴 30 step を odom で現在系に直して入れる。見えている・仮想のフラグも要る。
  採用が決まってから行う。

### このサーバーの run (2026-11-22 18 時〜)

* `hist_fromA45200` (GPU1)
* `hist_from42199` (GPU0)

どちらも 3000 iter、`--reset_noise_std 0.2`。別サーバーで回すなら、同じ 2 本を速く進めるか、
GPU 数を増やして env 数を増やした版にする。同じ設定を重複して回す必要は無い。

## 結果 (2026-11-23): 教師-生徒で観測劣化に対応した s5 を採用、実機で転倒がほぼ無くなった

成果物は `frozen_policies/best_steal_hist/` にある。booster_k1_locomotion の既定は `k1_steal_hist_2026-11-23_s5_1000.onnx`
(= `steal_hist_s5_1000.onnx`)。ユーザーの実機確認では、転倒はほぼ無くなった。

| 方策 | 静止: 奪取 / 転倒 | 動く敵 1.0 | 劣化ありの試合条件: 奪取 / 作動 1 分あたりの転倒 |
|---|---|---|---|
| 旧既定 w64440_42199 | 90.6% / 1.6% | 76.0% | 21.3% / 1.35 |
| 教師 t1 model_44700 (劣化なしで学習) | 88.6% / 1.3% | 71.1% | 22.4% / 0.70 (劣化なしなら 0.50) |
| **生徒 s5 model_1000 (採用)** | **86.1% / 1.9%** | **69.8%** | **31.1% / 0.50** |

静止・動く敵の奪取率は旧既定より下がった。増えた失敗はボールに届いた後の扱いで、敵に当てる・体で押し出すの 2 つ。
「近づかずに止まる」失敗 (時間切れ) は増えていない。

### 分かったこと (やり直さなくて良いこと)

* 観測劣化を入れて PPO で直接学習すると、方策全体が慎重になり、真値の観測でも成績が落ちる
  (静止 −5 pt、動く敵は時間切れが増えて 76 → 46〜55%)。以下はどれも効かなかった。
  * 履歴 encoder を入れる
  * 劣化の割合を 0.5 に下げる
  * 時間切れに罰を足す (報酬の `timeout` 項、既定 0)
  * 長く回す
  * 遅延を外す / 短くする
* 1 条件ずつの切り分け (+400 iter): 能力を落とすのは観測劣化だけ。歩行中開始・寄ってくる敵・キック・途中解除・
  速度 1.6 m/s・構造の変更は無害。劣化を要素ごとに入れると単独では小さく (遅延が最大で動く敵 −9 pt)、
  全部入れると足し算以上に落ちる。
* 対策: 劣化なしの試合条件で PPO の教師を作り、劣化ありの試合条件で生徒を DAgger 蒸留する。
  教師の行動を教師あり学習で真似るので、RL で慎重になる問題が起きない。
* 蒸留の改善は頭打ち。劣化なしの step の損失の重み 10 (`--clean_weight 10`) が少し効いた (静止 +1.5 pt)。
  30 倍、学習率 1e-4、劣化 0.5 は差なし。

### 再現手順

```bash
cd scripts/rsl_rl
# 1. 教師: 42199 起点の履歴版 (frozen_policies/steal_hist_init/model_hist_42199.pt を
#    logs/rsl_rl/k1_steal_hist/2026-11-22_00-00-00_init42199/model_42199.pt に置く) から、劣化なしで 3000 iter
uv run train.py --task Isaac-StealHist-K1-v0 --headless --num_envs 2048 --run_name t1_teacher --max_iterations 3000 \
    --resume --load_run 2026-11-22_00-00-00_init42199 --checkpoint model_42199.pt --reset_noise_std 0.2 \
    env.commands.steal.obs_degrade_prob=0.0
# 2. 生徒: DAgger 蒸留。1500 iter で頭打ち。s5 は s1 の 1200 から --clean_weight 10 で 1000 iter 追加したもの
uv run python distill_steal.py --headless --teacher <t1>/model_44700.pt --run_name s1 --num_envs 2048 --iterations 1500
uv run python distill_steal.py --headless --teacher <t1>/model_44700.pt --init <s1>/model_1200.pt \
    --run_name s5_cw10 --num_envs 2048 --iterations 1000 --clean_weight 10
# 3. ONNX。蒸留 run には params が無いので、--run_dir に教師の run (または frozen_policies/best_steal_hist) を渡す
uv run python export_steal_hist_policy.py --checkpoint <s5>/model_1000.pt --run_dir ../../frozen_policies/best_steal_hist \
    --out <out>.onnx
# 評価 (生徒・教師とも StealHist の Play で読める)
uv run python eval_steal.py --headless --task Isaac-StealHist-K1-Play-v0 --num_envs 1024 --steps 1500 --device cuda:0 \
    --checkpoint <ckpt> [--move_speed 0.0 | --move_speed 1.0 | --match --move_speed 1.0]
```

* 1 iter の時間 (RTX 5090): 教師 5 s、蒸留 1.8 s。
* `--match` 評価で教師を劣化なしで測るときは `env.commands.steal.obs_degrade_prob=0.0` を足す。

### 未解決

* 静止・動く敵の奪取率を旧既定の水準 (90.6% / 76.0%) に戻すこと。教師自体が 88.6% / 71.1% なので、まず教師の改善が要る。
* ノードは奪取の起動 tick に履歴を今の値で埋める。学習では起動前の歩行中から履歴が積まれているので、ここは学習とずれる。
