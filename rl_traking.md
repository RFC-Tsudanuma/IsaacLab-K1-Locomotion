# rlによる経路追従の学習
## 学習目的
この学習では、経由点を追従し、以下の2つのモードに従って目標の姿勢や位置に最速で到達するポリシーを学習する
- モード1
 - ロボットの相対位置で指定される位置に対して、ロボットの上体が+-90度以内を向き続けるようにしながら、目標位置に最速で到達する
- モード2
 - ロボットの相対位置で指定される目標姿勢(x,y,theta)に、最速で到達する

## 方策の実装方針
この方策は、すでに学習済の歩行専門家方策と、pivotターン専門家方策を活用する上位方策として実装する。それぞれの専門家間の遷移は、専門のゲーティングNNがあり、そちらが担当するので、トラッキング方策はそれら2つの専門家を活用して、最速で指定されたモードの目標を達成するようにすれば良い

### 観測
観測は、専門家の内界センサの観測(2つとも共通) + 目標相対位置(x,y) + 目標相対姿勢(theta) + 上体が保つ目標の相対位置 + モードを表す2値で構成される。
歩行の専門家は、短長の履歴を観測に入れているが、こちらはひとまず短い履歴のみを入力に入れる。（後で余裕があったら専門家と同じく長い履歴も入れてみる）

### 報酬関数
基本的には両方のモードとも、目標の位置や姿勢に最速で到達するのが目標。そして、転倒には重いペナルティを設定する。
モード1の報酬では、上体を可能な限り向ける目標と90度以内に保たれるようにペナルティ等を入れる。ただ、これはハード制約ではないので、少しくらいなら破れても良い。基本的には、モード1は、サッカーでディフェンダーが守備のためにボールを見ながら後ろに下がるときのような、あるものを見ながら（常に視界に入れながら）移動するために活用するものだということを念頭に置く。そのため、上体がぴったりその見つめる対象に向く必要はなく、ある程度ずれても首がそっちを向けば良いので許される。ただ、首の可動範囲（左右90度）を超えてはならない。



### カリキュラム
おそらく、モード1の方が難易度が高いため、モード2から学習を開始して、徐々に2を混ぜるといった方が良い気がする。


### アクションについて
歩行方策に関しては、xが+-2.0m/sまで、yは1.2m/s程度まで、thetaは1.5rad/sくらいまで出すことができる。yは学習の範囲は0.9m/sだが、実機でも1.2m/sまでは出せているのでその程度は出しても良い。
歩行方策へのコマンドと、pivotのコマンドは同時に出しても良い。むしろ、ある程度同時に出さないと、pivotしてすぐに歩き出すというのがうまくできない。今、人間の書いた追従アルゴリズムだと、pivotしたあとに0.3秒ほど立ち止まる時間ができてしまいそれがタイムロスである。その辺をうまく無くせるように上位方策が学習できると良い。振動を避けるため、基本的にはどちらのコマンドにもaction rateやaction smoothness系のペナルティは導入するべきだろう。
---

# 実装計画 (2026-11-05 レビュー反映版)

上の方針に対するレビューで決まったこと: 経由点は残す (fast/strict は廃止、通過半径は 1 つ)、
モード 1 のゴール到達時の向きは自由、ゴール判定は位置 0.1 m / yaw 0.1 rad、C++ の pivot 経験則は
この方策が有効なときだけ無効化する。以下はその実装計画。既存の経由点追従 (Isaac-Waypoint-K1,
`mdp/waypoint.py` / `agents/waypoint_policy.py`) を土台にし、新タスク名は **Isaac-Tracking-K1-v0** とする。

## 1. 下位 (凍結) の構成 — 融合 ONNX と同一にする

- 下位は **walk expert + turn expert + 凍結ゲート** (`GatingActorCritic`, gate_v4 `k1_gating/2026-11-01_08-19-20_gate_v4_newwalk/model_5897.pt`、expert は `frozen_policies/best_walk` / `best_turn` を自動解決) を
  そのまま内包する。実機の `k1_gating_walk_turn_2026-11-01_v4_5897.onnx` と等価な構成で学習しないと、
  上位がゲートの癖 (遷移の速さ・早期打ち切り時の挙動) を実機と違う形で学習してしまう。
- 上位 → 下位の受け渡しは経由点追従と同じ「`PPO.act` が差し替える」方式。上位の出力から
  `command` グループ・`policy` 履歴の最新フレームのコマンド枠・`expert_mode` を差し替えた観測で
  `GatingActorCritic.act_inference_blended` を評価し、12 次元の関節指令を env に渡す。

## 2. アクション (上位の出力, 5 次元)

| 枠 | 意味 | 後処理 |
|---|---|---|
| 0–2 | 歩行コマンド (vx, vy, ωz) | クランプ (2.0, 0.9, 1.0)、ノルム < 0.1 で 0 (停止)。vy 1.2 / ωz 1.5 は expert の学習分布外なので採用しない (v3 の実測: ωz 2.0 でも到達時間は縮まず精度だけ低下) |
| 3 | pivot 目標相対角 Δψ | クランプ ±π。**turn へ切り替えた step で env がラッチ**し、turn 中は固定目標として turn expert に残り角 Δψ/π を渡す (C++ の `/pivot_turn_cmd` と同じ契約) |
| 4 | mode (walk / turn) | `> 0` で turn。**最小滞在時間 0.3 s** を env が強制し、滞在中の切替要求は無視する |

- ガウス方策のまま 4 枠目をしきい値で 2 値化する (最小実装)。ばたつきは最小滞在時間で物理的に封じ、
  切替 1 回ごとの小さな罰 (−0.5) で無駄な切替を抑える。
- action 項は `WalkCommandAction` に `dim` (既定 3) を足して 5 次元で使う (Steal タスクが import 中なので既定値は変えない)。

## 3. コマンド項 (2 つ)

### 3.1 `ExternalTransitionCommand(TransitionCommand)` — 下位へのコマンド (名前は `base_velocity`)
- `TransitionCommand` の自律遷移 (`compute` / `_resample_command`) だけを「action 項の値に従う」実装に
  差し替える。mode・`heading_error`・切替時の履歴コマンド枠ゼロ埋め (`_zero_command_history`)・
  位相生成 (`phase_obs` は turn 中 cmd_xy=0、|Δψ|<9° でゼロ化) は親の実装をそのまま使う。
- `turn_early_exit_rad` / `yaw_assist_*` は **0 (無効)** にする。早期打ち切りも歩き出しの ωz も上位が自分で出す。
  C++ 側も同じ (§8)。
- エピソード開始時の mode は 20 % で turn から始める (ゲートの学習分布 `initial_turn_prob` に合わせる)。

### 3.2 `TrackingGoalCommand` — 経由点 + 目標 + 見つめる対象 (名前は `tracking`)
- `WaypointCommand` から fast/strict を外し (`pass_radius` は 1 値、**0.3 m** を提案。要確認)、以下を追加:
  - タスクモード `task_mode` (0 = 姿勢到達 / 1 = 見ながら到達)、50/50 で抽選。
  - 見つめる対象 `look_at_w` (xy)。経路の中点から 1.5–5 m、方向一様で抽選 (後退・横歩きが必要になる配置を含む)。
  - 判定: 位置 0.1 m、yaw 0.1 rad (task_mode 0 のみ)。task_mode 1 の到達判定は位置のみ。
- 経路生成は現行と同じ (経由点 0–4、セグメント 0.6–2.0 m、折れ角 ≤ 2.0 rad) だが、**最初のセグメントの
  向きは機体 yaw に対して ±π 一様**にする (pivot が有効な状況を十分に作る)。
- 観測 (`command`, base yaw 系, 16 次元): 次の 4 点 xy (8) + ゴール xy (2) + ゴール yaw 誤差 cos/sin (2, mode 1 では 0)
  + 見つめる対象 xy (2, mode 0 では 0) + task_mode one-hot (2)。
- メトリクス: `reached_goal`, `time_to_goal`, `goal_dist`, `goal_yaw_err`, `look_at_err` (|yaw 誤差|),
  `look_at_violation_frac` (|誤差| > 90° の時間割合), `pivot_frac` (turn モードの時間割合), `mode_switches`。

## 4. 観測 (上位, `nav_policy` / `nav_critic`)

v2 (22 次元) で到達率 99.97 % だったので、内界観測の履歴は入れずに始める。

| 項 | 次元 |
|---|---|
| `tracking` コマンド (§3.2) | 16 |
| 直前の自分の出力 (walk cmd 3 + mode 1) | 4 |
| 下位の状態: 現在の gate mode one-hot (2)、切替からの経過時間 (1)、ラッチ中の残り角 Δψ (1) | 4 |
| base 線速度 (3)、base 角速度 (3)、projected_gravity (3)、gait_phase (4) | 13 |
| 合計 | **37** |

critic は同じ項をノイズ無しで。停滞したときだけ関節位置・速度の短履歴 (4 フレーム) を足す。

## 5. 報酬 (dt = 0.02 を掛ける前の値)

| 項 | 重み | 備考 |
|---|---|---|
| `path_progress` (残り折れ線長の減少, ポテンシャル) | 50 | 現行と同じ |
| `goal_position_exp` (全経由点通過後, std 0.15) | 2 | 居続け = 最速到達の動機。task_mode 1 では「かつ 見つめ誤差 ≤ 90°」でゲート |
| `goal_heading_exp` (task_mode 0 のみ, std_yaw 0.15, std_pos 0.15) | 2 | |
| `look_at_penalty` (task_mode 1 のみ) | −1 | 誤差 e: |e| ≤ 60° で 0、60–90° で ((|e|−60°)/30°)²、90° 超で 1 + 定数 2 (首可動域外) |
| `walk_command_rate_l2` | −0.05 | v2 の教訓: −0.5 では探索ノイズのコストが進捗報酬を上回り学習しない |
| `mode_switch` (切替 1 回) | −0.5 | |
| `termination_penalty` | −200 | |

転倒を増やしたくなければ −400 (ゲート学習と同じ) まで上げる。モード間のカリキュラムは入れない (停滞したら検討)。

## 6. リセット・外乱・エピソード

- リセットは親 (`K1FlatStancePlanePolishCfg`) の乱れた状態 (関節 ±30°、初速 ±0.5) を**残す**。
  v3 で立位スタートに変えたら頑健性が落ちて v2 に負けた。試合中は移動中に新しい目標を受けるので多様な初期状態が必要。
- 評価だけ立位スタート (関節 ±5°、初速 ±0.1) で行い、到達率・到達時間・転倒率を v2 と同条件で比べる
  (`eval_waypoint_falls.py` を新タスク対応にする)。
- push は現行と同じ (0.5 m/s, 4–8 s ごと)。エピソード 20 s。

## 7. 方策・学習

- `agents/tracking_policy.py`: `TrackingActorCritic(ActorCritic)` (5 次元出力、凍結 `GatingActorCritic` 内包、
  `env_actions` = cat[関節指令 12, 上位出力 5]) と `TrackingPPO(PPO)` (storage は `nav_*` のみ)。
  `WaypointPPO` とほぼ同じなので共通化できる所は継承で済ませる。
- runner cfg: hidden [256, 128], init_noise_std 0.5, **entropy_coef 0.0** (クランプ付き連続コマンドの必須条件)。
- 学習: 2 GPU 分散 2048 env × 2、3000 iter (mode の学習が入るので経由点追従の 2 倍を見込む、約 4 時間)。
- 成否の判断 (立位スタート評価, 6000 エピソード): 到達率 ≥ 99.5 %、転倒 ≤ 0.2 %、
  task_mode 1 の 90° 超過時間割合 ≤ 2 %、pivot を使う経路で v2 (walk のみ) より到達時間が短いこと。
  pivot の効果は「初期 yaw 誤差 ≥ 120° の経路」に絞った到達時間で確認する。

## 8. デプロイ (booster_k1_locomotion)

- 上位だけ ONNX 化 (`export_waypoint_policy.py` を流用: `nav_obs (1,37)` → `[cmd 3, Δψ 1, mode 1]`)。
  クランプ・デッドゾーン・mode のしきい値は ONNX 内に焼き込む。
- 入力: `/waypoint_path` (nav_msgs/Path, 現行) に加え `/look_at_point` (geometry_msgs/PointStamped, 機体相対→odom 系固定)。
  look_at が有効なら task_mode 1、無ければ 0。
- `rl_policy_isaaclab_node.cpp`: `tracking_model_path` を追加。これが有効な間は
  **`PIVOT_DONE_*` / `PIVOT_EARLY_EXIT_RAD` / yaw assist を使わず**、mode 出力 → `set_mode`、
  turn 突入 step で Δψ をラッチ、最小滞在 0.3 s を env と同値で実装する。ゲートの mode 入力・履歴コマンド枠の
  ゼロ埋めは既存の `set_mode` がそのまま使える。
- 学習 env と C++ で一致させる定数: クランプ値、デッドゾーン 0.1、最小滞在 0.3 s、Δψ ラッチ、
  通過半径、ゴール判定 0.1 m / 0.1 rad。

## 9. 作業順序

1. `mdp/tracking.py` (action 項の dim 拡張、`ExternalTransitionCommand`、`TrackingGoalCommand`、報酬) → 64 env smoke
2. `agents/tracking_policy.py` + `tracking_env_cfg.py` + runner cfg + タスク登録 → 64 env × 20 iter で学習が回ることを確認
3. 2 GPU 本学習 3000 iter → 立位スタート評価 (v2 と同条件)
4. 上位 ONNX エクスポート → C++ 実装 (§8) → sim で `/waypoint_path_1` + `/look_at_point_1` の動作確認

## 決定事項 (2026-11-05 ユーザー回答)

- 通過半径 0.3 m、見つめる対象の抽選範囲 1.5–5 m で確定。

## 後で追加するカリキュラム (候補): 動く目標・動く対象

実際はサッカーの試合で使われ、目標位置・姿勢は頻繁に差し替わり、見つめる対象 (ボール・相手選手) も連続的に動く。
静止目標で学習した方策でも十分動くと見込むので最初は入れないが、最終性能を狙う段階で以下を env に追加する:

- エピソード途中での目標差し替え (経路・ゴール姿勢を一定確率で再抽選。`prev_path_length` のポテンシャルは
  差し替え step でリセットして進捗報酬の飛びを防ぐ)。
- 見つめる対象の連続移動 (等速 + ランダムウォーク、速度 0–2 m/s 程度)。
- 評価指標に「差し替えから到達までの時間」を追加。

## 実装メモ (2026-11-05 02:30 実装・学習開始)

- ファイル: `mdp/tracking.py` (ExternalTransitionCommand / TrackingGoalCommand / 報酬 / lowlevel_state)、
  `agents/tracking_policy.py` (TrackingActorCritic、PPO は WaypointPPO 流用)、`tracking_env_cfg.py`、
  `K1TrackingPPORunnerCfg`、タスク `Isaac-Tracking-K1-v0` / `-Play-v0`、`scripts/rsl_rl/eval_tracking.py`、
  `scripts/rsl_rl/export_tracking_policy.py`。凍結ゲートは `frozen_policies/best_gating/model_5897.pt` (gate_v4)。
- 計画からの変更: 上位観測は **34 次元** (gait_phase は入れない。`phase_obs` は位相アキュムレータを進める
  副作用があり、別グループから 2 回呼ぶと位相が 2 倍速になる)。上位出力は **1 step 遅れ**で下位に反映
  (ゲートの観測を学習時と同じ見え方に保つため。C++ も前 tick の出力を使う)。
- v1: `k1_tracking/<日時>_tracking_v1` (2 GPU × 2048 env, 3000 iter, 約 5 s/iter → 約 4 時間)、
  ログ `scripts/rsl_rl/tracking_v1_nohup.log`。
- **v1 の結果 (iter 300 で中断)**: 到達率 0.94 / 到達時間 6.7 s と学習は進んだが、pivot の使用率が 47 % → 2 %
  (iter 0 → 300) に落ちた。上位が出す Δψ を一から学習する方式では乱数の目標角で pivot しても得が無く、
  mode=turn が罰 (切替コスト + 進捗ゼロ) にしか見えないため。pivot expert は 180° を 0.33 s で回る
  (best_turn README) ので、歩行の ωz 1.0 rad/s で回る (3 s) より圧倒的に速く、使わないのは損。
- **v2 (`tracking_v2_resid`, 02:52 開始)**: pivot 目標 = **既定目標 + 上位の残差 (±1.0 rad)** に変更。既定目標は
  look では見つめる対象の方位、pose では全経由点通過後はゴール yaw、それ以前は次の目標点の方位
  (`TrackingGoalCommand.pivot_default`)。既定目標の cos/sin を上位観測に追加 (36 次元)。
  上位は「いつ pivot するか」を主に学び、Δψ は微調整だけになる。v1 は廃棄 (観測次元が変わり評価不可)。
- **v2 iter 300 の評価 (立位スタート, 2048 ep)**: 到達 99.3 % / 転倒 0.39 % / pose 5.62 s / look 5.86 s /
  見つめ違反 1.8 %。決定論方策の pivot 使用は 0.5–1.4 % で、ほぼ歩行だけで解いている。
  診断: 初期 |yaw 誤差| ≥ 120° の env で開始 0.6 s だけ pivot を強制すると、その部分集合の到達時間が
  pose 6.10 → 5.85 s、look 6.34 → 5.90 s に縮む → **pivot には価値があるが方策が見つけていない**。
- **v3 の変更 (準備済み、v2 の後に起動)**: turn 始まりのエピソード (initial_turn_prob 0.2 → 0.35) の目標角を
  乱数ではなく既定目標にする (pivot の「お手本」状態分布で、pivot 中の価値を critic に学ばせる)。
- **v2 iter 1000 の評価 (立位, 2048 ep)**: pose 到達 99.9 % / 5.49 s / 転倒 0 %、look 到達 97.9 % / 5.61 s / 転倒 0.86 % /
  違反 1.5 %。look の未到達 22 件中 13 件が経由点の手前で停止 (ゴールまで平均 3.75 m)。原因は見つめる対象が
  経由点の近くに抽選されたとき、真上を通ると方位が 180° 反転して違反が避けられないこと。
  → `look_at_clearance` 1.0 m (経路上の点から 1 m 未満は引き直し) を追加。
- **v3 (`tracking_v3_turnstart_lookfix`, 04:30 起動, 2000 iter)**: turn 始まり 35 % (目標 = 既定目標) + look_at_clearance。
  v2 は iter 1000 で頭打ち (学習時 到達 0.93–0.97 / 到達時間 6.4–6.9 s、pivot 3 %) だったので中断。
- **v3 iter 300**: pivot 使用 3.4 %、turn 到達率 0.16 (始めた pivot も 0.3 s で打ち切る)、到達 0.91 / 6.9 s。
  turn 始まりのお手本だけでは pivot を採用しない。
- **v4 の変更**: (1) `heading_progress` 報酬 (向き誤差の減少のポテンシャル, 1 rad あたり +2.0。look は見つめる
  対象への方位、pose は全経由点通過後のゴール yaw。接近中の向きには掛けない) で「早く向く」価値を密にする。
  (2) mode 次元の探索 std に下限 0.3 (`min_noise_std`)。v2/v3 では std が 0.1 まで潰れた後 mode 平均が
  walk 側に寄り、pivot を試す機会が消えた。
- v3 は iter 329 で中断 (model_300 を保持)。**v4 (`tracking_v4_headprog`, 04:58 起動, 2000 iter)**。
- **v4 iter 300**: 学習時 pivot 使用 4.7 % / turn 到達率 0.32 / 見つめ違反 0.4 %。立位評価 (2048 ep):
  pose 到達 99.0 % / 5.89 s / 転倒 0.97 %、look 到達 99.8 % / 5.47 s / 転倒 0 % / 違反 1.5 %、pivot 時間 1.3–1.5 %。
  look は v2 (97.9 % / 5.61 s) から改善 (look_at_clearance + heading_progress)。pose は iter 300 時点では v2 より遅い。
- 実機側 (booster_k1_locomotion, 未コミット): rl_policy_isaaclab_node.cpp に tracking_model_path / look_at_topic /
  tracking_step を実装 (§8 の契約どおり)、launch 引数と README 追記、スタブで構文確認済み。ONNX アセットは学習後。
- **v4 iter 1000 立位評価 (06:30)**: pose 98.8% / 5.79 s / 転倒 1.10%、look 99.6% / 5.53 s / 転倒 0.19% / 違反 1.47%、pivot 1.5〜1.9%。
  v2 model_1000 (pose 99.9% / 5.49 s / 0%) より pose が遅く転倒も増えた。look の改善は look-at 抽選修正の寄与が大きいと判断し、v4 を iter 1075 で停止。
- **v5a/v5b (06:33 開始、各 1 GPU × 4096 env × 1500 iter、約 8.2 s/iter → 10:05 頃終了)**: hydra override で
  `heading_progress.weight=0`、`initial_turn_prob=0.2` (= v2 設定) + look_at_clearance=1.0 (抽選修正)。
  v5a は mode 次元の std 下限なし (v2 と同じ)、v5b は std 下限 0.3 あり (pivot 探索の有無だけを比較)。
- **v5 iter 300 立位評価 (07:21)**: v5a pose 99.4 % / 5.47 s / 転倒 0.29 %、look 99.5 % / 5.36 s / 転倒 0 % / 違反 1.6 %
  (v2 model_1000 と v4 を両モードで上回る暫定ベスト)。v5b は look 到達 96.4 % (経由点で停滞、pivot 4.8 %) と劣る。
  → heading_progress 報酬は pose を遅くし転倒を増やしていた。look の改善は look_at_clearance の寄与。
- **v5 iter 700 立位評価 (08:18)**: v5a pose 99.7 % / 5.02 s / 転倒 0.20 %、look 99.8 % / 5.11 s / 転倒 0.10 % / 違反 1.7 % (pivot 0.4–1.0 %)。
  v5b pose 99.4 % / 5.28 s / 転倒 0.60 %、look 99.5 % / 4.96 s / 転倒 0.10 %。v5a が本命、v5b は look が僅かに速いが pose が劣る。
- **v5 最終 (10:10, 立位 2048 ep)**:
  | ckpt | pose 到達 / 時間 / 転倒 | look 到達 / 時間 / 転倒 / 違反 |
  |---|---|---|
  | v5a 1000 | 99.7 % / 5.05 s / 0.00 % | 99.3 % / 5.22 s / 0.29 % / 1.6 % |
  | **v5a 1200** | 99.8 % / 5.02 s / 0.10 % | 99.8 % / 5.14 s / 0.19 % / 1.7 % |
  | v5a 1499 | 99.9 % / 5.04 s / 0.19 % | 99.7 % / 5.27 s / 0.10 % / 1.7 % |
  | v5b 1000 | 99.8 % / 5.30 s / 0.19 % | 99.6 % / 4.97 s / 0.20 % / 1.4 % |
  | v5b 1499 | 99.7 % / 5.19 s / 0.29 % | 99.4 % / 4.92 s / 0.29 % / 1.4 % |
  **採用: v5a model_1200** (`logs/rsl_rl/k1_tracking/2026-11-05_06-33-17_tracking_v5a_v2lookfix`)。
  `frozen_policies/best_tracking/{model_1200.pt,tracking_1200.onnx}`、実機側 `assets/k1_tracking_2026-11-05_v5a_1200.onnx` (torch–onnxruntime 差 2.7e-06)。
  pivot 使用は 0.3–1.2 % (大きな向き誤差でも主に歩行で回る)。mode 次元に探索下限を付けても (v5b) pivot 採用は増えず pose がやや劣化。
- **v6 (`tracking_v6_v5a_seed1`, 10:14 開始, v5a 設定 + seed 1, 2 GPU)**: model_1200 立位評価は pose 99.6 % / 5.43 s / 0.10 %、
  look 98.7 % / 5.41 s / 0.20 % と v5a model_1200 より劣る (シード差の幅はこの程度)。採用は v5a model_1200 のまま。
- v6 model_1499: pose 99.4 % / 5.73 s、look 98.8 % / 5.43 s → 不採用。12:20 時点で学習は全て終了、GPU 空き。

## pivot 固定コスト低減後 (2026-11-06)
- w1:p5 の成果: pivot expert を model_9000 に再学習 (歩行中 1 m/s からの進入で転倒 12.9 % → 0.2 %)、ゲートは model_5897 のまま。
  融合 ONNX `best_gating/gating_5897_turn9000.onnx`。停止 → 180° → 1 m/s 復帰は 0.72 + 0.38 = 1.10 s。env 側設定は変更不要 (メモ §4)。
- **診断 (v5a model_1200 + 新 expert、立位 2048 ep)**: 通常 pose 5.01 s / look 5.02 s、|yaw 誤差| ≥ 120° の部分集合 pose 5.34 s / look 4.90 s。
  開始 0.8 s の pivot 強制で同部分集合は pose **5.70 s** / look **5.70 s** に悪化 (転倒は 0.1 % で同じ)。
  → 停止状態から回り始める最も有利な条件でも、pivot (1.1 s の無進捗) は歩行 (後退 2.0 m/s 可・曲がりながら進む) に負ける。
  pivot が勝つのは「移動がほぼ無く回転だけが必要」な状況だけで、今のタスク分布 (経由点区間 0.6–2.0 m、後退に罰則なし) ではほぼ出ない。
  設計上の論点: 後退歩行を制限する (vx の下限や後退ペナルティ) なら「向き直す必要」が生まれ pivot に価値が出る。ユーザー判断待ち。
- v7a (v5a 設定 + 新 expert) / v7b (+ mode std 下限 0.3) を 1 GPU ずつ 1500 iter で学習中 (15:00 起動)。
- **v7 iter 700 立位評価 (16:45)**: v7a pose 99.8 % / 5.05 s / 転倒 0.2 %、look 99.6 % / 5.17 s / 0 %。
  v7b pose 99.7 % / **4.93 s** / 0.2 %、look 99.9 % / **4.83 s** / 0 % / 違反 1.5 % → v5a model_1200 (5.02 / 5.14) を上回る。pivot 使用は両方 1.0–1.3 %。
- **v7 最終 (18:37, 立位 2048 ep)**:
  | ckpt | pose 到達 / 時間 / 転倒 | look 到達 / 時間 / 転倒 / 違反 | pivot |
  |---|---|---|---|
  | v7a 1000 | 99.8 % / 4.90 s / 0.20 % | 99.9 % / 4.99 s / 0.00 % / 1.4 % | 0.9–1.2 % |
  | v7a 1200 | 99.9 % / 4.96 s / 0.00 % | 99.5 % / 5.05 s / 0.00 % / 1.4 % | 1.1–1.4 % |
  | v7a 1499 | 99.8 % / 5.06 s / 0.20 % | 99.9 % / 5.03 s / 0.19 % / 1.4 % | 0.9–1.1 % |
  | v7b 1000 | 99.9 % / 5.03 s / 0.10 % | 99.8 % / 4.95 s / 0.10 % / 1.5 % | 1.0–1.1 % |
  | **v7b 1200** | **100 % / 4.84 s / 0.00 %** | **99.9 % / 4.91 s / 0.10 % / 1.6 %** | 0.7–0.9 % |
  | v7b 1499 | 99.9 % / 4.96 s / 0.10 % | 100 % / 4.94 s / 0.00 % / 1.5 % | 0.6–0.9 % |
  **採用: v7b model_1200** (`logs/rsl_rl/k1_tracking/2026-11-06_15-03-36_tracking_v7b_turn9000_modestd`、新 pivot expert model_9000 + mode std 下限 0.3)。
  v5a model_1200 (5.02 / 5.14) から pose −0.18 s、look −0.23 s。pivot 使用は 1 % 未満で、改善は新 expert のもとで探索の失敗コストが消えたことによる。
  `frozen_policies/best_tracking/{model_v7b_1200.pt,tracking_v7b_1200.onnx}`、実機 `assets/k1_tracking_2026-11-06_v7b_1200.onnx`。
- v7b model_1200 の学習時リセット (関節 ±30°, 初速 ±0.5) 評価は下記 (v5a model_1200 は到達 95.8 % / 転倒 4.2 %)。
    episodes 2053: reached 96.35%  falls 3.56%

## 自己位置推定の遅延 (2026-11-06 夜)
- 実機では目標との相対位置を自己位置推定から計算し、推定が真値から 100〜250 ms 遅れる → 相対位置の観測に 100〜300 ms の遅延 (ユーザー報告)。
- `TrackingGoalCommand` に遅延を実装: 真の (x, y, yaw) をリングバッファに積み、エピソードごとに一様乱数で選んだ `obs_delay_s` (既定 0.1〜0.3 s) 前の値を
  「推定姿勢」として観測・経由点通過判定・pivot 既定目標に使う。報酬・到達判定・metrics は真値のまま。C++ 側は変更なし (実機の遅延がそのまま相当)。
  `env.commands.tracking.obs_delay_s=[0,0]` で無効化できる。
- 遅延なしで学習した v7b model_1200 を遅延 0.1〜0.3 s で評価: pose 4.84 → **6.09 s**、look 4.91 → **5.65 s** (到達 99.9 / 99.6 %、転倒 0.1 / 0 %)。
  到達率は保つが、ゴール付近で行き過ぎて戻る分だけ遅くなる。
- **v8 (`tracking_v8_delay`)**: v7b 設定 (新 expert + mode std 下限 0.3) + 遅延。2 GPU 分散 2×2048 env × 1500 iter (約 2 時間)。
- **v8 結果 (21:05, 立位 2048 ep, 遅延あり)**: model_1000/1200/1499 とも到達 **96.4–96.7 %** / pose 6.1 s / look 5.9 s、転倒 0–0.3 %、pivot 4–6 %。
  未到達は経由点で停滞 (look が多い)。遅延なしで評価しても 97.3 % (look 96.3 %) で、遅延なし学習の v7b (遅延評価 99.9 / 99.6 %) に劣る。
  学習時指標も v7b より悪い (reached 0.90 vs 0.97、time 7.3 vs 5.5 s)。遅延そのものか、2 GPU 分散 (v6 も 1 GPU の v5a より悪かった) か、
  mode std 下限 × 遅延 (pivot 増) かを切り分けるため、v9a (遅延, std 下限なし, 1 GPU) / v9b (遅延, std 下限あり, 1 GPU) を並走 (21:10 開始)。
- **v9 iter 700 立位評価 (22:50, 遅延あり)**: v9a pose 99.7 % / 5.87 s、look 99.7 % / 5.52 s。v9b pose 99.6 % / 5.71 s、look 99.7 % / 5.68 s。
  どちらも v8 (96.5 %) を大きく上回り、v7b の遅延評価 (6.09 / 5.65 s) より pose が速い → **v8 の劣化は 2 GPU 分散学習が原因**
  (v6 でも同じ傾向)。遅延そのもの・std 下限は原因ではない。以後 tracking の学習は 1 GPU × 4096 env で行う。
- **v9 最終 (00:42, 立位 2048 ep, 遅延 0.1–0.3 s)**:
  | ckpt | pose 到達 / 時間 / 転倒 | look 到達 / 時間 / 転倒 / 違反 |
  |---|---|---|
  | v9a 1000 | 99.7 % / 5.82 s / 0.10 % | 99.8 % / 5.61 s / 0.20 % / 1.5 % |
  | v9a 1200 | 99.8 % / 5.84 s / 0.19 % | 99.6 % / 5.57 s / 0.00 % / 1.6 % |
  | v9a 1499 | 99.8 % / 5.84 s / 0.10 % | 99.3 % / 5.51 s / 0.00 % / 1.5 % |
  | v9b 1000 | 99.7 % / 5.54 s / 0.00 % | 99.5 % / 5.70 s / 0.20 % / 1.4 % |
  | v9b 1200 | 99.8 % / 5.64 s / 0.10 % | 99.7 % / 5.57 s / 0.10 % / 1.4 % |
  | **v9b 1499** | **100 % / 5.66 s / 0.00 %** | **99.6 % / 5.69 s / 0.00 %** / 1.5 % |
  **採用: v9b model_1499** (`tracking_v9b_delay_stdfloor_1gpu`、遅延 0.1–0.3 s で学習、新 expert、mode std 下限 0.3、1 GPU)。
  遅延評価で v7b model_1200 (6.09 / 5.65 s) より pose −0.43 s、転倒 0。差は誤差範囲のものが多く、転倒 0 と到達率で選んだ。
- v9b model_1499 の確認: 遅延なし評価 pose 99.6 % / 5.38 s、look 99.6 % / 5.58 s (遅延用に慎重になった分、遅延なしの v7b 4.84 / 4.91 より遅い)。
  学習時リセット評価 (遅延あり) 到達 96.1 % / 転倒 3.3 % (v7b 96.4 % / 3.6 %)。実機には遅延があるので遅延評価の値で比較する。

## 遅延の扱いの結論 (2026-11-07)
- 上流は相対位置の計算に使った自己位置推定の時刻を `header.stamp` に入れられる (ユーザー確認)。オドメトリ (booster_k1 内の inekf) は遅延 ~30 ms。
- → C++ ノードで経路・見つめる対象を「stamp 時刻のオドメトリ」を基準に odom 系へ固定し (`odom_anchor`)、以降は最新オドメトリとの差で追従する。
  推定の遅延中に動いた分はこれで戻るので、**遅延なし方策 v7b model_1200 (新 pivot expert 上で学習) に戻す**。
- v9b (遅延込み学習) は stamp が使えない場合の代替として best_tracking に残す (model_v9b_1499.pt / tracking_v9b_1499.onnx)。
  遅延を観測に入れて方策に融合させる案は、決定論的な補正と同じ結果にしかならないので採らない。

## 2026-11-16: 新歩行 (best_walk 56444, y ±1.9) + 新ゲート (gate_v7 2999) 向けの準備
- 採用設定 (v7b) を既定値に焼き込み: heading_progress 報酬を削除、initial_turn_prob 0.2、obs_delay_s (0, 0) (遅延は実機側で補正)。
- 上位方策のクランプを新歩行の指令領域に合わせた: (2.0, 1.9, 1.0) + |vy| > 1.0 で |vx| 上限を 0.3 (|vy|=1.9) まで線形に絞る台形
  (`lateral_x_knee` / `lateral_x_min`、mdp/commands.py のサンプリングと同じ)。ONNX 書き出しにも同じ処理を入れた (旧 checkpoint は従来どおり)。
- 新 expert / ゲートで 64 env × 3 iter の起動確認済み。別 PC 向けの手順は `scripts/rsl_rl/TRACKING_HANDOFF.md`。

## 2026-11-18〜19: v10 の停滞診断と v11〜v13 (報酬調整ループ、新歩行 56444 + gate_v7 2999 上)
- **v10 model_1499 の未到達の診断** (`eval_tracking.py --dump_miss`、立位 8192 ep): look の時間切れ約 20 件のうち 13 件は
  「開始 0.3 s で pivot に入り、完了 (残り角 ≈ 0) 後も mode=turn を出し続けて 20 s 動かない」。残りは経由点の 0.3〜0.5 m 手前
  (通過半径のすぐ外) で微小指令のまま停滞。見つめる対象と経路の距離は無関係 (全件 1.1 m 以上)。
- **turn 居座りガード** (`agent.policy.turn_exit_rad`、既定 0 = 無効): turn 中に |残り角| < 値 なら mode=walk を強制。
  `TrackingActorCritic.env_actions` と `export_tracking_policy.py --turn_exit_rad` の両方に実装 (nav_obs[24], [26] だけで決まるので C++ 変更不要)。
  v10 に評価時だけ 0.15 を掛けると look の時間切れ 20 → 4 件 (到達 99.17 → 99.58 %)。
- **停滞ペナルティ** (`rewards.turn_overstay` / `rewards.walk_stall`、既定 weight 0 = 無効): turn に 1 s 超で居座る / walk で
  ゴールから 0.5 m 以上離れて速度 0.3 m/s 未満、の step に 1。
- 学習はすべて 1 GPU × 4096 env × 1500 iter。評価は立位・8192 ep × 5 seed (123, 7, 11, 21, 33) の平均、model_1499:
  | run | 変更点 | pose 到達 / 時間 / 転倒 | look 到達 / 時間 / 転倒 | 時間切れ (pose / look, 件/8192ep) |
  |---|---|---|---|---|
  | v10 (採用中) | 既定 | 99.61 % / 4.55 s / 0.33 % | 99.20 % / 4.52 s / 0.30 % | 5.2 / 22.0 |
  | v11a | 既定、seed 1 | 99.71 % / 4.68 s / 0.31 % | 99.74 % / 4.38 s / 0.29 % | 0 / 1.2 |
  | v11b | 停滞ペナルティ 各 −2 (seed 123 のみ) | 99.69 % / 4.53 s / 0.39 % | 99.21 % / 4.56 s / 0.84 % | 0 / 3 |
  | v12a | turn_exit_rad 0.15 で学習・評価 | 99.51 % / 4.53 s / 0.54 % | 99.69 % / 4.17 s / 0.34 % | 0.4 / 0.4 |
  | v12b | termination −400 | 99.53 % / 4.51 s / 0.52 % | 99.56 % / 4.17 s / 0.45 % | 0 / 1.4 |
  | v12b + ガード (評価時のみ) | 同上 | 99.57 % / 4.50 s / 0.48 % | 99.53 % / 4.15 s / 0.50 % | 0 / 1.0 |
  | v13a | 停滞ペナルティ + termination −400 (seed 123 のみ) | 99.22 % / 4.65 s / 0.51 % | 99.11 % / 4.36 s / 0.37 % | 13 / 24 |
  学習時リセット評価 (seed 7、到達 pose / look): v10 96.8 / 96.7 %、v11a 96.7 / 96.9 %、v12a 95.7 / 96.4 %、v12b+ガード 96.1 / 96.7 %。
- わかったこと:
  - v10 の turn 居座りは**シード依存の癖** (同設定の v11a ではほぼ出ない)。同一 checkpoint・同一 seed でも評価は到達 ±0.25 %pt、
    時間 ±0.08 s 揺れるので、採否は複数 seed の平均で見る。
  - 停滞ペナルティは効果なし (v11b は look の転倒増、v13a は逆に停滞増)。termination −400 でも転倒は減らない (0.3〜0.5 % は下位由来)。
  - v12a / v12b は look が 4.17 s (v10 比 −0.35 s) と速いが、pose の転倒が 0.5 % (v10 / v11a は 0.3 %) とやや多い。v11a は転倒・到達が最良で pose が 0.13 s 遅い。
  - 評価を学習と同じ GPU で並走させると両方が約 2 倍遅くなる (5 分 → 10 分)。
- **採用 (2026-11-19 ユーザー決定): v11a model_1499** (`k1_tracking/2026-11-18_23-02-08_tracking_v11a_seed1`、既定設定・seed 1、ガードなし)。
  `frozen_policies/best_tracking/{model_v11a_1499.pt,tracking_v11a_1499.onnx}` (torch–onnxruntime 差 1.6e-06)。実機リポジトリへの配備は未実施。
  速さ優先の代替は `k1_tracking/2026-11-19_03-31-41_tracking_v12b_term400/model_1499.pt` (書き出し時に `--turn_exit_rad 0.15`)。評価ログは `scripts/rsl_rl/eval_tracking_v1*_s*.log`。
  安全優先 = `2026-11-18_23-02-08_tracking_v11a_seed1/model_1499.pt`。評価ログは `scripts/rsl_rl/eval_tracking_v1*_s*.log`。

## 2026-11-21: v14 — 新歩行 R9 (best_walk 64440, 摩擦 DR U(0.3, 3.0) per_env) + pivot 10000 + gate_v8 2999 上で再学習
- 別 PC で歩行 expert を摩擦 DR 拡張版 R9 (`model_64440`, 摩擦 U(0.3, 3.0) を床ごとに 1 つ) に差し替え、pivot (`best_turn/model_10000`) と
  ゲート (`best_gating/model_2999`, gate_v8, `gating_2999_walk64440_turn10000.onnx`) も同じ摩擦分布で再学習済み (commit 963f2e0)。
  ユーザー指示: 「同じ摩擦 DR の範囲の環境で経路追従を再学習」。
- tracking 環境の摩擦: 親 `K1FlatGatingCfg` が `randomize_floor_friction` per_env U(0.3, 3.0) になった (896cec3) のに、
  `K1TrackingEnvCfg.__post_init__` が 11-19 の上書き (0.2, 3.0) (be3a61c) を残していたので撤去し、親をそのまま継承させた。
  起動ログ `[randomize_floor_friction] per_env=True static μ quantiles: [0.38, 0.98, 1.76, 2.36, 3.0]`。他の設定は v11a と同じ既定値。
- 学習: 1 GPU × 4096 env × 1500 iter を 2 seed 並走 (v14a seed 1 / GPU0、v14b seed 7 / GPU1、約 8 s/iter、3.4 h)。
  `k1_tracking/2026-11-21_03-23-49_tracking_v14a_walk64440_seed1`、`2026-11-21_03-23-57_tracking_v14b_walk64440_seed7`。発散なし。
- 評価 (立位、2048 env × 4000 step ≈ 8192 ep / seed、摩擦 U(0.3, 3.0) 環境、下位は新スタック)。
  v14a/v14b model_1499 は 5 seed (123, 7, 11, 21, 33)、他は 2 seed (123, 7) の平均:
  | checkpoint | pose 到達 / 時間 / 転倒 | look 到達 / 時間 / 転倒 | 時間切れ (件 / 2 seed) |
  |---|---|---|---|
  | v11a model_1499 (旧上位方策を新スタックに載せたもの) | 99.58 % / 4.72 s / 0.58 % | 99.66 % / 4.44 s / 0.40 % | 2 |
  | **v14a model_1000 (採用)** | 99.54 % / 4.55 s / 0.48 % | 99.69 % / 4.38 s / 0.45 % | 13 |
  | v14a model_1200 | 99.49 % / 4.56 s / 0.62 % | 99.66 % / 4.36 s / 0.52 % | 5 |
  | v14a model_1499 (5 seed) | 99.50 % / 4.55 s / 0.63 % | 99.68 % / 4.35 s / 0.42 % | 18 / 5 seed |
  | v14b model_1499 (5 seed) | 99.21 % / 4.54 s / 0.78 % | 99.55 % / 4.17 s / 0.58 % | 23 / 5 seed |
  (v14a 1499 の 2 seed だけなら 99.42 / 4.54 / 0.73 %、99.63 / 4.33 / 0.42 %)
- わかったこと:
  - 転倒 0.4〜0.6 % は新しい評価環境 (μ 0.3 の床を含む) の床値。旧上位 v11a も同環境では pose 0.58 % 転ぶ (旧環境では 0.31 %)。
    v14 固有の悪化ではない。μ 0.3 では歩行 expert 単体でも前進 2.0 で転倒が残る (`best_walk/README.md`)。
  - 再学習の効果は到達時間: pose −0.17 s (4.72 → 4.55 s)、look −0.06〜−0.1 s。到達率・転倒は v11a と同等。
    v11a をそのまま新スタックに載せても大きくは壊れない (到達 99.6 %)。
  - v14a の checkpoint 間の差 (到達 ±0.1 %pt、転倒 ±0.15 %pt) は seed ノイズ (±0.25 %pt) の範囲。model_1000 は転倒と到達が最良で
    速度は 1499 と同じなので採用。経由点手前での時間切れが 2 seed で 13 件とやや多い (1499 は 5 seed で 18 件) が到達率に含めて評価済み。
  - v14b (seed 7) は look が 0.2 s 速いが pose の到達 99.2 %・転倒 0.8〜0.9 % で劣る。v11a 時代と同じく seed 依存が大きい。
  - 切替回数 pose 0.74 / ep (v11a 0.45)、pivot 時間 1.0〜1.5 % (v11a 0.6〜1.2 %)。新 pivot expert で少し使うようになったが依然 1 % 台。
- **採用: v14a model_1000** → `frozen_policies/best_tracking/{model_v14a_1000.pt, tracking_v14a_1000.onnx}` (torch–onnxruntime 差 3.4e-06)。
  実機リポジトリへの配備は未実施。実機側は融合 ONNX も `gating_2999_walk64440_turn10000.onnx` に差し替える必要がある (位相の式は 56444 と同じで C++ 変更なし)。
  評価ログ `scripts/rsl_rl/eval_tracking_v14*_s*.log`, `eval_tracking_v11a_1499_newenv_s*.log`。
