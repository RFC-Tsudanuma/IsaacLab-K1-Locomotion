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
