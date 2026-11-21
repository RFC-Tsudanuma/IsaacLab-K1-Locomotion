# 経路追従 + 姿勢到達 / 見ながら到達 (Isaac-Tracking-K1) 引き継ぎ (2026-11-16)

別 PC で tracking 上位方策を学習し直すエージェント向け。これ 1 ファイルで開始できる。
設計と経緯の全記録は `rl_traking.md` (リポジトリ直下、ファイル名は typo のまま)、
実機側の契約は booster_k1_locomotion の README「経路追従 + 姿勢到達 / 見ながら到達」節。

## 今回の目的

(初版 2026-11-16) 歩行 expert が `best_walk/model_56444.pt` (横 ±1.9 m/s) に、融合ゲートが
`best_gating/model_2999.pt` (gate_v7) に更新された。この上で tracking 上位方策を再学習し、
現行 `best_tracking/tracking_v7b_1200.onnx` (旧歩行 52700 + ゲート 5897 上で学習) を置き換える。
→ v11a で実施。その後 2026-11-21 に歩行 R9 (`model_64440`, 摩擦 DR 拡張) + pivot 10000 + gate_v8 で v14a を再学習 (末尾の追記)。
下位 expert が更新されたら、このファイルの手順どおり再学習 → 2 seed 以上で評価 → best_tracking 更新、を繰り返す。

## 構成 (コードはすべてコミット済み、既定値 = 採用設定)

| | |
|---|---|
| タスク | `Isaac-Tracking-K1-v0` (`tracking_env_cfg.py: K1TrackingEnvCfg`、ゲート学習環境 `K1FlatGatingCfg` を継承) |
| 上位方策 | `agents/tracking_policy.py: TrackingActorCritic` (MLP 256-128、出力 5 = vx, vy, ωz, Δψ 残差, mode) |
| 凍結する下位 | `frozen_policies/best_{walk,turn,gating}/model_*.pt` の**番号最大**を自動で拾う (起動ログの `expert 'walk' <-` 行で確認) |
| コマンド領域 | クランプ (2.0, 1.9, 1.0)、`|vy| > 1.0` では `|vx|` 上限を 2.0 → 0.3 (|vy| = 1.9) に線形に絞る (歩行 expert の学習分布。`RslRlTrackingActorCriticCfg.lateral_x_knee / lateral_x_min`) |
| 報酬 | path_progress 50 / 位置 exp 2 / 向き exp 2 / look_at −1 / コマンド変化 −0.05 / mode 切替 −0.5 / 転倒 −200。heading_progress は不採用 (外してある) |
| 探索 | entropy 0、mode 次元だけ std 下限 0.3 (`min_noise_std`)。旧 pivot expert では逆効果だったが、歩行中進入で転倒しない expert (model_9000 以降) では有効 |
| 遅延 | `TrackingGoalCommandCfg.obs_delay_s = (0, 0)`。実機側が header.stamp 時刻のオドメトリで補正するため。補正できない構成なら (0.1, 0.3) |
| 環境リセット | 学習は関節 ±30°・初速 ±0.5 の厳しい初期化のまま (評価は `--standing`)。立位スタート学習は頑健性を落とした (waypoint v3) |

## 学習の回し方

```bash
cd scripts/rsl_rl
nohup env CUDA_VISIBLE_DEVICES=0 uv run train.py --task Isaac-Tracking-K1-v0 --headless \
    --num_envs 4096 --max_iterations 1500 --run_name tracking_v10_walk56444 > tracking_v10_nohup.log 2>&1 &
```

- **1 GPU × 4096 env で回す (約 8.2 s/iter、1500 iter ≈ 3.5 時間)。2 GPU 分散 (2×2048) は使わない**:
  同設定で 2 回 (v6, v8) 明確に悪い方策になった (到達 99.7 % → 96.5 %)。原因未特定。
- 変種を並走させるときは hydra override で (例: `agent.policy.min_noise_std=[0,0,0,0,0]`、
  `env.commands.tracking.obs_delay_s=[0.1,0.3]`、`env.rewards.look_at.weight=-2`)。2 本目は 5 秒以上ずらして起動
  (同じ秒に起動すると `/tmp/IsaacLab/usd_<timestamp>` が衝突する)。
- 止めるときは `ps -eo pid,args | grep "train.py --task Isaac-Tracking"` の python を PID で kill (`pkill -f` は自分の shell も殺す)。
- resume は `--resume --load_run <run> --checkpoint model_N.pt --reset_noise_std 0.05` (省略すると std が負に落ちて落ちる)。
- 学習ログで見る指標: `Metrics/tracking/reached_goal` (厳しい初期化込みで 0.95 前後が上限)、`time_to_goal`、
  `Metrics/base_velocity/turn_frac` (pivot 使用率、これまで 1〜5 %)、`Episode_Termination/base_height` (転倒、0.05 前後)。
  学習時指標はノイズが大きいので採用判断は下の評価で行う。

## 評価と採用判断

```bash
# 立位スタート (実機相当の初期化)、2048 エピソード、約 5 分
CUDA_VISIBLE_DEVICES=1 uv run eval_tracking.py --headless --num_envs 1024 --steps 2000 --standing \
    --checkpoint /abs/path/logs/rsl_rl/k1_tracking/<run>/model_1200.pt
# 遅延ありで見るとき: "env.commands.tracking.obs_delay_s=[0.1,0.3]" を末尾に付ける
# pivot 強制の診断: --force_pivot_s 0.8 (初期向き誤差 120° 以上の env で開始 0.8 s を turn にする)
```

出力は pose / look 別に 到達率・到達時間・転倒・切替回数・pivot 時間・見つめ違反時間、
さらに「全経由点通過後にゴールから 0.1 / 0.3 / 0.5 m 以内に初めて入る時刻」(粗い到達の速さ)。

採用基準 (現行 v7b model_1200、旧歩行上、遅延なし):

| | 到達 | 到達時間 | 転倒 |
|---|---|---|---|
| pose | 100 % | 4.84 s | 0.0 % |
| look | 99.9 % | 4.91 s | 0.1 % |

- 到達 ≥ 99.5 %、転倒 ≤ 0.2 % を満たす中で到達時間が短い checkpoint を選ぶ (1000 / 1200 / 1499 を比べる。
  差の多くは誤差範囲なので、転倒 0 と到達率で選んで良い)。
- 新歩行は横移動が速いので look (見つめながら横移動・後退) が速くなるはず。pose は同等以上が目標。
- 学習時リセット (`--standing` なし) の転倒 3〜4 % はリセット直後 2 秒以内のもので、立位では出ない (既知)。
- 歩行 expert 自体が実機未検証なので、横 1.9 m/s 付近の指令が多く出るようなら実機で先に歩行単体を確認してもらう。

## 書き出しと配備

```bash
cd scripts/rsl_rl
CUDA_VISIBLE_DEVICES="" uv run python export_tracking_policy.py \
    --checkpoint /abs/path/logs/rsl_rl/k1_tracking/<run>/model_1200.pt \
    --out ../../frozen_policies/best_tracking/tracking_v10_1200.onnx     # torch–onnxruntime の差が 1e-5 未満で合格
cp /abs/path/logs/rsl_rl/k1_tracking/<run>/model_1200.pt ../../frozen_policies/best_tracking/model_v10_1200.pt
```

- ONNX は上位 actor のみ (観測正規化 + MLP + クランプ/台形/デッドゾーン/mode 閾値)。入力 `nav_obs (1,36)`、出力 `command (1,5)`。
  観測レイアウトは `export_tracking_policy.py` の docstring。**観測を変えたら C++ も変える** (36 次元の並び)。
- 実機リポジトリ booster_k1_locomotion (branch `feat/goalkeeper_block`): ONNX を `assets/k1_tracking_<日付>_v10_<iter>.onnx` として追加し、
  README の例を更新。融合モデル (歩行 + pivot + ゲート) の ONNX も新版 `best_gating/gating_2999_walk56444_turn9000.onnx` を
  `assets/k1_gating_walk_turn_*.onnx` として追加し launch 既定を切り替える。**新歩行は位相周波数に横加算があり、C++ の
  `cmd_phase_freq` を合わせないと位相がずれる** (`best_walk/README.md`)。これはゲート側の作業として依頼済みか確認すること。
- 旧版は `frozen_policies/best_tracking/` に併置して良い (v9b: 遅延込み学習の代替)。
- **2026-11-21 追記: 採用は v14a model_1000** (`best_tracking/tracking_v14a_1000.onnx`、歩行 R9 64440 (摩擦 U(0.3, 3.0) per_env) +
  pivot 10000 + gate_v8 2999 上、既定設定・seed 1)。tracking 環境の摩擦 DR は親 `K1FlatGatingCfg` の U(0.3, 3.0) per_env を継承
  (歩行 expert と同じ分布。tracking 側で上書きしないこと)。この環境での評価は μ 0.3 の床を含むため転倒の床値が 0.4〜0.6 % になる
  (旧環境の基準 0.2 % とは比べられない。同環境で旧 v11a も 0.58 %)。経緯は `rl_traking.md` 末尾。
- 2026-11-19 追記: 採用は v11a model_1499 (`best_tracking/tracking_v11a_1499.onnx`、新歩行 56444 + gate_v7 2999 上、既定設定・seed 1)。
  経緯と比較は `rl_traking.md` 末尾の節。採否は複数 seed の評価平均で決める (同一 seed でも到達 ±0.25 %pt 揺れる)。

## 既知の結論 (やり直さなくて良いこと)

- pivot の採用率は報酬を変えない限り 1 % 前後。後退 (vx −2.0) と曲がりながらの前進が、pivot (回転 0.72 + 再加速 0.38 s) に勝つ。
  heading_progress 報酬・turn 始まりのデモ・残差目標は効かなかった。pivot を使わせたいなら後退を制限するなどタスク側の変更が要る (未着手)。
- look-at 対象は経路上の点から 1 m 以上離して抽選する (`look_at_clearance`)。これが無いと真上を通ったとき方位が反転し停滞する。
- ωz のクランプを 2.0 にしても到達時間は縮まらない (waypoint v3)。
- 検討中で未着手: 「粗い / 正確」の精度モード入力 (ブロック用は 0.3 m / 0.3 rad を最速で、セットプレー用は正確に)。
  動く目標・見つめる対象のカリキュラム。
