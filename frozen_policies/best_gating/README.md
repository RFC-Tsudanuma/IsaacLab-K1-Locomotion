# best_gating — 歩行 ⇄ その場回転 ゲーティング融合モデル

`Isaac-Velocity-Flat-Gating` (`K1FlatGatingCfg` + `GatingPPO`) の凍結成果物。
凍結 expert 2 本 (現行は `best_walk/model_56444.pt`, `best_turn/model_9000.pt`) とゲート MLP を
1 つの ONNX に融合してある (`scripts/rsl_rl/export_gating_policy.py`)。

## 現行 (2026-11-16 採用、実機未検証)

歩行 expert を横移動拡張版 (`best_walk/model_56444.pt`、y ±1.9 m/s) に差し替えたのに合わせ、ゲートをスクラッチで再学習した。

| | |
|---|---|
| 融合 ONNX | `gating_2999_walk56444_turn9000.onnx` |
| ゲート重み | `model_2999.pt` (`2026-11-16_04-54-41_gate_v7_walk56444`、2 GPU × 2048 env、3000 it、歩行コマンド y ±1.9) |
| 凍結 expert | `best_walk/model_56444.pt` + `best_turn/model_9000.pt` (回転 expert は据え置き) |
| 検証 | torch 等価性 `max|Δaction| = 3.8e-6` (64 乱数入力) / mode=0 vs `best_walk/policy.onnx` 差 0 (小振幅乱数入力、w=0)。mode=1 の expert 単体パリティは乱数入力では w が 1 に寄らず未実施 |
| 旧ゲート | `old/model_5897.pt` (番号最大の自動解決で拾われないよう退避)、`gating_5897_turn9000.onnx` は旧歩行 52700 焼き込みのまま |

**C++ 側の注意**: 歩容位相の周波数に横加算が増えた。
`freq = 1.8 + max(||cmd_xy|| - 1.0, 0) * 0.25 + max(|vy| - 0.5, 0) * 0.75` (`best_walk/README.md` 参照)。
`cmd_phase_freq` を合わせないと横移動で位相がずれる。

pivot 計測 (`eval_pivot_overhead.py`、新歩行 56444 + 回転 9000、合計時間の中央値 / turn 中の転倒率):

| 条件 | 旧ゲート 5897 | 新ゲート 2999 |
|---|---|---|
| 進入 0 m/s、180° (1024 env) | 1.16 s / 0 % | 1.12 s / 0 % |
| 進入 1.0 m/s、120° (1024 env) | 0.92 s / 0.1 % | 0.98 s / 0 % |
| 進入 1.5 m/s、120° (1024 env) | 1.00 s / 0.1 % | 1.04 s / 0 % |
| 進入 2.0 m/s、120° (4096 env) | 1.16 s / 0.73 % | 1.20 s / 0.34 % |

新ゲートは高速進入の転倒が半分、所要時間は進入ありで 0.04〜0.06 s 長い。途中の model_2000 は時間が旧と同等 (2.0 m/s で 1.14 s) だが転倒 0.61 % で旧と差が無い。
横移動からの進入 (`--entry-vy`、120°、2048 env、旧 / 新): 横 1.0 m/s で 1.04 s / 0.15 % → 1.06 s / 0.05 %、
横 1.9 m/s で 0.86 s / 2.93 % → 0.96 s / 2.05 %。横 1.9 からの進入は新旧とも 2 % 台で転ぶ (回転 expert は横進入を学習していない)。

## 旧現行 (2026-11-06 採用)

| | |
|---|---|
| 融合 ONNX | `gating_5897_turn9000.onnx` |
| ゲート重み | `model_5897.pt` (`2026-10-31_..._gate_v4` 由来。2026-11-05/06 の追加学習 v5/v6 は高速進入で退行したため**不採用**) |
| 凍結 expert | `best_walk/model_52700.pt` + `best_turn/model_9000.pt` (**回転 expert を歩行中進入向けに再学習**) |
| 検証 | torch 等価性 `max|Δaction| < 1e-4` / mode=0 vs `best_walk/policy.onnx` `≤ 2.2e-3` / mode=1 vs `best_turn/policy.onnx` `≤ 3.8e-4` |
| 計測・判断の詳細 | [`PIVOT_FIXED_COST_MEMO.md`](PIVOT_FIXED_COST_MEMO.md) |

1.0 m/s で歩きながら pivot に入って転倒 0.20 % (1024 env)。旧 expert は 12.9 % (ゲート経由) /
29.7 % (expert 単体) で落ちていた。停止 → 180° pivot → 1.0 m/s 復帰は合計 1.12 s・転倒 0 %。

以下の「来歴」以降は gate_v1 (model_2500) 当時の記述。インタフェース・学習時の要点は現行でも有効。

## 来歴

| | |
|---|---|
| 学習 run | `scripts/rsl_rl/logs/rsl_rl/k1_gating/2026-09-23_01-11-33_gate_v1` (2 GPU × 2048 env, 3000 it) |
| checkpoint | `model_2500.pt` (平均報酬 72.0・転倒終了 0.089・turn 成功率 0.853 で系列ベスト。2999 は 63.8 / 0.090 / 0.785) |
| エクスポート | `gating_2500.onnx` (torch との等価性: 64 乱数入力で max |Δaction| = 3.3e-5) |
| 学習ログ | `scripts/rsl_rl/gating_v1_nohup.log` |

## インタフェース (C++: booster_k1_locomotion rl_policy_isaaclab_node, branch feat/gating_walk_turn)

```
入力  command      [1, 3]        現モードの意味: walk (vx, vy, ωz) / turn (0, 0, Δψ/π)
      obs_history  [1, 100, 49]  歩行ポリシーと同一レイアウト。コマンド枠 [6:9] は現モードの値
      mode         [1, 1]        0.0 = walk, 1.0 = turn
出力  actions      [1, 12]       混合後の生アクション (スケール 0.5 は C++ 側)
      turn_weight  [1, 1]        ゲート重み w (回転側)。ログ用
```

契約:
* **モード切替の瞬間に obs_history 全フレームのコマンド枠 [6:9] を 0 にする**
  (学習環境 `TransitionCommand.zero_command_history_on_switch`)。履歴のタイル埋めはしない。
* 非アクティブ expert へのコマンド枠マスクはモデル内部で行う (C++ は 1 本の履歴を持てばよい)。
* 歩容位相は歩行と同じマッピング (1.8 Hz @ ≤1.0 m/s → 2.0 Hz @ 1.8 m/s)。turn 中は cmd_xy=0 なので 1.8 Hz。
  C++ は `phase_freq_low_hz:=1.8` で起動すること (既定 1.5 は 2026-07 世代用)。
* 位相ゼロ化: 学習は ||cmd|| < 0.05 (turn では |Δψ| < 9°)、C++ は 0.1 + 1 周期の graceful stop (従来どおり)。

## 学習時の要点

* 両 expert 凍結。ゲート = 小 MLP (入力: モード one-hot / コマンド / 直近 4 フレーム /
  両 expert の提案アクション) → logit、`w = sigmoid(logit + ±4 のハード切替事前分布)`。
* 内発報酬: 平滑化 -5(Δw)^2、モード整合 -2|w - target| (切替後 0.5 s 猶予)。転倒罰 -400。
* 学習終了時 `Gate/turn_weight_in_turn` ≈ 0.74、`in_walk` ≈ 0.02 (turn 中も ~1/4 歩行 expert を混ぜる)。
