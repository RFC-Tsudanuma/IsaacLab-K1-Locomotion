# best_gating — 歩行 ⇄ その場回転 ゲーティング融合モデル

`Isaac-Velocity-Flat-Gating` (`K1FlatGatingCfg` + `GatingPPO`) の凍結成果物。
凍結 expert 2 本 (`best_walk/model_52500.pt`, `best_turn/model_7999.pt`) とゲート MLP を
1 つの ONNX に融合してある (`scripts/rsl_rl/export_gating_policy.py`)。

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
