# bhuman_walk_kick_s6c_9200 — B-Human 版インサイドキック (歩行 model_34995 の上に学習)、仕様 10 項目すべて合格

学習 run: `logs/rsl_rl/k1_bhuman_walk_kick/2026-10-27_16-10-56_s6c_vel8_strong4` (Stage 6 の 3 つ目)、checkpoint `model_9200.pt`。
仕様と合格条件は `docs/bhuman_walk_kick_spec.md`。2026-10-27 時点の一番良い checkpoint
(前の最良は `bhuman_walk_kick_s5_7500`。判定の条件ではどちらも 10/10 で、こちらは seed・PD の遅れを変えても通る場合が多い)。

学習の段: Stage 5 (s5 model_7699) → Stage 6 (README 5)、kick_direction_sigma_range 0.25、500 iteration) → 0.2 で 300 iteration →
0.2 のまま B-Human README の「ball_kick_velocity 8・ball_kick_velocity_strong 4」で 700 iteration (この checkpoint)。
B-Human から変えた点と理由は進捗ログ (`claude_tmp/bhuman_walk_kick_progress.md`) と `bhuman_walk_kick_env_cfg.py` のコメント。

## インタフェース (ONNX `policy.onnx`)

```
入力  command      [1, 3]        (vx, vy, wz)  ボールと蹴る方向から決める速度指令 (mdp/walk_command.py の式)
      obs_history  [1, 100, 61]  歩行の 49 項 (best_walk と同じ並び) + キック用 12 項 (locomotion/history_layout.py KICK_EXTRA_TERM_SPECS)
出力  actions      [1, 13]       脚 12 (scale 0.5) + 歩行周波数のずらし (±0.5 Hz に clip して基本周波数に足す)
```

- `policy.onnx` は `scripts/mujoco/sim2sim_bhuman_walk_kick.py` の書き出し (履歴入力の actor + 観測の正規化) と同じもの。
  学習時の actor (HistoryEncoderHead + EmpiricalNormalization) との出力の差は最大 2.9e-6 (ランダム入力 16 個)。
- ボールは半径 0.095 m・0.29 kg。**関節の目標は関節の可動範囲で切ること** (B-Human・学習の PhysX と同じ)。
- 実機では歩行方策とキック方策を切り替えて使う (蹴るときだけこの方策)。

MuJoCo での確認:
`DISPLAY=:2 ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_walk_kick.py --onnx frozen_policies/bhuman_walk_kick_s6c_9200/policy.onnx --view --scene left90_strong`

## 仕様表 (MuJoCo `sim2sim_bhuman_walk_kick.py --random 100`、ボール 0.095 m、ボール-足の摩擦 0.8、関節の目標を範囲で切る)

合格の判定は seed 0・PD 目標の遅れ 4 物理ステップ (既定)。ほかの列は余裕を見るための参考。

| 項目 | 条件 | seed 0 (判定) | seed 1 | seed 2 | 遅れ 1 | 遅れ 8 | B-Human 完成品 seed 0 |
|---|---|---|---|---|---|---|---|
| 蹴り方 | 90±25° が 80% 以上 | 99% PASS | 99% | 100% | 100% | 99% | 100% |
| 当て方 | 内側が 80% 以上 | 99% PASS | 98% | 100% | 100% | 98% | 100% |
| 止まる (減速) | 0.6 m/s 以下まで減速が 80% 以上 | 98% PASS | 98% | 97% | 96% | 100% | 91% |
| 軸足 | 0.1 秒以上前から接地が 80% 以上 | 87% PASS | 86% | 88% | 83% | 90% | 96% |
| 振り | 遊脚・1.0 m/s 以上が 80% 以上 | 99% PASS | 100% | 100% | 100% | 98% | 99% |
| 向きの精度 | 平均 ≤10°、30° 超 ≤3% | 2.6°・0% PASS | 3.0°・1% | 3.1°・0% | 3.6°・0% | 2.7°・0% | 2.4°・0% |
| 強キック | 平均 ≥3.0 m/s | 5.27 PASS | 5.27 | 5.42 | 5.00 | 5.26 | 4.41 |
| 弱キック | 必要な初速との差の平均 ≤0.8 m/s | 0.72 PASS | 0.75 | 0.73 | 0.90 (未達) | 0.60 | 0.37 |
| 転倒 | 100 場面で 2 以下 | 0 PASS | 0 | 0 | 0 | 0 | 1 |
| つながり | 90% 以上 | 100% PASS | 100% | 100% | 100% | 100% | 97% |
| 合格数 | | **10/10** | 10/10 | 10/10 | 9/10 | 10/10 | 10/10 |

残っている弱いところ: 距離指定キックはまだ強めに蹴る (必要な初速より平均 +0.72 m/s、必要な初速との相関 +0.30。
B-Human は +0.24・相関 +0.41)。近い距離 (1〜3 m) ほど強すぎる。蹴り足を振りの一番速い所でボールに当てていて、
B-Human のように当てる前に足を減速させる動きをまだ覚えていない。PD の遅れ 1 では弱キックが条件を外れる。
