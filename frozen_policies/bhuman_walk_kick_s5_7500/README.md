# bhuman_walk_kick_s5_7500 — B-Human 版インサイドキック (歩行 model_34995 の上に学習)、仕様 10 項目すべて合格

学習 run: `logs/rsl_rl/k1_bhuman_walk_kick/2026-10-26_20-09-22_s5` (Stage 5)、checkpoint `model_7500.pt`。
仕様と合格条件は `docs/bhuman_walk_kick_spec.md`。2026-10-27 時点の一番良い checkpoint
(前の最良は `bhuman_walk_kick_s4_4600` で、距離指定キックの強さだけ未達だった)。

## インタフェース (ONNX `policy.onnx`)

```
入力  command      [1, 3]        (vx, vy, wz)  ボールと蹴る方向から決める速度指令 (mdp/walk_command.py の式)
      obs_history  [1, 100, 61]  歩行の 49 項 (best_walk と同じ並び) + キック用 12 項 (locomotion/history_layout.py KICK_EXTRA_TERM_SPECS)
出力  actions      [1, 13]       脚 12 (scale 0.5) + 歩行周波数のずらし (±0.5 Hz に clip して基本周波数に足す)
```

- `policy.onnx` は `scripts/mujoco/sim2sim_bhuman_walk_kick.py` の書き出し (履歴入力の actor + 観測の正規化) と同じもの
  (`logs/.../2026-10-26_20-09-22_s5/exported/2026-10-26_20-09-22_s5_model_7500.onnx`)。学習時の actor
  (HistoryEncoderHead + EmpiricalNormalization) との出力の差は最大 2.4e-6 (ランダム入力 16 個)。
- ボールは半径 0.095 m・0.29 kg。**関節の目標は関節の可動範囲で切ること** (B-Human・学習の PhysX と同じ)。
- 実機では歩行方策とキック方策を切り替えて使う (蹴るときだけこの方策)。

MuJoCo での確認:
`DISPLAY=:2 ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_walk_kick.py --onnx frozen_policies/bhuman_walk_kick_s5_7500/policy.onnx --view --scene left90_strong`

## 仕様表 (MuJoCo `sim2sim_bhuman_walk_kick.py --random 100`、ボール 0.095 m、ボール-足の摩擦 0.8、関節の目標を範囲で切る)

合格の判定は seed 0・PD 目標の遅れ 4 物理ステップ (既定)。ほかの列は余裕を見るための参考。

| 項目 | 条件 | seed 0 (判定) | seed 1 | seed 2 | 遅れ 1 | 遅れ 8 | B-Human 完成品 seed 0 |
|---|---|---|---|---|---|---|---|
| 蹴り方 | 90±25° が 80% 以上 | 98% PASS | 100% | 100% | 96% | 99% | 100% |
| 当て方 | 内側が 80% 以上 | 99% PASS | 100% | 99% | 98% | 98% | 100% |
| 止まる (減速) | 0.6 m/s 以下まで減速が 80% 以上 | 98% PASS | 99% | 100% | 97% | 98% | 91% |
| 軸足 | 0.1 秒以上前から接地が 80% 以上 | 87% PASS | 89% | 91% | 76% (未達) | 82% | 96% |
| 振り | 遊脚・1.0 m/s 以上が 80% 以上 | 100% PASS | 100% | 100% | 99% | 100% | 99% |
| 向きの精度 | 平均 ≤10°、30° 超 ≤3% | 5.2°・2% PASS | 2.9°・0% | 3.2°・0% | 4.9°・2% | 3.7°・2% | 2.4°・0% |
| 強キック | 平均 ≥3.0 m/s | 5.25 PASS | 5.38 | 5.39 | 5.04 | 5.43 | 4.41 |
| 弱キック | 必要な初速との差の平均 ≤0.8 m/s | 0.73 PASS | 0.71 | 0.72 | 0.79 | 0.78 | 0.37 |
| 転倒 | 100 場面で 2 以下 | 0 PASS | 0 | 0 | 0 | 3 (未達) | 1 |
| つながり | 90% 以上 | 100% PASS | 100% | 98% | 99% | 97% | 97% |
| 合格数 | | **10/10** | 10/10 | 10/10 | 9/10 | 9/10 | 10/10 |

残っている弱いところ: 距離指定キックはまだ強めに蹴る (必要な初速より平均 +0.69 m/s、必要な初速との相関 +0.11。
B-Human は +0.24・相関 +0.41)。PD の遅れを 1 / 8 にすると軸足 / 転倒が条件を外れる (B-Human は 5 通りすべて合格)。
