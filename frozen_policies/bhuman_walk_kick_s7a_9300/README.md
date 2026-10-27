# bhuman_walk_kick_s7a_9300 — B-Human 版インサイドキック (歩行 model_34995 の上に学習)、仕様 10 項目すべて合格

学習 run: `logs/rsl_rl/k1_bhuman_walk_kick/2026-10-27_18-59-24_s7a` (Stage 7 の前半 = 7a)、checkpoint `model_9300.pt`。
仕様と合格条件は `docs/bhuman_walk_kick_spec.md`。2026-10-27 19:30 時点の一番良い checkpoint
(前の最良は `bhuman_walk_kick_s6c_9200`。判定の条件ではどちらも 10/10 で、こちらは seed・PD の遅れを変えた 5 通りすべてで 10/10、
弱キックの差が 0.72 → 0.42 m/s)。

学習の段: Stage 6c (s6c model_9200) → Stage 7a (README 6) の前半) を 100 iteration (この checkpoint)。
Stage 7a の値: ball_kick_direction 5、ball_walk_speed 0.75、ball_distance_offset 0.7、ball_rotation_offset 0.2、
ボールの置き直し・速度付与の確率は最終値の 10% (random_ball_reset (0.001, 0.002, 0.002)、reset_ball_request 0.0005、
add_ball_vel / add_ball_vel_in_direction 0.00001)、heavy_ball 0.00001、kick_pose_increase_value 0.025 (0.2 は 7b)、
重みは README どおり ball_kick_velocity 4・ball_kick_velocity_strong 5、速さ合わせの報酬の幅 sigma_scale 0.5 (K1_Ball.yaml の final)、
kick_direction_sigma_range 0.2。
B-Human から変えた点と理由は進捗ログ (`claude_tmp/bhuman_walk_kick_progress.md`) と `bhuman_walk_kick_env_cfg.py` のコメント。

## インタフェース (ONNX `policy.onnx`)

```
入力  command      [1, 3]        (vx, vy, wz)  ボールと蹴る方向から決める速度指令 (mdp/walk_command.py の式)
      obs_history  [1, 100, 61]  歩行の 49 項 (best_walk と同じ並び) + キック用 12 項 (locomotion/history_layout.py KICK_EXTRA_TERM_SPECS)
出力  actions      [1, 13]       脚 12 (scale 0.5) + 歩行周波数のずらし (±0.5 Hz に clip して基本周波数に足す)
```

- `policy.onnx` は `scripts/mujoco/sim2sim_bhuman_walk_kick.py` の書き出し (履歴入力の actor + 観測の正規化) と同じもの。
  学習時の actor (HistoryEncoderHead + EmpiricalNormalization) との出力の差は最大 2.6e-6 (ランダム入力 16 個)。
- ボールは半径 0.095 m・0.29 kg。**関節の目標は関節の可動範囲で切ること** (B-Human・学習の PhysX と同じ)。
- 実機では歩行方策とキック方策を切り替えて使う (蹴るときだけこの方策)。

MuJoCo での確認:
`DISPLAY=:2 ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_walk_kick.py --onnx frozen_policies/bhuman_walk_kick_s7a_9300/policy.onnx --view --scene left90_strong`

## 仕様表 (MuJoCo `sim2sim_bhuman_walk_kick.py --random 100`、ボール 0.095 m、ボール-足の摩擦 0.8、関節の目標を範囲で切る)

合格の判定は seed 0・PD 目標の遅れ 4 物理ステップ (既定)。ほかの列は余裕を見るための参考。
「Stage 7 の姿勢の値」は `--stage_cfg '{"ball_distance_offset": 0.7, "ball_rotation_offset": 0.2}'` (MuJoCo 側の歩行指令・キック姿勢の式を Stage 7 の値にしたもの)。

| 項目 | 条件 | seed 0 (判定) | seed 1 | seed 2 | 遅れ 1 | 遅れ 8 | Stage 7 の姿勢の値 | B-Human 完成品 seed 0 |
|---|---|---|---|---|---|---|---|---|
| 蹴り方 | 90±25° が 80% 以上 | 100% PASS | 98% | 100% | 99% | 99% | 95% | 100% |
| 当て方 | 内側が 80% 以上 | 100% PASS | 98% | 100% | 99% | 98% | 95% | 100% |
| 止まる (減速) | 0.6 m/s 以下まで減速が 80% 以上 | 96% PASS | 97% | 92% | 95% | 96% | 94% | 91% |
| 軸足 | 0.1 秒以上前から接地が 80% 以上 | 94% PASS | 88% | 86% | 87% | 94% | 84% | 96% |
| 振り | 遊脚・1.0 m/s 以上が 80% 以上 | 100% PASS | 100% | 100% | 100% | 100% | 97% | 99% |
| 向きの精度 | 平均 ≤10°、30° 超 ≤3% | 2.8°・0% PASS | 3.5°・1% | 3.9°・1% | 4.2°・1% | 3.3°・2% | 3.3°・0% | 2.4°・0% |
| 強キック | 平均 ≥3.0 m/s | 4.94 PASS | 5.08 | 5.22 | 4.62 | 5.33 | 5.23 | 4.41 |
| 弱キック | 必要な初速との差の平均 ≤0.8 m/s | 0.42 PASS | 0.49 | 0.44 | 0.48 | 0.40 | 0.64 | 0.37 |
| 転倒 | 100 場面で 2 以下 | 0 PASS | 0 | 0 | 1 | 0 | 0 | 1 |
| つながり | 90% 以上 | 100% PASS | 98% | 99% | 97% | 100% | 100% | 97% |
| 合格数 | | **10/10** | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |

距離指定キック (seed 0、51 場面、必要な初速の平均 1.40 m/s):
| | 実際の速さ | 差 | 相関 | 近い半分の差 | 遠い半分の差 | 蹴り足の速さ 近い / 遠い |
|---|---|---|---|---|---|---|
| この checkpoint | 1.74 | 0.42 | +0.35 | 0.60 | 0.23 | 2.29 / 2.46 m/s |
| s6c 9200 (前の最良) | 2.11 | 0.72 | +0.30 | 0.97 | 0.46 | 2.60 / 2.62 m/s |
| B-Human 完成品 | 1.63 | 0.37 | +0.41 | 0.40 | 0.34 | 1.92 / 2.31 m/s |

残っている弱いところ: 近い距離 (必要な初速が小さい半分) ではまだ強めに蹴る (差 0.60、B-Human 0.40)。
軸足の接地の余裕は B-Human (96%) より小さい (86〜94%)。
