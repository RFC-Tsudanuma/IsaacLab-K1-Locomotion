# bhuman_walk_kick_s7b_9400 — B-Human 版インサイドキック (歩行 model_34995 の上に学習)、仕様 10 項目すべて合格

学習 run: `logs/rsl_rl/k1_bhuman_walk_kick/2026-10-27_20-04-48_s7b` (Stage 7 の後半 = 7b、B-Human README 6) の最終値)、checkpoint `model_9400.pt`。
仕様と合格条件は `docs/bhuman_walk_kick_spec.md`。2026-10-27 20:25 時点の一番良い checkpoint
(前の最良は `bhuman_walk_kick_s7a_9300`。どちらも seed・PD の遅れを変えた 5 通りすべてで 10/10 で、こちらは弱キックの差が
5 通りとも小さい: 0.34 / 0.46 / 0.38 / 0.35 / 0.35 m/s、前は 0.42 / 0.49 / 0.44 / 0.48 / 0.40)。

学習の段: Stage 6c (s6c model_9200) → Stage 7a (確率は最終値の 10%) を 100 iteration (s7a model_9300) →
Stage 7b (最終値) を 100 iteration (この checkpoint)。
Stage 7b の値 (env cfg `K1BHumanWalkKickStage7EnvCfg` そのまま + kick_direction_sigma_range 0.2):
ball_kick_direction 5、ball_walk_speed 0.75、ball_distance_offset 0.7、ball_rotation_offset 0.2、kick_pose_increase_value 0.2、
random_ball_reset (0.01, 0.02, 0.02)、reset_ball_request 0.005、add_ball_vel / add_ball_vel_in_direction 0.0001、
heavy_ball 0.0001 (新しいボールを出すたびに引き直す。B-Human は env を作る時に 1 回)、
重み ball_kick_velocity 4・ball_kick_velocity_strong 5、速さ合わせの報酬の幅 sigma_scale 0.5、desired_kl 0.003。
B-Human から変えた点と理由は進捗ログ (`claude_tmp/bhuman_walk_kick_progress.md`) と `bhuman_walk_kick_env_cfg.py` のコメント。

## インタフェース (ONNX `policy.onnx`)

```
入力  command      [1, 3]        (vx, vy, wz)  ボールと蹴る方向から決める速度指令 (mdp/walk_command.py の式)
      obs_history  [1, 100, 61]  歩行の 49 項 (best_walk と同じ並び) + キック用 12 項 (locomotion/history_layout.py KICK_EXTRA_TERM_SPECS)
出力  actions      [1, 13]       脚 12 (scale 0.5) + 歩行周波数のずらし (±0.5 Hz に clip して基本周波数に足す)
```

- `policy.onnx` は `scripts/mujoco/sim2sim_bhuman_walk_kick.py` の書き出し (履歴入力の actor + 観測の正規化) と同じもの。
  学習時の actor (HistoryEncoderHead + EmpiricalNormalization) との出力の差は最大 1.5e-6 (ランダム入力 16 個)。
- ボールは半径 0.095 m・0.29 kg。**関節の目標は関節の可動範囲で切ること** (B-Human・学習の PhysX と同じ)。
- 実機では歩行方策とキック方策を切り替えて使う (蹴るときだけこの方策)。

MuJoCo での確認:
`DISPLAY=:2 ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_walk_kick.py --onnx frozen_policies/bhuman_walk_kick_s7b_9400/policy.onnx --view --scene left90_strong`

## 仕様表 (MuJoCo `sim2sim_bhuman_walk_kick.py --random 100`、ボール 0.095 m、ボール-足の摩擦 0.8、関節の目標を範囲で切る)

合格の判定は seed 0・PD 目標の遅れ 4 物理ステップ (既定)。ほかの列は余裕を見るための参考。
「Stage 7 の姿勢の値」は `--stage_cfg '{"ball_distance_offset": 0.7, "ball_rotation_offset": 0.2}'` (MuJoCo 側の歩行指令・キック姿勢の式を Stage 7 の値にしたもの)。

| 項目 | 条件 | seed 0 (判定) | seed 1 | seed 2 | 遅れ 1 | 遅れ 8 | Stage 7 の姿勢の値 | B-Human 完成品 seed 0 |
|---|---|---|---|---|---|---|---|---|
| 蹴り方 | 90±25° が 80% 以上 | 100% PASS | 100% | 100% | 99% | 99% | 97% | 100% |
| 当て方 | 内側が 80% 以上 | 100% PASS | 100% | 100% | 99% | 100% | 96% | 100% |
| 止まる (減速) | 0.6 m/s 以下まで減速が 80% 以上 | 94% PASS | 97% | 93% | 95% | 94% | 92% | 91% |
| 軸足 | 0.1 秒以上前から接地が 80% 以上 | 88% PASS | 92% | 90% | 88% | 94% | 87% | 96% |
| 振り | 遊脚・1.0 m/s 以上が 80% 以上 | 100% PASS | 100% | 100% | 100% | 100% | 99% | 99% |
| 向きの精度 | 平均 ≤10°、30° 超 ≤3% | 2.6°・1% PASS | 3.7°・2% | 1.9°・0% | 4.2°・2% | 2.0°・0% | 2.5°・0% | 2.4°・0% |
| 強キック | 平均 ≥3.0 m/s | 5.37 PASS | 5.45 | 5.44 | 4.73 | 5.50 | 5.51 | 4.41 |
| 弱キック | 必要な初速との差の平均 ≤0.8 m/s | 0.34 PASS | 0.46 | 0.38 | 0.35 | 0.35 | 0.54 | 0.37 |
| 転倒 | 100 場面で 2 以下 | 0 PASS | 0 | 0 | 0 | 1 | 0 | 1 |
| つながり | 90% 以上 | 100% PASS | 100% | 100% | 100% | 99% | 100% | 97% |
| 合格数 | | **10/10** | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |

距離指定キック (seed 0、51 場面、必要な初速の平均 1.40 m/s):
| | 実際の速さ | 差 | 相関 | 近い半分の差 | 遠い半分の差 | 蹴り足の速さ 近い / 遠い |
|---|---|---|---|---|---|---|
| この checkpoint | 1.61 | 0.34 | +0.36 | 0.48 | 0.20 | 2.12 / 2.33 m/s |
| s7a 9300 (前の最良) | 1.74 | 0.42 | +0.35 | 0.60 | 0.23 | 2.29 / 2.46 m/s |
| s6c 9200 | 2.11 | 0.72 | +0.30 | 0.97 | 0.46 | 2.60 / 2.62 m/s |
| B-Human 完成品 | 1.63 | 0.37 | +0.41 | 0.40 | 0.34 | 1.92 / 2.31 m/s |

残っている弱いところ: 近い距離 (必要な初速が小さい半分) ではまだ少し強めに蹴る (差 0.48、B-Human 0.40)。
軸足の接地の余裕は B-Human (96%) より小さい (88〜94%)。強キックは B-Human より強い (5.4 vs 4.4 m/s)。
