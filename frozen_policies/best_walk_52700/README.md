# best_walk_52700 — 試合で使う新しい歩行 (平面特化 ±2.0 m/s + 省エネ・上体揺れ低減の仕上げ)

別ブランチ `origin/feat/inoue_walk_double_encoder` (commit `53ce175`) の `frozen_policies/best_walk/` から写したもの。
`model_52700.pt`・`policy.onnx`・`env.yaml` はそこのものとバイト単位で同じ (`policy.onnx` の元名は `policy_054042.onnx`、
`env.yaml` は元の run の params/env.yaml)。このブランチ (bhuman_walk_kick) では、キック方策をこの歩行の上に学習し直すとき・
歩行からキックへの受け渡しを学習と MuJoCo で再現するときに使う。

| | |
|---|---|
| 学習 run (元) | `scripts/rsl_rl/logs/rsl_rl/k1_flat/2026-11-01_05-40-42` (feat/inoue_walk_double_encoder) |
| checkpoint | `model_52700.pt` (`K1FlatStancePlanePolishCfg` + 上体揺れ低減の finetune +200 iter) |

系譜: model_34995 → Stance (stance_ratio 0.60・no_fly 4) → StancePlane (平面 0.7・摩擦 0.6〜1.0・x ±2.0) →
Polish (joint_power_l2 -1.5e-5・flat_orientation_l2 -30) → 関節初期角 ±30° → 上体揺れ低減
(ang_vel_xy_l2 -0.65・flat_orientation_l2 -40・base_ang_acc_l2 -2e-5)。

## インタフェース

ネットワークの形は model_34995 と同じ (どのテンソルも同じ形)。

```
入力  command      [1, 3]        (vx, vy, wz)
      obs_history  [1, 100, 49]
出力  actions      [1, 12]
```

位相の周波数は model_34995 と同じ (|cmd_xy| ≤ 1.0 m/s で 1.8 Hz、1.8 m/s で 2.0 Hz、その先も同じ傾き)。

## 測った速さ

- 指令 2.0 m/s で Isaac 1.87 m/s、MuJoCo 1.70 m/s。
- MuJoCo で 2.0 (実速度 1.70) から指令 0 にすると、0.40 秒・0.52 m で 0.6 m/s、0.59 秒・0.60 m で 0.3 m/s まで落ちる (転倒なし)。

## このブランチでの使い方

- キックの形に移す: `python scripts/rsl_rl/convert_walk_to_bhuman_kick.py --walk frozen_policies/best_walk_52700/model_52700.pt --out logs/rsl_rl/k1_bhuman_walk_kick/walk52700/model_0.pt`
- 学習での受け渡し (この歩行で遠くのボールへ近づいてからキック方策へ渡す) は `env.commands.kick.handoff_prob` で有効にする
  (`bhuman_walk_kick/mdp/walk_handoff.py`、既定は無効)。歩行の重みはこのフォルダの `model_52700.pt` を読む。
- MuJoCo の受け渡しの確認は `scripts/mujoco/sim2sim_bhuman_walk_kick.py --handoff ...` (歩行の ONNX は既定でこのフォルダの `policy.onnx`)。
