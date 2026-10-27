# bhuman_walk_kick_s4_4600 — B-Human 版インサイドキック (歩行 model_34995 の上に学習) の途中経過の最良

学習 run: `logs/rsl_rl/k1_bhuman_walk_kick/2026-10-26_19-18-47_s4` (Stage 4)、checkpoint `model_4600.pt`。
仕様と合格条件は `docs/bhuman_walk_kick_spec.md`。2026-10-26 時点で 10 項目中 9 項目合格、
未達は距離指定キックの強さ (必要な初速との差 平均 1.48 m/s、条件 0.8 以内)。

## インタフェース (ONNX `policy.onnx`)

```
入力  command      [1, 3]        (vx, vy, wz)  ボールと蹴る方向から決める速度指令 (mdp/walk_command.py の式)
      obs_history  [1, 100, 61]  歩行の 49 項 (best_walk と同じ並び) + キック用 12 項 (locomotion/history_layout.py KICK_EXTRA_TERM_SPECS)
出力  actions      [1, 13]       脚 12 (scale 0.5) + 歩行周波数のずらし (±0.5 Hz に clip して基本周波数に足す)
```

ボールは半径 0.095 m・0.29 kg。関節の目標は関節の可動範囲で切ること。
MuJoCo での確認: `scripts/mujoco/sim2sim_bhuman_walk_kick.py --onnx frozen_policies/bhuman_walk_kick_s4_4600/policy.onnx --view --scene left90_strong`
