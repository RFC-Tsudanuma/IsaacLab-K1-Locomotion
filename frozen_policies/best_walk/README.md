# best_walk — 歩行の最良ポリシー (平面特化 ±2.0 m/s + 省エネ仕上げ)

`Isaac-Velocity-Flat-Stance-Plane-Polish` (`K1FlatStancePlanePolishCfg`) の凍結成果物。
追加学習 (warm-start / finetune) の起点として使うことを想定。

## 来歴

| | |
|---|---|
| 学習 run | `scripts/rsl_rl/logs/rsl_rl/k1_flat/2026-09-13_04-35-39` |
| checkpoint | `model_52500.pt` (2500 iter 仕上げ中の iter ~1000、膝判定で採用) |
| ONNX | `policy.onnx` (元名 `dual_policy_2ms_random_power_043539.onnx`) |
| 凍結日 | 2026-09-16 |

系譜 (すべて warm-start の連鎖):

```
model_34995 (±1.5 歩行 + posture/joint_power finetune 済, 2026-08-10_03-07-48)
  → K1FlatStanceCfg      : stance_ratio 0.60 + no_fly 4 (model_41000, 2026-09-04_13-50-47)
  → K1FlatStancePlaneCfg : 平面 0.7 地形 + 摩擦 0.6-1.0 + x±2.0 (model_50999, 2026-09-09_18-32-40)
  → Polish (jp -1.5e-5 + ori -30)              (model_51500, 2026-09-12_01-09-17)
  → Polish + 関節初期角オフセット ±30°          (model_52500, 本成果物)
```

## 実測値 (measure_stride.py, Isaac 平面分布上)

| 指令 vx | 実速度 | 着地歩幅 | fly% (両足空中率) |
|---|---|---|---|
| 1.0 | 0.98 | 0.226 m | 0.2 |
| 1.5 | 1.46 | 0.296 m | 0.5 |
| 2.0 | **1.88** | 0.336 m | 1.3 |

* 関節パワー Σ(τ·ω)² 相当: 16026 (仕上げ前 model_50999 の 22236 比 **-28%**)
* extreme corner 頑健化済み (prob 0.35 学習)、±30° 初期姿勢オフセット耐性あり
* mujoco で動作確認済み (仕上げ前世代 2026-09-11、仕上げは追従同等)

## インタフェース

歩行ポリシー標準。C++ デプロイ側の変更は不要。

```
入力  command      [1, 3]   (vx, vy, wz)
      obs_history  [1, 100, 49]
出力  actions      [1, 12]
```

位相周波数マッピングは通常版のまま: `||cmd_xy|| ≤ 1.0 m/s → 1.8 Hz`、
1.8 m/s で 2.0 Hz (以降同傾きで外挿)。`k1_constants_isaaclab.hpp` の
(PHASE_FREQ_LOW, PHASE_FREQ_HIGH) = (1.8, 2.0) と整合する。

## 追加学習での使い方

元 run が logs に残っている間はそこから resume するのが簡単:

```bash
uv run torchrun --standalone --nproc_per_node=2 train.py \
    --task <派生タスク> --headless --distributed --num_envs 2048 \
    --resume --load_run 2026-09-13_04-35-39 --checkpoint model_52500.pt \
    --reset_noise_std 0.05
```

logs を消した場合は、本ディレクトリの `model_52500.pt` を
`scripts/rsl_rl/logs/rsl_rl/k1_flat/<任意のrun名>/` に置けば同様に resume できる。

**注意 (この系譜共通):**

* resume には `--reset_noise_std 0.05` が必須 (action noise std が収束済みで、
  optimizer state ごと載せると std が負に落ちてクラッシュする)。
* `--max_iterations N` は「追加 N イテレーション」の意味。
* 派生タスクの環境は `K1FlatStancePlaneCfg` 系から派生させること。
  素の `K1FlatEnvCfg` 系 (凹凸 0.7 地形・摩擦 0.3〜・stance_ratio 0.5・no_fly なし・±1.5)
  に載せると学習分布が食い違い、本ポリシーの特性 (接地重視・平面高速) を壊す。

## 学習時の環境設定 (要点、詳細は同梱 `env.yaml`)

| 項目 | 値 |
|---|---|
| 地形 | 平面 0.7 / ランダム凹凸 0.3 (noise 0.01-0.04) |
| 摩擦 DR (static/dynamic) | U(0.6, 1.0) |
| stance_ratio (feet_phase) | 0.60 |
| no_fly (両足空中ペナルティ) | weight 4.0 |
| 速度レンジ | x ±2.0 / y ±0.9 / yaw ±1.0 (固定、extreme prob 0.35) |
| 高速域ゲート緩和 (1.5→1.8 m/s) | base_height 床 0.53→0.48、feet_parallel 0.5 倍 |
| 仕上げ報酬 | joint_power_l2 -1.5e-5、flat_orientation_l2 -30 |
| 関節初期角 | オフセット加算 ±30° (reset_joints_by_offset、リミット clamp) |

## 既知の性質

* 速度上限は「フライト禁止 + 最大 2.0 Hz」の物理制約でほぼ天井 (2.0 指令 → 1.88)。
  これ以上はケイデンス増かフライト許可が必要。
* 2.0 m/s 指令時は接地中の足速度 ~0.5 m/s の滑り成分を含む (摩擦 0.6+ で学習済みだが
  実機の床材次第では要確認)。
* 仕上げ学習をさらに回すとパワーは減るが 2.0 m/s 追従が壊れる
  (iter ~53500 で実速度 1.69 まで劣化する事を確認済み。膝が本 checkpoint)。
