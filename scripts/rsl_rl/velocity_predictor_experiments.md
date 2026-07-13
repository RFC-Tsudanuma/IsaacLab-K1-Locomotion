# 速度推定モデル（VelocityPredictor）実験記録

経路計画用途で、歩行コマンド列 `[vx, vy, wz]` から **実際に実現される本体速度** を推定する
モデルのアーキテクチャ／学習設定の探索記録。

- モデル: `VelocityPredictor` = 1次系ベースライン（`v_base = α·v_base_prev + (1−α)·gain·cmd`, `α=exp(−dt/τ)`）+ GRU残差
- データ収集ポリシー: `logs/rsl_rl/k1_flat/0524_walk/0524_walk.pt`（flat env, Isaac-Velocity-Flat-K1-v0）
- データセット: `data/rollouts_switch.zarr`（81,920エピソード×250step, dt=0.02=50Hz, **proprioなし**）
  - 経路計画用途のため固有感覚は使えない → proprioなしで統一
  - コマンドは頻繁切替（piecewise-constant, hold 0.25–1.25s）を約37.5%含む分布
- 損失: 終了マスク付き per-axis MSE、最適化: AdamW(lr 1e-3, wd 1e-4) + CosineAnnealingLR(T_max=epochs)
- train/val = 73,728 / 8,192、評価指標は **val MSE**

> 50Hz固定前提（dt=0.02をベースラインのαとGRUの時間ダイナミクス両方が前提とする）。別レートは再収集・再学習が必要。

---

## 実験1: proprioの有無（参考・初期検討）

| 構成 | val MSE | val MAE (vx/vy/wz) |
|---|---|---|
| proprioなし h64 50ep（非切替データ） | 0.0491 | 0.100 / 0.135 / 0.220 |
| proprioあり(joint_pos+joint_vel) h128 | **0.0176**(epoch4時点) | 0.059 / 0.053 / 0.143 |

→ proprioは大幅に効くが、**経路計画では推論時に固有感覚が無いため使用不可**。以降はproprioなしで進める。

---

## 実験2: アーキテクチャ × エポック数スイープ（本実験）

データ: `data/rollouts_switch.zarr`（頻繁切替・proprioなし）。各runは独立のcosineスケジュール。

| 構成 | epochs | best val | best epoch | final val | 過学習(final gap = val−train) |
|---|---|---|---|---|---|
| hidden=128, 1層（基準） | 50 | 0.05533 | ~45 | — | ~0 |
| hidden=256, 1層 | 50 | 0.05535 | 32 | 0.05535 | +0.00019 |
| hidden=256, 1層 | 100 | 0.05538 | 32 | 0.05633 | +0.00237 |
| hidden=256, 1層 | 150 | 0.05538 | 32 | 0.05776 | +0.00543 |
| **hidden=128, 2層** | **50** | **0.05533** | 39 | 0.05533 | +0.00012 |
| hidden=128, 2層 | 100 | 0.05537 | 38 | 0.05579 | +0.00111 |
| hidden=128, 2層 | 150 | 0.05539 | 29 | 0.05659 | +0.00290 |

### 結論
1. **全構成が val MSE ≈ 0.0553 で頭打ち（飽和）**。256次元化・2層化とも基準(h128/1層, 0.05533)を**改善しない**（差は0.0001未満＝ノイズ）。
2. **容量を増やすほど過学習が早く・強く出る**。val最小エポックは早期化（h256≈32, h128/2層≈39）し、以降valは悪化。
3. **50エポック超は全構成で過学習が進むだけで無意味**（100/150は不要）。最適エポックは概ね **40–50**。

### 最適パラメータ（採用）
**hidden_dim=128 / num_layers=1 / epochs=50**

- 最良タイ（0.05533）かつ最もシンプル・最も過学習しにくい。
- これは既にデプロイ済みの `logs/velocity_predictor/predictor_planning_noproprio.{pt,onnx}` と同一 → **再学習・再エクスポート不要**。
- h128/2層/50ep も同値タイだが複雑化するだけでメリットなし。h256は僅かに劣り過学習しやすい。

### 解釈
command→実速度（proprioなし）のタスクは情報量が限られ、1次系+小GRUで表現力が飽和。容量追加は汎化を変えず過学習リスクのみ増える。**小さいモデルを維持するのが正解**。

---

## 実験3: データセットサイズ（82k vs 328k episodes）

最適構成（hidden=128, 1層, 50ep）で、データ量のみを4倍に変えて比較。

| データセット | episodes | 総ステップ | best val MSE | best epoch | final gap (val−train) |
|---|---|---|---|---|---|
| 現行 `rollouts_switch.zarr` | 81,920 | 2,048万 | **0.05533** | ~45 | ~0 |
| 4倍 `rollouts_switch_large.zarr` | 327,680 | 8,192万 | 0.05594 | 43 | −0.00022 |

### 結論
- **データを4倍にしても val MSE は改善しない**（差はノイズレベル、むしろ僅かに悪化）。
- 4倍データでは **final gap が負（val ≤ train）= 過学習ゼロ**。trainerror も同じ床に張り付き、データ量律速でないことを示す。

### 総合解釈（実験2+3）
容量増（実験2）でもデータ増（実験3）でも ~0.055 を下回らない → これは **モデル容量でもデータ量でもなく、タスク本来の予測不能性（アレアトリック誤差）による床**。
command（指令速度）だけからは、接触・歩行位相・外乱に依存する実速度のばらつきを原理的に説明できない。proprioで0.018まで下がる（実験1）のはこの不確かさを観測で埋めるため。
→ **現行の82kデータ・h128/1層/50ep で十分。データ拡張・モデル拡大とも投資対効果なし**。

---

## 再現コマンド

```bash
# 基準/最適（hidden=128, 1層, 50ep）
uv run python train_velocity_predictor.py \
  --data data/rollouts_switch.zarr \
  --output logs/velocity_predictor/predictor_planning_noproprio.pt \
  --hidden_dim 128 --num_layers 1 --epochs 50

# 比較に使った構成例
uv run python train_velocity_predictor.py --data data/rollouts_switch.zarr \
  --hidden_dim 256 --num_layers 1 --epochs 50 --output /tmp/h256_l1_e50.pt
uv run python train_velocity_predictor.py --data data/rollouts_switch.zarr \
  --hidden_dim 128 --num_layers 2 --epochs 50 --output /tmp/h128_l2_e50.pt
```

実施日: 2026-06（feat/inoue_walk ブランチ）
