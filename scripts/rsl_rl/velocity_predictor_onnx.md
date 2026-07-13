# 速度推定モデルの ONNX 化・推論ガイド

経路計画用途で、歩行コマンド列 `[vx, vy, wz]` から **実際に実現される本体速度** を推定する
`VelocityPredictor`（1次系ベースライン + GRU残差）を ONNX 化し、自己回帰的に推論するための
2スクリプトの使い方をまとめる。

対象スクリプト:

| スクリプト | 役割 | Isaac Sim |
|---|---|---|
| `export_velocity_predictor_onnx.py` | 学習済み `.pt` → 単一ステップ ONNX | 不要（pure PyTorch + onnx） |
| `velocity_onnx_inference.py` | ONNX を状態保持で逐次推論するクラス | 不要（onnxruntime のみ） |

> 前段（データ収集 → 学習）は `collect_velocity_data.py` → `train_velocity_predictor.py`。
> 本ガイドは学習済みチェックポイント（例: `logs/velocity_predictor/predictor_planning_noproprio.pt`）が
> ある前提で進める。

すべて `scripts/rsl_rl/` 直下で実行する。Python は `uv run python ...`（= `_labpython2`）。

---

## 1. ONNX へのエクスポート — `export_velocity_predictor_onnx.py`

`VelocityPredictor.step()` と等価な「1ステップ分」のグラフを ONNX に書き出す。
GRU 隠れ状態 `h` と 1次系の `v_base` を **入出力として明示** するため、呼び出し側が再帰状態を
保持して任意長のコマンド列を回せる。

### 実行

```bash
# 出力は既定で <チェックポイントと同じ場所>/<同名>.onnx
uv run python export_velocity_predictor_onnx.py \
  --checkpoint logs/velocity_predictor/predictor_planning_noproprio.pt

# 出力先を明示する場合
uv run python export_velocity_predictor_onnx.py \
  --checkpoint logs/velocity_predictor/predictor_planning_noproprio.pt \
  --output exported/velocity_predictor.onnx
```

### 引数

| 引数 | 既定 | 説明 |
|---|---|---|
| `--checkpoint` | （必須） | `train_velocity_predictor.py` が出力した `.pt` |
| `--output` | `<checkpoint>.onnx` | 出力 ONNX パス |
| `--opset` | `17` | ONNX opset バージョン |

### 入出力仕様（`N` = バッチ＝同時に回すストリーム数、動的軸）

入力:

| 名前 | 形状 | 説明 |
|---|---|---|
| `cmd_t` | `(N, 3)` | 速度コマンド `[vx, vy, wz]` |
| `v_base_prev` | `(N, 3)` | 前ステップの 1次系ベースライン速度（初期 0） |
| `h_prev` | `(1, N, hidden)` | 前ステップの GRU 隠れ状態（初期 0） |
| `proprio_t` | `(N, P)` | **proprio 版モデルのみ**。固有感覚ベクトル |

出力:

| 名前 | 形状 | 説明 |
|---|---|---|
| `v_pred` | `(N, 3)` | 推定される実速度 `[vx, vy, wz]` ← これが欲しい値 |
| `v_base` | `(N, 3)` | 更新後ベースライン（次ステップの `v_base_prev`） |
| `h_new` | `(1, N, hidden)` | 更新後隠れ状態（次ステップの `h_prev`） |

`dt` / `hidden_dim` / `proprio_dim` / `residual_scale` / `num_layers` / `use_proprio` /
`proprio_keys` は ONNX の `metadata_props` に埋め込まれ、推論クラスが自動で読み取る。

> **proprio 版について**: 学習時に `--use_proprio` を付けたモデルなら、エクスポート時に
> 自動で `proprio_t` 入力が追加される。ただし経路計画では固有感覚は使えないため、通常は
> proprio なしモデル（`predictor_planning_noproprio.pt`）を使う。

---

## 2. 逐次推論 — `velocity_onnx_inference.py`

`VelocityPredictorOnnx` クラスが ONNX をラップし、`h` と `v_base` を **内部で保持**する。
呼び出し側はコマンドを渡すだけで推定速度が返る。

### Python から使う

```python
from velocity_onnx_inference import VelocityPredictorOnnx
import numpy as np

pred = VelocityPredictorOnnx("logs/velocity_predictor/predictor_planning_noproprio.onnx")

# --- 1ステップずつ（オンライン／プランナのループ内） ---
pred.reset(num_envs=1)                  # 状態を 0 で初期化
for cmd in command_stream:              # cmd: (3,) または (N, 3)
    v_hat = pred.step(cmd)              # -> (N, 3) 推定実速度
    # v_hat を使って経路コスト評価など

# --- コマンド列を一括ロールアウト ---
cmd_seq = np.random.randn(100, 3).astype(np.float32) * 0.5   # (T, 3)
v_seq = pred.predict_sequence(cmd_seq)                        # (T, 3)
# 複数ストリームなら cmd_seq: (T, N, 3) -> v_seq: (T, N, 3)
```

### API

| メソッド / 属性 | 説明 |
|---|---|
| `VelocityPredictorOnnx(onnx_path, providers=None)` | ロード。`providers` 既定は `["CPUExecutionProvider"]` |
| `reset(num_envs=1)` | 再帰状態（`v_base`, `h`）を 0 初期化。ストリーム数を設定 |
| `step(cmd, proprio=None)` | 1ステップ進める。`cmd`: `(3,)` or `(N,3)` → 返り値 `(N,3)`。バッチ数が変わると自動 `reset` |
| `predict_sequence(cmd_seq, proprio_seq=None)` | `(T,3)` or `(T,N,3)` を一括処理し同形状を返す（内部で `reset` 実行） |
| `.dt` `.hidden_dim` `.proprio_dim` `.use_proprio` `.num_layers` `.proprio_keys` | ONNX メタから自動設定 |

> **状態のリセット**: エピソード／軌道が切り替わったら `reset()` を呼ぶ。連続したコマンド列の
> 途中では呼ばない（再帰状態が途切れる）。
>
> **proprio 版**: `use_proprio=True` のモデルでは `step(cmd, proprio=...)` のように固有感覚を
> 渡す必要がある（未指定だと例外）。

### CLI スモークテスト

```bash
uv run python velocity_onnx_inference.py \
  --onnx logs/velocity_predictor/predictor_planning_noproprio.onnx --steps 50
# dt / hidden_dim 等のメタと、一定コマンドを50ステップ入れた後の v_pred を表示
```

---

## 3. 全体パイプライン（参考）

```bash
# (1) ポリシーからデータ収集（頻繁切替コマンド込み）
uv run collect_velocity_data.py \
  --task Isaac-Velocity-Flat-K1-v0 \
  --checkpoint logs/rsl_rl/k1_flat/0524_walk/0524_walk.pt \
  --headless --num_envs 4096 --num_rollouts 20 --episode_length 250 \
  --output data/rollouts_switch.zarr

# (2) 速度推定モデルを学習（proprio なし・epochs 既定50）
uv run python train_velocity_predictor.py \
  --data data/rollouts_switch.zarr \
  --output logs/velocity_predictor/predictor_planning_noproprio.pt \
  --hidden_dim 128

# (3) ONNX 化
uv run python export_velocity_predictor_onnx.py \
  --checkpoint logs/velocity_predictor/predictor_planning_noproprio.pt

# (4) 推論（velocity_onnx_inference.py を import して利用）
```

---

## 補足

- ONNX 出力は PyTorch の `model.step()` と数値一致を確認済み（max 誤差 ~1e-7、バッチ可変対応）。
- 必要依存: `onnx`, `onnxruntime`（`pyproject.toml` に追加済み）。
- バッチ軸 `N` は動的なので、1ストリームでも多数ストリーム（候補軌道の並列評価など）でも同一 ONNX で回せる。
