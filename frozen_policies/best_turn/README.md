# best_turn — その場高速回転ポリシー

`Isaac-Velocity-Flat-Turn` (`K1FlatTurnCfg`) の凍結成果物。

## 来歴

| | |
|---|---|
| 学習 run | `scripts/rsl_rl/logs/rsl_rl/k1_turn/2026-09-16_03-56-13` |
| checkpoint | `model_6100.pt` (8000 iter 中 6100、スクラッチ学習) |
| 学習ログ | `scripts/rsl_rl/turn_v11_nohup.log` |
| エクスポート日 | 2026-09-16 |

## インタフェース

ONNX の入出力は**歩行ポリシーと完全に同一**。C++ デプロイ側のモデル読み込み・
バッファ確保のコードは変更不要。

```
入力  command      [1, 3]
      obs_history  [1, 100, 49]
出力  actions      [1, 12]
```

**ただしコマンド 3 枠に書き込む値の意味が歩行と異なる。**

```cpp
// 歩行ポリシー
command[0] = vx;  command[1] = vy;  command[2] = wz;

// 本ポリシー ← これに変える
command[0] = 0.0f;
command[1] = 0.0f;
command[2] = wrap_to_pi(psi_target - psi_current) / M_PI;   // ∈ [-1, 1]
```

`wrap_to_pi` は `atan2(sin(d), cos(d))`。`psi_current` は機体のワールド yaw 角。
**この変更を忘れてもモデルは正常にロードできてしまうので気づきにくい。**

`obs_history` の 49 次元の中身 (gait_phase 含む) は歩行と同一なので、そちらの
生成コードはそのまま使える。

### 引き渡し時の履歴初期化

歩行ポリシーから切り替える際は、**履歴を歩行時のものから持ち越さずタイル埋めする**
(最初の観測を 100 ステップ全部にコピー)。学習時の CircularBuffer も
`num_pushes == 0` で同じタイル埋めをするので分布が一致する。

歩行の履歴には `velocity_commands` 枠に非ゼロの vx/vy が入っており、本タスクの
学習データには **slot 0/1 が非ゼロの履歴が 1 サンプルも存在しない** ため、
持ち越すと完全な分布外になる。

## 学習時の環境設定 (要点)

| 項目 | 値 |
|---|---|
| コマンド | 残り角 Δψ、`turn_angle_range` (0, π)、再サンプリング 1.0-3.0 s |
| 到達後の保持 | `hold_after_settle_s = 2.0` |
| 摩擦 DR (static/dynamic) | U(0.3, 3.5) |
| 足裏 torsional patch radius | 0.051 (USD に焼き込み、`usd_torsional/`) |
| Hip_Pitch 原点 z | -0.062 (修正後) |
| 足首 armature | 0.0565 (修正後、MuJoCo と一致) |
| 初期姿勢 | 関節オフセット ±30° |
| 初期線速度 | カリキュラム ±0.5 → ±1.4 m/s |
| 学習 | スクラッチ、2048 env × 2 GPU |

## 実測値 (it6100, Isaac 上)

| 指標 | 値 |
|---|---|
| `settle_time_s` | 0.336 s |
| `success_rate` (5° 以内到達) | 0.939 |
| 転倒率 | 0.066 |
| `Mean episode length` | 923 / 1000 |
| `settled_speed` (到達中の並進速度) | 0.129 m/s |
| `time_in_target` | 0.797 |

NOTE: `success_rate` は「5° 以内に一度でも入ったか」の二値。**最終到達精度は別物**で、
`scripts/rsl_rl/eval_turn_accuracy.py` で角度帯別に測れる。
評価時は `Isaac-Velocity-Flat-Turn-Finetune` を使うこと
(`Isaac-Velocity-Flat-Turn` だとカリキュラムが目標角を stage 0 = ±45° に戻すため
全角度帯を測れない)。

## 既知の性質

* **接地したまま足裏を捻るピボット旋回**をする (片足支持時間はほぼ 0)。
  人型の正当な旋回方法なので罰していない。摩擦 DR を広げても踏み替えは獲得しなかった。
* yaw 方向の揺れは抑制済み (`settle_ang_stillness` + `settle_ang_vel_l1`)。
  Isaac 上でゲート内 yaw 角速度は約 0.12 rad/s (1 Hz 振動なら振幅 ±1.1° 相当)。
