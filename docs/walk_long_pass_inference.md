# walk_long_pass ポリシー 推論側実装ガイド

対象: `walk_long_pass_kickfoot_9600_policy.onnx`
(`logs/rsl_rl/k1_walk_long_pass/2026-08-17_22-56-06/exported/`)

| 項目 | 値 |
|---|---|
| ONNX 入力 | `obs` : `float32[1, 100, 55]` |
| ONNX 出力 | `actions` : `float32[1, 12]` |
| 制御周期 | **0.02 s (50 Hz)** — `decimation(4) × sim.dt(0.005)` |
| 正規化 | **ONNX 内に焼き込み済み**。生の観測をそのまま渡す |

---

## 1. 全体の流れ

```python
# 50 Hz ループ
obs_t   = build_observation()          # (55,) float32  ← 第 2 章
buf.push(obs_t)                        # 古い順に 100 フレーム
actions = session.run(None, {"obs": buf.as_array()[None]})[0][0]   # (12,)
q_target = default_joint_pos + 0.5 * actions                       # 第 4 章
send_joint_position_targets(q_target)
```

**バッファは古い順**。`buf[0]` が最も古く、`buf[99]` が現在フレーム。
起動直後・リセット直後は **現在フレームで 100 個すべてを埋める**（ゼロ埋めしない。
学習時の `CircularBuffer` がリセット時に同じことをするため）。

```python
class ObsHistory:
    def __init__(self, length=100, dim=55):
        self.buf = np.zeros((length, dim), dtype=np.float32)
        self.primed = False

    def push(self, obs):
        if not self.primed:                 # 初回は全フレームを現在値で埋める
            self.buf[:] = obs
            self.primed = True
        else:
            self.buf[:-1] = self.buf[1:]    # 古い方へシフト
            self.buf[-1] = obs

    def reset(self):
        self.primed = False
```

---

## 2. 観測 55 次元の作り方

すべて **yaw-aligned ボディフレーム**（胴体の yaw だけを打ち消し、roll/pitch は残す）
で表す。以下 `R_yaw⁻¹(v)` は world ベクトル `v` をそのフレームへ変換する操作。

```python
def yaw_inv(v_w, base_quat_w):
    """world ベクトルを yaw-aligned ボディフレームへ。"""
    yaw = yaw_from_quat(base_quat_w)
    c, s = cos(yaw), sin(yaw)
    return np.array([ c*v_w[0] + s*v_w[1],
                     -s*v_w[0] + c*v_w[1],
                      v_w[2]])
```

| # | 位置 | 項 | 次元 | 作り方 |
|---|---|---|---:|---|
| 1 | 0:3 | `projected_gravity` | 3 | IMU の重力方向を**フルの姿勢**で body frame へ（yaw だけでなく roll/pitch も打ち消す） |
| 2 | 3:6 | `base_ang_vel` | 3 | IMU の角速度（body frame） |
| 3 | **6:9** | **`ball_pos`** | 3 | `yaw_inv(ball_center_w − right_foot_link_origin_w)` ← **第 3 章** |
| 4 | 9:11 | `gait_phase` | 2 | `[sin(φ), cos(φ)]`, `φ = 2π · 1.6 · t` |
| 5 | 11:23 | `joint_pos` | 12 | `q − q_default`（下記の関節順） |
| 6 | 23:35 | `joint_vel` | 12 | `q̇`（生の関節速度） |
| 7 | 35:47 | `prev_joint_request` | 12 | **前ステップの ONNX 出力 `actions` そのもの**（関節角ではない） |
| 8 | 47:48 | `gait_phase_factor_offset` | 1 | **0.0 固定**（周波数 DR は sim 専用） |
| 9 | 48:50 | `kick_direction` | 2 | `yaw_inv([cos θ, sin θ, 0])[:2]`, θ = world での目標蹴り方向 |
| 10 | 50:51 | `target_kick_velocity` | 1 | 目標ボール速度 [m/s]（**3.2–5.0** が学習帯） |
| 11 | 51:53 | `ball_vel` | 2 | `yaw_inv(ball_vel_w)[:2]`（水平 2 成分のみ） |
| 12 | 53:55 | `prev_ball_pos` | 2 | **1 ステップ前**の `yaw_inv(ball_center_w − base_origin_w)[:2]` ← **胴体基準** |

### 関節順（12、`preserve_order=True`）

```
Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw, Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll,
Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw, Right_Knee_Pitch, Right_Ankle_Pitch, Right_Ankle_Roll
```

観測 `joint_pos`/`joint_vel`、出力 `actions`、`prev_joint_request` すべてこの順。

### 歩行位相 `gait_phase` の注意

```python
phase = 2.0 * math.pi * 1.6 * t          # t = エピソード開始からの経過時間 [s]
gait_phase = [sin(phase), cos(phase)]
# 停止時はゼロ埋め（学習時は base_velocity 指令 < 0.05 で 0 になる）
if is_stopped:
    gait_phase = [0.0, 0.0]
```

* `t` は**リセットからの経過時間**。`episode_length_buf × 0.02` に対応する
* 周波数は基準値 **1.6 Hz** 固定。sim では ±0.05 Hz の DR があるが実機は基準値でよい
  （そのぶんを開示するのが項 8 で、実機では 0.0）
* **位相の初期値は 0**（`t=0` で `[0, 1]`）。一歩目の足が固定されるのは仕様

### `prev_ball_pos` は胴体基準（項 3 と原点が違う）

項 3 `ball_pos` は**右足リンク原点**基準、項 12 `prev_ball_pos` は**胴体原点**基準。
混同しないこと。`prev_ball_pos` は 1 ステップ（0.02 s）前の値を渡す。
起動直後は前ステップが無いので現在値で初期化する。

---

## 3. `ball_pos`（3 番目のスロット）— 最重要

このスロットが今回の変更点。**回り込み方向による蹴りタイミングの非対称を解消**する
ために、胴体基準ではなく**蹴り足（右足）基準**にしてある。

```python
ball_pos = yaw_inv(ball_center_w - right_foot_link_origin_w, base_quat_w)   # (3,)
```

### 基準点は「リンク原点」であって足裏ではない

`right_foot_link` の **URDF で定義されたリンク座標系の原点**。足裏でも接触点でもない。
参考までに、足裏はリンク原点の **0.038 m 下**（コード内 `_SOLE_OFFSET`）。

```python
# 誤り: 足裏を基準にすると z が 0.038 ずれる
ball_pos = yaw_inv(ball_center_w - right_sole_w)      # ✗

# 正しい: リンク原点
ball_pos = yaw_inv(ball_center_w - right_foot_link_origin_w)   # ✓
```

sim と実機で同じ URDF を使っているかを必ず確認すること。リンク原点の定義が違うと
この観測がずれる。

### 回転の基準は胴体（足ではない）

原点は足だが、向きは**胴体の yaw**。足首の角度やつま先の向きは反映されない。

---

## 4. 出力の扱い

```python
q_target = default_joint_pos + 0.5 * actions
```

* `scale = 0.5`、`use_default_offset = True`（`JointPositionActionCfg`）
* `default_joint_pos` は sim の初期姿勢と同じ値を使うこと
* 出力は関節**位置指令**。PD ゲインも sim と揃える

`prev_joint_request`（項 7）に入れるのは `q_target` ではなく **生の `actions`**。

---

## 5. センサ遅延について

学習時は以下の DR を掛けているが、**推論側で遅延を再現する必要はない**。
実機の実レイテンシがこの範囲に収まっていれば、ポリシーは吸収できる。

| グループ | 対象 | 学習時の遅延 |
|---|---|---|
| `imu` | `projected_gravity`, `base_ang_vel` | 0–0.02 s |
| `encoder` | `joint_pos`, `joint_vel` | 0–0.02 s |
| `vision` | `ball_pos`, `ball_vel`, `prev_ball_pos` | **0.02–0.08 s** |

* ボール系 3 項は**同じカメラフレームから作る**こと（3 項でレイテンシがずれると
  学習時の前提と食い違う）
* カメラ遅延が 0.08 s を超えるなら、学習側の `_BALL_OBS_DELAY_MAX_S` を上げて再学習

学習時のノイズ（参考。実機で足す必要はない）:
`ball_pos` ±0.07 m / `ball_vel` ±0.5 m/s / `joint_pos` ±0.03 / `joint_vel` ±1.5 /
`projected_gravity` ±0.05 / `base_ang_vel` ±0.2

---

## 6. 動作確認のチェックリスト

1. **入力幅が 55 か** — 別途エクスポートした 58 次元版 ONNX
   (`walk_long_pass_58dim_9500_policy.onnx`) と取り違えない
2. **バッファが古い順か** — `buf[99]` が現在フレーム
3. **3 番目のスロットが足リンク原点基準か** — 胴体基準だと蹴りのタイミングがずれる
4. **`prev_joint_request` が生の action か** — 関節角を入れると壁にぶつかる
5. **関節順が `JOINT_NAMES_K1` か** — 左脚 6 → 右脚 6、各 Pitch/Roll/Yaw/Knee/Ankle 順
6. **`gait_phase_factor_offset` が 0.0 か**
7. **静止状態で出力が暴れないか** — 初回は全フレーム同一値で埋まるので、
   出力も安定しているはず

## 7. 性能の目安（sim 実測、model_9600）

| 指標 | 値 |
|---|---|
| キック成功率 | 0.990 |
| 速度追従率 | 0.884 |
| 方向誤差 | 7.16° |
| 仰角 | 約 21.6° |
| 転倒率 | 2.8% |
| 速度帯 | 3.2–5.0 m/s（距離 5–10 m 相当） |

回り込み方向による差は方向誤差で 0.22°、速度追従で 0.014 と実質対称。
