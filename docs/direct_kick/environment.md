# 環境・制御

数値の完全な原本は [DirectKicking.yaml](source/DirectKicking.yaml)。以下の出典は [固定したコピー元](README.md) に対する参照。

## 基本構成・物理

| 項目 | 仕様・設定値 |
|---|---|
| actor 配置 | 各環境に robot、ball の順で 2 actor。root state は `[N, 2, 13]` |
| 環境数 | YAML は 1、元学習 shell の既定値は 8192 |
| 平面 | `terrain.type = plane`、重力 `[0,0,-9.81]`、Z-up |
| 周期 | physics dt 0.002 s、substeps 1、decimation 10、policy dt 0.02 s |
| PhysX | solver type 1、position iterations 8、velocity iterations 4、CPU threads 4 |
| 接触設定 | contact offset 0.02、rest offset 0、bounce threshold 0.2、max depenetration velocity 1 |
| その他 PhysX | buffer multiplier 5、GPU contact pairs 8388608、subscenes 4、contact collection 1 |
| 環境原点 | 元コードの `env_origins` tensor は全環境で `[0,0,0]`。`create_env` の範囲は `(-5,0,-5)`〜`(5,5,5)`、行あたり `int(sqrt(N))` |
| env spacing | YAML の 5 はこの plane 配置経路で参照されない |
| 地面の物性 | nominal static/dynamic friction 1、restitution 0。摩擦は run ごとに共通倍率 U[0.3,0.9] |
| viewer | pos `[3,-3,2]`、lookat `[15,0,0]`。record_video true、record_interval 5 s、record_env_idx 0 |

`terrain` の trimesh 用設定も YAML に残しているが、現在選択されている地形は plane。
根拠: `envs/base_task.py:24`、`envs/K1/kicking_k1.py:40,268,272`、`envs/K1/direct_kicking.py:162`。

## ロボット・ボール

ロボットはコピー元の [K1_locomotion.urdf](source/K1_locomotion.urdf)。Trunk を base とし、左右の `left_foot_link`、`right_foot_link` を足として使用する。可動関節は脚の 12 関節で、上半身は固定関節。URDF のリンク質量の合計はランダム化前 19.666 kg。各リンクの重心・慣性・関節原点・axis・衝突形状は原本を参照する。移植先に元からある K1 asset と同一であるとは仮定しない。

asset 設定は effort drive (`default_dof_drive_mode=3`)、固定関節 collapse 有効、base 非固定、重力有効、self collision 有効 (`self_collisions=0`)。cylinder→capsule 置換と visual flip は無効。density 0.001、linear/angular damping 0、最大 linear/angular velocity 1000、asset armature 0、thickness 0.01。

ボールは [ball.urdf](source/ball.urdf) の sphere。半径 0.075 m、質量 0.2 kg、対角慣性各 0.00045 kg m²、積慣性 0。nominal friction 1、rolling friction 1、restitution 0。DirectKicking は親クラスの一時的な半径 0.05 を YAML の 0.075 で上書きする。

`ball.contact_offset=0.005` は設定されているが、親の shape property への代入はコメントアウトされている。YAML の値を「適用済みの個別 contact offset」と解釈しない。球の linear/angular damping は 0、最大速度は両方 1000、重力有効。根拠: `kicking_k1.py:40`、`direct_kicking.py:245`。

## 関節順序と制御

Actor 関節位置・速度、action の 12 要素は次の順序。元実装では Gym asset が返した `dof_names` の順序を使う。URDF の可動関節定義は左脚、右脚の順に並ぶ。

```text
Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw,
Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll,
Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw,
Right_Knee_Pitch, Right_Ankle_Pitch, Right_Ankle_Roll
```

| 関節種別 | nominal q [rad] | stiffness | damping | effort [Nm] | velocity 設定 [rad/s] | armature 設定 |
|---|---:|---:|---:|---:|---:|---:|
| Hip_Pitch | -0.26 | 200 | 5 | 30 | 8 | 0.0478125 |
| Hip_Roll | 0 | 200 | 5 | 35 | 12.9 | 0.0339552 |
| Hip_Yaw | 0 | 200 | 5 | 20 | 18 | 0.0282528 |
| Knee_Pitch | 0.52 | 200 | 5 | 40 | 12.5 | 0.095625 |
| Ankle_Pitch | -0.26 | 50 | 1.5 | 20 | 18 | 0.0565056 |
| Ankle_Roll | 0 | 50 | 1.5 | 20 | 18 | 0.0565056 |

| 関節種別 | 左脚 URDF lower〜upper [rad] | 右脚 URDF lower〜upper [rad] |
|---|---|---|
| Hip_Pitch | -3〜2.21 | -3〜2.21 |
| Hip_Roll | -0.4〜1.57 | -1.57〜0.4 |
| Hip_Yaw | -1〜1 | -1〜1 |
| Knee_Pitch | 0〜2.23 | 0〜2.23 |
| Ankle_Pitch | -0.87〜0.345 | -0.87〜0.345 |
| Ankle_Roll | -0.345〜0.345 | -0.345〜0.345 |

制御は各 physics step で次を行う。`kp`,`kd` と関節摩擦 `f` は環境生成時にランダム化済み。

```text
a = clip(policy_sample, -1, 1)
q_target = q_default + 1.0 * a
q_delayed = target history から当該 substep の遅延に応じて取り出す
tau_pd = kp * (q_delayed - q) - kd * dq
friction = min(f, abs(tau_pd)) * sign(tau_pd)
tau = clip(tau_pd - friction, -effort_limit, effort_limit)
```

action target 自体を URDF 位置制限に clamp する処理はこの step 内にはない。報酬で使う `torques` は 10 physics step の平均。`normalization.filter_weight=0.1` により body 線速度・角速度の filtered buffer は `0.1*current + 0.9*previous` で更新される。

YAML の velocity/armature と URDF の値は区別する。親は取得した DOF property 配列へ設定値を書き込むが、この `_create_envs` 内に `set_actor_dof_properties` の呼び出しはない。明示 PD で使用する effort limit tensor の clamp は確認できる一方、velocity/armature 設定が simulator に反映されるとの保証はここではしない。

action delay は reset ごと・環境ごとに整数一様 **1〜25 physics steps**（2〜50 ms、両端含む）。target history は `ceil(25/10)+1=4` フレーム。reset 時には現在の実関節姿勢で全履歴を初期化する。substep 0 で新targetを追加し、遅延 `d`、substep `i=0..9` に対して履歴年齢 `age=max(floor((d-i+9)/10),0)`、index=`(cursor-age)%4` を使う。遅延が 1 policy step を超える場合も以前の target を参照する。根拠: `kicking_k1.py:967`、`direct_kicking.py:177,397,421,1216`、`utils/action_delay.py`。

## Reset とキック目標

1. default pose から開始。`walking_policy_initial_state` は enabled、probability 1、blend `[1,1]` なので指定 walking nominal pose を選ぶ。現在は default pose と同じ角度。
2. 各関節 q に Gaussian σ=0.03 rad を加え、URDF 位置制限へ clamp。dq は Gaussian σ=0.15 rad/s。
3. root XY は nominal `[0,0]` と `env_origins` の和。`init_base_pos_xy` の U[-1,1] は代入箇所がコメントアウトされている。
4. roll/pitch は独立 Gaussian σ=0.04 rad、yaw は U[-0.1,0.1] rad。world XY 速度は Gaussian σ=0.1 m/s、world XYZ 角速度は Gaussian σ=0.15 rad/s。root Z 速度は nominal 0。
5. root 高さは接触を考慮して決定。FK で両足の接触点を現在の q・root 回転に従って変換し、`max(terrain_height + 0.003 - rotated_contact_z)` を root Z とする。固定高さ 0.545 m はこの機能を無効にした場合の値。
6. ボールを下記分布から初期化。物性の restitution と認識系の episode 値も再サンプル。
7. episode・キック・phase・外乱スケジュール・action delay を reset。歩行指令とキック方向をサンプルする。

接触を考慮する reset の足接触点は `(0.119493, ±0.041048, -0.038234)` と `(-0.065899, ±0.041048, -0.038234)` m。通常の接地判定に使う `asset.feet_edge_pos` の `(0.11, ±0.04,-0.02)`、`(-0.07, ±0.04,-0.02)` m とは別。

FK のリンク原点は左右符号 `s=+1/-1` として hip pitch `(0,s*0.096,-0.077)`、hip roll `(0,0,-0.026)`、hip yaw `(0.012,0,-0.0485)`、knee `(-0.014,0,-0.117)`、ankle `(0.00019706,s*0.0002,-0.24519)`。回転軸は順に Y,X,Z,Y,Y,X。

歩行指令 vx,vy,yaw-rate と gait frequency は常に 0、still proportion 1、command curriculum false。キック目標 yaw は **world frame** の U[-1.04,1.04] rad で、方向は `[cos(yaw),sin(yaw)]`。目標距離 U[3,5] m を保持し、reset ボール位置から目標点を作るが、DirectKicking の観測・主要方向報酬では単位方向を使う。command resampling の予定値 2〜4 s は保持されるが、現在の step で `_resample_commands` は呼ばれず、episode 内の指令は固定。

根拠: `kicking_k1.py:675,726,854,870,967`、`direct_kicking.py:569,620,691,731,1216`、`utils/k1_foot_kinematics.py:60`。

## ボールの初期運動

| サンプル | 範囲 |
|---|---|
| incoming/outgoing | incoming 確率 0.5 |
| spawn 距離 | どちらも U[1.5,3.0] m |
| spawn bearing | robot yaw 相対 U[-0.87266463,0.87266463] rad（約 ±50°） |
| closest approach offset | U[-0.25,0.25] m |
| XY speed | U[0,1] m/s |
| kick warmup | 0.1 s = 5 policy steps |

`build_ball_trajectory` で最近接 offset を満たす直線を作る。bearing を b、距離を d、offset を o とすると、radial=`(cos(b),sin(b))`、tangent=`(-sin(b),cos(b))`、k=`o/d`。位置は `d*radial`、速度は `speed*(s*sqrt(1-k²)*radial+k*tangent)` で、s は incoming が -1、outgoing が +1。反転するのは radial 成分で、tangent 成分の符号は維持する。生成時の robot yaw で world へ回転し、robot XY を足して配置。ball Z は地面+0.075、quaternion は `[0,0,0,1]`。rolling 初速は `omega_x=-vy/r`、`omega_y=vx/r`、`omega_z=0`。

根拠: `direct_kicking.py:99,1129,2330`、`utils/ball_trajectory.py`。

## ランダム化の単位

Gaussian の `range: [0,s]` はこのコードでは平均 0・**標準偏差 s**。uniform の `[l,u]` は `l+(u-l)*U[0,1]`。`additive`/`scaling` の全値は原本を維持する。

| 対象 | 分布・操作 | サンプルする時点 |
|---|---|---|
| 地面 static/dynamic friction | 共通倍率 U[0.3,0.9] | run 生成時に 1 回 |
| ball friction | nominal 1 × U[0.9,1.3] | 各環境の生成時 |
| ball rolling friction | nominal 1 × U[0.85,1.4] | 各環境の生成時 |
| ball restitution | U[0,0.7] | 各 ball reset |
| 関節 kp / kd | それぞれ nominal × U[0.8,1.2] | 各環境・各関節、生成時 |
| 明示制御の関節摩擦 | U[0,2] | 各環境・各関節、生成時 |
| 足 shape friction / compliance / restitution | U[0.9,1.1] / U[0.5,1.5] / U[0.1,0.9] | 各足 shape、生成時 |
| base COM xyz | 各成分に U[-0.1,0.1] m を加算 | 各環境、生成時 |
| base mass | nominal × U[0.8,1.2] | 各環境、生成時 |
| 他 body の COM xyz | 各成分に U[-0.005,0.005] m を加算 | 各環境・body、生成時 |
| 他 body の mass | nominal × U[0.98,1.02] | 各環境・body、生成時 |
| 初期姿勢・速度 | 前節の分布 | 各 episode reset |
| action delay | 1〜25 physics steps | 各 episode reset |
| 認識 Q/R、カメラ、遅延、ego bias | [観測仕様](observations.md) | 各 perception reset |

body の質量・COM 変更後は `recomputeInertia=True` で設定。`base_mass_scaled` の特権観測には変換前の乱数 U[0,1] が保存される。

外乱は環境ごとに独立した次回時刻を持つ。world XY 速度加算は間隔 U[4,7] s、各軸 U[-0.3,0.3] m/s。local body force/torque は間隔 U[3,5] s、各軸 Gaussian σ=5 N / 0.5 Nm、設定持続時間 0.2 s（10 policy steps）。時刻の判定には `common_step_counter` を使い、発生ごとに次回間隔を取り直す。`_push_robots` は post-physics で `gymapi.LOCAL_SPACE` を指定して外力 tensor を渡す。

これは設定されたスケジュールと呼出順の記録であり、物理エンジンでの作用時間を実測した結果ではない。終了時と reset 時の外力ゼロ化は元コードで advanced-indexed tensor への `.zero_()` として記述されているため、忠実な実装比較ではその tensor 更新挙動も確認対象になる。

根拠: `kicking_k1.py:40,238,261`、`direct_kicking.py:162,245,332,439,473,507,1234`、`utils/utils.py:5`、`utils/disturbance_schedule.py`。

## 1 policy step の順序

1. DirectKicking が前 step の ball、base、足首位置、足接触を保存。
2. action を clip し、遅延つき PD を 10 physics steps 実行。
3. root、body、接触 tensor を refresh。速度 filter、足状態、episode・ball 運動カウンタを更新。
4. 速度外乱、force/torque 外乱を処理。
5. 終了判定。続いて DirectKicking のキック検出・phase/attempt 更新・報酬計算。
6. 親のキック検出再呼出は当該 step の処理済みフラグで二重計算を抑止。
7. 終了環境を reset。観測を計算し、最後に action・速度・足位置の履歴を更新。
8. `(obs, reward, done, extras)` を返す。終了 step の観測は reset 後の観測。

終了・報酬のより詳しい条件は [報酬仕様](rewards.md)。`extras` は `privileged_obs`、`time_outs`、`rew_terms`、`post_kick_phase_target` を運ぶ。根拠: `direct_kicking.py:1313,1661,1783,1860`、`kicking_k1.py:967`。
