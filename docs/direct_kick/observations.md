# Actor 観測・Critic 特権観測・認識

出典は [コピー元コミット](README.md) の `envs/K1/direct_kicking.py`。以下の slice は Python と同じ **0 起点・終端を含まない**表記。

## Actor: 132 次元

| slice | 次元 | 値 | 倍率・ノイズ |
|---|---:|---|---|
| `0:3` | 3 | body frame の projected gravity（world `[0,0,-1]` を逆回転） | ×1、Gaussian σ=0.01 |
| `3:6` | 3 | body frame の base angular velocity | ×1、Gaussian σ=0.15。z に episode yaw-rate bias/drift を追加 |
| `6:9` | 3 | command vx,vy,yaw-rate | ×1。現設定は常に 0 |
| `9:11` | 2 | `cos(2π*gait_process)`, `sin(2π*gait_process)` | `gait_frequency>1e-8` の mask。現設定は両方 0 |
| `11:23` | 12 | `q-q_default` | Gaussian σ=0.01 rad を加え ×1 |
| `23:35` | 12 | dq | Gaussian σ=0.1 rad/s を加え **×0.1** |
| `35:47` | 12 | `self.actions` | 直前に環境へ渡して clip した action。遅延後の PD target ではない |
| `47:125` | 78 | 13 horizon × 6 要素の belief token | 下記 |
| `125:127` | 2 | belief から計算した現在 ball 相対 vx,vy | ×1 |
| `127:130` | 3 | measurement age、updated、valid | age は 0.5 s で正規化し `[0,1]` へ clip、flag は 0/1 |
| `130:132` | 2 | 固定 kick target の単位方向を観測 base yaw frame へ回した XY | 距離を含まない |

関節順は [環境仕様](environment.md#関節順序と制御) の左脚6→右脚6。
Actor に base linear velocity の独立したスロットはない。ノイズを加えた yaw frame の base linear velocity は、belief の相対速度と未来予測の計算に使用する。
ball 真値の位置・速度を Actor の ball feature へ直接連結しない。

根拠: `direct_kicking.py:1661,1544`、`utils/direct_kicking_observation.py:8`。

各 horizon の token は次の順。

```text
[relative_x, relative_y, log_std_x, log_std_y, rho_xy, normalized_horizon]
horizons [s] = [0, .05, .10, .15, .20, .30, .50, .75, 1, 1.5, 2, 2.5, 3]
token i の actor slice = [47 + 6*i : 47 + 6*(i+1)]
normalized_horizon = h / 3
log_std = 0.25 * clip(log(std / 0.01), -4, 4)
rho_xy = clip(cov_xy / (std_x * std_y), -0.999, 0.999)
```

位置は予測時刻の robot base-yaw frame、倍率 1。std を作るときは分散の下限を `1e-12` として平方根を取る。log_std の出力範囲は `[-1,1]`。

無効な belief は各 token を `[0,0,1,1,0,h/3]`、相対速度を `[0,0]` とする。status は `[clip(age/.5,0,1),updated,valid]`、reset 直後は `[1,0,0]`。target direction は別枠のため無効化しない。

## Critic: Actor 観測 + 20 次元の特権観測

元 API は `model.est_value(actor_obs, privileged_obs)` として **132 と 20 を別引数**で受け取る。Critic は Actor と別の horizon encoder を使い、118 次元に符号化した観測と 20 次元を連結する。入力情報量は 152 次元だが、実際の value MLP 入力は 138 次元。

| privileged slice | 次元 | 値 |
|---|---:|---|
| `0:4` | 4 | base COM x,y,z と mass ランダム化の **変換前の一様乱数** |
| `4:7` | 3 | body frame base linear velocity + 独立 Gaussian σ=0.1、倍率1 |
| `7:8` | 1 | base Z − terrain height + Gaussian σ=0.02 m |
| `8:11` | 3 | `pushing_forces[:,0,:]` ×0.1 |
| `11:14` | 3 | `pushing_torques[:,0,:]` ×0.5 |
| `14:16` | 2 | ball rigid-body state の真値 world vx,vy |
| `16:20` | 4 | 左足 world x,y、右足 world x,y（`feet_pos` の値） |

`base_mass_scaled` という名前でも、物理的な COM・mass やその倍率ではない。`apply_randomization(..., return_noise=True)` は `noise_val` ではなく元の乱数 `noise` を返す。現設定では 4 要素とも U[0,1]。特権観測の base velocity noise は Actor の belief 生成時と別にサンプルされ、ego-motion bias/drift の追加もない。

force/torque は local-space 適用に使う外乱 tensor の値。足位置は body state の位置をそのまま用い、base からの相対位置や足首 FK 位置ではない。環境原点が異なる移植先では座標の対応を確認する必要がある。

根拠: `direct_kicking.py:1750`、`kicking_k1.py:238,272,1126`、`utils/utils.py:5`、`utils/models/DirectKickingAC.py:131`。

## 認識の時間軸・CVKF

制御周期 `Δ=0.02 s`。環境ごとに世界座標の state `[x,y,vx,vy]` と 4×4 covariance を持つ。デフォルト dtype は float32。フィルタの時間軸は現在から固定 latency `L` だけ過去。

```text
F(t) = [[1,0,t,0], [0,1,0,t], [0,0,1,0], [0,0,0,1]]
H    = [[1,0,0,0], [0,1,0,0]]
Q(t) = sigma_a² * [[t⁴/4,0,t³/2,0], [0,t⁴/4,0,t³/2],
                   [t³/2,0,t²,0], [0,t³/2,0,t²]]
R    = sigma_filter² * I₂

predict: x ← F(Δ)x
         P ← F(Δ)PF(Δ)ᵀ + Q(Δ)
update:  innovation = z - Hx
         S = HPHᵀ + R
         NIS = innovationᵀ S⁻¹ innovation
```

観測と NIS が finite、かつ `NIS <= 13.82` の場合だけ更新する。棄却時は predict 後の状態を保つ。gain は `K=PHᵀS⁻¹`。共分散は Joseph 形式 `(I-KH)P(I-KH)ᵀ+KRKᵀ`、その後 `(P+Pᵀ)/2` で対称化。

初回観測では位置を測定値、速度をゼロにし、`P=diag(sigma_filter²,sigma_filter²,0.5²,0.5²)` で初期化する。真値速度は使わない。

任意 horizon の共分散予測は、単に `Q(t)` を一度足すのではなく、制御周期の離散ノイズを累積する。`n=floor(t/Δ+1e-6)`, `r=max(t-nΔ,0)` とすると各 x/vx、y/vy の組について次になる。

```text
A = sigma_a² * Δ⁴ * n*(4*n²-1)/12
B = sigma_a² * Δ³ * n²/2
C = sigma_a² * Δ² * n
Q_pp = A + 2*r*B + r²*C + sigma_a²*r⁴/4
Q_pv = B + r*C + sigma_a²*r³/2
Q_vv = C + sigma_a²*r²
P_forecast = F(t) P F(t)ᵀ + Q_accumulated(t)
```

根拠: `utils/ball_kalman_filter.py:37,89,128,156,277,311,447`。

## Episode ごとの認識ランダム化

| 値 | サンプル・計算 |
|---|---|
| 実測生成の基準 std | `sigma_sensor_base ~ U[0.005,0.015] m` |
| filter 測定の基準 std | `sigma_sensor_base * U[0.8,1.2]`（std の倍率） |
| process acceleration std | `0.8 * U[0.8,1.2] * sqrt(ball_rolling_friction_scale * ground_friction_scale)` m/s² |
| FPS | U[25,30] |
| latency | U[0,0.06] s をサンプルして `round(latency/Δ)`。有効値は 0,0.02,0.04,0.06 s |
| history length | `ceil(0.06/Δ)+2 = 5`、latency step は history length−2 以下 |
| camera timer 初期値 | `L + U[0,1]/FPS` |
| age 初期値 | 0.5 s |
| filter 初期値 | state=0、P=0、initialized=false |

latency の離散4値は整数一様サンプルではない。episode 中は固定し、reset 時に取り直す。history の全スロットには reset 時の真値 ball XY と観測 base XY/yaw を入れるが、有効履歴数は0にし、filter は invalidate する。これによって history 内の真値がそのまま有効な belief にはならない。

根拠: `direct_kicking.py:868,1234`。

## カメラ・視野・欠測・外れ値

1. policy step ごとに history cursor を進め、ball 真値位置と観測 base 姿勢を書き込む。reset 直後以外は有効履歴数、初期化済み filter、measurement age、camera timer を進める。
2. `!just_reset && valid_history_steps>=latency_steps && camera_timer<=0` の環境でフレームが発生。
3. timer に `(1+0.15*(2U-1))/FPS` を加算し、最低 `0.25*Δ=0.005 s` に clamp。1環境・1 policy step で最大1フレーム。
4. `(cursor-latency_steps)%5` の撮影状態を取り出す。
5. dropout burst を処理。残数があれば欠測して残数−1、なければ確率0.05で長さ1〜3フレームの整数一様 burst を開始する。開始フレームも欠測。
6. 撮影時点の観測 base yaw frame で、`abs(atan2(y,x)) < 3.49065850/2`、`0.05 < distance < 6` のとき可視。約200°の水平 FOV で、境界は strict 不等号。垂直 FOV・roll/pitch 判定はない。
7. 距離倍率 `s(d)=clip(d/1.0,1,8)` を実測 std と filter std に掛け、相対 XY に Gaussian noise を加える。
8. 利用可能フレームの確率0.02を外れ値候補にする。filter 未初期化ならフレームを棄却。初期化済みなら長さ U[0.10,0.30] m、角度 U[0,2π] の変位を加え、通常の NIS gate に通す。
9. ノイズ込み相対位置を同じ撮影時の観測 base 姿勢で world に戻し、CVKF を初期化または更新。

dropout の抽選は可視判定より先。位置ノイズは撮影距離に依存し、filter 用 R と実際の測定ノイズは calibration scale 分だけ異なる。初期化・NIS 採用があったときだけ updated=true、age=L になる。フレームが届いたことと、filter が採用したことは別。

`belief_valid = initialized && age<=0.5`。0.5 s 超の欠測で Actor は無効表現になるが、filter 自体は invalidate せず predict を継続する。次回の位置観測は既存 track に NIS gate を適用する。

明示的な `reset()` で観測を計算するときは知覚時間を進めず `perception_just_reset` を解除する。通常 step 中の自動 reset は、その step の知覚更新が reset 直後を判定して進行を抑止する。

根拠: `direct_kicking.py:1329,1661`、`utils/ball_visibility.py:6`、`utils/ball_kalman_filter.py:6`。

## Ego-motion と予測座標

| 成分 | episode bias の std | step ごとの drift std / √s | white noise std |
|---|---:|---:|---:|
| base yaw frame XY 速度 | 0.03 m/s | 0.01 m/s/√s | 0.10 m/s |
| body angular z | 0.05 rad/s | 0.02 rad/s/√s | 0.15 rad/s |
| base world XY 位置 | 0 | なし | 0 |
| base yaw 角 | 0 | なし | 0 |

drift は `drift += N(0,1)*std*sqrt(Δ)`、reset で0。base XY/yaw の観測は真値+bias+white noise。位置/yaw ノイズはコードに機能があるが、現 YAML は明示的に0とする。キー省略時は位置/yaw の white noise 各0.005、bias各0.01なので、省略で置換しない。

base world XY 速度を観測 yaw で回し、Gaussian noise と velocity bias/drift を加える。yaw-rate は body angular z に noise と bias/drift を加えたものを使用し、Euler yaw の時間微分へ変換しない。同じ noisy angular velocity を Actor の `3:6` と belief 予測に使う。

ball は遅延 filter の時刻から `L+h` 先、base は現在の観測から **h だけ**先を予測する。観測 body twist `v=(vx,vy), omega` に対し、

```text
a = sin(omega*h)/omega
b = (cos(omega*h)-1)/omega
d_body = (a*vx+b*vy, -b*vx+a*vy)
abs(omega)<1e-6 のとき a=h, b=0
p_base(h) = p_base(0) + R_world_from_yaw(psi_0)*d_body
psi(h) = psi_0 + omega*h
r(h) = R_yaw_from_world(psi(h)) * (p_ball(L+h)-p_base(h))
```

共分散は ball XY covariance の回転に ego-motion 分散を加える。各 σ は表の white noise(n)、episode bias(b)、drift(d)。

```text
V_p(h)   = sigma_p,n² + sigma_p,b²
           + h²*(sigma_v,n²+sigma_v,b²) + h³*sigma_v,d²/3
V_psi(h) = sigma_psi,n² + sigma_psi,b²
           + h²*(sigma_omega,n²+sigma_omega,b²) + h³*sigma_omega,d²/3
g = (-r_y,r_x)
P_relative = R*P_ball,xy*Rᵀ + V_p*I₂ + V_psi*g*gᵀ
```

現在の相対速度は `R(psi_0)*v_ball - v_base - omega×r(0)`、`omega×r=(-omega*r_y,omega*r_x)`。horizon 0 の未スケール相対位置を使う。正規化済み horizon token の位置をそのまま速度計算へ流用しない。

根拠: `direct_kicking.py:991,1515,1544,1683`、`utils/ego_motion.py:8,60,105`。
