# 報酬・キック検出・終了条件

出典は [固定したコピー元](README.md)。以下で `D` は `envs/K1/direct_kicking.py`、`K` は `envs/K1/kicking_k1.py`、`U` は `utils/kick_foot_symmetry.py`、`O` は `utils/direct_kicking_outcome.py` を指す。

## 報酬の合成

policy dt は0.02秒。通常項は `raw * YAML係数 * 0.02` を加算する。**初回キックの `kick_direction` だけは dt を掛けず別加算**。`only_positive_rewards=false`。`ball_rolling_scale` は現在の YAML にない。係数0の項は関数評価をスキップする。

通常報酬は終了 step にも算出する。`kick_direction` は `~reset_buf` を掛けるため、終了 step には加算しない。通常項の評価順は名前のソート順。

記号: `n` は episode step、`V` は valid kick のラッチ、`P` は kick attempt pending、`I(condition)` は0/1、`[x]+ = max(x,0)`、`wrap` は `[-π,π)` への角度正規化。`a` は clip 後 action、`tau` は10 physics steps の平均出力 torque。

根拠: `K:510,1180`、`D:1860`。

## 通常報酬の全項目

以下の係数は **dt 乗算前の YAML 値**。角度、速度、距離の単位は rad、m/s、m。

| 項目 | 係数 | raw の定義 | 実装 |
|---|---:|---|---|
| `survival` | 0.3 | 1 | K:1307 |
| `fall` | -1000 | `I(fall_buf)`。実効一回値 -20 | K:1311 |
| `base_height` | -200 | `(base高さ - terrain高さ - 0.54)²` | K:1340 |
| `orientation` | -20 | projected gravity の x²+y² | K:1357 |
| `torques` | -0.0002 | 全関節の `sum(tau²)` | K:1361 |
| `ankle_torques` | -0.001 | 左右 Ankle Pitch/Roll 4関節の `sum(tau²)` | D:549,1889 |
| `torque_tiredness` | -0.01 | `sum(min((tau/torque_limit)²,1))` | K:1407 |
| `power` | -0.002 | `sum([tau*dq]+)`。絶対値ではない | K:1411 |
| `lin_vel_z` | -2 | filtered body linear velocity z² | K:1349 |
| `ang_vel_xy` | -0.1 | filter 前 body angular velocity x²+y² | K:1353 |
| `dof_vel` | -0.05 | `sum(dq²)` | K:1365 |
| `dof_acc` | -0.0000001 | `sum(((dq_prev-dq)/dt)²)` | K:1369 |
| `root_acc` | -0.0001 | world root 線速度3+角速度3の `sum(((v_prev-v)/dt)²)` | K:1373 |
| `action_rate` | -1.5 | `sum((a_prev-a)²)` | K:1377 |
| `action_second_difference` | -0.80 | `sum((a-2*a_prev+a_prevprev)²)`。評価後に `a_prevprev←a_prev` | D:1894 |
| `dof_pos_limits` | -1 | URDF 範囲外の関節数。超過量ではない。soft limit=1 | K:1381 |
| `feet_slip` | -1 | `I(n>1)*sum_feet(contact * norm((foot_prev-foot)/dt)²)`。XYZ全成分 | K:1415 |
| `feet_airborne` | -4.5 | `I(両足非接地)*I(!V)` | D:2208 |
| `edge_only_support` | -4 | 下記の端部支持条件が4 step以上継続する足があれば1 | D:2251 |
| `feet_yaw_diff` | -1 | `wrap(yaw_right-yaw_left)²` | K:1442 |
| `feet_yaw_mean` | -3 | `wrap(base_yaw-foot_yaw_mean)²`。mean は下記 | K:1445 |
| `feet_roll` | -3 | `wrap(roll_left)²+wrap(roll_right)²` | K:1439 |
| `swing_feet_pitch` | -1 | `sum_feet([abs(wrap(pitch))-0.17453293]+² * I(足がCoM前方または同位置) * I(!V&&!P))` | D:2303 |
| `feet_distance` | -1 | `clip(0.192-abs(cos(base_yaw)*delta_foot_y-sin(base_yaw)*delta_foot_x),0,0.1)` | K:1449 |
| `parallel_step` | -2 | `I(n>1&&!V)*max_feet((([abs(v_foot_local.y)-0.05]+/0.25)²)*I(片脚支持の遊脚 && v_foot_local.x>0))` | D:1904 |
| `body_approach_ball` | 15 | `I(n>1&&!V)*(J_prev-J_now)/0.05`。J は下記 | D:2089 |
| `kicking_foot_approach_ball_stationary` | 5 | `I(n>1&&!V)*(min_foot norm(ankle_prev-ball_now)_XY - min_foot norm(ankle_now-ball_now)_XY)/0.05` | D:2135 |
| `body_heading_to_ball` | -1 | `I(!V)*(1-clip(cos_alignment,-1,1))²`。base 前方と robot→ball XY | D:2163 |
| `body_alignment_for_kick` | 1 | `clip(exp((alignment-1)/0.5),0,1)`。base 前方と固定 target 単位方向の内積 | D:2057 |
| `body_angle` | 0.1 | `1/(0.1+(wrap(roll)²+wrap(pitch)²)²)-1`。直立時 raw=9 | K:1616 |
| `kicking_foot_height` | -2 | `I(!V)*max_feet(([foot_height-0.18]+/0.10)²)` | D:2182 |
| `failed_kick_attempt` | -10 | attempt失敗イベントの1 stepだけ1。実効一回値 -0.2 | D:2005 |
| `post_kick_walking_pose` | 1 | `I(V)*exp(-mean((q-q_walking)²)/0.25²)` | D:2009 |
| `waiting` | -0.5 | `I(!V)*clip(n*dt/2,0,1)²` | K:1654 |
| `tracking_lin_vel_x` | 0 | 無効 | K:1314 |
| `tracking_lin_vel_y` | 0 | 無効 | K:1318 |
| `tracking_ang_vel` | 0 | 無効 | K:1322 |
| `ball_velocity_target_direction` | 0 | 現設定では評価をスキップ。関数は親の報酬×valid kick mask | D:2048 |
| `ball_acceleration` | 0 | 現設定では評価をスキップ。関数は親の報酬×valid kick mask | D:2054 |

全39項は非ゼロ34項・ゼロ5項。`collision` は親に関数と対象リンク設定があるが、現在の `rewards.scales` に含まれない。

足の接地は接触力 threshold ではなく、`asset.feet_edge_pos` の4点を足姿勢で変換し、1点でも地面から **0.01 m未満**なら true。`feet_yaw_mean=(yaw_left+yaw_right)/2 + π*I(abs(yaw_right-yaw_left)>π)`。

`swing_feet_pitch` は名前に swing とあるが、接地/遊脚を条件に含めない。足が全身 CoM より前方かは base yaw frame の前後成分で判断する。CoM は各 body の位置とランダム化済み質量の加重平均。`kicking_foot_approach_ball_stationary` も現 `progress` モードではボール静止を条件に含めない。

`body_approach_ball` の `J` は、body-local ball XY と左右 nominal strike point `(0.185,±0.096)` の最短距離。前回・今回の base 姿勢で **同じ現在 ball 位置**を変換して比較する。ball が勝手に近づいた量をそのまま body の進捗にしない。foot approach も同じ現在 ball に対して前回・今回足首を比較する。足首位置は rigid-body の foot COM 位置から、回転済み local COM offset を引いて復元する。

`parallel_step` の足速度は world 位置差分から base world XY 線速度と `yaw_rate×[-offset_y,offset_x]` を引いて base yaw frame へ変換する。`body_heading_to_ball` の cosine の分母はノルム積+1e-6。`body_alignment_for_kick` は base 前方XYのノルム下限1e-6で正規化し、kick後も有効。

端部支持 `edge_only_support` は足ごとに次の条件を連続カウントする。

- toe（local x>0）と heel（local x<0）の接触が XOR。各グループで1点以上が terrain+0.01 m未満なら接触。
- 正の垂直接触力が `0.10 * robot総質量 * 9.81` 以上。
- kick前、かつ attempt pending ではない。

条件が切れれば0へ戻し、`ceil(0.08/0.02)=4` step以上でraw=1。任意の足が満たせばよく、左右の和ではない。

`previous_previous_actions` は生成時0だが、確認した episode reset 処理には action 履歴を0へ戻す処理がない。移植で履歴resetを追加すると、この元仕様から変わる。

根拠: `K:675,1004,1078,1126`、`D:363,1216,1894,1904,2089,2135,2213,2285,2294`、`U:110,125,164,171,203`、`utils/action_smoothness.py:6`。

## 物理キックと初回方向報酬

各足の候補条件:

1. 現在/前回の足↔ball **3D距離の小さい方が 0.23 m以下**。
2. world 足差分速度 `(foot_now-foot_prev)/dt` を前回足→ball方向へ射影し、0.2 m/s以上。方向の分母は距離+1e-6。base相対速度ではない。

どちらかの足が候補、かつ ball world XY の速度変化ノルムが **0.5 m/s以上**、かつ reset 後5 policy stepsの warmup 終了 (`episode_step>=block_until_step`) で物理キックと判定する。目標方向は検出条件に使わない。`valid_kick_buf` は一度成立するとラッチ。

初回のみ `new_valid_kick` と `first_valid_kick_step` を記録する。候補が左右両方なら現在/前回距離の小さい足を選び、同距離なら index0。選択した足を phase 教師用に保存する。

```text
kick_direction =
  10 * exp(4*(cos(resulting_ball_velocity_xy, target_direction_xy)-1))
     * sigmoid(10*(norm(ball_velocity_xy_now-ball_velocity_xy_prev)-0.5))
     * I(new_valid_kick) * I(!reset_buf)
```

cosine は `[-1,1]` に clamp。結果 ball 速度または target ノルムが1e-6以下なら方向報酬0。評価するのは**結果速度の方向**であり、速度差の方向ではない。速度差ノルムは sigmoid の倍率に使用する。距離は評価せず、dtは掛けない。

固定 target direction は reset 時の `kick_target_pos_world-kick_start_ball_pos` を正規化したもの。現在 ball→target の向きではない。
根拠: `D:1783,1860,2086`、`K:1294`、`U:6,63`、`O:22`。

## キック試行と失敗

kick前、n>1、左右どちらかの足が次を満たすと attempt candidate。

- 本人が非接地、反対足が接地している。
- base yaw frame の足位置 x>=0.12 m。
- world 足差分速度から base world 線速度を引き、base yaw frame へ回した前方速度>=1.0 m/s。
- base world XY速度ノルム<=0.35 m/s。

この速度定義では `parallel_step` と違い yaw 回転速度を引かない。candidate の false→true の立上がりで pending にし、deadline=`n+ceil(0.30/dt)=n+15`。期限までの `new_valid_kick` を成功優先で処理し pending を解除する。deadline と同stepのkickも成功。

pending で `n>=deadline` になれば1 stepだけ失敗を返す。candidate がtrueのままでは再試行を開始せず、新しい立上がりが必要。失敗イベントと同stepに次の試行は始めない。

pending 中は `edge_only_support` と `swing_feet_pitch` を止める。他のkick前の接近・高さ・両足離地報酬は継続する。
根拠: `D:829,1947,1983,2005`、`U:230`。

## Post-kick phase 教師

物理kickで選んだ足が、前stepまたは以後に一度非接地となり、その足が接地したら `post_kick_phase_target=true` をラッチする。キックイベント自体と着地後phaseは別の信号。

`post_kick_walking_pose` 報酬はこのphaseではなく valid kick で直ちに有効になる。目標 walking q は Hip Pitch=-0.26、Knee Pitch=0.52、Ankle Pitch=-0.26、他0。
根拠: `D:1848,2009`、`O:137,193`。

## 終了条件

physics・状態更新・episode counter増加・外乱の後に**終了判定を先に行い**、続いて当該stepのkick検出と報酬を計算する。このため終了判定で参照する valid kick は前stepまでの状態。

以下のORでepisodeを終了する。

| 条件 | 判定 |
|---|---|
| 低姿勢 | base の地面からの高さ <0.45 m |
| 過大root速度 | world 線速度3+角速度3の**二乗和 >50** |
| 指定body接触 | 対象 contact force norm >1。ただし現 `terminate_contacts_on=[]` |
| episode時間 | `n>ceil(17/0.02)=850`。通常は851 step目 |
| 正方向ball速度の継続 | `min_ball_vel_buf>ceil(2/0.02)=100` |
| ball静止継続 | still counter >2 s |
| ball運動継続 | moving counter >5 s |
| 初回kick後時間 | `V && first_step>=0 && n-first_step>=100`（2秒） |

ball のカウンタは次の定義。

- `min_ball_vel_buf` は ball world **vx>0.1** のとき+1、他は0。target方向やXY速度normではない。
- moving は ball world **3D速度norm>0.1**。movingなら still=0、moving+=dt。そうでなければ逆。
- 終了判定の直前、まだ valid kick でない環境の上記3 counter はすべて0にする。初期 ball が移動・静止しているだけではこれらの時間終了に達しない。初回kick検出stepも先に0に戻る。

`fall_buf` は低姿勢・過大速度・指定接触のみのOR。時間終了はfallではない。
`time_out_buf` は親が設定する command resampling 時刻のtimeoutを上書きし、実際の17秒超だけにする。post-kick終了だけが新たに成立した環境は `post_kick_terminal_buf=true`、timeout=false。既に別条件で終了していればその判定を保持する。

YAML にある `ball_rest_speed_threshold_after_kick=0.02` と `min_steps_travel_after_kick=10` は、この終了判定で参照される条件ではない。

報酬計算後、終了環境をresetして観測を計算するため、返す観測は新episodeのもの。timeout時の学習処理は [モデル・学習仕様](training.md#timeoutgae) を参照。

根拠: `K:967,1015,1024,1148`、`D:1313,2021`、`O:172`。

## 集計値

`env_successes` は親の `min_ball_vel_buf>100` の環境数を加算する指標であり、単純な physical kick 回数や方向精度ではない。`env_falling` は fall の件数、`env_resets` は `_reset_idx` の対象数で初期の明示resetも含む。`extras.rew_terms` は通常項と別加算の `kick_direction` の値を持つ。

根拠: `K:675,1174`、`D:1886`。
