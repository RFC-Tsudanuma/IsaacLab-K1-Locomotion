# K1 Kick Analysis and Tuning

## 1. Current problem analysis

今回の調整対象は `source\isaaclab_k1_locomotion\...\kick\kick_env_cfg.py` と `...\kick\mdp\rewards.py`。  
既存の kick 学習では、ボール接触や前進速度の信号は出る一方で、`reward_kick_success` が 0 に張り付きやすく、post-kick 回復報酬や yaw 抑制報酬も十分に働きにくかった。

今回の修正では次を実施した。

1. `kick_success` を「接触済み + 前進速度または前進距離」で判定するよう変更
2. yaw 抑制/回復報酬を追加
3. post-kick の default pose 回帰・左右対称・足位置整列を強化
4. `reward_kick_success` の重みを引き上げ、ログ上でも死に報酬になりにくくした
5. post-kick termination は即終了ではなく、引き続き recovery window 後にのみ発火する構成を維持

## 2. Yaw rotation analysis

Yaw 回転問題は、並進ではなく `root_ang_vel_z` / heading の崩れが支配的だった。  
短期 smoke 学習でも `Metrics/base_velocity/error_vel_xy` は小さく、`Metrics/base_velocity/error_vel_yaw` が相対的に大きい傾向だった。

そのため以下を追加した。

- `penalty_yaw_rate`
- `penalty_post_kick_yaw`
- `reward_yaw_stabilization`
- `reward_heading_recovery`

実装上は `root_ang_vel_w[:, 2]` と初期 heading との差を使って評価している。  
結果として、yaw はまだ残るが、reward/penalty として明示的に最適化対象へ乗るようになった。

## 3. Kick Success analysis

原因は成功条件が厳しすぎ、かつ first-hit 型の成功報酬がログ上で見えにくかったことだった。

変更前の問題:

- ボール接触と速度増加が出ても success 判定へ届きにくい
- `reward_kick_success` は 1 step のみなので、ログ集計で 0.0000 に見えやすい

変更後:

- `KICK_SUCCESS_DISTANCE = 0.15`
- `KICK_SUCCESS_SPEED = 0.40`
- success は `had_ball_contact` を前提に
  - 前進距離達成
  - もしくは前進速度達成
  - もしくは最小距離 + 速度達成
- `reward_kick_success.weight = 96.0`

短期学習では `reward_kick_success` が `0.1257`, `0.0800` まで上がる iteration を確認した。  
これで success 報酬は dead reward ではなくなった。

## 4. Dead reward analysis

### 改善された項目

- `reward_kick_success`
- `reward_post_kick_stability`
- `reward_recover_to_stand`
- `reward_symmetric_posture`
- `reward_return_to_default_pose`
- `reward_joint_symmetry`
- `reward_double_support`
- `reward_feet_alignment`
- `reward_yaw_stabilization`
- `reward_heading_recovery`

短期 smoke でも上記は非 0 を確認した。

### まだ弱い項目

- `reward_post_kick_balance`
- `Episode_Termination/ball_travel_distance`
- `Episode_Termination/post_kick_settle_time`

これらは「配線不良」よりも「短期 smoke 中に十分な安定 recovery window へ到達しづらい」影響が大きい。  
現状では base height / fall 系 termination が先に来やすい。

## 5. Termination analysis

現行 termination は以下。

| Name | Threshold / condition | Note |
| --- | --- | --- |
| `time_out` | `episode_length_s = 7.0` | 通常 timeout |
| `no_kick_timeout` | timeout かつ no contact | no-kick 局所解を崩す |
| `base_height` | `minimum_height = 0.40` | 転倒 |
| `trunk_contact` | trunk 接触 | 転倒 |
| `ball_travel_distance` | `>= 1.5 m` かつ success 後 `3.0 s` 経過 | 長距離移動確認用 |
| `post_kick_settle_time` | success 後 `3.0 s` | 回復観察用 |

重要なのは、`ball_travel_distance` / `post_kick_settle_time` が success 直後には発火しないこと。  
短期 smoke で両者が 0 のままだったのは、即終了していないことを意味しており、以前の「成功直後に終わる」問題は再発していない。

## 6. Actor observation analysis

`kick` task の observation group は `K1PolicyCfg` / `K1CriticCfg` をそのまま使っており、flat task 互換を維持している。

### Actor (49 dims)

- `base_ang_vel`
- `projected_gravity`
- `velocity_commands`
- `joint_pos`
- `joint_vel`
- `actions`
- `gait_phase`

### Critic (52 dims)

Actor の内容に加えて:

- `base_lin_vel`

### 現状の結論

- Ball observation: **含まれない**
- Relative Ball Position: **含まれない**
- Ball Velocity: **含まれない**

これは `k1_flat` checkpoint strict load 互換のため意図的。  
`logs` 上でも actor/critic はそれぞれ `49` / `52` 入力のまま。

### 将来の ball-aware policy 案

互換性を壊さず拡張するなら次の 2 段構えが安全。

1. 現行 `Isaac-K1-Kick-v0` は flat-compatible actor のまま維持
2. 別 task で ball-aware 観測を actor に追加し、partial load か adapter 層で移行

この分離なら既存 resume 経路は壊れない。

## 7. Recommended reward weights

現行の主要 weight は以下。

| Reward / penalty | Weight |
| --- | ---: |
| `first_ball_contact_bonus` | 60.0 |
| `reward_ball_contact` | 20.0 |
| `reward_ball_speed_increase` | 10.0 |
| `reward_ball_forward_velocity` | 8.0 |
| `reward_kick_distance_progress` | 2.5 |
| `reward_kick_speed_progress` | 1.5 |
| `reward_kick_success` | 96.0 |
| `reward_post_kick_stability` | 2.5 |
| `reward_recover_to_stand` | 4.0 |
| `reward_return_to_default_pose` | 7.0 |
| `reward_joint_symmetry` | 4.0 |
| `reward_double_support` | 3.0 |
| `reward_feet_alignment` | 2.0 |
| `reward_post_kick_balance` | 4.0 |
| `reward_post_kick_settling` | 3.0 |
| `reward_yaw_stabilization` | 2.5 |
| `reward_heading_recovery` | 2.0 |
| `reward_stand_still` | 0.6 |
| `reward_base_position_hold` | 0.8 |
| `penalty_yaw_rate` | -3.5 |
| `penalty_post_kick_yaw` | -2.0 |
| `penalty_no_kick_timeout` | -120.0 |
| `penalty_fall` | -80.0 |

推奨方針:

- pre-kick は `ball_contact / speed / forward_velocity / kick_success` を最優先
- post-kick は `return_to_default_pose / joint_symmetry / double_support` を主軸
- yaw 問題が残る場合は `penalty_yaw_rate` をさらに強め、`reward_heading_recovery` を少し上げる

## 8. Recommended termination thresholds

| Termination | Current value | Recommendation |
| --- | ---: | --- |
| `KICK_SUCCESS_DISTANCE` | 0.15 m | 妥当。これ以上上げると success が死にやすい |
| `KICK_SUCCESS_SPEED` | 0.40 m/s | 妥当。探索初期でも届きやすい |
| `BALL_TRAVEL_DISTANCE_THRESHOLD` | 1.5 m | GUI 観察重視なら維持 |
| `POST_KICK_RECOVERY_TIME_S` | 3.0 s | recovery 観察用として維持 |
| `base_height minimum_height` | 0.40 m | 現状維持。過度に緩めると転倒を許しすぎる |

短期学習では `ball_travel_distance` / `post_kick_settle_time` は低頻度だった。  
この 2 つを増やしたい場合は threshold を下げるより、先に転倒率を下げる方が筋が良い。

## 9. Changed files

- `source\isaaclab_k1_locomotion\isaaclab_k1_locomotion\tasks\manager_based\locomotion\kick\kick_env_cfg.py`
- `source\isaaclab_k1_locomotion\isaaclab_k1_locomotion\tasks\manager_based\locomotion\kick\mdp\rewards.py`
- `docs\K1_KICK_ANALYSIS_AND_TUNING.md`

## Validation summary

確認した内容:

1. kick task は import / compile 可能
2. `Isaac-K1-Kick-v0` は短期学習で起動可能
3. `reward_kick_success` は非 0 を確認
4. yaw 関連 reward / penalty は非 0 を確認
5. post-kick recovery reward 群は複数項目で非 0 を確認
6. `k1_flat` resume 互換は維持

残課題:

- yaw error はまだ高めで、追加学習での収束確認が必要
- `reward_post_kick_balance` は依然として弱い
- `ball_travel_distance` / `post_kick_settle_time` は長め学習または GUI 観察で再確認したい
