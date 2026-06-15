# K1 Kick Reward Redesign

## 現状の問題点

初期版の Kick 環境には次の問題があった。

- 学習開始時に立位・姿勢安定・歩行まで再学習しやすく、キック学習へ到達しにくい
- ボールへ近づく報酬が強く、`ball chasing` が発生しやすい
- ball 観測を actor/critic に追加していたため、`k1_flat` 学習済み checkpoint と shape 不整合になり、RSL-RL の strict load で resume できない

## 修正した報酬

`Isaac-K1-Kick-v0` の報酬は、歩いて追いかけるより「その場で立って蹴る」ことを優先する形へ変更した。

さらに、立っているだけで高報酬になる局所解を防ぐため、**キックしない episode には大きな負報酬**を与える構成へ変更した。

### 正の報酬

- `first_ball_contact_bonus`: 最初の足ボール接触に大きな bonus
- `reward_stand_still`: ベース線速度・yaw 角速度が小さいほど加点
- `reward_base_position_hold`: 初期 base XY からのずれが小さいほど加点
- `reward_ball_contact`: 足でボールへ接触したとき加点
- `reward_ball_speed_increase`: ボール速度の増加を加点
- `reward_ball_forward_velocity`: ボールが前方へ進むほど加点
- `reward_kick_success`: 初期位置から前方 0.35 m 以上ボールを動かすと加点
- `reward_post_kick_stability`: キック成立後も立位を維持すると加点
- `reward_recover_to_stand`: kick 成功後、default joint pose へ戻るほど加点
- `reward_symmetric_posture`: kick 成功後、左右対称姿勢へ戻るほど加点
- `reward_double_support`: kick 成功後、左右両足接地で加点
- `reward_post_kick_balance`: kick 成功後、両足接地 + upright + 低速度で加点

### 負の報酬

- `penalty_base_position_drift`: 初期 base XY からの移動量を減点
- `penalty_unnecessary_walking`: ベース平面速度を減点
- `penalty_no_kick_timeout`: episode 中に一度もボール接触せず timeout したら大きく減点
- `penalty_fall`: base 高さ低下時に大きく減点
- 既存の姿勢・行動平滑化コスト (`flat_orientation_l2`, `action_rate_l2`, `joint_acc_l2`, `joint_torques_l2`) を維持

### 報酬優先順位

学習初期の探索を優先して、reward の主眼を次の順にした。

1. `first_ball_contact_bonus` / `reward_ball_contact`
2. `reward_ball_speed_increase`
3. `reward_ball_forward_velocity`
4. `reward_kick_success`
5. `reward_post_kick_stability`
6. `reward_recover_to_stand` / `reward_symmetric_posture` / `reward_double_support` / `reward_post_kick_balance`

`reward_stand_still` と `reward_base_position_hold` は **kick 成功後のみ有効** にし、事前の standing-only 局所解を避けている。

## 修正した終了条件

- `time_out`
- `no_kick_timeout`
- `base_height`
- `trunk_contact`
- `ball_travel_distance`: kick success 後 **2.5 秒以上経過してから**、ボールが 1.2 m 以上移動していれば終了
- `post_kick_settle_time`: kick success 後 **2.5 秒** 経過で終了

これにより、キック後の安定姿勢まで評価したうえで episode を閉じる構成にした。

`no_kick_timeout` は manager-based termination term として追加してあるため、ログには `Episode_Termination/no_kick_timeout` が出力される。

## 環境初期化の見直し

- velocity command は常に 0
- robot base の x/y/yaw reset は 0 固定
- joint reset は default pose 固定
- ball 初期位置は前方約 0.20 m、reset 時も前方 0.15-0.25 m / 左右 +/-0.03 m に制限
- train episode 長は 6.0 s、play episode 長は 7.0 s に延長
- kick success 後は 2.5 s の recovery window を確保し、GUI で ball 軌道・飛距離・姿勢回復を観察できるようにした

これにより、学習初期から「立位維持 -> 足を振る -> ボール接触」に入りやすくした。

## Resume 学習方法

RSL-RL の resume は strict `load_state_dict` を使うため、kick 側の actor/critic shape を `k1_flat` に合わせた。

- policy observation: `(49,)`
- critic observation: `(52,)`
- network: flat と同じ MLP 構成

さらに `scripts/rsl_rl/train.py` に resume path 解決を追加し、以下の両方を使えるようにした。

### 1. `--load_run` 形式

```bash
python scripts/rsl_rl/train.py --task Isaac-K1-Kick-v0 --resume --load_run 2026-06-01_13-43-10 --checkpoint model_1499.pt
```

### 2. checkpoint 直指定形式

```bash
python scripts/rsl_rl/train.py --task Isaac-K1-Kick-v0 --resume --checkpoint logs/rsl_rl/k1_flat/2026-06-01_13-43-10/model_1499.pt
```

`K1KickPPORunnerCfg.resume_experiment_name = "k1_flat"` を追加しているため、Kick task からでも `k1_flat` run を既定の resume 元にできる。

## 検証結果

- `train.py --task Isaac-K1-Kick-v0 --resume --load_run 2026-06-01_13-43-10 --checkpoint model_1499.pt` で `k1_flat` checkpoint の resume を確認
- `play.py --task Isaac-K1-Kick-v0 --checkpoint logs/rsl_rl/k1_flat/2026-06-01_13-43-10/model_1499.pt` で Kick task への直接読込を確認
- Kick task 上で 120 step 推論:
  - base 最低高さ `0.532`
  - base 最大 XY drift `0.048 m`
  - 立位維持したままほぼ定位置に留まることを確認
- Flat task 上で同一 checkpoint を 120 step 推論:
  - base 最低高さ `0.513`
  - base XY 移動量 `0.620 m`
  - 同一モデルの歩行能力が保持されていることを確認
- 追加の 7 iteration 学習確認:
  - `Episode_Reward/reward_ball_contact` が iteration 0 から非ゼロ
  - `Episode_Reward/reward_ball_speed_increase` が iteration 0 から非ゼロ
  - `Episode_Reward/reward_ball_forward_velocity` が iteration 0 から非ゼロ
  - `Episode_Reward/reward_kick_success` が iteration 1 以降で非ゼロ
  - `Episode_Termination/ball_travel_distance` が iteration 0 から非ゼロ
  - `Episode_Termination/no_kick_timeout` が iteration 1 で非ゼロ
- recovery phase validation:
  - `reward_recover_to_stand` と `reward_symmetric_posture` は短期学習ログで非ゼロ化を確認
  - 直接 logic probe で、kick success step から **125 step (= 2.5 s)** 後に `post_kick_settle_time` と `ball_travel_distance` が発火することを確認

## 推奨学習コマンド

```bash
python scripts/rsl_rl/train.py --task Isaac-K1-Kick-v0 --resume --load_run 2026-06-01_13-43-10 --checkpoint model_1499.pt
```

または

```bash
python scripts/rsl_rl/train.py --task Isaac-K1-Kick-v0 --resume --checkpoint logs/rsl_rl/k1_flat/2026-06-01_13-43-10/model_1499.pt
```
