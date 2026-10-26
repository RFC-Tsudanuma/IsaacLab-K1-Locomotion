# bhuman_walk_kick の仕様と合格条件

作業者 (人・エージェント) はこのファイルを読んでから作業する。ここに反する変更は、数字が良くても採用しない。

## 目的

歩行方策 `/home/akira/workspace/model_34995.pt` (K1FlatEnvCfg 履歴入力版) の上に、
B-Human の walk-kick (https://github.com/bhuman/MachineLearning, `IsaacGymRL/envs/t1_ball.py`,
`envs/K1_Ball.yaml`, README「Training kick」) の**蹴り方**を学習で再現する。
B-Human が配っている K1 のキックは `K1_kick_inner` = **インサイドキック** (蹴り足を横に向け、足の内側でボールを押し出す)。
これが再現できていない結果は、ボールがどれだけ速く・正確に飛んでも不合格。

## 変えてはいけないこと

- 出発点は model_34995 だけ。他のキック方策 (k1_walk_kick_360 など) を使わない・真似させない。既存キックとの比較もしない。
- B-Human README の段階 (Stage 1→2→3→4→5→6→7) のキック関係の値は README どおりにする:
  - Stage 2: `target_sole_yaw_angle = 1.5705`, `ball_sole_yaw` 重み 25, `disable_sole_yaw_direction_penalty = true`,
    `disable_deviation_flag = false`, `ball_pos_noise_x/y` の下限 0.03
  - Stage 3: `ball_walk_speed` 1, `ball_sole_yaw` 15, 探索ノイズを下げる, 学習率下限 1e-5, `target_sole_yaw_sigma` を段階的に 0.1 へ
  - Stage 4: `disable_sole_yaw_direction_penalty = false`, `ball_sole_yaw` 5, `kick_direction_negative_factor` 3.0,
    `ball_walk_target_overshoot` -10, 足上げ報酬を下げる (こちらでは feet_phase の相当分)
  - Stage 5: `kick_direction_sigma_range` 0.75, `kick_direction_sigma_just_hit` 0.5
  - Stage 6: `ball_sole_yaw` 0, `kick_direction_sigma_range` 0.25 (良ければ 0.15, 0.1, 0.05), `desired_kl` 0.003,
    `random_ball_distance_reset` 0.0005, `ball_walk_speed` 1.0, 探索ノイズをさらに下げる
  - Stage 7: リセット系の確率を最終値へ (`ball_kick_direction` 5, `heavy_ball`, `ball_walk_speed` 0.75,
    `kick_pose_increase_value` 0.2, `ball_distance_offset` 0.7, `ball_rotation_offset` 0.2)
  - 「蹴り足の向き 90°」をやめる・弱める・段を飛ばすことは禁止。
- 歩行側 (土台) の報酬・DR・制御は、インサイドキックの妨げになると**測って**分かったものだけ外す。外したら理由と測定値をコード
  のコメントと進捗ログに書く。
- GPU は cuda:1 だけ。push はしない。

## 合格条件 (MuJoCo `scripts/mujoco/sim2sim_bhuman_walk_kick.py`、ランダム 100 場面、ノイズなし)

すべて満たして合格。1 つでも外れたら「未達」と報告する (「ほぼ合格」と言わない)。

| 項目 | 条件 |
|---|---|
| 蹴り方 | ボールに当たった瞬間の蹴り足の向き (足の前方向とキック方向のなす角) が 90° ± 25° の場面が 80% 以上 |
| 当て方 | 当たった瞬間、ボール中心が蹴り足の足首から見て足の横方向 (内側) にある場面が 80% 以上 |
| 向きの精度 | 蹴った後のボールの進行方向とキック方向のずれの平均 ≤ 10°、最大 ≤ 30° |
| 強キックの強さ | 強キック (20 m 指令) で当たった直後のボール速度の平均 ≥ 3.0 m/s |
| 弱キックの強さ | 距離指定キックで、必要な初速との差の平均 ≤ 0.8 m/s |
| 転倒 | 100 場面で 2 回以下 |
| 歩行 | ボールに向かって歩き、当たる前に蹴り足が接地から離れる (歩きながら足先で押していない) |

上の「蹴り足の向き」「当て方」を測る項目を評価スクリプトに**先に**追加し、歩行のままの方策 (model_34995 変換直後) で
値が出ることを確かめてから学習に入る。

## 報告の仕方

- 各段の終わりに、上の表の全項目の数字を進捗ログ (`/home/akira/workspace/claude_tmp/bhuman_walk_kick_progress.md`) に書く。
- B-Human から変えた点は、理由・測定値つきで列挙する。変えずに済む方法がないか先に試す。
- 合格条件を満たせないまま終えるときは、どの項目がいくつ足りないかを書く。
