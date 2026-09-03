# Booster-K1 起き上がり(get-up)ポリシー 学習ハンドオフ

別PC/別エージェントで学習と改善を継続するための引き継ぎ資料。IsaacLab + rsl_rl で学習し、
`~/booster_k1_locomotion` の C++ ノード経由で MuJoCo / 実機にデプロイする。

---

## 0. ゴールと現状

- **ゴール**: K1(22-DOF 人型)が うつ伏せ/仰向けから **足だけで**立ち上がる、sim2real 可能なポリシー。
- **現状の最大の壁**: IsaacLab では立てるが、**MuJoCo に移すと大暴れして立てない**(sim2sim ギャップ)。
  - 摩擦・アクション遅延・トルク上限・PDゲイン・default角・関節順・重力ベクトルは一致確認済み(下記「調査済み」参照)。
  - 残る主因候補は **接触モデルの差**(MuJoCo は衝突ジオメトリが素のプリミティブ箱のみ、mesh は contype=0)。→ 全身接触に頼る起き上がりが転移しない。
  - このため方針を **「足だけで起き上がる」二段階カリキュラム**に変更(セクション6が本命の現行手法)。

---

## 1. 基本設定(必ず守る)

| 項目 | 値 |
|---|---|
| 学習タスク | `Isaac-Getup-K1-v0` |
| 再生/viser タスク | `Isaac-Getup-K1-Play-v0` |
| PD スケール環境変数 | `GETUP_PD_SCALE=0.6`(全学習・エクスポートで必須) |
| num_envs | 4096 |
| actor obs | **75次元**(base_lin_vel を除く: base_ang_vel3 + projected_gravity3 + velocity_commands3(=0) + joint_pos22 + joint_vel22 + last_action22) |
| critic obs | **78次元**(先頭に base_lin_vel3 を追加した特権情報。actor には入れない=実機で線速度取得困難のため) |
| action | 22-DOF 全身、action_scale 0.5、`q_target = default + 0.5*action` |
| 制御周波数 | 50Hz(decimation 4, physics 200Hz) |

### アクチュエータ(実機トルク上限に一致させ済み)
`GETUP_PD_SCALE=0.6` を掛けた PD ゲインと、実機/MuJoCo と一致させた effort_limit:

| グループ | kp | kd | effort_limit [Nm] |
|---|---|---|---|
| 頭 (AAHead_yaw, Head_pitch) | 12.0 | 3.0 | 6 |
| 腕 (肩P/R, 肘P/Y) | 24.0 | 6.0 | 14 |
| 脚 (股P/R/Y, 膝P) | 96.0 | 2.4 | 30/35/20/40 |
| 足首 (P/R) | 30.0 | 1.5 | 20 |

- **DelayedPD 遅延 2〜8 制御ステップ**(全関節、ランダム)。
- deploy 側ゲイン YAML: `~/booster_k1_locomotion/resource/k1_isaaclab_getup_gains.yaml`(上表と一致)。
- 関節順 = `k1_constants_isaaclab.hpp` JOINT_NAMES: 0-1頭, 2-5左腕, 6-9右腕, 10-15左脚, 16-21右脚。

### 初期姿勢(default角 = deploy `GETUP_DEFAULT_ANGLES` と一致)
脚 Hip_P -0.26 / Knee 0.52 / Ankle_P -0.26(他0)、腕 Shoulder_Roll ±1.3744(=±0.7853981634×1.75、他0)、頭0。

---

## 2. 主要ファイル

**IsaacLab (`~/IsaacLab-K1-Locomotion`)**
- env: `source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/getup_env_cfg.py`
- 報酬: `.../locomotion/mdp/getup_rewards.py`
- reset イベント(prone/supine): `.../locomotion/mdp/events.py::reset_root_state_prone_supine`
- タスク登録: `.../locomotion/__init__.py`(`Isaac-Getup-K1-v0` / `-Play-v0`)
- 学習: `scripts/rsl_rl/train.py`、エクスポート: `scripts/rsl_rl/export_policy.py`、再生: `scripts/rsl_rl/play.py`
- override 機構: `scripts/rsl_rl/config_overrides.py`
- viser 再生スクリプト: `scripts/rsl_rl/play_getup_viser.sh`

**deploy (`~/booster_k1_locomotion`)**
- getup 推論ノード: `src/rl_policy_getup_node.cpp`(obs 75, action 22, base_lin_vel 不要)
- 定数: `src/k1_constants_isaaclab_getup.hpp`(GETUP_OBS_DIM=75)
- MuJoCo sim: `src/mujoco_sim_node.cpp`、XML: `assets/rfc_assets/booster/K1/K1_22dof_soccer_field.xml`
- ゲイン: `resource/k1_isaaclab_getup_gains.yaml`、CMake target `rl_policy_getup_node_cpp`

---

## 3. コマンド

```bash
# --- 学習(スクラッチ) ---
cd ~/IsaacLab-K1-Locomotion/scripts/rsl_rl
GETUP_PD_SCALE=0.6 CUDA_VISIBLE_DEVICES=1 uv run train.py \
  --task Isaac-Getup-K1-v0 --headless --num_envs 4096 \
  --override_json /path/to/override.json

# --- 学習(warm-start: 既存 .pt から継続) ---
#   ※ --resume と --checkpoint の両方が必須(--checkpoint 単独ではロードされない)
GETUP_PD_SCALE=0.6 CUDA_VISIBLE_DEVICES=1 uv run train.py \
  --task Isaac-Getup-K1-v0 --headless --num_envs 4096 \
  --resume --checkpoint /abs/path/model_XXXX.pt --override_json /path/to/override.json

# --- ONNX エクスポート(学習後は毎回行う) ---
GETUP_PD_SCALE=0.6 CUDA_VISIBLE_DEVICES=1 uv run python export_policy.py \
  --task Isaac-Getup-K1-v0 --headless --num_envs 1 --checkpoint /abs/path/model_XXXX.pt
#   出力: <run>/exported/policy.onnx (obs[1,75]->actions[1,22], MLP)

# --- viser 再生(最新 checkpoint 自動選択) ---
CUDA_VISIBLE_DEVICES=1 bash play_getup_viser.sh   # http://localhost:8080
```

- **進捗指標**: TensorBoard/ログの `Curriculum/com_height`(全身 CoM 高さ[m])。
  prone ≈ 0.08、完全起立 ≈ 0.46–0.50。各報酬は `Episode_Reward/<term>`。
- 速度 ~4.4s/iter(4096 envs, RTX A4000/4000 Ada)。2000 iter ≈ 2.4h、5000 iter ≈ 6h。
- **注意**: 学習は nohup でバックグラウンド化。`kill -9 <PID>` で停止(`pkill -f "...Isaac-Getup..."` は
  自分のコマンド行にマッチして誤爆するので PID 指定推奨)。

---

## 4. 報酬項リファレンス(config 現在値)

`getup_env_cfg.py` の RewardsCfg。weight は override(`rewards.<name>.weight`)で上書き可。
`require_upright` などの params は `rewards.<name>.params.<key>`(**キーが既に存在する項のみ**)で上書き可。

**起き上がり駆動(タスク報酬)**
| name | weight | 備考 |
|---|---|---|
| base_height_increase | 80 | Δ胴高。require_upright ゲート(スクラッチ時は false に ungate 必須) |
| base_height | 15 | 胴高(min_height~0.2閾値)。require_upright ゲート |
| head_height | 25 | 頭高。require_upright ゲート。handstand 拒否に効く |
| upright_posture | 3 | (1-g_z)/2、反転拒否。ゲート無し |
| feet_ground_contact | 3 | 両足接地率。require_upright ゲート |
| feet_vertical_force | 20 | 足裏の垂直反力/体重(≤1.0)。**require_upright ゲート追加済**(寝farm防止) |
| stand_still | 4 | 起立後(com>0.4 & upright)の静止 |

**足だけ起き上がり誘導**
| name | weight | 備考 |
|---|---|---|
| non_foot_contact | -3 | 手(_hand_link)・膝(_Shank)の接地力/体重。足だけ起立へ誘導(Stage1 の主誘導) |
| feet_flat | -8 | 接地足の水平からのズレ。require_contact ゲート |
| feet_low | 0.5 | Σexp(-10·h_foot)、足を低く |
| feet_slide | -1.0 | 接地足の滑り(mdp.feet_slide) |
| feet_reaction_increase | 10 | 垂直反力の増分。ゲート無し(スクラッチでは寝farm するので0推奨) |

**sim2real 整形 penalty**
| name | weight |
|---|---|
| dof_acc_l2 | -6e-8 |
| dof_vel_l2 | -3e-3 |
| action_rate_l2 | -0.03 |
| action_smoothness_l2 | -0.03 |
| dof_torques_l2 | -3e-5 |
| joint_power | -1e-4(=Σ\|torque·vel\|) |
| torque_over_limit | -0.03(effort_limit×0.7超過分) |
| jump | -10(両足浮き、upright時) |
| body_symmetry | 0.25(exp対称報酬) |
| body_symmetry_l1 | -2.0(Σ\|q_L - sign·q_R\|、roll/yawはミラー符号) |
| dof_pos_error / arm_pos_error | -0.5 / -0.5 |
| ang_vel_xy_l2 / base_lin_vel_xy_l2 | -0.01 / -0.1 |

**接触センサ** `contact_forces`: `(Trunk|.*_foot_link|.*_hand_link|.*_Shank)`。
net_forces_w_history は **index 0 = 現ステップ, 1 = 前ステップ**(roll実装)。

---

## 5. これまでの検証と教訓(重要)

1. **base_lin_vel は critic 専用**にした(実機で線速度取得困難)。actor 78→75 化はチェックポイント手術
   (actor.0.weight と actor_obs_normalizer の先頭3列を削除)+ fine-tune で温存した。以降は 75/78 で統一。

2. **MuJoCo で暴れる真因は摩擦でも遅延でもなかった**:
   - 摩擦: MuJoCo 実効 μ≈0.7–1.0(地面0.7+足1.0)で**高い**。低摩擦説は誤り。
   - **トルク上限の不一致だった**: 歩行から流用した effort_limit(膝112 等)で学習していたが、deploy は
     膝40Nmでクランプ → 持ち上げ切れず暴れる。→ **effort_limit を実機値(膝40/股30・35・20/足首20)に一致**させ解決。
   - アクション遅延(DelayedPD 2–8): MuJoCo `action_delay_ticks` を合わせても暴れは**変化なし** → 主因ではない。
   - 残る主因候補 = **接触モデル差**(MuJoCo 衝突は素のプリミティブ箱、mesh非衝突)。全身接触起立が転移しない。

3. **スクラッチ学習は「そのまま」では失敗する**(局所最適 com~0.16 or prone 0.08 で停滞):
   - 原因: 起き上がり報酬(base/head_height 等)が全て **require_upright ゲート** → 寝姿勢では勾配ゼロ。
     さらにゲート無しの feet_vertical_force が「寝たまま足を押して farm」できる局所最適を作る。
   - **解決(スクラッチ成功のレシピ)**: (a) head_height/base_height/base_height_increase を **ungate**
     (`params.require_upright=false`)、(b) 寝farm する feet_vertical_force / feet_reaction_increase を **0
     または upright ゲート**、(c) その他 penalty を全0。→ **com 0.08→0.50 を ~2500 iter で達成**。
   - 低姿勢の駆動役は base_height_increase(Δ) + upright_posture、com>0.2 で base/head_height が加勢。

4. **warm-start は「手動ペナルティ・カリキュラム」**。弱penaltyで起立を覚えさせ、段階的にpenaltyを足すのが確実。
   逆に強penaltyからのスクラッチは探索が潰れて立てない。→ これを formalize したのがセクション6の二段階。

5. **並列実験・重み一括変更**は `--override_json` で(config を触らず変種を回せる)。GPU2枚並列可。

---

## 6. 現行の本命手法: 二段階カリキュラム(足だけ起き上がり)

### Stage 1 — 自由探索(penalty 少・ランダム化 少・足だけ誘導)
**目的**: 「足だけで起き上がる」動きをスクラッチで発見させる。sim2real 整形はまだしない。

- 報酬: 高さ ungate(スクラッチ勾配)+ `feet_vertical_force`(20, upright ゲート)+ `feet_ground_contact`(3)
  + **`non_foot_contact`(-3)= 手・膝の接地を罰し足だけ起立へ誘導**。sim2real penalty は全0。
- ランダム化 最小: 初期姿勢は**うつ伏せ/仰向けの2択のみ**(pose_range 全0, prone_prob 0.5)、
  **関節固定**(reset_robot_joints position_range [1,1])、摩擦 0.9–1.0、質量/COM/ゲイン/push 無効。
- 起動: スクラッチ(`--resume` 無し)。com>0.4 の起立まで ~2500–5000 iter を目安に回す。
- **未収束**(GPU譲渡のため中断)。次エージェントはここから再開 or 回し直す。
- 注意点: `non_foot_contact -3` が強すぎて立てない場合は -1〜-2 に緩める。

**Stage 1 override(`getup_stage1.json`。別PCではこの内容でファイル作成):**
```json
{
  "env": {
    "rewards.base_height_increase.params.require_upright": false,
    "rewards.base_height.params.require_upright": false,
    "rewards.head_height.params.require_upright": false,
    "rewards.feet_reaction_increase.weight": 0.0,
    "rewards.body_symmetry.weight": 0.0,
    "rewards.body_symmetry_l1.weight": 0.0,
    "rewards.feet_flat.weight": 0.0,
    "rewards.feet_low.weight": 0.0,
    "rewards.feet_slide.weight": 0.0,
    "rewards.jump.weight": 0.0,
    "rewards.torque_over_limit.weight": 0.0,
    "rewards.dof_acc_l2.weight": 0.0,
    "rewards.dof_vel_l2.weight": 0.0,
    "rewards.action_rate_l2.weight": 0.0,
    "rewards.action_smoothness_l2.weight": 0.0,
    "rewards.dof_torques_l2.weight": 0.0,
    "rewards.joint_power.weight": 0.0,
    "rewards.dof_pos_error.weight": 0.0,
    "rewards.arm_pos_error.weight": 0.0,
    "rewards.ang_vel_xy_l2.weight": 0.0,
    "rewards.base_lin_vel_xy_l2.weight": 0.0,
    "events.reset_robot_joints.params.position_range": [1.0, 1.0],
    "events.reset_base.params.pose_range": {"x": [0.0,0.0], "y": [0.0,0.0], "yaw": [0.0,0.0], "roll": [0.0,0.0]},
    "events.physics_material.params.static_friction_range": [0.9, 1.0],
    "events.physics_material.params.dynamic_friction_range": [0.9, 1.0],
    "events.add_base_mass.params.mass_distribution_params": [0.0, 0.0],
    "events.base_com.params.com_range": {"x": [0.0,0.0], "y": [0.0,0.0], "z": [0.0,0.0]},
    "events.randomize_actuator_gains.params.stiffness_distribution_params": [1.0, 1.0],
    "events.randomize_actuator_gains.params.damping_distribution_params": [1.0, 1.0],
    "events.push_robot.params.velocity_range": {"x": [0.0,0.0], "y": [0.0,0.0]}
  }
}
```

### Stage 2 — sim2real 化(penalty 増・ランダム化 増)— 未実装
**目的**: Stage 1 の足だけ起立ポリシーから warm-start し、実機で動く穏やかな動きに整形。

- **penalty を戻す・強める**: 速度(dof_vel_l2)/トルク(dof_torques_l2, torque_over_limit)を**大きめに**、
  action_rate / action_smoothness / joint_power / 対称(body_symmetry_l1)を追加。足姿勢(feet_flat/feet_low/feet_slide)も戻す。
- **ランダム化を増やす**: **関節角を大きくランダム化**(reset_robot_joints position_range を [0.5,1.5] 以上に、
  又は reset_joints_by_offset で角度オフセット付与)して**多様な初期姿勢**から。pose_range(yaw/roll/x/y)も復活。
  摩擦 0.3–1.0、質量 ±1.5、ゲイン 0.8–1.2、push 復活。
- 実装方針: Stage2 用 override JSON を作り、`--resume --checkpoint <Stage1 model> --override_json <stage2.json>`。
  weight は段階的に(いきなり最大にすると起立が壊れる)。com 維持を見ながら up。

---

## 7. 利用可能なチェックポイント(`scripts/rsl_rl/logs/rsl_rl/k1_getup/<run>/`)

| run | 内容 | com | 備考 |
|---|---|---|---|
| 2026-07-28_19-47-11 | 実機トルク準拠 warm-start版 | 0.469 | effort_limit 実機一致。全身接触起立 |
| 2026-07-31_01-19-04 | **スクラッチ素起立**(高さ ungate) | 0.499 | penalty無し。踵ピボット起立(MuJoCo不可) |
| 2026-07-31_15-32-02 | 素起立から足接地整形(shape1) | 0.49 | 足接地改善も踵ピボット残存 |

いずれも `<run>/exported/policy.onnx` にエクスポート済み。**ただし全て MuJoCo では大暴れ**(踵/全身接触依存)。
→ Stage 1(足だけ起立)からやり直すのが現行方針。

---

## 8. 次エージェントへの推奨アクション

1. **Stage 1 をスクラッチで回す**(`getup_stage1.json`)。com>0.4 まで(~2500–5000 iter)。
   足だけで立てているか viser で確認(`play_getup_viser.sh`)。手・膝を使わず立てば成功。
2. 立てたら **MuJoCo で検証**(`policy.onnx` を `rl_policy_getup_node` の model_path へ)。
   ここで暴れなければ「接触依存が主因」だった確証になる。
3. **Stage 2** で penalty(速度/トルク大)+ 関節ランダム化を段階的に足して sim2real 品質へ。
4. どうしても MuJoCo 転移しない場合の代替: **接触モデル差の直接調査**(MuJoCo XML の衝突ジオメトリを
   IsaacLab の collision と揃える)、または **MuJoCo 側でのドメインランダム化強化 / MuJoCo で直接学習**。
5. **学習完了ごとに必ず ONNX エクスポート**まで行う(ユーザー要望)。

---

## 9. sim2sim 調査で「一致確認済み」の項目(再調査不要)
default角 / PDゲイン(60%) / 関節順(JOINT_NAMES) / トルク上限(実機一致) / action_scale 0.5 / 50Hz /
projected_gravity(IMU rpy→body frame)/ アクション遅延(合わせても暴れ変化なし=無関係)。
→ **残る主因は接触モデル差**の可能性が最も高い。
