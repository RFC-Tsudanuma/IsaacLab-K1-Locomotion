# K1 Kick Porting Report

## 調査結果

### 移植元: Booster T1 kick task
- 参照元: `booster_kick/kick_env/kick_env/source/kick_env/kick_env/tasks/manager_based/kick_env`
- task 登録: `.../kick_env/__init__.py`
- 主構成:
  - `kick_env_env_cfg.py` に env/scene/obs/reward/event/curriculum/termination を集約
  - `mdp/observations.py` にボール相対位置
  - `mdp/rewards.py` にボール接近・接触・方向・姿勢系報酬
  - `agents/rsl_rl_ppo_cfg.py` に PPO 設定
- ボール処理:
  - `RigidObjectCfg` で球体を生成
  - reset 時に `soccer_ball` の root state を更新
- kick 判定:
  - 明示的な success termination はなく、主に接触/蹴り方報酬で学習

### 移植先: Booster K1 locomotion task
- 参照先: `source/isaaclab_k1_locomotion/.../tasks/manager_based/locomotion`
- task 登録: `locomotion/__init__.py`
- train/play 経路:
  1. `scripts/rsl_rl/train.py` / `play.py`
  2. `hydra_task_config(...)`
  3. `load_cfg_from_registry(...)`
  4. `gym.make(task, cfg=env_cfg)`
  5. `isaaclab.envs:ManagerBasedRLEnv`
- K1 asset:
  - `rough_env_cfg.py::K1_LOCOMOTION_CFG`
  - `JOINT_NAMES_K1`
  - foot/body/base 名として `left_foot_link`, `right_foot_link`, `Trunk` が利用可能

### T1 → K1 差分整理
- T1 は USD ベース、K1 は URDF ベース
- T1 kick は腰・腕も action に含むが、K1 locomotion は脚 12DoF を `JOINT_NAMES_K1` で厳密管理
- T1 は単一 policy group、K1 は `policy` / `critic` の 2 観測群を使う
- K1 既存規約に合わせ、manager-based config と gym registration を維持して新 task を追加する方針にした

## 実装内容

### 追加 task
- task 名:
  - `Isaac-K1-Kick-v0`
  - `Isaac-K1-Kick-Play-v0`

### 実装方針
- 既存 locomotion env は直接変更せず、新規 `locomotion/kick/` パッケージを追加
- K1 の既存 asset/action 規約を再利用
- flat ground 上に soccer ball を置く manager-based RL env を新設
- 追加の redesign で、`k1_flat` 学習済み policy をそのまま fine-tuning できる構成へ変更

### 新規 scene / env
- `kick/kick_env_cfg.py`
  - K1 robot + ground + rigid soccer ball + foot-ball contact sensors を定義
  - ball spawn / reset を実装
  - policy / critic 観測群は `k1_flat` と同 shape に維持
  - reward / event / termination manager を stationary kick 用に再設計

### 観測
- resume 互換性を優先し、actor/critic 観測は `k1_flat` と同じ構成へ統一:
  - `policy (49)`:
   - base angular velocity
   - projected gravity
   - velocity commands
   - joint pos/vel
   - previous actions
   - gait phase
  - `critic (52)`:
   - base linear velocity
   - 上記 policy 観測一式
- ball 情報は actor/critic shape を崩さないため、reward / reset / termination 側で扱う方針へ変更

### 報酬
- redesign 後の主要報酬:
  - `first_ball_contact_bonus`
  - `reward_stand_still`
  - `reward_base_position_hold`
  - `reward_ball_contact`
  - `reward_ball_speed_increase`
  - `reward_ball_forward_velocity`
  - `reward_kick_success`
  - `reward_post_kick_stability`
  - `reward_recover_to_stand`
  - `reward_symmetric_posture`
  - `reward_double_support`
  - `reward_post_kick_balance`
- 追従歩行を抑えるペナルティ:
  - `penalty_base_position_drift`
  - `penalty_unnecessary_walking`
  - `penalty_no_kick_timeout`
  - `penalty_fall`
- 安定化コスト:
  - `flat_orientation_l2`
  - `action_rate_l2`
  - `joint_acc_l2`
  - `joint_torques_l2`

### 終了条件
- `time_out`
- `no_kick_timeout`
- `base_height`
- `trunk_contact`
- `ball_travel_distance`
- `post_kick_settle_time`

### PPO
- `kick/agents/rsl_rl_ppo_cfg.py`
- `K1FlatPPORunnerCfg` を継承し、network 形状を flat と一致させたまま `k1_kick` 用設定へ調整
- `resume_experiment_name = "k1_flat"` を追加

### Resume / checkpoint 対応
- `scripts/rsl_rl/train.py`
  - `resolve_resume_path(...)` を追加
  - `--checkpoint` に既存ファイルパスを渡した場合はそのままロード
  - そうでない場合は `resume_experiment_name` を優先して `logs/rsl_rl/k1_flat/...` を探索
- これにより以下の両方をサポート:
  - `--resume --load_run ... --checkpoint model_1499.pt`
  - `--resume --checkpoint logs/rsl_rl/k1_flat/.../model_1499.pt`

### 追加の学習安定化変更
- ball spawn を前方 `0.15-0.25 m` に固定
- `reward_stand_still` / `reward_base_position_hold` は kick success 後のみ有効化
- kick 成功後の recovery rewards を追加し、終了まで 2.5 秒の post-kick 観察ウィンドウを確保
- train `episode_length_s = 6.0`、play `episode_length_s = 7.0`
- `KICK_SUCCESS_DISTANCE = 0.35`
- `BALL_TRAVEL_DISTANCE_THRESHOLD = 1.2`
- `POST_KICK_RECOVERY_TIME_S = 2.5`

## 変更ファイル一覧

### 変更
- `source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/__init__.py`
- `scripts/rsl_rl/train.py`

### 追加
- `docs/K1_KICK_PORTING_PLAN.md`
- `docs/K1_KICK_PORTING_REPORT.md`
- `docs/K1_KICK_REWARD_REDESIGN.md`
- `source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/kick/__init__.py`
- `source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/kick/kick_env_cfg.py`
- `source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/kick/agents/__init__.py`
- `source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/kick/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/kick/mdp/__init__.py`
- `source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/kick/mdp/events.py`
- `source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/kick/mdp/observations.py`
- `source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/kick/mdp/rewards.py`

## train コマンド

```bash
python scripts/rsl_rl/train.py --task Isaac-K1-Kick-v0 --resume --load_run 2026-06-01_13-43-10 --checkpoint model_1499.pt
```

今回の確認では以下を実行した:

```bash
python scripts/rsl_rl/train.py --task Isaac-K1-Kick-v0 --num_envs 2 --max_iterations 1 --headless
python scripts/rsl_rl/train.py --task Isaac-K1-Kick-v0 --resume --load_run 2026-06-01_13-43-10 --checkpoint model_1499.pt --num_envs 2 --max_iterations 1 --headless
python scripts/rsl_rl/train.py --task Isaac-K1-Kick-v0 --resume --checkpoint logs/rsl_rl/k1_flat/2026-06-01_13-43-10/model_1499.pt --num_envs 2 --max_iterations 1 --headless
```

## play コマンド

```bash
python scripts/rsl_rl/play.py --task Isaac-K1-Kick-v0 --checkpoint logs/rsl_rl/k1_flat/2026-06-01_13-43-10/model_1499.pt
```

今回の確認では checkpoint を指定して以下を実行した:

```bash
python scripts/rsl_rl/play.py --task Isaac-K1-Kick-v0 --num_envs 1 --checkpoint logs/rsl_rl/k1_kick/2026-06-10_15-59-25/model_0.pt
python scripts/rsl_rl/play.py --task Isaac-K1-Kick-v0 --num_envs 1 --checkpoint logs/rsl_rl/k1_flat/2026-06-01_13-43-10/model_1499.pt --headless --video --video_length 1
```

## 検証結果
- import/syntax: 新規 kick package は `compileall` を通過
- task 登録: `train.py` / `play.py` から `Isaac-K1-Kick(-Play)-v0` が正しく解決
- train 起動: 1 iteration 実行完了
- play 起動: `Isaac-K1-Kick-v0` / `Isaac-K1-Kick-Play-v0` の両方で checkpoint 読み込み・推論起動完了
- scene 生成: 成功
- ボール生成/リセット: `reset_ball` と球体 asset の生成を確認
- resume:
  - `load_run` 形式で `logs/rsl_rl/k1_flat/2026-06-01_13-43-10/model_1499.pt` をロード
  - direct checkpoint 形式でも同 checkpoint をロード
- shape 互換:
  - kick policy observation `49`
  - kick critic observation `52`
  - actor/critic MLP は `k1_flat` と同形状でロード成功
- 立位維持確認:
  - Kick task 上で flat checkpoint を 120 step 推論
  - `min_height = 0.532`
  - `max_xy_drift = 0.048 m`
- 歩行能力維持確認:
  - Flat task 上で同 checkpoint を 120 step 推論
  - `min_height = 0.513`
  - `max_xy_drift = 0.620 m`
- reward / termination 活性確認:
  - `Episode_Reward/reward_ball_contact` 非ゼロ
  - `Episode_Reward/reward_ball_speed_increase` 非ゼロ
  - `Episode_Reward/reward_ball_forward_velocity` 非ゼロ
  - `Episode_Reward/reward_kick_success` 非ゼロ
  - `Episode_Termination/ball_travel_distance` 非ゼロ
  - `Episode_Termination/no_kick_timeout` 非ゼロ
- post-kick recovery 確認:
  - `reward_recover_to_stand` 非ゼロ
  - `reward_symmetric_posture` 非ゼロ
  - logic probe で success 後 `125 step` 継続してから `post_kick_settle_time` / `ball_travel_distance` が有効化
- GUI 起動経路: non-headless play で `Creating window for environment.` まで確認
- 生成物:
  - `logs/rsl_rl/k1_kick/2026-06-10_15-59-25/model_0.pt`
  - `logs/rsl_rl/k1_kick/2026-06-10_15-59-25/videos/play/rl-video-step-0.mp4`
  - `logs/rsl_rl/k1_flat/2026-06-01_13-43-10/videos/play/rl-video-step-0.mp4`

## 残課題
- reward weight / termination threshold は初期 tuning のため、長期学習で追加調整余地あり
- actor/critic を flat 互換優先にしたため、ball 観測を policy に入れた高度な kick policy へ拡張する場合は checkpoint 変換または別学習系列が必要
- Isaac Sim / GPU / URDF import 由来の warning は出るが、今回の task 起動自体は継続可能
- 初回 fine-tuning は `k1_flat` の既存学習済み checkpoint を前提とする
