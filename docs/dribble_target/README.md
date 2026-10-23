# K1 ドリブル (walk → dribble) 学習環境の設計

Booster K1 ヒューマノイドが、サッカーボールをワールド座標の目標点まで運ぶ (ドリブルする) ポリシーを
IsaacLab + rsl_rl で学習する環境の説明。この文書だけ読めば、環境の全体像・カリキュラム・報酬・
学習手順・既知の問題が分かるように書いている。値はすべて 2026-10-20 のラン
(`logs/rsl_rl/k1_dribble_target/2026-10-20_08-19-15`) で実際に使われたもの (各ランの
`params/env.yaml` と同じ)。

最終更新: 2026-10-22 (コース 10 レベル・障害物廃止の改訂) / ブランチ `feat/dribble_target_rebuild`

---

## 1. 要点

- **方式**: 実機で使っている歩行ポリシー (HistoryActorCritic) を、そのままドリブル報酬で **直接 fine-tune** する。
  階層型 (凍結した歩行ポリシー + 上位ポリシー) ではない。ネットワークは 1 つ。
- **観測**: 歩行ポリシーの 1 ステップ 49 次元の **後ろに** ドリブル用の 10 次元を足した 59 次元を、
  100 ステップ分の履歴で入れる。先頭 49 次元は歩行と完全に同じ並び (実機で歩行の履歴を引き継ぐため)。
- **段階**: Stage0 (重みの拡張) → Stage1 (歩行のみ、ドリブル用の観測は 0) → Stage2 (ドリブル)。
- **Stage2 では歩行の速度コマンドを常にゼロ** にし、歩容の位相 (gait_phase) は 1.8Hz で回し続ける。
  「どこへ歩くか」はコマンドではなくポリシー自身が、ボール・目標の観測から決める。
- **カリキュラム**: コースの形で難しくする 10 レベル (直線 → C 字・O 字・S 字 → L 字・V 字・U ターン → Z 字・8 の字)。
  成功率に応じて進級。障害物はやめた (2026-10-22): 障害物の回避は戦略側の経路計画 (pathplan_lib) に任せ、
  ポリシーは渡されたウェイポイント列をたどることに専念する。
- **実機制約**: 実機 (C++ / ONNX Runtime、50Hz) で用意できない値は actor に入れない。特権情報は critic だけ。

---

## 2. ファイル構成

すべて `source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/dribble_target/` 以下。

| ファイル | 内容 |
|---|---|
| `k1_dribble_target_env_cfg.py` | **環境の本体**。Stage2 (`K1DribbleTargetEnvCfg`)、Stage1 (`K1DribbleTargetWalkPhaseEnvCfg`)、再生用 (`*_PLAY`)、成熟済み ckpt の再開用 (`*_ResumeMature`) |
| `flat_env_cfg.py` / `rough_env_cfg.py` / `velocity_env_cfg.py` | 歩行の環境 (親クラス)。`K1FlatStancePlaneCfg` を親に使う |
| `history_layout.py` | 観測の項の並び (`DRIBBLE_TARGET_POLICY_TERM_SPECS` など)。HistoryActorCritic・左右反転・エクスポータが全部これを参照する |
| `agents/history_actor_critic.py` | HistoryActorCritic (履歴 CNN + MLP の dual encoder)。`obs_layout` で歩行 / ドリブルのレイアウトを切り替える |
| `agents/history_policy_exporter.py` | ONNX / JIT への書き出し |
| `agents/rsl_rl_ppo_cfg.py` | PPO の設定 (`K1DribbleTargetPPORunnerCfg` など) |
| `mdp/observations.py` `rewards.py` `terminations.py` `commands.py` `curriculums.py` `events.py` | ドリブル用の MDP 関数 (各ファイルの末尾「ドリブル用」の節) |
| `mdp/obs_noise_models.py` | センサの劣化 (遅延・ホールド・バイアス) を模すノイズモデル |
| `mdp/symmetry.py` | mirror loss 用の左右反転 |

この `dribble_target` パッケージの歩行部分は、`feat/inoue_walk_double_encoder` ブランチ (21d53cf) の
`locomotion` パッケージの複製。このブランチ自身の `locomotion` は walk_kick 系が依存する別系統なので混ぜていない。

スクリプト (`scripts/rsl_rl/`):

| スクリプト | 用途 |
|---|---|
| `train_dribble_target.sh` | Stage0〜2 を通しで実行 (`STAGE=1` / `STAGE=2` で個別に) |
| `widen_history_checkpoint.py` | 歩行 ckpt (49 次元) をドリブル用 (59 次元) に広げる。変換前後の出力一致を float64 で検証 |
| `train.py` | 学習。`--reset_critic` / `--reestimate_obs_norm_terms` / `--critic_warmup_iters` を追加済み |
| `stage_transition_utils.py` | 上のオプションの中身、カリキュラム状態の保存・復元 |
| `stage2_progress.py` | Stage2 の転倒率を、累計サンプル数で前回のランと比べる |
| `dribble_watchdog.py` | 学習の監視 (NaN で停止、iteration 22000 の回復判定、Level3 到達時のエクスポート、停滞の報告) |
| `export_dribble_target.py` | ckpt → ONNX + 仕様 JSON (Isaac Sim 不要) |
| `play_dribble_target.sh` / `.py` | レベルを指定して再生・録画 (`./scripts/rsl_rl/play_dribble_target.sh 5`) |
| `eval_stage1_base.py` | Stage1 の環境の系を推論だけで比べる (歩行の親 cfg を選んだときの比較) |

---

## 3. ロボット・行動

| 項目 | 値 |
|---|---|
| ロボット | Booster K1。制御は脚の 12 関節 |
| 関節の順序 | L: Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, Ankle_Pitch, Ankle_Roll → R: 同じ並び |
| 行動 | `q_target = default + 0.5 * action` (clip なし) |
| 既定姿勢 [rad] | Hip_Pitch -0.26, Hip_Roll 0, Hip_Yaw 0, Knee_Pitch 0.52, Ankle_Pitch -0.26, Ankle_Roll 0 (左右共通) |
| アクチュエータ | DelayedPD。脚 (Hip 3 軸・Knee) stiffness 160 / damping 4、足首 50 / 2.5 |
| effort_limit [Nm] | Hip_Pitch 68, Hip_Roll 76, Hip_Yaw 38.3, Knee_Pitch 112, Ankle 38.3 |
| 制御周期 | 50Hz (物理 200Hz、decimation 4)。エピソード 25 秒 = 1250 ステップ |
| URDF | `assets_soccer/.../K1_locomotion.urdf` (Hip_Pitch の z = -0.077)。§10 参照 |

ドメインランダム化: PD ゲイン ×U(0.8, 1.2)、行動の遅延 2〜7 制御ステップ、リンク質量・慣性 ×U(0.9, 1.1)、
摩擦 U(0.6, 1.0)、重心位置、押し (7〜10 秒ごとに ±0.5m/s)、歩容周波数のオフセット ±0.05Hz、
地形は平面 0.7 / 軽い凹凸 0.3。

---

## 4. 観測

### 4.1 actor (policy): 1 ステップ 59 次元 × 履歴 100 ステップ

| ch | 項 | 次元 | 内容 | 学習時のノイズ |
|---|---|---|---|---|
| 0–2 | base_ang_vel | 3 | IMU 角速度 | ±0.2、遅延 0〜20ms |
| 3–5 | projected_gravity | 3 | 重力方向 | ±0.05、遅延 0〜20ms |
| 6–8 | velocity_commands | 3 | 歩行の速度コマンド。**Stage2 では常に 0** | なし |
| 9–20 | joint_pos | 12 | 関節角 − 既定姿勢 | ±0.03、遅延 0〜20ms |
| 21–32 | joint_vel | 12 | 関節角速度 | ±1.5 |
| 33–44 | actions | 12 | 前回の行動 | なし |
| 45–48 | gait_phase | 4 | [sin φL, cos φL, sin φR, cos φR]、φR = φL + π | なし |
| 49–50 | target_pos_rel | 2 | 目標点 (今のウェイポイント) の相対位置 | ±0.05、ホールド 0.6〜0.8、遅延 0〜0.1 秒 |
| 51–52 | ball_pos_rel | 2 | ボール中心の相対位置 | ±0.05、距離依存のホールド |
| 53–54 | ball_vel | 2 | ボールのワールド速度を回したもの | ±0.1、距離依存のホールドと欠損 (0) |
| 55–56 | foot_left_ball_pos_rel | 2 | ボール中心 − 左 foot_link 原点 | ±0.05 |
| 57–58 | foot_right_ball_pos_rel | 2 | ボール中心 − 右 foot_link 原点 | ±0.05 |

- 0–48 は歩行ポリシーと完全に同じ。**この並びは絶対に変えない** (実機で歩行の履歴を引き継ぐ)。
- 49–58 はすべて **body yaw 座標** (ロボットの yaw だけを打ち消した座標系。roll/pitch は残す) の xy。
  `offset_b = quat_apply_inverse(yaw_quat(robot_quat_w), p_w - robot_pos_w)[:2]`
- ボール観測の劣化 (実機の vision_filter を模す。足元のボールはカメラの死角に入るので近いほど劣化が大きい):
  ボール距離 0.35m〜2.0m の間を線形補間して、ホールド (前回の値をそのまま出す) 確率 0.6 → 0.05、
  速度欠損 (厳密に 0 を返す) 確率 0.4 → 0.02。補間距離は仮の値 (実機ログで要調整)。
- 白色ノイズは観測関数の中で掛ける (ホールドした値や欠損の 0 にノイズが乗らないように)。
- 別に `command` グループ (最新のコマンド 3 次元、履歴なし) があり、ネットワークに直接入る。Stage2 では常に 0。

### 4.2 critic: 1 ステップ 64 次元 × 履歴 100 ステップ (ノイズなし)

base_lin_vel (3) + actor の先頭 49 次元 + zmp_position (2) + target_pos_rel (2) + ball_pos_rel (2) +
ball_vel (2) + 足→ボール (2 + 2)。2026-10-22 に obstacle_pos_rel (2) を外した (66 → 64 次元)。

### 4.3 ネットワーク (HistoryActorCritic)

actor / critic とも: 「command + 履歴 (正規化済み)」から、(a) 各項の直近 4 ステップをそのまま、
(b) 100 ステップの履歴を 1-D CNN (kernel 8/4, filter 32/16, stride 4/2) で圧縮した特徴、を MLP
[512, 256, 128] (ELU) に入れる。観測の正規化 (EmpiricalNormalization) は入力全体に掛かる。
mirror loss (左右反転の対称性、係数 0.5) を使う。足→ボールの 2 項は左右反転で左右を入れ替える。

---

## 5. コマンドと歩容

- **dribble_target** (`WaypointPose2dCommand`、`mdp/commands.py`): ボールを運ぶウェイポイント列 (ワールド座標)。
  実機では戦略の経路計画が、ボールを始点に障害物を避けるウェイポイント列を出し、推論ノードが今のウェイポイントを
  target_pos_rel として入れる。学習ではその点列をコースの形から作る。heading は使わない。
  - 始点 S = ボールの現在位置 (エピソードの開始時)。進行方向は通常ワールド +x (リセット直後のロボットの向き)、25% は全方位。左右はランダムに反転。
  - コースの形は env ごとのレベルで決まる (§7、コースの例は `courses.png`)。
  - **ランダムな折れ線** (`random_polyline`): エピソードの 20% は、そのレベルのコースと同じ長さの折れ線にする
    (点の数 2〜5、区間 0.5〜3m、各点での曲がり 0〜90°。長さが 15m を超えるときは 15m)。実機の経路計画の出力
    (障害物を迂回する経路) に常に慣れさせるため。例は `polylines.png`。
  - **通過判定** (`waypoint_passed`、実機の推論ノードも同じ式): W = 今のウェイポイント、P = 1 つ前の点
    (最初は経路の始点)、B = ボール中心、u = (W − P)/|W − P| として、
    `|B − W| < 0.3`  または  `(B − W)・u ≥ 0 かつ |B − W| < 1.0` なら次へ (1 制御周期に 1 つ)。
    後者は「W を通り、入ってきた区間に垂直な直線を越えた」(行き過ぎて引き返そうとして転ぶのを防ぐ)。二等分線ではないので
    V 字・U ターンでも角まで行かないと通過しない。1.0m の上限は、O 字・8 の字でループの内側を突っ切ったときに
    離れたウェイポイントを次々に通過とみなすのを防ぐ (幾何のテストで、上限なしだと最大 4.6m 離れた通過が出た)。
    最終ウェイポイントでは進まない (成功は ball_reached_target)。
  - **経路の引き直し** (`replan_dribble_path` イベント): env ごとに 2〜5 秒間隔で、確率 0.3 で経路を計画し直す。実機と同じく、
    新しい列は「今のボール位置を始点 (= 通過判定の P)、未通過の点を先頭」にし、途中の点を入ってきた区間に垂直に 0.2〜0.8m
    ずらす (env ごとに同じ量、迂回路のように)。最終目標は動かさない。エピソードの途中で目標を全面的に引き直すことはしない。
- **base_velocity** (歩行の速度コマンド): Stage2 では **定数ゼロ**。速度のカリキュラムも無効。
- **gait_phase**: 周波数はコマンドの速さから決まり、コマンドがゼロなら 1.8Hz。歩行ポリシーでは
  「コマンドが cmd_threshold (0.05) 未満なら位相を 0 にする (止まれの合図)」だったが、Stage2 では
  最初の 500 iteration で cmd_threshold を 0.05 → −1.0 に線形に下げ、**位相を常に回す** ようにする
  (いきなり切り替えると転倒率が 99% に張り付いた経験があるため)。feet_phase 報酬の閾値も同時に動かす。
  2026-10-20 のランでは開始から 24 iteration 目に閾値が 0 をまたぎ、転倒率は上がらなかった。

---

## 6. リセット・終了条件

リセット (Stage2):

| 対象 | 範囲 |
|---|---|
| ロボット位置 / 向き | env 原点 ±0.1m / yaw ±0.2rad (ボールが最初の一歩の進路上に来るように絞っている) |
| ボール | 既定位置 (0.30, 0, 0.11) + x U(0.4, 0.6)、y U(−0.1, 0.1) (env 原点基準のワールド座標) |
| ロボットの速度 | ±0.5 |
| 関節角 | 既定姿勢のまま (`reset_joints_by_scale (1.0, 1.0)`、ランダム化なし。best_walk の系の値) |

ロボットから見た開始時のボール (body yaw 座標、上の範囲から計算): 前方 0.56〜1.01m (99% は 0.62〜0.97m)、
横 ±0.38m (99% は ±0.27m)、方位 ±28° (99% は ±19°)、静止。実機はこの範囲に入ったらドリブルに切り替える。

終了条件:

| 名前 | 条件 | 扱い |
|---|---|---|
| ball_reached_target | **ボールとロボットの両方** が最終目標から 1.0m 以内 (途中のウェイポイントでは判定しない) | 成功 (+300) |
| base_height | 胴体の高さ < 0.35m (転倒) | 失敗 (−300) |
| base_contact | 胴体が接触 | 失敗 (−300) |
| ball_too_far | ロボット−ボール間 > 4.0m | 失敗 (−300) |
| time_out | エピソードごとに max(30 秒, 経路の長さ ÷ 0.5m/s + 5 秒)、上限 60 秒 (`dribble_time_out`) | 失敗扱い (カリキュラム上) |

time_out は進級の判定で失敗に数えるので、遅くても完走できる時間にしている。0.5m/s は、曲がるコースでのドリブル中の
平均の速さ (0.5〜0.6m/s、model_24600 の C 字・L 字で測定)。結果として O 字は最長 約 40 秒、8 の字は最長 約 50 秒、
遠い目標 (10%) の L 字・Z 字は最長 43〜47 秒、それ以外はほぼ 30 秒。

---

## 7. カリキュラム

### 7.1 レベル (dribble_target_range、2026-10-22 改訂)

| Level | 区分 | コース | 経路の長さ | 進級に必要な失敗率 EMA |
|---|---|---|---|---|
| 1 | 直線 | 近距離 (2.0–2.5m) | 2.0–2.5m | < 0.20 |
| 2 | 直線 | 中距離 (2.5–4.0m、横 ±0.5m) | 2.5–4m | < 0.30 |
| 3 | 連続旋回 (止まらず曲がる) | C 字 (中点を横にずらす) | 4.8–6.6m | < 0.40 |
| 4 | 〃 | O 字 (楕円 半軸 2–3 / 1.5–2.5m を 1 周、16 点) | 11.7–15.0m | < 0.50 |
| 5 | 〃 | S 字スラローム (振幅 0.5–1m、波長 3–4m、1.5–2 周期、12 点) | 6.3–9.4m | < 0.50 |
| 6 | 方向転換 (減速→ピボット→再加速) | L 字 (直角 1 回) | 6.4–8.8m | < 0.50 |
| 7 | 〃 | V 字 (120–150° の鋭角ターン、各区間 2–4m) | 4.2–6.7m | < 0.55 |
| 8 | 〃 | U ターン (幅 1–2m で折り返す、各区間 2–4m) | 6.6–8.9m | < 0.60 |
| 9 | 複合 | Z 字 (切り返し 2 回) | 9.2–10.7m | < 0.60 |
| 10 | 〃 | 8 の字 (半径 1.2–1.8m の円を左・右 1 周ずつ、24 点) | 15.4–22.1m | (最終) |

旧 Level6〜8 (障害物) は 2026-10-22 に廃止した。

**レベルは env ごと**: エピソードの開始時に決める。25% は通過済みのレベルから一様に選ぶ (前のレベルの混合、
2026-10-20 のランで最後の 13000 iteration を直線コースだけで学んで曲がり方を忘れた対策)。それ以外は今のレベル。
さらに 20% はランダムな折れ線 (§5)。

**進級判定の仕組み** (`mdp/curriculums.py` の `dribble_target_range_curriculum`):

- 「失敗」= ball_reached_target **以外** で終わったエピソード (time_out・転倒・ボールを見失うを含む)。
- 数えるのは今のレベルの「混合でない・折れ線でない」エピソードだけ。EMA はエピソード 1 つごとに重み 1/2000 で更新する。
- そのレベルに **最低 300 iteration** いて、EMA が閾値を下回ったら次のレベルへ。進級したら EMA を 1.0 に戻す。
- 状態は ckpt と同じ名前の `model_*_curriculum.json` に保存し、再開時に読み込む。

**TensorBoard** (`Curriculum/dribble_target_range/...`、混合・折れ線のエピソードも含む):
`level`、`failure_ema`、`iterations_in_level`、`L{k}_success` / `L{k}_fall` / `L{k}_timeout` (レベル k の成功率・転倒率・time_out の割合、エピソードごとに重み 1/500 の EMA)、
`fall_replan` / `fall_corner` / `fall_overshoot` / `fall_other` (終わったエピソードのうち、その状況で転倒した割合。4 つの和 = 転倒率)。
状況はリセット前の状態で判定し、優先順は replan (引き直しから 1 秒以内) > corner (途中のウェイポイントから 1m 以内、
またはウェイポイントを進めてから 1 秒以内) > overshoot (今のウェイポイントがロボットの後ろ側) > other。

### 7.2 報酬のフェードイン

Stage2 の開始から、ドリブル用の報酬を weight 0 → 最終値へ線形に上げる (Stage1 のポリシーにいきなり満額を当てないため)。
基準は `env.common_step_counter` (iteration × 48)。期間は原則 500 iteration、robot_progress_to_target だけ 150 iteration。
歩行の速度追従の報酬 (track_lin_vel_* / track_ang_vel_z) は Stage2 では最初から 0 (コマンドがゼロなので意味がない)。

---

## 8. 報酬 (Stage2、フェードイン後の最終値)

### 8.1 ドリブル用

| 報酬 | weight | 中身 |
|---|---|---|
| ball_progress_to_target | 40 | ボール→目標の距離の 1 ステップの減少量 (ポテンシャル形式、±0.05m でクランプ) |
| robot_progress_to_target | 150 | ロボット→目標の距離の減少量 (同上)。「その場足踏み」局所解の対策 |
| approach_ball_progress | 40 | ロボット→ボールの距離の減少量 (±0.1m でクランプ) |
| ball_reached_target_bonus | 300 | 成功したステップに 1 |
| ball_velocity_toward_target | 3 | ボール速度の目標方向成分 / 0.8m/s を [0, 1] に (速く蹴っても得にならない) |
| push_ball_toward_target | 1 | 足がボールに加える力の向きが目標方向なら cos を [0, 1] で |
| robot_facing_ball | 2 | ボールがロボットの正面方向にあるほど大きい [0, 1] |
| ball_control_radius_penalty | −2 | ロボット−ボール間が 1.0m を超えた分 [m] |
| ball_trajectory_deviation_penalty | −5 | ボール速度の目標方向に直交する成分 (速度 0.5m/s 以上のとき) |
| penalize_ball_clamping | −10 | 両足で同時にボールに触れている (挟んで運ぶ局所解の対策) |
| termination_penalty | −300 | 転倒・胴体接触・ボールを見失う (成功と time_out は除く) |
| touch_ball / impact_kick_reward / ball_proximity_to_target | 0 | 無効 (過去に局所解を誘発したため) |

### 8.2 歩行用 (歩行ポリシーと共通、主なもの)

feet_phase 1.5 (stance_ratio 0.6) / no_fly 4 (両足が浮くと罰) / feet_parallel_to_ground 90 /
base_height_penalty −100 (0.53m 未満) / feet_close_penalty −20 / flat_orientation_l2 −40 /
zmp_stability 0.2 / action_rate_l2 −0.5 / action_smoothness_l2 −0.15 / feet_slide −0.5 /
joint_deviation_hip_yaw −1 / feet_air_time 0.2 (コマンドがゼロだと 0) など。全項目は `params/env.yaml` を参照。

---

## 9. 学習手順

```bash
./scripts/rsl_rl/train_dribble_target.sh            # Stage0〜2 を通し (既定 2048 env / ミニバッチ 16)
STAGE=2 WALK_CKPT=logs/rsl_rl/k1_dribble_target_walk_phase/<run>/model_19999.pt ./scripts/rsl_rl/train_dribble_target.sh
```

| 段階 | 内容 | 主な設定 |
|---|---|---|
| Stage0 | `frozen_policies/best_walk/model_52500.pt` (49 次元) を 59/66 次元に広げる。追加分の重みは 0 なので出力は変わらない | iteration は 0 に戻す。optimizer の状態は捨てる |
| Stage1 | `...-Dribble-Target-WalkPhase-v0`。ボールなし、ドリブル用の観測は厳密に 0 (ノイズも外す)、通常の歩行コマンドで歩行だけ | `--resume --checkpoint <拡張後> --reset_noise_std 0.05`、20000 iteration |
| Stage2 | `...-Dribble-Target-v0`。Stage1 の最終 ckpt から | `--resume --reset_critic --reestimate_obs_norm_terms <ドリブル用 5 項>`、20000 iteration |

- `--reset_critic`: critic とその正規化器を作り直し、optimizer の状態を読まず、`common_step_counter` を 0 から始める
  (フェードインを最初から回すため。iteration の番号は Stage1 から続く)。
- `--reestimate_obs_norm_terms`: Stage1 では常に 0 だったドリブル用 5 項の actor 正規化統計を、Stage2 の環境で
  200 ステップ集めた値で上書きする。これをしないと Stage2 で実際の値が誤ったスケールで正規化される。
- `--critic_warmup_iters N` (既定 0): 最初の N 回の更新は actor と std を固定し critic だけ学習する。2026-10-20 のランでは未使用。
- PPO: num_steps_per_env 48、学習率 1.26e-4 (adaptive、目標 KL 0.01)、エポック 5、γ 0.99、λ 0.95、clip 0.2。
- **GPU メモリ**: 12GB の GPU (RTX 4070 SUPER) では 4096 env が動かない (rollout storage だけで約 9.8GB)。
  既定は 2048 env / ミニバッチ 16。大きい GPU なら `NUM_ENVS=4096 MINI_BATCHES=8`。
- Python 環境: rsl-rl-lib **3.0.1** と tensordict 0.9.1 が必要 (`requirements-k1.txt`、`docs/env/README.md`)。
  IsaacLab が既定で入れる rsl-rl-lib 5.x では動かない。

---

## 10. 2026-10 の再構築ランの結果

| ラン | 期間 | 結果 |
|---|---|---|
| Stage1 `k1_dribble_target_walk_phase/2026-10-18_14-20-56` | 37.5 時間 | 転倒率 (終わったエピソードのうち転倒の割合) 0.15〜0.25 で横ばい。iteration 3000〜4500 で極端な速度コマンドを導入 |
| Stage2 `k1_dribble_target/2026-10-20_08-19-15` | 43.3 時間 | 最終で Level8、成功率 0.79、転倒 0.12、障害物衝突 0.09、time_out 0、ボールを見失う 0 |

Stage2 の推移 (直近 50 iteration の平均):

| iteration | レベル | 成功 | 転倒 | 障害物衝突 |
|---|---|---|---|---|
| 20999 | 1 | 0.50 | 0.05 | 0 |
| 21999 | 1→3 | 0.67 | 0.02 | 0 |
| 24999 | 4 | 0.31 | 0.12 | 0 |
| 26999 | 8 | 0.51 | 0.13 | 0.24 |
| 39998 | 8 | 0.79 | 0.12 | 0.09 |

- 以前のランで起きていた「Stage2 の開始直後に転倒率が 0.31 → 0.98 に上がる」現象は、今回は起きなかった
  (開始直後 0.01)。
- 転倒率は Level4 (L 字) 以降 0.12 前後に上がり、下がらなかった。以前のランの最終値は 0.035 (到達レベルは記録なし)。
- ボールが body yaw 座標で前方 ±45° にある時間の割合 (`Curriculum/ball_front_ratio`) は最後まで約 0.98。
- レベルごとの動画: `logs/rsl_rl/k1_dribble_target/2026-10-20_08-19-15/videos/levels/`。

---

## 11. 未決の検討事項・既知の問題

1. ~~進級が速すぎる~~ → 2026-10-22 に修正 (各レベル最低 300 iteration、エピソード単位の EMA、前のレベルの混合)。
2. **転倒率が高止まり** (Level4 以降 0.12)。
3. 関節角のリセットはランダム化していない (`reset_joints_by_scale (1.0, 1.0)`、best_walk の系の値。2026-10-22 に
   これを正とした)。リセット時の多様性は、7 の「歩行で始めて途中でドリブルに切り替える」でカバーする方針。
4. **URDF**: チームとしては Hip_Pitch の z = −0.062 が正しい寸法 (rfc_assets 2026-09-07、commit 4b6514c)。
   ドリブルは best_walk に合わせて −0.077 のまま。MuJoCo (−0.062) での sim2sim を寸法の違いへの耐性確認に使う。
   次に歩行を学習し直すときに −0.062 に揃える (実機の FK も同時に更新するので事前に知らせる)。
5. ボール観測の劣化の補間距離 (0.35m / 2.0m) は仮の値。実機 vision_filter のログ (距離ごとの mask と vel_valid の割合) で合わせ込む。
6. 障害物回避はポリシーに学習させない (2026-10-22 決定)。戦略側の経路計画 (pathplan_lib、ダイクストラ法) が障害物を
   避けるウェイポイント列を出し、ポリシーはそれをたどる。
7. 次のサイクルで試す改善 (1 回のランで変えるのは 1 か所だけ):
   critic のウォームアップ (`--critic_warmup_iters`) / 一部のエピソードを「歩行で始めて途中でドリブルに切り替える」形にする。

---

## 12. デプロイ (ONNX)

`python scripts/rsl_rl/export_dribble_target.py --checkpoint <ckpt>` で `<run>/exported/model_<N>/` に書き出す。

- ONNX: opset 18、batch=1、全軸固定、正規化器を含む。
  入力 `command (1, 3)` (Stage2 では常に 0)、`obs_history (1, 100, 59)` (各行 = 1 ステップ分のフレーム、古い → 新しい)。
  出力 `actions (1, 12)` (scale を掛ける前の値)。
- 仕様 JSON: フレームの並び (座標系・単位・学習時のノイズ)、action_scale、既定姿勢、gait_phase、コマンド、
  PD ゲイン・effort_limit、ボールの初期位置、終了条件、ラン ID と iteration。書き出し時に torch と ONNX の出力の一致を確認する。
- 実機側の対応: ドリブル開始時は現在のフレームで履歴全体を埋める。歩容の位相はリセットせず 1.8Hz で引き継ぐ。
  ボール観測は vision_filter から (body yaw 座標、速度が推定できないときは 0)。目標点は /localization_result で body yaw 座標に変換する。

---

## 13. 次の次の候補 (2026-10-22 時点、まだ入れていない)

- V 字・U ターン・8 の字は「曲がる前に減速する」動きが必要。これらのレベルで成功率が伸びなければ、減速を促す報酬を足す。
- 2026-10-20 のランの最終ポリシーは、転倒の直前の速さが 1.3〜1.45 m/s と速かった。実機の安全のため、ドリブル中の速度の上限
  (例: 1.0 m/s) を検討する。
- ボールに触れた直後 0.5 秒は、それ以外の 2〜3 倍転びやすかった (`scripts/rsl_rl/eval_dribble_falls.py` の結果)。
