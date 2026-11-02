# ドリブルのポリシーを booster_k1_locomotion に入れる (後日の作業)

このディレクトリには、学習したドリブルのポリシーを実機・MuJoCo で動かすための一式を置いている。
推論ノードの本体は futbol_main のサブモジュール
[booster_k1_locomotion](https://github.com/RFC-Tsudanuma/booster_k1_locomotion) の `rl_policy_isaaclab_node` にあるので、
**後日、ここの一式を booster_k1_locomotion に入れる**。それまで booster_k1_locomotion は main のまま (ドリブルなし) にしておく。

作成: 2026-11-02 / ブランチ `feat/dribble_target_rebuild`

## 1. 置いてあるもの

| ファイル | 中身 |
|---|---|
| `booster_k1_locomotion_dribble.patch` | booster_k1_locomotion への変更 (`git format-patch`)。元になったコミットは `5c8178c` (PR #17 のマージ)。モデルの ONNX は含まない |
| `model/k1_dribble_target_2026-11-01_12-04-42_model_74991.onnx` | 学習済みのポリシー (ラン `logs/rsl_rl/k1_dribble_target/2026-11-01_12-04-42`、ボールの観測に vision_filter の模倣を入れたもの) |
| `model/k1_dribble_target_2026-11-01_12-04-42_model_74991_spec.json` | 同じモデルの仕様 (`scripts/rsl_rl/export_dribble_target.py` が書き出したもの。観測の並び・通過判定・終了条件など) |

## 2. booster_k1_locomotion でやること

```bash
cd futbol_main/src/ros2_ws/src/booster_k1_locomotion
git checkout -b feat/dribble_policy origin/main
git am -3 <このディレクトリ>/booster_k1_locomotion_dribble.patch
cp <このディレクトリ>/model/k1_dribble_target_2026-11-01_12-04-42_model_74991* assets/
git add assets/k1_dribble_target_2026-11-01_12-04-42_model_74991*
git commit --amend --no-edit
cd ../.. && ./build_pkg_select.sh booster_k1_locomotion
```

- `5c8178c` にはそのまま当たる。2026-11-02 時点の origin/main (`062427c`: pivot_model_path の既定を変更、`736038f`: V キーのカメラ視野) では、
  `launch/k1_isaaclab_sim.launch.py` の引数の宣言の所が 1 か所だけぶつかる (試して確かめた)。パッチ側の
  `dribble_model_path`・`log_ball_contacts`・`dribble_ball_world_source` の 3 つをそのまま残し、その後ろの
  `pivot_model_path` の既定だけ main 側の `default_value=default_pivot_model` にする。直したら `git add` して `git am --continue`。
- futbol_main 側で、サブモジュールの参照を新しいコミットに更新してコミットする。

### パッチの中身

| ファイル | 変更 |
|---|---|
| `src/rl_policy_isaaclab_node.cpp` | `DribbleBehavior` (下の 3 章)。`/soccer_dribble_cmd` (nav_msgs/Path) の受信、ボールのフィールド座標 (vision_filter の `filtered_ball.global_position`) の取り込み |
| `src/k1_dribble_constants.hpp` (新規) | 観測の並び (59 次元)、ウェイポイントの通過判定・成功判定の定数 |
| `launch/k1_isaaclab_deploy.launch.py` / `k1_isaaclab_sim.launch.py` | `dribble_model_path` (既定は空 = ドリブル無効)、`dribble_cmd_topic`、`dribble_ball_world_source` |
| `resource/k1_kick_params.yaml` | ドリブルの開始窓・見失い・時間切れ (`dribble_ball_*`) |
| `src/mujoco_sim_node.cpp` | 足とボールの接触の記録 (`log_ball_contacts`、既定 false、sim 専用) |
| `scripts/` (新規 4 つ) | MuJoCo でコースをドリブルさせるテスト (`run_dribble_mujoco_test.sh`)。8 の字は `--shapes figure8 --timeout 60` |

## 3. 推論ノードの振る舞い (学習に合わせてある所)

- 入力は `command` (1, 3) が常に 0、`obs_history` (1, 100, 59)。先頭 49 次元は歩行と同じ並び、後ろ 10 次元は
  目標点・ボール・ボールの速度・左右の足→ボール (すべて body yaw 座標)。歩容の位相は 1.8 Hz で回し続ける。
- ボールは vision_filter の出力 (撮影時刻のロボット座標)、目標点は遅れた自己位置で見る。学習の観測の遅れの作り方と同じ。
- Walk → Dribble は学習の開始窓 (正面 0.6〜1.0 m・方位 ±19°・ボール 0.3 m/s 未満) のときだけ。入るときに履歴を最初のフレームで埋め、歩行の位相を引き継ぐ。
- ウェイポイントの通過: 0.3 m 以内、または入ってきた区間に垂直な線を越えた (1.0 m 以内に限る)。
- 終了: **最後のウェイポイントに進んでから**、ボールとロボットの両方が目標から 1 m 以内 (学習の `ball_reached_target` と同じ。
  8 の字・O 字は終点 = 始点なので、この条件が無いと走る前に終わる)。ほかにボールが 4 m 超、1 s 見失い、60 s。
- 同じ経路を送り直しても、ドリブルは最初からにならない (点の数が違うか、どれかが 0.2 m 以上動いたときだけ入れ替える)。空の経路で取り消し。

## 4. 確認の結果 (MuJoCo、vision_filter 経由、ボールの遅れ 100 ms・自己位置の遅れ 170 ms)

| モデル | 全コース 30 試行 (成功 / 転倒) | 8 の字 |
|---|---|---|
| model_69992 (観測に模倣なし) | 0.40 / 0.60 | — |
| model_69992、ボールの観測を「遅れだけの真値」に差し替え | 0.90 / 0.10 | — |
| model_74991 (このディレクトリのモデル) | 未測定 | 6 試行で成功 4・転倒 2 (途中で打ち切り) |

model_74991 の全コースの 30 試行は、まだ測っていない。入れたら最初に測る:

```bash
SIM_ARGS="self_position_delay_ms:=170.0 detection_delay_ms:=100.0" scripts/run_dribble_mujoco_test.sh --repeats 5
# 実機寄り (検出のノイズとカメラの視野) でも
SIM_ARGS="self_position_delay_ms:=170.0 detection_delay_ms:=100.0 detection_position_noise:=true ball_detection_only_in_camera_fov:=true" \
  scripts/run_dribble_mujoco_test.sh --repeats 5
```

## 5. 戦略 (futbol_main の strategy) の状態

- ブランチ `strategy/dribble_ogawa` のコミット `ec6286c7` (2026-11-02 時点で未 push)。
  両ペナルティエリアの前端の間、通常プレー (キックオフ・セットプレー以外) で、ゴールへのキックの代わりにドリブルを選ぶ。
  `task_config.dribble_in_midfield` は既定 false。sim の試合で片方のチームだけ試すときは環境変数 `STRATEGY_DRIBBLE_ROBOT_IDS=1,2,5`。
- 既知の問題 (未対応): ドリブルを始める位置へ回り込む `dribble_ball_position` で、目標の「ボールの後ろ 80 cm」が
  経路計画のボールの避ける半径 (`pathplan_clearance` 80 cm) の境目に乗り、経路が見つからずに立ち止まる。
  3v3 の試合で、ドリブルを選んでから経路を送るまで 2〜8 秒かかっていた。
