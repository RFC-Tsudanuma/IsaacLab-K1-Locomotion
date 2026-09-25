# DirectKicking 移植元仕様と IsaacLab 実装

`rl_humanoid_htwk` の `direct_kick/safety` にある **K1/DirectKicking** の仕様を、IsaacLab 移植の参照用に保存する。
当初の仕様コピーに続き、IsaacLab環境・認識・専用PPO runnerを追加した。**現在の学習方法、承認された変更、検証状況は [実装・学習手順](implementation.md) を参照**。現行実装は位置・速度・4×4共分散をLSTMへ渡すActor325次元／特権20次元。以下の仕様書と原本は移植元の記録であり、旧CVKFの設定値もそのまま保存している。

| 項目 | 固定したコピー元 |
|---|---|
| リポジトリ | `/home/akage/futbol_local/rl_humanoid_htwk` |
| ブランチ | `direct_kick/safety` |
| コミット | `3af2acc97f1081b4cbfd9556efd39408dea92cc6` |
| 環境クラス | `DirectKicking → KickingK1 → BaseTask` |
| シミュレータ | Isaac Gym / PhysX |
| 設定 | `envs/K1/DirectKicking.yaml` |
| 学習入口 | `train_direct_kicking_k1.sh → train.py → utils/runner.py` |

仕様書の `path:line` は、上記コミットのコピー元ファイルを指す。設定値は原本 YAML を正とし、実装内の定数、処理順序、上書き、未使用の設定は各仕様書で補足する。説明はソース読解によるもので、シミュレータ実行による動作保証ではない。

## 内容

| ファイル | 内容 |
|---|---|
| [実装・学習手順](implementation.md) | IsaacLab task、学習・再開・再生、最新VisionFilterへの変更、物理差、検証範囲 |
| [観測・認識](observations.md) | Actor 132 次元、特権観測 20 次元、座標系、ノイズ、CVKF、未来予測 |
| [環境・制御](environment.md) | ロボット・ボール、物理、12 関節制御、reset、ボール初期運動、ランダム化 |
| [報酬・終了](rewards.md) | 通常報酬全項目、初回キック報酬、キック検出、終端、phase 教師信号 |
| [モデル・学習](training.md) | Actor/Critic の LSTM・MLP、PPO、対称性、補助損失、checkpoint、入口の既定値 |
| [DirectKicking.yaml](source/DirectKicking.yaml) | コメント・無効な項目を含む設定原本のバイト単位コピー |
| [K1_locomotion.urdf](source/K1_locomotion.urdf) | 質量、慣性、関節・リンク・形状参照を含むロボット定義原本 |
| [ball.urdf](source/ball.urdf) | ボール形状、質量、慣性の定義原本 |
| [provenance.json](provenance.json) | 参照元コミットと各ファイルの SHA-256、サイズ、原本のコピー先 |

URDF のメッシュ参照は原文のまま。仕様資料・実装にはSTLを重複配置せず、実装の `direct_kick/assets/K1/meshes` は既存の `assets_soccer/booster_robotics_robots/K1/meshes` への相対シンボリックリンクとする。参照する24個のSTLは移植元と同一で、URDFの値とメッシュ参照名も原本を維持する。出典とハッシュは `provenance.json` にある。学習済み重み・ログは含めない。原本にある `asset.mujoco_file = resources/K1/K1_locomotion.xml` は、固定したコミットに対応ファイルが存在せず、調べた学習経路でも使用されない。

## 全設定の所在

YAML の全 18 トップレベル節を削除・再構成せず保存した。

| 節 | 主な意味 | 説明 |
|---|---|---|
| `basic` | task、model、seed、device、反復数、ログ | モデル・学習 |
| `env` | 環境数、観測・action 次元、spacing | 環境・制御 |
| `runner` | rollout、更新回数、chunk、保存・動画 | モデル・学習 |
| `viewer` | カメラ、動画 | 環境・制御 |
| `algorithm` | 学習率、discount、損失係数 | モデル・学習 |
| `sim` | dt、重力、PhysX | 環境・制御 |
| `asset` | K1、足、衝突、asset options | 環境・制御・URDF |
| `ball` | ボール定義、物性、初期値 | 環境・制御・URDF |
| `init_state` | 初期姿勢・速度 | 環境・制御 |
| `control` | PD、制限、armature、decimation | 環境・制御 |
| `terrain` | plane と未選択の trimesh 設定 | 環境・制御 |
| `commands` | ゼロ速度指令、キック目標、curriculum | 環境・制御 |
| `direct_kicking` | 初期姿勢、対称性、観測、認識、結果、物性 | 各仕様書 |
| `normalization` | 観測倍率、action clip、速度 filter | 観測・認識・環境・制御 |
| `vision` / `noise` / `randomization` | 可視域、ノイズ、物性・外乱 | 観測・認識・環境・制御 |
| `rewards` | 全報酬係数と補助パラメータ、終端 | 報酬・終了 |

## 移植元の契約

以下は仕様コピー時の基準。VisionFilterへの置換、LSTM入力の全状態・共分散化とActor325次元への変更、rolling friction・complianceの省略については、後から承認された [実装差分](implementation.md#承認された変更) が優先する。

- Actor 132 次元と特権観測 20 次元の順番・意味を維持する。Critic は両者を利用するが、単純な 152 次元 MLP ではない。
- Actor のボール入力は認識処理と CVKF の出力。シミュレータ真値を直接代入しない。既存の NIS gate、欠測、遅延、無効時の表現も仕様に含む。
- Actor/Critic の LSTM は 13 個の未来 horizon を符号化する。時系列の recurrent policy へ置き換えない。
- 物理 dt 0.002 s、制御 dt 0.02 s、物理 step 単位の action delay と明示 PD を区別する。
- YAML の通常報酬には dt を掛け、別加算の `kick_direction` には掛けない。終了判定・報酬・reset・観測の順序を保つ。
- Quaternion は元コードで xyzw。足位置の特権観測は元シミュレータの位置 tensor の XY をそのまま使う。IsaacLab の環境原点・body frame の意味を確認してから対応づける。
- ボールの rolling friction、足の compliance、PhysX/asset の各設定は移植先での対応確認が必要。未対応値を別の物理モデルで代用する判断は、この仕様コピーには含めない。
- 元の PPO の timeout 処理、full-batch 更新、対称性・phase 補助損失を含めて学習契約とする。標準 RSL-RL の既定値で置き換えた状態を同一仕様とは扱わない。

ADR: not required。対象プロジェクトのADR保存規約は確認範囲では見つからなかった。今回合意した実行方式は [実装・学習手順](implementation.md) に記録した。
