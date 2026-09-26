# IsaacLab DirectKick の実装・学習

タスク名は `Isaac-K1-DirectKick-v0`。元の観測・制御・報酬ロジックと PPO を専用の DirectRLEnv／runner に接続した。標準 RSL-RL runner には置き換えていない。

実装は追加済みだが、**GPU シミュレーションでの学習確認は未完了**。この作業環境では Isaac Sim 5.1 の起動後、`libcuda.so.1` が見つからず PhysX 初期化が停止した。CPU の数値・ロジック・runner テストと、実物理での検証は分けて扱う。

## 学習・再生

このリポジトリで使用している IsaacLab 2.3.2 / Isaac Sim 5.1 と NVIDIA GPU・ドライバが必要。Git clone全体をLinux上で使用する（既存の `assets_soccer` とメッシュの相対シンボリックリンクを含む）。Pythonパッケージだけを別の場所へコピーする配布は対象外。依存関係はルートで `uv sync`、または使用中の IsaacLab Python に `pip install -e source/isaaclab_k1_locomotion` で反映する。W&B を使用する既定設定のため、package dependencies に `wandb` を追加している。

まず GPU 上で 2 環境・1 iteration の確認を行う。

```bash
.venv/bin/python scripts/direct_kick/train.py \
  --headless --device cuda:0 --num_envs 2 --max_iterations 1 \
  --no_wandb --log_dir logs/direct_kick/smoke
```

通常の学習例。元 shell と同じ既定値は 8192 環境、seed 42、最大 200000 iterations。GPU メモリに応じて `--num_envs` を指定する。

```bash
.venv/bin/python scripts/direct_kick/train.py \
  --headless --device cuda:0 --num_envs 8192 --no_wandb
```

`--no_wandb` を外すと W&B に記録する。環境0の報酬・状態 CSV は既定で有効、無効化は `--no_csv`。学習損失は `learning.csv`、全環境の完了エピソード集計は `episode_metrics.csv`、実効設定は `config.yaml`、モデル／認識契約は `policy_contract.json` に保存する。

```bash
# 同じ移植実装の checkpoint から再開。max_iterations は通算の到達値。
.venv/bin/python scripts/direct_kick/train.py \
  --headless --device cuda:0 --num_envs 8192 --no_wandb \
  --checkpoint logs/direct_kick/RUN/model_500.pth

# 確定 action で再生。--headless も指定可能。
.venv/bin/python scripts/direct_kick/play.py \
  --device cuda:0 --checkpoint logs/direct_kick/RUN/model_500.pth \
  --export logs/direct_kick/RUN/policy.pt
```

checkpoint は500 iterationsごとと終了時に保存する。学習終了時には `policy.pt` も出力する。TorchScript Actor の入力は `[N,325]`、出力は **12 action + phase確率の13要素**。環境に渡すのは先頭12要素。phaseは関節指令ではない。LSTM の hidden state は呼出間で持ち越さない。

位置だけを符号化する旧132次元の移植checkpointと旧Gym checkpointは、観測形状・意味が異なるため再開には使えない。新しい入力契約で学習し直す。`port_metadata` でモデル契約と認識 revision を照合する。既存歩行タスクの checkpoint も対象外。

## ターミナルの学習表示

2026-09-26から、移植前のIsaacLab／RSL-RLに合わせた複数行表示を各iterationで出力する。`Learning iteration`、学習速度（steps/s、収集・更新時間）、action noise、各損失、entropy・KL・学習率、報酬内訳、完了数・キック成立率・転倒率、総step数、経過時間・ETAを表示する。

- `Mean reward` / `Mean episode length`：直近100件の完了エピソードの報酬合計／step数の平均。未完了エピソードは含めず、1件も完了していない間は表示しない。
- `Episode_Reward/<name>`：当iterationで完了した各エピソードの重み適用済み報酬合計を平均し、エピソード上限時間（17秒）で割った値。IsaacLabの正規化単位に合わせる。完了エピソードを同じ重みで扱い、当iterationに完了例がない場合は内訳を表示しない。リセットが同時に起きた群ごとの平均や前iterationの値の再表示にはしない。
- 報酬はPPOによるtimeout bootstrap用の書換前に集計する。途中のエピソードの累積はiterationをまたいで保持する。学習開始・再開時の環境resetで表示用の集計を初期化する。
- ETAは今回の実行で消化したiterationの平均所要時間から算出する。checkpointに保存済みのiteration数を経過時間の分母に使わない。

表示用の集計はPPO・環境へ値を戻さない。既存の`learning.csv`、`episode_metrics.csv`、W&B記録とcheckpoint形式は維持する。`learning.csv`の`reward`は従来どおりrolloutのstep平均なので、ターミナルの`Mean reward`とは集計単位が異なる。

表示修正後のCPU検証：47 tests / 258 subtests 成功。報酬の完了時点・反復間の保持・直近100件・入力テンソルの非変更・学習ループの表示値・再開後ETAを確認した。

## 再開と成績の集計

checkpoint再開ではAdamを読み込んだ直後、その学習率をPPOの適応学習率にも復元する。従来形式のcheckpointに保存済みのoptimizer情報を使い、形式は変更しない。`resume=False` の重み読込ではoptimizerと学習率は初期設定のまま。同一rolloutによる2回の追加更新で、中断なしの場合と再開後の全重み・Adam状態・学習率が一致することをテストする。

`episode_metrics.csv` は全環境を対象とし、学習時はiteration内に**完了したエピソード**を集計する。再生時は終了時に、その実行で完了したエピソード全体をcheckpointディレクトリ配下の `evaluation/episode_metrics.csv` に出力する（再生を再実行すると置換）。初期reset、途中での手動reset、再生終了時に未完了のエピソードは分母に含めない。学習再開後の集計は新しい実行での完了分から始める。

- `all`、`stationary`、`moving`：全体／静止／移動。区分は生成時の抽選結果に固定する。
- `speed_0_1_mps` ～ `speed_5_6_mps`：移動ボールの初速を1 m/s幅で区分。下限を含み上限を含まないが、最後の区分は6 m/sを含む。
- `offset_0_0p25_m`、`offset_0p25_0p50_m`、`offset_0p50_0p75_m`：移動ボールの生成時の最接近ずれの絶対値で区分。同じく下限を含み、最後のみ上限0.75 mを含む。静止ボールは軌道の最接近ずれを持たないため除外する。
- `episodes` と `kicks`、`kick_rate`：完了数、そのうち既存の `valid_kick` が成立した数・割合。**方向の正しいパス成功率ではない**。キック後に転倒した場合もキック成立には含め、転倒も別に記録する。従来の `env_successes` も、この完了エピソード内のキック成立数へ統一する。
- `falls`、`timeouts`、`post_kick_completions`、`other_terminations` と各rate：終了理由。重複時は転倒→時間切れ→キック後の所定時間完了→その他の順に分類する。報酬や終了判定には使用しない。

完了例がない区分のrateはCSVでは空欄、W&Bには送信しない。0%と未評価を区別し、件数も併記する。W&Bでは `episodes/<group>/<metric>` に記録する。終了状態をreset前に読み、二重計上を防ぐ。記録は環境ごとの固定サイズの区分マスクと件数カウンタのみで、未使用だった無制限のボール速度リストは削除した。

後方へ通過した場合の終了条件・報酬はユーザー指示により維持する。現在の待機ペナルティは未キック中、2秒まで二次的に増加し、その後は毎step同じ額を加算する（係数−0.5、dt=0.02なので2秒以降は−0.01/step）。これは報酬の一項であり、総報酬や実際の見送り頻度への影響は学習結果で評価する。

## 承認された変更

| 項目 | 実装内容 |
|---|---|
| VisionFilter | ローカル `futbol_main/main` の `32ece6ee0676b1008d5bc58c3533d45613440568` を固定。stationary / rolling / high_speed / bounce の4仮説、MAP選択、bounce再初期化、confirmed/tentative 2観測取得を移植 |
| NIS・欠測・再捕捉 | 元 DirectKick の単一KFを使わず、VisionFilterの NIS 9.21、strict `>3 s` timeout、再捕捉処理を使用 |
| LSTM入力 | 2026-09-25の明示要求により、13時刻それぞれの位置・速度と4×4共分散全成分を入力。Actorのボール速度をMLPへ直接渡す経路を廃止。Critic特権の真値速度は維持 |
| ボール初期条件 | 静止10%、移動90%。移動時は0〜6 m/s・最頻値3 m/sの対称三角分布で、接近方向のみ（`incoming_probability=1.0`）。軌道の符号付き最接近距離は±0.75 mの一様分布 |
| 生成距離 | 移動時は最接近時間Tを1.0〜1.4秒で一様抽選し、速度v・通過ずれbから `d=max(1.5, sqrt((v*T)^2+b^2))` mで決定。静止時は1.5〜3 mの一様分布 |
| 視認距離 | 高速時の最大生成距離約8.43 mに対応し、視認上限を6 mから9 mに拡大 |
| rolling friction | ボールへの係数適用を省略。代替の転がり減速度や抵抗力は追加しない |
| compliance | 足shapeへの係数適用を省略。新APIのspring stiffness/dampingへの換算は行わない |
| 通常の摩擦・反発 | 地面、足、ボールの元のランダム化範囲を適用 |

認識のQ/Rは最新VisionFilterの固定値を使う。旧DirectKickのQ/R倍率サンプルとrolling frictionによるQ補正は使用しない。実際のカメラ測定値に加える距離依存ノイズ、カメラ周期、遅延、FOV、dropout、外れ値、ego-motion noise は移植元を維持する。

ボール生成の変更も元YAMLを出典として保持したまま `load_config()` で実効設定へ反映する。接近ボールの旧 `incoming_spawn_distance_range_m` は、速度と時間から距離を求める設定へ置き換えた。速度分布は独立な一様乱数2個の平均による対称三角分布で、静止を選んだ環境は並進・回転速度を0にする。10%は各リセットでの抽選確率であり、各バッチの正確な比率ではない。

最接近時間は、ロボットが静止し、ボールが初期速度で直進した場合の基準。距離下限1.5 mが働く低速時は1.4秒を超えることがある。静止ボールに到達時間は設定しない。移動ボールの約2/3は初期直線軌道が中心から0.25 m以上ずれるが、身体形状・ロボットの動作を含む非衝突率を保証する値ではない。実際の衝突率・キック成功率はGPUシミュレーションでの確認が必要。

## 公開状態から Actor 325 への接続

VisionFilter内部はfloat64のSI単位。実機のcm内部計算と同等になるよう、校正値をSIへ換算した。公開スナップショットはfloat32の `[x,y,vx,vy]` と4×4共分散。LOST時には公開位置・速度・共分散をゼロにする。

- 初期共分散は `diag(0.25²,0.25²,2.5²,2.5²)`。
- Q は加速度標準偏差0.8 m/s²による単一区間のCV行列。
- R は観測local距離[m] × `[[0.0477733905268419846,-0.0004131492504468594],[-0.0004131492504468594,0.0153060614974156124]]`。Rの距離clamp・yaw回転は追加しない。
- フィールド外観測gateは main が読む22×14 mフィールド＋5 m margin、すなわち `abs(x)>16` または `abs(y)>12`。各環境の座標で評価する。
- フィルタbankを進めるのはカメラframe処理時。見えないframeも欠測として渡す。policy周期に合わせてbankを余分にpredictしない。
- 実機公開値に仮説IDはないため、13 horizonは公開state/Pだけから `F(t) state`、`F(t) P F(t)^T + Q(t)` で生成する。`t = 現在時刻 − 公開snapshot時刻 + horizon`。選択仮説の隠れた状態は使わない。
- ego poseは現在の観測からhorizon分だけ予測する。snapshotの経過時間を二重加算しない。
- Actorは13×21 token、measurement age / updated / valid、目標方向を受け取る。特権20次元には元と同じ真値・ノイズを用いる。

PowerPointの6・7枚目にある各時刻の「Σ・v・x → LSTM」に合わせ、1 tokenを次の順序で入力する。

```text
[x, y, vx, vy, P00, P01, P02, P03, P10, ..., P33, horizon / 3.0]
```

状態はその予測時刻の自機座標系。4×4共分散の行・列の順序も `[x,y,vx,vy]`。16成分を行優先で格納し、位置・速度間の相互共分散も保持する。対称要素も省略しない。有効時の共分散に対数化やクリップは行わない。位置/速度の観測係数を並べた対角行列Dにより、状態はD s、共分散はD P Dᵀで同じ単位系へ変換する（現行係数はどちらも1.0）。

| Actor入力 | 次元 | 経路 |
|---|---:|---|
| 従来の歩行観測 | 47 | MLPへ直接 |
| 13時刻の位置・速度・全共分散・時刻 | 273 | 21入力のLSTM → 最終hidden 64 |
| measurement age / updated / valid | 3 | MLPへ直接 |
| 目標方向 | 2 | MLPへ直接 |
| 合計 | 325 | 符号化後のActor MLP入力は116 |

Actorのボール位置・速度・共分散はすべてLSTM経由。Criticにも同じ形式の観測を専用LSTMで符号化して渡し、特権20次元を加える（MLP入力136）。特権のボール速度2成分は引き続き世界座標系のシミュレータ真値。LSTMは未来horizonを処理し、制御stepをまたいでhidden/cell stateを持ち越さない。

共分散の座標変換も速度の定義と揃える。Rをworld→local回転、Jを90度回転行列、ωを自機yaw rateとすると、p=R(p_ball−p_ego)、v=R v_ball−v_ego−ωJp。ball側のヤコビアンは `A=[[R,0],[-ωJR,R]]`。ego側は位置・yaw・並進速度・yaw rateのヤコビアンBを使い、`P_local=A P_ball Aᵀ+B P_ego Bᵀ` とする。

自機誤差は元の独立XY軸・積分random walk近似を維持し、位置↔速度、yaw↔yaw rateの共分散を加えて全状態へ伝播する。従来の位置2×2共分散は変わらない。これは旋回軌道の厳密な誤差積分ではなく、yaw等の変換にも一次近似を使う。ball/ego誤差間は従来同様に独立と仮定する。

LOST時の無効マスクとstatusは維持。無効tokenは推定分布ではなくプレースホルダーであり、状態4成分をゼロ、位置分散を従来の最大log-std表示に対応する値 `(0.01 exp(4))²`、追加した速度分散・相互共分散をゼロとする。ゼロ速度分散を高信頼な推定値と解釈せず、valid=0と合わせて扱う。元YAMLは出典資料として132次元のまま保存し、`load_config()`で実効観測数325を設定する。モデル契約は `direct_kicking_horizon_lstm_state_covariance_v3`。

この処理はROS nodeの起動を必要としない。実機側のROS通信・policy切替への組込みはこのリポジトリの変更には含まない。

## 元仕様を維持する境界

URDFは移植元の原本を `direct_kick/assets/K1` に保持。参照する24 STLは既存の `assets_soccer/booster_robotics_robots/K1/meshes` と全てSHA-256が一致するため、`direct_kick/assets/K1/meshes` から同ディレクトリへの相対シンボリックリンクで再利用する。STLの重複コピーは追加しない。既存URDFとの差である腕固定角（左−1.35、右＋1.35 rad）は移植元のまま維持する。関節は名前で元の左脚6→右脚6の順へ並べ替える。Labのwxyzと元ロジックのxyzwを境界で変換し、各環境の配置原点を引く。actor/criticの観測にclone配置座標を混入させない。

質量・COMは環境ごと・bodyごとにランダム化し、`sim.reset()` 前にUSDへ書き込む。対角慣性と主軸属性を自動計算指定へ戻し、PhysXにcollision geometryと指定質量・COMから再計算させる。単純な質量比での慣性近似は使用しない。

body状態は元の足原点補正・全身COM計算が期待するCOM位置＋link姿勢へ対応づけている。ただし、旧Gym SDKがこの環境にないため、**旧Gymが返すbody位置との実測同値は未確認**。足接地位置・COM・初期姿勢はGPUでの移植検証項目になる。

外力は元Gymの「次の1 physics timestepに作用する」扱いに合わせ、Labの瞬時wrenchを使う。永続wrenchで10 substepsすべてに加力しない。COMへの加力をlink原点での同等な力・トルクへ変換する。[PhysXの力の寿命](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxArticulationLink.html)

元コードのadvanced indexing後の `.zero_()` により外力bufferが消えない挙動、およびresetでaction履歴の一部を保持する挙動は修正していない。旧DOF propertyへ書いたvelocity/armatureの値がGymへ適用されていない点も踏襲する。報酬・観測用の制限値とシミュレータに適用される値を区別する。

終了判定→当該stepのキック検出・報酬→reset→観測の順を維持。通常34報酬はdt=0.02倍、初回方向報酬はdt倍なし。phase教師はstep前の観測に対応する値をrolloutへ保存する。PPOのtimeout報酬置換、full-batch勾配蓄積、対称性、class-balanced phase loss、全epoch後のKLによる学習率調整を維持する。

Gym固有のCPU thread/subscene数・buffer倍率はLabの公開設定にそのまま対応しないため原本に保存し、Lab側では対応する物理周期・solver反復数・接触設定を明示する。contact sensorはphysics stepごとに更新し、制御step末尾の値を読む。シミュレータとURDF importerが異なるため、上記の係数維持を物理軌跡の数値的一致と同一視しない。

## 検証

2026-09-25の学習再開・成績集計の修正後、**40 tests / 258 subtests 成功**。初回移植時には学習CLIの引数読込、元YAML・2 URDF・24 STLのバイト一致も確認した。出典ハッシュは [implementation_provenance.json](implementation_provenance.json) に記録している。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/direct_kick
```

- 固定元Runnerの独立AST oracle：同じ新モデルの複製を両学習器へ渡し、full-batch／不均等chunkで損失、全重み、Adam、timeout、KL、学習率を比較。旧モデルと新モデルの数値同値を主張するテストではない。
- 固定元VisionFilter C++から生成したoracle：stationary共分散、MAP／bounce、取得・欠測・再捕捉、3秒境界、複数環境を比較。rollingとhigh_speedは同じ運動モデルのため、状態・共分散が一致する数値的同率だけ選択ラベル差を許容。
- tensor backend：Actor325／特権20、報酬・終端、phase、action delay、reset、実perception処理と13 horizonの共分散を確認。
- 独立な有限差分ヤコビアンで4×4共分散変換を照合。全16成分の左右反射、13時刻の位置・速度・共分散の実入力とD P Dᵀ正規化、LSTM以外へのボール入力経路がないことを確認。
- ボールreset：128環境×2回、異なるロボット位置・yawで静止／接近方向、通過ずれ、最接近時間を確認。8192回の生成結果から静止約10%・三角分布の累積確率・通過ずれの分布を確認。0 / 0.1 / 1 / 3 / 6 m/sの境界例で距離下限・速度連動・最大距離・回転速度も検証。
- fake環境でのrollout→PPO→checkpoint再開→13出力TorchScript保存・再loadを確認。適応学習率が変わったcheckpointから同一rolloutを2回更新し、中断なしの場合との全重み・Adam・学習率の一致を確認。
- 完了エピソードの集計：生成条件の保持、カテゴリ境界、空欄と0%の区別、集計後も継続するエピソード、初期／手動reset除外、二重計上防止、キック後2秒終了時の成立数、CSV出力を確認。native backendの実resetメソッドを使い、Labがepisode lengthをクリアする前に集計されることも確認。
- Labの実書き込み処理をASTで実行し、10 physics writes中の外力適用が1回であること、COMまわりのモーメント保存、環境原点とquaternionの変換を確認。

これらはGPU物理のテストを代替しない。再現用C++driverと固定入力は `tests/direct_kick/fixtures/vision_filter`、元PPOのfixtureは `tests/direct_kick/reference` に保存している。

ADR: not required。対象のADR保存規約は確認できず、今回合意した認識・物理の変更はこの文書に記録する。
