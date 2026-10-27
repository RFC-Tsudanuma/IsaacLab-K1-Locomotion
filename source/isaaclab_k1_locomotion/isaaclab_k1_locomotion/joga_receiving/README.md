# Joga論文に基づくK1受球タスク

タスクは `Isaac-K1-JogaReceive-v0`。Joga論文のreceivingを、ユーザーが選択した **K1＋IsaacLab/PhysX** で再実装する。公式コードの移植・完全再現ではない。ボール観測は既存のVisionFilter、未来状態・共分散予測、2段MLPへ置き換える。Mocapデータ・PAM・WB-UAN・DSC・キック後phase教師・キック用対称性損失・アブレーション切替は使わない。

原資料はリポジトリの `docs/joga/joga.pdf`（13ページ）、特にAppendix A/C、Tables I–V。公式サイトは <https://joga.csail.mit.edu/>。既存の `Isaac-K1-DirectKick-v0` とそのcheckpoint形式は変更しない。

## 学習・再開・再生

既存リポジトリと同じIsaacLab 2.3.2 / Isaac Sim 5.1、Python環境、K1メッシュを使う。新しい依存パッケージの追加は不要。GPU搭載環境で、最初に2環境・1iterationを確認する。

```bash
.venv/bin/python scripts/joga_receiving/train.py \
  --headless --device cuda:0 --num_envs 2 --max_iterations 1 \
  --log_dir logs/joga_receiving/smoke

.venv/bin/python scripts/joga_receiving/train.py \
  --headless --device cuda:0 --num_envs 16384 --max_iterations 30000
```

既定はTable Iの16,384環境・24steps/environment/iteration。`--num_envs` で実行環境に合わせられる。30,000iterationsは本実装の設定値で、原論文の総学習回数を示さない。最終カリキュラム開始の20,000iterations以降も学習するための値である。自動でW&Bへ送信せず、学習情報はターミナルとCSVに出力する。

```bash
# max_iterationsは通算iteration数。同じ設定で学習を再開する。
.venv/bin/python scripts/joga_receiving/train.py \
  --headless --device cuda:0 \
  --checkpoint logs/joga_receiving/RUN/model_20000.pt \
  --max_iterations 30000

.venv/bin/python scripts/joga_receiving/play.py \
  --device cuda:0 --num_envs 1 --steps 2000 \
  --checkpoint logs/joga_receiving/RUN/model_30000.pt \
  --export logs/joga_receiving/RUN/policy.pt
```

checkpointには設定、モデル・観測正規化、Adam、iteration、総step数を保存する。再開ではAdamの学習率をPPOの適応学習率へ復元し、保存iterationのカリキュラムで環境をresetする。設定・モデル契約が異なるcheckpointや、既存DirectKickのcheckpointは読み込まない。シミュレータ内部状態・乱数生成器の途中状態を復元する機能はなく、同一の物理軌道からの再開ではない。

`learning.csv` はiterationごとのPPO損失・学習率と、そのiterationで完了したエピソードの成功率・転倒率・timeout率・平均return・長さ・最終球速を記録する。完了例がなければrateは空欄。初期resetや途中の手動resetは成績に含めない。成功・転倒・timeoutは重複計数しない。再生結果はcheckpoint隣の `evaluation/metrics.json` へ保存する。

## 論文から適用した仕様

| 対象 | 適用内容 |
|---|---|
| 制御周期 | policy 50 Hz、physics 200 Hz、decimation 4 |
| 方策・価値ネットワーク | 3step観測履歴、独立Actor/Critic、後段MLP `[512,256,128]` ELU、観測の経験平均・分散による正規化、初期action標準偏差1 |
| PPO | 5epochs、4mini-batches、LR 1e-3 adaptive、γ0.99、GAEλ0.95、clip0.2、clipped value loss係数1、entropy0.01、gradient norm上限1。実装は既存環境のRSL-RL 3.0.1 |
| 受球目標 | commanded world ball velocity = 0。停止させ、所定時間保持する |
| 正報酬 | Table IIのball in zone、face ball、ball trapped、settle progress、side-foot first touch、touch velocity kill、near foot、approach loose ball、stand still、upright、poseの式と重み |
| ペナルティ | 適用可能なjoint limits、self collisions、leg-ball contact、action rate、torso angular velocity、angular momentum、pelvis height、foot on ball、sole contact、move when blind |
| 球 | radius 0.11m、mass 0.43kg、hollow-shell inertia `(2/3)mr²` をUSDへ設定 |
| 駆動制約 | Eq.(9)の速度依存torque上限。毎physics stepでPD目標角を制限してeffortを計算する |
| DR | 起動時torso COM xyz±(0.025,0.025,0.03)m、encoder bias±0.015rad。reset時robot xy±0.5m、yaw±π。1–3秒ごとの線速度xyz±(0.5,0.5,0.4)m/s・角速度rpy±(0.52,0.52,0.78)rad/sの外乱 |
| 地形 | 50%flat、50%高さ0–0.01mのrough tile。共有する12m四方・61×61頂点の乱数meshを環境原点へclone。外周高さ0、tile外は平面。地形の実現値・空間解像度は本実装の補完 |

受球カリキュラムはTable IVに合わせる。値はエピソード開始時に取得し、そのエピソード中は保持する。

| PPO iteration | 距離[m] | bearing | lateral offset[m] | 球速[m/s] | 保持時間[s] |
|---:|---|---|---|---|---:|
| 0 | 1.5–2.5 | ±20° | ±0.10 | 1.5–2.8 | 0.5 |
| 8,000 | 1.5–3.0 | ±30° | ±0.30 | 2.0–3.5 | 1.0 |
| 14,000 | 1.5–3.8 | ±40° | ±0.48 | 2.3–4.5 | 1.5 |
| 20,000 | 1.8–5.0 | ±45° | ±0.60 | 2.6–5.5 | 2.0 |

側面接触の `|u_y|` 閾値は別スケジュールで、iteration 0/5,000/10,000/15,000に0/0.35/0.55/0.70。`u` は完全な足姿勢でworld→foot変換した単位ベクトルで、yawのみの近似ではない。自己接触、球と脚・足の接触は部位別の接触sensorを使い、4physics steps内の最大力を利用する。自己接触有無の判定はpairごとの力を先に評価し、反対方向の接触の相殺を防ぐ。

## 独自観測とモデル契約

Actorは1frame 323次元、3frameを古い順に並べた **969次元**を受け取る。

| frame内 | 次元 | 内容 |
|---|---:|---|
| proprioception | 47 | projected gravity3、torso angular velocity3、q−q0の12、dqの12、前action12、commanded world ball velocity3（0）、heading cos/sin2 |
| 予測 | 273 | 13時刻×`[x,y,vx,vy,P(16),horizon/3]`。現在のVisionFilter公開状態からのCV予測とego-motion共分散伝播 |
| 観測状態 | 3 | measurement age / updated / valid |

既存認識のNIS、欠測・再捕捉、カメラ周期・遅延・距離ノイズ・FOV・外れ値・ego-motion noiseは変更しない。原論文のPAM、camera XYZ+visibilityの入力形式、原論文の知覚DRは、この独自方式への明示的な置換対象である。Actorの関節等のノイズも既存K1設定を使用し、encoder biasだけTable IIIの起動時ランダム化を加える。

各frameの273成分だけを `Linear(273,128) → ELU → Linear(128,64)` へ通し、残り50成分と連結する。Actor MLPの入力は `3×114=342`。時刻をまたぐRNN状態はない。3step履歴は明示的な入力bufferであり、resetした環境だけ初期frameで埋め直す。

Criticはノイズなしの同形式履歴969次元に、base線速度3、足高さ2、ground contact2、airtime2、ground force6、body frameの真の球位置・速度6を加えた **990次元**。ノイズなし予測は現在の真値からのCV外挿で、未来の物理シミュレーション真値ではない。共分散は0。Actorとは別の予測MLPを持ち、後段への入力は `342+21=363`。

自己接触やball接触の力を除去して、足のground forceとground contactを構成する。球センサはball1body→各robot linkの個別filter、自己接触は各linkごとに独立したone-to-many sensorで取得する。

TorchScript出力は `[N,969] → [N,12]` の関節actionのみ。観測正規化を含む。既存DirectKickの325次元入力・13要素出力とは別契約で、学習済み重みの変換は行わず、新規に学習する。

## K1・PhysXへの変更と未記載部分の補完

設定の実体は `joga_receiving/config.py`。以下は原論文の数値であると主張しない。

| 項目 | 本実装の判断 |
|---|---|
| 機体 | 既存K1 URDF・脚12関節・既存の基準姿勢とPD gains/effort/velocity limits。腕・腰・首は固定。G1の31関節・能動視線制御は再現しない |
| 固定上半身の報酬 | 腕/腰の姿勢偏差・首速度項は制御対象がなく適用しない。arm-ball接触はURDFの固定joint集約後に腕とTrunkを識別できず、省略する。Trunk-ballペナルティへの代替は加えない |
| pelvis height | G1の0.75m閾値をK1へ適用せず0.50m。既存K1の立位高さを基準にした適用値 |
| 終了 | 最大8秒。gravity.z > −0.5またはbase ground height <0.25mで転倒。保持完了で成功終了。球が後方へ通過しただけでは終了せず、取り戻す時間を残す |
| パス初期化 | yaw frameでdistance・bearingから初期位置を作り、`x=0,y=offset` の交点へ初速を向ける。robot resetと同時に発射。球の初期回転は初速に対応する純転がり |
| reset姿勢 | 基準joint角・関節速度0。K1の足接触点のFKと実際の地形高さからroot高さを求め、3mm浮かせて開始 |
| trapのBoolean | 側面接触経験あり、球がyaw frameの(0.2,0)から0.2m以内かつx>0、球中心がground+radius+0.03m以下、球速<0.5m/s。連続保持時間を加算し、条件を外れたら0へ戻す |
| touch | 足-ball法線力>1N。どちらかの足の接触がfalse→trueになった回数を数える。初回bonusは1回だけ。`I_c`は接触中、`w_k=2^(1-k)`は現在のtouch回数に適用 |
| その他の幾何 | `d_foot`は足link原点と球中心の3D距離。foot on ballはXY距離<0.11mかつ足原点が球中心より上。sole項の`x_hat_foot,z`はworldでの足の+X軸z成分 |
| pose sigma | 全脚joint共通0.5rad。受球はcommand速度0で一定 |
| blind | 現在stepの認識を報酬計算前に更新。filterが測定を採用したかをカメラframe間で保持し、2policy steps非可視でblindとする。カメラframeのないstep単独では欠測扱いしない |
| 報酬の時間係数 | Table IIの重みを各stepの0.02秒で乗算。初回接触bonusも同じ時間係数。原論文はdt適用の詳細を記載していない |
| PPO補完 | 適応LRのdesired KL=0.01、action標準偏差はlogパラメータで学習。数値未記載部分にはRSL-RLの標準方式を使う |

### 接触モデルの限界

MuJoCoのsolref/solimp、接触優先度、torsional/rolling frictionをPhysXの同じ数値とみなして転記しない。特にdamping ratio0.18/0.14をrestitutionへ転記しない。

本実装はK1の既存足collision形状を使用する。Table Vのball-ground sliding friction `[0.5,0.9]`、foot-ground friction `[0.3,1.2]` をreset時に設定する。materialのfriction combineをminとし、ground側の係数2.0はそれらの範囲を切り詰めないための実装値。**foot-ball摩擦はmin(foot,ball)となり、論文の独立した0.3には一致しない。** per-pair専用collider・7本sole capsuleへの変更は行っていない。

ball-groundのrestitutionは `sqrt(0.4)`。理想的な剛体衝突なら2m落下で0.8m反発する値だが、**PhysXでFIFA試験に合格したと実測した値ではない**。groundはmax合成、robotはmin合成、ballはaverage合成とし、foot-ballのrestitutionは0。これも原論文のMuJoCo接触モデルの同定結果ではない。

線形・角減衰と追加のrolling resistanceは与えていないため、純転がりの減速特性はTable Vと一致しない。法線接触sensorを使うので、報酬閾値を適用する力は法線力であり、接線摩擦力を含む総接触力ではない。これらは実機性能・論文との比較で残る物理差分である。

## 検証と記録

CPUテストは以下で実行する。外部ROS用pytest pluginを自動読込しない。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/joga_receiving tests/direct_kick -q
```

カリキュラム境界、パスの交点・速度、Eq.(9)、角運動量、報酬重み・初回接触・touch減衰・連続保持、完全な足姿勢での側面判定、部分reset、地形三角形と高さ、実VisionFilter処理、native境界の履歴・真値分離・接触力分類・終了優先度、PPO学習・Adam再開・TorchScriptをCPUで確認する。

CPUテストはPhysXシミュレーション・学習収束・実機受球の成功を保証しない。原論文の最大5.2m/s実演との同等性能も主張しない。

2026-10-27の検証では、既存DirectKickの回帰を含め66 tests / 262 subtestsが成功した。2環境・1iterationのnative起動も試したが、実行環境に `libcuda.so.1` がなく、PhysX tensors/fabric pluginの初期化が失敗し、学習ループには到達していない。GPU搭載環境での起動・接触sensor・材質・物理学習の確認は未完了。

ADR: not required。ADR保存規約がないため、新しいADRを作成せず、この文書にユーザー選択と再現範囲・補完判断を記録する。
