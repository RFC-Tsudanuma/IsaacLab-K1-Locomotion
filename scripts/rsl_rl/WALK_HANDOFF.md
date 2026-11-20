# 歩行学習 引き継ぎ (2026-11-12)

別 PC で歩行学習を続けるエージェント向けの前提知識。これ 1 ファイルを読めば開始できる。

> **2026-11-19 追記あり。** 現状・採用モデル・次にやることは末尾の
> 「追記 (2026-11-19)」が最新。以下の本文は 2026-11-12 時点の記述で、
> 「次の目的」「現在の最良ポリシー」「推奨プラン」は追記の内容で置き換わっている
> (学習の回し方・検証プロトコル・下流との契約は今も有効)。
詳細な実験経緯は `scripts/rsl_rl/phase_freq_weight_tuning.md`、各設定クラスの docstring
(`source/.../locomotion/flat_env_cfg.py`) にある。

## 次の目的: 速度分布の拡張

参照動作ベースの歩行で、同じ K1 実機が **x 2.8 m/s / y 1.9 m/s** 程度まで出せることが
判明した。現行ポリシーの学習分布 (x ±2.0 / y ±0.9) をこれに向けて広げる。

重要: この新事実は過去の 2 つの前提を覆す可能性がある。
1. 「ケイデンス 2.5 Hz 以上は実機では無理」(2026-08-25 のユーザー判断で 2.5/2.8 Hz 系は
   保管扱いにした) — 2.8 m/s が出るなら高ケイデンスか飛行期を実機が許容している。
   **どちらで速度を出しているのか (周期・接地時間) を参照動作側で確認してから設計すること。**
2. 「フライト (両足空中) 禁止 + 最大 2.0 Hz では速度上限 ≈ 1.9 m/s」(実測で確認済みの
   物理の壁)。x 2.8 を狙うなら、位相周波数マッピングの高速側引き上げ
   (保管済みの 2.5 Hz 系 K1FlatFastCfg / 2.8 Hz 系 K1FlatFast2Cfg が再利用できる) か、
   高速域限定のフライト許可 (no_fly の速度ゲート減衰) のどちらかが必須。
   odometry は接地前提なので、フライト許可は要ユーザー確認。

y ±1.9 への拡張は、未解決の実機要件「y 1.2 m/s (ゲート学習側の指令域)」も同時に解決する。
横系の知見はゴールキーパー系 (K1FlatGoalkeeperCfg / LateralVelocityCommand) にある。

## 現在の最良ポリシー (出発点)

**`frozen_policies/best_walk/model_52700.pt`** (コミット済み。README.md に全詳細)

系譜: model_34995 (±1.5 + posture/power 仕上げ) → Stance (stance_ratio 0.60 + no_fly 4)
→ Stance-Plane (平面 0.7 地形・摩擦 0.6-1.0・x ±2.0) → 省エネ仕上げ (jp -1.5e-5)
→ 関節初期角 ±30° → 上体揺れ低減 (+200it)。

実測 (measure_stride.py): 2.0 m/s 指令 → 実速度 1.87、fly% 1.3、傾き 52500 比 -12%、
パワー素の Stance-Plane 比 -31%。インタフェース: command 3 + obs_history 100×49、
位相 low 1.8 Hz / high 2.0 Hz (@1.8 m/s、以降外挿)。

追加学習は `K1FlatStancePlaneCfg` 系 (`flat_env_cfg.py`) から派生させること。
素の `K1FlatEnvCfg` 系は分布 (凹凸 0.7・摩擦 0.3〜・stance 0.5・no_fly なし) が
食い違い、特性を壊す。

## 学習の回し方

```bash
# 2 GPU 分散 (num_envs は per-rank: 2048×2 = 4096 相当)
cd scripts/rsl_rl
uv run torchrun --standalone --nproc_per_node=2 train.py \
    --task <タスク名> --headless --distributed --num_envs 2048 \
    --resume --load_run <run名> --checkpoint model_52700.pt --reset_noise_std 0.05
```

- **resume には `--reset_noise_std 0.05` が必須** (この系譜は action noise std が
  ~0.05 まで収束済みで、optimizer state ごと載せると std が負に落ちて
  `normal expects all elements of std >= 0.0` でクラッシュする)。
- `--max_iterations N` は resume では「追加 N イテレーション」。
- `--override_json` は**ファイルパス**を取り、中身は `{"env": {...}, "agent": {...}}`
  のネスト必須 (bare キーはエラーになる)。dict 内キーの dot-path
  (`rewards.X.params.Y`) 対応。
- resume 元を frozen_policies から使う場合は、.pt を
  `logs/rsl_rl/k1_flat/<任意run名>/` に置けば `--load_run` で解決できる。
- タスク登録は `source/.../locomotion/__init__.py`。runner は
  `agents/rsl_rl_ppo_cfg.py` (experiment_name "k1_flat" を共有すると resume 解決が楽)。

このマシン (2× RTX 4000 Ada) 固有だった注意 — 新 PC では該当しない可能性あり:
- NCCL が PHB 接続 + AMD IOMMU で P2P ハングするため train.py に
  `NCCL_P2P_DISABLE=1` を焼き込み済み (害はないのでそのままで良い)。
- GPU0 が使用中のとき cuda:1 指定の Isaac Sim 起動が cuda:0 のストリーム生成で落ちる
  → GPU1 単独起動は `CUDA_VISIBLE_DEVICES=1` + `--device cuda:0`。
- 20 GB GPU 向けに num_mini_batches=8 (rollout が大きいため)。VRAM が大きい PC なら
  4 に戻しても良い。

## 検証プロトコル (重要)

1. **重み・設定変更はまず 300〜1000it のプローブ** (model_52700 から resume、
   extreme 即時有効 override: `curriculum.extreme_commands.params.num_steps=0,
   ramp_steps=300`)。2 GPU で 2 水準並走が基本。
2. **採用判定は学習ログの err でなく直接計測で行う**:
   ```bash
   MEASURE_SPEEDS=0.5,1.0,1.5,2.0 uv run measure_stride.py \
       --task <タスク> --headless --num_envs 256 --device cuda:0 \
       --checkpoint <full path>
   ```
   実速度・着地歩幅・ケイデンス・接地中の滑り・fly% (両足空中率) が出る。
   ログの err_vel_xy は extreme コマンドと初期化乱れ込みで悪化を過大評価する。
3. **仕上げ系 (posture/power/揺れ) は短期が適量**: 効果は最初の 200〜1000it に集中し、
   以降は追従を食うだけ。100it 刻みの checkpoint から「膝」を直接計測で選ぶ。
4. **物理発散に注意**: 報酬が -10^5 以下へ飛ぶイベントが稀に起き (過去 3 回)、
   以降の checkpoint は損傷する。ログの `Mean reward: -[0-9]{5,}` を監視し、
   発散前から checkpoint を選ぶこと。数十〜数百程度の単発の負報酬は無害。
5. warm-start では EMA ゲートの速度カリキュラムは習得済み範囲を数十 it で正当に
   全通過する (安全ランプとしてのみ機能)。

## 速度拡張で効いた/効かなかったレバー (実測済み)

- **効く**: ケイデンス増 (2.5 Hz で ±1.8、2.8 Hz で ±2.0 を sim 達成。cfg は
  K1FlatFastCfg / K1FlatFast2Cfg に保管済み。obs×2 + feet_phase 報酬の 3 箇所の
  params を同時 override する方式)。学習分布の実機整合 (平面化・摩擦 DR 縮小) も
  天井自体を上げた。
- **効く (副作用あり)**: 滑り (スケーティング) の抑制。ベース方策は高速指令の速度の
  半分近くを接地足の前方滑りで稼いでいた。feet_slide 強化で歩幅が伸びるが、
  extreme corner と強く競合する (-8 で 1.8 m/s 追従を放棄した)。デッドバンド版
  `feet_slide_deadband` (足速度 0.15 m/s 以下無罰) が rewards.py に実装済み。
- **効かない**: 歩幅報酬 (feet_stride_length) 単独 — 上体の揺れが増えて放棄。
  σ が小さいと勾配が届かない点にも注意 (目標と現状の距離に合わせて σ=0.25)。
- ペナルティ類 (base_height 床、feet_parallel) は速度ゲート付き緩和が実装済み
  (`minimum_height` / `feet_parallel_to_ground` の gate_* params、None で無効)。

## デプロイ・下流との契約 (破ると静かに壊れる)

1. **frozen_policies/best_walk**: ゲート学習 (別セッション/別 PC) の
   `export_gating_policy.py` が `best_walk/model_*.pt` の**番号最大**を自動解決して
   融合 ONNX に焼き込む。新ポリシー確定時は best_walk の model/policy.onnx/policy.pt/
   env.yaml/README を**必ず**更新してコミットすること。更新せず学習だけ差し替えると
   古い expert 入りの ONNX ができて気づけない。
2. **歩行ポリシー確定後はゲート (GatingPPO) の再学習が必要** (このマシンのセッション
   isaaclab-k1-locomotion-3f が担当。booster_k1_locomotion リポジトリに配備)。
3. **C++ 側の位相周波数アンカー** (`k1_constants_isaaclab.hpp` の cmd_phase_freq):
   現行 (low 1.8 Hz, high 2.0 Hz @1.8 m/s)。**マッピングを変えるポリシーを作ったら
   C++ も同時に変える必要がある** (変え忘れてもロードは通ってしまう)。
   実機 onnx_inference 側は速度→周波数の積分位相を自前実装している前提。
4. 位相はコマンド依存周波数の**積分**位相 (`mdp/events.py::get_gait_phase`)。
   周波数 params は policy obs / critic obs / feet_phase 報酬の 3 箇所を**同時に**
   変えること (1 箇所でも食い違うと位相整合が壊れる)。

## エクスポート

```bash
bash export_policy.sh --checkpoint <full path の .pt>   # --checkpoint はフルパス必須
# → <run>/exported/policy_<HHMMSS>.onnx (run 名の時刻から命名)
```
入出力: command [1,3] + obs_history [1,100,49] → actions [1,12]。

## 新 PC へ持っていくもの

- このリポジトリ (ブランチ feat/inoue_walk_double_encoder)。
  frozen_policies/best_walk がコミット済みなので checkpoint はこれで足りる
  (logs/ は git 管理外、持ち出し不要)。
- uv 環境は pyproject.toml から再構築 (`uv sync`)。python/pip は必ず uv のものを使う。

## 速度拡張学習の推奨プラン (たたき台)

1. 参照動作の実測から目標ケイデンスを決める (2.8 m/s を 2.0 Hz で出すには歩幅 0.7 m/歩
   が必要で非現実的 → おそらく 2.5〜2.8 Hz 系の再開になる)。
2. `K1FlatStancePlaneCfg` を継承し、(a) 位相マッピング高速側を目標値へ (3 箇所同時)、
   (b) stages_x を ±2.0→±2.4→±2.8、stages_y を ±0.9→±1.4→±1.9 の段階拡張、
   (c) no_fly の扱いをユーザーに確認 (高ケイデンスなら禁止のままで届く可能性あり)。
3. model_52700 から warm-start、プローブ → 10000it 本学習 → 直接計測で膝選定 →
   export → best_walk 更新 → ゲート再学習依頼、の順。

---

# 追記 (2026-11-19): 横 ±1.9 拡張の完了、前進拡張の調査、摩擦 DR 拡張 (未完)

2026-11-14〜19 に行った作業のまとめ。**次にやることは「A. 摩擦 DR の拡張」**。

> **2026-11-20 更新: A 節の摩擦 DR 拡張は実施済み。結果と最終版は末尾の L 節。A 節は当時の計画として残してある。**

## 0. 現状の一覧

| 項目 | 状態 |
|---|---|
| 採用歩行 | `frozen_policies/best_walk/model_56444.pt` (x ±2.0 / y ±1.9、コミット `66ce5e4`) |
| 採用ゲート | `frozen_policies/best_gating/model_2999.pt` + `gating_2999_walk56444_turn9000.onnx` (同コミット) |
| 学習タスク | `Isaac-Velocity-Flat-Stance-Plane-Wide` (`K1FlatStancePlaneWideCfg`、既定値 = model_56444 の最終段の設定) |
| 実機 | **高速域で転倒が出た (2026-11-19 ユーザー報告)。摩擦 DR 不足の疑い → A 節** |
| C++ 側 | `~/workspace/booster_k1_locomotion` ブランチ `feat/new_fast_side_walk` コミット `a7c6151` (未 push、**未コンパイル**) |
| 調査用に足したコード | no_fly 前進ゲート (既定無効)・飽和診断など (D 節) |

下流 (経路追従・ボール奪取・トラップ・パスカット) は別セッションが model_56444 を
前提に再学習済み。**歩行を差し替えたら下流すべてが再学習対象になる** (ゲートは必須)。

## A. 次にやること: 摩擦 DR を 0.3〜3.0 に広げて再学習 (ユーザー指示 2026-11-19)

ユーザーの指示: 「横の速度コマンドを拡張した今の状態で、摩擦の DR を 0.3〜3.0 まで
拡張して再学習。それで速度範囲が少し小さくなるのは許容する」。

現状の摩擦 DR は `K1FlatStancePlaneCfg` の U(0.6, 1.0) (static/dynamic とも。
`make_consistent=True` なので dynamic ≤ static)。地形側は 1.0 × multiply 結合なので、
ロボット側の値がそのまま実効摩擦になる。経路追従タスクは同じ理由で既に U(0.2, 3.0) に
広げてある (`tracking_env_cfg.py`、コミット `be3a61c`)。

**着手済みで中断したもの** (ユーザーの中止指示で停止。結果は無い):

- 上書き設定 `probe_wide_R1.json` (摩擦 0.3〜3.0 のみ変更) は作成済み。
- 学習 R1 は 38 it で停止 (`logs/rsl_rl/k1_flat/2026-11-19_15-27-07_wideR1_friction0.3-3.0`、
  checkpoint 無し、捨ててよい)。起動と上書きの反映 (`params/env.yaml`) は確認済み。
- 現行モデルを摩擦固定で測る基準値計測は結果が出る前に停止。**基準値は未取得**。

再開手順:

```bash
cd scripts/rsl_rl
# 1) 学習 (単一 GPU、約 10 s/it。2000 it で 5〜6 時間)
LOAD_RUN=2026-11-16_03-49-34_wideS2b_angvel-0.9_ori-50.0 CKPT=model_56444.pt \
    nohup ./probe_wide.sh R1 <gpu> 2000 > probe_wide_R1.log 2>&1 &

# 2) 摩擦を固定して計測 (hydra 上書き。基準 model_56444 と新モデルの両方で、mu = 0.3 / 0.5 / 3.0)
MU=0.3; CUDA_VISIBLE_DEVICES=<gpu> MEASURE_AXIS=0 MEASURE_SPEEDS=1.0,1.5,2.0 MEASURE_OTHER=0 \
  _labpython2 measure_stride.py --task Isaac-Velocity-Flat-Stance-Plane-Wide --headless \
  --num_envs 512 --device cuda:0 --checkpoint <full path の .pt> \
  "env.events.physics_material.params.static_friction_range=[$MU,$MU]" \
  "env.events.physics_material.params.dynamic_friction_range=[$MU,$MU]"
```

- `LOAD_RUN` に S2b の run を指定するのは、`logs/rsl_rl/k1_flat/best_walk/` に model_56444 を
  置いていないため。別 PC では `frozen_policies/best_walk/model_56444.pt` を
  `logs/rsl_rl/k1_flat/<任意名>/` にコピーして `LOAD_RUN=<任意名>` とする。
- 2) の hydra 上書きは**起動まで確認、結果は未確認** (途中で止めたため)。通らなければ
  同じキーを `--override_json` 相当で渡すか、measure_stride.py に env 変数を足す。
- 見るべき列: `vx_act` (実速度)、`falls/env` (転倒)、`slide` (滑り)、`fly%`。
  採用基準はユーザーと未合意。たたき台: mu 0.3 と 3.0 の転倒が基準より明確に減り、
  mu 0.6〜1.0 の実速度の低下が 1 割以内。
- 予想される副作用: 低摩擦では横の高速 (開き幅 × ケイデンスで稼いでいる) と前進 2.0
  (速度の 2 割強を接地足の滑りで稼いでいる) が落ちる。高摩擦では滑れなくなるので
  斜め・切り返しの転倒を重点的に見る (`MEASURE_OTHER` で斜め、`MEASURE_FLIP=100` で切り返し)。
- 一度に 0.3〜3.0 で崩れる場合は、0.4〜2.0 → 0.3〜3.0 の 2 段に刻む
  (横の範囲拡張は ±1.2 → ±1.5 → ±1.9 と手で刻んで成功した)。
- 採用したら: `K1FlatStancePlaneWideCfg` に摩擦範囲を既定値として書く → best_walk 更新
  (E 節) → ゲート再学習 (F 節) → 下流の再学習を依頼。gating / tracking は
  `K1FlatStancePlaneCfg` 系を別経路で継承しているので、Wide cfg を変えても影響しない。

## B. 横 ±1.9 拡張でやったこと (完了、採用済み)

model_52700 → model_56444。合計 +3750 it の warm-start 連鎖
(Y4 → Y4b → Y12 → W2 → P4 → S2b)。設定はすべて `K1FlatStancePlaneWideCfg` の既定値に
反映済みで、コメントに根拠がある。要点:

1. **位相周波数に横加算** (policy obs / critic obs / feet_phase 報酬の 3 箇所同時):
   ```
   freq = 1.8 + max(||cmd_xy|| - 1.0, 0) * 0.25 + max(|vy| - 0.5, 0) * 0.75
   ```
   純横 1.9 m/s で約 3.1 Hz。横速度は「1 周期の移動量 0.35〜0.41 m × 周波数」で決まり、
   移動量はどの設定でもほぼ一定だったので、周波数を足すのが唯一効いたレバー。
   係数 0.5 は横 1.9 → 1.44 止まり、1.0 (3.4 Hz) は 1.73、2.0 (4.8 Hz) は実機非現実的。
   実機の周波数上限は約 3 Hz (3.4 ならいけるかも) というユーザー情報から 0.75 を採用。
2. **粗い追従項** `track_lin_vel_xy_vcoarse` (std 1.0、重み 2.4): 届かない指令を学習分布に
   入れると既存の追従項 (std 0.25 / 0.5) が飽和して「諦め歩行」に落ちる
   (x 上限を 2.2 にしただけで指令 2.2 → 実 1.07 m/s)。これを防ぐ。**範囲を広げる
   プローブではこの項を外さないこと。**
3. **no_fly の横ゲート**: |vy| 1.0 → 1.3 で罰を 0 へ。横の高速域のみフライト許可。
4. **指令領域**: |vy| ≤ 1.0 は x 全域、それより外は |vx| 上限を |vy| = 1.9 で 0.3 まで絞る
   (`lateral_x_knee` / `lateral_x_min`)。x 2.0 / y 1.0 の斜めは必須要件で含む。
5. **仕上げ**: no_fly 12 + feet_slide -1.5 (片方だけ上げると他方が悪化するので同時)、
   ang_vel_xy -0.9 + flat_orientation -50 (揺れ。+300 it まで。それ以上は 2.0 の追従を食う)。

実測値は `frozen_policies/best_walk/README.md` の表を参照
(横 1.9 → 1.60、斜め (2.0, 1.0) → (1.77, 0.79)、前進 2.0 → 1.81)。

効かなかったもの: 同じ設定で長く回す、追従重みの倍増、横の接地比緩和
(`lateral_stance_ratio` 0.45。効果小)、高速側アンカーの引き上げ (2.8 Hz@2.0 など。
再適応中に 2.0 m/s の追従ごと失う)。

## C. 前進 2.2 m/s への拡張 (調査のみ、未達)

model_56444 から +500 it のプローブ 6 本。指令 2.2 → 実速度 (512 env):

| プローブ | 変更 | 実速度 |
|---|---|---|
| 基準 56444 | — | 1.93 |
| X1 | x 上限 2.2 | 1.94 |
| X2 | X1 + high_freq 2.2 | 1.92 (歩幅が縮み滑り増) |
| F1 | X1 + フライトゲート 2.0→2.4 | 1.96 |
| F2 | X1 + フライトゲート 2.0→2.2 | 1.99 |
| G1 | F2 + high_freq 2.2 | 2.01 (歩幅 0.34 → 0.30 m、滑り 0.50 → 0.58) |
| G2 | F2 + 探索ノイズ 0.2 | 2.00 |

- どれも 2.0 m/s で頭打ち。フライトは罰を 0 にしても使われない (fly% 1.4〜1.6 のまま)。
- 前進 2.0 以下の fly% はどのプローブでも増えていない (2.0 で 1.0〜1.1 %)。
- 接地時の歩幅は 2.0 以上で約 0.34 m に飽和。

**原因の有力候補は関節速度上限**。`measure_stride.py` の末尾に出る飽和診断
(`=== actuator saturation ...`) で、model_56444 は指令 2.0 以上で膝の最大角速度が
12.5 rad/s (上限 12.57)、足首ピッチが 18.1 (上限 17.59) に張り付く。トルクが上限で
切られるのは 1 % 未満。最大値どうしの比較で、歩行周期のどの瞬間に当たっているかは未確認。

ユーザーへの未回答の確認事項: sim の速度上限 (膝 12.57 / 足首 17.59 / 股関節ピッチ 14.66 rad/s)
は実機モーターの実上限か、余裕を見た値か。前者なら今の歩き方では 2.0 が限界に近く、
膝をあまり曲げ伸ばししない歩き方へ寄せる報酬設計が要る。後者なら上限を実機値に上げる。

ユーザー要件 (前進): フライト無しで行ける範囲 (2.0 以下) はフライト禁止を保つ。
2.0 超から徐々にフライトを許可してよい。まず保守的に 2.2 から。

## D. 調査用に足したコード (この文書と同じコミットに含む)

- `mdp/rewards.py` `both_feet_not_in_contact` に前進ゲート
  (`forward_gate_low` / `forward_gate_high`。vx 指令が low → high で罰を 0 へ)。
  `flat_env_cfg.py` の Wide cfg では既定 `None` (無効)。
- `probe_wide.sh`: 環境変数 `NOISE` で `--reset_noise_std` を変えられる (既定 0.05)。
- `measure_stride.py`: 関節ごとの飽和診断 (トルクが切られた割合と最大角速度)。
- `probe_wide_R1.json` (摩擦 DR 0.3〜3.0 の上書き設定)。
- X1〜G2 の上書き設定とログは git 管理外 (内容は C 節の表の「変更」列のとおり)。
- コミットは明示的に頼まれたときだけ行う運用。

## E. 歩行を採用するときの手順

1. `bash export_policy.sh --checkpoint <full path>` で ONNX / policy.pt を作る。
2. `frozen_policies/best_walk/` に `model_<番号>.pt` / `policy.onnx` / `policy.pt` /
   `env.yaml` (run の `params/env.yaml`) を置き、README の先頭に新しい節を足す。
   **番号最大の model が下流から自動解決される**ので、旧モデルより番号が小さい場合は
   旧モデルを `old/` に退避する。
3. 位相の式を変えたら C++ 側も変える (G 節)。摩擦 DR だけなら C++ の変更は不要。

## F. ゲート (歩行 ⇄ その場回転) の再学習手順

歩行を差し替えたら毎回必要。スクラッチで 3000 it (2 GPU × 2048 env、約 6〜7 時間)。

```bash
cd scripts/rsl_rl
./train_gating.sh --max_iterations 3000 --run_name gate_v8_walk<番号>   # expert は best_walk / best_turn の番号最大
# 旧ゲートとの比較 (進入速度 / 回転角ごと。転倒率は 4096 env で見る)
CUDA_VISIBLE_DEVICES=<gpu> _labpython2 eval_pivot_overhead.py --headless --device cuda:0 \
    --num_envs 4096 --checkpoint <abs>/model_2999.pt --entry-speed 2.0 --turn-deg 120   # 横進入は --entry-vy 1.9
PYTHONUNBUFFERED=1 _labpython2 export_gating_policy.py --headless \
    --checkpoint <abs>/model_2999.pt --out <出力>.onnx   # PYTHONUNBUFFERED が無いと検証 print が出ない
```

- gating env の歩行位相 params は Wide cfg と揃えてある (lateral_gain 0.75)。
  歩行の位相式を変えたら gating 側も合わせる。
- 採用したら `frozen_policies/best_gating/` に `model_<番号>.pt` と融合 ONNX を置き、
  旧 model は `old/` へ退避 (`tracking_policy.py` が番号最大を自動解決するため)。
- 今回の結果 (model_2999 vs 旧 5897): 進入 2.0 m/s / 120° の転倒 0.34 % vs 0.73 %、
  所要時間 +0.04〜0.06 s。途中の model_2000 は旧と差なし。詳細は `best_gating/README.md`。
- **残課題**: 横 1.9 m/s からの pivot 進入は転倒 2.05 % (旧 2.93 %)。回転 expert
  (`best_turn/model_9000.pt`) が横進入を学習していないため。下げるには回転 expert の
  再学習が要る。要否はユーザー未回答。
- 融合 ONNX の mode=1 (回転) の expert 単体パリティは未検証 (乱数入力ではゲート重みが
  1 に寄らないため)。torch との等価性は 3.8e-6 で確認済み。

## G. C++ 側 (`~/workspace/booster_k1_locomotion`)

ブランチ `feat/new_fast_side_walk`、コミット `a7c6151` (未 push)。

- `src/k1_constants_isaaclab.hpp`: `PHASE_LATERAL_LOW = 0.5`。
- `src/rl_policy_isaaclab_node.cpp`: `cmd_phase_freq(lin_speed, vy)` に横加算、
  ノードパラメータ `phase_freq_lateral_gain` (既定 0.0 = 従来動作)。
- `launch/k1_isaaclab_sim.launch.py`: 既定モデルを
  `k1_gating_walk_turn_2026-11-16_v7_2999_walk56444_turn9000.onnx` に、
  `phase_freq_lateral_gain` 既定 0.75。
- **未対応**: 実機用 `k1_isaaclab_deploy.launch.py` は旧ゲート (v4_5897) のまま。
  kick の build_frame と `getup_walk_supervisor.hpp` の周波数計算は横加算なし。Python ノードは非対応。
- **未検証**: この PC には ROS と third_party SDK が無く C++ ノードをビルドできなかった。
  SDK のある PC でビルドと動作確認が要る。

## H. この PC 固有の注意 (2× GPU: RTX 4000 Ada 20 GB + RTX A4000 16 GB)

- python は `_labpython2` (`~/.bash_functions`、IsaacLab 2.3.2 の kit)。本文の `uv run` は
  別 PC 向けの記述で、この PC では `_labpython2` に読み替える。
- SSH の X 転送 (`DISPLAY`) が残っていると Isaac Sim が起動時に固まる → `unset DISPLAY`
  (probe_wide.sh は対応済み)。
- train.py は SIGTERM で終わらず GPU メモリを掴み続ける → `kill -9`。
  `pkill -f` / `pgrep -f` は自分のシェルにもマッチするので `"[p]robe_wide_R1"` のように書く。
- Isaac プロセスを同じ秒に 2 本起動すると `/tmp/IsaacLab/usd_*` が衝突して片方が落ちる
  → 60 秒ずらす。
- GPU 0 が他ジョブで埋まっていることがある (2026-11-19 は別プロジェクト FastBFM の学習が
  15 GB 使用)。`nvidia-smi` で確認し、他人のジョブは止めない。
- 単一 GPU プローブは 2048 env で約 10 s/it。

## I. 計測の読み方 (追加分)

- 学習ログの `Episode_Termination/base_height` は転倒率ではない (大半が 40 m 四方の
  地形端からの落下で、速い指令ほど増える)。`measure_stride.py` の `falls/env`
  (端付近を除外) で見る。
- `measure_stride.py` の環境変数: `MEASURE_AXIS` (0 = 前進, 1 = 横)、`MEASURE_SPEEDS`、
  `MEASURE_OTHER` (もう一方の軸の固定指令。斜め用)、`MEASURE_FLIP` (切り返しの周期 step)、
  `MEASURE_EDGE`。出力列に tilt / angv_xy / power もある。
- 速度カリキュラムの EMA 関門は warm-start では十数 it で素通りする。範囲は単段で固定し、
  直接計測で判断する。
- 位相の params を試すときは hydra 上書きを 3 箇所同時に付ける
  (`env.observations.policy.gait_phase.params.X` / `env.observations.critic.gait_phase.params.X`
  / `env.rewards.feet_phase.params.X`)。

## J. ユーザー要件の一覧 (判断の根拠)

- 上位は model-free RL なので指令への追従率は低くてよい (実速度が出ることが大事)。
- 実機の歩行周波数の上限は約 3 Hz、3.4 Hz ならいけるかもしれない。
- 横の高速域はキーパーのセーブ用途でほぼ純横。x 2.0 と y 1.9 の両立は不要、
  **x 2.0 / y 1.0 の斜めは必須**。
- 前進側はフライトしないのが望ましい。2.0 超のみ徐々に許可可。
- 摩擦 DR 拡張で速度範囲が少し狭くなるのは許容。

## K. ユーザーに未確認のまま残っていること

1. 摩擦 DR 拡張の採用基準 (どの摩擦でどの程度の速度低下・転倒率なら可か)。
2. sim の関節速度上限は実機の実上限か (C 節)。
3. 回転 expert を横進入込みで再学習するか (F 節)。
4. 実機用 launch の既定モデルを新ゲートに切り替えるか (G 節)。
5. 後退 2.8 m/s は必要か。参照動作 (2.8 m/s) のケイデンスと接地時間。

## L. 摩擦 DR 拡張の結果 (2026-11-19〜20、vast.ai RTX5090 で実施)

**最終版 = R9 `model_64440.pt`** (run `2026-11-19_20-01-03_wideR9_perenv_clearance0.15_w-150`、
設定 `probe_wide_R9.json`、model_56444 から合計 +7996 it)。`logs/` は git 管理外なので
checkpoint は学習した PC と vast.ai にしか無い。**2026-11-20 に採用: `frozen_policies/best_walk/model_64440.pt` (実機未検証)**。

### 変更点 (model_56444 比)

1. 摩擦 DR を U(0.3, 3.0) に拡張し、**1 env につき 1 つの摩擦を引く** (`mdp/events.py`
   `randomize_floor_friction`、`per_env: true`)。標準の `randomize_rigid_body_material` は
   shape ごとに独立に引くので「両足とも滑る床」がほぼ出ず、範囲だけ広げても (R1) μ0.3 の転倒が残った。
2. 振り出し足が低いことへの罰 `feet_swing_clearance` (`mdp/rewards.py`、重み -150、
   目標の足リンク高さ 0.15 m)。高摩擦の転倒の約 9 割は凹凸タイル (地形の 3 割、1〜4 cm) での
   前のめりのつまずきで、平面のみではもともとほぼ転ばない。

採用に伴い `K1FlatStancePlaneWideCfg` の既定値を R9 の設定にした (`probe_wide_R9.json` の上書きと同じ値)。

### 試した系列

| run | 内容 | 結果 |
|---|---|---|
| R1 59443 | 摩擦 U(0.3, 3.0)、shape ごと | μ0.3 の転倒 0.16〜0.19 回/9 s が残る |
| R3 | 転倒罰 -400 | 改善なし、物理発散 17 回。不採用 |
| R4 | per_env + 対数一様 | 転倒は減るが速い指令を諦める (斜めの前進成分 0.46)。不採用 |
| R6 61442 | per_env + 一様 | 速度維持。高摩擦の転倒は基準と同程度 |
| R8 62941 | + 足上げ罰 (目標 0.13) | μ3.0 前進 2.0 の転倒 0.19 → 0.09 |
| **R9 64440** | 目標 0.15 | **最終版** |
| R10 65639 | 目標 0.17 | 転倒は R9 と同じ、パワー増、μ0.3 横 1.9 の転倒 0.59。不採用 |

### 結果 (基準 56444 → R9)

転倒 [回/台]、4096 env・15 s、前進 2.0 / 横 1.9 / 斜め (2.0, 1.0):

| 条件 | 基準 | R9 |
|---|---|---|
| μ3.0 混合地形 | 0.22 / 0.17 / 0.19 | 0.03 / 0.02 / 0.04 |
| μ3.0 平面のみ | 0.02 / 0.00 / 0.04 | 0.00 / 0.00 / 0.00 |
| μ0.8 混合地形 (R6 → R9) | 0.10 / 0.01 / 0.08 | 0.00 / 0.00 / 0.02 |

μ0.3 (512 env・9 s、実速度 / 転倒): 前進 1.0 0.75/0.99 → 0.89/0.04、前進 2.0 1.20/0.42 → 1.39/0.06、
横 1.9 1.28/0.03 → 1.38/0.05、斜め 1.26/0.28 → 1.22/0.17 (**残る弱点**)。

μ0.8 の定常速度・立ち上がり・停止距離はほぼ不変 (前進 2.0: 1.82 → 1.80 m/s、停止 0.73 → 0.72 m)。
変わった点: 横 0.5 の定常 0.44 → 0.38・立ち上がり 0.18 → 0.60 s、横 1.5 の立ち上がり 0.48 → 0.76 s、
斜め (1,1) の立ち上がり 0.56 → 0.88 s、斜め (2,1) の前進成分 1.77 → 1.70、前進 2.0 + 旋回 1.0 の旋回 0.79 → 0.73。
歩容 (前進 2.0): 足上げ 5.4 → 6.9 cm、パワー 273 → 305 W、歩幅・ケイデンスは不変。横 1.9 のフライト率 9.4 → 16.5 %。
μ0.3 では R9 でも前進 2.0 の定常 1.40 m/s・立ち上がり 2.4 s・停止 0.78 m (上位方策から見ると指令の 7 割・反応が倍遅い)。

全 16 指令 × μ 3 種の生データは `friction_dr_eval/grid_base.out` / `grid_R9.out`。

### 計測の道具 (`friction_dr_eval/`、vast.ai のパス前提なので他 PC では cd 先と VK_* を直す)

- `falls.sh <名前> <ckpt> <mu>`: 転倒の精密計測 (4096 env・15 s)。平面のみは
  `EXTRA="env.scene.terrain.terrain_generator.sub_terrains.random_rough.proportion=0.0"`。
- `measf.sh <名前> <ckpt> <mu>`: 速度・歩容の一覧 (512 env・9 s。転倒率はこの台数だと粗い)。
- `grid.sh <名前> <ckpt> <mu> [num_envs]`: 指令応答 (`measure_stride.py` の `MEASURE_GRID="vx,vy,wz;..."`。
  2 s 静止 → 8 s 指令 → 3 s ゼロ。定常速度・90 % 立ち上がり・停止時間/距離・転倒率)。
- `measure_stride.py` は転倒の内訳 (リセット直後/以降、方向、終了項) と足の高さも出す。

### vast.ai サーバーの注意

- Isaac Sim が Vulkan の列挙でハングする。存在しない lib を指す ICD json を作り
  `export VK_DRIVER_FILES=/tmp/no_icd.json VK_ICD_FILENAMES=/tmp/no_icd.json` で回避 (headless 学習は問題なし)。
- `rsl-rl-lib` は 3.0.1 に合わせた。2048 env で約 3.1 s/it。学習を 2 本並べると 7.7 s/it で逆に遅い。
- GPU 32 GB: 学習 13.4 GB + 4096 env 計測 1 本 14.7 GB が上限。計測は 1 本ずつ。

### 残り

- best_walk は更新済み。残りは E・F 節の手順のゲートと下流方策の再学習。位相の式と入出力は不変なので C++ の変更は不要。
- 実機の床の種類が未確認。今回減らしたのは「高摩擦 + 凹凸でのつまずき」。硬く平らな床なら
  シミュレーション上は基準でもほぼ転ばないので、実機の転倒は別原因の可能性がある。
- R9 の指令 2.2 / 2.4 (範囲外) での実速度は未計測。μ0.3 の斜めの転倒 (0.17 回/9 s) も未解決。
