# 歩行学習 引き継ぎ (2026-11-12)

別 PC で歩行学習を続けるエージェント向けの前提知識。これ 1 ファイルを読めば開始できる。
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
