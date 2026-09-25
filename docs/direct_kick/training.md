# モデル・学習

出典は [コピー元コミット](README.md) の `utils/models/DirectKickingAC.py` と `utils/runner.py`。数値原本は [DirectKicking.yaml](source/DirectKicking.yaml)。

## モデル

| 部分 | 構成 |
|---|---|
| モデル名 | `DirectKickingActorCritic` |
| Actor encoder | horizon 13×6 → LSTM(input=6, hidden=64, layers=1, batch_first=true) |
| 非予測部分 | locomotion 47 + relative velocity 2 + status 3 + target 2 = 54 |
| encoder 出力 | 非予測54 + LSTM最後の hidden64 = 118 |
| Actor action head | 118 → 256 → 128 → 128 → 12、隠れ層 ELU |
| Actor phase head | 118 → 128 → 1、隠れ層 ELU、確率出力時 sigmoid |
| Critic encoder | Actor と同構造、別重み |
| Critic value head | encoder118 + privileged20 = 138 → 256 → 256 → 128 → 1、隠れ層 ELU |
| action distribution | 12次元の要素独立 Normal、平均 action head、std=`exp(logstd)` |
| logstd | 学習可能 shape `[1,12]`、初期値 -2（std ≈0.135335） |
| phase 初期値 | 最終 Linear weight=0、bias=-4（初期確率 ≈0.017986） |
| 他層の初期化 | PyTorch Linear/LSTM の既定初期化 |
| 観測正規化 | モデル内に RunningMeanStd/BatchNorm/LayerNorm はない |

LSTM の走査軸は未来 horizon。呼出ごとにゼロ状態から開始し、rollout 時刻間の hidden/cell state は保持しない。Actor と Critic の encoder は共有しない。Actor の action head と phase head は encoder を共有する。

`model.act(obs)` は 12次元 Normal を返し、学習では sample、play では mean を環境へ渡す。action mean に tanh/clip はない。環境側が `[-1,1]` に clip する。

`model.actor.forward(obs)` は **12 action means + phase probability = 13 次元**。入力は132次元。export はこの Actor を `torch.jit.script` したもので、Critic、特権観測、logstd は含まない。外部 hidden-state 入出力もない。phase を13番目の関節 action として扱わない。

根拠: `DirectKickingAC.py:21,78,131,179,251,352,410`、`utils/runner.py:845,1300`、`export_model.py:59`。

## Rollout・更新

| 項目 | 仕様 |
|---|---|
| rollout 長 | 24 policy steps / env（0.48 s） |
| mini epochs | 20 |
| optimization chunk size | 16384 samples |
| optimizer | Adam、明示指定は learning rate のみ |
| 初期 learning rate | 1e-5 |
| gamma / lambda | 0.995 / 0.95 |
| PPO clip | 0.2。`surrogate_loss` の既定引数 |
| value clip | YAML 未指定のため無効 |
| gradient norm clip | 全モデル parameters に 1.0 |
| desired KL | 0.01 |
| 最大 iterations | 200000 |
| 保存 interval | 500 iterations |

buffer は `[24, N, ...]`。各 step で観測、特権観測、clip 前の sampled action、reward、done、timeout を保存する。phase 教師は **step 前**にその観測と組で保存。hidden state は保持しない。rollout の各 step で新しい action を生成する。

更新前に old log probability、mean、std、value、最後の観測の value を計算する。old values、returns、advantages は20回の epoch 中固定。

chunk はメモリ節約のための分割であり、chunk ごとの optimizer 更新ではない。各 epoch に1回 `zero_grad`、各 chunk の平均損失に `chunk_count/total_count` を掛けて backward を蓄積し、全 chunk 後に norm clip と1回の optimizer step。flatten 順のままで shuffle はない。

8192環境なら `24*8192=196608 samples`、12 chunks/epoch、20 optimizer steps/rollout。chunk size 未指定/null は全件、正の指定値は全 sample 数で上限を切る。

根拠: `utils/buffer.py:6`、`utils/runner.py:112,845,872,980,1088`。

## Timeout・GAE

元 Runner は timeout 報酬を次のように**置換**する。

```text
rewards[time_outs] = old_values[time_outs]
done_for_gae = dones OR time_outs
delta_t = reward_t + gamma*(1-done_t)*V_next - V_t
A_t = delta_t + gamma*lambda*(1-done_t)*A_next
returns_t = V_t + A_t
A_normalized = (A - mean(A)) / (std(A) + 1e-8)
```

正規化前の timeout step の advantage は0、return は old value。報酬への `gamma*V` の加算とは異なる。advantage は rollout 全体で正規化する。
根拠: `utils/runner.py:940`、`utils/utils.py:33`。

## 損失・学習率

```text
L = L_value + L_PPO + L_bound - 0.01*entropy
    + 10.0*L_symmetry + 0.1*L_phase

L_value = mean((V-returns)²)
ratio = exp(logprob_new-logprob_old)
L_PPO = mean(max(-A*ratio, -A*clip(ratio,0.8,1.2)))
L_bound = mean(max(mu-1,0)²) + mean(min(mu+1,0)²)
```

現経路の value loss に0.5係数はない。log probability は12 action 成分の和を各sampleごとに保持して ratio を計算する。entropy は12成分の和を sample 平均する。bound loss は batch・action 要素全体の平均であり、action mean の範囲外罰則。entropy の min/max は両方 null なので範囲罰則は無効。

20 epoch 後に `KL(old Normal || final Normal)` を action 方向に合計し、全 sample で平均する。KL>0.02なら `lr=max(1e-5,lr/1.5)`、KL<0.005なら `lr=min(1e-2,lr*1.5)`、他は維持。KL による epoch の early stopping はない。

根拠: `utils/runner.py:980,1020,1103`、`utils/utils.py:47`。

## 鏡像整合性

現 YAML で enabled=true、係数10。鏡映は左右脚を交換し、各脚の成分に `[+1,-1,-1,+1,+1,-1]` を掛ける。

| 観測 | 鏡映 |
|---|---|
| projected gravity | `[+,-,+]` |
| angular velocity | `[-,+,-]`（軸性ベクトル） |
| command | `[+,-,-]` |
| gait cosine/sine | 両方反転 |
| joint q / dq / action | 左右交換 + 脚符号 |
| horizon token | y と rho の符号反転 |
| belief relative velocity | y の符号反転 |
| target direction | y の符号反転 |
| その他 | 不変 |

左右 strike point `(0.185,±0.096)` に対する全 horizon 予測位置の最短距離を `J_left/right` とし、

```text
gap = abs(J_left-J_right)
u = clip((gap-0.02)/(0.10-0.02),0,1)
weight = u²*(3-2u)*clip(belief_valid,0,1)
L_symmetry = mean_samples(weight * sum_actions(
    (mu(mirror(obs))-mirror(mu(obs)))²))
```

strike point と閾値は `normalization.ball_pos` に合わせてスケールする。重みの総和による再正規化はしない。元・鏡映両方の mean に勾配が流れ、phase head はこの loss に含めない。
根拠: `utils/direct_kicking_symmetry.py:16,137`、`DirectKickingAC.py:371`。

## Phase 補助損失

enabled=true、係数0.1。教師は [キック足着地後のラベル](rewards.md#post-kick-phase-教師)。Actor の入力へ追加する値ではない。

negative は target<0.5、positive はそれ以外。negative（premature）の重み3、positive（delayed）の重み1。rollout 全体の sample 数を `N`、各クラス数を `N_neg/N_pos`、存在するクラスの重み合計を `W` とする。

```text
multiplier_neg = N*3/N_neg/W
multiplier_pos = N*1/N_pos/W
L_phase = mean(BCEWithLogits(logit,target,reduction='none') * multiplier)
```

両クラスがあると各クラス平均の寄与比は3:1。一方だけならそのクラスの単純平均。chunk ごとに class balance を計算し直さない。
根拠: `utils/runner.py:142,849,879,1042`、`utils/post_kick_phase.py:7`。

## 起動既定値・ログ

| 設定 | YAML | 元 `train_direct_kicking_k1.sh` |
|---|---|---|
| task / model | K1/DirectKicking / DirectKickingActorCritic | task を同じ値で指定 |
| num_envs | 1 | 8192 |
| headless | false | True |
| sim/rl device | cpu/cpu | DEVICE があれば採用、なければ `nvidia-smi` の存在で cuda:0/cpu |
| seed | 42 | 上書きなし |
| max_iterations | 200000 | 上書きなし |
| checkpoint | 未指定 | -1（既存 checkpoint 自動選択を要求） |
| save/video interval | 500/500 | 上書きなし |
| wandb / CSV | true/true | 上書きなし |
| env status print interval | 0 | 上書きなし |

shell は `NUM_ENVS`, `HEADLESS`, `SIM_DEVICE`, `RL_DEVICE`, `DEVICE`, `CHECKPOINT`, `PYTHON_BIN` を利用する。`CHECKPOINT=none/None/null/NULL` は空に変換され、load しない。Python の既定は元 repo の `venv_isaac/bin/python`。追加 CLI 引数は末尾へ渡す。

Runner の学習 CLI は `--task`, `--checkpoint`, `--init_from_checkpoint`, `--num_envs`, `--headless`, `--sim_device`, `--rl_device`, `--seed`, `--max_iterations`, `--model`。num_envs は env 節、他は basic 節へ反映。`--headless` は元コードで `argparse type=bool`。

学習本体では viewer recording を無効にし、動画を別プロセスで取得する。YAML に `log_video_duration` はないため Runner の既定10秒。viewer の設定と学習時の有効値を混同しない。
根拠: `train_direct_kicking_k1.sh:17`、`utils/runner.py:196,217,246,771`。

## Checkpoint と export

保存 payload は `model`, `optimizer`, `curriculum`, `iteration`, `model_metadata`。環境に `get_curriculum_state()` があれば `stage_curriculum` も追加。

```yaml
model_class: DirectKickingActorCritic
observation_schema: direct_kicking_horizon_lstm_direction_only_v2
num_actions: 12
num_observations: 132
num_privileged_observations: 20
prediction_horizons_s: [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50,
                       0.75, 1.00, 1.50, 2.00, 2.50, 3.00]
horizon_token_size: 6
lstm_hidden_size: 64
lstm_num_layers: 1
policy_output_schema: joint_action_12_post_kick_phase_probability_v1
num_policy_outputs: 13
post_kick_phase_output_index: 12
```

通常 load は metadata 辞書の完全一致と strict state load を要求し、resume では optimizer と iteration も復元する。`--init_from_checkpoint` はモデル重みだけを読み、optimizer/iteration は新規のまま。この専用初期化経路のみ、phase head 追加前の metadata/state を受理して phase head の初期値を維持する。

| parameter | 宣言から導出した shape |
|---|---|
| actor/critic `encoder.lstm.weight_ih_l0` | (256,6) |
| actor/critic `encoder.lstm.weight_hh_l0` | (256,64) |
| 各 LSTM bias | (256,) |
| actor `network.0/2/4/6.weight` | (256,118), (128,256), (128,128), (12,128) |
| actor `post_kick_phase_head.0/2.weight` | (128,118), (1,128) |
| critic `network.0/2/4/6.weight` | (256,138), (256,256), (128,256), (1,128) |
| logstd | (1,12) |

Linear bias は各出力次元と同じ長さ。実 checkpoint をロードして検証した表ではない。
根拠: `DirectKickingAC.py:301`、`utils/model_factory.py:79`、`utils/runner.py:246,757`。
