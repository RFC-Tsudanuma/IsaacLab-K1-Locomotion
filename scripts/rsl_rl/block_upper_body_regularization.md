# パスカット方策 (Isaac-Block-K1) の上体正則化の試行記録 (2026-11-12 〜 11-14、不採用)

目的: 頭部カメラの揺れ (ボール検出率) と転倒率を下げるため、上位方策の報酬に上体の姿勢・角速度・角加速度のペナルティを加える。
結論: **試したすべての変種で失点率が v22d より 2〜4 pt 悪化し、採用しなかった。** 優先順位は「失点 > 転倒」(ユーザー指示)。
コードは元に戻してあり、best_block は v22d model_14699 (失点 21.6 % / 転倒 3.1 %) のまま。

同じことをやり直さないための記録。再挑戦するなら末尾の「もしやるなら」から。

## 前提

- Block の報酬 cfg (`K1BlockRewardsCfg`) は歩行 env の報酬を丸ごと差し替えているため、歩行 expert の学習で使っていた
  `flat_orientation_l2` / `ang_vel_xy_l2` / `base_ang_acc_l2` は Block には一切入っていなかった。
- 評価は新分布 (射出 1〜12 m、狙い ±2 m) で 1024 env × 1200 step、`eval_block.py`。
- 基準 v22d@14699 の上体指標 (step 平均): 傾き gxy² 0.0239 (RMS 約 9°)、角速度 ωxy² 0.925 (rad/s)²、角加速度 (Δω/dt)² 2283 (rad/s²)²。
- 重みの桁合わせ: 各項が 1 step あたり 0.1〜0.2 (既存 shaping 項 0.5/step の一段下) になるように
  姿勢 −5、角速度 −0.2、角加速度 −5e-5 とした。平均エピソード ≈210 step なので 1 項あたり ≈20〜40/episode
  (阻止 +100、失点 −200 と比べて無視できない大きさ)。

## 試した変種と結果

すべて v22d@14699 から `--resume --reset_noise_std 0.1` で再開 (v23d のみ v23b@16600 から)。失点 / 転倒は %。

| run | 変更 | resume +500 iter | 最終 (最良付近) | 上体指標 (最終) 傾き / 角速度 / 角加速度 |
|---|---|---|---|---|
| v23i_ctrl | **報酬変更なし (対照)** | 24.5 / 3.5 | 600 iter で終了 | 0.0241 / 1.010 / 2505 |
| v23a | 姿勢 −5 + 角速度 −0.2 (常時) | 28.1 / 2.1 | 25.2 / 2.1 @16600 (+1900) | 0.0110 / 0.521 / 1816 |
| v23b | v23a + 角加速度 −5e-5 (常時) | 26.1 / 2.1 | 25.2 / 1.5 @16600 (+1900) | 0.0112 / 0.452 / 1098 |
| v23e | 角加速度 −5e-5 のみ | 26.8 / 3.3 | 27.0 / 3.5 @15600 (+900、停滞) | 0.0260 / 0.856 / 1418 |
| v23f | v23e と同じ、std 下限 0.05 | (+100 で 36.2) | 中止 | — |
| v23g | v23b の重みを到達 1 s 前から免除 | 32.7 / 2.3 | 中止 | 0.0173 / 0.722 / 1641 (@15200) |
| v23h | v23b の重みを到達 2 s 前から免除 | 28.2 / 3.1 | 25.5 / 2.5 @15800 (+1100) | 0.0176 / 0.673 / 1664 |
| v23d | v23b@16600 から姿勢 −1 / 角速度 −0.05 / 角加速度 −1e-4 に緩和 | 29.8 / 3.9 | 中止 | 0.0112 / 0.511 / 1043 |

学習ログ: `logs/rsl_rl/k1_block/2026-11-12_*_v23a_orient`, `_v23b_angacc`, `2026-11-14_*_v23d_acc`, `_v23e_acconly`, `_v23g_tta1`, `_v23h_tta2`, `_v23i_ctrl`。

## 分かったこと

1. **resume 自体にコストがある。** 報酬を変えない対照でも +100 / +300 / +500 iter で 33.5 / 26.9 / 24.5 % と、
   v22d の 21.6 % に戻るのに 500 iter では足りない (過去の v22c の履歴では +1200 で完全回復)。
   原因は `--reset_noise_std` の挙動: これは「std の下限クランプ + optimizer 状態を読み込まない」であり、
   v22d の std は ≈0.04 なので 0.1 にクランプされ探索ノイズが 2.5 倍になる。無指定だと初回更新で std<0 になり
   `normal expects all elements of std >= 0` で落ちるため、真の継続はできない (train.py:267-293)。
   **resume 直後 +500 iter 程度の比較は resume コストに埋もれる。報酬比較には必ず対照 run を並走させること。**
2. **ペナルティには resume コストとは別に 2〜4 pt の失点コストがある。** 同時点 (+500) で対照 24.5 % に対し
   v23a〜h は 26〜33 %。常時掛け (v23a/b) は +1900 iter でも 25.2 % で停滞。
3. **角加速度項だけ (v23e) でも失点は悪化し、転倒も揺れ (傾き・角速度) も改善しない。** v23a→v23b の比較で
   「角加速度項は無料」に見えたのは誤り。
4. **到達直前の踏み込みだけ免除するゲート (`until_ball_arrives`, v23g/h) でもコストは消えず、むしろ回復が遅い。**
   失点悪化は踏み込み局面の問題ではなく、ボールが遠い間の素早い位置取り (歩行に伴う上体の揺れ) を抑えてしまうため。
5. 転倒が明確に下がったのは常時掛けの v23b のみ (3.1 → 1.5 %)。揺れも半減。失点 +3.6 pt との交換。
   トレードオフを許容するなら候補は v23b@16600 (`2026-11-12_19-06-43_v23b_angacc/model_16600.pt`)。
6. 悪化する帯は横 1〜2 m (v22d 35.8 % → v23 系 38〜48 %)。遠距離・低速球よりも横移動が要る球で効く。

## もしやるなら (未検証の案)

- resume ではなく、v22 系の学習の最初から軽い重み (姿勢 −1 / 角速度 −0.05 程度) を掛けて長く回す。
  resume の後遺症と切り分けられ、方策が最初から制約下で位置取りを学べる。
- 方策ではなく検出側で揺れを扱う (IMU で画像座標を補正、首の制御)。失点を犠牲にしない。
- resume を使うなら `--reset_noise_std` を現状の std (≈0.04〜0.05) に合わせ、対照 run を必ず並走させる。
  ただし v23f (下限 0.05) の +100 時点は 36 % で、下限を下げてもディップは消えなかった (1 点のみ、要確認)。

## 戻したコード (再利用用)

### 評価側: eval_block.py に上体指標を出す (ループ前 / ループ内 / 速度表の前)

```python
body = {"tilt": 0.0, "ang_vel": 0.0, "ang_acc": 0.0}
prev_w = robot.data.root_ang_vel_b.clone()
...
    w = robot.data.root_ang_vel_b
    body["tilt"] += robot.data.projected_gravity_b[:, :2].square().sum(1).mean().item()
    body["ang_vel"] += w[:, :2].square().sum(1).mean().item()
    acc_ok = raw_env.episode_length_buf >= 2  # リセット直後の差分は除く
    body["ang_acc"] += (((w - prev_w) / dt).square().sum(1) * acc_ok).sum().item() / max(acc_ok.sum().item(), 1)
    prev_w = w.clone()
...
print("  上体の揺れ (step 平均): 傾き gxy² {:.4f}, 角速度 ωxy² {:.3f} (rad/s)², 角加速度 {:.0f} (rad/s²)²".format(
    *(body[k] / args_cli.steps for k in ("tilt", "ang_vel", "ang_acc"))))
```

### 報酬側: block_env_cfg.K1BlockRewardsCfg に weight 0 で追加し hydra で上書き

```python
from .mdp.rewards import base_ang_acc_l2
flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
base_ang_acc_l2 = RewTerm(func=base_ang_acc_l2, weight=0.0)
# 起動例: env.rewards.flat_orientation_l2.weight=-5.0 env.rewards.ang_vel_xy_l2.weight=-0.2 env.rewards.base_ang_acc_l2.weight=-5e-5
```

### 到達時刻ゲート (mdp/block.py、v23g/h で使用)

```python
from isaaclab.utils.string import string_to_callable

def until_ball_arrives(env, func, min_tta: float = 0.0, command_name: str = "ball_launch") -> torch.Tensor:
    """ボールのライン到達予測が min_tta [s] より先の間だけ func を掛ける (min_tta 0 = ボールが前にある間は常時)。"""
    x, _, vx, _ = env.command_manager.get_term(command_name).ball_line_state()
    fn = string_to_callable(func) if isinstance(func, str) else func  # hydra 経由だと "module:name" 文字列になる
    return fn(env) * (x / (-vx).clamp(min=0.1) > min_tta)

# RewTerm(func=until_ball_arrives, weight=-5.0, params={"func": mdp.flat_orientation_l2, "min_tta": 2.0})
```
