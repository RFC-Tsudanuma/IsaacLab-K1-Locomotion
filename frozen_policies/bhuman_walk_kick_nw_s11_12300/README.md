# bhuman_walk_kick nw_s11_overspeed model_12300 (新しい歩行 model_52700 の上、2026-11-03)

歩行は frozen_policies/best_walk_52700 (model_52700)。全速 (指令 2.0、実速 1.7〜1.9 m/s) のまま受け渡しても転ばないようにした版。

## 学習の流れ (vast.ai RTX 5070 Ti、1024 envs)
- Stage 1〜9 (nw_s1 … nw_s9c): 前回採用した流れ (s9c 11400) と同じ段・回数・設定に、新しい歩行の学習設定と受け渡し (handoff_prob 0.5) を足したもの。
  手順は scripts/vast_chain_new_walk.sh。nw_s9c は 11000 で止めた。
- nw_s10_fasthandoff (11000 → 11999): 足の摩擦 [0.3, 1.0]、受け渡しの半分を全速 (handoff_fast_prob 0.5)。
  理由: 速い受け渡しの直後の転倒は、Isaac の足が MuJoCo より滑りにくいせいで学習中に起きていなかった。
- nw_s11_overspeed (11999 → 12300): 距離指定キックの速すぎを引く ball_kick_overspeed (重み -4.0)。
- コード: ブランチ fix/handoff-falls (0b03444, 9a9b918)。

## MuJoCo (sim2sim_bhuman_walk_kick.py、摩擦 0.7)
- ランダム 100 場面 (seed 0): 仕様 10/10、転倒 0、強キック 3.96 m/s、弱キックのずれ 0.67 m/s、方向のずれ 2.7°。
- 受け渡し (歩行 2.0 m/s、ボール 4 m、24 場面、成功 / ぶつかり / 転倒):
  入る距離 1.2 m: 23 / 1 / 0、2.0 m: 24 / 0 / 0、0.8 m: 1 / 21 / 2。
  比較: 前の s9c 11400 は 1.2 m で 14 / 4 / 2、0.8 m で 0 / 24 / 0。
- 実機の受け渡し距離 ball_approach_m = 1.2 のままで使う想定。0.8 m では使わない。
