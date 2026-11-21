#!/usr/bin/env bash
source /home/satoshi/.bash_functions
# 首 (ヨー・ピッチ) 安定化の残差方策の学習 (2026-11-20)。設計は neck_compensate.md。スクラッチ学習。
# 歩行 expert の既定は frozen_policies/best_walk/model_*.pt (番号最大)。差し替えは agent.policy.expert_checkpoint=/abs/path
#
#
# 評価 (状況別のブラー。--mode ik = 先読み付き解析 IK のみ、--mode none = 安定化なし):
#   _labpython2 eval_neck.py --headless --checkpoint /abs/path/model_1499.pt
#   _labpython2 eval_neck.py --headless --mode ik
# 首指令の遅延を脚と同じ 10〜35 ms にする (既定は ≤ 5 ms)。hydra の上書きは整数・小数の型を合わせること (0 と 0.0):
#   ./train_neck.sh env.scene.robot.actuators.neck.min_delay=2 env.scene.robot.actuators.neck.max_delay=7
# 同時に複数起動するときは数秒ずらす (同じ秒に起動すると URDF → USD 変換の一時ディレクトリが衝突して落ちる)。
_labpython2 train.py --task Isaac-Neck-K1-v0 --headless --num_envs 4096 $@
