#!/usr/bin/env bash
source /home/satoshi/.bash_functions
# 凍結歩行 expert を事前分布にしたボール奪取方策の学習 (2026-11-03)。スクラッチ学習。
#
# expert の checkpoint 既定は frozen_policies/best_walk/model_*.pt (番号最大)。差し替えは
#   agent.policy.expert_checkpoint=/abs/path/model.pt   (hydra 形式)
#
# 使い方:
#   ./train_steal.sh                         # 2 GPU 分散 (2048 env × 2)
#   NUM_GPUS=1 ./train_steal.sh --num_envs 4096 --run_name v1
#   TASK=Isaac-StealDir-K1-v0 ./train_steal.sh --run_name dir_v1   # 方向指定版
NUM_GPUS=${NUM_GPUS:-2}
TASK=${TASK:-Isaac-Steal-K1-v0}
if [ "$NUM_GPUS" -gt 1 ]; then
    _labpython2 -m torch.distributed.run --nnodes=1 --nproc_per_node=${NUM_GPUS} \
        train.py --task ${TASK} --headless --num_envs 2048 --distributed $@
else
    _labpython2 train.py --task ${TASK} --headless $@
fi
