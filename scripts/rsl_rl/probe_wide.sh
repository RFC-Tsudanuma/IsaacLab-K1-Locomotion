#!/bin/bash
# 速度拡張プローブ: probe_wide.sh <名前> <gpu> [it]  (単一 GPU、既定 1000it、model_52700 起点)
# DISPLAY を外す: SSH X 転送が残っていると Isaac Sim が起動時に無限に固まる。
source ~/.bash_functions
unset DISPLAY
CUDA_VISIBLE_DEVICES=$2 _labpython2 train.py --task Isaac-Velocity-Flat-Stance-Plane-Wide \
    --headless --num_envs 2048 --device cuda:0 --resume --load_run ${LOAD_RUN:-best_walk} \
    --checkpoint ${CKPT:-model_52700.pt} --reset_noise_std 0.05 --max_iterations ${3:-1000} \
    --override_json probe_wide_$1.json
