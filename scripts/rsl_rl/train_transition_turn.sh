source /home/satoshi/.bash_functions
# 歩行 ⇄ 回転の遷移学習: 回転 expert を学習、歩行 expert を凍結 (2026-09-14)。
#
# 使い方:
#   ./train_transition_turn.sh \
#       --checkpoint /abs/path/logs/rsl_rl/k1_turn/<run>/model_XXXX.pt \
#       --frozen_ckpt walk=/abs/path/logs/rsl_rl/k1_flat/2026-09-12_01-09-17/model_51500.pt
NUM_GPUS=${NUM_GPUS:-2}
_labpython2 -m torch.distributed.run --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    train.py --task Isaac-Velocity-Flat-Transition-Turn --headless --num_envs 2048 --distributed \
    --resume --reset_noise_std 0.05 --max_iterations 2000 $@
