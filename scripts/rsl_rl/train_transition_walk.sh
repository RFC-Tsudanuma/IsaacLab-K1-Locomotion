source /home/satoshi/.bash_functions
# 歩行 ⇄ 回転の遷移学習: 歩行 expert を学習、回転 expert を凍結 (2026-09-14)。
#
# 学習元の歩行 checkpoint は --checkpoint に絶対パスで渡す (experiment_name が k1_transition
# なので load_run による相対解決は使えない)。凍結側は --frozen_ckpt turn=/abs/path/model.pt。
# --reset_noise_std は resume 時に必須 (Adam モーメントで std が負に潰れるのを防ぐ)。
#
# 使い方:
#   ./train_transition_walk.sh \
#       --checkpoint /abs/path/logs/rsl_rl/k1_flat/2026-09-12_01-09-17/model_51500.pt \
#       --frozen_ckpt turn=/abs/path/logs/rsl_rl/k1_turn/<run>/model_XXXX.pt
NUM_GPUS=${NUM_GPUS:-2}
_labpython2 -m torch.distributed.run --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    train.py --task Isaac-Velocity-Flat-Transition-Walk --headless --num_envs 2048 --distributed \
    --resume --reset_noise_std 0.05 --max_iterations 2000 $@
