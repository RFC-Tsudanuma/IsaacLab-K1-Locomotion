source /home/satoshi/.bash_functions
# 凍結 expert (歩行 / 回転) を混合するゲート網の学習 (2026-09-23)。スクラッチ学習。
#
# expert の checkpoint 既定は frozen_policies/best_walk/ と frozen_policies/best_turn/ の
# model_*.pt のうち番号最大 (2026-11-05 時点: model_52700.pt / model_9000.pt)。差し替えるときは
#   --frozen_ckpt walk=/abs/path/model.pt --frozen_ckpt turn=/abs/path/model.pt
#
# 使い方:
#   ./train_gating.sh
#   ./train_gating.sh --max_iterations 1500 --run_name smooth2 \
#       --override_json overrides.json   # 例: {"agent": {"algorithm.gate_smooth_coef": 2.0}}
NUM_GPUS=${NUM_GPUS:-2}
_labpython2 -m torch.distributed.run --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    train.py --task Isaac-Velocity-Flat-Gating --headless --num_envs 2048 --distributed $@
