source /home/satoshi/.bash_functions
# 回転 expert の「歩行中からの進入」追加学習 (2026-11-05)。
# ゲート学習環境 (実機指令域・切替頻度↑) で MultiExpertPPO (learner=turn)。
#
# 使い方:
#   ./train_transition_turnfast.sh \
#       --checkpoint /home/satoshi/IsaacLab-K1-Locomotion/frozen_policies/best_turn/model_7999.pt \
#       --frozen_ckpt walk=/home/satoshi/IsaacLab-K1-Locomotion/frozen_policies/best_walk/model_52700.pt
NUM_GPUS=${NUM_GPUS:-2}
_labpython2 -m torch.distributed.run --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    train.py --task Isaac-Velocity-Flat-Transition-TurnFast --headless --num_envs 2048 --distributed \
    --resume --reset_noise_std 0.05 --max_iterations 1500 $@
