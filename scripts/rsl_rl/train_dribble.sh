source /home/satoshi/.bash_functions

# Path to the frozen low-level (walking) policy checkpoint.
# Override with: FROZEN_CKPT=/path/to/model_XXX.pt ./train_dribble.sh
FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_flat/latest/model_0.pt}

_labpython2 train_dribble.py \
    --task Isaac-Dribble-K1-v0 \
    --frozen_checkpoint "${FROZEN_CKPT}" \
    --low_level_obs_group low_level \
    --headless --num_envs 4096 $@
