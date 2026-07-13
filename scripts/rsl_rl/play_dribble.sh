source /home/satoshi/.bash_functions

# Frozen low-level (walking) checkpoint. Override with:
#   FROZEN_CKPT=/path/to/walking_model.pt ./play_dribble.sh path/to/dribble_model.pt
FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_flat/latest/model_0.pt}

_labpython2 play_dribble.py \
    --task Isaac-Dribble-K1-Play-v0 \
    --frozen_checkpoint "${FROZEN_CKPT}" \
    --low_level_obs_group low_level \
    --num_envs 32 \
    --checkpoint $@
