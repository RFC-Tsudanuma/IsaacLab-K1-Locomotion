source /home/satoshi/.bash_functions
NUM_GPUS=${NUM_GPUS:-2}
_labpython2 -m torch.distributed.run --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    train.py --task Isaac-Velocity-Flat-Turn --headless --num_envs 2048 --distributed $@
