source /home/satoshi/.bash_functions
# 学習済み回転ポリシーからの warm-start (追加学習)。
#
# カリキュラムは K1FlatTurnFinetuneCfg が最終状態 (±180° / 初期速度 ±1.4 m/s) に
# 固定するので、resume で common_step_counter が 0 に戻っても能力が引き戻されない。
#
# --reset_noise_std は必須。train.py はこの引数があるとオプティマイザ状態の
# 読み込みをスキップするので、adaptive KL の LR が 1e-5 に張り付く問題を回避できる。
#
# 使い方:
#   ./train_turn_finetune.sh --load_run 2026-09-13_04-31-39 --checkpoint model_4200.pt
NUM_GPUS=${NUM_GPUS:-2}
_labpython2 -m torch.distributed.run --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    train.py --task Isaac-Velocity-Flat-Turn-Finetune --headless --num_envs 2048 --distributed \
    --resume --reset_noise_std 0.05 --max_iterations 2000 $@
