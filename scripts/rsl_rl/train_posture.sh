source /home/satoshi/.bash_functions
# 学習済 Flat ポリシーに対し上体の傾き・上下動を抑える仕上げ学習。
# extreme_command_curriculum は最初から発動する (K1FlatImprovePostureCfg 参照)。
# 既存 run から resume して使う:
#   ./train_posture.sh --resume --load_run <既存run名> --max_iterations 2500
# 特定 checkpoint を指定する場合:
#   ./train_posture.sh --resume --load_run <run名> --checkpoint model_xxxx.pt --max_iterations 2500
_labpython2 train.py --task Isaac-Velocity-Flat-Posture --headless --num_envs 4096 $@
