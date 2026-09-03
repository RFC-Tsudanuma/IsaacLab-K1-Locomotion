source /home/satoshi/.bash_functions
# 学習済 Flat ポリシーに対し角速度(yaw)追従を強化する追加学習。
# 既存 run から resume して使う:
#   ./train_ang_tracking.sh --resume --load_run <既存run名>
# 特定 checkpoint を指定する場合:
#   ./train_ang_tracking.sh --resume --load_run <run名> --checkpoint model_xxxx.pt
_labpython2 train.py --task Isaac-Velocity-Flat-AngTracking --headless --num_envs 4096 $@
