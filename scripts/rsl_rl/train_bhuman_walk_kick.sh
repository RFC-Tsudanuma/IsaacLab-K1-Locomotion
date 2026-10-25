#!/bin/bash
# B-Human 版 walk-kick を歩行 model_34995 の上で学習した手順 (2026-10-24〜25 に実際に回した順番と値)。
# 1 段ずつ前の段の checkpoint から --resume で続ける。各段の iteration 数は「良くなりが止まった所」で手で止めた値。
#
# コンテナ内で実行する (CLAUDE.md 参照)。このコンテナの kit python は cuDNN が初期化できないので、
# cu12 版の cuDNN を先に読み込む sitecustomize を PYTHONPATH に入れている
# (scripts/rsl_rl/cudnn_fix/sitecustomize.py。中身は ctypes で
#  /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cudnn/lib/libcudnn*.so.9 を RTLD_GLOBAL で読むだけ)。
# 評価: scripts/rsl_rl/eval_bhuman_walk_kick.py (IsaacLab)、scripts/mujoco/sim2sim_bhuman_walk_kick.py (MuJoCo, ホスト)。
set -e
cd /workspace/IsaacLab-K1-Locomotion
export PYTHONPATH=/workspace/IsaacLab-K1-Locomotion/scripts/rsl_rl/cudnn_fix:/workspace/IsaacLab-K1-Locomotion/source/isaaclab_k1_locomotion:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
PY=/workspace/isaaclab/_isaac_sim/python.sh
T="$PY scripts/rsl_rl/train.py --headless --device cuda:0 --num_envs 1024"
EXP=logs/rsl_rl/k1_bhuman_walk_kick

# 0) 歩行の重みをキック用の形に移す (ホストの ~/k1_deploy_env/bin/python でも可)
#    python scripts/rsl_rl/convert_walk_to_bhuman_kick.py --walk /workspace/model_34995.pt --out $EXP/walk34995/model_0.pt

# 1) Stage 1。最初の 800 iter はノイズを歩行のまま (0.1 前後) で回したが強く蹴る動きが育たなかったので、
#    ノイズ下限 0.2 で 1100 まで、そこから歩き込み速度と強キック重み 10 を入れて 2300 まで回した。
#    (今の設定 = 歩き込み速度・強キック重み 10・ボール摩擦ランダム化 が入った状態なので、最初からこれで回してよい)
$T --task Isaac-K1-BHuman-Walk-Kick-v0 --max_iterations 2300 --resume --load_run walk34995 --checkpoint model_0.pt \
   --reset_noise_std 0.2 --run_name s1 agent.algorithm.std_range=[0.2,1.0]

# 2) Stage 2 (2300 → 2700)
$T --task Isaac-K1-BHuman-Walk-Kick-Stage2-v0 --max_iterations 400 --resume --load_run '.*_s1$' --checkpoint model_2300.pt \
   --reset_noise_std 0.2 --run_name s2 agent.algorithm.std_range=[0.2,1.0] agent.algorithm.actor_freeze_iterations=20

# 3) Stage 3 (2700 → 3000)。ノイズを B-Human のエントロピー [-15,-14] 相当の 0.13〜0.17 に下げる
$T --task Isaac-K1-BHuman-Walk-Kick-Stage3-v0 --max_iterations 300 --resume --load_run '.*_s2$' --checkpoint model_2700.pt \
   --reset_noise_std 0.13 --run_name s3 agent.algorithm.std_range=[0.13,0.17] agent.algorithm.actor_freeze_iterations=10

# 4) Stage 4 (3000 → 3500)。feet_phase は下げない・当てる高さの係数あり (bhuman_walk_kick_env_cfg.py の Stage 4)
$T --task Isaac-K1-BHuman-Walk-Kick-Stage4-v0 --max_iterations 500 --resume --load_run '.*_s3$' --checkpoint model_3000.pt \
   --reset_noise_std 0.13 --run_name s4d_strike agent.algorithm.std_range=[0.13,0.17] agent.algorithm.actor_freeze_iterations=10

# 5) Stage 5 (3500 → 3900)
$T --task Isaac-K1-BHuman-Walk-Kick-Stage5-v0 --max_iterations 400 --resume --load_run '.*_s4d_strike$' --checkpoint model_3500.pt \
   --reset_noise_std 0.13 --run_name s5 agent.algorithm.std_range=[0.13,0.17] agent.algorithm.actor_freeze_iterations=10

# 6) Stage 6 (3900 → 4400)、続けて sigma 0.15 (4400 → 4800)。ノイズ 0.12〜0.13 (エントロピー [-16.6,-16.5] 相当)、desired_kl 0.003
$T --task Isaac-K1-BHuman-Walk-Kick-Stage6-v0 --max_iterations 500 --resume --load_run '.*_s5$' --checkpoint model_3900.pt \
   --reset_noise_std 0.12 --run_name s6 agent.algorithm.std_range=[0.12,0.13] agent.algorithm.desired_kl=0.003 \
   agent.algorithm.actor_freeze_iterations=10
$T --task Isaac-K1-BHuman-Walk-Kick-Stage6-v0 --max_iterations 400 --resume --load_run '.*_s6$' --checkpoint model_4400.pt \
   --reset_noise_std 0.12 --run_name s6b_sig015 agent.algorithm.std_range=[0.12,0.13] agent.algorithm.desired_kl=0.003 \
   agent.algorithm.actor_freeze_iterations=0 env.commands.kick.kick_direction_sigma_range=0.15
# ↑ この model_4800 を最終の方策にした (MuJoCo で一番ずれが小さく転倒 0)。
# Stage 7 (確率を最終値の 10% → 最終値) も回したが (s7a / s7b_final)、MuJoCo ではずれが 7.0° → 8.5〜9.2° に増えたので採用していない:
#   --task Isaac-K1-BHuman-Walk-Kick-Stage7-v0 ... env.commands.kick.kick_direction_sigma_range=0.15 \
#   env.commands.kick.random_ball_reset=[0.001,0.002,0.002] env.commands.kick.reset_ball_request=0.0005 \
#   env.commands.kick.add_ball_vel=0.00001 env.commands.kick.add_ball_vel_in_direction=0.00001   (7a、その後 7b は既定値)
