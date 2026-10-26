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

# ---- 2026-10-26 からやり直し: Stage 2 以降を B-Human どおり (インサイドキック) にする ----
# 前回 (2026-10-25) の Stage 2〜7 は蹴り足の向き 0° のまま (つま先キック) で、仕様 (docs/bhuman_walk_kick_spec.md) の
# 蹴り方・当て方が 0% だった。今回は Stage 1 の s1c model_2300 から、README どおりの値で続ける。
# ボールは B-Human と同じ 0.095 m・0.29 kg (Stage 1 は 0.11 m・0.45 kg で学習した checkpoint のまま始める)。
# 各段の iteration 数は、評価 (MuJoCo ランダム 100 場面 + IsaacLab) を見ながら手で止めた値を後で書き入れる。

# 2) Stage 2: 足の向き 90° (target_sole_yaw_angle 1.5705)、ball_sole_yaw 25、足の向きの罰なし、逸脱フラグあり。
#    探索ノイズは Stage 1 と同じ [0.2, 1.0] (B-Human もエントロピーの範囲は Stage 1 のまま)。
$T --task Isaac-K1-BHuman-Walk-Kick-Stage2-v0 --max_iterations 3000 --resume \
   --load_run 2026-10-24_22-59-02_s1c_thru_strong10 --checkpoint model_2300.pt \
   --reset_noise_std 0.2 --run_name s2_inside agent.algorithm.std_range=[0.2,1.0] agent.algorithm.actor_freeze_iterations=20

# 3) Stage 3〜7 は Stage 2 の結果を見てから (target_sole_yaw_sigma 0.5 → 0.25 → 0.1 は
#    env.commands.kick.target_sole_yaw_sigma=... で上書き、ノイズ [0.13, 0.17]、Stage 6 は desired_kl 0.003・ノイズ [0.12, 0.13])。

# 評価:
#   IsaacLab: bash /workspace/claude_tmp/bwk_run.sh scripts/rsl_rl/eval_bhuman_walk_kick.py --task Isaac-K1-BHuman-Walk-Kick-Stage2-v0 \
#             --checkpoint <model.pt> --num_envs 256 --steps 1500 --headless --device cuda:0
#   MuJoCo (ホスト): ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_walk_kick.py --checkpoint <model.pt> --random 100
#   B-Human の完成品 (比較の基準): ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_reference.py --random 100

# ---- 以下は前回 (2026-10-25、つま先キック、仕様の蹴り方・当て方 0%) の Stage 2〜7。記録として残す ----
# 2) Stage 2 (2300 → 2700)
# $T --task Isaac-K1-BHuman-Walk-Kick-Stage2-v0 --max_iterations 400 --resume --load_run '.*_s1$' --checkpoint model_2300.pt \
#    --reset_noise_std 0.2 --run_name s2 agent.algorithm.std_range=[0.2,1.0] agent.algorithm.actor_freeze_iterations=20

# 3) Stage 3 (2700 → 3000)。ノイズを B-Human のエントロピー [-15,-14] 相当の 0.13〜0.17 に下げる
# $T --task Isaac-K1-BHuman-Walk-Kick-Stage3-v0 --max_iterations 300 --resume --load_run '.*_s2$' --checkpoint model_2700.pt \
#    --reset_noise_std 0.13 --run_name s3 agent.algorithm.std_range=[0.13,0.17] agent.algorithm.actor_freeze_iterations=10

# 4) Stage 4 (3000 → 3500)。feet_phase は下げない・当てる高さの係数あり (bhuman_walk_kick_env_cfg.py の Stage 4)
# $T --task Isaac-K1-BHuman-Walk-Kick-Stage4-v0 --max_iterations 500 --resume --load_run '.*_s3$' --checkpoint model_3000.pt \
#    --reset_noise_std 0.13 --run_name s4d_strike agent.algorithm.std_range=[0.13,0.17] agent.algorithm.actor_freeze_iterations=10

# 5) Stage 5 (3500 → 3900)
# $T --task Isaac-K1-BHuman-Walk-Kick-Stage5-v0 --max_iterations 400 --resume --load_run '.*_s4d_strike$' --checkpoint model_3500.pt \
#    --reset_noise_std 0.13 --run_name s5 agent.algorithm.std_range=[0.13,0.17] agent.algorithm.actor_freeze_iterations=10

# 6) Stage 6 (3900 → 4400)、続けて sigma 0.15 (4400 → 4800)。ノイズ 0.12〜0.13 (エントロピー [-16.6,-16.5] 相当)、desired_kl 0.003
# $T --task Isaac-K1-BHuman-Walk-Kick-Stage6-v0 --max_iterations 500 --resume --load_run '.*_s5$' --checkpoint model_3900.pt \
#    --reset_noise_std 0.12 --run_name s6 agent.algorithm.std_range=[0.12,0.13] agent.algorithm.desired_kl=0.003 \
#    agent.algorithm.actor_freeze_iterations=10
# $T --task Isaac-K1-BHuman-Walk-Kick-Stage6-v0 --max_iterations 400 --resume --load_run '.*_s6$' --checkpoint model_4400.pt \
#    --reset_noise_std 0.12 --run_name s6b_sig015 agent.algorithm.std_range=[0.12,0.13] agent.algorithm.desired_kl=0.003 \
#    agent.algorithm.actor_freeze_iterations=0 env.commands.kick.kick_direction_sigma_range=0.15
# ↑ この model_4800 を最終の方策にした (MuJoCo で一番ずれが小さく転倒 0)。
# Stage 7 (確率を最終値の 10% → 最終値) も回したが (s7a / s7b_final)、MuJoCo ではずれが 7.0° → 8.5〜9.2° に増えたので採用していない:
#   --task Isaac-K1-BHuman-Walk-Kick-Stage7-v0 ... env.commands.kick.kick_direction_sigma_range=0.15 \
#   env.commands.kick.random_ball_reset=[0.001,0.002,0.002] env.commands.kick.reset_ball_request=0.0005 \
#   env.commands.kick.add_ball_vel=0.00001 env.commands.kick.add_ball_vel_in_direction=0.00001   (7a、その後 7b は既定値)
