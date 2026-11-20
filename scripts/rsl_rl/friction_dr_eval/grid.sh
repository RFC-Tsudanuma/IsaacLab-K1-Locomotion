#!/bin/bash
# grid.sh <名前> <ckpt> <mu> [num_envs]: 指令応答グリッドの計測 (MEASURE_GRID)。ログ grid_<名前>_mu<mu>.log
cd /workspace/IsaacLab-K1-Locomotion/scripts/rsl_rl; source ~/.bash_functions; unset DISPLAY
export VK_DRIVER_FILES=/tmp/no_icd.json VK_ICD_FILENAMES=/tmp/no_icd.json
FR="env.events.physics_material.params.static_friction_range=[$3,$3] env.events.physics_material.params.dynamic_friction_range=[$3,$3]"
G="${GRID:-0.5,0,0;1.0,0,0;1.5,0,0;2.0,0,0;-1.0,0,0;-2.0,0,0;0,0.5,0;0,1.0,0;0,1.5,0;0,1.9,0;1.0,1.0,0;2.0,1.0,0;0,0,1.0;1.0,0,1.0;2.0,0,1.0;0,1.0,1.0}"
MEASURE_GRID="$G" _labpython2 measure_stride.py --task Isaac-Velocity-Flat-Stance-Plane-Wide --headless --num_envs ${4:-2048} --device cuda:0 --checkpoint $2 $FR $EXTRA > grid_$1_mu$3.log 2>&1
echo "== $1 mu=$3"; grep -A40 "command response" grid_$1_mu$3.log | grep "^cmd\|^(" ; grep -B3 "HYDRA_FULL_ERROR" grid_$1_mu$3.log | head -4
