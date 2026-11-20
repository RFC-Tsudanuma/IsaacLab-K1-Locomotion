#!/bin/bash
# falls.sh <名前> <ckpt> <mu>: 転倒の精密計測 (4096 env, 15 s)。前進 2.0 / 横 1.9 / 斜め (2.0,1.0)
cd /workspace/IsaacLab-K1-Locomotion/scripts/rsl_rl; source ~/.bash_functions; unset DISPLAY
export VK_DRIVER_FILES=/tmp/no_icd.json VK_ICD_FILENAMES=/tmp/no_icd.json
N=$1; C=$2; MU=$3
FR="env.events.physics_material.params.static_friction_range=[$MU,$MU] env.events.physics_material.params.dynamic_friction_range=[$MU,$MU]"
m(){ MEASURE_STEPS=900 MEASURE_AXIS=$2 MEASURE_SPEEDS=$3 MEASURE_OTHER=$4 _labpython2 measure_stride.py --task Isaac-Velocity-Flat-Stance-Plane-Wide --headless --num_envs 4096 --device cuda:0 --checkpoint $C $FR $EXTRA > falls_$1.log 2>&1; echo "-- $1"; grep -A3 "stride measurement" falls_$1.log | grep -v "Warning\|===" | awk "NF>8{print \$1, \$2, \$7, \$8, \$(NF-3)}"; grep -A2 "falls/env within\|fall direction" falls_$1.log | grep -v "===\|^--"; }
echo "== $N mu=$MU (cmd act slide fly% falls/15s)"
m ${N}_mu${MU}_x 0 1.0,2.0 0; m ${N}_mu${MU}_y 1 1.9 0; m ${N}_mu${MU}_diag 0 2.0 1.0
