#!/bin/bash
# measf.sh <名前> <ckpt> <mu>: 摩擦固定で 前進/横/斜め を計測 (512 env)。出力ログ measure_fr_*.log
cd /workspace/IsaacLab-K1-Locomotion/scripts/rsl_rl; source ~/.bash_functions; unset DISPLAY
[ -f /tmp/no_icd.json ] || echo "{\"file_format_version\":\"1.0.1\",\"ICD\":{\"library_path\":\"/nonexistent.so\",\"api_version\":\"1.3.0\"}}" > /tmp/no_icd.json
export VK_DRIVER_FILES=/tmp/no_icd.json VK_ICD_FILENAMES=/tmp/no_icd.json
N=$1; C=$2; MU=$3
FR="env.events.physics_material.params.static_friction_range=[$MU,$MU] env.events.physics_material.params.dynamic_friction_range=[$MU,$MU]"
m(){ MEASURE_AXIS=$2 MEASURE_SPEEDS=$3 MEASURE_OTHER=$4 _labpython2 measure_stride.py --task Isaac-Velocity-Flat-Stance-Plane-Wide --headless --num_envs 512 --device cuda:0 --checkpoint $C $FR > measure_fr_$1.log 2>&1; echo "-- $1"; grep -A7 "stride measurement" measure_fr_$1.log | grep -v Warning | awk "NR>1 && NF>8{print \$1, \$2, \$7, \$8, \$(NF-3), \$(NF-2), \$(NF-1), \$NF}"; }
echo "== $N mu=$MU (cmd act slide fly% falls tilt angv power)"
m ${N}_mu${MU}_x 0 1.0,1.5,2.0 0; m ${N}_mu${MU}_y 1 1.0,1.5,1.9 0; m ${N}_mu${MU}_diag 0 2.0 1.0
