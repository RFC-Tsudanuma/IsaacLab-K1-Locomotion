#!/usr/bin/env bash
# vast.ai のインスタンス上で scripts/rsl_rl/train.py をバックグラウンドで回す (ssh を切っても止まらない)。
# setup_vast.sh の後に、リモートの repo の直下で実行する。
#
# 使い方:
#   ./scripts/vast_train.sh [--gpu N] <run 名> <train.py の引数...>
#   例 (Stage 1、新しい歩行から):
#     ./scripts/vast_train.sh --gpu 0 nw_s1 --task Isaac-K1-BHuman-Walk-Kick-v0 --num_envs 4096 --max_iterations 2300 \
#         --resume --load_run walk52700 --checkpoint model_0.pt --reset_noise_std 0.2 agent.algorithm.std_range=[0.2,1.0]
#   ローカルから:
#     ssh -p 40712 root@ssh4.vast.ai 'cd /workspace/IsaacLab-K1-Locomotion && ./scripts/vast_train.sh nw_s1 ...'
#
# - --headless と --run_name <run 名> は自動で付ける。--device は付けない (CUDA_VISIBLE_DEVICES で 1 枚に絞るので
#   train.py の既定 cuda:0 = 指定した GPU)。
# - ログ: logs/vast/<run 名>.log、PID: logs/vast/<run 名>.pid。止めるときは kill $(cat logs/vast/<run 名>.pid)。
# - 結果は logs/rsl_rl/<experiment>/<日時>_<run 名>/ に出る。ローカルへは scripts/fetch_ckpt.sh で取ってくる。
# - python の探し方は setup_vast.sh と同じ (VAST_PYTHON → _labpython2 → isaaclab.sh / python.sh → .venv → PATH)。
set -euo pipefail

GPU="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "${1:-}" == "--gpu" ]]; then GPU="$2"; shift 2; fi
if [[ $# -lt 1 || "$1" == -* ]]; then
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  exit 1
fi
RUN="$1"; shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LAB_PY="${VAST_PYTHON:-}"
if [[ -z "$LAB_PY" ]]; then
  if [[ -f "$HOME/.bash_functions" ]]; then . "$HOME/.bash_functions"; fi
  if type _labpython2 >/dev/null 2>&1; then
    LAB_PY="_labpython2"
  else
    for _c in ./isaaclab.sh /workspace/isaaclab/isaaclab.sh /isaac-sim/python.sh; do
      if [[ -x "$_c" ]]; then
        case "$_c" in
          *isaaclab.sh) LAB_PY="$_c -p" ;;
          *)            LAB_PY="$_c" ;;
        esac
        break
      fi
    done
  fi
fi
if [[ -z "$LAB_PY" && -x "$ROOT/.venv/bin/python" ]]; then
  LAB_PY="$ROOT/.venv/bin/python"
fi
if [[ -z "$LAB_PY" ]]; then
  for _c in python python3; do
    if command -v "$_c" >/dev/null 2>&1 && "$_c" -c "import isaaclab" >/dev/null 2>&1; then
      LAB_PY="$_c"
      break
    fi
  done
fi
if [[ -z "$LAB_PY" ]]; then
  echo "[vast_train] ERROR: IsaacLab の python が見つかりません。先に setup_vast.sh を流すか VAST_PYTHON で指定してください。" >&2
  exit 20
fi

mkdir -p logs/vast
LOG="logs/vast/$RUN.log"
if [[ -f "logs/vast/$RUN.pid" ]] && kill -0 "$(cat "logs/vast/$RUN.pid")" 2>/dev/null; then
  echo "[vast_train] ERROR: $RUN はもう動いています (pid $(cat "logs/vast/$RUN.pid"))" >&2
  exit 1
fi

# DISPLAY が残っていると Kit が --headless でも XOpenDisplay を呼んで segfault する (fetch_ckpt.sh 参照)
unset DISPLAY XAUTHORITY
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/source/isaaclab_k1_locomotion${PYTHONPATH:+:$PYTHONPATH}"

echo "[vast_train] run    : $RUN (GPU $GPU)"
echo "[vast_train] python : $LAB_PY"
echo "[vast_train] log    : $ROOT/$LOG"
# _labpython2 はシェル関数なので nohup に直接渡せない。bash -c の中で読み直す。
nohup bash -c "$( [[ "$LAB_PY" == _labpython2 ]] && echo '. "$HOME/.bash_functions";' ) \
  $LAB_PY scripts/rsl_rl/train.py --headless --run_name $(printf '%q' "$RUN") $(printf '%q ' "$@")" \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "logs/vast/$RUN.pid"
echo "[vast_train] pid    : $(cat "logs/vast/$RUN.pid")"
echo "  tail -f $ROOT/$LOG"
