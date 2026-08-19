#!/usr/bin/env bash
# vast.ai のインスタンスから最新 run の最新チェックポイントを取ってくる。
#
# 使い方:
#   VAST_SSH="-p 40712 root@ssh4.vast.ai" ./scripts/fetch_ckpt.sh k1_walk_kick
#   ./scripts/fetch_ckpt.sh -H ssh4.vast.ai -P 40712 k1_walk_kick
#   ./scripts/fetch_ckpt.sh --list k1_walk_kick     # 取ってこずに候補だけ表示
#   ./scripts/fetch_ckpt.sh --full k1_walk_kick     # run ディレクトリ丸ごと
#   ./scripts/fetch_ckpt.sh --onnx k1_walk_kick     # exported/ (policy*.onnx) も取ってくる
#   ./scripts/fetch_ckpt.sh --video k1_walk_kick    # videos/ も取ってくる
#   ./scripts/fetch_ckpt.sh --no-tfevents k1_walk_kick # TensorBoard のログは要らない
#
# --onnx / --video を付けたときは、**ssh 越しにリモートで生成してから回収する**
# (既定)。リモートに古い成果物が残っていても必ず作り直す。生成には Isaac Sim の
# 起動を伴うので数分かかる。
#   --no-remote-exec        生成せず、無ければ従来どおり WARN だけ出す
#
# TensorBoard のログ (events.out.tfevents.*) は**既定で回収する**。既に学習が
# 書き出したものを取ってくるだけなので、リモートでの生成は伴わない (Isaac Sim も
# 起動しない)。要らないときは --no-tfevents を付ける。--full なら run ディレクトリ
# 丸ごとなので、いずれにせよ含まれる。
#
#   --gym-task <ID>         gym タスク ID の自動解決を上書きする
#   --video-length <N>      録画するステップ数 (既定 200)
#   VAST_PYTHON=...         リモートの IsaacLab python を明示する
#                           (例: VAST_PYTHON=/workspace/isaaclab/isaaclab.sh\ -p)
#
# run ディレクトリは YYYY-MM-DD_HH-MM-SS で始まるものだけを対象にする
# (10525 のような非日付ディレクトリは無視される)。
set -euo pipefail

REMOTE_ROOT="${VAST_REMOTE_ROOT:-/workspace/IsaacLab-K1-Locomotion}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TASK=""
FULL=0
LIST=0
ONNX=0
VIDEO=0
TFEVENTS=1
REMOTE_EXEC=1
GYM_TASK=""
VIDEO_LENGTH=200
HOST="${VAST_HOST:-}"
PORT="${VAST_PORT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -H|--host) HOST="$2"; shift 2 ;;
    -P|--port) PORT="$2"; shift 2 ;;
    --full)    FULL=1; shift ;;
    --list|-l) LIST=1; shift ;;
    --onnx)    ONNX=1; shift ;;
    --video)   VIDEO=1; shift ;;
    --tfevents) TFEVENTS=1; shift ;;
    --no-tfevents) TFEVENTS=0; shift ;;
    --no-remote-exec) REMOTE_EXEC=0; shift ;;
    --gym-task) GYM_TASK="$2"; shift 2 ;;
    --video-length) VIDEO_LENGTH="$2"; shift 2 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 1 ;;
    *)  TASK="$1"; shift ;;
  esac
done

# --- ssh 接続先の組み立て -----------------------------------------------------
if [[ -n "${VAST_SSH:-}" ]]; then
  # 例: VAST_SSH="-p 40712 root@ssh4.vast.ai"  (vast.ai が出す ssh コマンドの引数部分)
  read -r -a _args <<<"$VAST_SSH"
  SSH_TARGET="${_args[-1]}"
  SSH_OPTS=("${_args[@]:0:${#_args[@]}-1}")
  SCP_OPTS=()
  for a in "${SSH_OPTS[@]}"; do
    if [[ "$a" == "-p" ]]; then SCP_OPTS+=("-P"); else SCP_OPTS+=("$a"); fi
  done
elif [[ -n "$HOST" ]]; then
  if [[ "$HOST" == *"@"* ]]; then SSH_TARGET="$HOST"; else SSH_TARGET="root@$HOST"; fi
  SSH_OPTS=(); SCP_OPTS=()
  if [[ -n "$PORT" ]]; then SSH_OPTS=(-p "$PORT"); SCP_OPTS=(-P "$PORT"); fi
else
  echo "ssh 接続先が未指定です。VAST_SSH か -H/-P を指定してください。" >&2
  echo '  例: VAST_SSH="-p 40712 root@ssh4.vast.ai" ./scripts/fetch_ckpt.sh k1_walk_kick' >&2
  exit 1
fi

SSH=(ssh -o StrictHostKeyChecking=accept-new "${SSH_OPTS[@]}" "$SSH_TARGET")
SCP=(scp -o StrictHostKeyChecking=accept-new "${SCP_OPTS[@]}")

# --- タスク名が無ければリモートの一覧を出して終わる ---------------------------
if [[ -z "$TASK" ]]; then
  echo "タスク名を指定してください。リモートにあるのは:" >&2
  "${SSH[@]}" "ls -1 '$REMOTE_ROOT/logs/rsl_rl' 2>/dev/null" >&2 || true
  exit 1
fi

REMOTE_TASK_DIR="$REMOTE_ROOT/logs/rsl_rl/$TASK"

# --- 最新 run と最大 step を解決 ----------------------------------------------
read -r RUN STEP < <("${SSH[@]}" bash -s <<EOF
set -e
cd "$REMOTE_TASK_DIR" 2>/dev/null || { echo "リモートに $REMOTE_TASK_DIR がありません" >&2; exit 2; }
run=\$(ls -1d */ 2>/dev/null | tr -d '/' \
      | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}' \
      | sort | tail -1)
[ -n "\$run" ] || { echo "日付形式の run ディレクトリが見つかりません" >&2; exit 3; }
step=\$(ls -1 "\$run"/model_*.pt 2>/dev/null \
       | sed -e 's#.*/model_##' -e 's#\.pt\$##' | sort -n | tail -1)
[ -n "\$step" ] || { echo "\$run に model_*.pt がありません" >&2; exit 4; }
echo "\$run \$step"
EOF
)

echo "task : $TASK"
echo "run  : $RUN"
echo "ckpt : model_${STEP}.pt"

[[ "$LIST" -eq 1 ]] && exit 0

REMOTE_RUN_DIR="$REMOTE_TASK_DIR/$RUN"
REMOTE_CKPT="$REMOTE_RUN_DIR/model_${STEP}.pt"

# --------------------------------------------------------------------------- #
# experiment_name (= logs/rsl_rl/ 配下のディレクトリ名) から gym タスク ID を引く。
#
# RunnerCfg (agents/rsl_rl_ppo_cfg.py) の experiment_name 代入から
# クラス名を特定し、そのクラスを rsl_rl_cfg_entry_point に指定している
# gym.register の id を __init__.py から拾う。**ローカルのソースを読むだけ**で、
# Isaac Sim も import も要らない。
#
# 例: k1_walk_kick_360_moving_ball
#       -> K1WalkKick360MovingBallPPORunnerCfg
#       -> Isaac-Velocity-Flat-K1-Walk-Kick-360-Moving-Ball-v0        (学習用)
#          Isaac-Velocity-Flat-K1-Walk-Kick-360-Moving-Ball-Play-v0   (再生用)
#
# 学習用 ID は export に、Play 版は録画に使う (Play 版は num_envs=20・
# 観測ノイズ off・外乱イベント off なので、動画を撮るならこちら)。
# ローカルとリモートでソースがずれていると解決結果も食い違うので、
# その場合は --gym-task で明示すること。
# --------------------------------------------------------------------------- #
resolve_gym_task() {
  python3 - "$LOCAL_ROOT" "$1" <<'PYEOF'
import pathlib
import re
import sys

root, exp = pathlib.Path(sys.argv[1]), sys.argv[2]

# 1. experiment_name = "<exp>" を代入している RunnerCfg クラスを探す
needle = re.compile(r'experiment_name\s*=\s*["\']%s["\']' % re.escape(exp))
cls = None
for path in sorted(root.glob("source/**/agents/rsl_rl_ppo_cfg.py")):
    current = None
    for line in path.read_text().splitlines():
        m = re.match(r"class\s+(\w+)", line)
        if m:
            current = m.group(1)
        if needle.search(line):
            cls = current
            break
    if cls:
        break
if not cls:
    sys.exit(1)

# 2. そのクラスを rsl_rl_cfg_entry_point にしている gym.register の id を集める
ids = []
ref = "rsl_rl_ppo_cfg:%s" % cls
for path in sorted(root.glob("source/**/__init__.py")):
    text = path.read_text()
    if ref not in text:
        continue
    current = None
    for line in text.splitlines():
        m = re.search(r'id="([^"]+)"', line)
        if m:
            current = m.group(1)
        # 前方一致で別クラスを拾わないよう、閉じ引用符まで含めて照合する
        if (ref + '"') in line and current:
            ids.append(current)
if not ids:
    sys.exit(1)

play = next((i for i in ids if i.endswith("-Play-v0")), None)
base = next((i for i in ids if not i.endswith("-Play-v0")), None)
print("%s %s" % (base or play, play or base))
PYEOF
}

GYM_BASE=""
GYM_PLAY=""
if [[ -n "$GYM_TASK" ]]; then
  GYM_BASE="$GYM_TASK"
  GYM_PLAY="$GYM_TASK"
elif [[ "$ONNX" -eq 1 || "$VIDEO" -eq 1 ]] && [[ "$REMOTE_EXEC" -eq 1 ]]; then
  if read -r GYM_BASE GYM_PLAY < <(resolve_gym_task "$TASK"); then
    echo "gym  : $GYM_BASE (play: $GYM_PLAY)"
  else
    GYM_BASE=""; GYM_PLAY=""
    echo "WARN: '$TASK' に対応する gym タスク ID をローカルのソースから解決できませんでした。" >&2
    echo "      リモート生成をスキップします (--gym-task <ID> で明示できます)。" >&2
  fi
fi

# --------------------------------------------------------------------------- #
# リモートで IsaacLab の python を解決するための snippet。
# train_walk_kick.sh の LAB_PY 解決と同じ順序 (VAST_PYTHON → ~/.bash_functions の
# _labpython2 → isaaclab.sh / python.sh → PATH 上の python)。
# --------------------------------------------------------------------------- #
REMOTE_PY_RESOLVER='
LAB_PY="$VAST_PYTHON_REMOTE"
if [ -z "$LAB_PY" ]; then
  if [ -f "$HOME/.bash_functions" ]; then . "$HOME/.bash_functions"; fi
  if type _labpython2 >/dev/null 2>&1; then
    LAB_PY="_labpython2"
  else
    for _c in ./isaaclab.sh /workspace/isaaclab/isaaclab.sh /isaac-sim/python.sh; do
      if [ -x "$_c" ]; then
        case "$_c" in
          *isaaclab.sh) LAB_PY="$_c -p" ;;
          *)            LAB_PY="$_c" ;;
        esac
        break
      fi
    done
  fi
  if [ -z "$LAB_PY" ]; then
    for _c in python python3; do
      if command -v "$_c" >/dev/null 2>&1 && "$_c" -c "import isaaclab" >/dev/null 2>&1; then
        LAB_PY="$_c"
        break
      fi
    done
  fi
fi
if [ -z "$LAB_PY" ]; then
  echo "[remote] ERROR: IsaacLab の python が見つかりません。VAST_PYTHON で指定してください。" >&2
  exit 20
fi
echo "[remote] python: $LAB_PY"
'

# リモートで IsaacLab の python にスクリプトを食わせる。
#   $1    = 進捗表示用のラベル
#   $2... = python に渡す引数 (scripts/rsl_rl/xxx.py --foo ...)
remote_run_py() {
  local label="$1"; shift
  local cmd
  printf -v cmd '%q ' "$@"

  echo
  echo "[fetch_ckpt] === $label ==="
  echo "[fetch_ckpt] remote: cd $REMOTE_ROOT && \$LAB_PY $cmd"
  echo "[fetch_ckpt] (Isaac Sim の起動を伴うので数分かかります)"

  "${SSH[@]}" bash -s <<EOF
set -euo pipefail
# DISPLAY が残っていると Kit が --headless でも XOpenDisplay を呼んで segfault する
# (vast.ai コンテナで実測。対話シェルからの学習は通るのに ssh 経由だけ落ちる原因)。
unset DISPLAY XAUTHORITY
VAST_PYTHON_REMOTE='${VAST_PYTHON:-}'
cd '$REMOTE_ROOT'
$REMOTE_PY_RESOLVER
set -x
\$LAB_PY $cmd
EOF
}

# --------------------------------------------------------------------------- #
# 成果物をリモートで生成する (--onnx / --video 指定時)
# --------------------------------------------------------------------------- #
NEED_ONNX=0
NEED_VIDEO=0

if [[ "$REMOTE_EXEC" -eq 1 && -n "$GYM_BASE" ]] && [[ "$ONNX" -eq 1 || "$VIDEO" -eq 1 ]]; then
  # リモートに既に exported/ や videos/ があっても**必ず作り直す**。
  # cfg (カメラ・報酬など) をいじった直後に、古い成果物がそのまま回収されて
  # 「変更が効いていない」ように見えるのを防ぐ。
  NEED_ONNX="$ONNX"
  NEED_VIDEO="$VIDEO"

  # play.py は再生のついでに exported/ も書き出すので、動画を撮るなら
  # そちらで onnx もまとめて片付く (Isaac Sim の起動を 1 回減らせる)。
  if [[ "$NEED_VIDEO" -eq 1 ]]; then
    if remote_run_py "動画を生成 ($GYM_PLAY, ${VIDEO_LENGTH} steps)" \
        scripts/rsl_rl/play.py \
        --task "$GYM_PLAY" \
        --headless \
        --video \
        --video_length "$VIDEO_LENGTH" \
        --checkpoint "$REMOTE_CKPT"; then
      NEED_ONNX=0  # play.py が exported/ も書いた
    else
      echo "WARN: リモートでの動画生成に失敗しました。回収は続行します。" >&2
    fi
  fi

  if [[ "$NEED_ONNX" -eq 1 ]]; then
    if ! remote_run_py "ONNX をエクスポート ($GYM_BASE)" \
        scripts/rsl_rl/export_policy.py \
        --task "$GYM_BASE" \
        --headless \
        --num_envs 1 \
        --checkpoint "$REMOTE_CKPT"; then
      echo "WARN: リモートでの ONNX エクスポートに失敗しました。回収は続行します。" >&2
    fi
  fi
fi

# --- 転送 ---------------------------------------------------------------------
LOCAL_RUN_DIR="$LOCAL_ROOT/logs/rsl_rl/$TASK/$RUN"
mkdir -p "$LOCAL_RUN_DIR"

if [[ "$FULL" -eq 1 ]]; then
  "${SCP[@]}" -r "$SSH_TARGET:$REMOTE_RUN_DIR/." "$LOCAL_RUN_DIR/"
else
  # play に必要なのは checkpoint と params/ (env.yaml, agent.yaml)
  "${SCP[@]}" "$SSH_TARGET:$REMOTE_CKPT" "$LOCAL_RUN_DIR/"
  "${SCP[@]}" -r "$SSH_TARGET:$REMOTE_RUN_DIR/params" "$LOCAL_RUN_DIR/" 2>/dev/null || true
fi

if [[ "$ONNX" -eq 1 && "$FULL" -eq 0 ]]; then
  # export_policy.py / play.py の出力 (exported/policy*.onnx)
  "${SCP[@]}" -r "$SSH_TARGET:$REMOTE_RUN_DIR/exported" "$LOCAL_RUN_DIR/" \
    || echo "WARN: リモートに exported/ がありません (エクスポートに失敗した?)" >&2
fi

if [[ "$VIDEO" -eq 1 && "$FULL" -eq 0 ]]; then
  # --video 付き学習の録画、または play.py --video の出力 (videos/)
  "${SCP[@]}" -r "$SSH_TARGET:$REMOTE_RUN_DIR/videos" "$LOCAL_RUN_DIR/" \
    || echo "WARN: リモートに videos/ がありません (録画に失敗した?)" >&2
fi

if [[ "$TFEVENTS" -eq 1 && "$FULL" -eq 0 ]]; then
  # rsl_rl の SummaryWriter は run ディレクトリ直下に events.out.tfevents.* を書く。
  # glob はリモートのシェルが展開するのでクォートしない。学習が途中でも
  # 追記中のファイルをそのままコピーできる (TensorBoard は末尾が欠けても読める)。
  "${SCP[@]}" "$SSH_TARGET:$REMOTE_RUN_DIR/events.out.tfevents.*" "$LOCAL_RUN_DIR/" \
    || echo "WARN: リモートに events.out.tfevents.* がありません" >&2
fi

echo
echo "→ $LOCAL_RUN_DIR"
ls -la "$LOCAL_RUN_DIR"
