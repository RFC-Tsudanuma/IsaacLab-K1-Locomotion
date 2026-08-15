#!/usr/bin/env bash
# vast.ai の新しいインスタンスに、このリポジトリを clone して pip install するまでを一発でやる。
#
# 使い方 (ローカルから ssh 越しに流す。スクリプト自体が転送されるのでリモートに置く必要は無い):
#   VAST_SSH="-p 40712 root@ssh4.vast.ai" ./scripts/setup_vast.sh
#   ./scripts/setup_vast.sh -H ssh4.vast.ai -P 40712
#   ./scripts/setup_vast.sh -H ssh4.vast.ai -P 40712 -b master    # ブランチ指定
#
# ブランチ既定値はローカルで今チェックアウトしているブランチ。
#
# 使い方 (リモートのシェルに入って直接実行する場合):
#   bash <(curl -fsSL https://raw.githubusercontent.com/RFC-Tsudanuma/IsaacLab-K1-Locomotion/master/scripts/setup_vast.sh) -b feat/walk_kick
#   ./scripts/setup_vast.sh --here          # 既に clone 済みのディレクトリで再実行
#
# オプション:
#   -b, --branch <NAME>   チェックアウトするブランチ (既定: ローカルの現在ブランチ)
#   -r, --root <DIR>      clone 先の親ディレクトリ (既定: /workspace, 環境変数 VAST_REMOTE_PARENT)
#   -u, --repo <URL>      clone 元 (既定: https://github.com/RFC-Tsudanuma/IsaacLab-K1-Locomotion.git)
#       --here            ssh を使わず、このマシン上で実行する
#       --no-pull         既存の clone がある場合に fetch/pull しない
#   -H, --host / -P, --port / VAST_SSH   接続先 (fetch_ckpt.sh と同じ指定方法)
#   VAST_PYTHON=...       リモートの IsaacLab python を明示する
#                         (例: VAST_PYTHON=/workspace/isaaclab/isaaclab.sh\ -p)
set -euo pipefail

REPO_URL_DEFAULT="https://github.com/RFC-Tsudanuma/IsaacLab-K1-Locomotion.git"
PARENT_DEFAULT="${VAST_REMOTE_PARENT:-/workspace}"

BRANCH=""
PARENT="$PARENT_DEFAULT"
REPO_URL="$REPO_URL_DEFAULT"
HERE=0
PULL=1
HOST="${VAST_HOST:-}"
PORT="${VAST_PORT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--branch) BRANCH="$2"; shift 2 ;;
    -r|--root)   PARENT="$2"; shift 2 ;;
    -u|--repo)   REPO_URL="$2"; shift 2 ;;
    --here)      HERE=1; shift ;;
    --no-pull)   PULL=0; shift ;;
    -H|--host)   HOST="$2"; shift 2 ;;
    -P|--port)   PORT="$2"; shift 2 ;;
    -h|--help)   awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

# --------------------------------------------------------------------------- #
# リモート実行モード (--here): このマシン上で clone → pip install する
# --------------------------------------------------------------------------- #
if [[ "$HERE" -eq 1 ]]; then
  [[ -n "$BRANCH" ]] || BRANCH="master"
  NAME="$(basename "$REPO_URL" .git)"
  DEST="$PARENT/$NAME"

  echo "[setup] repo   : $REPO_URL"
  echo "[setup] branch : $BRANCH"
  echo "[setup] dest   : $DEST"

  mkdir -p "$PARENT"
  if [[ -d "$DEST/.git" ]]; then
    echo "[setup] 既に clone 済み。fetch して切り替えます。"
    cd "$DEST"
    if [[ "$PULL" -eq 1 ]]; then
      git fetch --prune origin
      git switch "$BRANCH" 2>/dev/null || git switch -c "$BRANCH" --track "origin/$BRANCH"
      # ローカル変更で pull がこけても install まで進める
      git pull --ff-only || echo "[setup] WARN: pull できませんでした (ローカル変更あり?)。現在の状態のまま続けます。" >&2
    else
      git switch "$BRANCH" 2>/dev/null || git switch -c "$BRANCH" --track "origin/$BRANCH"
    fi
  else
    git clone --branch "$BRANCH" "$REPO_URL" "$DEST"
    cd "$DEST"
  fi

  echo "[setup] HEAD   : $(git rev-parse --short HEAD) $(git log -1 --pretty=%s)"

  # --- IsaacLab の python を解決 ---------------------------------------------
  # 順序は fetch_ckpt.sh / train_walk_kick.sh と同じ:
  #   VAST_PYTHON → ~/.bash_functions の _labpython2 → isaaclab.sh / python.sh → PATH の python
  LAB_PY="${VAST_PYTHON:-}"
  if [[ -z "$LAB_PY" ]]; then
    if [[ -f "$HOME/.bash_functions" ]]; then . "$HOME/.bash_functions"; fi
    if type _labpython2 >/dev/null 2>&1; then
      LAB_PY="_labpython2"
    else
      for _c in ./isaaclab.sh "$PARENT/isaaclab/isaaclab.sh" /workspace/isaaclab/isaaclab.sh /isaac-sim/python.sh; do
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
  if [[ -z "$LAB_PY" ]]; then
    for _c in python python3; do
      if command -v "$_c" >/dev/null 2>&1 && "$_c" -c "import isaaclab" >/dev/null 2>&1; then
        LAB_PY="$_c"
        break
      fi
    done
  fi
  if [[ -z "$LAB_PY" ]]; then
    echo "[setup] ERROR: IsaacLab の python が見つかりません。VAST_PYTHON で指定してください。" >&2
    exit 20
  fi
  echo "[setup] python : $LAB_PY"

  # --- 拡張のインストール -----------------------------------------------------
  # rsl-rl-lib はここで pyproject の固定版 (3.0.1) に揃う。
  # イメージに別バージョンが入っていても上書きされる。
  echo "[setup] pip install -e source/isaaclab_k1_locomotion"
  $LAB_PY -m pip install -e "$DEST/source/isaaclab_k1_locomotion" --root-user-action=ignore

  echo
  echo "[setup] 完了:"
  $LAB_PY -m pip show rsl-rl-lib 2>/dev/null | sed -n '1,2p' || true
  echo
  echo "  cd $DEST"
  echo "  ./scripts/rsl_rl/train_walk_kick.sh        # 例"
  exit 0
fi

# --------------------------------------------------------------------------- #
# ローカル実行モード: ssh 越しに自分自身を流す
# --------------------------------------------------------------------------- #
if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --abbrev-ref HEAD 2>/dev/null || echo master)"
fi

if [[ -n "${VAST_SSH:-}" ]]; then
  # 例: VAST_SSH="-p 40712 root@ssh4.vast.ai"  (vast.ai が出す ssh コマンドの引数部分)
  read -r -a _args <<<"$VAST_SSH"
  SSH_TARGET="${_args[-1]}"
  SSH_OPTS=("${_args[@]:0:${#_args[@]}-1}")
elif [[ -n "$HOST" ]]; then
  if [[ "$HOST" == *"@"* ]]; then SSH_TARGET="$HOST"; else SSH_TARGET="root@$HOST"; fi
  SSH_OPTS=()
  [[ -n "$PORT" ]] && SSH_OPTS=(-p "$PORT")
else
  echo "ssh 接続先が未指定です。VAST_SSH か -H/-P を指定してください。" >&2
  echo '  例: VAST_SSH="-p 40712 root@ssh4.vast.ai" ./scripts/setup_vast.sh' >&2
  echo '  (リモートのシェル上で直接動かすなら --here)' >&2
  exit 1
fi

echo "[setup] remote : $SSH_TARGET ${SSH_OPTS[*]-}"
echo "[setup] branch : $BRANCH"

REMOTE_ARGS=(--here --branch "$BRANCH" --root "$PARENT" --repo "$REPO_URL")
[[ "$PULL" -eq 0 ]] && REMOTE_ARGS+=(--no-pull)

# スクリプト本体を stdin で送り込むので、リモートに置いておく必要は無い。
# VAST_PYTHON はローカルで指定されていればそのまま引き継ぐ。
ssh -o StrictHostKeyChecking=accept-new "${SSH_OPTS[@]}" "$SSH_TARGET" \
  "VAST_PYTHON='${VAST_PYTHON:-}' bash -s -- $(printf '%q ' "${REMOTE_ARGS[@]}")" \
  < "${BASH_SOURCE[0]}"
