#!/usr/bin/env bash
# vast.ai の新しいインスタンスに、このリポジトリを置いて学習できる状態にするまでを一発でやる。
#
# 使い方 (ローカルから ssh 越しに流す。スクリプト自体が転送されるのでリモートに置く必要は無い):
#   VAST_SSH="-p 40712 root@ssh4.vast.ai" ./scripts/setup_vast.sh --rsync     # ローカルの作業ツリーをそのまま送る
#   ./scripts/setup_vast.sh -H ssh4.vast.ai -P 40712 --rsync
#   ./scripts/setup_vast.sh -H ssh4.vast.ai -P 40712 -b master               # GitHub から clone (ブランチ指定)
#
# --rsync: GitHub を使わず、ローカルの作業ツリー (コミットしていない変更も含む) を rsync で送る。push 不要で、
#          リモートに GitHub の認証も要らない。.git (約 750 MB) は送らず、代わりに送った時点のコミットと差分を
#          VAST_SOURCE.txt / VAST_SOURCE.diff に書いて一緒に送る。logs/ は送らず、学習の出発点に使う
#          checkpoint だけを同じ相対パスで送る (下の CKPTS。--ckpt <repo からの相対パス> で追加)。
#          2 回目以降は差分だけ送られる。
#
# ブランチ既定値はローカルで今チェックアウトしているブランチ (clone の場合)。
#
# 使い方 (リモートのシェルに入って直接実行する場合):
#   bash <(curl -fsSL https://raw.githubusercontent.com/RFC-Tsudanuma/IsaacLab-K1-Locomotion/master/scripts/setup_vast.sh) -b feat/walk_kick
#   ./scripts/setup_vast.sh --here          # 既に clone 済みのディレクトリで再実行
#
# オプション:
#   -b, --branch <NAME>   チェックアウトするブランチ (既定: ローカルの現在ブランチ)
#   -r, --root <DIR>      clone 先の親ディレクトリ (既定: /workspace, 環境変数 VAST_REMOTE_PARENT)
#   -u, --repo <URL>      clone 元 (既定: https://github.com/RFC-Tsudanuma/IsaacLab-K1-Locomotion.git)
#       --rsync           clone せず、ローカルの作業ツリーを rsync で送る (上の説明)
#       --ckpt <PATH>     --rsync で追加で送るファイル (repo からの相対パス、何度でも指定可)
#       --uv              イメージに IsaacLab があっても使わず、pyproject.toml (uv) で入れた環境を使う
#       --here            ssh を使わず、このマシン上で実行する
#       --no-pull         既存の clone がある場合に fetch/pull しない
#   -H, --host / -P, --port / VAST_SSH   接続先 (fetch_ckpt.sh と同じ指定方法)
#   VAST_PYTHON=...       リモートの IsaacLab python を明示する
#                         (例: VAST_PYTHON=/workspace/isaaclab/isaaclab.sh\ -p)
#
# IsaacLab の python の探し方 (--here):
#   VAST_PYTHON → ~/.bash_functions の _labpython2 → isaaclab.sh / python.sh → <repo>/.venv/bin/python
#   → PATH の python (import isaaclab できるもの)。どれも無ければ (または --uv)、uv を入れて pyproject.toml の
#   固定版 (Isaac Sim 5.1.0・IsaacLab 2.3.2.post1・rsl-rl-lib 3.0.1・torch 2.7.0+cu128 = ML3 の docker コンテナと同じ版) で
#   <repo>/.venv を作る。初回は Isaac Sim の wheel を落とすので 10〜20 分かかる。Isaac Sim は RT コアのある GPU
#   (RTX 4090 / 3090 / L40S / A6000 など) と Ubuntu 22.04 以降 (glibc 2.35+) が前提。
set -euo pipefail

REPO_URL_DEFAULT="https://github.com/RFC-Tsudanuma/IsaacLab-K1-Locomotion.git"
PARENT_DEFAULT="${VAST_REMOTE_PARENT:-/workspace}"

# --rsync で送る学習の出発点 (repo からの相対パス。無いものは飛ばす)
CKPTS=(
  logs/rsl_rl/k1_bhuman_walk_kick/walk34995/model_0.pt
  logs/rsl_rl/k1_bhuman_walk_kick/walk52700/model_0.pt
  logs/rsl_rl/k1_bhuman_walk_kick/2026-10-31_06-40-27_s9c/model_11400.pt
)

BRANCH=""
PARENT="$PARENT_DEFAULT"
REPO_URL="$REPO_URL_DEFAULT"
HERE=0
PULL=1
RSYNC=0
NO_CLONE=0
USE_UV=0
HOST="${VAST_HOST:-}"
PORT="${VAST_PORT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--branch) BRANCH="$2"; shift 2 ;;
    -r|--root)   PARENT="$2"; shift 2 ;;
    -u|--repo)   REPO_URL="$2"; shift 2 ;;
    --rsync)     RSYNC=1; shift ;;
    --ckpt)      CKPTS+=("$2"); shift 2 ;;
    --uv)        USE_UV=1; shift ;;
    --no-clone)  NO_CLONE=1; shift ;;  # 内部用: --rsync で送った後のリモート側
    --here)      HERE=1; shift ;;
    --no-pull)   PULL=0; shift ;;
    -H|--host)   HOST="$2"; shift 2 ;;
    -P|--port)   PORT="$2"; shift 2 ;;
    -h|--help)   awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

# --------------------------------------------------------------------------- #
# リモート実行モード (--here): このマシン上で clone → install する
# --------------------------------------------------------------------------- #
if [[ "$HERE" -eq 1 ]]; then
  [[ -n "$BRANCH" ]] || BRANCH="master"
  NAME="$(basename "$REPO_URL" .git)"
  DEST="$PARENT/$NAME"

  echo "[setup] dest   : $DEST"
  mkdir -p "$PARENT"
  if [[ "$NO_CLONE" -eq 1 ]]; then
    # --rsync で送ったツリーをそのまま使う
    cd "$DEST"
    [[ -f VAST_SOURCE.txt ]] && sed -n '1,3p' VAST_SOURCE.txt | sed 's/^/[setup] source : /'
  else
    echo "[setup] repo   : $REPO_URL"
    echo "[setup] branch : $BRANCH"
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
  fi

  # --- IsaacLab の python を解決 ---------------------------------------------
  # 順序は fetch_ckpt.sh / vast_train.sh と同じ:
  #   VAST_PYTHON → ~/.bash_functions の _labpython2 → isaaclab.sh / python.sh → .venv → PATH の python
  LAB_PY="${VAST_PYTHON:-}"
  if [[ -z "$LAB_PY" && "$USE_UV" -eq 0 ]]; then
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
  if [[ -z "$LAB_PY" && "$USE_UV" -eq 0 && -x "$DEST/.venv/bin/python" ]]; then
    LAB_PY="$DEST/.venv/bin/python"
  fi
  if [[ -z "$LAB_PY" && "$USE_UV" -eq 0 ]]; then
    for _c in python python3; do
      if command -v "$_c" >/dev/null 2>&1 && "$_c" -c "import isaaclab" >/dev/null 2>&1; then
        LAB_PY="$_c"
        break
      fi
    done
  fi

  if [[ -z "$LAB_PY" ]]; then
    # --- IsaacLab が無い (または --uv): pyproject.toml の固定版で .venv を作る -----------
    echo "[setup] IsaacLab の python が無いので uv で .venv を作ります (pyproject.toml / uv.lock の固定版)。"
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
    fi
    export OMNI_KIT_ACCEPT_EULA=YES
    # Isaac Sim の wheel は大きいので、タイムアウトを長めにする
    UV_HTTP_TIMEOUT=600 uv sync --frozen </dev/null
    LAB_PY="$DEST/.venv/bin/python"
  fi
  echo "[setup] python : $LAB_PY"

  # --- 拡張のインストール -----------------------------------------------------
  if [[ "$LAB_PY" == "$DEST/.venv/bin/python" ]]; then
    # uv sync が pyproject の [tool.uv.sources] で editable に入れている
    echo "[setup] isaaclab_k1_locomotion は uv sync で editable に入っています"
  else
    echo "[setup] pip install -e source/isaaclab_k1_locomotion"
    $LAB_PY -m pip install -e "$DEST/source/isaaclab_k1_locomotion" --root-user-action=ignore </dev/null
    # 拡張の依存には rsl-rl-lib が無く、固定版 (3.0.1) は root の pyproject.toml / uv.lock にしか無いので、ここで揃える。
    # (vast の IsaacLab イメージには 3.1.2 が入っている)
    echo "[setup] pip install rsl-rl-lib==3.0.1"
    $LAB_PY -m pip install "rsl-rl-lib==3.0.1" --root-user-action=ignore </dev/null
  fi

  # --- 動作確認 (Isaac Sim は起動しない) ----------------------------------------
  echo "[setup] 確認: import と GPU"
  OMNI_KIT_ACCEPT_EULA=YES $LAB_PY - <<'PY' || echo "[setup] WARN: 確認に失敗しました (上のエラーを見てください)" >&2
import importlib.metadata as md
import torch
for p in ("isaacsim", "isaaclab", "rsl-rl-lib", "torch"):
    try:
        print(f"  {p:12s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p:12s} (not found)")
print("  cuda", torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
  for _c in "${CKPTS[@]}"; do
    [[ -f "$DEST/$_c" ]] && echo "[setup] ckpt   : $_c"
  done

  echo
  echo "[setup] 完了:"
  echo "  cd $DEST"
  echo "  ./scripts/vast_train.sh <run 名> <train.py の引数...>    # 学習 (nohup、ログは logs/vast/<run 名>.log)"
  exit 0
fi

# --------------------------------------------------------------------------- #
# ローカル実行モード: ssh 越しに自分自身を流す
# --------------------------------------------------------------------------- #
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git -C "$LOCAL_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo master)"
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
  echo '  例: VAST_SSH="-p 40712 root@ssh4.vast.ai" ./scripts/setup_vast.sh --rsync' >&2
  echo '  (リモートのシェル上で直接動かすなら --here)' >&2
  exit 1
fi
SSH_OPTS+=(-o StrictHostKeyChecking=accept-new)

echo "[setup] remote : $SSH_TARGET ${SSH_OPTS[*]}"

REMOTE_ARGS=(--here --root "$PARENT" --repo "$REPO_URL")
[[ "$USE_UV" -eq 1 ]] && REMOTE_ARGS+=(--uv)

if [[ "$RSYNC" -eq 1 ]]; then
  DEST="$PARENT/$(basename "$REPO_URL" .git)"
  echo "[setup] rsync  : $LOCAL_ROOT -> $SSH_TARGET:$DEST"
  # 送った時点のコミットと、コミットしていない変更を記録する (.git は送らないので)
  {
    echo "branch: $BRANCH"
    echo "commit: $(git -C "$LOCAL_ROOT" rev-parse HEAD) $(git -C "$LOCAL_ROOT" log -1 --pretty=%s)"
    echo "sent  : $(date '+%Y-%m-%d %H:%M:%S') from $(hostname)"
    echo "status:"
    git -C "$LOCAL_ROOT" status --short
  } > "$LOCAL_ROOT/VAST_SOURCE.txt"
  git -C "$LOCAL_ROOT" diff HEAD > "$LOCAL_ROOT/VAST_SOURCE.diff"

  RSH="ssh $(printf '%q ' "${SSH_OPTS[@]}")"
  # リモートに rsync が無いイメージがあるので先に入れる
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
    "mkdir -p '$DEST' && (command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null))"
  rsync -az --info=stats1 -e "$RSH" \
    --exclude '.git/' --exclude 'logs/' --exclude 'outputs/' --exclude '.venv/' \
    --exclude '__pycache__/' --exclude '*.pyc' --exclude 'MUJOCO_LOG.TXT' --exclude '*.egg-info/' \
    "$LOCAL_ROOT/" "$SSH_TARGET:$DEST/"
  # 学習の出発点の checkpoint (と同じ run の params/。--resume で env.yaml / agent.yaml は読まないが記録として)
  _list="$(mktemp)"
  for _c in "${CKPTS[@]}"; do
    if [[ -f "$LOCAL_ROOT/$_c" ]]; then
      echo "$_c" >> "$_list"
      _p="$(dirname "$_c")/params"
      [[ -d "$LOCAL_ROOT/$_p" ]] && (cd "$LOCAL_ROOT" && find "$_p" -type f) >> "$_list"
    else
      echo "[setup] WARN: ckpt がローカルに無いので飛ばします: $_c" >&2
    fi
  done
  rsync -az --info=stats1 -e "$RSH" --files-from="$_list" "$LOCAL_ROOT/" "$SSH_TARGET:$DEST/"
  rm -f "$_list" "$LOCAL_ROOT/VAST_SOURCE.txt" "$LOCAL_ROOT/VAST_SOURCE.diff"
  REMOTE_ARGS+=(--no-clone)
else
  echo "[setup] branch : $BRANCH"
  REMOTE_ARGS+=(--branch "$BRANCH")
  [[ "$PULL" -eq 0 ]] && REMOTE_ARGS+=(--no-pull)
fi

# スクリプト本体を stdin で送り込むので、リモートに置いておく必要は無い。
# VAST_PYTHON はローカルで指定されていればそのまま引き継ぐ。
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
  "VAST_PYTHON='${VAST_PYTHON:-}' bash -s -- $(printf '%q ' "${REMOTE_ARGS[@]}")" \
  < "${BASH_SOURCE[0]}"
