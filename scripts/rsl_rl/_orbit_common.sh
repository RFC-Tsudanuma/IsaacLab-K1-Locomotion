#!/usr/bin/env bash
# orbit 系 (walk_weak_kick_orbit / walk_long_pass_orbit) の通しスクリプト共通部。
#
# 単体では実行しない。各通しスクリプトから source して使う。
# 提供するもの:
#   * REPO_ROOT へ cd 済み (train.py は logs/ を CWD 基準で作るため)
#   * $LAB_PY_CMD[@] … IsaacLab の python (train_walk_kick_360.sh と同じ解決手順)
#   * should_run N   … STAGE 環境変数に N が含まれるか (STAGE=all なら常に真)
#   * find_latest_ckpt DIR … experiment ディレクトリの最新 run の最終 checkpoint
#   * run_stage ...  … 1 段ぶんの train.py 実行 (下の説明参照)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# IsaacLab の python を解決する (train_walk_kick_360.sh と同じ手順)。
# --------------------------------------------------------------------------- #
for _bf in "${BASH_FUNCTIONS:-}" "$HOME/.bash_functions"; do
    if [[ -n "$_bf" && -f "$_bf" ]]; then
        # shellcheck disable=SC1090
        source "$_bf"
        break
    fi
done

# NOTE: isaaclab.sh は "-p" を伴うので **必ず配列で持つ**。1 個の文字列にして
#       クォート付きで渡すと "isaaclab.sh -p" という名前の実行ファイルを探しに行き、
#       "No such file or directory" になる (既存スクリプトはクォート無し展開の
#       単語分割に頼っているが、ここは配列で明示的に扱う)。
LAB_PY_CMD=()
if [[ -n "${LAB_PYTHON:-}" ]]; then
    # ユーザー指定は "path" でも "path -p" でもよいよう単語分割する。
    read -r -a LAB_PY_CMD <<< "$LAB_PYTHON"
elif type _labpython2 >/dev/null 2>&1; then
    # bash 関数のこともあるが、配列展開でも通常のコマンド検索が走るので呼べる。
    LAB_PY_CMD=(_labpython2)
else
    for _cand in "$REPO_ROOT/isaaclab.sh" /workspace/isaaclab/isaaclab.sh /isaac-sim/python.sh; do
        if [[ -x "$_cand" ]]; then
            case "$_cand" in
                *isaaclab.sh) LAB_PY_CMD=("$_cand" -p) ;;
                *)            LAB_PY_CMD=("$_cand") ;;
            esac
            break
        fi
    done
    if [[ ${#LAB_PY_CMD[@]} -eq 0 ]]; then
        for _cand in python python3; do
            if command -v "$_cand" >/dev/null 2>&1 && "$_cand" -c "import isaaclab" >/dev/null 2>&1; then
                LAB_PY_CMD=("$_cand")
                break
            fi
        done
    fi
fi

if [[ ${#LAB_PY_CMD[@]} -eq 0 ]]; then
    echo "[ERROR] IsaacLab の python が見つかりません。LAB_PYTHON で明示してください。" >&2
    exit 1
fi
echo "[INFO] python: ${LAB_PY_CMD[*]}"

NUM_ENVS=${NUM_ENVS:-4096}
STAGE=${STAGE:-all}

# --------------------------------------------------------------------------- #
# マルチ GPU (torchrun による DDP)
#
# GPUS=1 (既定) では従来どおり単一プロセスで起動する。2 以上にすると
#   <python> -m torch.distributed.run --nnodes=1 --nproc_per_node=$GPUS \
#            scripts/rsl_rl/train.py ... --distributed
# の形で起動する。train.py 側は --distributed を受けると sim/agent の device を
# cuda:<local_rank> に振り、seed に local_rank を足して rank ごとに散らす。
#
# NOTE: --num_envs は **GPU 1 枚あたり** の数。IsaacLab は rank ごとに
#       num_envs 個の env を作るので、合計は NUM_ENVS × GPUS になる。
#       train.py を直接叩いたときと意味を揃えるため、ここでは割り算しない。
#       合計を据え置きたいなら NUM_ENVS を GPUS で割って渡すこと。
#
# 使う GPU を選ぶときは CUDA_VISIBLE_DEVICES で絞る:
#   CUDA_VISIBLE_DEVICES=0,1 GPUS=2 ./scripts/rsl_rl/train_walk_weak_kick_orbit.sh
#
# 同じマシンで 2 本同時に回すときは MASTER_PORT をずらす (既定 29500)。
# --------------------------------------------------------------------------- #
GPUS=${GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29500}

if [[ ! "$GPUS" =~ ^[0-9]+$ ]] || [[ "$GPUS" -lt 1 ]]; then
    echo "[ERROR] GPUS は 1 以上の整数で指定してください (指定値: $GPUS)" >&2
    exit 1
fi
if [[ "$GPUS" -gt 1 ]]; then
    echo "[INFO] マルチ GPU: nproc_per_node=$GPUS  master_port=$MASTER_PORT"
    echo "[INFO] env 数: $NUM_ENVS / GPU  (合計 $((NUM_ENVS * GPUS)))"
fi

should_run() { [[ "$STAGE" == "all" || "$STAGE" == *"$1"* ]]; }

# 指定 experiment ディレクトリの最新 run から最終 checkpoint を拾う。
# run 名は YYYY-MM-DD_HH-MM-SS (辞書順=時刻順)、model_*.pt は sort -V で数値順。
# NOTE: 「最新 run」を取ってから checkpoint を探すのではなく、**model_*.pt を実際に
#       持っている run のうち最新** を取る。理由が 2 つある:
#         * 中断した run (model_*.pt が無い / model_0.pt だけ) が混ざっていても飛ばせる
#         * マルチ GPU 時、train.py は log_dir を各プロセスのタイムスタンプで作る
#           (rank 0 限定のガードが無い)。秒をまたぐと rank ごとに別ディレクトリが
#           できるので、checkpoint を持たない方を掴まないようにする
find_latest_ckpt() {
    local run ckpt
    while IFS= read -r run; do
        ckpt=$(find "$run" -maxdepth 1 -name 'model_*.pt' 2>/dev/null | sort -V | tail -n 1)
        if [[ -n "$ckpt" ]]; then
            echo "$ckpt"
            return 0
        fi
    done < <(find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r)

    echo "[ERROR] checkpoint を持つ run が見つかりません: $1" >&2
    echo "[ERROR] 先に前段を回すか、<STAGE名>_CKPT で明示してください。" >&2
    return 1
}

# run_stage <見出し> <task> <iters> <checkpoint or ""> [追加引数...]
#
# checkpoint が空文字なら --load_pretrained を付けない (Stage 1 用)。
# EXTRA_ARGS (呼び出し側が "$@" を入れる配列) は最後に展開する。
run_stage() {
    local title="$1" task="$2" iters="$3" ckpt="$4"
    shift 4
    echo "=============================================================="
    echo " $title"
    echo " task=$task  iters=$iters  num_envs=$NUM_ENVS/GPU  gpus=$GPUS"
    [[ -n "$ckpt" ]] && echo " pretrained: $ckpt"
    echo "=============================================================="

    local -a cmd=("${LAB_PY_CMD[@]}")
    # GPUS>1 なら torchrun (torch.distributed.run) を挟む。isaaclab.sh -p は
    # 残りの引数をそのまま python に渡すので -m がそのまま効く。
    if [[ "$GPUS" -gt 1 ]]; then
        cmd+=(-m torch.distributed.run
            --nnodes=1 --nproc_per_node="$GPUS" --master_port="$MASTER_PORT")
    fi
    cmd+=(scripts/rsl_rl/train.py
        --task "$task" --headless
        --num_envs "$NUM_ENVS" --max_iterations "$iters")
    [[ -n "$ckpt" ]] && cmd+=(--load_pretrained "$ckpt")
    [[ "$GPUS" -gt 1 ]] && cmd+=(--distributed)
    cmd+=("$@")

    "${cmd[@]}"
}
