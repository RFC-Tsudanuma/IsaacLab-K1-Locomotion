#!/usr/bin/env bash
# ML3 (tailscale 越し) にある onnx を、手元の booster_k1_locomotion/assets に配置する。
#
# 使い方:
#   ./scripts/fetch_onnx_ml3.sh k1_walk_lob_hist_2026-08-18_11-26-58.onnx
#   ./scripts/fetch_onnx_ml3.sh lob_hist            # 部分一致でも可 (新しい順に候補提示)
#   ./scripts/fetch_onnx_ml3.sh -l                  # ML3 上の onnx 一覧
#   ./scripts/fetch_onnx_ml3.sh -l walk_kick        # 一覧を絞り込み
#   ./scripts/fetch_onnx_ml3.sh -y lob_hist         # 候補が複数でも最新を黙って取る
#   ./scripts/fetch_onnx_ml3.sh -o policy.onnx lob_hist   # 別名で置く
#   ./scripts/fetch_onnx_ml3.sh a b c               # 複数まとめて
#
# オプション:
#   -l, --list [PAT]   転送せずに候補を出すだけ
#   -o, --output NAME  配置先のファイル名を上書き (1 件指定時のみ)
#   -d, --dest DIR     配置先ディレクトリ (既定: assets)
#   -H, --host HOST    ssh 接続先 (既定: ml3 / ML3_HOST)
#   -r, --root PATH    ML3 側の探索ルート (複数回指定可)
#   -y, --yes          候補が複数でも最新を選ぶ / 上書き確認もしない
#   -n, --dry-run      転送せずに何をするかだけ表示
#       --no-login     tailscale が未ログインでも自動で up しない
#
# 環境変数:
#   ML3_HOST           ssh 接続先          (既定: ml3)
#   ML3_TS_NAME        tailscale のホスト名 (既定: hayashibaralab-ml3)
#   ML3_ROOTS          ML3 側の探索ルート   (空白区切り)
#   ONNX_DEST_DIR      配置先ディレクトリ
#   SSH_CONNECT_TIMEOUT ssh の接続待ち秒数 (既定: 120 / 認証待ちがあるので長め)
#
# 認証まわり:
#   * Tailscale SSH は接続のたびに追加認証を要求することがあり、そのとき ssh は
#       # To authenticate, visit: https://login.tailscale.com/a/xxxxxxxx
#     を出したままブラウザでの承認を待つ。このスクリプトはその URL を検出して
#     大きく表示し (WSL なら Windows 側のブラウザも開く)、認証が済んだら
#     Enter で再試行できるようにする。
#   * 手元の tailscale 自体が未ログイン (BackendState=NeedsLogin) なら
#     `tailscale up` を代わりに叩いて、こちらの認証 URL も同じように表示する。
set -uo pipefail

REMOTE_HOST="${ML3_HOST:-ml3}"
TS_NAME="${ML3_TS_NAME:-hayashibaralab-ml3}"
DEST_DIR="${ONNX_DEST_DIR:-$HOME/futbol_main/src/ros2_ws/src/booster_k1_locomotion/assets}"
CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-120}"

# ML3 側で onnx を探すルート。~ はリモートのシェルが展開する。
if [[ -n "${ML3_ROOTS:-}" ]]; then
  read -r -a ROOTS <<<"$ML3_ROOTS"
else
  ROOTS=(\~/workspace/IsaacLab-K1-Locomotion/logs \~/workspace/IsaacLab-K1-Locomotion/checkpoint)
fi

LIST=0
YES=0
DRY=0
AUTO_LOGIN=1
OUT_NAME=""
QUERIES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -l|--list)    LIST=1; shift ;;
    -o|--output)  OUT_NAME="$2"; shift 2 ;;
    -d|--dest)    DEST_DIR="$2"; shift 2 ;;
    -H|--host)    REMOTE_HOST="$2"; shift 2 ;;
    -r|--root)    ROOTS+=("$2"); shift 2 ;;
    -y|--yes)     YES=1; shift ;;
    -n|--dry-run) DRY=1; shift ;;
    --no-login)   AUTO_LOGIN=0; shift ;;
    -h|--help)    awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)           echo "unknown option: $1" >&2; exit 1 ;;
    *)            QUERIES+=("$1"); shift ;;
  esac
done

if [[ -t 1 ]]; then
  C_B=$'\033[1m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_R=$'\033[31m'; C_0=$'\033[0m'
else
  C_B=""; C_G=""; C_Y=""; C_R=""; C_0=""
fi
info() { echo "${C_B}[onnx]${C_0} $*"; }
warn() { echo "${C_Y}[onnx] WARN:${C_0} $*" >&2; }
die()  { echo "${C_R}[onnx] ERROR:${C_0} $*" >&2; exit 1; }

TMPDIR_SELF="$(mktemp -d)"
AUTH_FLAG="$TMPDIR_SELF/auth_url"
cleanup() { rm -rf "$TMPDIR_SELF"; }
trap cleanup EXIT

# --------------------------------------------------------------------------- #
# 認証 URL の表示
# --------------------------------------------------------------------------- #
URL_RE='https://login\.tailscale\.com/[A-Za-z0-9/._~%-]+'

show_auth_url() {
  local url="$1"
  echo >&2
  echo "${C_Y}================================================================${C_0}" >&2
  echo "${C_Y}  tailscale の認証が必要です。下の URL をブラウザで開いてください${C_0}" >&2
  echo "${C_Y}================================================================${C_0}" >&2
  echo >&2
  echo "    ${C_B}${url}${C_0}" >&2
  echo >&2
  # WSL なら Windows 側の既定ブラウザで開ける
  if command -v wslview >/dev/null 2>&1; then
    wslview "$url" >/dev/null 2>&1 && echo "    (既定のブラウザを開きました)" >&2
  elif command -v explorer.exe >/dev/null 2>&1; then
    explorer.exe "$url" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || true
  fi
}

# ssh / rsync の stderr を素通ししつつ、認証 URL を見つけたら表示する。
# 見つけた URL は $AUTH_FLAG に書いて、呼び出し側から参照できるようにする。
watch_auth() {
  local line url
  while IFS= read -r line; do
    printf '%s\n' "$line" >&2
    if [[ ! -s "$AUTH_FLAG" && "$line" =~ $URL_RE ]]; then
      url="${BASH_REMATCH[0]}"
      url="${url%$'\r'}"
      printf '%s\n' "$url" >"$AUTH_FLAG"
      show_auth_url "$url"
      echo "${C_B}[onnx]${C_0} 承認が終わるまで待っています... (Ctrl-C で中止)" >&2
    fi
  done
}

# --------------------------------------------------------------------------- #
# 手元の tailscale の状態
# --------------------------------------------------------------------------- #
ts_json() {
  tailscale status --json 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
$1
" 2>/dev/null
}

ts_state()    { ts_json 'print(d.get("BackendState", ""))'; }
ts_auth_url() { ts_json 'print(d.get("AuthURL", "") or "")'; }

ts_peer_online() {
  ts_json "
for p in (d.get('Peer') or {}).values():
    names = [p.get('HostName', ''), (p.get('DNSName', '') or '').split('.')[0]]
    if '$TS_NAME' in names:
        print('online' if p.get('Online') else 'offline')
        break
"
}

tailscale_login() {
  local logf="$TMPDIR_SELF/tsup.log"
  local -a cmd=(tailscale up --qr=false)

  # 権限が足りなければ sudo で叩き直す (--operator 未設定の環境向け)
  : >"$logf"
  if ! tailscale up --qr=false --timeout=1s >/dev/null 2>"$logf"; then
    if grep -qiE 'access denied|permission denied|must be run as root|operator' "$logf"; then
      info "tailscale up に root 権限が要ります (sudo で実行します)"
      cmd=(sudo tailscale up --qr=false)
    fi
  fi
  : >"$logf"

  info "tailscale up を実行中..."
  "${cmd[@]}" >"$logf" 2>&1 &
  local pid=$! url="" shown=0 waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [[ $shown -eq 0 ]]; then
      url="$(grep -m1 -oE "$URL_RE" "$logf" 2>/dev/null)"
      [[ -z "$url" ]] && url="$(ts_auth_url)"
      if [[ -n "$url" ]]; then show_auth_url "$url"; shown=1; info "認証を待っています... (Ctrl-C で中止)"; fi
    fi
    sleep 1
    waited=$((waited + 1))
    if [[ $waited -ge 300 ]]; then
      kill "$pid" 2>/dev/null
      [[ -n "$url" ]] && show_auth_url "$url"
      die "tailscale の認証が 5 分以内に完了しませんでした"
    fi
  done
  wait "$pid"; local rc=$?
  if [[ $rc -ne 0 ]]; then
    url="$(grep -m1 -oE "$URL_RE" "$logf" 2>/dev/null)"
    [[ -n "$url" ]] && show_auth_url "$url"
    sed -e 's/^/    /' "$logf" >&2
    die "tailscale up に失敗しました"
  fi
  info "${C_G}tailscale ログイン完了${C_0}"
}

ensure_tailscale() {
  command -v tailscale >/dev/null 2>&1 || { warn "tailscale コマンドが見つかりません (素の ssh で続行します)"; return 0; }

  local state; state="$(ts_state)"
  case "$state" in
    Running) ;;
    NeedsLogin|NoState|Stopped|Starting|"")
      warn "手元の tailscale が使える状態ではありません (BackendState=${state:-unknown})"
      local url; url="$(ts_auth_url)"
      [[ -n "$url" ]] && show_auth_url "$url"
      if [[ $AUTO_LOGIN -eq 1 ]]; then
        tailscale_login
      else
        die "'tailscale up' してから再実行してください"
      fi
      ;;
    *) warn "未知の BackendState=$state のまま続行します" ;;
  esac

  local peer; peer="$(ts_peer_online)"
  if [[ "$peer" == "offline" ]]; then
    warn "$TS_NAME が tailnet 上でオフラインです (ML3 側で tailscale が落ちている?)"
  fi
}

# --------------------------------------------------------------------------- #
# ML3 側の onnx を探す
# --------------------------------------------------------------------------- #
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout="$CONNECT_TIMEOUT" -o ServerAliveInterval=15)

# ssh を実行し、Tailscale SSH の認証 URL が出たら表示する。
# 認証待ちで落ちた場合は、承認後に Enter で再試行できる。
# stdout は呼び出し側に返す。
ssh_capture() {
  local remote_cmd="$1" out rc attempt=0
  while :; do
    : >"$AUTH_FLAG"
    out="$(ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$remote_cmd" 2> >(watch_auth))"
    rc=$?
    if [[ $rc -eq 0 ]]; then printf '%s' "$out"; return 0; fi
    # 認証 URL が出ていて、まだ承認前に切れたなら 1 度だけ待って再試行
    if [[ -s "$AUTH_FLAG" && $attempt -eq 0 && -t 0 ]]; then
      attempt=1
      echo >&2
      show_auth_url "$(cat "$AUTH_FLAG")"
      read -r -p "ブラウザで承認したら Enter を押してください (q で中止): " ans
      [[ "$ans" == "q" ]] && return "$rc"
      info "再接続します..."
      continue
    fi
    return "$rc"
  done
}

remote_find_cmd() {
  local cmd="for r in ${ROOTS[*]}; do [ -d \"\$r\" ] || continue; find \"\$r\" -type d -name .git -prune -o -type f -name '*.onnx' -printf '%T@\t%s\t%p\n'; done"
  printf 'bash -lc %q' "$cmd"
}

declare -a FILES=()
load_files() {
  local out rc
  out="$(ssh_capture "$(remote_find_cmd)")"; rc=$?
  if [[ $rc -ne 0 ]]; then
    echo >&2
    warn "$REMOTE_HOST に ssh できませんでした。確認するなら:"
    echo "    tailscale status | grep $TS_NAME" >&2
    echo "    tailscale ping $TS_NAME" >&2
    echo "    ssh -v $REMOTE_HOST" >&2
    exit 1
  fi
  [[ -n "$out" ]] || die "探索ルート (${ROOTS[*]}) に onnx が 1 つもありません。-r で指定してください"
  mapfile -t FILES < <(printf '%s\n' "$out" | sort -t$'\t' -k1,1nr)
}

fmt_row() {
  # $1 = "mtime\tsize\tpath", $2 = 連番 (空なら付けない)
  python3 - "$1" "${2:-}" <<'PY'
import datetime, os, sys
mtime, size, path = sys.argv[1].split("\t", 2)
ts = datetime.datetime.fromtimestamp(float(mtime)).strftime("%Y-%m-%d %H:%M")
mb = int(size) / 1024 / 1024
idx = sys.argv[2]
head = "  %2s) " % idx if idx else "  "
print("%s%-58s %s  %6.2f MB" % (head, os.path.basename(path), ts, mb))
print("      %s" % os.path.dirname(path))
PY
}

# クエリに一致する行を新しい順で返す。完全一致があればそれだけ。
match_files() {
  local q="${1,,}"; q="${q%.onnx}"
  local -a exact=() partial=()
  local row base lb
  for row in "${FILES[@]}"; do
    base="$(basename "${row##*$'\t'}")"
    lb="${base,,}"
    if [[ "${lb%.onnx}" == "$q" ]]; then
      exact+=("$row")
    elif [[ "$lb" == *"$q"* ]]; then
      partial+=("$row")
    fi
  done
  if [[ ${#exact[@]} -gt 0 ]]; then
    printf '%s\n' "${exact[@]}"
  elif [[ ${#partial[@]} -gt 0 ]]; then
    printf '%s\n' "${partial[@]}"
  fi
}

# rsync (無ければ scp) で 1 ファイル取ってくる。こちらも認証 URL を拾う。
transfer() {
  local src="$1" dst="$2" rc attempt=0
  while :; do
    : >"$AUTH_FLAG"
    if command -v rsync >/dev/null 2>&1; then
      rsync -h --partial --info=progress2 \
        -e "ssh ${SSH_OPTS[*]}" \
        "$REMOTE_HOST:$src" "$dst" 2> >(watch_auth)
    else
      scp "${SSH_OPTS[@]}" "$REMOTE_HOST:$src" "$dst" 2> >(watch_auth)
    fi
    rc=$?
    [[ $rc -eq 0 ]] && return 0
    if [[ -s "$AUTH_FLAG" && $attempt -eq 0 && -t 0 ]]; then
      attempt=1
      show_auth_url "$(cat "$AUTH_FLAG")"
      read -r -p "ブラウザで承認したら Enter を押してください (q で中止): " ans
      [[ "$ans" == "q" ]] && return "$rc"
      continue
    fi
    return "$rc"
  done
}

# --------------------------------------------------------------------------- #
ensure_tailscale
info "ML3 ($REMOTE_HOST) の onnx を検索中..."
load_files
info "${#FILES[@]} 件の onnx が見つかりました"

if [[ $LIST -eq 1 || ${#QUERIES[@]} -eq 0 ]]; then
  pat="${QUERIES[0]:-}"
  n=0
  for row in "${FILES[@]}"; do
    if [[ -n "$pat" ]]; then
      base="$(basename "${row##*$'\t'}")"
      [[ "${base,,}" == *"${pat,,}"* ]] || continue
    fi
    fmt_row "$row"
    n=$((n + 1))
  done
  [[ $n -eq 0 ]] && warn "'$pat' に一致する onnx はありません"
  echo
  info "配置先: $DEST_DIR"
  [[ $LIST -eq 1 ]] && exit 0
  echo "onnx 名を指定してください: $(basename "${BASH_SOURCE[0]}") <名前>" >&2
  exit 1
fi

if [[ -n "$OUT_NAME" && ${#QUERIES[@]} -gt 1 ]]; then
  die "-o は 1 件だけ指定したときにしか使えません"
fi

mkdir -p "$DEST_DIR" || die "配置先を作れません: $DEST_DIR"

failed=0
for q in "${QUERIES[@]}"; do
  echo
  mapfile -t hits < <(match_files "$q")

  if [[ ${#hits[@]} -eq 0 ]]; then
    warn "'$q' に一致する onnx が ML3 にありません (-l で一覧)"
    failed=1
    continue
  fi

  pick="${hits[0]}"
  if [[ ${#hits[@]} -gt 1 ]]; then
    if [[ $YES -eq 1 || ! -t 0 ]]; then
      info "'$q' に ${#hits[@]} 件一致 → 最新を選びます"
    else
      echo "${C_B}'$q' に ${#hits[@]} 件一致 (新しい順):${C_0}"
      i=1
      for row in "${hits[@]}"; do fmt_row "$row" "$i"; i=$((i + 1)); done
      read -r -p "どれにしますか? [1-${#hits[@]}, 既定 1, q=中止]: " ans
      [[ "$ans" == "q" ]] && { info "中止しました"; continue; }
      [[ -z "$ans" ]] && ans=1
      if ! [[ "$ans" =~ ^[0-9]+$ ]] || (( ans < 1 || ans > ${#hits[@]} )); then
        warn "不正な選択: $ans (スキップ)"; failed=1; continue
      fi
      pick="${hits[$((ans - 1))]}"
    fi
  fi

  src="${pick##*$'\t'}"
  name="${OUT_NAME:-$(basename "$src")}"
  dst="$DEST_DIR/$name"

  fmt_row "$pick"
  info "$REMOTE_HOST:$src"
  info "  → $dst"

  if [[ -e "$dst" && $YES -eq 0 && $DRY -eq 0 && -t 0 ]]; then
    read -r -p "既に存在します。上書きしますか? [y/N]: " ow
    [[ "$ow" =~ ^[yY]$ ]] || { info "スキップしました"; continue; }
  fi

  if [[ $DRY -eq 1 ]]; then
    info "(dry-run) 転送はしません"
    continue
  fi

  if ! transfer "$src" "$dst"; then
    warn "転送に失敗しました ($src)"
    failed=1
    continue
  fi

  # ハッシュ照合 (リモートに sha256sum があるときだけ)
  rsum="$(ssh_capture "sha256sum $(printf %q "$src") 2>/dev/null | cut -d' ' -f1")"
  if [[ -n "$rsum" ]]; then
    lsum="$(sha256sum "$dst" | cut -d' ' -f1)"
    if [[ "$rsum" == "$lsum" ]]; then
      info "${C_G}OK${C_0} sha256 一致 ($(du -h "$dst" | cut -f1))"
    else
      warn "sha256 が一致しません! (再転送してください)"
      failed=1
      continue
    fi
  else
    info "${C_G}OK${C_0} $(du -h "$dst" | cut -f1)"
  fi
done

echo
info "配置先: $DEST_DIR"
ls -lt "$DEST_DIR" | head -8
exit $failed
