#!/usr/bin/env bash
# vast.ai: 新しい歩行 model_52700 の上の bhuman_walk_kick を、Stage 2 (nw_s2) の後に Stage 3〜9c まで間を空けずに続けて回す。
# 各段は前の段が作った run の最後の checkpoint から --resume する。段の並び・iteration 数・上書きは、
# 歩行 model_34995 の上で採用した系列 (s2_inside → s2b_ballfric → s3a_sig05 → s3b_sig025 → s3b_mirrorfull → s3c_sig01 →
# s4 → s5 → s6 → s6b_sig02 → s6c_vel8_strong4 (9200 まで) → s7a → s7b → s9a (10100 まで) → s9b → s9c (11400)) と同じ。
# 値の出どころは logs/rsl_rl/k1_bhuman_walk_kick/<旧 run>/params/{env,agent}.yaml と claude_tmp/bwk2/train_*.sh。
#
# 使い方 (リモートの repo の直下で):
#   nohup setsid ./scripts/vast_chain_new_walk.sh > /dev/null 2>&1 < /dev/null &   # 本番
#   ./scripts/vast_chain_new_walk.sh --dry-run                                     # コマンドを表示するだけ
#
# - ログ: logs/vast/chain.log (時刻つき)。PID: logs/vast/chain.pid。止めるときは kill $(cat logs/vast/chain.pid)
#   (今動いている段の学習は止まらない。止めるなら kill $(cat logs/vast/<段の名前>.pid) と、その下の python も)。
# - 段の iteration 数 N は旧 run の「読み込んだ checkpoint の番号 S → 次の段が読んだ番号 E」から N = E - S + 1
#   (rsl_rl は読み込んだ番号から数え始め、最後のファイルが model_<S+N-1>.pt になる。旧 run で手で止めた所と update 回数が同じ)。
# - 各段の後に「最後の checkpoint (model_<S+N-1>.pt) があるか」を確かめ、無ければ鎖を止めて理由をログに書く。
# - train.py が他に動いていたら段を始めない (10 分待っても消えなければ鎖を止める)。
# - Stage 2 を早めに切り上げたいとき (旧系列の Stage 2 は s2_inside + s2b で合計 1101 update = 新しい番号で model_3400 相当):
#     echo 3400 > logs/vast/chain_s2_stop_at
#   と書いておくと、nw_s2 に model_3400.pt ができた時点で鎖が nw_s2 を止めて、そこから Stage 3 を始める。
#   ファイルが無ければ nw_s2 には手を出さず、終わる (model_5298.pt) のを待つ。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EXP_DIR="logs/rsl_rl/k1_bhuman_walk_kick"
VLOG="logs/vast"
CHAIN_LOG="$VLOG/chain.log"
mkdir -p "$VLOG"

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; fi

# Stage 2 (今動いている run)。nw_s1 の model_2299 から --max_iterations 3000 → 最後は model_5298.pt
S2_NAME="${S2_NAME:-nw_s2}"
S2_FINAL="${S2_FINAL:-5298}"
S2_STOP_FILE="$VLOG/chain_s2_stop_at"

# ---- すべての段に付ける上書き (scripts/rsl_rl/train_bhuman_walk_kick.sh の NEW_WALK_OVERRIDES / HANDOFF_OVERRIDES と同じ) ----
COMMON=(
  env.rewards.feet_phase.params.stance_ratio=0.6
  env.rewards.no_fly.weight=4.0
  env.rewards.joint_power_l2.weight=-1.5e-5
  env.rewards.flat_orientation_l2.weight=-40.0
  env.rewards.ang_vel_xy_l2.weight=-0.65
  env.rewards.base_ang_acc_l2.weight=-2.0e-5
  "env.events.physics_material.params.static_friction_range=[0.6,1.0]"
  "env.events.physics_material.params.dynamic_friction_range=[0.6,1.0]"
  "env.events.push_robot.interval_range_s=[0.5,8.0]"
  "env.events.push_robot.params.velocity_range.x=[-0.7,0.7]"
  "env.events.push_robot.params.velocity_range.y=[-0.7,0.7]"
  "env.events.push_robot.params.velocity_range.roll=[-0.06,0.06]"
  "env.events.push_robot.params.velocity_range.pitch=[-0.06,0.06]"
  env.commands.kick.handoff_prob=0.5
)

# ---- 段ごとの値 (名前|task|N|上書き)。旧 run → 新 run ----
N3="--reset_noise_std 0.13 agent.algorithm.std_range=[0.13,0.17]"
N6="--reset_noise_std 0.12 agent.algorithm.std_range=[0.12,0.13] agent.algorithm.desired_kl=0.003"
S7B="env.commands.kick.kick_direction_sigma_range=0.2 env.commands.kick.random_ball_reset=[0.01,0.02,0.02] \
env.commands.kick.reset_ball_request=0.005 env.commands.kick.add_ball_vel=0.0001 env.commands.kick.add_ball_vel_in_direction=0.0001 \
env.commands.kick.heavy_ball=0.0001 env.commands.kick.kick_pose_increase_value=0.2"
S9="$S7B env.commands.kick.weak_rel_prob=0.25 env.commands.base_velocity.min_side_off=0.15 \
env.commands.base_velocity.near_yaw_off_deg=20.0 env.commands.kick.walk_side_hysteresis=0.05"
STAGES=(
  # s3a_sig05: s2b 3400 → 3600
  "nw_s3a_sig05|Stage3|201|$N3 agent.algorithm.actor_freeze_iterations=10 env.commands.kick.target_sole_yaw_sigma=0.5"
  # s3b_sig025: 3600 → 3700
  "nw_s3b_sig025|Stage3|101|$N3 agent.algorithm.actor_freeze_iterations=0 env.commands.kick.target_sole_yaw_sigma=0.25"
  # s3b_mirrorfull: 3700 → 4000 (旧はここから mirror loss が履歴全体の版。今の source は最初から全体の版なので、値は上と同じ)
  "nw_s3b_mirrorfull|Stage3|301|$N3 agent.algorithm.actor_freeze_iterations=0 env.commands.kick.target_sole_yaw_sigma=0.25"
  # s3c_sig01: 4000 → 4300 (target_sole_yaw_sigma は Stage 3 の既定 0.1)
  "nw_s3c_sig01|Stage3|301|$N3 agent.algorithm.actor_freeze_iterations=0"
  # s4: 4300 → 4700
  "nw_s4|Stage4|401|$N3 agent.algorithm.actor_freeze_iterations=10"
  # s5: 4700 → 7699 (--max_iterations 3000 を最後まで)
  "nw_s5|Stage5|3000|$N3 agent.algorithm.actor_freeze_iterations=0"
  # s6: 7699 → 8200。旧 run の時の Stage 6 の source は ball_kick_velocity の sigma_scale 0.5 (今の source は 1.0) なので上書きで合わせる
  "nw_s6|Stage6|502|$N6 agent.algorithm.actor_freeze_iterations=10 env.rewards.ball_kick_velocity.params.sigma_scale=0.5"
  # s6b_sig02: 8200 → 8500 (sigma_scale は 1.0 = 今の Stage 6 の既定)
  "nw_s6b_sig02|Stage6|301|$N6 agent.algorithm.actor_freeze_iterations=0 env.commands.kick.kick_direction_sigma_range=0.2"
  # s6c_vel8_strong4: 8500 → 9200 (旧 run は 9700 まで回したが、採用した s7a は model_9200 から)
  "nw_s6c_vel8_strong4|Stage6|701|$N6 agent.algorithm.actor_freeze_iterations=10 env.commands.kick.kick_direction_sigma_range=0.2 env.rewards.ball_kick_velocity.weight=8.0 env.rewards.ball_kick_velocity_strong.weight=4.0"
  # s7a: 9200 → 9300 (確率は最終値の 10%)
  "nw_s7a|Stage7|101|$N6 agent.algorithm.actor_freeze_iterations=10 env.commands.kick.kick_direction_sigma_range=0.2 env.commands.kick.random_ball_reset=[0.001,0.002,0.002] env.commands.kick.reset_ball_request=0.0005 env.commands.kick.add_ball_vel=0.00001 env.commands.kick.add_ball_vel_in_direction=0.00001 env.commands.kick.heavy_ball=0.00001 env.commands.kick.kick_pose_increase_value=0.025"
  # s7b: 9300 → 9400 (Stage 7 の既定 + sigma_range 0.2)
  "nw_s7b|Stage7|101|$N6 agent.algorithm.actor_freeze_iterations=10 env.commands.kick.kick_direction_sigma_range=0.2"
  # s9a: 9400 → 10100 (旧 run は 10200 まで、採用した s9b は model_10100 から)
  "nw_s9a|Stage7|701|$N6 agent.algorithm.actor_freeze_iterations=10 $S9"
  # s9b: 10100 → 10300
  "nw_s9b|Stage7|201|$N6 agent.algorithm.actor_freeze_iterations=10 $S9 env.rewards.kick_foot_speed.weight=4.0"
  # s9c: 10300 → 11400
  "nw_s9c|Stage7|1101|$N6 agent.algorithm.actor_freeze_iterations=10 $S9 env.rewards.kick_foot_speed.weight=20.0"
)

log() {
  local msg="[$(date -u '+%F %T UTC') / $(TZ=Asia/Tokyo date '+%H:%M JST')] $*"
  if (( DRY )); then echo "$msg"; else echo "$msg" >> "$CHAIN_LOG"; fi
}
die() { log "STOP: $*"; log "鎖を止めました。上の理由を直してから、続きの段は手で起動してください。"; exit 1; }

# 名前が *_<name> で終わる run のうち一番新しいもの (無ければ空)
newest_run() { ls -d "$EXP_DIR"/*_"$1" 2>/dev/null | sort | tail -1 | xargs -r basename; }

pid_alive() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null; }

train_running() { pgrep -f "scripts/rsl_rl/train.py" >/dev/null 2>&1; }

wait_no_train() {
  local t=0
  while train_running; do
    if (( t >= 600 )); then
      die "train.py がまだ動いています (10 分待った): $(pgrep -af scripts/rsl_rl/train.py | head -3 | cut -c1-200)"
    fi
    sleep 10; t=$((t + 10))
  done
}

# ---- Stage 2 の終わりを待つ ----
s2_run="$(newest_run "$S2_NAME")"
if (( DRY )); then
  echo "# dry-run: Stage 2 run = ${s2_run:-<*_$S2_NAME が無い>}、最後の checkpoint = model_$S2_FINAL.pt"
else
  echo $$ > "$VLOG/chain.pid"
  log "chain start (pid $$). Stage 2 = ${s2_run:-?} (pid $(cat "$VLOG/$S2_NAME.pid" 2>/dev/null)), expect model_$S2_FINAL.pt"
  [[ -n "$s2_run" ]] || die "$EXP_DIR/*_$S2_NAME がありません"
  s2_pid="$(cat "$VLOG/$S2_NAME.pid" 2>/dev/null || true)"
  while pid_alive "$s2_pid"; do
    if [[ -f "$S2_STOP_FILE" ]]; then
      stop_at="$(tr -dc '0-9' < "$S2_STOP_FILE")"
      f="$EXP_DIR/$s2_run/model_$stop_at.pt"
      # 書き終わってから (更新から 30 秒以上) 止める
      if [[ -n "$stop_at" && -f "$f" ]] && (( $(date +%s) - $(stat -c %Y "$f") >= 30 )); then
        log "chain_s2_stop_at=$stop_at: $f ができたので $S2_NAME を止めます"
        pkill -TERM -f -- "--run_name $S2_NAME " ; kill -TERM "$s2_pid" 2>/dev/null
        for _ in $(seq 1 24); do pgrep -f -- "--run_name $S2_NAME " >/dev/null || break; sleep 5; done
        pkill -KILL -f -- "--run_name $S2_NAME " 2>/dev/null
        S2_FINAL="$stop_at"
        break
      fi
    fi
    sleep 20
  done
  log "$S2_NAME は終わりました"
fi
prev_run="${s2_run:-<*_$S2_NAME>}"
prev_ckpt="$S2_FINAL"
[[ -f "$EXP_DIR/$prev_run/model_$prev_ckpt.pt" ]] || (( DRY )) || \
  die "$EXP_DIR/$prev_run/model_$prev_ckpt.pt がありません (Stage 2 が途中で落ちた?)。tail: $(tail -5 "$VLOG/$S2_NAME.log" | tr '\n' ' ' | cut -c1-600)"

# ---- Stage 3〜9c ----
total=0
for spec in "${STAGES[@]}"; do
  IFS='|' read -r name stage n extra <<< "$spec"
  task="Isaac-K1-BHuman-Walk-Kick-$stage-v0"
  final=$((prev_ckpt + n - 1))
  total=$((total + n))
  # $extra は空白で分ける ([0.13,0.17] などをファイル名として展開しないよう、その間だけ glob を切る)
  set -f
  # shellcheck disable=SC2206
  args=(--task "$task" --num_envs 1024 --max_iterations "$n" --resume --load_run "$prev_run" --checkpoint "model_$prev_ckpt.pt" $extra "${COMMON[@]}")
  set +f
  if (( DRY )); then
    echo
    echo "# $name: $prev_run/model_$prev_ckpt.pt → model_$final.pt ($n iterations)"
    echo "./scripts/vast_train.sh $name ${args[*]}"
    prev_run="<newest *_$name>"; prev_ckpt=$final
    continue
  fi

  wait_no_train
  [[ -f "$EXP_DIR/$prev_run/model_$prev_ckpt.pt" ]] || die "$EXP_DIR/$prev_run/model_$prev_ckpt.pt がありません"
  log "start $name: $task, from $prev_run/model_$prev_ckpt.pt, $n iterations → model_$final.pt"
  log "  cmd: ./scripts/vast_train.sh $name ${args[*]}"
  t0=$(date +%s)
  ./scripts/vast_train.sh "$name" "${args[@]}" >> "$CHAIN_LOG" 2>&1 || die "vast_train.sh が失敗しました ($name)"
  pid="$(cat "$VLOG/$name.pid" 2>/dev/null || true)"
  [[ -n "$pid" ]] || die "$VLOG/$name.pid がありません"
  while pid_alive "$pid"; do sleep 20; done
  run="$(newest_run "$name")"
  if [[ -z "$run" ]] || (( $(stat -c %Y "$EXP_DIR/$run") < t0 - 5 )); then
    die "$name の run ディレクトリができていません。tail: $(tail -8 "$VLOG/$name.log" | tr '\n' ' ' | cut -c1-800)"
  fi
  if [[ ! -f "$EXP_DIR/$run/model_$final.pt" ]]; then
    die "$EXP_DIR/$run/model_$final.pt がありません (最後: $(ls "$EXP_DIR/$run" | grep -E '^model_' | sort -t_ -k2 -n | tail -1))。tail: $(tail -8 "$VLOG/$name.log" | tr '\n' ' ' | cut -c1-800)"
  fi
  log "done $name: $run/model_$final.pt ($(( ($(date +%s) - t0) / 60 )) 分)"
  prev_run="$run"; prev_ckpt=$final
done

if (( DRY )); then
  echo
  echo "# 合計 $total iterations (Stage 3〜9c)。最後: $prev_run/model_$prev_ckpt.pt"
else
  log "chain finished: $prev_run/model_$prev_ckpt.pt (Stage 3〜9c 合計 $total iterations)"
  rm -f "$VLOG/chain.pid"
fi
