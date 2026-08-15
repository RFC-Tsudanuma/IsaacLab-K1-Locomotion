#!/usr/bin/env python3
"""階層版 GK Stage2 ランの状態を 1 画面に要約する (gk_status.py の階層版)。

gk_status.py は直接制御版専用 (foot_clearance タグ前提・LOG_ROOT 固定) で、
階層版のランに向けるとタグ欠落で落ちるため別スクリプトにした。
ホストの python3 で動く (torch / Isaac Sim 不要、tensorboard パッケージのみ)。

使い方 (リポジトリ直下):
    python3 scripts/rsl_rl/gk_hier_status.py                 # 最新の run
    python3 scripts/rsl_rl/gk_hier_status.py <run ディレクトリ>

判定基準 (2026-08-14 の会話で決めたもの):
  * aim_y_range が 0.4 のまま 1 時間 (実測 ~1400 iter) → カリキュラム停滞。止めて切り分け
  * mean_episode_length が 100 step 未満で推移 → 学習が壊れている (COM 事故の教訓)
  * hard_ball started は aim_stage 最終段まで 0 が正常 (序盤に 1 なら誤発火 = バグ)
  * 転倒 (base_contact + base_height) は Stage1 実績 0.2% 台。1% 超なら悪化
  * 報酬 < -1000 のスパイクが 1 件でもあれば ckpt 選定でその前後を避ける
"""

import glob
import os
import statistics
import sys

from tensorboard.backend.event_processing import event_accumulator

LOG_ROOT = "logs/rsl_rl/k1_gk_hier_stage2"


def load(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "*.tfevents*")))
    if not files:
        sys.exit(f"tfevents が見つかりません: {run_dir}")
    ea = event_accumulator.EventAccumulator(files[-1], size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def tail(ea, key, n=20):
    if key not in ea.Tags()["scalars"]:
        return None
    return statistics.fmean(v.value for v in ea.Scalars(key)[-n:])


def fmt(v, spec=".3f"):
    return format(v, spec) if v is not None else "  (無し)"


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else max(
        glob.glob(os.path.join(LOG_ROOT, "*/")), key=os.path.getmtime
    )
    ea = load(run)
    reward = ea.Scalars("Train/mean_reward")
    first, last = reward[0].step, reward[-1].step
    wall_min = (reward[-1].wall_time - reward[0].wall_time) / 60.0
    print(f"run  : {run}")
    print(f"iter : {first} -> {last}  ({last - first} 回, {wall_min:.0f} 分)")

    # --- 序盤の健全性 (COM 事故の教訓: まずここを見る) ------------------
    ep_len = tail(ea, "Train/mean_episode_length")
    print(f"\n[健全性] mean_episode_length = {fmt(ep_len, '.1f')}")
    if ep_len is not None and ep_len < 100:
        print("  → 100 step 未満。学習が壊れている可能性。即止めて切り分け")

    # --- カリキュラム ---------------------------------------------------
    aim = tail(ea, "Curriculum/difficulty/aim_y_range", n=1)
    hi = tail(ea, "Curriculum/difficulty/ball_speed_hi", n=1)
    ema = tail(ea, "Curriculum/difficulty/success_ema")
    cd = tail(ea, "Curriculum/difficulty/cooldown_left", n=1)
    print(f"\n[カリキュラム] aim_y_range = {fmt(aim, '.2f')} (0.4→0.6→0.8→1.1)"
          f" / ball_speed_hi = {fmt(hi, '.2f')} m/s (cap 5.0)")
    print(f"               success_ema = {fmt(ema)} (昇格 0.85 / 降格 0.55)"
          f" / cooldown_left = {fmt(cd, '.0f')}")
    if aim is not None and aim <= 0.4 and (last - first) > 1400:
        print("  → 1 時間相当を超えて最易段のまま。前回の停滞と同じ。止めて切り分け")

    hb_started = tail(ea, "Curriculum/hard_ball/started", n=1)
    hb_prob = tail(ea, "Curriculum/hard_ball/hard_ball_prob", n=1)
    print(f"[到達不能球] started = {fmt(hb_started, '.0f')} / prob = {fmt(hb_prob, '.2f')}")
    if hb_started and aim is not None and aim < 1.05:
        print("  → aim が最終段でないのに発火している。トリガのバグを疑う")

    # --- セーブ実績 (Episode_Reward はエピソード長 25s で割った値) --------
    ep_s = 25.0
    touch = tail(ea, "Episode_Reward/save_touch_bonus")
    conceded = tail(ea, "Episode_Termination/goal_conceded")
    if touch is not None and conceded is not None:
        saves = touch * ep_s / 2.0     # weight 100 × dt 0.02 = 2.0 / セーブ
        rate = saves / max(saves + conceded, 1e-6)
        print(f"\n[セーブ] {saves:.2f} 本/ep / 失点 {conceded:.2f} 本/ep"
              f" → 1球あたり {rate:.1%}")

    # --- 転倒・位置維持 ---------------------------------------------------
    bc = tail(ea, "Episode_Termination/base_contact") or 0.0
    bh = tail(ea, "Episode_Termination/base_height") or 0.0
    oob = tail(ea, "Episode_Termination/out_of_bounds") or 0.0
    print(f"\n[転倒] base_contact {bc:.4f} + base_height {bh:.4f} = {bc + bh:.4f}"
          f"   (Stage1 実績 0.002)   out_of_bounds {oob:.4f}")
    if bc + bh > 0.01:
        print("  → 1% 超。到達不能球かゴースト速度でバランスを崩している可能性")
    print(f"[位置維持] stay_on_goal_line_fine = "
          f"{fmt(tail(ea, 'Episode_Reward/stay_on_goal_line_fine'))}"
          f" / hold_at_target = {fmt(tail(ea, 'Episode_Reward/hold_at_target'))}")

    # --- 安定性 -----------------------------------------------------------
    spikes = sum(1 for v in reward if v.value < -1000)
    vf = tail(ea, "Loss/value_function")
    noise = tail(ea, "Policy/mean_noise_std")
    print(f"\n[安定性] 報酬スパイク(<-1000) {spikes} 件 / value loss {fmt(vf)}"
          f" / noise_std {fmt(noise)}")
    if spikes:
        print("  → ckpt を選ぶときはスパイクの前後を避けること")


if __name__ == "__main__":
    main()
