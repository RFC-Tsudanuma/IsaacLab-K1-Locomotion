# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""pivot の「固定コスト」を測る台本評価 (2026-11-05)。

経路追従セッションの指摘: 上位方策が pivot を使わないのは回転そのものではなく前後の
固定コスト (歩行を止める → turn へ → 最小滞在 → walk へ戻ると履歴ゼロ埋めで静止から再加速)
のせい。本スクリプトはゲーティング方策だけを台本で動かし、その固定コストを分解して測る。

台本 (env ごとに 1 回):
  1. warmup: walk モードで ``--entry-speed`` [m/s] を ``--warmup`` 秒出す (0 なら立位スタート)
  2. turn:   mode=turn にして目標角 = 現在 yaw ± ``--turn-deg``。残り角が ``--exit-deg`` 以内
             (または 2.5 s) で抜ける
  3. resume: mode=walk にして前進 1.0 m/s を指令し、実速度が ``--resume-speed`` に達するまでの
             時間を測る。``--yaw-assist`` でデプロイ側と同じ残り角のヨー上乗せを再現する

出力: 回転時間 / 再加速時間 / 合計オーバーヘッド / 転倒率 (turn 中・resume 中)。

例 (gate_v4 を現行契約と提案契約で比較):
  uv run eval_pivot_overhead.py --checkpoint <abs>/model_5897.pt --entry-speed 0 --exit-deg 5
  uv run eval_pivot_overhead.py --checkpoint <abs>/model_5897.pt --entry-speed 0 --exit-deg 15 \
      --no-zero-fill --yaw-assist
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="pivot fixed-cost (overhead) evaluation.")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Gating-Play")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--warmup", type=float, default=1.5, help="turn へ入る前に歩かせる秒数")
parser.add_argument("--entry-speed", type=float, default=0.0, help="turn 突入時の前進指令 [m/s]")
parser.add_argument("--turn-deg", type=float, default=180.0, help="pivot の目標相対角 [deg]")
parser.add_argument("--exit-deg", type=float, default=5.0, help="この残り角 [deg] で walk へ戻す")
parser.add_argument("--resume-speed", type=float, default=0.5, help="再加速の判定速度 [m/s]")
parser.add_argument("--resume-cmd", type=float, default=1.0, help="resume 時の前進指令 [m/s]")
parser.add_argument("--no-zero-fill", action="store_true",
                    help="切替時の履歴コマンド枠ゼロ埋めを無効にする")
parser.add_argument("--yaw-assist", action="store_true",
                    help="resume 中、残り角を ωz = clamp(4*err, ±3) として上乗せする (デプロイと同じ)")
parser.add_argument("--observe", type=float, default=3.0, help="resume を待つ上限秒数")
parser.add_argument("--force-w", type=float, default=None,
                    help="指定すると turn 中のゲート重みをこの値に固定する (1.0 = 旋回 expert 単体)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.math import wrap_to_pi  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_k1_locomotion.tasks  # noqa: F401, E402

PHASE_WARMUP, PHASE_TURN, PHASE_RESUME, PHASE_DONE = 0, 1, 2, 3


def _median(x: torch.Tensor) -> float:
    return float(x.median()) if x.numel() else float("nan")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    # 台本が完走するまでリセットされないように
    env_cfg.episode_length_s = args_cli.warmup + 6.0 + args_cli.observe
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.initial_turn_prob = 0.0            # 全 env を walk から開始
    cmd_cfg.walk_duration_range = (1.0e6, 1.0e6)   # 自律遷移を止める
    cmd_cfg.turn_hold_range = (1.0e6, 1.0e6)
    cmd_cfg.turn_interrupt_prob = 0.0
    cmd_cfg.turn_max_duration = 1.0e6
    cmd_cfg.turn_early_exit_rad = 0.0          # 抜けるのは台本側
    cmd_cfg.zero_command_history_on_switch = not args_cli.no_zero_fill
    cmd_cfg.walk_cfg.heading_command = False   # ωz を台本で直接指令する
    cmd_cfg.walk_cfg.rel_standing_envs = 0.0

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    policy = runner.alg.act_inference

    raw = env.unwrapped
    dev = raw.device
    n = raw.num_envs
    dt = raw.step_dt
    cmd = raw.command_manager.get_term("base_velocity")
    robot = raw.scene["robot"]

    # 計測器の自己検証: 実際に term が持っている cfg と、ゼロ埋めが何回走ったか
    print(f"[check] zero_command_history_on_switch = {cmd.cfg.zero_command_history_on_switch}", flush=True)
    zf_calls = [0]
    _orig_zero = cmd._zero_command_history

    def _counting_zero(ids):
        zf_calls[0] += 1
        return _orig_zero(ids)

    cmd._zero_command_history = _counting_zero
    gate = runner.alg.policy
    _forced = {"on": torch.zeros(n, dtype=torch.bool, device=dev)}
    if args_cli.force_w is not None:
        _orig_tw = gate.turn_weight

        def _forced_tw(z):
            w = _orig_tw(z)
            return torch.where(_forced["on"].unsqueeze(1), torch.full_like(w, args_cli.force_w), w)

        gate.turn_weight = _forced_tw
    w_turn_sum = torch.zeros(n, device=dev)
    w_turn_cnt = torch.zeros(n, device=dev)

    phase = torch.zeros(n, dtype=torch.long, device=dev)
    timer = torch.zeros(n, device=dev)
    sign = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    turn_time = torch.full((n,), float("nan"), device=dev)
    resume_time = torch.full((n,), float("nan"), device=dev)
    fell_turn = torch.zeros(n, dtype=torch.bool, device=dev)
    fell_resume = torch.zeros(n, dtype=torch.bool, device=dev)

    exit_rad = math.radians(args_cli.exit_deg)
    target_rad = math.radians(args_cli.turn_deg)
    obs = env.get_observations()

    def set_walk_cmd(ids, vx, wz=None):
        cmd.walk.vel_command_b[ids, 0] = vx
        cmd.walk.vel_command_b[ids, 1] = 0.0
        cmd.walk.vel_command_b[ids, 2] = 0.0 if wz is None else wz
        cmd.walk.time_left[ids] = 1.0e6

    set_walk_cmd(slice(None), args_cli.entry_speed)

    steps = int((env_cfg.episode_length_s - 0.2) / dt)
    with torch.inference_mode():
        for _ in range(steps):
            obs, _, _, _ = env.step(policy(obs))
            terminated = raw.termination_manager.terminated
            timer += dt
            err = cmd.heading_error.abs()
            speed = robot.data.root_lin_vel_b[:, :2].norm(dim=-1)
            w = gate.turn_weight(gate.act_inference(obs))[:, 0]
            in_turn = phase == PHASE_TURN
            _forced["on"] = in_turn
            w_turn_sum += torch.where(in_turn, w, torch.zeros_like(w))
            w_turn_cnt += in_turn.float()

            # 転倒の記録 (その env はそこで打ち切る)
            fell = terminated & (phase != PHASE_DONE)
            fell_turn |= fell & (phase == PHASE_TURN)
            fell_resume |= fell & (phase == PHASE_RESUME)
            phase = torch.where(fell, torch.full_like(phase, PHASE_DONE), phase)

            # warmup → turn
            to_turn = (phase == PHASE_WARMUP) & (timer >= args_cli.warmup)
            if to_turn.any():
                ids = to_turn.nonzero(as_tuple=False).flatten()
                cmd._enter_turn(ids, count_switch=True)
                cmd.turn.heading_target[ids] = wrap_to_pi(
                    robot.data.heading_w[ids] + sign[ids] * target_rad)
                if cmd.cfg.zero_command_history_on_switch:
                    cmd._zero_command_history(ids)   # compute() の switched 経路を通らないので明示的に
                phase[ids] = PHASE_TURN
                timer[ids] = 0.0

            # turn → resume (残り角が閾値以内、または 2.5 s で打ち切り)
            to_walk = (phase == PHASE_TURN) & ((err <= exit_rad) | (timer >= 2.5))
            if to_walk.any():
                ids = to_walk.nonzero(as_tuple=False).flatten()
                turn_time[ids] = timer[ids]
                cmd._enter_walk(ids, count_switch=True, resample_walk=False)
                if cmd.cfg.zero_command_history_on_switch:
                    cmd._zero_command_history(ids)
                set_walk_cmd(ids, args_cli.resume_cmd)
                phase[ids] = PHASE_RESUME
                timer[ids] = 0.0

            # resume 中のヨー上乗せ (デプロイと同じ式) と再加速の判定
            active = phase == PHASE_RESUME
            if active.any():
                if args_cli.yaw_assist:
                    ids = active.nonzero(as_tuple=False).flatten()
                    assist = (4.0 * cmd.heading_error[ids]).clamp(-3.0, 3.0)
                    keep = cmd.heading_error[ids].abs() > math.radians(5.0)
                    set_walk_cmd(ids, args_cli.resume_cmd, torch.where(keep, assist, torch.zeros_like(assist)))
                done = active & (speed >= args_cli.resume_speed)
                if done.any():
                    ids = done.nonzero(as_tuple=False).flatten()
                    resume_time[ids] = timer[ids]
                    phase[ids] = PHASE_DONE
                lost = active & (timer >= args_cli.observe)
                phase = torch.where(lost, torch.full_like(phase, PHASE_DONE), phase)

            if bool((phase == PHASE_DONE).all()):
                break

    ok_turn = turn_time.isfinite()
    ok_resume = resume_time.isfinite()
    total = turn_time + resume_time
    print("", flush=True)
    print(f"=== pivot overhead  n={n}  entry={args_cli.entry_speed:.1f} m/s  "
          f"turn={args_cli.turn_deg:.0f} deg  exit={args_cli.exit_deg:.0f} deg  "
          f"zero_fill={not args_cli.no_zero_fill}  yaw_assist={args_cli.yaw_assist}", flush=True)
    print(f"  回転時間      median {_median(turn_time[ok_turn]):.3f} s  (n={int(ok_turn.sum())})", flush=True)
    print(f"  再加速時間    median {_median(resume_time[ok_resume]):.3f} s  "
          f"(>= {args_cli.resume_speed} m/s に到達: {float(ok_resume.float().mean()):.3f})", flush=True)
    print(f"  合計          median {_median(total[ok_turn & ok_resume]):.3f} s", flush=True)
    print(f"  転倒 (turn中)   {float(fell_turn.float().mean()):.4f}", flush=True)
    print(f"  転倒 (resume中) {float(fell_resume.float().mean()):.4f}", flush=True)
    w_mean = w_turn_sum / w_turn_cnt.clamp(min=1.0)
    print(f"  turn 中の w   median {_median(w_mean[w_turn_cnt > 0]):.3f}  "
          f"(1.0 = 旋回 expert 単体、小さいほど歩行 expert が混ざって回転が鈍る)", flush=True)
    print(f"  ゼロ埋め呼出  {zf_calls[0]} 回", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
