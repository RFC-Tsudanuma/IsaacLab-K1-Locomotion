# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ドリブルのポリシーの転倒が、どんな状況で起きているかを調べる (推論だけ、学習の更新はしない)。

カリキュラムのレベルを固定し、学習時と同じ環境 (観測ノイズ・押しあり) で --steps ステップ動かす。
転倒 (base_height で終了) ごとに、その瞬間の状況を記録する:

* touch:    直前 --touch_window ステップ以内に足がボールに触れていた
* corner:   途中のウェイポイント (曲がり角) から --corner_radius m 以内、または次のウェイポイントへ進んでから
            --corner_window ステップ以内
* resample: エピソードの途中で経路が引き直されてから --resample_window ステップ以内

あわせて、それぞれの状況が全ステップに占める割合 (曝露) を数え、「その状況にいる 1 万ステップあたりの転倒数」
で比べる (状況そのものが多いだけなのか、その状況で転びやすいのかを区別するため)。

    python scripts/rsl_rl/eval_dribble_falls.py --levels 1 3 4 8
"""

import argparse
import glob
import json
import os
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, 9)))
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--stochastic", action="store_true", help="学習時と同じく行動ノイズ (std) を入れてサンプリングする")
parser.add_argument("--touch_window", type=int, default=25)
parser.add_argument("--corner_radius", type=float, default=1.0)
parser.add_argument("--corner_window", type=int, default=50)
parser.add_argument("--resample_window", type=int, default=50)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output", default=None)
parser.add_argument("--polyline_prob", type=float, default=0.0,
                    help="ランダムな折れ線にするエピソードの割合 (学習では 0.2。既定 0 = そのレベルのコースだけ)")
parser.add_argument("--no_replan", action="store_true", help="エピソードの途中の経路の引き直しを切る")
parser.add_argument("--ball_delay_ms", type=float, default=None,
                    help="ボール検出の遅れを固定する [ms] (既定: 学習と同じランダム化。0 で遅れなし)")
parser.add_argument("--ball_delay_mode", choices=["current", "capture"], default=None,
                    help="ボールの遅れの作り方 (既定: cfg のまま)。capture = 実機と同じ「撮影時刻のロボット座標で見た撮影時刻のボール」")
parser.add_argument("--loc_delay_ms", type=float, default=None,
                    help="自己位置の遅れを固定する [ms] (既定: 学習と同じランダム化。0 で遅れなし)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import isaaclab_k1_locomotion.tasks  # noqa: E402, F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_transition_utils  # noqa: E402

TASK = "Isaac-Velocity-Flat-K1-Dribble-Target-v0"


def _latest_checkpoint() -> str:
    runs = sorted(glob.glob(os.path.join("logs", "rsl_rl", "k1_dribble_target", "20*")))
    ckpts = glob.glob(os.path.join(runs[-1], "model_*.pt"))
    return max(ckpts, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))


def main():
    ckpt = os.path.abspath(args_cli.checkpoint or _latest_checkpoint())
    env_cfg = parse_env_cfg(TASK, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.commands.dribble_target.debug_vis = False
    stages = env_cfg.curriculum.dribble_target_range.params["stages"]
    env_cfg.curriculum.dribble_target_range.params["error_threshold"] = [-1.0] * len(stages)  # 進級させない
    env_cfg.curriculum.dribble_target_range.params["prev_level_mix_prob"] = 0.0  # 前のレベルを混ぜない
    env_cfg.commands.dribble_target.random_polyline_prob = args_cli.polyline_prob
    if args_cli.no_replan:
        env_cfg.events.replan_dribble_path = None
    if args_cli.ball_delay_mode is not None:
        for _name in ("ball_pos_rel", "ball_vel", "foot_left_ball_pos_rel", "foot_right_ball_pos_rel"):
            getattr(env_cfg.observations.policy, _name).params["ball_delay_mode"] = args_cli.ball_delay_mode
    delay_ev = getattr(env_cfg.events, "randomize_observation_delays", None)
    if delay_ev is not None:
        if args_cli.ball_delay_ms is not None:
            d = args_cli.ball_delay_ms / 1000.0
            delay_ev.params["ball_delay_range_s"] = (d, d)
        if args_cli.loc_delay_ms is not None:
            d = args_cli.loc_delay_ms / 1000.0
            delay_ev.params["localization_delay_range_s"] = (d, d)
    # 報酬のフェードイン・gait_phase の閾値は学習の最後の状態 (フェード済み) にする
    for term in vars(env_cfg.curriculum).values():
        p = getattr(term, "params", None)
        if isinstance(p, dict) and "end_step" in p:
            p["start_step"], p["end_step"] = -1, 0

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(
        wrapped, stage_transition_utils.filter_unsupported_alg_args(agent_cfg.to_dict()), log_dir=None, device="cuda:0"
    )
    stage_transition_utils.load_for_inference(runner, ckpt)
    policy_nn = runner.alg.policy
    policy_nn.eval()
    act = policy_nn.act if args_cli.stochastic else policy_nn.act_inference

    cm = u.curriculum_manager
    term = cm._term_cfgs[cm.active_terms.index("dribble_target_range")].func
    cmd = u.command_manager.get_term("dribble_target")
    robot, ball = u.scene["robot"], u.scene["soccer_ball"]
    s_left, s_right = u.scene.sensors["contact_balls_left"], u.scene.sensors["contact_balls_right"]
    n, dev = u.num_envs, u.device
    results = {}

    for level in args_cli.levels:
        term.load_state_dict({"level": level - 1, "failure_ema": 1.0})
        obs, _ = wrapped.reset()
        obs = wrapped.get_observations()
        big = torch.full((n,), 10**6, device=dev)
        last_touch, last_adv, last_resample = big.clone(), big.clone(), big.clone()
        prev_wp = cmd.waypoint_index.clone()
        prev_goal = cmd.waypoints_w[torch.arange(n, device=dev), cmd.num_waypoints - 1, :2].clone()
        c = {k: 0 for k in ("episodes", "success", "fall", "obstacle", "contact", "ball_too_far", "time_out")}
        falls = {"touch": 0, "corner": 0, "resample": 0, "none": 0}
        combos: dict[str, int] = {}
        expo = {"touch": 0, "corner": 0, "resample": 0, "none": 0, "all": 0}
        fall_ctx = {"speed": [], "turn_deg": [], "ball_dist": []}
        reset_prev = torch.zeros(n, dtype=torch.bool, device=dev)
        speed_sum = 0.0
        pass0 = dict(cmd.pass_count)
        ep_len_sum, ep_len_n = 0.0, 0
        with torch.no_grad():
            for step in range(args_cli.steps):
                # --- 状況のフラグは「ステップを進める前」の状態で計算する。IsaacLab は終了した env を
                # ステップの中で自動的にリセットするので、ステップ後に読むとリセット後の状態になってしまう。
                # 終了した env については、これが終了の 1 ステップ (20ms) 前の状態になる。
                touch_now = (torch.norm(s_left.data.force_matrix_w[:, 0, 0], dim=1) > 0.1) | (
                    torch.norm(s_right.data.force_matrix_w[:, 0, 0], dim=1) > 0.1
                )
                touch_now &= ~reset_prev
                last_touch = torch.where(touch_now, torch.zeros_like(last_touch), last_touch + 1)
                adv = (cmd.waypoint_index > prev_wp) & ~reset_prev
                last_adv = torch.where(adv, torch.zeros_like(last_adv), last_adv + 1)
                # 経路の引き直し (replan_dribble_path イベント) からの経過ステップ
                last_resample = (int(u.common_step_counter) - cmd.last_replan_step).clamp(min=0)
                prev_wp = cmd.waypoint_index.clone()
                rxy = robot.data.root_pos_w[:, :2]
                idx = torch.arange(cmd.waypoints_w.shape[1], device=dev).unsqueeze(0)
                inter = idx < (cmd.num_waypoints - 1).unsqueeze(1)
                dwp = torch.norm(cmd.waypoints_w[:, :, :2] - rxy.unsqueeze(1), dim=2)
                near_corner = ((dwp < args_cli.corner_radius) & inter).any(dim=1) | (last_adv <= args_cli.corner_window)
                f_touch = last_touch <= args_cli.touch_window
                f_res = last_resample <= args_cli.resample_window
                v_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)[:, :2]
                tgt3 = torch.cat([cmd.pos_command_w[:, :2] - rxy, torch.zeros(n, 1, device=dev)], 1)
                tgt_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), tgt3)
                ctx_speed = torch.norm(v_b, dim=1)
                ctx_turn = torch.rad2deg(torch.atan2(tgt_b[:, 1], tgt_b[:, 0])).abs()
                ctx_ball = torch.norm(ball.data.root_pos_w[:, :2] - rxy, dim=1)
                ctx_tick = u.episode_length_buf.clone()
                expo["all"] += n
                speed_sum += float(ctx_speed.sum())
                expo["touch"] += int(f_touch.sum())
                expo["corner"] += int(near_corner.sum())
                expo["resample"] += int(f_res.sum())
                expo["none"] += int((~f_touch & ~near_corner & ~f_res).sum())

                obs, _, dones, _ = wrapped.step(act(obs))
                dones = dones.bool()
                tm = u.termination_manager
                if dones.any():
                    c["episodes"] += int(dones.sum())
                    for key, name in (("success", "ball_reached_target"), ("fall", "base_height"),
                                      ("obstacle", "obstacle_collision"), ("contact", "base_contact"),
                                      ("ball_too_far", "ball_too_far"), ("time_out", "time_out")):
                        if name in tm.active_terms:  # obstacle_collision は 2026-10-22 に廃止
                            c[key] += int((tm.get_term(name) & dones).sum())
                    fell = tm.get_term("base_height") & dones
                    succ_ids = tm.get_term("ball_reached_target") & dones
                    ep_len_sum += float(ctx_tick[succ_ids].sum()) * u.step_dt
                    ep_len_n += int(succ_ids.sum())
                    for i in torch.nonzero(fell).flatten().tolist():
                        flags = [k for k, f in (("touch", f_touch), ("corner", near_corner), ("resample", f_res)) if bool(f[i])]
                        for k in flags:
                            falls[k] += 1
                        if not flags:
                            falls["none"] += 1
                        key = "+".join(flags) or "none"
                        combos[key] = combos.get(key, 0) + 1
                        fall_ctx["speed"].append(float(ctx_speed[i]))
                        fall_ctx["turn_deg"].append(float(ctx_turn[i]))
                        fall_ctx["ball_dist"].append(float(ctx_ball[i]))
                        fall_ctx.setdefault("time_s", []).append(float(ctx_tick[i]) * u.step_dt)
                    last_touch[dones] = 10**6
                    last_adv[dones] = 10**6
                    last_resample[dones] = 10**6
                reset_prev = dones
        per10k = {k: (falls[k] / expo[k] * 1e4 if expo[k] else None) for k in ("touch", "corner", "resample", "none")}
        med = lambda xs: (sorted(xs)[len(xs) // 2] if xs else None)  # noqa: E731
        results[level] = {
            "stage": stages[level - 1],
            "episodes": c,
            "fall_per_episode": c["fall"] / max(c["episodes"], 1),
            "falls_by_flag": falls,
            "falls_by_combination": combos,
            "exposure_fraction": {k: expo[k] / max(expo["all"], 1) for k in ("touch", "corner", "resample", "none")},
            "falls_per_10k_steps_in_situation": per10k,
            "falls_per_10k_steps_overall": c["fall"] / max(expo["all"], 1) * 1e4,
            "at_fall_median": {k: med(v) for k, v in fall_ctx.items()},
            "mean_robot_speed": speed_sum / max(expo["all"], 1),
            "waypoint_passes": {k: cmd.pass_count[k] - pass0[k] for k in pass0},
            "mean_success_time_s": ep_len_sum / max(ep_len_n, 1),
        }
        print(f"[RESULT] level {level}: {json.dumps(results[level], ensure_ascii=False, default=str)}", flush=True)

    if args_cli.output:
        with open(args_cli.output, "w") as f:
            json.dump({"checkpoint": ckpt, "stochastic": args_cli.stochastic, "results": results}, f, indent=2, ensure_ascii=False, default=str)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
