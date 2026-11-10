# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール阻止方策の結果 (失点 / 阻止 / 転倒 / 時間切れ) を発射速度・狙い別に集計する (headless)。

例::

    uv run eval_block.py --headless --num_envs 1024 --steps 1500 --checkpoint /abs/path/model_N.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Block policy evaluation.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--task", type=str, default="Isaac-Block-K1-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=1500, help="評価 step 数 (50 Hz)")
parser.add_argument("--zero_policy", action="store_true", help="診断用: 残差 0・コマンド 0 (歩行 expert の立位) で評価")
parser.add_argument("--stochastic", action="store_true", help="平均ではなくサンプルした行動で評価 (学習時と同条件)")
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
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_k1_locomotion.tasks  # noqa: F401, E402

OUTCOMES = ("conceded", "resolved", "fell", "timeout")
REASONS = ("out_wide", "away", "stopped", "lost_after_touch")  # resolved の内訳 + 触った後に転倒/時間切れ (コマンド項の Metrics から)


def table(title, key, edges, rows):
    print(f"\n  by {title}:")
    print(f"    {'bin':>14s} {'n':>6s} " + " ".join(f"{o:>9s}" for o in OUTCOMES))
    lo = -float("inf")
    for hi in list(edges) + [float("inf")]:
        sel = [r for r in rows if lo < r[key] <= hi]
        n = max(len(sel), 1)
        pct = " ".join(f"{100.0 * sum(r['outcome'] == o for r in sel) / n:8.1f}%" for o in OUTCOMES)
        print(f"    ({lo:>5.2f},{hi:>5.2f}] {len(sel):6d} {pct}")
        lo = hi


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    path = retrieve_file_path(args_cli.checkpoint)
    # map_location 必須: 既定 (None) だと checkpoint が保存時の GPU (cuda:0) に載り、別 GPU の方策へは copy_ が無言で失敗する
    runner.load(path, map_location=agent_cfg.device)
    pol = runner.alg.policy
    policy = (lambda o: pol.env_actions(o, pol.act(o))) if args_cli.stochastic else runner.alg.act_inference
    if args_cli.zero_policy:
        policy = lambda o: pol.env_actions(o, torch.zeros(env.num_envs, pol.num_policy_actions, device=agent_cfg.device))
    raw_env = env.unwrapped
    term = raw_env.command_manager.get_term("ball_launch")
    robot = raw_env.scene["robot"]
    tm = raw_env.termination_manager
    dt = raw_env.step_dt

    rows = []
    logged = {k: [] for k in ("conceded", "resolved") + REASONS}  # 学習ログと同じ Metrics/ の値 (reset 呼び出しごとの平均)
    obs = env.get_observations()
    # 弾き出し速度: 最初に足 (xy 0.35 m 以内) に触れた後のボール速さの最大値 (接触の次 step から)
    touched_foot = torch.zeros(env.num_envs, dtype=torch.bool, device=raw_env.device)
    peak_after = torch.zeros(env.num_envs, device=raw_env.device)
    with torch.inference_mode():
        for step in range(args_cli.steps):
            speed = term.metrics["launch_speed"].clone()
            lateral = term.launch_lateral.abs().clone()
            t_ep = (raw_env.episode_length_buf + 1).float() * dt
            base_x, base_y = term.body_line_xy()
            base_y = base_y.abs()
            touched = torch.norm(robot.data.root_pos_w[:, :2] - raw_env.scene["soccer_ball"].data.root_pos_w[:, :2], dim=1) < 0.45
            ball_w = raw_env.scene["soccer_ball"].data.root_pos_w
            ball_z = ball_w[:, 2] - raw_env.scene.env_origins[:, 2]
            feet_d = torch.norm(robot.data.body_link_pos_w[:, term._feet_idx, :] - ball_w.unsqueeze(1), dim=-1).min(dim=1).values
            ball_v = torch.norm(raw_env.scene["soccer_ball"].data.root_lin_vel_w[:, :2], dim=1)
            peak_after = torch.where(touched_foot, torch.maximum(peak_after, ball_v), peak_after)
            feet_d_xy = torch.norm(robot.data.body_link_pos_w[:, term._feet_idx, :2] - ball_w[:, :2].unsqueeze(1), dim=-1).min(dim=1).values
            touched_foot |= feet_d_xy < 0.35
            rebound = torch.where(touched_foot, peak_after, torch.full_like(peak_after, float("nan")))
            obs, _, dones, extras = env.step(policy(obs))
            for k in logged:
                if f"Metrics/ball_launch/{k}" in extras.get("log", {}):
                    logged[k].append(extras["log"][f"Metrics/ball_launch/{k}"])
            d = dones.bool()
            if not d.any():
                continue
            fell = tm.get_term("base_contact") | tm.get_term("base_height")
            outcome = torch.full_like(d, 3, dtype=torch.long)  # timeout
            outcome[tm.get_term("ball_resolved")] = 1
            outcome[tm.get_term("ball_conceded")] = 0
            outcome[fell] = 2  # 転倒は同時に失点していても転倒扱い
            for i in d.nonzero().flatten().tolist():
                rows.append({"speed": speed[i].item(), "lateral": lateral[i].item(), "t": t_ep[i].item(), "outcome": OUTCOMES[outcome[i]],
                             "base_x": base_x[i].item(), "base_y": base_y[i].item(), "near": touched[i].item(),
                             "ball_z": ball_z[i].item(), "feet_d": feet_d[i].item(), "ball_v": ball_v[i].item(), "rebound": rebound[i].item()})
            touched_foot[d] = False
            peak_after[d] = 0.0
            if step % 500 == 0:
                print(f"[step {step}] episodes={len(rows)}")

    n = max(len(rows), 1)
    print(f"\n=== {args_cli.checkpoint}  envs={env.num_envs} steps={args_cli.steps} episodes={len(rows)} ===")
    print("  " + "  ".join(f"{o}: {100.0 * sum(r['outcome'] == o for r in rows) / n:5.1f}%" for o in OUTCOMES))
    print("  logged Metrics/ (学習ログ方式, 阻止内訳はこちら): " + "  ".join(f"{k}: {100.0 * sum(v) / max(len(v), 1):5.1f}%" for k, v in logged.items()))
    res = [r for r in rows if r["outcome"] == "resolved"]
    print(f"  終了直前に胴体がボール 0.45 m 以内 (阻止のみ): {100.0 * sum(r['near'] for r in res) / max(len(res), 1):5.1f}%")
    print(f"  終了時の胴体の横ずれ |y| 平均: 全体 {sum(r['base_y'] for r in rows) / n:.3f} m, 阻止 {sum(r['base_y'] for r in res) / max(len(res), 1):.3f} m")
    print(f"  終了時の胴体のライン前方 x 平均: 全体 {sum(r['base_x'] for r in rows) / n:+.3f} m, 阻止 {sum(r['base_x'] for r in res) / max(len(res), 1):+.3f} m,"
          f" 0.5 m 以上前に出た割合 {100.0 * sum(r['base_x'] > 0.5 for r in rows) / n:5.1f}%")
    print(f"  阻止時 (終了 1 step 前): ボール高さ z 平均 {sum(r['ball_z'] for r in res) / max(len(res), 1):.3f} m, 最寄り足との距離 平均 {sum(r['feet_d'] for r in res) / max(len(res), 1):.3f} m, ボール速度 平均 {sum(r['ball_v'] for r in res) / max(len(res), 1):.3f} m/s, 1 m 以内で 0.3 m/s 未満 {100.0 * sum(r['feet_d'] < 1.0 and r['ball_v'] < 0.3 for r in res) / max(len(res), 1):5.1f}%")
    reb = [r["rebound"] for r in res if r["rebound"] == r["rebound"]]
    print(f"  阻止のうち足に触れた割合 {100.0 * len(reb) / max(len(res), 1):5.1f}%, 接触後のボール最大速度 (弾き出し) 平均 {sum(reb) / max(len(reb), 1):.3f} m/s,"
          f" 1 m/s 未満 {100.0 * sum(v < 1.0 for v in reb) / max(len(reb), 1):5.1f}%")
    table("launch speed [m/s]", "speed", [1.0, 2.0, 3.0, 4.5], rows)
    table("|target lateral| [m]", "lateral", [0.25, 0.5, 0.75], rows)
    print("\n  失点率 [%] 速度帯 (行) × |狙い横| (列):")
    lat_edges, spd_edges = [0.25, 0.5, 0.75, float("inf")], [1.0, 2.0, 3.0, 4.5, float("inf")]
    print("    " + " ".join(f"{'<' + str(e):>7s}" for e in lat_edges))
    lo_s = -float("inf")
    for hi_s in spd_edges:
        cells = []
        lo_l = -float("inf")
        for hi_l in lat_edges:
            sel = [r for r in rows if lo_s < r["speed"] <= hi_s and lo_l < r["lateral"] <= hi_l]
            cells.append(f"{100.0 * sum(r['outcome'] == 'conceded' for r in sel) / max(len(sel), 1):6.1f}%")
            lo_l = hi_l
        print(f"    v<{hi_s:<4} " + " ".join(cells))
        lo_s = hi_s
    table("episode time at end [s]", "t", [0.5, 1.0, 2.0, 4.0], rows)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
