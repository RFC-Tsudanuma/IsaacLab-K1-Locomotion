# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール奪取方策の結果 (奪取 / 敵に当てた / 転倒 / 時間切れ) と奪取までの時間を集計する (headless)。

例::

    uv run eval_steal.py --headless --num_envs 1024 --steps 1500 --checkpoint /abs/path/model_N.pt
    uv run eval_steal.py --headless --num_envs 1024 --steps 1500 --checkpoint /abs/path/model_N.pt --zero_policy  # ベースライン
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Steal policy evaluation.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--task", type=str, default="Isaac-Steal-K1-Play-v0")
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

OUTCOMES = ("stolen", "lost", "fell", "timeout", "no_foot")


def table(title, key, edges, rows):
    print(f"\n  by {title}:")
    print(f"    {'bin':>14s} {'n':>6s} " + " ".join(f"{o:>9s}" for o in OUTCOMES) + "   t_steal[s]")
    lo = -float("inf")
    for hi in list(edges) + [float("inf")]:
        sel = [r for r in rows if lo < r[key] <= hi]
        n = max(len(sel), 1)
        pct = " ".join(f"{100.0 * sum(r['outcome'] == o for r in sel) / n:8.1f}%" for o in OUTCOMES)
        st = [r["t_steal"] for r in sel if r["outcome"] == "stolen"]
        print(f"    ({lo:>5.2f},{hi:>5.2f}] {len(sel):6d} {pct}   {sum(st) / max(len(st), 1):6.2f}")
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
    # map_location 必須: 既定 (None) だと checkpoint が保存時の GPU に載り、別 GPU の方策へは copy_ が無言で失敗する
    runner.load(path, map_location=agent_cfg.device)
    pol = runner.alg.policy
    policy = (lambda o: pol.env_actions(o, pol.act(o))) if args_cli.stochastic else runner.alg.act_inference
    if args_cli.zero_policy:
        policy = lambda o: pol.env_actions(o, torch.zeros(env.num_envs, pol.num_policy_actions, device=agent_cfg.device))
    raw_env = env.unwrapped
    term = raw_env.command_manager.get_term("steal")
    tm = raw_env.termination_manager
    dt = raw_env.step_dt

    rows = []
    sensors = [raw_env.scene[n] for n in ("contact_balls_right", "contact_balls_left")]
    foot_touch = torch.zeros(env.num_envs, dtype=torch.bool, device=agent_cfg.device)  # 足がボールに触れた (接触センサ)
    bumped = torch.zeros_like(foot_touch)  # 胴体が障害物に接近しすぎた (敵にぶつかった目安)
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            # 終了 step の値 (reset で消える前に控える)
            dist0 = torch.norm(term.obstacle_xy() - raw_env.scene["robot"].data.root_pos_w[:, :2], dim=1)
            ball_r = term.ball_obstacle_dist().clone()
            t_ep = (raw_env.episode_length_buf + 1).float() * dt
            t_steal = term.stolen_at.clone()
            obs, _, dones, _ = env.step(policy(obs))
            for sen in sensors:
                foot_touch |= torch.norm(sen.data.force_matrix_w, dim=-1).amax(dim=(1, 2)) > 1.0
            bumped |= torch.norm(term.obstacle_xy() - raw_env.scene["robot"].data.root_pos_w[:, :2], dim=1) < term.cfg.obstacle_radius + 0.25
            d = dones.bool()
            if not d.any():
                continue
            fell = tm.get_term("base_contact") | tm.get_term("base_height")
            outcome = torch.full_like(d, 3, dtype=torch.long)  # timeout
            outcome[tm.get_term("ball_stolen")] = 0
            outcome[tm.get_term("ball_lost")] = 1
            outcome[tm.get_term("ball_no_foot")] = 4
            outcome[fell] = 2  # 転倒は同時に失敗していても転倒扱い
            for i in d.nonzero().flatten().tolist():
                rows.append({"t": t_ep[i].item(), "outcome": OUTCOMES[outcome[i]], "t_steal": t_steal[i].item(),
                             "dist0": dist0[i].item(), "ball_r": ball_r[i].item(), "touched": t_steal[i].item() >= 0.0,
                             "foot": foot_touch[i].item(), "bumped": bumped[i].item()})
            foot_touch[d] = False
            bumped[d] = False
            if step % 500 == 0:
                print(f"[step {step}] episodes={len(rows)}")

    n = max(len(rows), 1)
    stolen = [r for r in rows if r["outcome"] == "stolen"]
    print(f"\n=== {args_cli.checkpoint}  envs={env.num_envs} steps={args_cli.steps} episodes={len(rows)} ===")
    print("  " + "  ".join(f"{o}: {100.0 * sum(r['outcome'] == o for r in rows) / n:5.1f}%" for o in OUTCOMES))
    print(f"  圏外に出した後に転倒/時間切れ (lost_after_touch): {100.0 * sum(r['touched'] and r['outcome'] != 'stolen' for r in rows) / n:5.1f}%")
    print(f"  奪取までの時間 (ラッチ時刻) 平均 {sum(r['t_steal'] for r in stolen) / max(len(stolen), 1):.2f} s (奪取のみ)")
    print(f"  終了時の ball-obstacle 距離 平均: 全体 {sum(r['ball_r'] for r in rows) / n:.2f} m")
    print(f"  足がボールに触れた (接触センサ): 全体 {100.0 * sum(r['foot'] for r in rows) / n:5.1f}%, 奪取のみ {100.0 * sum(r['foot'] for r in stolen) / max(len(stolen), 1):5.1f}%")
    print(f"  胴体が障害物に 0.25 m 以内まで接近: 全体 {100.0 * sum(r['bumped'] for r in rows) / n:5.1f}%")
    table("終了時の robot-obstacle 距離 [m] (≈ 初期距離の目安)", "dist0", [1.0, 1.5, 2.0, 2.5], rows)
    table("episode time at end [s]", "t", [1.0, 2.0, 3.0, 5.0], rows)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
