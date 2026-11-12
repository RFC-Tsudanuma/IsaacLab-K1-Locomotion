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
parser.add_argument("--move_speed", type=float, default=None, help="敵・ボールの速度上限 [m/s] を固定 (カリキュラム無効)。既定は cfg の初期値")
parser.add_argument("--diag_back", action="store_true", help="診断: 敵が背後 (|方位|>135°) の開始 1 s の方策出力 (コマンド・残差) を集計")
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

OUTCOMES = ("stolen", "lost", "fell", "timeout", "no_foot", "wrong_dir")
DIRS = ("front", "left", "back", "right")


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

    if hasattr(env_cfg.curriculum, "steal_move_speed"):
        env_cfg.curriculum.steal_move_speed = None  # 評価中に上限が変わらないように
    if args_cli.move_speed is not None:
        env_cfg.commands.steal.move_speed_max = args_cli.move_speed
        env_cfg.commands.steal.static_prob = 0.0  # 速度指定時は全エピソードを U(0, move_speed) で動かす
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    path = retrieve_file_path(args_cli.checkpoint)
    # map_location 必須: 既定 (None) だと checkpoint が保存時の GPU に載り、別 GPU の方策へは copy_ が無言で失敗する
    # 評価に使うのは actor だけ。critic の入力次元が今の cfg と違う checkpoint (critic 観測を足す前のもの) も
    # 読めるように、形が合わない critic のテンソルは今の (未使用の) 値を残す。optimizer も読まない。
    ckpt = torch.load(path, map_location=agent_cfg.device, weights_only=False)
    own = runner.alg.policy.state_dict()
    sd = ckpt["model_state_dict"]
    for k, v in list(sd.items()):
        if k in own and own[k].shape != v.shape:
            assert k.startswith("critic"), f"actor 側の形が合わない: {k} {tuple(v.shape)} vs {tuple(own[k].shape)}"
            sd[k] = own[k]
    runner.alg.policy.load_state_dict(sd)
    pol = runner.alg.policy
    policy = (lambda o: pol.env_actions(o, pol.act(o))) if args_cli.stochastic else runner.alg.act_inference
    if args_cli.zero_policy:
        policy = lambda o: pol.env_actions(o, torch.zeros(env.num_envs, pol.num_policy_actions, device=agent_cfg.device))
    raw_env = env.unwrapped
    term = raw_env.command_manager.get_term("steal")
    tm = raw_env.termination_manager
    dt = raw_env.step_dt

    rows = []
    diag = []
    sensors = [raw_env.scene[n] for n in ("contact_balls_right", "contact_balls_left")]
    foot_touch = torch.zeros(env.num_envs, dtype=torch.bool, device=agent_cfg.device)  # 足がボールに触れた (接触センサ)
    bumped = torch.zeros_like(foot_touch)  # 胴体が障害物に接近しすぎた (敵にぶつかった目安)
    trunk_hit = torch.zeros_like(foot_touch)  # 胴体と障害物の接触センサが反応した
    # 上体の揺れ (全 env・全 step の平均)。角加速度・並進加速度は有限差分、リセット直後の step は除く
    robot = raw_env.scene["robot"]
    sway = {"tilt": 0.0, "ang_vel_xy": 0.0, "ang_acc": 0.0, "lin_acc": 0.0, "n": 0}
    prev_w = robot.data.root_ang_vel_b.clone()
    prev_v = robot.data.root_lin_vel_w.clone()
    trunk_sensor = raw_env.scene["contact_trunk_obstacle"] if "contact_trunk_obstacle" in raw_env.scene.sensors else None
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            # 終了 step の値 (reset で消える前に控える)
            dist0 = torch.norm(term.obstacle_xy() - raw_env.scene["robot"].data.root_pos_w[:, :2], dim=1)
            ball_r = term.ball_obstacle_dist().clone()
            side0 = term.ball_side0.abs().rad2deg()
            bear0 = term.bearing0.abs().rad2deg()
            ospeed = torch.norm(term.obstacle_vel, dim=1)
            t_ep = (raw_env.episode_length_buf + 1).float() * dt
            t_steal = term.stolen_at.clone()
            act = policy(obs)
            if args_cli.diag_back:
                early = (raw_env.episode_length_buf < 50) & (term.bearing0.abs() > 2.356)
                if early.any():
                    cmd = act[early, 12:15]
                    res = act[early, 15:27]
                    diag.append(torch.cat([cmd.abs().mean(0), cmd.mean(0), res.abs().amax(1).mean().view(1)]).cpu())
            obs, _, dones, _ = env.step(act)
            for sen in sensors:
                foot_touch |= torch.norm(sen.data.force_matrix_w, dim=-1).amax(dim=(1, 2)) > 1.0
            bumped |= torch.norm(term.obstacle_xy() - raw_env.scene["robot"].data.root_pos_w[:, :2], dim=1) < term.cfg.obstacle_radius + 0.25
            if trunk_sensor is not None:
                trunk_hit |= torch.norm(trunk_sensor.data.force_matrix_w, dim=-1).amax(dim=(1, 2)) > 1.0
            w, v = robot.data.root_ang_vel_b, robot.data.root_lin_vel_w
            ok = raw_env.episode_length_buf >= 2  # 直前 step と同じエピソード
            if ok.any():
                sway["tilt"] += torch.norm(robot.data.projected_gravity_b[ok, :2], dim=1).sum().item()
                sway["ang_vel_xy"] += torch.norm(w[ok, :2], dim=1).sum().item()
                sway["ang_acc"] += torch.norm((w - prev_w)[ok, :2], dim=1).sum().item() / dt
                sway["lin_acc"] += torch.norm((v - prev_v)[ok], dim=1).sum().item() / dt
                sway["n"] += int(ok.sum())
            prev_w, prev_v = w.clone(), v.clone()
            d = dones.bool()
            if not d.any():
                continue
            fell = tm.get_term("base_height")
            if "base_contact" in tm.active_terms:
                fell = fell | tm.get_term("base_contact")
            outcome = torch.full_like(d, 3, dtype=torch.long)  # timeout
            outcome[tm.get_term("ball_stolen")] = 0
            outcome[tm.get_term("ball_lost")] = 1
            outcome[tm.get_term("ball_no_foot")] = 4
            if "ball_wrong_dir" in tm.active_terms:
                outcome[tm.get_term("ball_wrong_dir")] = 5
            outcome[fell] = 2  # 転倒は同時に失敗していても転倒扱い
            for i in d.nonzero().flatten().tolist():
                rows.append({"t": t_ep[i].item(), "outcome": OUTCOMES[outcome[i]], "t_steal": t_steal[i].item(),
                             "dist0": dist0[i].item(), "ball_r": ball_r[i].item(), "touched": t_steal[i].item() >= 0.0,
                             "foot": foot_touch[i].item(), "bumped": bumped[i].item(), "dir": int(term.target_idx[i]),
                             "side0": side0[i].item(), "trunk": trunk_hit[i].item(), "bear0": bear0[i].item(),
                             "ospeed": ospeed[i].item()})
            foot_touch[d] = False
            bumped[d] = False
            trunk_hit[d] = False
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
    print(f"  胴体が障害物に接触 (センサ): 全体 {100.0 * sum(r['trunk'] for r in rows) / n:5.1f}%, 奪取のみ {100.0 * sum(r['trunk'] for r in stolen) / max(len(stolen), 1):5.1f}%")
    table("終了時の robot-obstacle 距離 [m] (≈ 初期距離の目安)", "dist0", [1.0, 1.5, 2.0, 2.5], rows)
    m = max(sway["n"], 1)
    print(f"  上体の揺れ (平均): 傾き |g_xy| {sway['tilt'] / m:.4f} / ロール・ピッチ角速度 {sway['ang_vel_xy'] / m:.3f} rad/s / "
          f"ロール・ピッチ角加速度 {sway['ang_acc'] / m:.2f} rad/s² / 並進加速度 {sway['lin_acc'] / m:.2f} m/s²")
    table("episode time at end [s]", "t", [1.0, 2.0, 3.0, 5.0], rows)
    table("|ball side angle| at start [deg] (0 = robot side, 180 = behind the obstacle)", "side0", [45.0, 90.0, 135.0], rows)
    table("|opponent bearing| at start [deg] (0 = in front of the robot, 180 = behind it)", "bear0", [45.0, 90.0, 135.0], rows)
    if term.move_speed_max > 0:
        table("opponent speed [m/s]", "ospeed", [0.25, 0.5, 0.75], rows)
    if term.cfg.directed:
        table(f"target direction (bins = {', '.join(DIRS)})", "dir", [0.5, 1.5, 2.5], rows)
    if diag:
        d = torch.stack(diag).mean(0).tolist()
        print(f"\n  [diag_back] 背後スタートの開始 1 s: |cmd| mean vx {d[0]:.2f} vy {d[1]:.2f} wz {d[2]:.2f} / "
              f"cmd mean vx {d[3]:+.2f} vy {d[4]:+.2f} wz {d[5]:+.2f} / |residual| max mean {d[6]:.2f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
