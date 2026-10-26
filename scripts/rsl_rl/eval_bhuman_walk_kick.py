"""bhuman_walk_kick の方策を IsaacLab 上で決定論的に回し、キックの出来を数える。

    python scripts/rsl_rl/eval_bhuman_walk_kick.py --task Isaac-K1-BHuman-Walk-Kick-Play-v0 \
        --checkpoint logs/rsl_rl/k1_bhuman_walk_kick/<run>/model_XXXX.pt --num_envs 256 --steps 3000 --headless

出力 (標準出力):
* キック回数、1 キックあたりの時間 (ボールを置いてから片付くまで)
* ボールの最高速度 (強キック / 距離指定キック別)、距離指定キックの |速度 - 必要な初速|
* 目標方向とボールの進んだ向きのずれ (平均・中央値・15°/30° 以内の割合)
* 転倒 (base_contact / base_height で終わったエピソード) の回数
* 仕様 (docs/bhuman_walk_kick_spec.md) の項目 (キック 1 回ごとの割合。MuJoCo の場面ごとの値とは分母が違う):
  蹴り足の向き 90±25°、ボールが蹴り足の内側、止まる (当たる前 1.5 秒に 0.6 m/s 以下まで減速)、
  軸足 (0.1 秒以上前から接地)、振り、つながり (蹴って 3 秒以内に歩き出す)

--task に段ごとのタスク (Isaac-K1-BHuman-Walk-Kick-Stage2-v0 など) を渡してよい。観測ノイズ・ボール観測ノイズ・押しは
切って回す (Play と同じ。--keep_noise で残す)。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-K1-BHuman-Walk-Kick-Play-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--stochastic", action="store_true", help="方策のノイズを入れたまま回す")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--json_out", type=str, default=None)
parser.add_argument("--keep_noise", action="store_true", help="観測ノイズと押しを学習のまま残す")
parser.add_argument("--ball_max_depen", type=float, default=None, help="ボールの max_depenetration_velocity を上書き")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_k1_locomotion.tasks  # noqa: F401, E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    if not args_cli.keep_noise:
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.commands.kick.obs_noise = False
        env_cfg.events.push_robot = None
    if args_cli.ball_max_depen is not None:
        env_cfg.scene.soccer_ball.spawn.rigid_props.max_depenetration_velocity = args_cli.ball_max_depen
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ckpt = os.path.abspath(args_cli.checkpoint)
    runner.load(ckpt, load_optimizer=False)
    policy = runner.alg.policy
    policy.eval()
    if args_cli.stochastic:
        act = lambda o: policy.act(o)  # noqa: E731
    else:
        act = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    k = uenv.command_manager.get_term("kick")
    k.eval_log = []
    k.eval_walkoff = []
    tm = uenv.termination_manager
    fall_terms = [n for n in tm.active_terms if n in ("base_contact", "base_height", "bad_orientation")]
    n_falls = 0
    n_episode_end = 0
    freq_abs = 0.0
    cmd_abs = torch.zeros(3, device=uenv.device)
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            actions = act(obs)
            obs, _, dones, _ = env.step(actions)
            fell = torch.zeros(uenv.num_envs, dtype=torch.bool, device=uenv.device)
            for n in fall_terms:
                fell |= tm.get_term(n)
            n_falls += int(fell.sum().item())
            n_episode_end += int(dones.sum().item())
            freq_abs += uenv._gait_freq_action_offset.abs().mean().item()
            cmd_abs += uenv.command_manager.get_command("base_velocity").abs().mean(dim=0)

    rec = k.eval_log
    sim_s = args_cli.steps * uenv.step_dt * uenv.num_envs
    out = {"checkpoint": ckpt, "env_seconds": sim_s, "falls": n_falls, "episode_ends": n_episode_end,
           "falls_per_min": n_falls / sim_s * 60.0,
           "mean_abs_freq_offset_hz": freq_abs / args_cli.steps,
           "mean_abs_cmd": (cmd_abs / args_cli.steps).tolist()}
    if rec:
        cat = {key: torch.cat([r[key] for r in rec]) for key in rec[0]}
        sp, tsp, st, err = cat["speed"], cat["target_speed"], cat["strong"], cat["dir_err_deg"]
        out.update({
            "kicks": int(sp.numel()),
            "kicks_per_min": sp.numel() / sim_s * 60.0,
            "sec_per_ball": float(cat["ball_steps"].float().mean() * uenv.step_dt),
            "speed_all_mean": float(sp.mean()),
            "speed_strong_mean": float(sp[st].mean()) if st.any() else None,
            "speed_strong_p90": float(sp[st].quantile(0.9)) if st.any() else None,
            "speed_normal_abs_err": float((sp - tsp)[~st].abs().mean()) if (~st).any() else None,
            "speed_normal_target_mean": float(tsp[~st].mean()) if (~st).any() else None,
            "speed_normal_mean": float(sp[~st].mean()) if (~st).any() else None,
            # 蹴ってからボールを片付けるまでの移動距離 (ボールの減速 0.75/s を戻して初速に直した値も出す)
            "disp_m_mean": float(cat["disp_m"].mean()),
            "disp_m_strong_mean": float(cat["disp_m"][st].mean()) if st.any() else None,
            "speed_from_disp_strong": float((cat["disp_m"][st] * 0.75 / (1 - torch.exp(-0.75 * cat["kick_counter"][st] * uenv.step_dt))).mean()) if st.any() else None,
            "strike_h_strong_mean": float(cat["strike_h"][st].mean()) if st.any() else None,
            "strike_vz_strong_mean": float(cat["strike_vz"][st].mean()) if st.any() else None,
            "dir_err_mean_deg": float(err.mean()),
            "dir_err_median_deg": float(err.median()),
            "dir_within_15deg": float((err < 15).float().mean()),
            "dir_within_30deg": float((err < 30).float().mean()),
            "dir_err_strong_mean_deg": float(err[st].mean()) if st.any() else None,
            # 仕様の項目 (キック 1 回ごとの割合)
            "spec_yaw_ok": float(((cat["foot_yaw_abs"] - 90).abs() <= 25).float().mean()),
            "spec_foot_yaw_abs_mean_deg": float(cat["foot_yaw_abs"].mean()),
            "spec_foot_yaw_inside_mean_deg": float(cat["foot_yaw_in"].mean()),
            "spec_yaw_inside_ok": float(((cat["foot_yaw_in"] - 90).abs() <= 25).float().mean()),
            "spec_inside_ok": float((cat["ball_yin_foot"] > cat["ball_x_foot"].abs()).float().mean()),
            "spec_ball_x_foot_mean": float(cat["ball_x_foot"].mean()),
            "spec_ball_yin_foot_mean": float(cat["ball_yin_foot"].mean()),
            "spec_stop_ok": float((cat["min_speed"] <= 0.6).float().mean()),
            "spec_min_speed_mean": float(cat["min_speed"].mean()),
            "spec_stop_s_mean": float(cat["stop_s"].mean()),
            "spec_plant_ok": float(cat["plant"].float().mean()),
            "spec_swing_ok": float((cat["foot_air"] & (cat["foot_hvel"] >= 1.0)).float().mean()),
            "spec_foot_air": float(cat["foot_air"].float().mean()),
            "spec_foot_hvel_mean": float(cat["foot_hvel"].mean()),
            "trunk_yaw_inside_mean_deg": float(cat["trunk_yaw_in"].mean()),
            "hip_yaw_out_mean_deg": float(cat["hip_yaw_out"].mean()),
        })
        if k.eval_walkoff:
            wo = torch.cat(k.eval_walkoff)
            out["spec_walkoff_ok"] = float((wo >= 0.3).float().mean())
            out["spec_walkoff_n"] = int(wo.numel())
        # キック方向 (ロボットがボールを置かれた時の向き基準) 別のずれ
        rel = cat["kick_dir_rel_deg"].abs()
        for lo, hi in ((0, 45), (45, 90), (90, 135), (135, 181)):
            m = (rel >= lo) & (rel < hi)
            if m.any():
                out[f"dir_err_mean_deg_rel{lo}-{hi}"] = float(err[m].mean())
                out[f"speed_mean_rel{lo}-{hi}"] = float(sp[m].mean())
    else:
        out["kicks"] = 0
    print("[EVAL]", json.dumps(out, indent=1), flush=True)
    if args_cli.json_out:
        with open(args_cli.json_out, "w") as f:
            json.dump(out, f, indent=1)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
