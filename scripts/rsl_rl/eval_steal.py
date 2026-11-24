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
import math
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
parser.add_argument("--match", action="store_true", help="試合条件 (歩行中開始・寄ってくる敵・蹴られるボール・観測劣化・途中解除) で評価。既定は無効 (従来の評価)")
parser.add_argument("--move_speed", type=float, default=None, help="敵・ボールの速度上限 [m/s] を固定 (カリキュラム無効)。既定は cfg の初期値")
parser.add_argument("--node_dir", choices=["away", "lr", "right", "opp_near", "opp_far"], default=None,
                    help="実機ノードの方向の決め方を再現 (起動 tick の観測から決めて固定)。away = 観測している敵 (仮想を含む) → ボール、"
                         "lr = ロボット→ボールの線の真横 (観測している敵のいない側)、right = ロボット→ボールの線から右へ --node_dir_deg "
                         "(敵の方向と 60° 以内なら真後ろ、それも駄目なら away。今の方策は左足で右へ払う蹴りしか持たないため)、"
                         "opp_near / opp_far = 敵→ボールの線の真横 (near = ロボットがいる側 = 回り込みが少ない側、far = その反対側)")
parser.add_argument("--node_track", type=float, default=0.0, help="--node_dir opp_*: > 0 なら起動 tick で固定せず、ボールがロボットからこの距離 [m] に入るまで、敵が見えている tick ごとに向きを更新する (左右の側は起動 tick のまま。ノードの steal_dir_track_m と同じ)")
parser.add_argument("--node_dir_deg", type=float, default=110.0, help="--node_dir right の角度 [deg] (ロボット→ボールの向きから右回り)")
parser.add_argument("--expert_sym", action="store_true", help="診断: 歩行 expert を左右対称にして評価 (E(o) と 反転 E(反転 o) の平均)。--mirror の検算用")
parser.add_argument("--dump_rows", type=str, default=None, help="エピソードごとの集計行を JSON で書き出す (立ち位置などの解析用)")
parser.add_argument("--mirror", choices=["step", "latch", "full", "full_opp"], default=None,
                    help="方策を左右対称にして評価する (目標方向が左のときは観測と出力を左右反転)。step = 毎 tick 判定、latch = 作動直後の判定を保つ、full = latch + 歩行 expert も反転 (制御全体を鏡像にする)")
parser.add_argument("--dump_traj", type=str, default=None, help="先頭 256 env の毎 step の軌跡 (T, 256, 15) を torch.save する (解析用)")
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

OUTCOMES = ("stolen", "lost", "fell", "timeout", "no_foot", "wrong_dir", "neutral")
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

    if hasattr(env_cfg.commands.steal, "match_enable"):
        env_cfg.commands.steal.match_enable = bool(args_cli.match)
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
    if args_cli.mirror:
        from isaaclab_k1_locomotion.tasks.manager_based.locomotion.agents.steal_policy import steal_mirror_policy

        policy = steal_mirror_policy(pol, env.unwrapped, args_cli.mirror, agent_cfg.policy.ball_history)
    esym = {"d": 0.0, "a": 0.0, "n": 0}
    if args_cli.expert_sym:
        from tensordict import TensorDict

        from isaaclab_k1_locomotion.tasks.manager_based.locomotion.history_layout import POLICY_TERM_SPECS
        from isaaclab_k1_locomotion.tasks.manager_based.locomotion.mdp.symmetry import _mirror_history_group, _mirror_joints

        expert_actions = pol.expert_actions

        def expert_sym(o, cmd):
            a = expert_actions(o, cmd)
            om = {k: v for k, v in o.items()}
            om["policy"] = _mirror_history_group(o["policy"], "policy", POLICY_TERM_SPECS)
            am = _mirror_joints(expert_actions(TensorDict(om, batch_size=o.batch_size), cmd * torch.tensor([1.0, -1.0, -1.0], device=cmd.device)))
            esym["d"] += float((a - am).pow(2).mean())
            esym["a"] += float(a.pow(2).mean())
            esym["n"] += 1
            return 0.5 * (a + am)

        pol.expert_actions = expert_sym
    if args_cli.zero_policy:
        policy = lambda o: pol.env_actions(o, torch.zeros(env.num_envs, pol.num_policy_actions, device=agent_cfg.device))
    raw_env = env.unwrapped
    term = raw_env.command_manager.get_term("steal")
    tm = raw_env.termination_manager
    dt = raw_env.step_dt

    rows = []
    active = {"steps": 0, "falls": 0, "falls_pre": 0}
    diag = []
    traj = []
    sensors = [raw_env.scene[n] for n in ("contact_balls_right", "contact_balls_left")]
    foot_touch = torch.zeros(env.num_envs, dtype=torch.bool, device=agent_cfg.device)  # 足がボールに触れた (接触センサ)
    bumped = torch.zeros_like(foot_touch)
    between = torch.zeros_like(foot_touch)  # ボールと敵の間に入った  # 胴体が障害物に接近しすぎた (敵にぶつかった目安)
    trunk_hit = torch.zeros_like(foot_touch)  # 胴体と障害物の接触センサが反応した
    opp_hit = torch.zeros_like(foot_touch)  # ボールが敵に触れた (opp_touch_lost=false では終了しないので別に数える)
    # 足が最初にボールへ触れた瞬間の立ち位置: ボールから見た「ロボットの方向」と「敵の方向」の角度 [deg]
    # (0 = ボールと敵の間に立っている, 90 = 真横, 180 = 敵の反対側)。触れなかったエピソードは NaN
    touch_th = torch.full((env.num_envs,), float("nan"), device=agent_cfg.device)
    # 解析用 (--dump_rows): 符号付きの立ち位置 (ボール→敵を 0°、目標方向のある側を +) の開始時 / 最初に触れたとき、同じ系での目標方向
    start_sg, touch_sg, touch_tg = touch_th.clone(), touch_th.clone(), touch_th.clone()
    dir_latched = torch.zeros_like(foot_touch)  # --node_dir: このエピソードで目標方向を固定済み
    trk_close = torch.zeros_like(foot_touch)  # --node_track: ボールが近づいたので目標方向の更新をやめた
    opp_sign = torch.ones(env.num_envs, device=agent_cfg.device)  # --node_dir opp_*: ボール→敵の線の左 (+1) / 右 (−1)、起動 tick に決める
    # 回り込み量: 奪取の作動中 (w > 0)、足で触れるまでにボールのまわりを回った角度の累計 [deg] (ボールから見たロボットの方位の変化)
    circ = torch.zeros(env.num_envs, device=agent_cfg.device)
    prev_ang = torch.full_like(circ, float("nan"))
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
            tdir = term.target_idx.clone()  # 2026-11-24 修正: reset 後に読むと次のエピソードの方向になる
            ospeed = torch.norm(term.obstacle_vel, dim=1)
            kind = term.contest.float() + (term.contest & term.kick_on_reach).float()  # 0 = 敵が保持, 1 = 寄ってきてドリブル, 2 = 寄ってきて蹴る
            # 目標方向と「ボール→敵 (開始時のボール位置から)」の角度 [deg]。180 = 敵から真っすぐ離す方向
            to_o = term.obstacle_xy() - term.ball_start_xy
            to_o = to_o / torch.norm(to_o, dim=1, keepdim=True).clamp(min=1e-6)
            tgt_opp = torch.rad2deg(torch.acos((term.target_dir_w * to_o).sum(1).clamp(-1.0, 1.0)))
            t_ep = (raw_env.episode_length_buf + 1).float() * dt
            t_steal = term.stolen_at.clone()
            # 試合条件: 奪取の作動時間 (w > 0 か解除後) と、その間の転倒 (試合 sim の「作動 1 分あたりの転倒」と同じ数え方)
            steal_on = (term.w > 0.0) | (term.phase == 4)
            active["steps"] += int(steal_on.sum())
            if args_cli.node_dir:
                # ノードの steal_update_dir (指令なし): 起動 tick の観測での「敵→ボール」。反映は次 tick の観測から
                new = (term.phase != 0) & ~dir_latched
                ball_b, opp_b = term.obs_ball_b[:, :2], term.obs_opp_b
                if args_cli.node_dir == "lr":
                    opp_left = ball_b[:, 0] * (opp_b[:, 1] - ball_b[:, 1]) - ball_b[:, 1] * (opp_b[:, 0] - ball_b[:, 0]) > 0.0
                    left = torch.stack([-ball_b[:, 1], ball_b[:, 0]], dim=1)
                    away = term._to_world(torch.where(opp_left.unsqueeze(1), -left, left))
                elif args_cli.node_dir == "right":
                    fwd = ball_b / ball_b.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    to_opp = (opp_b - ball_b) / (opp_b - ball_b).norm(dim=1, keepdim=True).clamp(min=1e-6)
                    a = -math.radians(args_cli.node_dir_deg)
                    side = torch.stack([fwd[:, 0] * math.cos(a) - fwd[:, 1] * math.sin(a), fwd[:, 0] * math.sin(a) + fwd[:, 1] * math.cos(a)], dim=1)
                    free = lambda d: ((d * to_opp).sum(1) < 0.5).unsqueeze(1)  # 敵の方向から 60° より離れている
                    away = term._to_world(torch.where(free(side), side, torch.where(free(-fwd), -fwd, -to_opp)))
                elif args_cli.node_dir in ("opp_near", "opp_far"):
                    # ボール→敵の線の真横。側 = ロボットがいる側 (near) かその反対 (far)。線上・仮想の敵は右 (lr と同じ)
                    e = opp_b - ball_b
                    robot_left = ball_b[:, 0] * e[:, 1] - ball_b[:, 1] * e[:, 0] > 0.0  # ロボットはボール→敵の線の左にいる
                    opp_sign = torch.where(new, torch.where(robot_left == (args_cli.node_dir == "opp_near"), 1.0, -1.0), opp_sign)
                    away = term._to_world(torch.stack([-e[:, 1], e[:, 0]], dim=1) * opp_sign.unsqueeze(1))
                    if args_cli.node_track > 0.0:
                        trk_close |= (term.phase != 0) & (ball_b.norm(dim=1) < args_cli.node_track)
                        new = new | ((term.phase != 0) & ~trk_close & ~term.opp_virtual)
                else:
                    away = term._to_world(ball_b - opp_b)
                away = away / away.norm(dim=1, keepdim=True).clamp(min=1e-6)
                term.target_dir_w[new] = away[new]
                dir_latched |= new
            act = policy(obs)
            rb, ob = robot.data.root_pos_w[:, :2] - term.ball_xy(), term.obstacle_xy() - term.ball_xy()
            cross = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            tg_now = torch.rad2deg(torch.atan2(cross(ob, term.target_dir_w), (ob * term.target_dir_w).sum(1)))
            sg_now = torch.rad2deg(torch.atan2(cross(ob, rb), (ob * rb).sum(1))) * torch.sign(tg_now)
            start_sg = torch.where(start_sg.isnan(), sg_now, start_sg)
            ang = torch.atan2(rb[:, 1], rb[:, 0])
            d_ang = (torch.remainder(ang - prev_ang + torch.pi, 2 * torch.pi) - torch.pi).abs().nan_to_num(0.0)
            circ += torch.rad2deg(d_ang) * ((term.w > 0.0) & ~term.foot_touched & (rb.norm(dim=1) > 0.15))
            prev_ang = ang
            th_now = torch.rad2deg(torch.acos(((rb * ob).sum(1) / (rb.norm(dim=1) * ob.norm(dim=1)).clamp(min=1e-6)).clamp(-1.0, 1.0)))
            if args_cli.diag_back:
                early = (raw_env.episode_length_buf < 50) & (term.bearing0.abs() > 2.356)
                if early.any():
                    cmd = act[early, 12:15]
                    res = act[early, 15:27]
                    diag.append(torch.cat([cmd.abs().mean(0), cmd.mean(0), res.abs().amax(1).mean().view(1)]).cpu())
            if args_cli.dump_traj:  # [ロボット xy, yaw, ボール xy, 目標方向 xy, 敵 xy, 触れた, w, コマンド 3, 経過 step]
                traj.append(torch.cat([robot.data.root_pos_w[:, :2], robot.data.heading_w.unsqueeze(1), term.ball_xy(), term.target_dir_w, term.obstacle_xy(),
                                       term.foot_touched.float().unsqueeze(1), term.w.reshape(-1, 1), act[:, 12:15], raw_env.episode_length_buf.float().unsqueeze(1)], dim=1)[:256].cpu())
            obs, _, dones, _ = env.step(act)
            for sen in sensors:
                foot_touch |= torch.norm(sen.data.force_matrix_w, dim=-1).amax(dim=(1, 2)) > 1.0
            new_touch = term.foot_touched & touch_th.isnan()
            touch_th = torch.where(new_touch, th_now, touch_th)
            touch_sg = torch.where(new_touch, sg_now, touch_sg)
            touch_tg = torch.where(new_touch, tg_now.abs(), touch_tg)
            # ボールと敵の間に入った: ロボット中心をボール→敵の線分に射影して内側 (端から 0.1 m 以上) かつ横 0.25 m 以内
            rp = raw_env.scene["robot"].data.root_pos_w[:, :2]
            seg = term.obstacle_xy() - term.ball_xy()
            L = torch.norm(seg, dim=1).clamp(min=1e-6)
            u = seg / L.unsqueeze(1)
            rel = rp - term.ball_xy()
            along = (rel * u).sum(1)
            lat = (rel[:, 0] * u[:, 1] - rel[:, 1] * u[:, 0]).abs()
            # 足で触れる前だけ数える: 蹴った後はボールが動いて線分がロボットの上を掃くので、追っているだけで「間」に数えてしまう
            between |= (along > 0.1) & (along < L - 0.1) & (lat < 0.25) & ~term.foot_touched
            opp_hit |= term.lost_obstacle()
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
            if "steal_neutral" in tm.active_terms:
                outcome[tm.get_term("steal_neutral")] = 6
            outcome[fell] = 2  # 転倒は同時に失敗していても転倒扱い
            active["falls"] += int((fell & steal_on).sum())
            active["falls_pre"] += int((fell & ~steal_on).sum())
            for i in d.nonzero().flatten().tolist():
                rows.append({"t": t_ep[i].item(), "outcome": OUTCOMES[outcome[i]], "t_steal": t_steal[i].item(),
                             "dist0": dist0[i].item(), "ball_r": ball_r[i].item(), "touched": t_steal[i].item() >= 0.0,
                             "foot": foot_touch[i].item(), "bumped": bumped[i].item(), "dir": int(tdir[i]),
                             "side0": side0[i].item(), "trunk": trunk_hit[i].item(), "bear0": bear0[i].item(),
                             "ospeed": ospeed[i].item(), "between": between[i].item(), "tgt_opp": tgt_opp[i].item(), "kind": kind[i].item(), "touch_th": touch_th[i].item(),
                             "start_sg": start_sg[i].item(), "touch_sg": touch_sg[i].item(), "touch_tg": touch_tg[i].item(), "circ": circ[i].item(), "opp_hit": opp_hit[i].item()})
            foot_touch[d] = False
            bumped[d] = False
            between[d] = False
            opp_hit[d] = False
            dir_latched[d] = False
            trk_close[d] = False
            for buf in (touch_th, start_sg, touch_sg, touch_tg, prev_ang):
                buf[d] = float("nan")
            circ[d] = 0.0
            trunk_hit[d] = False
            if step % 500 == 0:
                print(f"[step {step}] episodes={len(rows)}")

    if args_cli.dump_traj:
        torch.save(torch.stack(traj), args_cli.dump_traj)
    if esym["n"]:
        print(f"  [expert_sym] 歩行 expert の非対称 RMS(E(o) − 反転 E(反転 o)) = {(esym['d'] / esym['n']) ** 0.5:.4f}、出力の RMS = {(esym['a'] / esym['n']) ** 0.5:.4f} (アクション単位)")
    if args_cli.dump_rows:
        import json
        with open(args_cli.dump_rows, "w") as f:
            json.dump(rows, f)
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
    print(f"  ボールと敵の間に入った: 全体 {100.0 * sum(r['between'] for r in rows) / n:5.1f}%, 奪取のみ {100.0 * sum(r['between'] for r in stolen) / max(len(stolen), 1):5.1f}%")
    for name, sel in (("触れた全エピソード", [r for r in rows if r["touch_th"] == r["touch_th"]]), ("奪取のみ", stolen)):
        c = [r["circ"] for r in sel]
        m = max(len(c), 1)
        print(f"  回り込み量 (触れるまでにボールのまわりを回った角度) {name}: 平均 {sum(c) / m:5.1f}°, >45° {100.0 * sum(x > 45 for x in c) / m:5.1f}%, >90° {100.0 * sum(x > 90 for x in c) / m:5.1f}%")
    table("終了時の robot-obstacle 距離 [m] (≈ 初期距離の目安)", "dist0", [1.0, 1.5, 2.0, 2.5], rows)
    if args_cli.match:
        mins = active["steps"] * dt / 60.0
        print(f"  [match] 奪取の作動 (w > 0 か解除後) {mins:.1f} 分 (延べ) で転倒 {active['falls']} 回 = "
              f"{active['falls'] / max(mins, 1e-9):.2f} 回/分、起動前 (歩行 expert のみ) の転倒 {active['falls_pre']} 回")
    m = max(sway["n"], 1)
    print(f"  上体の揺れ (平均): 傾き |g_xy| {sway['tilt'] / m:.4f} / ロール・ピッチ角速度 {sway['ang_vel_xy'] / m:.3f} rad/s / "
          f"ロール・ピッチ角加速度 {sway['ang_acc'] / m:.2f} rad/s² / 並進加速度 {sway['lin_acc'] / m:.2f} m/s²")
    table("episode time at end [s]", "t", [1.0, 2.0, 3.0, 5.0], rows)
    table("|ball side angle| at start [deg] (0 = robot side, 180 = behind the obstacle)", "side0", [45.0, 90.0, 135.0], rows)
    table("|opponent bearing| at start [deg] (0 = in front of the robot, 180 = behind it)", "bear0", [45.0, 90.0, 135.0], rows)
    if term.move_speed_max > 0:
        table("opponent speed [m/s]", "ospeed", [0.25, 0.5, 0.75], rows)
    if args_cli.match:
        table("敵の動き (bins = 敵がボールを保持, 寄ってきてドリブル, 寄ってきて蹴る)", "kind", [0.5, 1.5], rows)
    if term.cfg.directed:
        table(f"target direction (bins = {', '.join(DIRS)})", "dir", [0.5, 1.5, 2.5], rows)
        table("目標方向とボール→敵の角度 [deg] (0 = 敵へ向けて, 180 = 敵から真っすぐ離す)", "tgt_opp", [30.0, 60.0, 90.0, 120.0, 150.0], rows)
        print("  ボールが敵に触れた: " + ", ".join(
            f"{name} {100.0 * sum(r['opp_hit'] for r in sel) / max(len(sel), 1):5.1f}%"
            for name, sel in (("全体", rows), ("奪取のみ", stolen), ("目標が敵側 (<60°)", [r for r in rows if r["tgt_opp"] < 60.0]),
                              ("目標が敵側の奪取", [r for r in stolen if r["tgt_opp"] < 60.0]))))
        for lo, hi in ((0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 120.0), (120.0, 150.0), (150.0, 181.0)):
            sel = [r for r in rows if lo <= r["tgt_opp"] < hi]
            th = [r["touch_th"] for r in sel if r["touch_th"] == r["touch_th"]]
            print(f"    角度 [{lo:.0f},{hi:.0f}): ボールと敵の間に入った {100.0 * sum(r['between'] for r in sel) / max(len(sel), 1):5.1f}% (n={len(sel)})"
                  f" / 最初に触れたときの立ち位置 (0 = 敵側, 180 = 敵の反対側) 平均 {sum(th) / max(len(th), 1):5.1f}°、"
                  f"敵側 (<60°) {100.0 * sum(t < 60.0 for t in th) / max(len(th), 1):5.1f}% (n={len(th)})")
        th = [r["touch_th"] for r in rows if r["touch_th"] == r["touch_th"]]
        print(f"  最初に触れたときの立ち位置: 平均 {sum(th) / max(len(th), 1):5.1f}°、敵側 (<60°) {100.0 * sum(t < 60.0 for t in th) / max(len(th), 1):5.1f}% (n={len(th)})")
    if diag:
        d = torch.stack(diag).mean(0).tolist()
        print(f"\n  [diag_back] 背後スタートの開始 1 s: |cmd| mean vx {d[0]:.2f} vy {d[1]:.2f} wz {d[2]:.2f} / "
              f"cmd mean vx {d[3]:+.2f} vy {d[4]:+.2f} wz {d[5]:+.2f} / |residual| max mean {d[6]:.2f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
