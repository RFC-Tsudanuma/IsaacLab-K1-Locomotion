"""奪取方策の教師-生徒蒸留 (DAgger、2026-11-23)。

教師 = 観測劣化なしの試合条件で PPO 学習した StealHist の checkpoint。actor 入力は真値の
``teacher_policy`` (69) + ``teacher_hist`` (30 × 8)。
生徒 = 同じ構造 (StealHistActorCritic) で、actor 入力は劣化した ``steal_policy`` + ``steal_hist``。
初期値は教師の重み。

生徒の決定的行動で環境を進め (DAgger: 生徒が訪れた状態に教師がラベルを付ける)、
同じ状態での教師の出力に生徒の出力を合わせる。損失は環境に渡す空間で取る:
MSE(コマンド (クランプ後)) + MSE(残差 (scale 後))。
RL の報酬を使わないので、劣化観測を PPO で直接学ぶと方策全体が慎重になる問題 (2026-11-22 の切り分け) を避ける。

保存は rsl_rl と同じ形式 (model_state_dict)。eval_steal.py --task Isaac-StealHist-K1-Play-v0 でそのまま評価できる。

    uv run distill_steal.py --headless --teacher <t1/model_N.pt> --run_name s1 --num_envs 2048 --iterations 1500
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Steal teacher-student distillation (DAgger).")
parser.add_argument("--task", type=str, default="Isaac-StealStudent-K1-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--teacher", type=str, required=True, help="教師の checkpoint (StealHist、劣化なしで学習)")
parser.add_argument("--init", type=str, default=None, help="生徒の初期値 (既定は教師と同じ)")
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--iterations", type=int, default=1500)
parser.add_argument("--steps_per_iter", type=int, default=24)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--mini_batches", type=int, default=4)
parser.add_argument("--lr", type=float, default=3.0e-4)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--clean_weight", type=float, default=1.0, help="劣化なしエピソードの step の損失の重み (劣化ありは 1)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_k1_locomotion.tasks  # noqa: F401, E402
from isaaclab_k1_locomotion.tasks.manager_based.locomotion.agents.steal_policy import StealHistActorCritic  # noqa: E402

STUDENT_GROUPS = {"policy": ["steal_policy", "steal_hist"], "critic": ["steal_critic", "steal_hist_true"]}
TEACHER_GROUPS = {"policy": ["teacher_policy", "teacher_hist"], "critic": ["steal_critic", "steal_hist_true"]}


def load(policy, path, device):
    sd = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    policy.load_state_dict(sd)
    return policy


def env_space(policy, raw):
    """方策の生出力 (15) → 環境に渡す空間 [コマンド (クランプ・デッドゾーン・膝), 残差 (scale 後)]。"""
    return policy.command(raw[:, :3]), policy.residual_scale * raw[:, 3:]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    dev = args_cli.device or "cuda:0"
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = dev
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    obs = env.get_observations().to(dev)

    pcfg = agent_cfg.policy.to_dict()
    pcfg.pop("class_name", None)
    teacher = load(StealHistActorCritic(obs, TEACHER_GROUPS, env.num_actions, **copy.deepcopy(pcfg)).to(dev), args_cli.teacher, dev)
    student = load(StealHistActorCritic(obs, STUDENT_GROUPS, env.num_actions, **copy.deepcopy(pcfg)).to(dev), args_cli.init or args_cli.teacher, dev)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    # 観測の正規化は教師の統計で固定 (更新しない)。学習するのは生徒の actor (encoder + MLP) だけ
    params = list(student.actor.parameters())
    opt = torch.optim.Adam(params, lr=args_cli.lr)

    log_dir = os.path.join("logs", "rsl_rl", "k1_steal_hist", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args_cli.run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[distill] teacher {args_cli.teacher} -> {log_dir}")
    raw_env = env.unwrapped
    tm = raw_env.termination_manager

    def save(it):
        torch.save({"model_state_dict": student.state_dict(), "optimizer_state_dict": opt.state_dict(), "iter": it, "infos": None},
                   os.path.join(log_dir, f"model_{it}.pt"))

    for it in range(args_cli.iterations):
        t0 = time.time()
        xs, cmd_t, res_t, cl = [], [], [], []
        term = raw_env.command_manager.get_term("steal")
        n_done = n_fell = 0
        with torch.no_grad():
            for _ in range(args_cli.steps_per_iter):
                x = student.get_actor_obs(obs)
                raw_s = student.actor(student.actor_obs_normalizer(x))
                raw_t = teacher.actor(teacher.actor_obs_normalizer(teacher.get_actor_obs(obs)))
                c, r = env_space(teacher, raw_t)
                xs.append(x)
                cl.append(~term.obs_degrade)
                cmd_t.append(c)
                res_t.append(r)
                obs, _, dones, _ = env.step(student.env_actions(obs, raw_s))
                obs = obs.to(dev)
                d = dones.bool()
                n_done += int(d.sum())
                n_fell += int((d & tm.get_term("base_height")).sum())
        t_col = time.time() - t0
        X, C, R, CL = torch.cat(xs), torch.cat(cmd_t), torch.cat(res_t), torch.cat(cl)
        W = torch.where(CL, args_cli.clean_weight, 1.0)
        n = X.shape[0]
        tot = 0.0
        for _ in range(args_cli.epochs):
            perm = torch.randperm(n, device=dev)
            for idx in perm.chunk(args_cli.mini_batches):
                c, r = env_space(student, student.actor(student.actor_obs_normalizer(X[idx])))
                per = (c - C[idx]).pow(2).mean(1) + (r - R[idx]).pow(2).mean(1)
                loss = (W[idx] * per).sum() / W[idx].sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                tot += loss.item()
        loss_mean = tot / (args_cli.epochs * args_cli.mini_batches)
        with torch.no_grad():  # 更新後の誤差を劣化なし / ありに分けて (一部サンプルで)
            sub = torch.randperm(n, device=dev)[:8192]
            c, r = env_space(student, student.actor(student.actor_obs_normalizer(X[sub])))
            per = (c - C[sub]).pow(2).mean(1) + (r - R[sub]).pow(2).mean(1)
            e_cl, e_dg = per[CL[sub]].mean().item(), per[~CL[sub]].mean().item()
        print(f"[distill] it {it:5d}  loss {loss_mean:.5f}  clean {e_cl:.5f}  degraded {e_dg:.5f}  done {n_done:5d}  fell {n_fell:4d} ({100.0 * n_fell / max(n_done, 1):.1f}%)"
              f"  collect {t_col:.1f}s  total {time.time() - t0:.1f}s", flush=True)
        if (it + 1) % args_cli.save_interval == 0:
            save(it + 1)
    save(args_cli.iterations)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
