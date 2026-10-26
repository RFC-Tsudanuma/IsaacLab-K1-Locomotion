# 使い方 (コンテナ内): python.sh scripts/rsl_rl/check_ball_friction.py --headless --device cuda:0 [--task <task>]
"""ボールの摩擦の効き方を確かめる。回転なしでボールを滑らせ、短い時間の回転の速さから摩擦係数を逆算する
(滑っている間: 角加速度 = 5 μ g / (2 r))。(1) 床の上、(2) 立っているロボットの胴体の箱の上面 (ロボットの材質との組み合わせ)。"""
import argparse, sys
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--task", default="Isaac-K1-BHuman-Walk-Kick-Stage2-v0")
AppLauncher.add_app_launcher_args(ap); a, rest = ap.parse_known_args(); sys.argv = [sys.argv[0]] + rest
app = AppLauncher(a).app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_k1_locomotion.tasks  # noqa
cfg = parse_env_cfg(a.task, device="cuda:0", num_envs=8)
env = gym.make(a.task, cfg=cfg)
u = env.unwrapped
ball, robot = u.scene["soccer_ball"], u.scene["robot"]
r = cfg.scene.soccer_ball.spawn.radius
n = u.num_envs
dt = u.sim.get_physics_dt()
bm = ball.root_physx_view.get_material_properties()
rm = robot.root_physx_view.get_material_properties()  # (N, shapes, 3)
trunk_shape = 0
print("robot shape0 (Trunk) dynamic friction:", [round(float(x), 2) for x in rm[:, trunk_shape, 1]], flush=True)
print("ball dynamic friction:", [round(float(x), 2) for x in bm[:, 0, 1]], flush=True)

def slide(pos, v, t):
    st = torch.zeros(n, 13, device=u.device); st[:, :3] = pos; st[:, 3] = 1.0; st[:, 7] = v
    ball.write_root_state_to_sim(st)
    steps = int(round(t / dt))
    for _ in range(steps):
        u.scene.write_data_to_sim(); u.sim.step(render=False); u.scene.update(dt)
    return ball.data.root_ang_vel_w[:, 1] * 2 * r / (5 * 9.81 * steps * dt)

pos = u.scene.env_origins.clone(); pos[:, 0] += 2.5; pos[:, 2] = r + 0.001
mu_floor = slide(pos, 3.0, 0.04)
top = robot.data.root_pos_w.clone(); top[:, 0] -= 0.03; top[:, 2] += 0.2 + r + 0.001
mu_trunk = slide(top, 1.0, 0.03)
for i in range(n):
    t = float(rm[i, trunk_shape, 1]); b = float(bm[i, 0, 1])
    print(f"env {i}: 床 μ≈{mu_floor[i]:.2f} | 胴体 μ≈{mu_trunk[i]:.2f} (胴体 {t:.2f}, ボール {b:.2f}: 平均なら {(t+b)/2:.2f}, 積なら {t*b:.2f})", flush=True)
env.close(); app.close()
