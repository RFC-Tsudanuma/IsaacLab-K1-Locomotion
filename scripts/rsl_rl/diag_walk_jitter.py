# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""歩行ポリシーの **停止指令時の震え** を測る (obs 49 / locomotion タスク用)。

実機で walk_0524 は実用され、07-28 以降は振動している。両者は同じ 12 関節直接制御・
同じデプロイスタックなので、差が学習側にあるのかを切り分けるために、シムで同一条件の
数値を取る。

指令はゼロ (立っているべき状態) に固定し、以下を測る:
    ||Δaction||   1階差分。指令の 1 ステップ変化
    2階差分        高周波成分。滑らかな動きと区別するため

使い方 (コンテナ内・リポジトリ直下):
    /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/diag_walk_jitter.py \\
        --checkpoint logs/rsl_rl/k1_flat/main_walk/0524_walk.pt \\
        --num_envs 16 --steps 900 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Measure standstill jitter of a walk policy.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-K1-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--warmup", type=int, default=300)
parser.add_argument("--override_json", type=str, default=None)
parser.add_argument(
    "--cmd", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"),
    help=(
        "Velocity command held constant during the measurement. Default (0,0,0) = standstill. "
        "★ 2026-08-21 追加。それまで停止指令固定で、**移動中の高周波を測る手段が無かった**。"
        "振動対策の比ペナルティ (action_jitter_ratio) は移動中に効く項なので、"
        "同じ指標を移動中にも取れないと効果が判定できない。例: '--cmd 0 1.3 0' で横移動。"
    ),
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.override_json is not None:
        from config_overrides import apply_overrides_from_file
        apply_overrides_from_file(args_cli.override_json, env_cfg=env_cfg, agent_cfg=agent_cfg)

    # ★ 指令を --cmd に固定する。レンジを 1 点に潰し、リサンプルも止める。
    #   既定 (0,0,0) は従来どおり「立っているべき状態」。
    _cvx, _cvy, _cwz = [float(v) for v in args_cli.cmd]
    _standing = (abs(_cvx) + abs(_cvy) + abs(_cwz)) < 1e-6
    r = env_cfg.commands.base_velocity.ranges
    r.lin_vel_x = (_cvx, _cvx)
    r.lin_vel_y = (_cvy, _cvy)
    r.ang_vel_z = (_cwz, _cwz)
    if hasattr(r, "heading"):
        r.heading = (0.0, 0.0)
    # ☠ standing env は指令を強制的にゼロにするので、移動を測るときは 0.0 にすること。
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0 if _standing else 0.0
    # ☠ heading_command=True だと wz が heading 誤差のフィードバックで上書きされる。
    #   固定指令の意味が壊れるので切る。
    if getattr(env_cfg.commands.base_velocity, "heading_command", False):
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        if hasattr(r, "heading"):
            r.heading = None
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    # ☠ 2026-08-21: 指令レンジのカリキュラムを切る。
    #
    #   K1FlatEnvCfg には lin_vel_command カリキュラムがあり、**毎エピソード
    #   ranges.lin_vel_x/y をステージ値で上書きする**。上でレンジを 1 点に潰しても
    #   カリキュラムが元に戻すので --cmd が一切効かず、指令を変えても数値が
    #   1 ビットも変わらない (実測: vy=0.66 / 1.0 / 1.3 で完全同一)。
    #   停止指令の測定は rel_standing_envs=1.0 が指令を強制ゼロにするため
    #   この不具合の影響を受けない (過去の測定結果は有効)。
    if getattr(getattr(env_cfg, "curriculum", None), "lin_vel_command", None) is not None:
        env_cfg.curriculum.lin_vel_command = None
        print("[cfg] lin_vel_command カリキュラムを無効化しました (指令固定のため)")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else \
        get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    inner_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(inner_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # ★ 古い ckpt は critic の観測次元が違う (後から ZMP を足したため)。推論に critic は
    #   要らないので strict=False で actor だけ読む。actor の形が合わなければここで落ちる。
    ckpt = torch.load(resume_path, weights_only=False, map_location=agent_cfg.device)
    # strict=False は「キーが無い」を許すだけで、**形状の不一致は許さない**。
    # critic は推論に使わないので、キーごと落としてから読む。
    msd = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("critic")}
    # ★ rsl_rl の ActorCritic.load_state_dict は bool を返す (nn.Module と規約が違う)。
    runner.alg.policy.load_state_dict(msd, strict=False)
    # 実際に載ったかを重みの一致で確かめる。黙って初期値のまま走るのが一番危ない。
    w_ck = msd["actor.0.weight"].to(agent_cfg.device)
    w_live = runner.alg.policy.state_dict()["actor.0.weight"]
    if not torch.allclose(w_ck, w_live):
        raise RuntimeError("actor の重みが ckpt と一致しません (読み込み失敗)")
    print("[load] actor OK (critic はスキップ)")
    policy = runner.get_inference_policy(device=inner_env.unwrapped.device)
    raw = inner_env.unwrapped
    robot = raw.scene["robot"]

    d1, d2, spd = [], [], []
    # ★ 2026-08-22 追加: **機体側の実挙動**も測る。
    #   それまで `action` (方策の出力) しか見ておらず、「指令は震えているが PD と
    #   armature が吸収して関節は滑らか」という可能性を排除できていなかった。
    #   その場合、指令の平滑度をいくら追っても実機とはズレたものを最適化している。
    #   吸収率 = 関節の2階差分 / 指令の2階差分 が小さければ、シムが吸収している証拠。
    q1, q2, qv2, w2 = [], [], [], []
    obs = inner_env.get_observations()
    prev = prev2 = None
    qprev = qprev2 = None
    qvprev = qvprev2 = None
    wprev = wprev2 = None
    with torch.inference_mode():
        for i in range(args_cli.steps + args_cli.warmup):
            a = policy(obs)
            if i >= args_cli.warmup and prev is not None:
                d1.append(torch.norm(a - prev, dim=1).clone())
                if prev2 is not None:
                    d2.append(torch.norm(a - 2.0 * prev + prev2, dim=1).clone())
                spd.append(torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1).clone())
            # --- 機体側 (実際の関節・胴体) ---
            q = robot.data.joint_pos.clone()
            qv = robot.data.joint_vel.clone()
            wz = robot.data.root_ang_vel_w.clone()
            if i >= args_cli.warmup and qprev is not None:
                q1.append(torch.norm(q - qprev, dim=1))
                if qprev2 is not None:
                    q2.append(torch.norm(q - 2.0 * qprev + qprev2, dim=1))
                    qv2.append(torch.norm(qv - 2.0 * qvprev + qvprev2, dim=1))
                    w2.append(torch.norm(wz - 2.0 * wprev + wprev2, dim=1))
            qprev2 = None if qprev is None else qprev.clone()
            qprev = q
            qvprev2 = None if qvprev is None else qvprev.clone()
            qvprev = qv
            wprev2 = None if wprev is None else wprev.clone()
            wprev = wz
            prev2 = None if prev is None else prev.clone()
            prev = a.clone()
            obs, _, _, _ = inner_env.step(a)

    d1 = torch.stack(d1); d2 = torch.stack(d2); spd = torch.stack(spd)
    print("\n" + "=" * 62)
    print(f"checkpoint : {resume_path}")
    _cmd_s = f"({_cvx:g}, {_cvy:g}, {_cwz:g})" + ("  = 停止指令" if _standing else "")
    print(f"task       : {args_cli.task}")
    print(f"cmd        : {_cmd_s} 固定")
    print(f"sampled    : {d1.shape[0]} x {d1.shape[1]} env")
    print(f"  ベース水平速度   mean={spd.mean():.4f}  max={spd.max():.4f}")
    print(f"  ||Δaction||     mean={d1.mean():.4f}  p95={torch.quantile(d1.float(), 0.95):.4f}")
    print(f"  2階差分          mean={d2.mean():.4f}  p95={torch.quantile(d2.float(), 0.95):.4f}")
    # ★ 比 = 報酬 action_jitter_ratio が最適化している量そのもの。
    #   正弦波なら 2*sin(pi*f/fs) に等しいので、そこから支配周波数が逆算できる。
    _ratio = float(d2.mean() / d1.mean().clamp(min=1e-9))
    _s = min(max(_ratio / 2.0, -1.0), 1.0)
    _fs = 1.0 / float(raw.step_dt)
    _f = math.asin(_s) * _fs / math.pi
    print(f"  2階/1階の比      {_ratio:.4f}   → 支配周波数 約 {_f:.1f} Hz  (fs={_fs:.0f}Hz)")
    print(f"     参考: 1.6Hz→0.20 / 3.5Hz→0.44 / 5Hz→0.62 / 10Hz→1.17 / Nyquist→2.00")
    # --- 機体側 ---
    if q2:
        q1t = torch.stack(q1); q2t = torch.stack(q2)
        qv2t = torch.stack(qv2); w2t = torch.stack(w2)
        print("  --- 機体側 (実際に震えているか) ---")
        print(f"  関節角 1階差分   mean={q1t.mean():.5f}  [rad]")
        print(f"  関節角 2階差分   mean={q2t.mean():.5f}  p95={torch.quantile(q2t.float(), 0.95):.5f}  [rad]")
        print(f"  関節速度 2階差分 mean={qv2t.mean():.4f}  [rad/s]")
        print(f"  胴体角速度 2階差分 mean={w2t.mean():.5f}  [rad/s]")
        # ★ 吸収率: 指令の高周波がどれだけ関節に出ているか。
        #   1 に近い = 指令どおり関節も震えている (シムで振動が再現できている)。
        #   0 に近い = PD/armature が吸収 = **シムでは震えていない** →
        #              指令の平滑度を追っても実機とはズレたものを最適化している。
        #   ☠ action は scale 0.5 の関節目標角なので、指令側を 0.5 倍して単位を揃える。
        absorb = float(q2t.mean() / (d2.mean() * 0.5).clamp(min=1e-9))
        print(f"  ★ 吸収率 (関節2階差分 / 指令2階差分) = {absorb:.3f}")
        print("     1 に近い = 指令どおり機体も震えている / 0 に近い = PD が吸収 = シムでは震えていない")
    print("=" * 62 + "\n")
    inner_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
