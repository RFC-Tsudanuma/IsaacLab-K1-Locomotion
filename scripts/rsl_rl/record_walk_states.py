# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""歩行ポリシーの「歩いている最中の状態」をサンプリングして npz に落とす。

用途
----
walk (model_34995.pt) と walk_kick で歩容が違うため、実機で walk → walk_kick に
切り替えた瞬間に姿勢が飛ぶ。これを消すには walk_kick 側を **walk ポリシーの歩行状態
から reset して学習** すればよい (Reference State Initialization)。そのための状態
プールを作るのがこのスクリプト。

``record_amp_rollout.py`` との違い
---------------------------------
あちらは AMP の参照モーション用に **連続クリップ** を録る。こちらは reset 用なので
連続性は不要で、多数の env × 時間方向の間引きで **無相関なスナップショット** を集める。

実行するブランチ
----------------
**model_34995.pt を学習したブランチ (feat/inoue_walk_double_encoder) で実行すること。**
このポリシーは観測が「command (3) + 49 次元 × 100 ステップ履歴」の dual-encoder 構成で、
そのブランチの ``K1FlatEnvCfg`` / ``HistoryActorCritic`` でしかロードできない。
別ブランチで走らせると入力層が形違いで捨てられ、**エラー無しでランダム方策の
「歩行データ」が録れてしまう** ので、下の ``_verify_loaded`` で必ず検算している。

使い方::

    # 学習元ブランチの worktree で
    _labpython2 scripts/rsl_rl/record_walk_states.py \
        --task Isaac-Velocity-Flat-K1-Play-v0 \
        --checkpoint /path/to/model_34995.pt \
        --num_envs 128 --record_time 60 --headless \
        --out walk_states.npz

出力 (npz, 1 行 = 1 サンプル)::

    root_height      (M,)     地面からの base 高さ [m]
    root_quat_wxyz   (M, 4)   yaw を抜いた姿勢 (roll/pitch のみ)。reset 時の yaw は
                              タスク側が自由に決めてよい
    root_lin_vel_b   (M, 3)   base フレームの線速度 [m/s]
    root_ang_vel_b   (M, 3)   base フレームの角速度 [rad/s]
    joint_pos        (M, 12)  **絶対角** [rad] (default 差分ではない)
    joint_vel        (M, 12)  [rad/s]
    last_action      (M, 12)  そのステップでポリシーが出した action (obs の
                              prev_joint_request 相当)。並びは action_joint_names
    command          (M, 3)   その時の速度コマンド (vx, vy, wz)
    gait_phase       (M,)     左脚の歩行位相 [rad, 0..2π)
    phase_freq       (M,)     その env の歩行周波数 [Hz]
    joint_names      (12,)    joint_pos / joint_vel の並び (articulation 順)
    action_joint_names (12,)  last_action の並び
    meta             (json 文字列) task / checkpoint / dt など
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="歩行ポリシーの歩行中状態をサンプリングする。")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-K1-Play-v0", help="タスク名。")
parser.add_argument("--num_envs", type=int, default=128, help="同時に歩かせる env 数。")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent cfg entry point。")
parser.add_argument("--seed", type=int, default=None, help="乱数シード。")
parser.add_argument("--settle_time", type=float, default=5.0,
                    help="歩容が定常になるまで捨てる時間 [s]。")
parser.add_argument("--record_time", type=float, default=60.0, help="収録時間 [s]。")
parser.add_argument("--stride", type=int, default=5,
                    help="何ステップおきにスナップショットを取るか (dt=0.02 なら 5 で 0.1 s ごと)。")
parser.add_argument("--min_steps_after_reset", type=int, default=100,
                    help="リセット直後は歩容が立ち上がっていないので、この step 数未満の env は捨てる。")
parser.add_argument("--min_cmd_speed", type=float, default=0.1,
                    help="||cmd|| がこれ未満の env (立ち止まり) は捨てる。0 で無効。")
parser.add_argument("--max_samples", type=int, default=200000, help="収集する最大サンプル数。")
parser.add_argument("--vx_range", type=float, nargs=2, default=(0.0, 1.0), help="vx コマンド範囲 [m/s]。")
parser.add_argument("--vy_range", type=float, nargs=2, default=(-0.4, 0.4), help="vy コマンド範囲 [m/s]。")
parser.add_argument("--wz_range", type=float, nargs=2, default=(-1.0, 1.0), help="wz コマンド範囲 [rad/s]。")
parser.add_argument("--resampling_time", type=float, nargs=2, default=(2.0, 5.0),
                    help="コマンド再サンプル間隔 [s]。")
parser.add_argument("--allow_partial_load", action="store_true",
                    help="checkpoint の重みが一部しか載らなくても続行する (デバッグ用)。")
parser.add_argument("--out", type=str, default="walk_states.npz", help="出力 npz のパス。")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_conjugate, quat_mul, yaw_quat

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401


def _policy_module(runner):
    """rsl_rl のバージョン差 (policy / actor_critic) を吸収する。"""
    try:
        return runner.alg.policy
    except AttributeError:
        return runner.alg.actor_critic


def _verify_loaded(policy_nn, resume_path: str, allow_partial: bool) -> None:
    """checkpoint の重みが本当に載ったかを検算する。

    ``load_state_dict(..., strict=False)`` 系の経路は、観測次元やネットワーク構成が
    違う checkpoint を渡しても **例外を出さずに黙って捨てる**。そのまま収録すると
    ランダム方策の「歩行データ」が出来上がるので、ここで必ず突き合わせる。
    """
    raw = torch.load(resume_path, map_location="cpu", weights_only=False)
    ckpt = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"checkpoint の中身が dict ではない: {type(ckpt)}")

    live = policy_nn.state_dict()
    missing, mismatched, worst = [], [], 0.0
    for k, v in ckpt.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k not in live:
            missing.append(k)
            continue
        if tuple(live[k].shape) != tuple(v.shape):
            mismatched.append(f"{k}: env={tuple(live[k].shape)} ckpt={tuple(v.shape)}")
            continue
        worst = max(worst, float((live[k].detach().cpu().float() - v.float()).abs().max()))

    if missing or mismatched or worst > 1e-6:
        msg = ["checkpoint がポリシーに載っていない:"]
        if missing:
            msg.append(f"  ロードされなかったキー ({len(missing)}): {missing[:8]}")
        if mismatched:
            msg.append(f"  形が違うキー ({len(mismatched)}): {mismatched[:8]}")
        if worst > 1e-6:
            msg.append(f"  値が一致しない (max|Δ| = {worst:.3g})")
        msg.append("  → この checkpoint を学習したブランチ (観測構成・ネットワーク) で実行すること。")
        text = "\n".join(msg)
        if not allow_partial:
            raise RuntimeError(text)
        print(f"[WARN] {text}")
    else:
        print(f"[INFO] checkpoint 検算 OK: {len(ckpt)} キーが一致 (max|Δ| = {worst:.3g})")


def _gait_phase(env) -> torch.Tensor | None:
    """左脚の歩行位相 [rad] を env から取り出す。取れなければ None。

    - dual-encoder 系 (feat/inoue_walk_double_encoder): 位相は毎ステップ積分され
      ``env._gait_phase_left`` に入っている。
    - 旧構成: 位相は ``2π * pf * t + offset`` の閉じた式なので、その場で組み立てる。
    """
    phase = getattr(env, "_gait_phase_left", None)
    if phase is not None:
        return phase.clone()

    pf = getattr(env, "_phase_freq_per_env", None)
    if pf is None:
        return None
    offset = getattr(env, "_phase_offset_per_env", 0.0)
    t = env.episode_length_buf * env.step_dt
    return torch.remainder(2.0 * math.pi * pf * t + offset, 2.0 * math.pi)


def _phase_freq(env) -> torch.Tensor | None:
    """その env の歩行周波数 [Hz]。コマンド依存の構成なら現在のコマンドでの値。"""
    try:
        from isaaclab_k1_locomotion.tasks.manager_based.locomotion.mdp.events import (
            compute_cmd_phase_freq,
        )
    except ImportError:
        pass
    else:
        return compute_cmd_phase_freq(env)

    pf = getattr(env, "_phase_freq_per_env", None)
    return None if pf is None else pf.clone()


def _action_joint_names(env, fallback: list[str]) -> list[str]:
    """action 項の関節並び。取れなければ articulation 順で代用する。"""
    try:
        term = env.action_manager.get_term("joint_pos")
    except Exception:
        return fallback
    for attr in ("_joint_names", "joint_names"):
        names = getattr(term, attr, None)
        if names is not None:
            return list(names)
    return fallback


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    # 収録中にエピソードが切れると、そのぶん立ち上がり待ちが増えるだけなので長くする。
    env_cfg.episode_length_s = max(
        float(env_cfg.episode_length_s), args_cli.settle_time + args_cli.record_time + 5.0
    )

    # コマンド分布: walk_kick の接近フェーズで実際に出る範囲に寄せる。
    # 立ち止まり (rel_standing_envs) は歩行状態が欲しいので切る。
    try:
        cmd = env_cfg.commands.base_velocity
        cmd.ranges.lin_vel_x = tuple(args_cli.vx_range)
        cmd.ranges.lin_vel_y = tuple(args_cli.vy_range)
        cmd.ranges.ang_vel_z = tuple(args_cli.wz_range)
        cmd.resampling_time_range = tuple(args_cli.resampling_time)
        cmd.rel_standing_envs = 0.0
        cmd.heading_command = False
    except AttributeError as exc:
        print(f"[WARN] コマンド設定を上書きできなかった: {exc}")

    # 収録を汚す外乱は切る (PLAY cfg では既に None のことが多い)。
    for ev in ("push_robot", "base_external_force_torque"):
        if getattr(env_cfg.events, ev, None) is not None:
            setattr(env_cfg.events, ev, None)

    # コマンド系のカリキュラムは iteration 0 相当の狭い range に戻してしまうので無効化する。
    curriculum = getattr(env_cfg, "curriculum", None)
    for name in list(vars(curriculum)) if curriculum is not None else []:
        if "command" in name and getattr(curriculum, name, None) is not None:
            setattr(curriculum, name, None)
            print(f"[INFO] curriculum '{name}' を無効化した (コマンド範囲を固定するため)")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy_nn = _policy_module(runner)
    _verify_loaded(policy_nn, resume_path, args_cli.allow_partial_load)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.scene["robot"]
    joint_names = list(robot.joint_names)
    action_names = _action_joint_names(unwrapped, joint_names)
    env_origins = unwrapped.scene.env_origins  # (N, 3)
    dt = unwrapped.step_dt
    settle_steps = int(args_cli.settle_time / dt)
    record_steps = int(args_cli.record_time / dt)

    print(f"[INFO] dt={dt:.4f}s envs={unwrapped.num_envs} settle={settle_steps} rec={record_steps} "
          f"stride={args_cli.stride}")
    print(f"[INFO] joint order (articulation): {joint_names}")
    print(f"[INFO] joint order (action term) : {action_names}")

    obs = env.get_observations()
    if hasattr(policy_nn, "reset"):
        policy_nn.reset(torch.ones(unwrapped.num_envs, dtype=torch.bool, device=device))

    for _ in range(settle_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

    buckets: dict[str, list[np.ndarray]] = {
        k: [] for k in ("root_height", "root_quat_wxyz", "root_lin_vel_b", "root_ang_vel_b",
                        "joint_pos", "joint_vel", "last_action", "command", "gait_phase", "phase_freq")
    }
    n_samples = 0

    for t in range(record_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        if t % args_cli.stride != 0:
            continue

        # 採用条件: リセット直後でない / 今このステップで終了していない / 歩いている
        keep = unwrapped.episode_length_buf >= args_cli.min_steps_after_reset
        keep &= ~dones.reshape(-1).bool()
        command = unwrapped.command_manager.get_command("base_velocity")
        if args_cli.min_cmd_speed > 0.0:
            keep &= torch.norm(command[:, :3], dim=1) >= args_cli.min_cmd_speed
        idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue

        quat_w = robot.data.root_quat_w[idx]                    # (n, 4) wxyz
        # yaw を抜いた姿勢。reset 時の yaw はタスク側が決めるので roll/pitch だけ残す。
        quat_noyaw = quat_mul(quat_conjugate(yaw_quat(quat_w)), quat_w)

        phase = _gait_phase(unwrapped)
        freq = _phase_freq(unwrapped)
        nan = torch.full((idx.numel(),), float("nan"), device=device)

        cap = {
            "root_height": robot.data.root_pos_w[idx, 2] - env_origins[idx, 2],
            "root_quat_wxyz": quat_noyaw,
            "root_lin_vel_b": robot.data.root_lin_vel_b[idx],
            "root_ang_vel_b": robot.data.root_ang_vel_b[idx],
            "joint_pos": robot.data.joint_pos[idx],
            "joint_vel": robot.data.joint_vel[idx],
            "last_action": unwrapped.action_manager.action[idx],
            "command": command[idx, :3],
            "gait_phase": nan if phase is None else phase[idx],
            "phase_freq": nan if freq is None else freq[idx],
        }
        for k, v in cap.items():
            buckets[k].append(v.detach().cpu().numpy().astype(np.float32))
        n_samples += idx.numel()

        if n_samples >= args_cli.max_samples:
            print(f"[INFO] max_samples ({args_cli.max_samples}) に到達したので収録を打ち切る。")
            break

    if n_samples == 0:
        raise RuntimeError(
            "サンプルが 1 つも採れなかった。--min_steps_after_reset を下げるか "
            "(転倒し続けている可能性)、--min_cmd_speed を下げること。"
        )

    out = {k: np.concatenate(v, axis=0) for k, v in buckets.items()}
    out["joint_names"] = np.array(joint_names)
    out["action_joint_names"] = np.array(action_names)
    out["meta"] = np.array(json.dumps({
        "task": args_cli.task,
        "checkpoint": resume_path,
        "dt": float(dt),
        "num_envs": int(unwrapped.num_envs),
        "stride": int(args_cli.stride),
        "settle_time": float(args_cli.settle_time),
        "record_time": float(args_cli.record_time),
        "vx_range": list(args_cli.vx_range),
        "vy_range": list(args_cli.vy_range),
        "wz_range": list(args_cli.wz_range),
        "min_cmd_speed": float(args_cli.min_cmd_speed),
    }))

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)) or ".", exist_ok=True)
    np.savez_compressed(args_cli.out, **out)

    h = out["root_height"]
    print(f"[SAVE] {args_cli.out}: {n_samples} サンプル")
    print(f"       root_height  mean={h.mean():.3f} min={h.min():.3f} max={h.max():.3f}")
    print(f"       |cmd_vx|     mean={np.abs(out['command'][:, 0]).mean():.3f}")
    print(f"       gait_phase   {'なし (NaN)' if np.isnan(out['gait_phase']).all() else 'あり'}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
