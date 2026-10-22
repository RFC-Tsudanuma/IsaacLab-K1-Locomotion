# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ドリブル (Isaac-Velocity-Flat-K1-Dribble-Target) のポリシーを、カリキュラムのレベルを指定して再生する。

カリキュラム (dribble_target_range) を指定したレベルに固定し (進級させない)、そのレベルの
目標距離・経路形状・障害物の環境で再生する。再生用 cfg (K1DribbleTargetEnvCfg_PLAY) を使うので、
観測ノイズ・外乱 (押し) は無効、カメラは env 0 のロボットを追従する。

    # 画面で見る (ウィンドウを閉じると終了)
    python scripts/rsl_rl/play_dribble_target.py --level 5
    # 動画を撮る (headless、<run>/videos/levels/level5-*.mp4)
    python scripts/rsl_rl/play_dribble_target.py --level 5 --video --video_length 1000

--checkpoint を省略すると logs/rsl_rl/k1_dribble_target の最新ランの最終 checkpoint を使う。
"""

import argparse
import glob
import os
import re
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--level", type=int, required=True, help="カリキュラムのレベル (1〜8)")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--num_envs", type=int, default=None, help="既定: 画面 16、動画 1")
parser.add_argument("--video", action="store_true", help="動画を撮って終了する")
parser.add_argument("--video_length", type=int, default=1000, help="動画の長さ [step] (50 step = 1 秒)")
parser.add_argument("--with_noise", action="store_true", help="学習時と同じ観測ノイズを入れる")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--no_real_time", action="store_true", help="画面再生を実時間に合わせない")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import isaaclab_k1_locomotion.tasks  # noqa: E402, F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_transition_utils  # noqa: E402

TASK = "Isaac-Velocity-Flat-K1-Dribble-Target-Play-v0"


def _latest_checkpoint() -> str:
    runs = sorted(glob.glob(os.path.join("logs", "rsl_rl", "k1_dribble_target", "20*")))
    if not runs:
        raise SystemExit("logs/rsl_rl/k1_dribble_target にランがありません。--checkpoint で指定してください")
    ckpts = glob.glob(os.path.join(runs[-1], "model_*.pt"))
    return max(ckpts, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))


def main():
    ckpt = os.path.abspath(args_cli.checkpoint or _latest_checkpoint())
    num_envs = args_cli.num_envs or (1 if args_cli.video else 16)
    env_cfg = parse_env_cfg(TASK, num_envs=num_envs)
    env_cfg.seed = args_cli.seed
    if args_cli.with_noise:
        env_cfg.observations.policy.enable_corruption = True
    # 再生中は進級させない (レベルは下で固定する)
    cur = env_cfg.curriculum.dribble_target_range
    stages = cur.params["stages"]
    if not 1 <= args_cli.level <= len(stages):
        raise SystemExit(f"--level は 1〜{len(stages)} で指定してください")
    cur.params["error_threshold"] = [-1.0] * len(stages)

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    u = env.unwrapped
    term = u.curriculum_manager._term_cfgs[u.curriculum_manager.active_terms.index("dribble_target_range")].func
    term.load_state_dict({"level": args_cli.level - 1, "failure_ema": 1.0, "num_updates": 0})
    print(f"[INFO] Level {args_cli.level}: {stages[args_cli.level - 1]}", flush=True)

    if args_cli.video:
        video_dir = os.path.join(os.path.dirname(ckpt), "videos", "levels")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix=f"level{args_cli.level}_{os.path.splitext(os.path.basename(ckpt))[0]}",
            disable_logger=True,
        )
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(
        wrapped, stage_transition_utils.filter_unsupported_alg_args(agent_cfg.to_dict()), log_dir=None, device="cuda:0"
    )
    runner.load(ckpt, load_optimizer=False)
    policy = runner.get_inference_policy(device="cuda:0")
    print(f"[INFO] checkpoint: {ckpt}", flush=True)

    # レベルを固定してからリセットし、そのレベルの目標・経路・障害物で始める
    obs, _ = wrapped.reset()
    obs = wrapped.get_observations()
    dt = u.step_dt
    step = 0
    successes = falls = episodes = 0
    with torch.no_grad():  # inference_mode だと env.reset の in-place 操作でエラーになる
        while simulation_app.is_running():
            t0 = time.time()
            obs, _, dones, _ = wrapped.step(policy(obs))
            step += 1
            if dones.any():
                tm = u.termination_manager
                episodes += int(dones.sum().item())
                successes += int(tm.get_term("ball_reached_target")[dones.bool()].sum().item())
                falls += int(tm.get_term("base_height")[dones.bool()].sum().item())
            if args_cli.video and step >= args_cli.video_length:
                break
            if not args_cli.video and not args_cli.no_real_time:
                time.sleep(max(0.0, dt - (time.time() - t0)))
    print(
        f"[RESULT] level={args_cli.level} steps={step} episodes={episodes} success={successes} fall={falls}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
