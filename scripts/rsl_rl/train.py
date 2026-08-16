# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--reset_noise_std",
    type=float,
    default=None,
    help="If set, clamp the policy action-noise std to this minimum after loading a checkpoint (resume only).",
)
parser.add_argument(
    "--reset_obs_normalizer",
    action="store_true",
    default=False,
    help=(
        "Reset the observation-normalizer running statistics (mean/var/count) to their initial "
        "state after loading a checkpoint. Use when resuming across a stage boundary that changes "
        "what the observation slots mean (e.g. gk Stage1 -> Stage2, where the task slots go from "
        "zeros_obs dummies to real values). See the block at the load site for why."
    ),
)
parser.add_argument(
    "--warmstart_actor",
    type=str,
    default=None,
    help=(
        "Path to a walking checkpoint (e.g. logs/rsl_rl/k1_flat/main_walk/0524_walk.pt). "
        "Partially initializes the actor (first-layer column mapping + obs-normalizer stats) "
        "for tasks whose observation extends the walking layout (see scripts/rsl_rl/warmstart.py). "
        "Ignored when --resume is set."
    ),
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--override_json",
    type=str,
    default=None,
    help="JSON file with dot-path overrides for env_cfg / agent_cfg (used by Optuna tuning).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import isaaclab_k1_locomotion.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # apply Optuna-style JSON overrides last so they win over Hydra/CLI defaults
    if args_cli.override_json is not None:
        from config_overrides import apply_overrides_from_file

        apply_overrides_from_file(args_cli.override_json, env_cfg=env_cfg, agent_cfg=agent_cfg)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # print joint order
    try:
        print("[INFO] K1 joint order (USD native):")
        robot = env.unwrapped.scene["robot"]
        for i, name in enumerate(robot.joint_names):
            print(f"  {i:2d}: {name}")
        print("[INFO] K1 joint order (action space):")
        action_term = env.unwrapped.action_manager._terms["joint_pos"]
        for i, name in enumerate(action_term._joint_names):
            print(f"  {i:2d}: {name}")
    except Exception as e:
        print(f"[WARNING] Could not retrieve joint names: {e}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        # --checkpoint に実在するファイルパスが渡された場合はそれを直接使う。
        # これにより load_run を介さず、別 experiment / 任意の場所にある .pt からでも
        # 追加学習を開始できる (追加学習で checkpoint パスだけ指定したいケース)。
        # ファイルが見つからない場合は従来通り logs/rsl_rl/<experiment_name>/<load_run>/
        # 配下を load_run・load_checkpoint の正規表現で解決する (後方互換)。
        if agent_cfg.load_checkpoint and os.path.isfile(agent_cfg.load_checkpoint):
            resume_path = os.path.abspath(agent_cfg.load_checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # --- ゴールキーパーのカリキュラム進捗の永続化パスを env に渡す (2026-08-16) ---
    # ★ RslRlVecEnvWrapper より **前** に設定すること。wrapper の __init__ が env.reset()
    #   を呼び、そこで CurriculumManager が走って adaptive_difficulty が初期化される。
    #   その時点でパスが無いと保存済みの進捗を読めず、最易段から始まってしまう。
    #
    # rsl_rl の save() はモデル・オプティマイザ・iter しか保存しないため、ゴールキーパーの
    # カリキュラム到達点 (ball_speed_hi / aim_stage) は goalkeeper/mdp/curriculums.py が
    # curriculum_state.json として自前で永続化する。これが無いと --resume のたびに
    # 最易段 (初速 1.0 / 狙い先 ±0.4) へ巻き戻る。
    #   load: resume 元のランディレクトリ / save: 今回のランディレクトリ
    # goalkeeper 以外のタスクでは curriculums.py 側が参照しないので無害。
    if hasattr(env.unwrapped.cfg, "goalkeeper"):
        _curr_load_dir = os.path.dirname(resume_path) if (
            agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation"
        ) else None
        env.unwrapped._gk_curriculum_paths = {"load": _curr_load_dir, "save": log_dir}

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # When resetting std, also drop the optimizer state. The saved Adam moments for the std
        # parameter carry strong "push std down" momentum that immediately drives std negative on
        # the first update, regardless of any post-load clamp.
        load_optimizer = args_cli.reset_noise_std is None
        runner.load(resume_path, load_optimizer=load_optimizer)
        if not load_optimizer:
            print("[INFO]: Skipped optimizer state load (--reset_noise_std set).")

        policy = runner.alg.policy

        # Force-load workaround: runner.load() can silently no-op for some checkpoints
        # (see PLAY_LOAD_ISSUE.md). Detect and force-load if needed.
        ckpt = torch.load(resume_path, weights_only=False, map_location=agent_cfg.device)
        ckpt_msd = ckpt["model_state_dict"]
        live_msd = policy.state_dict()
        mismatched = [
            k for k in ckpt_msd if k in live_msd and not torch.equal(live_msd[k].cpu(), ckpt_msd[k].cpu())
        ]
        if mismatched:
            print(f"[WARN]: runner.load no-op detected; force-loading {len(mismatched)} mismatched keys.")
            policy.load_state_dict(ckpt_msd, strict=False)

        # --- 観測正規化の統計をリセットする (ステージ境界をまたぐ resume 用) ---
        #
        # ★ 2026-08-16: ゴールキーパー Stage1 -> Stage2 の resume で実害を確認した。
        #
        #   rsl_rl の EmpiricalNormalization は ckpt の model_state_dict に mean/var/count を
        #   含むので、--resume すると **前ステージの統計をそのまま引き継ぐ**。しかも
        #   ``until`` は既定 None で、学習の最後まで更新し続ける (更新率 rate = 今回の
        #   サンプル数 / 累積 count なので、累積が大きいほど新しいデータが効かなくなる)。
        #
        #   ゴールキーパー Stage1 はボール系スロット (ball_pos_rel / ball_vel / ball_active /
        #   target_y / self_state = obs[49:59]) が全て zeros_obs のダミーで、Stage2 で初めて
        #   実値が入る。したがって Stage2 の統計は「ゼロが大量に入ったあとに実値が乗る」
        #   二峰分布の平均になり、実値が不当に縮む。
        #
        #   実測 (k1_gk_direct_stage2/2026-08-15_11-31-55, model_48200):
        #     Stage1 count 5,406,720,000 -> Stage2 count 8,670,609,408
        #     Stage2 の寄与率 = 3,263,889,408 / 8,670,609,408 = 0.376
        #     cos(yaw) は常に ~1.0 のはずが mean = 0.3736  (予測 1.0 * 0.376 = 0.3764)
        #     self_x は guard_x=0.9 付近のはずが mean = 0.2994
        #   → タスク観測 8 スロットが一律 **約 0.38 倍に潰されて** 方策に入っていた。
        #
        #   重みの引き継ぎ (warmstart の目的) にダミーゼロの統計まで要らないので、
        #   ステージ境界では統計だけ捨てる。count=0 に戻せば Stage2 の実データで
        #   最初から推定し直す。
        #
        # ★ 重みは触らない。統計 (mean/var/std/count) だけをリセットする。
        # ★ opt-in。通常の「同一ステージの続きから」では **使わないこと** (せっかく貯めた
        #   統計を捨てて、序盤の観測スケールが暴れる)。
        if args_cli.reset_obs_normalizer:
            _n_reset = 0
            for _name, _mod in policy.named_modules():
                if not (hasattr(_mod, "_mean") and hasattr(_mod, "count")):
                    continue
                with torch.no_grad():
                    _mod._mean.zero_()
                    _mod._var.fill_(1.0)
                    _mod._std.fill_(1.0)
                    _mod.count.zero_()
                _n_reset += 1
                print(f"[INFO]: Reset obs-normalizer statistics: {_name}")
            if _n_reset == 0:
                print("[WARN]: --reset_obs_normalizer set but no normalizer module was found.")

        # re-inject action noise std if requested (recover from collapsed std after long training)
        if args_cli.reset_noise_std is not None:
            import math

            with torch.no_grad():
                if policy.noise_std_type == "scalar":
                    before = policy.std.data.clone()
                    policy.std.data.clamp_(min=args_cli.reset_noise_std)
                    print(
                        f"[INFO]: Clamped policy std to min={args_cli.reset_noise_std}\n"
                        f"        before: {before.tolist()}\n"
                        f"        after : {policy.std.data.tolist()}"
                    )
                elif policy.noise_std_type == "log":
                    log_floor = math.log(args_cli.reset_noise_std)
                    before = policy.log_std.data.exp().clone()
                    policy.log_std.data.clamp_(min=log_floor)
                    print(
                        f"[INFO]: Clamped policy std to min={args_cli.reset_noise_std} (log_std clamp)\n"
                        f"        before std: {before.tolist()}\n"
                        f"        after  std: {policy.log_std.data.exp().tolist()}"
                    )

    # 歩行 ckpt からの actor ウォームスタート (B-Human 方式 Stage 2 用)。
    # --resume (完全な再開) とは排他: resume が優先。
    if args_cli.warmstart_actor is not None:
        if agent_cfg.resume:
            print("[WARN] --warmstart_actor は --resume と併用できないため無視します。")
        else:
            from warmstart import warmstart_actor_from_checkpoint

            warmstart_actor_from_checkpoint(runner.alg.policy, args_cli.warmstart_actor)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
