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
    "--load_pretrained", type=str, default=None,
    help="Path to a pretrained checkpoint (.pt) to initialize weights (strict=False, for transfer learning)."
)
parser.add_argument(
    "--warm_start_from_single_frame",
    action="store_true",
    default=False,
    help="Treat --load_pretrained as a 1-frame-observation checkpoint and graft its actor onto the "
    "history-input actor (ActorCriticHistoryCNN). The policy starts out behaving exactly like the "
    "pretrained one and learns to use the history from there. Without this flag the actor weights "
    "are silently dropped (shape/name mismatch).",
)
parser.add_argument(
    "--allow_untransferred_actor",
    action="store_true",
    default=False,
    help="Allow --load_pretrained to proceed even when no actor tensor could be transferred "
    "(the actor stays randomly initialized). Off by default because a silently random actor "
    "looks like a normal run in the logs but restarts locomotion from scratch, which quietly "
    "invalidates every fine-tuning curriculum. Only pass this when critic-only transfer is intended.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--reset_noise_std",
    type=float,
    default=None,
    help="If set, clamp the policy action-noise std to this minimum after loading a checkpoint "
    "(--resume or --load_pretrained).",
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name in {"DirectKickingOnPolicyRunner", "WalkKickLikelihoodOnPolicyRunner"}:
        from isaaclab_k1_locomotion.tasks.manager_based.walk_kick_likelihood.agents.runner import (
            DirectKickingOnPolicyRunner,
        )

        runner = DirectKickingOnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
        )
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
        if "model_state_dict" in ckpt:
            ckpt_msd = ckpt["model_state_dict"]
        elif agent_cfg.class_name in {
            "DirectKickingOnPolicyRunner",
            "WalkKickLikelihoodOnPolicyRunner",
        } and "model" in ckpt:
            expected_metadata = policy.checkpoint_metadata()
            actual_metadata = ckpt.get("model_metadata")
            if actual_metadata != expected_metadata:
                raise ValueError(
                    "DirectKicking checkpoint model_metadata does not match the configured policy. "
                    f"Expected {expected_metadata}, got {actual_metadata}"
                )
            ckpt_msd = ckpt["model"]
        else:
            raise KeyError("Checkpoint does not contain model_state_dict")
        live_msd = policy.state_dict()
        mismatched = [
            k for k in ckpt_msd if k in live_msd and not torch.equal(live_msd[k].cpu(), ckpt_msd[k].cpu())
        ]
        if mismatched:
            print(f"[WARN]: runner.load no-op detected; force-loading {len(mismatched)} mismatched keys.")
            policy.load_state_dict(ckpt_msd, strict=False)

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

        # sync common_step_counter so curriculum resumes from the correct phase
        if agent_cfg.resume:
            synced_steps = runner.current_learning_iteration * runner.cfg["num_steps_per_env"]
            env.unwrapped.common_step_counter = synced_steps
            print(f"[INFO]: Set common_step_counter to {synced_steps} (iteration {runner.current_learning_iteration})")
    # transfer learning: load pretrained weights with strict=False (observation dim may differ)
    elif args_cli.load_pretrained is not None:
        pretrained_path = os.path.abspath(args_cli.load_pretrained)
        print(f"[INFO]: Loading pretrained weights (strict=False) from: {pretrained_path}")
        loaded = torch.load(pretrained_path, map_location=agent_cfg.device)

        policy = getattr(runner.alg, 'actor_critic', None) or getattr(runner.alg, 'policy', None)
        if (
            agent_cfg.class_name
            in {"DirectKickingOnPolicyRunner", "WalkKickLikelihoodOnPolicyRunner"}
            and isinstance(loaded, dict)
            and "model" in loaded
        ):
            expected_metadata = policy.checkpoint_metadata()
            actual_metadata = loaded.get("model_metadata")
            if actual_metadata != expected_metadata:
                raise ValueError(
                    "DirectKicking checkpoint model_metadata does not match the configured policy. "
                    f"Expected {expected_metadata}, got {actual_metadata}"
                )
            state_dict = loaded["model"]
        else:
            state_dict = loaded.get("model_state_dict", loaded)

        # 1 フレーム観測の checkpoint を履歴入力の actor へ移植する。
        #
        # 素の ActorCritic の actor は名前 (actor.0.* vs actor.mlp.0.*) も 1 層目の入力次元も
        # 違うので、何もしないと下のフィルタに全部捨てられ、actor だけゼロから学習になる。
        # 移植すると初期状態の出力が旧ポリシーと一致するので、歩き方・蹴り方を保ったまま
        # 履歴の使い方だけを追加で学習できる (詳細は remap_single_frame_actor の docstring)。
        if args_cli.warm_start_from_single_frame:
            from isaaclab_k1_locomotion.tasks.manager_based.locomotion.networks import (
                ActorCriticHistoryCNN,
                remap_single_frame_actor,
            )

            if not isinstance(policy, ActorCriticHistoryCNN):
                raise ValueError(
                    "--warm_start_from_single_frame は履歴入力の policy (ActorCriticHistoryCNN) 専用です。"
                    f" 現在の policy: {type(policy).__name__}"
                )
            state_dict, notes = remap_single_frame_actor(state_dict, policy)
            if any("->" in note for note in notes):
                print("[INFO]: Grafting 1-frame actor onto the history actor:")
                for note in notes:
                    print(f"          {note}")
            else:
                print(
                    "[WARN]: --warm_start_from_single_frame が指定されましたが、checkpoint に"
                    " 1 フレーム版の actor (actor.<N>.weight) がありません。"
                    " 既に履歴入力版の checkpoint の可能性があります (移植は何もしていません)。"
                )

        # 形の合うテンソルだけをロードする。
        # obs次元が違う転移では入力層 (actor.0.weight) と normalizer の形が合わないので
        # 自動的に除外され、新しい次元で初期化されたままになる。
        # obs次元が同じ転移 (例: walk_kick の walk phase → kick phase) では入力層も
        # normalizer の統計もそのまま引き継がれる。
        live_msd = policy.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k in live_msd and live_msd[k].shape == v.shape:
                filtered[k] = v
            else:
                shape_info = f"{tuple(v.shape)} -> {tuple(live_msd[k].shape)}" if k in live_msd else "not in model"
                skipped.append(f"{k} ({shape_info})")
        print(f"[INFO]: Loaded {len(filtered)} tensors from pretrained checkpoint.")
        if skipped:
            print(f"[INFO]: Skipped {len(skipped)} tensors (shape mismatch / unknown key):")
            for s in skipped:
                print(f"          {s}")

        # actor が 1 本も引き継げていないなら止める。
        #
        # このフィルタは形の合わないテンソルを黙って捨てるので、1 フレーム観測の
        # checkpoint を履歴入力タスクへ --warm_start_from_single_frame 無しで渡すと
        # 「critic と正規化統計と std だけ載って actor は乱数のまま」で学習が始まる。
        # ログ上は正常な起動に見えるうえ、歩けないところからのやり直しなので
        # fine-tune 前提のカリキュラム (キック報酬のランプ / 帯のランプ) が全部空振りする。
        # 実際 k1_walk_long_pass/2026-08-11_16-31-27 はこれで 5000 iteration を潰した
        # (base_height 終了 99.9%、kick_rate ≈ 0 のまま歩行だけを再獲得)。
        actor_loaded = [k for k in filtered if k.startswith("actor.")]
        if not actor_loaded:
            message = (
                "[ERROR]: 引き継いだテンソルに actor が 1 本も含まれていません"
                " (critic / 正規化統計 / std だけがロードされました)。\n"
                "         このまま学習すると actor は乱数初期化のままなので、歩行から"
                " やり直しになります。\n"
                "         1 フレーム観測の checkpoint を履歴入力タスクへ渡した場合は"
                " --warm_start_from_single_frame を付けてください。\n"
                "         critic だけを引き継ぐのが意図どおりなら"
                " --allow_untransferred_actor を付けてください。"
            )
            if not args_cli.allow_untransferred_actor:
                raise RuntimeError(message)
            print(message.replace("[ERROR]", "[WARN] "))

        result = policy.load_state_dict(filtered, strict=False)
        if isinstance(result, tuple):
            missing, unexpected = result
            if missing:
                print(f"[INFO]: Missing keys: {missing}")
            if unexpected:
                print(f"[INFO]: Unexpected keys: {unexpected}")

        # re-inject action noise std if requested. Unlike the --resume path, load_pretrained
        # never touches the optimizer (runner.alg was built fresh, we only overwrote policy
        # weights via state_dict), so there is no stale Adam momentum on std to worry about here.
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
