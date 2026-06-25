# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained velocity predictor against IsaacLab simulation.

Runs the same command-sampling protocol as collect_velocity_data.py, but instead
of writing zarr it compares the predictor's online output (model.step) against
the actual robot velocity measured in simulation, and prints aggregated error
metrics.

Compared baselines (per axis):
    naive:    v_pred = cmd_t  (assume robot perfectly tracks command)
    baseline: 1st-order baseline only (residual = 0)
    model:    full predictor (1st-order + GRU residual)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate the velocity predictor in IsaacLab.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_rollouts", type=int, default=4)
parser.add_argument("--episode_length", type=int, default=250, help="Steps per rollout.")
parser.add_argument("--predictor_path", type=str, required=True,
                    help="Path to the .pt produced by train_velocity_predictor.py")
parser.add_argument("--settle_steps", type=int, default=30,
                    help="Steps treated as transient phase (~tau*~3 / dt).")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.util
import os
from pathlib import Path

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
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_k1_locomotion.tasks  # noqa: F401
from isaaclab_k1_locomotion.velocity_model import PATTERN_NAMES, pattern_partition, sample_commands

# Load VelocityPredictor directly to avoid pulling the parent package init twice.
from isaaclab_k1_locomotion.velocity_model.predictor import VelocityPredictor

# Reuse override_command and proprio helpers from the collector module.
_collect_path = Path(__file__).resolve().parent / "collect_velocity_data.py"
_spec = importlib.util.spec_from_file_location("_collector", _collect_path)
_collector = importlib.util.module_from_spec(_spec)
# Skip the AppLauncher block at the top by faking sys.argv-aware execution? Not needed —
# collect_velocity_data.py *does* run AppLauncher at module import time. To avoid that,
# we reimplement the two small helpers we need locally instead of importing it.


PROPRIO_SOURCE_KEYS = (
    "joint_pos",
    "joint_vel",
    "base_quat",
    "projected_gravity",
    "base_lin_vel_b",
    "base_ang_vel_b",
)


def _proprio_source(robot, key: str) -> torch.Tensor:
    if key == "joint_pos":
        return robot.data.joint_pos
    if key == "joint_vel":
        return robot.data.joint_vel
    if key == "base_quat":
        return robot.data.root_quat_w
    if key == "projected_gravity":
        return robot.data.projected_gravity_b
    if key == "base_lin_vel_b":
        return robot.data.root_lin_vel_b
    if key == "base_ang_vel_b":
        return robot.data.root_ang_vel_b
    raise KeyError(f"Unknown proprio key: {key}")


def override_command(env, cmd_n3: torch.Tensor) -> None:
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    if hasattr(cmd_term, "vel_command_b"):
        cmd_term.vel_command_b[:] = cmd_n3
    if hasattr(cmd_term, "command"):
        try:
            cmd_term.command[:] = cmd_n3
        except (AttributeError, TypeError):
            pass
    if hasattr(cmd_term, "heading_target"):
        cmd_term.heading_target[:] = 0.0
    if hasattr(cmd_term, "is_heading_env"):
        cmd_term.is_heading_env[:] = False
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = False


def get_velocity(robot) -> torch.Tensor:
    """Return (num_envs, 3) actual body-frame velocity = [vx, vy, wz]."""
    lin = robot.data.root_lin_vel_b[:, :2]
    ang = robot.data.root_ang_vel_b[:, 2:3]
    return torch.cat([lin, ang], dim=-1)


def make_pattern_labels(num_envs: int, device) -> torch.Tensor:
    """Label each env by its command pattern, mirroring sample_commands()'s partition."""
    labels = torch.zeros(num_envs, dtype=torch.long, device=device)
    start = 0
    for idx, (_name, cnt) in enumerate(pattern_partition(num_envs)):
        labels[start : start + cnt] = idx
        start += cnt
    return labels


AXIS_NAMES = ["vx", "vy", "wz"]


class ErrorStats:
    """Accumulate per-axis squared/abs error split by pattern and by phase.

    For each (predictor, pattern, phase) bucket we keep:
        sum_abs:  sum of |err| per axis  (3,)
        sum_sq:   sum of err**2 per axis (3,)
        count:    number of valid (env, t) samples included
    """

    PREDICTORS = ["naive", "baseline", "model"]
    PHASES = ["settle", "steady"]

    def __init__(self, num_envs: int, device):
        # buckets[pred][pat][phase] = dict(sum_abs, sum_sq, count)
        self.buckets = {
            p: {pat: {ph: self._empty(device) for ph in self.PHASES} for pat in PATTERN_NAMES}
            for p in self.PREDICTORS
        }

    @staticmethod
    def _empty(device):
        return {
            "sum_abs": torch.zeros(3, device=device),
            "sum_sq": torch.zeros(3, device=device),
            "count": torch.zeros(1, device=device),
        }

    def update(
        self,
        v_actual: torch.Tensor,        # (N, 3)
        preds: dict,                   # {"naive": (N,3), "baseline": (N,3), "model": (N,3)}
        pattern_labels: torch.Tensor,  # (N,) long
        valid_mask: torch.Tensor,      # (N,) bool — env still alive
        is_settle: bool,
    ):
        phase = "settle" if is_settle else "steady"
        for pred_name, v_pred in preds.items():
            err = v_pred - v_actual                    # (N, 3)
            abs_err = err.abs()
            sq_err = err * err
            for pat_idx, pat in enumerate(PATTERN_NAMES):
                m = valid_mask & (pattern_labels == pat_idx)
                if not bool(m.any()):
                    continue
                bucket = self.buckets[pred_name][pat][phase]
                bucket["sum_abs"] += abs_err[m].sum(dim=0)
                bucket["sum_sq"] += sq_err[m].sum(dim=0)
                bucket["count"] += m.float().sum()

    def _agg(self, pred: str, patterns: list[str], phases: list[str]):
        sa = torch.zeros(3)
        ss = torch.zeros(3)
        c = 0.0
        for pat in patterns:
            for ph in phases:
                b = self.buckets[pred][pat][ph]
                sa = sa + b["sum_abs"].cpu()
                ss = ss + b["sum_sq"].cpu()
                c += float(b["count"].item())
        if c == 0:
            return None
        mae = sa / c
        rmse = (ss / c).sqrt()
        return mae.numpy(), rmse.numpy(), int(c)

    def report(self):
        sep = "=" * 78

        def print_row(label, mae, rmse, c):
            print(
                f"  {label:<24} "
                f"MAE: vx={mae[0]:.4f}  vy={mae[1]:.4f}  wz={mae[2]:.4f}  | "
                f"RMSE: vx={rmse[0]:.4f}  vy={rmse[1]:.4f}  wz={rmse[2]:.4f}  | n={c:,}"
            )

        # --- Overall (all patterns, all phases) ---
        print(f"\n{sep}\n  Overall (全パターン, settle+steady)\n{sep}")
        for pred in self.PREDICTORS:
            r = self._agg(pred, PATTERN_NAMES, self.PHASES)
            if r:
                mae, rmse, c = r
                print_row(pred, mae, rmse, c)

        # --- Phase split ---
        for ph in self.PHASES:
            print(f"\n{sep}\n  Phase = {ph}\n{sep}")
            for pred in self.PREDICTORS:
                r = self._agg(pred, PATTERN_NAMES, [ph])
                if r:
                    mae, rmse, c = r
                    print_row(pred, mae, rmse, c)

        # --- Per pattern (steady only — fairer comparison once transient is gone) ---
        print(f"\n{sep}\n  Per command pattern (steady phase only)\n{sep}")
        for pat in PATTERN_NAMES:
            print(f"\n  [{pat}]")
            for pred in self.PREDICTORS:
                r = self._agg(pred, [pat], ["steady"])
                if r:
                    mae, rmse, c = r
                    print_row(pred, mae, rmse, c)

        # --- Improvement vs baselines (overall steady) ---
        print(f"\n{sep}\n  Model vs baselines (steady, all patterns) — RMSE 改善率\n{sep}")
        for ref in ("naive", "baseline"):
            r_ref = self._agg(ref, PATTERN_NAMES, ["steady"])
            r_mod = self._agg("model", PATTERN_NAMES, ["steady"])
            if r_ref and r_mod:
                _, rmse_ref, _ = r_ref
                _, rmse_mod, _ = r_mod
                impr = (rmse_ref - rmse_mod) / (rmse_ref + 1e-9) * 100.0
                print(
                    f"  vs {ref:<10}  "
                    f"vx: {impr[0]:+6.1f}%  vy: {impr[1]:+6.1f}%  wz: {impr[2]:+6.1f}%"
                )
        print(f"\n{sep}\n")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    needed_s = args_cli.episode_length * 0.02 + 1.0
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, needed_s)
    try:
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    except AttributeError:
        pass
    try:
        env_cfg.commands.base_velocity.heading_command = False
    except AttributeError:
        pass

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

    print(f"[INFO] Loading policy checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    raw_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    ckpt_msd = raw_ckpt.get("model_state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else None
    if isinstance(ckpt_msd, dict):
        policy_to_load = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        target_device = next(policy_to_load.parameters()).device
        ckpt_on_device = {k: (v.to(target_device) if isinstance(v, torch.Tensor) else v)
                          for k, v in ckpt_msd.items()}
        policy_to_load.load_state_dict(ckpt_on_device, strict=False)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    device = env.unwrapped.device

    # ---- Load velocity predictor ----
    print(f"[INFO] Loading predictor: {args_cli.predictor_path}")
    pred_ckpt = torch.load(args_cli.predictor_path, map_location=device, weights_only=False)
    pred_args = pred_ckpt.get("args", {})
    use_proprio = bool(pred_args.get("use_proprio", False))
    proprio_keys = list(pred_args.get("proprio_keys", []))
    hidden_dim = int(pred_ckpt["hidden_dim"])
    proprio_dim = int(pred_ckpt["proprio_dim"])
    dt = float(pred_ckpt["dt"])
    residual_scale = float(pred_ckpt["residual_scale"])

    sim_dt = float(env.unwrapped.step_dt)
    if abs(sim_dt - dt) > 1e-6:
        print(f"[WARN] predictor dt ({dt}) != sim dt ({sim_dt}); using sim dt for stepping.")
    model = VelocityPredictor(
        dim=3, dt=sim_dt, hidden_dim=hidden_dim,
        proprio_dim=proprio_dim, residual_scale=residual_scale,
    ).to(device)
    model.load_state_dict(pred_ckpt["model"])
    model.eval()

    print(
        f"[INFO] sim_dt={sim_dt:.4f}s, num_envs={args_cli.num_envs}, "
        f"episode_length={args_cli.episode_length}, num_rollouts={args_cli.num_rollouts}, "
        f"settle_steps={args_cli.settle_steps}, "
        f"use_proprio={use_proprio}, proprio_keys={proprio_keys}, "
        f"hidden_dim={hidden_dim}, proprio_dim={proprio_dim}"
    )

    num_envs = env.num_envs
    pattern_labels = make_pattern_labels(num_envs, device)
    stats = ErrorStats(num_envs, device)

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    for r in range(args_cli.num_rollouts):
        commands = sample_commands(num_envs, args_cli.episode_length, sim_dt, device)
        v_base = torch.zeros(num_envs, 3, device=device)
        h = torch.zeros(1, num_envs, hidden_dim, device=device)
        # alive mask: once an env terminates within a rollout we drop it for the rest of that rollout.
        alive = torch.ones(num_envs, dtype=torch.bool, device=device)

        for t in range(args_cli.episode_length):
            cmd_t = commands[:, t]
            override_command(env, cmd_t)

            proprio_t = None
            if use_proprio:
                robot = env.unwrapped.scene["robot"]
                # match VelocityDataset: sorted concat over proprio_keys
                proprio_t = torch.cat(
                    [_proprio_source(robot, k) for k in sorted(proprio_keys)], dim=-1
                )

            with torch.inference_mode():
                action = policy(obs)
                # Predictor inference: model.step is wrapped in torch.no_grad already.
                v_pred, v_base_new, h_new = model.step(cmd_t, v_base, h, proprio_t=proprio_t)

            obs, _, dones, _ = env.step(action)
            v_actual = get_velocity(env.unwrapped.scene["robot"])

            naive_pred = cmd_t                       # (N, 3)
            baseline_pred = v_base_new               # 1st-order only
            preds = {"naive": naive_pred, "baseline": baseline_pred, "model": v_pred}

            stats.update(
                v_actual=v_actual,
                preds=preds,
                pattern_labels=pattern_labels,
                valid_mask=alive,
                is_settle=(t < args_cli.settle_steps),
            )

            v_base = v_base_new
            h = h_new
            alive = alive & ~dones.bool()

        n_alive = int(alive.sum().item())
        print(f"[{r+1}/{args_cli.num_rollouts}] rollout finished — alive_envs={n_alive}/{num_envs}")

    stats.report()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
