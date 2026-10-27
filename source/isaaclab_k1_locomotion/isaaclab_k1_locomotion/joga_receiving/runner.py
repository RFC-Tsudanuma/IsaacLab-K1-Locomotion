"""RSL-RL PPO for receiving; no kicking losses, phase head, or ablation modes."""
import copy
import csv
import json
from pathlib import Path

import torch
import yaml
from rsl_rl.algorithms import PPO
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from ..direct_kick.config import VISION_REVISION, PERCEPTION_SCHEMA
from .model import ReceivingActorCritic


def tensor_observations(observations, device):
    # Native environments may reuse history buffers across step/reset.
    return TensorDict({name: value.detach().to(device).clone() for name, value in observations.items()},
                      batch_size=[observations["policy"].shape[0]])


class ReceivingRunner:
    def __init__(self, env, cfg, log_dir, device="cpu"):
        self.env = env
        self.cfg = copy.deepcopy(cfg)
        self.device = torch.device(device)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        initial, _ = env.reset()
        self.observations = tensor_observations(initial, self.device)
        self.model = ReceivingActorCritic(self.observations, self.cfg).to(self.device)
        self.ppo = PPO(self.model, device=str(self.device), **self.cfg["ppo"])
        self.ppo.init_storage("rl", env.unwrapped.num_envs, cfg["runner"]["steps_per_env"],
                              self.observations, [12])
        self.iteration = 0
        self.total_steps = 0
        self.metadata = {"contract": cfg["contract"], "task": cfg["task"],
                         "policy_size": cfg["policy_size"], "critic_size": cfg["critic_size"],
                         "num_actions": 12, "model": copy.deepcopy(cfg["model"]),
                         "physics_dt": cfg["physics_dt"], "decimation": cfg["decimation"],
                         "history_length": cfg["history_length"],
                         "perception_revision": VISION_REVISION,
                         "perception_schema": PERCEPTION_SCHEMA}
        (self.log_dir / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))
        (self.log_dir / "policy_contract.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    def save(self):
        path = self.log_dir / f"model_{self.iteration}.pt"
        torch.save({"metadata": self.metadata, "model": self.model.state_dict(),
                    "optimizer": self.ppo.optimizer.state_dict(), "config": self.cfg,
                    "iteration": self.iteration, "total_steps": self.total_steps}, path)
        return path

    def load(self, path, resume=True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("metadata") != self.metadata:
            raise ValueError("Checkpoint contract differs from K1 Joga receiving")
        if checkpoint["config"] != self.cfg:
            raise ValueError("Checkpoint task/training configuration differs; use its saved configuration")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        if resume:
            self.ppo.optimizer.load_state_dict(checkpoint["optimizer"])
            self.ppo.learning_rate = float(self.ppo.optimizer.param_groups[0]["lr"])
            self.iteration = int(checkpoint["iteration"])
            self.total_steps = int(checkpoint["total_steps"])
        self.env.unwrapped.set_curriculum_iteration(int(checkpoint["iteration"]))
        # Reinitialize the simulator at the checkpoint curriculum, not stage 0.
        observations, _ = self.env.reset()
        self.observations = tensor_observations(observations, self.device)
        return checkpoint

    def train(self, max_iterations):
        self.model.train()
        csv_path = self.log_dir / "learning.csv"
        with SummaryWriter(log_dir=str(self.log_dir / "tensorboard")) as summary, \
                csv_path.open("a", newline="") as stream:
            writer = None
            while self.iteration < max_iterations:
                self.env.unwrapped.set_curriculum_iteration(self.iteration)
                reward_sum = torch.zeros((), device=self.device)
                reward_term_sums = {}
                with torch.inference_mode():
                    for _ in range(self.cfg["runner"]["steps_per_env"]):
                        actions = self.ppo.act(self.observations)
                        observations, rewards, terminated, truncated, extras = self.env.step(
                            actions.to(self.env.unwrapped.device))
                        reward_sum += rewards.detach().to(self.device).sum()
                        # The environment already includes weights and dt in
                        # rew_terms; keep the actual signed reward contributions.
                        for name, values in extras.get("rew_terms", {}).items():
                            term_sum = values.detach().to(self.device).sum()
                            if name in reward_term_sums:
                                reward_term_sums[name] += term_sum
                            else:
                                reward_term_sums[name] = term_sum
                        self.observations = tensor_observations(observations, self.device)
                        self.ppo.process_env_step(self.observations, rewards.to(self.device),
                                                  (terminated | truncated).to(self.device),
                                                  {**extras, "time_outs": truncated.to(self.device)})
                    self.ppo.compute_returns(self.observations)
                losses = self.ppo.update()
                self.iteration += 1
                self.total_steps += self.cfg["runner"]["steps_per_env"] * self.env.unwrapped.num_envs
                episode_metrics = self.env.unwrapped.drain_episode_metrics()
                metrics = {"iteration": self.iteration, "total_steps": self.total_steps,
                           "learning_rate": self.ppo.learning_rate, **losses,
                           **episode_metrics}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(metrics))
                    if stream.tell() == 0:
                        writer.writeheader()
                writer.writerow(metrics)
                stream.flush()
                for name, value in metrics.items():
                    # No completed episodes means no measured episode statistic.
                    if value != "":
                        group = "Episode" if name in episode_metrics else "Train"
                        summary.add_scalar(f"{group}/{name}", value, self.iteration)
                samples = self.cfg["runner"]["steps_per_env"] * self.env.unwrapped.num_envs
                summary.add_scalar("Reward/mean_step_total", (reward_sum / samples).item(), self.iteration)
                for name, value in reward_term_sums.items():
                    summary.add_scalar(f"Reward/mean_step/{name}", (value / samples).item(), self.iteration)
                summary.flush()
                print(f"Receiving iteration {self.iteration}/{max_iterations}: "
                      f"return={metrics['mean_return']} success={metrics['success_rate']} "
                      f"completed={metrics['episodes']} lr={self.ppo.learning_rate:.3g}", flush=True)
                if self.iteration % self.cfg["runner"]["save_interval"] == 0:
                    self.save()
        return self.save()

    def export(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.script(self.model.export_actor()).save(str(path))
        path.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2) + "\n")
