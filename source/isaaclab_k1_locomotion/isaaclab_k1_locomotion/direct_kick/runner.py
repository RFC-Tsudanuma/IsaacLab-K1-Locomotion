"""Rollout/checkpoint boundary for DirectKick's original full-batch PPO."""
import csv
import json
from pathlib import Path
import time

import torch
import yaml

from .learning import DirectKickPPO
from .model import DirectKickingActorCritic


class DirectKickRunner:
    def __init__(self, env, cfg, log_dir, device):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model = DirectKickingActorCritic.from_config(12, cfg['env']['num_observations'], 20, cfg).to(self.device)
        self.ppo = DirectKickPPO(self.model, cfg, self.device)
        self.iteration = 0
        self.total_steps = 0
        self.metadata = {'model': self.model.checkpoint_metadata(), 'migration': cfg['migration']}
        (self.log_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        (self.log_dir / 'policy_contract.json').write_text(json.dumps(self.metadata, indent=2) + '\n')

    def load(self, path, resume=True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get('port_metadata') != self.metadata:
            raise ValueError('Checkpoint perception/model contract differs from this DirectKick VisionFilter port')
        self.model.load_state_dict(checkpoint['model'], strict=True)
        if resume:
            self.ppo.optimizer.load_state_dict(checkpoint['optimizer'])
            self.iteration = int(checkpoint['iteration'])
            self.total_steps = int(checkpoint['total_steps'])
        return checkpoint

    def save(self):
        path = self.log_dir / f'model_{self.iteration}.pth'
        torch.save({'model': self.model.state_dict(), 'optimizer': self.ppo.optimizer.state_dict(),
                    'iteration': self.iteration, 'total_steps': self.total_steps,
                    'model_metadata': self.model.checkpoint_metadata(), 'port_metadata': self.metadata,
                    'config': self.cfg}, path)
        return path

    def collect(self, observations, extras):
        horizon = self.cfg['runner']['horizon_length']
        n = self.env.unwrapped.num_envs
        specs = {'obses': (self.model.num_observations,), 'privileged_obses': (20,), 'actions': (12,),
                 'rewards': (), 'dones': (), 'time_outs': (), 'post_kick_phase_targets': ()}
        rollout = {key: torch.empty((horizon, n, *shape), device=self.device,
                                   dtype=torch.bool if key in ('dones', 'time_outs') else torch.float32)
                   for key, shape in specs.items()}
        with torch.no_grad():
            for t in range(horizon):
                obs = observations['policy'].to(self.device)
                privileged = observations['critic'].to(self.device)
                actions = self.model.act(obs).sample()
                rollout['obses'][t] = obs
                rollout['privileged_obses'][t] = privileged
                rollout['actions'][t] = actions
                # This target belongs to the observation BEFORE stepping/resetting.
                rollout['post_kick_phase_targets'][t] = extras['post_kick_phase_target'].to(self.device)
                observations, rewards, terminated, truncated, extras = self.env.step(actions)
                rollout['rewards'][t] = rewards.to(self.device)
                rollout['dones'][t] = (terminated | truncated).to(self.device)
                rollout['time_outs'][t] = truncated.to(self.device)
        return rollout, observations, extras

    def train(self, max_iterations):
        observations, extras = self.env.reset()
        wandb_run = None
        if self.cfg['runner']['use_wandb']:
            import wandb
            wandb_run = wandb.init(project='K1-DirectKick-IsaacLab', config=self.cfg, dir=str(self.log_dir))
        start = time.monotonic()
        csv_path = self.log_dir / 'learning.csv'
        try:
            with csv_path.open('a', newline='') as stream:
                writer = None
                while self.iteration < max_iterations:
                    rollout, observations, extras = self.collect(observations, extras)
                    reward = rollout['rewards'].mean().item()
                    metrics = self.ppo.update(rollout, observations['policy'].to(self.device), observations['critic'].to(self.device))
                    self.iteration += 1
                    self.total_steps += rollout['rewards'].numel()
                    metrics.update(iteration=self.iteration, total_steps=self.total_steps, reward=reward,
                                   elapsed_seconds=time.monotonic() - start)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(metrics))
                        if stream.tell() == 0:
                            writer.writeheader()
                    writer.writerow(metrics)
                    stream.flush()
                    print(f"iteration={self.iteration} reward={reward:.5f} value_loss={metrics['value_loss']:.5f} kl={metrics['kl']:.6f}", flush=True)
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=self.iteration)
                    if self.iteration % self.cfg['runner']['save_interval'] == 0:
                        self.save()
                return self.save()
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    def export(self, path):
        """Stateless future-horizon LSTM; output = 12 actions + phase probability."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        actor = self.model.actor.cpu().eval()
        torch.jit.script(actor).save(str(path))
        path.with_suffix('.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
        self.model.to(self.device)
