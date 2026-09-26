"""Rollout/checkpoint boundary for DirectKick's original full-batch PPO."""
import csv
import json
from pathlib import Path
import time

import torch
import yaml

from .learning import DirectKickPPO
from .model import DirectKickingActorCritic
from .episode_metrics import episode_metric_scalars, write_episode_metrics
from .terminal import EpisodeStatistics, format_duration


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
        self.terminal_statistics = EpisodeStatistics(
            env.unwrapped.num_envs, self.device, cfg['rewards']['episode_length_s'])
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
            self.ppo.learning_rate = float(self.ppo.optimizer.param_groups[0]['lr'])
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
                # Snapshot raw environment rewards before PPO timeout bootstrapping.
                self.terminal_statistics.record(
                    rollout['rewards'][t], rollout['dones'][t], extras.get('rew_terms', {}))
        return rollout, observations, extras

    def train(self, max_iterations):
        observations, extras = self.env.reset()
        self.terminal_statistics.reset()
        start_iteration = self.iteration
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
                    iteration_start = time.monotonic()
                    rollout, observations, extras = self.collect(observations, extras)
                    collection_end = time.monotonic()
                    reward = rollout['rewards'].mean().item()
                    metrics = self.ppo.update(rollout, observations['policy'].to(self.device), observations['critic'].to(self.device))
                    learning_end = time.monotonic()
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
                    episode_summary = self.env.unwrapped.episode_metrics.summary(reset=True)
                    write_episode_metrics(self.log_dir / 'episode_metrics.csv', episode_summary, self.iteration)
                    self._print_progress(
                        max_iterations, metrics, episode_summary,
                        collection_end - iteration_start, learning_end - collection_end,
                        time.monotonic() - start, self.iteration - start_iteration)
                    if wandb_run is not None:
                        wandb_run.log({**metrics, **episode_metric_scalars(episode_summary)}, step=self.iteration)
                    if self.iteration % self.cfg['runner']['save_interval'] == 0:
                        self.save()
                return self.save()
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    def _print_progress(self, max_iterations, metrics, episode_summary,
                        collection_time, learning_time, elapsed_time, completed_iterations):
        """Use the former IsaacLab/RSL-RL console layout with this task's values."""
        width, pad = 80, 35
        iteration_time = collection_time + learning_time
        batch_steps = self.cfg['runner']['horizon_length'] * self.env.unwrapped.num_envs
        fps = int(batch_steps / iteration_time) if iteration_time > 0 else 0
        episodes = self.terminal_statistics.finish_iteration()
        title = f' \033[1m Learning iteration {self.iteration}/{max_iterations} \033[0m '
        lines = ['#' * width, title.center(width), '']

        def row(label, value):
            lines.append(f'{label + ":":>{pad}} {value}')

        row('Computation', f'{fps} steps/s (collection: {collection_time:.3f}s, learning {learning_time:.3f}s)')
        row('Mean action noise std', f'{self.model.logstd.detach().exp().mean().item():.2f}')
        for key, label in (('value_loss', 'value_function'), ('actor_loss', 'surrogate'),
                           ('bound_loss', 'bound'), ('symmetry_loss', 'symmetry'), ('phase_loss', 'phase')):
            row(f'Mean {label} loss', f'{metrics[key]:.4f}')
        row('Mean entropy', f'{metrics["entropy"]:.4f}')
        row('KL divergence', f'{metrics["kl"]:.6f}')
        row('Learning rate', f'{metrics["learning_rate"]:.6f}')
        if episodes["reward"] is not None:
            row('Mean reward', f'{episodes["reward"]:.2f}')
            row('Mean episode length', f'{episodes["episode_length"]:.2f}')
        for name, value in episodes['reward_terms'].items():
            row(f'Episode_Reward/{name}', f'{value:.4f}')
        completed = episode_summary['all']
        row('Completed episodes', completed['episodes'])
        if completed['episodes']:
            row('Kick rate', f'{completed["kick_rate"]:.3f}')
            row('Fall rate', f'{completed["fall_rate"]:.3f}')
        lines.append('-' * width)
        row('Total timesteps', self.total_steps)
        row('Iteration time', f'{iteration_time:.2f}s')
        row('Time elapsed', format_duration(elapsed_time))
        remaining = max_iterations - self.iteration
        row('ETA', format_duration(elapsed_time / completed_iterations * remaining))
        print('\n'.join(lines) + '\n', flush=True)

    def export(self, path):
        """Stateless future-horizon LSTM; output = 12 actions + phase probability."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        actor = self.model.actor.cpu().eval()
        torch.jit.script(actor).save(str(path))
        path.with_suffix('.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
        self.model.to(self.device)
