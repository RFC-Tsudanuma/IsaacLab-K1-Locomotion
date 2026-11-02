"""Isaac Sim 上で BlockActorCritic と export_block_policy.py の ONNX の出力を突き合わせる (2026-11-02)。

例::

    uv run check_block_onnx.py --headless --num_envs 8 --device cuda:1 \\
        --checkpoint /abs/model_999.pt --onnx /abs/block_999.onnx
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Block policy ONNX parity check.")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--task", type=str, default="Isaac-Block-K1-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--onnx", type=str, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_k1_locomotion.tasks  # noqa: F401, E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint), map_location=agent_cfg.device)
    policy = runner.alg.policy
    policy.eval()
    sess = ort.InferenceSession(args_cli.onnx, providers=["CPUExecutionProvider"])
    T = policy.actor.seq_len

    obs = env.get_observations()
    max_cmd, max_res = 0.0, 0.0
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            raw = policy.act_inference(obs)
            act = policy.env_actions(obs, raw)  # [expert + res (12), cmd (3), res (12)]
            ref_cmd, ref_res = act[:, 12:15].cpu().numpy(), act[:, 15:27].cpu().numpy()
            block_obs = obs["block_policy"].cpu().numpy().astype(np.float32)
            ball = obs["ball_noisy"].cpu().numpy().astype(np.float32).reshape(-1, T, 4)
            for i in range(block_obs.shape[0]):
                cmd, res = sess.run(None, {"block_obs": block_obs[i : i + 1], "ball_history": ball[i : i + 1]})
                max_cmd = max(max_cmd, float(np.abs(cmd[0] - ref_cmd[i]).max()))
                max_res = max(max_res, float(np.abs(res[0] - ref_res[i]).max()))
            obs, _, _, _ = env.step(act)
    print(f"[check_block_onnx] {args_cli.steps} steps x {args_cli.num_envs} envs: max |Δcmd| = {max_cmd:.3e}, max |Δres| = {max_res:.3e}")
    assert max_cmd < 1e-3 and max_res < 1e-3, "ONNX と BlockActorCritic の出力が一致しません"
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
