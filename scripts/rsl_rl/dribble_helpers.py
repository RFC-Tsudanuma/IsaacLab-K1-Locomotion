# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the hierarchical dribble train/play scripts.

This module is imported AFTER ``AppLauncher`` has started in both
``train_dribble.py`` and ``play_dribble.py``. It is kept side-effect free
(no argparse, no AppLauncher) so it can be imported from either script
without re-parsing ``sys.argv``.
"""

import math
import os

import torch
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.utils import resolve_obs_groups

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper


def _term_slot(observation_manager, group: str, term_name: str) -> tuple[int, int]:
    """Return (start, end) column indices of ``term_name`` within the concatenated tensor of ``group``."""
    if group not in observation_manager.active_terms:
        raise ValueError(
            f"Observation group '{group}' not found. Available: {list(observation_manager.active_terms.keys())}"
        )
    term_names = list(observation_manager.active_terms[group])
    term_dims = list(observation_manager.group_obs_term_dim[group])
    if term_name not in term_names:
        raise ValueError(
            f"Term '{term_name}' not found in group '{group}'. Available terms: {term_names}"
        )
    idx = term_names.index(term_name)
    start = sum(int(math.prod(d)) for d in term_dims[:idx])
    end = start + int(math.prod(term_dims[idx]))
    return start, end


class HierarchicalVecEnvWrapper:
    """rsl_rl VecEnv-compatible wrapper exposing a high-level (walking command) action space.

    Each step:
        1. Read the current env observation.
        2. Build a copy of the chosen ``low_level_obs_group`` tensor with the
           ``low_level_cmd_term_name`` slice overwritten by the high-level action.
        3. Run the frozen low-level policy on the modified observation → joint targets.
        4. Step the inner env with those joint targets.

    The env's ``command_manager`` is not touched.
    """

    def __init__(
        self,
        inner_env: RslRlVecEnvWrapper,
        low_level_policy,
        low_level_obs_group: str = "policy",
        low_level_cmd_term_name: str = "velocity_commands",
        action_clip=(1.0, 1.0, 1.0),
        high_action_dim: int = 3,
    ):
        self.env = inner_env
        self.low_level_policy = low_level_policy
        self.low_level_obs_group = low_level_obs_group
        # rsl_rl VecEnv-required attrs
        self.num_envs = inner_env.num_envs
        self.device = inner_env.device
        self.max_episode_length = inner_env.max_episode_length
        self.num_actions = int(high_action_dim)
        # Per-axis clipping: accept a single float (uniform) or a per-axis sequence.
        if isinstance(action_clip, (int, float)):
            clip_vals = [float(action_clip)] * self.num_actions
        else:
            clip_vals = [float(v) for v in action_clip]
            if len(clip_vals) != self.num_actions:
                raise ValueError(
                    f"action_clip must be a scalar or length-{self.num_actions} sequence, got {clip_vals}"
                )
        self.action_clip = torch.tensor(clip_vals, device=self.device)  # (num_actions,)
        # Resolve the slice of the walking-command observation term inside the chosen group.
        om = inner_env.unwrapped.observation_manager
        self._cmd_slot = _term_slot(om, low_level_obs_group, low_level_cmd_term_name)
        slot_dim = self._cmd_slot[1] - self._cmd_slot[0]
        if slot_dim != self.num_actions:
            raise ValueError(
                f"Slot dim of obs term '{low_level_cmd_term_name}' in group '{low_level_obs_group}' is"
                f" {slot_dim}, but high_action_dim is {self.num_actions}."
            )

        # Buffer holding the previous high-level (clipped) action. The env's
        # ``last_high_action`` observation reads this; ``reset_prev_high_action``
        # event zeroes per-env slots at reset time.
        inner_env.unwrapped._prev_high_action = torch.zeros(
            self.num_envs, self.num_actions, device=self.device
        )

    # -- pass-through properties expected by rsl_rl / IsaacLab utilities --
    @property
    def cfg(self):
        return self.env.cfg

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def render_mode(self):
        return self.env.render_mode

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.env.episode_length_buf = value

    # -- VecEnv API --
    def reset(self):
        return self.env.reset()

    def get_observations(self):
        return self.env.get_observations()

    def step(self, action: torch.Tensor):
        # Per-axis clip: self.action_clip is (num_actions,); broadcast over batch.
        cmd = torch.clamp(action.to(self.device), -self.action_clip, self.action_clip)

        # Frozen policy is fixed: never need autograd for it. Wrap the whole
        # low-level inference in inference_mode so that we don't build a
        # computation graph even if the outer caller (rsl_rl rollout) forgets to.
        with torch.inference_mode():
            env_obs = self.env.get_observations()

            low_group_tensor = env_obs[self.low_level_obs_group].clone()
            s, e = self._cmd_slot
            # cmd may carry grad info from the high-level sampler; detach defensively
            # before stuffing into the frozen policy's input slot.
            low_group_tensor[:, s:e] = cmd.detach()
            low_obs = {k: env_obs[k] for k in env_obs.keys()}
            low_obs[self.low_level_obs_group] = low_group_tensor

            joint_action = self.low_level_policy.act_inference(low_obs)

        # Stash *before* env.step so that the post-step obs (incl. any auto-reset
        # branches) observes the correct prev high action. For envs that get
        # auto-reset inside env.step, the reset event clears this slot.
        self.env.unwrapped._prev_high_action.copy_(cmd.detach())

        return self.env.step(joint_action)

    def close(self):
        return self.env.close()


class _JitFrozenPolicy:
    """TorchScript wrapper exposing the same ``act_inference`` API as RSL-RL policies."""

    def __init__(self, jit_module: torch.jit.ScriptModule, low_level_obs_group: str, device: str):
        self._module = jit_module.to(device).eval()
        for p in self._module.parameters():
            p.requires_grad_(False)
        self._low_level_obs_group = low_level_obs_group
        self._device = device

    @torch.inference_mode()
    def act_inference(self, obs: dict) -> torch.Tensor:
        return self._module(obs[self._low_level_obs_group].to(self._device))


class _OnnxFrozenPolicy:
    """ONNX-runtime wrapper exposing the same ``act_inference`` API as RSL-RL policies."""

    def __init__(self, onnx_path: str, low_level_obs_group: str, device: str):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device else ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        input_names = [i.name for i in self._session.get_inputs()]
        if input_names != ["obs"]:
            raise ValueError(
                f"ONNX frozen policy expects a single input named 'obs' (non-recurrent export),"
                f" got inputs={input_names}. Recurrent ONNX frozen policies are not supported."
            )
        self._output_name = self._session.get_outputs()[0].name
        self._low_level_obs_group = low_level_obs_group
        self._device = device

    def act_inference(self, obs: dict) -> torch.Tensor:
        x = obs[self._low_level_obs_group]
        x_np = x.detach().cpu().numpy().astype("float32", copy=False)
        out = self._session.run([self._output_name], {"obs": x_np})[0]
        return torch.from_numpy(out).to(self._device)


def _build_frozen_policy(
    env: RslRlVecEnvWrapper,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint_path: str,
    device: str,
    low_level_obs_group: str,
):
    """Construct a low-level policy and load weights from a ``.pt`` or ``.onnx`` checkpoint."""
    ext = os.path.splitext(checkpoint_path)[1].lower()
    if ext == ".onnx":
        return _OnnxFrozenPolicy(checkpoint_path, low_level_obs_group, device)

    # ``.pt`` can be either a TorchScript archive (export_policy_as_jit) or a
    # raw state_dict (rsl_rl ``model_*.pt``). Try TorchScript first.
    try:
        jit_module = torch.jit.load(checkpoint_path, map_location=device)
        return _JitFrozenPolicy(jit_module, low_level_obs_group, device)
    except RuntimeError:
        pass

    agent_dict = agent_cfg.to_dict()
    # 凍結する下位 (歩行) 方策は上位 (ドリブル) 方策とは別ネットワークなので、その構造は
    # ``low_level_policy`` で指定する (rsl_rl_ppo_cfg_dribble.py)。後方互換のため未指定なら
    # 従来どおり ``policy`` にフォールバックする。
    low_level_cfg = agent_dict.get("low_level_policy")
    if low_level_cfg is None:
        print("[dribble] 'low_level_policy' が未指定のため 'policy' の構造で凍結方策を構築します。")
        low_level_cfg = agent_dict["policy"]
    policy_cfg = dict(low_level_cfg)
    policy_class_name = policy_cfg.pop("class_name", "ActorCritic")
    policy_class = {"ActorCritic": ActorCritic, "ActorCriticRecurrent": ActorCriticRecurrent}[policy_class_name]

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    # Frozen policy is only used for inference; critic_* tensors may have a
    # different shape (privileged obs) and would fail the size check even with
    # strict=False, so drop them entirely.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(("critic", "critic_obs_normalizer"))}

    # 構造は ``low_level_policy`` で指定するが、念のため checkpoint から actor の隠れ層
    # サイズを推定し、指定とずれていれば checkpoint 側に合わせる (load 失敗を防ぐ安全網)。
    # ずれた場合は警告を出すので、本来は ``low_level_policy`` を checkpoint に合わせること。
    actor_layer_idxs = sorted(
        int(k.split(".")[1]) for k in state_dict if k.startswith("actor.") and k.endswith(".weight")
    )
    if actor_layer_idxs:
        actor_weights = [state_dict[f"actor.{i}.weight"] for i in actor_layer_idxs]
        # out_features of every linear except the last (= action head) are hidden dims.
        inferred_hidden = [int(w.shape[0]) for w in actor_weights[:-1]]
        if inferred_hidden and inferred_hidden != list(policy_cfg.get("actor_hidden_dims", [])):
            print(
                f"[dribble] frozen policy actor_hidden_dims {policy_cfg.get('actor_hidden_dims')} "
                f"-> {inferred_hidden} (inferred from checkpoint)"
            )
            policy_cfg["actor_hidden_dims"] = inferred_hidden
            policy_cfg["critic_hidden_dims"] = inferred_hidden

    obs = env.get_observations()
    forced_groups = {"policy": [low_level_obs_group], "critic": [low_level_obs_group]}
    obs_groups = resolve_obs_groups(obs, forced_groups, ["critic"])

    frozen = policy_class(obs, obs_groups, env.num_actions, **policy_cfg).to(device)
    frozen.load_state_dict(state_dict, strict=False)

    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    return frozen
