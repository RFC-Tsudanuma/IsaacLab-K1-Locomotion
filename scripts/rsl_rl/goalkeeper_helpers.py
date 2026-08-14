# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the hierarchical goalkeeper train/play scripts.

This module is imported AFTER ``AppLauncher`` has started in both
``train_goalkeeper.py`` and ``play_goalkeeper.py``. It is kept side-effect free
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
        action_deadband: float = 0.0,
        cmd_scale_range=None,
        cmd_delay_range=None,
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

        # -- 指令デッドバンド (norm 基準) --
        # ガウス方策は厳密な 0 を出せないので、放置すると上位が常に微小な指令を出し続け、
        # 下位の「指令ノルム < cmd_threshold なら位相ゼロ = 停止」規約に一生入れず、
        # キーパーがその場足踏みし続ける。3 軸のノルムがこの値未満なら全成分を 0 に落とす。
        #
        # ★ 軸別のデッドバンドにしてはいけない。07-28 の下位には横移動中に
        #   「yaw ≈ 10°/s のドリフト」と「後退 ≈ 0.10 m/s のドリフト」があり、上位は
        #   それを打ち消す小さな定常オフセット (wz ≈ -0.175, vx ≈ +0.10) を出し続ける
        #   必要がある。軸別に閾値を掛けるとこの補正が潰れる。
        #   下位自身の停止判定も norm 基準 (rough_env_cfg の _COMMAND_THRESHOLD) なので
        #   規約としてもこちらが整合する。
        self.action_deadband = float(action_deadband)

        # -- 下位エンベロープの DR (per-episode 固定) --
        # 上位は「sim の下位」の上でタイミング (何秒前に動き出せば間に合うか) を学ぶので、
        # 実機の下位が sim より遅い/遅延が大きいとその前提ごと崩れる。指令にスケールと
        # 遅延を掛けて「少し鈍い下位」も学習分布に入れておく。
        #   cmd_scale_range: (lo, hi) 指令に掛かる倍率。1.3 × U(0.8,1.0) = 実効 1.04〜1.30 m/s。
        #   cmd_delay_range: (lo, hi) 指令が下位に届くまでの遅延 [tick] (整数、両端含む)。
        # どちらも None なら無効 (既存タスクの挙動は変わらない)。
        self._cmd_scale_range = tuple(float(v) for v in cmd_scale_range) if cmd_scale_range else None
        self._cmd_delay_range = tuple(int(v) for v in cmd_delay_range) if cmd_delay_range else None
        self._has_cmd_dr = (self._cmd_scale_range is not None) or (self._cmd_delay_range is not None)

        self._cmd_scale = torch.ones(self.num_envs, 1, device=self.device)
        if self._cmd_delay_range is not None:
            lo, hi = self._cmd_delay_range
            if lo < 0 or hi < lo:
                raise ValueError(f"cmd_delay_range must satisfy 0 <= lo <= hi, got {self._cmd_delay_range}")
            self._max_delay = hi
            # 遅延用リングバッファ: (max_delay + 1, num_envs, action_dim)。
            # 添字 0 が最新、k が k tick 前。
            self._cmd_hist = torch.zeros(
                self._max_delay + 1, self.num_envs, self.num_actions, device=self.device
            )
            self._cmd_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            self._max_delay = 0
            self._cmd_hist = None
            self._cmd_delay = None
        # 起動時に全 env ぶんサンプルしておく (最初のエピソードから DR が効くように)
        self._resample_cmd_dr(torch.arange(self.num_envs, device=self.device))

    # -- 下位エンベロープ DR --
    def _resample_cmd_dr(self, env_ids: torch.Tensor) -> None:
        """``env_ids`` のエピソード固定 DR パラメータを引き直し、遅延履歴を消す。"""
        if env_ids.numel() == 0:
            return
        if self._cmd_scale_range is not None:
            lo, hi = self._cmd_scale_range
            self._cmd_scale[env_ids, 0] = torch.rand(env_ids.numel(), device=self.device) * (hi - lo) + lo
        if self._cmd_delay_range is not None:
            lo, hi = self._cmd_delay_range
            self._cmd_delay[env_ids] = torch.randint(
                lo, hi + 1, (env_ids.numel(),), device=self.device, dtype=torch.long
            )
            # 前のエピソードの指令が遅延バッファ経由で漏れないよう 0 で埋める
            self._cmd_hist[:, env_ids, :] = 0.0

    def _apply_cmd_delay(self, cmd: torch.Tensor) -> torch.Tensor:
        """指令をリングバッファに積み、env ごとの遅延ぶん過去の指令を返す。"""
        if self._cmd_hist is None:
            return cmd
        # 1 tick ずらして最新を先頭に入れる (max_delay は高々数 tick なので roll で十分)
        self._cmd_hist = torch.roll(self._cmd_hist, shifts=1, dims=0)
        self._cmd_hist[0] = cmd
        # gather: env ごとに異なる遅延 tick の行を引く
        idx = self._cmd_delay.view(1, -1, 1).expand(1, self.num_envs, self.num_actions)
        return self._cmd_hist.gather(0, idx).squeeze(0)

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
        self._resample_cmd_dr(torch.arange(self.num_envs, device=self.device))
        return self.env.reset()

    def get_observations(self):
        return self.env.get_observations()

    def step(self, action: torch.Tensor):
        # Per-axis clip: self.action_clip is (num_actions,); broadcast over batch.
        cmd = torch.clamp(action.to(self.device), -self.action_clip, self.action_clip).detach()

        # デッドバンド (norm 基準) → 遅延 → スケール の順で適用する。
        # デッドバンドは「上位側の出力整形」なので実機の推論ループにも同じものを実装する
        # 前提で先に掛ける。遅延とスケールは「下位/伝送側の性質」なので後段。
        if self.action_deadband > 0.0:
            below = torch.norm(cmd, dim=1, keepdim=True) < self.action_deadband
            cmd = torch.where(below, torch.zeros_like(cmd), cmd)
        cmd = self._apply_cmd_delay(cmd)
        # ★ _prev_high_action にはスケール前の値を書く。スケールは「下位が指令ほど
        #   動けない」という plant 側の性質であって上位の出力ではないので、これを
        #   観測に混ぜると実機に無い情報 (自分の指令が何倍で効いているか) を
        #   ポリシーが直接読めてしまう。ポリシーは自分の動きの結果から推定するべき。
        #   なお gait_phase もこのバッファから作られるが、位相の停止判定は norm と
        #   閾値の比較だけなので 0.8〜1.0 のスケール差で判定は変わらない。
        delivered = cmd * self._cmd_scale

        # Frozen policy is fixed: never need autograd for it. Wrap the whole
        # low-level inference in inference_mode so that we don't build a
        # computation graph even if the outer caller (rsl_rl rollout) forgets to.
        with torch.inference_mode():
            env_obs = self.env.get_observations()

            low_group_tensor = env_obs[self.low_level_obs_group].clone()
            s, e = self._cmd_slot
            low_group_tensor[:, s:e] = delivered
            low_obs = {k: env_obs[k] for k in env_obs.keys()}
            low_obs[self.low_level_obs_group] = low_group_tensor

            joint_action = self.low_level_policy.act_inference(low_obs)

        # Stash *before* env.step so that the post-step obs (incl. any auto-reset
        # branches) observes the correct prev high action. For envs that get
        # auto-reset inside env.step, the reset event clears this slot.
        self.env.unwrapped._prev_high_action.copy_(cmd)

        obs, rew, dones, extras = self.env.step(joint_action)

        # env.step 内で auto-reset された env は次のエピソードなので DR を引き直す。
        # DR を使わない (既存の) タスクでは毎ステップの nonzero() が無駄なので丸ごと飛ばす。
        if self._has_cmd_dr and dones is not None:
            done_ids = dones.to(device=self.device).flatten().nonzero(as_tuple=False).flatten()
            self._resample_cmd_dr(done_ids)

        return obs, rew, dones, extras

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
    # ``low_level_policy`` で指定する (rsl_rl_ppo_cfg.py)。後方互換のため未指定なら
    # 従来どおり ``policy`` にフォールバックする。
    low_level_cfg = agent_dict.get("low_level_policy")
    if low_level_cfg is None:
        print("[goalkeeper] 'low_level_policy' が未指定のため 'policy' の構造で凍結方策を構築します。")
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
                f"[goalkeeper] frozen policy actor_hidden_dims {policy_cfg.get('actor_hidden_dims')} "
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
