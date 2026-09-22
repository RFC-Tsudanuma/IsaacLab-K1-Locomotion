# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""凍結 expert (歩行 / 回転) の出力を混合するゲーティング方策とその PPO (2026-09-23)。

RPG (Robust Policy Gating) の gating network をコマンド追従設定に移植したもの。
専門家同士の遷移追加学習 (`MultiExpertPPO`) は各 expert の能力低下を招き調整が難しかった
ため、**両 expert を完全に凍結**し、遷移の滑らかさは小さなゲート網だけで学習する。

構成 (環境は `K1FlatTransitionCfg` をそのまま使う):

* 両 expert は `HistoryActorCritic` を checkpoint から復元して凍結 (ゲート方策の
  ``nn.ModuleDict`` として保持するので、ゲートの checkpoint だけで自己完結する)。
* ゲート actor: 小さな MLP。入力は
  ``[モード one-hot (2), muxed コマンド (3), 直近 k フレームの policy 観測, 両 expert の
  提案アクション (24)]``。出力は 1 次元の logit ``z``、``w = sigmoid(z)`` が回転 expert の
  重み。実行アクションは ``a = (1 - w) * a_walk + w * a_turn``。
* PPO のアクション空間は ``z`` (1 次元) で、環境には混合後の 12 次元を渡す
  (`GatingPPO.act` が差し替える)。critic はモード one-hot + コマンド + 直近 k フレームの
  critic 観測 + expert アクションを見る。
* **非アクティブ expert へのコマンド枠マスク** (``mask_inactive_command``): 各 expert に
  渡す観測のコマンド枠 (``command`` グループと ``policy`` 履歴の velocity_commands 枠) を、
  自モードでない間は 0 にする。歩行 expert に ``[0, 0, Δψ/π]`` を ωz 指令として、回転
  expert に ``(vx, vy, ωz)`` を残り角として読ませない (どちらも自タスクの「停止指令」を
  受け取る = in-distribution)。環境側の「切替時に履歴のコマンド枠を 0 埋め」と合わせると、
  各 expert の見るコマンド列は「自分が担当でない間はずっと 0」になる。
  デプロイ側 (C++) も同じマスクをすること。
* ゲート専用の報酬は `GatingPPO` が内発報酬として加える (環境の報酬関数は無改造):
  - 平滑化: ``-gate_smooth_coef * |w_t - w_{t-1}|``
  - モード整合: モード切替から ``consistency_grace_steps`` 経過後、
    ``-gate_consistency_coef * |w_t - target|`` (target = 1 if turn else 0)。
    これが無いとゲートは平滑化のために切替を無限に先延ばしにできる
    (RPG 論文の "Task ID Check" 報酬に相当)。
  ログは ``Gate/*`` (rsl_rl の extras["log"] 経由)。

rsl_rl の OnPolicyRunner は方策・アルゴリズムを ``eval(class_name)`` で解決するため、
本モジュール末尾で `GatingActorCritic` / `GatingPPO` を注入している。
"""

from __future__ import annotations

import os
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.storage import RolloutStorage

from ..history_layout import CRITIC_TERM_SPECS, HISTORY_LENGTH, POLICY_TERM_SPECS
from ..mdp.commands import MODE_NAMES, MODE_TURN, MODE_WALK
from .history_actor_critic import HistoryActorCritic, RslRlHistoryActorCriticCfg, _recent_step_indices

# expert の並び順 = モード値 (0: walk, 1: turn)
EXPERT_MODES: tuple[int, ...] = (MODE_WALK, MODE_TURN)
COMMAND_TERM_NAME = "velocity_commands"


def _find_repo_root(start: str) -> str:
    """frozen_policies/ を含むディレクトリまで遡る (見つからなければ start を返す)。"""
    path = os.path.abspath(start)
    for _ in range(12):
        if os.path.isdir(os.path.join(path, "frozen_policies")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return os.path.abspath(start)


_REPO_ROOT = _find_repo_root(os.path.dirname(os.path.abspath(__file__)))


def _default_expert_checkpoint(name: str) -> str:
    """frozen_policies/best_<name>/ にある model_*.pt (複数あれば番号最大) を返す。

    凍結成果物は更新のたびにファイル名 (iteration 番号) が変わるので、ファイル名を
    決め打ちせずディレクトリから解決する。見つからなければ空文字 (構築時にエラー)。
    """
    import glob
    import re

    folder = os.path.join(_REPO_ROOT, "frozen_policies", f"best_{name}")
    candidates = glob.glob(os.path.join(folder, "model_*.pt"))
    if not candidates:
        return ""

    def _iter(path: str) -> int:
        m = re.search(r"model_(\d+)\.pt$", path)
        return int(m.group(1)) if m else -1

    return max(candidates, key=_iter)


DEFAULT_EXPERT_CHECKPOINTS: dict[str, str] = {name: _default_expert_checkpoint(name) for name in MODE_NAMES}


def _term_block(term_specs, name: str, history_length: int) -> torch.Tensor:
    """flatten 履歴グループ内で項 ``name`` が占める添字 (history_length × dim)。"""
    offset = 0
    for term_name, dim, _ in term_specs:
        if term_name == name:
            return torch.arange(offset, offset + history_length * dim, dtype=torch.long)
        offset += history_length * dim
    raise KeyError(f"term '{name}' not found in layout")


class GatingActorCritic(ActorCritic):
    """凍結 expert の混合重みを出すゲート方策。"""

    is_gating = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,  # 環境のアクション次元 (12)。ゲートの出力次元ではないので参照しない。
        expert_checkpoints: dict | None = None,
        expert_policy: dict | None = None,
        actor_hidden_dims=(128, 128),
        critic_hidden_dims=(256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        gate_history_steps: int = 4,
        gate_uses_expert_actions: bool = True,
        mask_inactive_command: bool = True,
        hard_switch_prior_logit: float = 4.0,
        mode_obs_group: str = "expert_mode",
        **kwargs,
    ):
        self.num_env_actions = int(num_actions)
        self.num_gate_actions = 1
        # 基底は obs_groups の全次元で actor/critic/normalizer を作る。下で全て置き換える。
        super().__init__(
            obs,
            obs_groups,
            self.num_gate_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )
        if mode_obs_group not in obs.keys():
            raise ValueError(f"GatingActorCritic: 観測グループ '{mode_obs_group}' がありません (K1FlatTransitionCfg を使うこと)。")
        self.mode_obs_group = mode_obs_group
        self.gate_history_steps = int(gate_history_steps)
        self.gate_uses_expert_actions = bool(gate_uses_expert_actions)
        self.mask_inactive_command = bool(mask_inactive_command)
        # ハード切替の事前分布: logit に mode に応じた ±prior を足す。MLP の最終層はゼロ初期化する
        # ので、学習開始時のゲートは「モード信号どおりのハード切替」(w ≈ sigmoid(±4) = 0.98/0.02)
        # と一致し、MLP は「切替をいつ・どれだけ滑らかにするか」の残差だけを学ぶ。
        # 0.5 混合から始めると両 expert の半々アクションで転倒し続け、学習が立ち上がらない。
        self.hard_switch_prior_logit = float(hard_switch_prior_logit)

        # --- 凍結 expert ---
        expert_cfg = dict(expert_policy or {})
        expert_cfg.pop("class_name", None)
        checkpoints = dict(DEFAULT_EXPERT_CHECKPOINTS)
        checkpoints.update({str(k): v for k, v in (expert_checkpoints or {}).items() if v})
        self.experts = nn.ModuleDict()
        for name, mode_idx in MODE_NAMES.items():
            path = checkpoints.get(name)
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"GatingActorCritic: expert '{name}' の checkpoint が見つかりません: {path}")
            expert = HistoryActorCritic(obs, obs_groups, self.num_env_actions, **expert_cfg)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            expert.load_state_dict(ckpt["model_state_dict"], strict=True)
            expert.eval()
            for p in expert.parameters():
                p.requires_grad_(False)
            self.experts[str(mode_idx)] = expert
            print(f"[GatingActorCritic] expert '{name}' (mode {mode_idx}) <- {path}")

        # --- 観測スライス用の添字 ---
        k = self.gate_history_steps
        self.register_buffer(
            "_policy_recent_idx", _recent_step_indices(POLICY_TERM_SPECS, HISTORY_LENGTH, k), persistent=False
        )
        self.register_buffer(
            "_critic_recent_idx", _recent_step_indices(CRITIC_TERM_SPECS, HISTORY_LENGTH, k), persistent=False
        )
        self.register_buffer(
            "_policy_cmd_idx", _term_block(POLICY_TERM_SPECS, COMMAND_TERM_NAME, HISTORY_LENGTH), persistent=False
        )

        # --- ゲート actor / critic (基底の巨大 MLP を置き換える) ---
        n_modes = len(EXPERT_MODES)
        cmd_dim = obs["command"].shape[-1]
        expert_dim = n_modes * self.num_env_actions if self.gate_uses_expert_actions else 0
        policy_step_dim = sum(d for _, d, _ in POLICY_TERM_SPECS)
        critic_step_dim = sum(d for _, d, _ in CRITIC_TERM_SPECS)
        actor_in = n_modes + cmd_dim + policy_step_dim * k + expert_dim
        critic_in = n_modes + cmd_dim + critic_step_dim * k + expert_dim
        self.actor = MLP(actor_in, self.num_gate_actions, list(actor_hidden_dims), activation)
        self.critic = MLP(critic_in, 1, list(critic_hidden_dims), activation)
        self.actor_obs_normalizer = EmpiricalNormalization(actor_in) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(critic_in) if critic_obs_normalization else nn.Identity()
        last = [m for m in self.actor.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        print(f"Gate actor MLP ({actor_in} -> {self.num_gate_actions}): {self.actor}")
        print(f"Gate critic MLP ({critic_in} -> 1): {self.critic}")

        # 直前に expert を評価した obs (弱参照) とその結果。act → evaluate が同じ obs で
        # 連続して呼ばれるので 2 度計算しない。id() だけだと解放後の再利用で誤ヒットしうる。
        self._expert_cache: tuple[weakref.ref, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # expert
    # ------------------------------------------------------------------
    def modes(self, obs: TensorDict) -> torch.Tensor:
        return obs[self.mode_obs_group][:, 0].round().long()

    def _expert_view(self, obs: TensorDict, active: torch.Tensor) -> TensorDict:
        """expert に渡す観測。非アクティブ env のコマンド枠 (command / policy 履歴) を 0 にする。"""
        if not self.mask_inactive_command:
            return obs
        keep = active.to(obs["command"].dtype).unsqueeze(1)
        command = obs["command"] * keep
        policy = obs["policy"].clone()
        policy[:, self._policy_cmd_idx] *= keep
        groups = {key: value for key, value in obs.items()}
        groups["command"] = command
        groups["policy"] = policy
        return TensorDict(groups, batch_size=obs.batch_size)

    @torch.no_grad()
    def expert_actions(self, obs: TensorDict) -> torch.Tensor:
        """両 expert の決定論アクション。shape (N, num_experts, num_env_actions)、並びは EXPERT_MODES。"""
        if self._expert_cache is not None and self._expert_cache[0]() is obs:
            return self._expert_cache[1]
        modes = self.modes(obs)
        outs = []
        for mode_idx in EXPERT_MODES:
            expert = self.experts[str(mode_idx)]
            outs.append(expert.act_inference(self._expert_view(obs, modes == mode_idx)))
        actions = torch.stack(outs, dim=1)
        try:
            self._expert_cache = (weakref.ref(obs), actions)
        except TypeError:
            self._expert_cache = None
        return actions

    @staticmethod
    def blend(turn_weight: torch.Tensor, expert_actions: torch.Tensor) -> torch.Tensor:
        """``a = (1 - w) * a_walk + w * a_turn``。turn_weight: (N, 1)。"""
        w = turn_weight.unsqueeze(-1)  # (N, 1, 1)
        weights = torch.cat([1.0 - w, w], dim=1)  # (N, 2, 1)
        return (weights * expert_actions).sum(dim=1)

    def turn_weight(self, gate_action: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(gate_action)

    # ------------------------------------------------------------------
    # gate observations
    # ------------------------------------------------------------------
    def _features(self, obs: TensorDict, recent_idx: torch.Tensor, history_group: str) -> torch.Tensor:
        modes = self.modes(obs)
        parts = [
            F.one_hot(modes, num_classes=len(EXPERT_MODES)).to(obs["command"].dtype),
            obs["command"],
            obs[history_group].index_select(-1, recent_idx),
        ]
        if self.gate_uses_expert_actions:
            parts.append(self.expert_actions(obs).flatten(1))
        return torch.cat(parts, dim=-1)

    def get_actor_obs(self, obs):
        return self._features(obs, self._policy_recent_idx, "policy")

    def get_critic_obs(self, obs):
        return self._features(obs, self._critic_recent_idx, "critic")

    def _prior_logit(self, obs: TensorDict) -> torch.Tensor:
        """モードに応じたハード切替の事前 logit (turn: +prior, walk: -prior)。shape (N, 1)。"""
        sign = (self.modes(obs) == MODE_TURN).to(obs["command"].dtype) * 2.0 - 1.0
        return (self.hard_switch_prior_logit * sign).unsqueeze(1)

    def update_distribution(self, obs, prior: torch.Tensor | None = None):
        mean = self.actor(obs)
        if prior is not None:
            mean = mean + prior
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = torch.distributions.Normal(mean, std)

    def act(self, obs, **kwargs):
        features = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(features, self._prior_logit(obs))
        return self.distribution.sample()

    def act_inference(self, obs):
        features = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(features) + self._prior_logit(obs)

    def act_inference_blended(self, obs: TensorDict) -> torch.Tensor:
        """決定論のゲート重みで混合した 12 次元アクション (play / デプロイ確認用)。"""
        w = self.turn_weight(self.act_inference(obs))
        return self.blend(w, self.expert_actions(obs))


class GatingPPO(PPO):
    """ゲート logit を PPO で学習し、環境には混合アクションを渡す。ゲート報酬は内発的に加える。"""

    def __init__(
        self,
        policy: GatingActorCritic,
        gate_smooth_coef: float = 5.0,
        gate_consistency_coef: float = 2.0,
        consistency_grace_steps: int = 25,
        mode_obs_group: str = "expert_mode",
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        if not getattr(policy, "is_gating", False):
            raise TypeError("GatingPPO には GatingActorCritic を使うこと。")
        self.gate_smooth_coef = float(gate_smooth_coef)
        self.gate_consistency_coef = float(gate_consistency_coef)
        self.consistency_grace_steps = int(consistency_grace_steps)
        self.mode_obs_group = mode_obs_group
        # act() 時点の状態 (process_env_step で使う)
        self._w: torch.Tensor | None = None
        self._mode: torch.Tensor | None = None
        # 直前 step の重み / 切替からの経過 step
        self._prev_w: torch.Tensor | None = None
        self._prev_valid: torch.Tensor | None = None
        self._since_switch: torch.Tensor | None = None

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        # PPO のアクションはゲート logit (環境の 12 次元ではない)
        self.storage = RolloutStorage(
            training_type, num_envs, num_transitions_per_env, obs, [self.policy.num_gate_actions], self.device
        )

    def act(self, obs):
        z = super().act(obs)  # transition にはゲート logit が保存される
        w = self.policy.turn_weight(z)
        self._w = w[:, 0]
        self._mode = self.policy.modes(obs)
        return self.policy.blend(w, self.policy.expert_actions(obs))

    def act_inference(self, obs):
        return self.policy.act_inference_blended(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        w, mode = self._w, self._mode
        n = w.shape[0]
        if self._prev_w is None:
            self._prev_w = w.clone()
            self._prev_valid = torch.ones(n, dtype=torch.bool, device=self.device)
            self._since_switch = torch.full((n,), self.consistency_grace_steps, dtype=torch.long, device=self.device)

        zero = torch.zeros_like(w)
        # 平滑化は 2 乗差分: Σ|Δw| は単調な遷移なら所要時間に関係なく 1 (望遠鏡和) で
        # ゆっくり切り替える動機にならない。Σ(Δw)^2 は T ステップに広げると 1/T になり、
        # クロスフェードを伸ばすほど得になる。
        smooth_pen = torch.where(self._prev_valid, (w - self._prev_w).square(), zero)
        target = (mode == MODE_TURN).to(w.dtype)
        after_grace = self._since_switch >= self.consistency_grace_steps
        deviation = (w - target).abs()
        consistency_pen = torch.where(after_grace, deviation, zero)
        rewards = rewards - self.gate_smooth_coef * smooth_pen - self.gate_consistency_coef * consistency_pen

        # 次 step 用の状態更新 (obs は step 後の観測)
        mode_next = self.policy.modes(obs)
        done_mask = dones.to(self.device) > 0
        self._since_switch = torch.where(
            mode_next != mode, torch.zeros_like(self._since_switch), self._since_switch + 1
        )
        self._since_switch[done_mask] = self.consistency_grace_steps
        self._prev_w = w.clone()
        self._prev_valid = ~done_mask

        log = extras.setdefault("log", {})
        log["Gate/turn_weight_mean"] = w.mean()
        for name, mode_idx in MODE_NAMES.items():
            sel = mode == mode_idx
            if sel.any():
                log[f"Gate/turn_weight_in_{name}"] = w[sel].mean()
        log["Gate/smooth_penalty"] = smooth_pen.mean()
        log["Gate/consistency_penalty"] = consistency_pen.mean()
        log["Gate/mismatch_frac"] = torch.where(after_grace, (deviation > 0.5).to(w.dtype), zero).mean()

        super().process_env_step(obs, rewards, dones, extras)


@configclass
class RslRlGatingActorCriticCfg(RslRlPpoActorCriticCfg):
    """`GatingActorCritic` 用の設定。"""

    class_name: str = "GatingActorCritic"

    init_noise_std: float = 0.5
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [128, 128]
    critic_hidden_dims: list[int] = [256, 128]
    activation: str = "elu"

    expert_checkpoints: dict = dict(DEFAULT_EXPERT_CHECKPOINTS)
    """凍結 expert の checkpoint (キー "walk" / "turn")。既定は frozen_policies/best_<mode>/ 内の
    model_*.pt (番号最大)。train.py の ``--frozen_ckpt MODE=PATH`` で上書きできる。"""

    expert_policy: RslRlHistoryActorCriticCfg = RslRlHistoryActorCriticCfg(
        init_noise_std=0.7207805082202461,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    """expert (HistoryActorCritic) のアーキテクチャ。checkpoint と一致させること。"""

    gate_history_steps: int = 4
    """ゲート入力に使う直近フレーム数。"""

    gate_uses_expert_actions: bool = True
    """両 expert の提案アクション (24 次元) をゲート入力に含めるか。"""

    mask_inactive_command: bool = True
    """非アクティブ expert に渡す観測のコマンド枠を 0 にするか (デプロイ側と一致させる)。"""

    hard_switch_prior_logit: float = 4.0
    """モード信号どおりのハード切替を事前分布として logit に足す量 (0 で無効)。
    学習開始時のゲートが w ≈ 0.98 / 0.02 のハード切替になり、MLP は残差 (滑らかさ) だけ学ぶ。"""

    mode_obs_group: str = "expert_mode"


@configclass
class RslRlGatingPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """`GatingPPO` 用の設定。"""

    class_name: str = "GatingPPO"

    gate_smooth_coef: float = 5.0
    """``-coef * (w_t - w_{t-1})^2`` (ゲート重みの時間平滑化)。ハード切替 1 回 ≈ 0.9*coef、
    5 step のクロスフェード ≈ 0.2*coef。"""

    gate_consistency_coef: float = 2.0
    """``-coef * |w_t - target|`` (猶予後のモード整合)。無いとゲートは切替を無限に先延ばしにできる。"""

    consistency_grace_steps: int = 25
    """モード切替からこの step 数 (0.02 s 刻み、既定 0.5 s) はモード整合ペナルティを免除する
    (= クロスフェードに使える時間)。"""

    mode_obs_group: str = "expert_mode"


# OnPolicyRunner は eval(class_name) を on_policy_runner モジュールの名前空間で評価する。
import rsl_rl.runners.on_policy_runner as _rsl_rl_on_policy_runner  # noqa: E402

_rsl_rl_on_policy_runner.GatingActorCritic = GatingActorCritic
_rsl_rl_on_policy_runner.GatingPPO = GatingPPO
