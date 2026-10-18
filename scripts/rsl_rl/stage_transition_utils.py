# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""段階的な fine-tune (例: ドリブル Stage1 → Stage2) のための train.py 補助関数。

* :func:`reset_critic`: critic のネットワークと観測正規化器を初期化し直す (``--reset_critic``)
* :func:`reestimate_obs_normalizer`: 指定した観測項の正規化統計を、現在の環境で集めた統計で
  上書きする (``--reestimate_obs_norm_terms``)
* :func:`attach_curriculum_state_io` / :func:`load_curriculum_state`: ``state_dict`` を持つ
  カリキュラム項の状態を ``model_*_curriculum.json`` に保存・復元する
"""

from __future__ import annotations

import json
import math
import os

import torch


def reset_critic(policy) -> None:
    """critic のネットワーク (全層) と critic の観測正規化器を初期化し直す。

    Stage1 で学習した価値関数は Stage2 の報酬 (ドリブル報酬が加わる) と合わないため捨てる。
    critic の正規化器も、Stage1 ではドリブル用の項が常に 0 だったため統計が使えないので戻す。
    """
    count = 0
    for module in policy.critic.modules():
        if module is not policy.critic and hasattr(module, "reset_parameters"):
            module.reset_parameters()
            count += 1
    norm = getattr(policy, "critic_obs_normalizer", None)
    if norm is not None and hasattr(norm, "_mean"):
        with torch.no_grad():
            norm._mean.zero_()
            norm._var.fill_(1.0)
            norm._std.fill_(1.0)
            norm.count.zero_()
    print(f"[INFO]: --reset_critic: critic の {count} 層と critic の観測正規化器を初期化しました。")


def _actor_obs_term_indices(env, policy, term_names: list[str]) -> tuple[torch.Tensor, list[str]]:
    """actor 入力ベクトル (obs_groups['policy'] の連結) の中で、指定した観測項が占める添字を返す。"""
    obs_manager = env.unwrapped.observation_manager
    indices: list[int] = []
    offset = 0
    found: list[str] = []
    for group in policy.obs_groups["policy"]:
        names = obs_manager.active_terms[group]
        dims = [int(math.prod(shape)) for shape in obs_manager.group_obs_term_dim[group]]
        for name, dim in zip(names, dims):
            if name in term_names:
                indices.extend(range(offset, offset + dim))
                found.append(f"{group}/{name}")
            offset += dim
    missing = [t for t in term_names if not any(f.endswith("/" + t) for f in found)]
    if missing:
        raise ValueError(f"--reestimate_obs_norm_terms: actor の観測に無い項があります: {missing}")
    return torch.tensor(indices, dtype=torch.long), found


@torch.no_grad()
def reestimate_obs_normalizer(env, policy, term_names: list[str], num_steps: int) -> None:
    """指定した観測項の actor 正規化統計 (mean / var / std) を、現在の環境で集めた値で上書きする。

    前段 (Stage1) ではこれらの項が常に 0 で、統計が意味を持たない。そのまま Stage2 に入ると
    実際の値が誤ったスケールで正規化される (§9-2)。現在のポリシーで ``num_steps`` ステップ
    動かして観測を集め、その項の統計だけを置き換える (他の項と count は変えない)。

    環境の common_step_counter は集計前の値に戻す (報酬のフェードインの進み具合を変えないため)。
    """
    norm = getattr(policy, "actor_obs_normalizer", None)
    if norm is None or not hasattr(norm, "_mean"):
        raise ValueError("--reestimate_obs_norm_terms: actor の観測正規化が無効です")
    idx, found = _actor_obs_term_indices(env, policy, term_names)
    idx = idx.to(norm._mean.device)
    step_counter = env.unwrapped.common_step_counter

    was_training = policy.training
    policy.eval()
    total = None
    total_sq = None
    n = 0
    obs = env.get_observations()
    for _ in range(num_steps):
        x = policy.get_actor_obs(obs).index_select(-1, idx).double()
        total = x.sum(0) if total is None else total + x.sum(0)
        total_sq = (x * x).sum(0) if total_sq is None else total_sq + (x * x).sum(0)
        n += x.shape[0]
        obs, _, _, _ = env.step(policy.act_inference(obs))
    policy.train(was_training)
    env.unwrapped.common_step_counter = step_counter

    mean = (total / n).float()
    var = (total_sq / n - (total / n) ** 2).clamp(min=0.0).float()
    before_std = norm._std[0, idx].clone()
    norm._mean[0, idx] = mean
    norm._var[0, idx] = var
    norm._std[0, idx] = torch.sqrt(var)
    print(
        f"[INFO]: --reestimate_obs_norm_terms: {found} の正規化統計を {num_steps} ステップ × "
        f"{n // max(num_steps, 1)} env で集めた値で上書きしました。\n"
        f"        std (前→後): median {before_std.median().item():.4f} → {norm._std[0, idx].median().item():.4f}, "
        f"min {before_std.min().item():.4f} → {norm._std[0, idx].min().item():.4f}, "
        f"max {before_std.max().item():.4f} → {norm._std[0, idx].max().item():.4f}"
    )


def _stateful_curriculum_terms(env) -> dict:
    cm = env.unwrapped.curriculum_manager
    terms = {}
    for name, cfg in zip(cm.active_terms, cm._term_cfgs):
        func = cfg.func
        if hasattr(func, "state_dict") and hasattr(func, "load_state_dict"):
            terms[name] = func
    return terms


def curriculum_state_path(model_path: str) -> str:
    return os.path.splitext(model_path)[0] + "_curriculum.json"


def attach_curriculum_state_io(runner, env) -> None:
    """runner.save をラップし、checkpoint と同じ名前の ``*_curriculum.json`` にカリキュラムの状態も書く。"""
    terms = _stateful_curriculum_terms(env)
    if not terms:
        return
    original_save = runner.save

    def save(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        with open(curriculum_state_path(path), "w") as f:
            json.dump({name: term.state_dict() for name, term in terms.items()}, f, indent=2)

    runner.save = save
    print(f"[INFO]: カリキュラムの状態を model_*_curriculum.json に保存します: {list(terms)}")


def load_curriculum_state(env, model_path: str) -> None:
    """``model_path`` に対応する ``*_curriculum.json`` があれば、カリキュラムの状態を復元する。"""
    terms = _stateful_curriculum_terms(env)
    path = curriculum_state_path(model_path)
    if not terms:
        return
    if not os.path.isfile(path):
        print(f"[INFO]: カリキュラムの状態ファイルが無いので初期状態から始めます: {path}")
        return
    with open(path) as f:
        state = json.load(f)
    for name, term in terms.items():
        if name in state:
            term.load_state_dict(state[name])
            print(f"[INFO]: カリキュラム '{name}' の状態を復元しました: {state[name]}")
