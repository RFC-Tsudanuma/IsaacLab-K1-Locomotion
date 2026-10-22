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


def filter_unsupported_alg_args(train_cfg: dict) -> dict:
    """インストール済みの rsl-rl の PPO が受け付けない algorithm 設定を取り除く。

    新しい isaaclab_rl の RslRlPpoAlgorithmCfg には rsl-rl >= 4 向けの項目 (share_cnn_encoders など)
    があり、handle_deprecated_rsl_rl_cfg は rsl-rl 3.x 向けにそれを取り除かない。既定値のままの
    項目だけを捨て、既定値以外が指定されていたら (意図した設定が黙って無視されないよう) エラーにする。
    """
    import inspect

    alg_cfg = train_cfg.get("algorithm", {})
    class_name = alg_cfg.get("class_name", "PPO")
    import rsl_rl.algorithms as _algs

    alg_class = getattr(_algs, class_name, None)
    if alg_class is None:
        return train_cfg
    params = inspect.signature(alg_class.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return train_cfg
    defaults = {"share_cnn_encoders": False, "optimizer": "adam", "normalize_advantage_per_mini_batch": False}
    for key in list(alg_cfg):
        if key == "class_name" or key in params:
            continue
        if key in defaults and alg_cfg[key] == defaults[key]:
            print(f"[INFO]: algorithm.{key}={alg_cfg[key]!r} はインストール済みの rsl-rl の {class_name} に無いので外します。")
            del alg_cfg[key]
        else:
            raise ValueError(f"algorithm.{key}={alg_cfg[key]!r} はインストール済みの rsl-rl の {class_name} では使えません。")
    return train_cfg


def enable_critic_warmup(runner, num_iterations: int) -> None:
    """最初の ``num_iterations`` 回の PPO 更新では actor と行動ノイズ (std) を固定し、critic だけを学習する。

    Stage2 の開始直後、作り直したばかりの critic の価値推定が出鱈目なまま actor を更新すると
    歩き方が崩れる (開始直後に転倒率が 0.31 → 0.98 に上がる現象の対策候補、§9-1)。

    * actor / std の requires_grad を切る (勾配が None の引数は Adam が更新しない)。
      Adam のモーメントもこの間は積まれないので、解除後に古い勢いが残ることもない。
    * 学習率はこの間固定する (actor を固定すると KL が 0 になり、adaptive スケジュールが
      学習率を動かす前提が崩れるため。rsl-rl 3.0.1 は KL=0 では動かさないが念のため)。
    * actor の観測正規化器は通常どおり更新される (統計を集め続けるだけで、重みは固定)。
    """
    if num_iterations <= 0:
        return
    alg = runner.alg
    policy = alg.policy
    frozen = [p for name, p in policy.named_parameters() if not name.startswith("critic.") and p.requires_grad]
    for p in frozen:
        p.requires_grad_(False)
    fixed_lr = alg.learning_rate
    original_update = alg.update
    state = {"done": 0}

    def update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        state["done"] += 1
        if state["done"] < num_iterations:
            alg.learning_rate = fixed_lr
            for group in alg.optimizer.param_groups:
                group["lr"] = fixed_lr
        else:
            for p in frozen:
                p.requires_grad_(True)
            alg.update = original_update
            print(f"[INFO]: --critic_warmup_iters: {num_iterations} 回の更新が終わったので actor / std の固定を解除しました。")
        return result

    alg.update = update
    print(
        f"[INFO]: --critic_warmup_iters: 最初の {num_iterations} 回の更新は actor / std "
        f"({len(frozen)} テンソル) を固定し critic だけを学習します (学習率 {fixed_lr:.2e} で固定)。"
    )


def load_for_inference(runner, path: str) -> None:
    """推論用 (評価・再生) に checkpoint を読む。critic の形が今のレイアウトと違えば critic は読まない。

    2026-10-22 に critic を 66 → 64 次元にしたので、それ以前の checkpoint も評価できるようにする。
    actor が合わないときはエラーにする。
    """
    import torch

    ckpt = torch.load(path, weights_only=False, map_location=runner.device)
    state = ckpt["model_state_dict"]
    live = runner.alg.policy.state_dict()
    skipped = [k for k, v in state.items() if k.startswith("critic") and k in live and live[k].shape != v.shape]
    if not skipped:
        runner.load(path, load_optimizer=False)
        return
    state = {k: v for k, v in state.items() if k not in skipped}
    missing, unexpected = torch.nn.Module.load_state_dict(runner.alg.policy, state, strict=False)  # rsl_rl のものは bool を返す
    bad = [k for k in missing if not k.startswith("critic")]
    if bad or unexpected:
        raise RuntimeError(f"actor の読み込みが合いません: missing={bad}, unexpected={unexpected}")
    print(f"[INFO] critic の形が今のレイアウトと違うので読まない (推論には不要): {skipped}")
