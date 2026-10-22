# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""歩行ポリシー (HistoryActorCritic、1 ステップ 49 次元) の checkpoint を、ドリブル用の
観測レイアウト (policy 59 次元 / critic 64 次元) に広げる。

ドリブル用の項は各フレームの末尾に追加されている (history_layout.DRIBBLE_TARGET_*_TERM_SPECS)
ので、形が変わるのは次のテンソルだけ:

* ``{actor,critic}.cnn.0.weight`` (out, C, k): 追加チャネル C の重みを 0 で足す
* ``{actor,critic}.mlp.0.weight`` (out, direct + C×4 + cnn_out): 直近 4 ステップ部分の末尾
  (CNN 特徴の手前) に、追加項の列を 0 で挿入する
* ``{actor,critic}_obs_normalizer.{_mean,_var,_std}``: 末尾に追加項のブロック (H×次元) を
  mean=0 / var=std=1 で足す (count はそのまま)

追加分の重みが 0 なので、変換後のポリシーは「ドリブル用の観測が何であっても」変換前と
同じ行動・価値を出す。``--verify`` で実際の数値で確認する (同じ観測を入れて出力を比べる)。

optimizer の状態は形が合わないので捨てる (resume 時は --reset_noise_std を付けて
optimizer を読まないこと)。iteration は既定で 0 に戻す (Stage1 のカリキュラムを最初から回すため)。

使い方 (Isaac Sim は起動しない。rsl_rl 3.x が import できる python で実行する):

    python scripts/rsl_rl/widen_history_checkpoint.py \
        --input frozen_policies/best_walk/model_52500.pt \
        --output logs/rsl_rl/k1_dribble_target_walk_phase/best_walk_widened/model_0.pt --verify
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import types

import torch

_PKG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/dribble_target",
)


def _import_dribble_modules():
    """dribble_target パッケージの history_layout / HistoryActorCritic を、パッケージの
    __init__ (gym 登録 → env cfg → Isaac Sim の import) を通さずに読み込む。"""
    name = "_dribble_target_pkg"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [os.path.abspath(_PKG_DIR)]
        sys.modules[name] = pkg
        agents = types.ModuleType(f"{name}.agents")
        agents.__path__ = [os.path.join(os.path.abspath(_PKG_DIR), "agents")]
        sys.modules[f"{name}.agents"] = agents
    # history_actor_critic は設定クラス (RslRlHistoryActorCriticCfg) の定義のために
    # isaaclab_rl.rsl_rl / isaaclab.utils を import するが、これらは Isaac Sim (pxr) を要求する。
    # ここで使うのはネットワーク本体だけなので、import できない場合は最小のスタブで代用する。
    try:
        import isaaclab_rl.rsl_rl  # noqa: F401
    except Exception:
        stub_rl = types.ModuleType("isaaclab_rl.rsl_rl")
        stub_rl.RslRlPpoActorCriticCfg = type("RslRlPpoActorCriticCfg", (), {})
        sys.modules.setdefault("isaaclab_rl", types.ModuleType("isaaclab_rl"))
        sys.modules["isaaclab_rl.rsl_rl"] = stub_rl
        stub_utils = types.ModuleType("isaaclab.utils")
        stub_utils.configclass = lambda cls: cls
        sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
        sys.modules["isaaclab.utils"] = stub_utils
    layout = importlib.import_module(f"{name}.history_layout")
    return layout


def _insert_columns(weight: torch.Tensor, at: int, count: int) -> torch.Tensor:
    zeros = torch.zeros(weight.shape[0], count, dtype=weight.dtype, device=weight.device)
    return torch.cat([weight[:, :at], zeros, weight[:, at:]], dim=1)


def widen_state_dict(state: dict, layout, history_length: int, mlp_history_steps: int, direct_dim: int) -> dict:
    """model_state_dict を walk レイアウトから dribble_target レイアウトに広げた新しい dict を返す。"""
    out = dict(state)
    heads = {
        "actor": (layout.POLICY_TERM_SPECS, layout.DRIBBLE_TARGET_POLICY_TERM_SPECS),
        "critic": (layout.CRITIC_TERM_SPECS, layout.DRIBBLE_TARGET_CRITIC_TERM_SPECS),
    }
    for head, (old_specs, new_specs) in heads.items():
        if new_specs[: len(old_specs)] != old_specs:
            raise ValueError(f"{head}: 新しいレイアウトの先頭が旧レイアウトと一致しません (末尾追加のみ対応)")
        old_c = layout.term_specs_dim(old_specs)
        new_c = layout.term_specs_dim(new_specs)
        add_c = new_c - old_c

        cnn_w = state[f"{head}.cnn.0.weight"]
        if cnn_w.shape[1] != old_c:
            raise ValueError(f"{head}.cnn.0.weight の入力チャネル {cnn_w.shape[1]} が旧レイアウト {old_c} と違います")
        zeros = torch.zeros(cnn_w.shape[0], add_c, cnn_w.shape[2], dtype=cnn_w.dtype, device=cnn_w.device)
        out[f"{head}.cnn.0.weight"] = torch.cat([cnn_w, zeros], dim=1)

        mlp_w = state[f"{head}.mlp.0.weight"]
        recent_end = direct_dim + old_c * mlp_history_steps
        out[f"{head}.mlp.0.weight"] = _insert_columns(mlp_w, recent_end, add_c * mlp_history_steps)

        norm = f"{head}_obs_normalizer"
        if f"{norm}._mean" in state:
            old_len = direct_dim + old_c * history_length
            if state[f"{norm}._mean"].shape[-1] != old_len:
                raise ValueError(f"{norm} の次元 {state[f'{norm}._mean'].shape[-1]} が想定 {old_len} と違います")
            n_add = add_c * history_length
            for key, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
                t = state[f"{norm}.{key}"]
                out[f"{norm}.{key}"] = torch.cat([t, torch.full((1, n_add), fill, dtype=t.dtype, device=t.device)], 1)
    return out


def _build_policy(layout_name: str, num_actions: int, history_length: int, ckpt_state: dict):
    from tensordict import TensorDict

    hac = importlib.import_module("_dribble_target_pkg.agents.history_actor_critic")
    layout = importlib.import_module("_dribble_target_pkg.history_layout")
    pol_specs, cri_specs = layout.term_specs_by_name(layout_name)
    obs = TensorDict(
        {
            "command": torch.zeros(1, 3),
            "policy": torch.zeros(1, layout.term_specs_dim(pol_specs) * history_length),
            "critic": torch.zeros(1, layout.term_specs_dim(cri_specs) * history_length),
        },
        batch_size=[1],
    )
    policy = hac.HistoryActorCritic(
        obs,
        {"policy": ["command", "policy"], "critic": ["command", "critic"]},
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        obs_layout=layout_name,
    )
    # critic の次元はレイアウトの変更 (2026-10-22 に 66 → 64) で変わりうる。actor だけ使う用途 (エクスポート) のため、
    # 形の合わない critic のテンソルは読まない (actor が合わなければエラーにする)。
    live = policy.state_dict()
    skipped = [k for k, v in ckpt_state.items() if k.startswith("critic") and k in live and live[k].shape != v.shape]
    state = {k: v for k, v in ckpt_state.items() if k not in skipped}
    policy.load_state_dict(state, strict=not skipped)
    if skipped:
        print(f"[load] critic の形が今のレイアウトと違うので読まない: {skipped}")
    # float32 だと「0 の列が増えて行列積の足し算の順序が変わる」だけで 1e-5 程度の丸め誤差が出るので、
    # 等価性は float64 で確認する (追加分の重みが 0 なら float64 では誤差がほぼ 0 になるはず)。
    policy.double()
    policy.eval()
    return policy


def _random_frames(n: int, history_length: int, step_dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, history_length, step_dim, generator=g)


def _flatten(frames: torch.Tensor, term_specs) -> torch.Tensor:
    """(N, H, C) のフレーム列を「項ごとの (H × dim) ブロック」の flatten レイアウトにする。"""
    parts, ch = [], 0
    for _, dim, _ in term_specs:
        parts.append(frames[:, :, ch : ch + dim].reshape(frames.shape[0], -1))
        ch += dim
    return torch.cat(parts, dim=-1)


@torch.no_grad()
def verify(old_state: dict, new_state: dict, layout, num_actions: int, history_length: int, n: int = 64) -> float:
    """同じ観測 (歩行の 49 次元は共通、ドリブル用の 10 次元は乱数) を入れて、変換前後の出力差の最大値を返す。"""
    from tensordict import TensorDict

    old = _build_policy("walk", num_actions, history_length, old_state)
    new = _build_policy("dribble_target", num_actions, history_length, new_state)
    worst = 0.0
    for seed in range(3):
        cmd = torch.randn(n, 3, generator=torch.Generator().manual_seed(100 + seed)).double()
        pol_old_c = layout.term_specs_dim(layout.POLICY_TERM_SPECS)
        pol_new_c = layout.term_specs_dim(layout.DRIBBLE_TARGET_POLICY_TERM_SPECS)
        cri_old_c = layout.term_specs_dim(layout.CRITIC_TERM_SPECS)
        cri_new_c = layout.term_specs_dim(layout.DRIBBLE_TARGET_CRITIC_TERM_SPECS)
        pol = _random_frames(n, history_length, pol_new_c, seed).double() * 2.0
        cri = _random_frames(n, history_length, cri_new_c, 10 + seed).double() * 2.0
        obs_old = TensorDict(
            {
                "command": cmd,
                "policy": _flatten(pol[:, :, :pol_old_c], layout.POLICY_TERM_SPECS),
                "critic": _flatten(cri[:, :, :cri_old_c], layout.CRITIC_TERM_SPECS),
            },
            batch_size=[n],
        )
        obs_new = TensorDict(
            {
                "command": cmd,
                "policy": _flatten(pol, layout.DRIBBLE_TARGET_POLICY_TERM_SPECS),
                "critic": _flatten(cri, layout.DRIBBLE_TARGET_CRITIC_TERM_SPECS),
            },
            batch_size=[n],
        )
        a_old, a_new = old.act_inference(obs_old), new.act_inference(obs_new)
        v_old, v_new = old.evaluate(obs_old), new.evaluate(obs_new)
        diff_a = (a_old - a_new).abs().max().item()
        diff_v = (v_old - v_new).abs().max().item()
        print(
            f"[verify] seed={seed}: |Δaction|max={diff_a:.3e}  |Δvalue|max={diff_v:.3e}"
            f"  (action の大きさ max={a_old.abs().max().item():.3f}, value max={v_old.abs().max().item():.3f})"
        )
        worst = max(worst, diff_a, diff_v)
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="歩行ポリシーの checkpoint (model_*.pt)")
    parser.add_argument("--output", required=True, help="書き出し先 (model_*.pt)")
    parser.add_argument("--iter", type=int, default=0, help="書き出す checkpoint の iteration (既定 0)")
    parser.add_argument("--verify", action="store_true", help="変換前後で出力が一致することを数値で確認する")
    parser.add_argument("--tolerance", type=float, default=1e-9, help="float64 での出力差の許容値")
    args = parser.parse_args()

    layout = _import_dribble_modules()
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    num_actions = state["std"].shape[0] if "std" in state else state["log_std"].shape[0]
    history_length = layout.HISTORY_LENGTH

    new_state = widen_state_dict(state, layout, history_length, layout.MLP_HISTORY_STEPS, layout.COMMAND_DIM)
    for key in sorted(new_state):
        if new_state[key].shape != state[key].shape:
            print(f"[widen] {key}: {tuple(state[key].shape)} -> {tuple(new_state[key].shape)}")

    if args.verify:
        worst = verify(state, new_state, layout, num_actions, history_length)
        if worst > args.tolerance:
            raise SystemExit(f"[verify] NG: 変換前後の出力差 {worst:.3e} が許容 {args.tolerance:.1e} を超えました")
        print(f"[verify] OK: 変換前後の出力差の最大値 {worst:.3e} (許容 {args.tolerance:.1e})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": new_state,
            "iter": args.iter,
            "infos": {
                "widened_from": os.path.abspath(args.input),
                "widened_from_iter": ckpt.get("iter"),
                "obs_layout": "dribble_target",
            },
        },
        args.output,
    )
    print(f"[widen] wrote {args.output} (iter={args.iter}, optimizer state は含めない)")


if __name__ == "__main__":
    main()
