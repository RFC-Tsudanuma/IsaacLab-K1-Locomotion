#!/usr/bin/env python3
"""歩行 (HistoryActorCritic, 1 ステップ観測 49 / critic 54) の checkpoint を、bhuman_walk_kick
(キック用の項を後ろに足したレイアウト, locomotion/history_layout.py の KICK_*) の形に移す。

足した入力・出力につながる重みは 0、足した入力の正規化は平均 0・分散 1 で始めるので、
変換後の方策は**足した入力の値に関係なく**歩行と同じ関節目標を出し、13 番目の出力
(歩行周波数) は 0 (= 歩行と同じ周波数) になる。最後にそれを乱数入力で確かめる。

    ~/k1_deploy_env/bin/python scripts/rsl_rl/convert_walk_to_bhuman_kick.py \
        --walk /home/akira/workspace/model_34995.pt \
        --out logs/rsl_rl/k1_bhuman_walk_kick/walk34995/model_0.pt

学習は ``--resume --load_run walk34995 --checkpoint model_0.pt --reset_noise_std 0.05`` で始める
(``--reset_noise_std`` を付けると optimizer の状態は読まない)。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

HISTORY = 100
RECENT = 4
COMMAND = 3
# history_layout.py と同じ値 (このスクリプトは IsaacLab 無しで動かすので import しない)
WALK_POLICY_STEP = 49
WALK_CRITIC_STEP = 54
KICK_EXTRA = [2, 1, 2, 3, 2, 1, 1]  # KICK_EXTRA_TERM_SPECS の次元
CRITIC_EXTRA = KICK_EXTRA + [2]  # + kick_pose_state
WALK_POLICY_TERMS = [3, 3, 3, 12, 12, 12, 4]
WALK_CRITIC_TERMS = [3] + WALK_POLICY_TERMS + [2]
NEW_ACTION_STD = 0.1  # 歩行周波数の action の初期ノイズ [Hz]


def pad_head(sd: dict, prefix: str, walk_step: int, extra: int) -> None:
    """cnn の入力チャネルと mlp 1 層目の「直近ステップ」列を 0 で足す。"""
    w = sd[f"{prefix}.cnn.0.weight"]  # (out, step, k)
    sd[f"{prefix}.cnn.0.weight"] = torch.cat((w, torch.zeros(w.shape[0], extra, w.shape[2])), dim=1)
    w = sd[f"{prefix}.mlp.0.weight"]  # (hidden, command + RECENT*step + cnn)
    cut = COMMAND + RECENT * walk_step
    sd[f"{prefix}.mlp.0.weight"] = torch.cat(
        (w[:, :cut], torch.zeros(w.shape[0], RECENT * extra), w[:, cut:]), dim=1
    )


def pad_normalizer(sd: dict, prefix: str, extra_total: int) -> None:
    n = extra_total * HISTORY
    for key, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        v = sd[f"{prefix}.{key}"]
        sd[f"{prefix}.{key}"] = torch.cat((v, torch.full((1, n), fill)), dim=1)


def head_forward(sd: dict, prefix: str, x: torch.Tensor, term_dims: list[int]) -> torch.Tensor:
    """HistoryEncoderHead.forward と同じ計算 (検証用)。"""
    direct, hist = x[:, :COMMAND], x[:, COMMAND:]
    blocks = torch.split(hist, [HISTORY * d for d in term_dims], dim=-1)
    recent = torch.cat([b.reshape(-1, HISTORY, d)[:, -RECENT:].reshape(-1, RECENT * d) for b, d in zip(blocks, term_dims)], -1)
    frames = torch.cat([b.reshape(-1, HISTORY, d) for b, d in zip(blocks, term_dims)], dim=-1).transpose(1, 2)
    h = F.elu(F.conv1d(frames, sd[f"{prefix}.cnn.0.weight"], sd[f"{prefix}.cnn.0.bias"], stride=4))
    h = F.elu(F.conv1d(h, sd[f"{prefix}.cnn.2.weight"], sd[f"{prefix}.cnn.2.bias"], stride=2)).flatten(1)
    y = torch.cat((direct, recent, h), dim=-1)
    i = 0
    while f"{prefix}.mlp.{i}.weight" in sd:
        y = F.linear(y, sd[f"{prefix}.mlp.{i}.weight"], sd[f"{prefix}.mlp.{i}.bias"])
        if f"{prefix}.mlp.{i + 2}.weight" in sd:
            y = F.elu(y)
        i += 2
    return y


def normalize(sd: dict, prefix: str, x: torch.Tensor) -> torch.Tensor:
    return (x - sd[f"{prefix}._mean"]) / (sd[f"{prefix}._std"] + 1e-2)


def walk_obs_to_kick(x_walk: torch.Tensor, walk_terms: list[int], extra: list[int]) -> torch.Tensor:
    """歩行の観測ベクトルの後ろに、足した項のブロック (乱数) を付ける。"""
    new = torch.randn(x_walk.shape[0], sum(extra) * HISTORY) * 3.0
    return torch.cat((x_walk, new), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    ck = torch.load(a.walk, map_location="cpu", weights_only=False)
    walk = {k: v.clone() for k, v in ck["model_state_dict"].items()}
    assert walk["actor.cnn.0.weight"].shape[1] == WALK_POLICY_STEP, "歩行の policy 観測が 49 次元ではない"
    assert walk["critic.cnn.0.weight"].shape[1] == WALK_CRITIC_STEP, "歩行の critic 観測が 54 次元ではない"
    sd = {k: v.clone() for k, v in walk.items()}

    pad_head(sd, "actor", WALK_POLICY_STEP, sum(KICK_EXTRA))
    pad_head(sd, "critic", WALK_CRITIC_STEP, sum(CRITIC_EXTRA))
    pad_normalizer(sd, "actor_obs_normalizer", sum(KICK_EXTRA))
    pad_normalizer(sd, "critic_obs_normalizer", sum(CRITIC_EXTRA))
    # 13 番目の出力 (歩行周波数) = 0
    w = sd["actor.mlp.6.weight"]
    sd["actor.mlp.6.weight"] = torch.cat((w, torch.zeros(1, w.shape[1])), dim=0)
    sd["actor.mlp.6.bias"] = torch.cat((sd["actor.mlp.6.bias"], torch.zeros(1)))
    std_key = "std" if "std" in sd else "log_std"
    new_std = NEW_ACTION_STD if std_key == "std" else torch.log(torch.tensor(NEW_ACTION_STD)).item()
    sd[std_key] = torch.cat((sd[std_key], torch.tensor([new_std])))

    # ---- 検証: 足した入力が何であっても、歩行と同じ出力になること ----
    torch.manual_seed(0)
    n = 64
    for prefix, norm, terms, extra, out_dim in (
        ("actor", "actor_obs_normalizer", WALK_POLICY_TERMS, KICK_EXTRA, 12),
        ("critic", "critic_obs_normalizer", WALK_CRITIC_TERMS, CRITIC_EXTRA, 1),
    ):
        x_walk = torch.randn(n, COMMAND + sum(terms) * HISTORY) * walk[f"{norm}._std"] + walk[f"{norm}._mean"]
        y_walk = head_forward(walk, prefix, normalize(walk, norm, x_walk), terms)
        x_kick = walk_obs_to_kick(x_walk, terms, extra)
        y_kick = head_forward(sd, prefix, normalize(sd, norm, x_kick), terms + extra)
        diff = (y_kick[:, :out_dim] - y_walk).abs().max().item()
        print(f"[check] {prefix}: max |kick - walk| = {diff:.2e}", end="")
        if prefix == "actor":
            print(f", 13th output max |.| = {y_kick[:, 12].abs().max().item():.2e}")
        else:
            print()
        assert diff < 1e-4, f"{prefix} の出力が歩行と一致しない"

    out = {"model_state_dict": sd, "iter": 0, "infos": {"converted_from": str(a.walk)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"[convert] saved {a.out}")
    for k in ("actor.cnn.0.weight", "actor.mlp.0.weight", "actor.mlp.6.weight", std_key,
              "actor_obs_normalizer._mean", "critic.cnn.0.weight", "critic.mlp.0.weight", "critic_obs_normalizer._mean"):
        print(f"  {k}: {tuple(walk[k].shape)} -> {tuple(sd[k].shape)}")


if __name__ == "__main__":
    main()
