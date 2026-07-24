# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""歩行チェックポイントからの actor 部分ウォームスタート。

B-Human 方式の Stage 2 (観測と報酬を差し替えてキックポリシーを学習) では、
歩行ポリシーの重みを初期値として引き継ぐ。ただし観測次元が違う
(歩行 49 → キック 62 など) ため `load_state_dict(strict=False)` は第1層を
黙ってスキップして実質未初期化になる。本モジュールは第1層だけ列単位で
マッピングして引き継ぐ:

    * actor 第1層 (in→256): 先頭 `old_in` 列をコピー、追加観測の列はゼロ初期化。
      → 初期ポリシーは追加観測を完全に無視し、挙動は歩行ポリシーと一致する
        (velocity_commands スロットに入るボール位置を速度コマンドと誤読する分を除く)。
      前提: 新観測の先頭 `old_in` 次元が歩行時とスロット互換であること
      (ball_kick_env_cfg.K1BallKickPolicyCfg の docstring 参照)。
    * actor 第2層以降・バイアス: 形状一致が必須。そのままコピー。
    * actor 観測正規化 (EmpiricalNormalization): 先頭 `old_in` 次元の統計をコピー、
      追加次元は mean=0 / var=1 (= 素通し) で初期化。count は引き継ぐ。
    * 行動ノイズ std: **引き継がない**。収束済みの小さい std を引き継ぐと
      新タスクの探索が死ぬため、agent cfg の init_noise_std を使う。
    * critic: **引き継がない**。報酬が全く異なるため価値関数の引き継ぎは有害。

使い方 (train.py が --warmstart_actor で呼ぶ):
    from warmstart import warmstart_actor_from_checkpoint
    warmstart_actor_from_checkpoint(runner.alg.policy, "logs/rsl_rl/k1_flat/main_walk/0524_walk.pt")
"""

from __future__ import annotations

import torch


def warmstart_actor_from_checkpoint(
    policy: torch.nn.Module,
    ckpt_path: str,
    verbose: bool = True,
) -> None:
    """歩行 ckpt の actor 重みを、観測次元の広い新 policy へ部分コピーする (in-place)。

    Args:
        policy: rsl_rl の ActorCritic (runner.alg.policy)。actor 隠れ層の形状は
            ckpt と一致していること (K1BallKickPPORunnerCfg: [256, 128, 128])。
        ckpt_path: rsl_rl 形式のチェックポイント (.pt)。model_state_dict を含むもの。
        verbose: コピー内容のレポートを表示する。

    Raises:
        ValueError: 隠れ層・出力層の形状が一致しない、または新観測次元が
            旧観測次元より小さい場合。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    old_sd = ckpt.get("model_state_dict", ckpt)
    new_sd = policy.state_dict()  # パラメータ実体への参照 (in-place コピーに使う)

    if "actor.0.weight" not in old_sd or "actor.0.weight" not in new_sd:
        raise ValueError(
            f"actor.0.weight が見つかりません (ckpt: {'actor.0.weight' in old_sd}, "
            f"policy: {'actor.0.weight' in new_sd})。MLP actor 前提のローダです。"
        )

    old_in = old_sd["actor.0.weight"].shape[1]
    new_in = new_sd["actor.0.weight"].shape[1]
    if new_in < old_in:
        raise ValueError(
            f"新観測次元 ({new_in}) が歩行時 ({old_in}) より小さいため列マッピングできません。"
        )
    if new_in != old_in:
        # 次元一致方式 (61=61 の純粋コピー) が現行の正規ルート。次元が違う ckpt は
        # 「先頭 old_in スロットの意味が完全一致」している場合しか正しく動かない。
        # 旧 0524_walk.pt (49 次元, ang_vel が先頭・gait_phase 4 次元) は現行の
        # 61 次元レイアウト (gravity が先頭・gait_phase 2 次元) と互換性が無いので使えない。
        print(
            f"[WARN] warmstart: 観測次元が不一致です ({old_in} → {new_in})。"
            "先頭スロットの互換性を必ず確認してください "
            "(現行レイアウトでは Stage 1 = Isaac-BallKick-Walk-K1-v0 の ckpt のみ互換)。"
        )

    # actor の全レイヤーキーを集める (actor.0.weight, actor.0.bias, actor.2.weight, ...)
    actor_keys = sorted(k for k in old_sd if k.startswith("actor.") and not k.startswith("actor_obs_normalizer"))
    copied, report = 0, []
    with torch.no_grad():
        for k in actor_keys:
            src, dst = old_sd[k], new_sd[k]
            if k == "actor.0.weight":
                # 第1層: 先頭 old_in 列コピー + 追加列ゼロ初期化
                if dst.shape[0] != src.shape[0]:
                    raise ValueError(
                        f"{k}: 出力幅が不一致 (ckpt {tuple(src.shape)} vs policy {tuple(dst.shape)})。"
                        "actor_hidden_dims を ckpt と揃えてください (0524_walk.pt は [256, 128, 128])。"
                    )
                dst.zero_()
                dst[:, :old_in].copy_(src)
                report.append(f"  {k}: 先頭 {old_in}/{new_in} 列コピー, 残り {new_in - old_in} 列ゼロ初期化")
            else:
                if dst.shape != src.shape:
                    raise ValueError(
                        f"{k}: 形状が不一致 (ckpt {tuple(src.shape)} vs policy {tuple(dst.shape)})。"
                        "actor_hidden_dims を ckpt と揃えてください (0524_walk.pt は [256, 128, 128])。"
                    )
                dst.copy_(src)
                report.append(f"  {k}: {tuple(src.shape)} コピー")
            copied += 1

        # 観測正規化の統計 (EmpiricalNormalization: _mean/_var/_std は shape (1, in))
        for stat, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
            k = f"actor_obs_normalizer.{stat}"
            if k in old_sd and k in new_sd:
                dst = new_sd[k]
                dst.fill_(fill)
                dst[:, :old_in].copy_(old_sd[k])
                report.append(f"  {k}: 先頭 {old_in} 次元コピー, 追加次元は {fill} で初期化")
        k = "actor_obs_normalizer.count"
        if k in old_sd and k in new_sd:
            new_sd[k].copy_(old_sd[k])
            report.append(f"  {k}: {old_sd[k].item()} を引き継ぎ")

    if verbose:
        skipped = sorted(
            k for k in old_sd
            if k not in actor_keys and not k.startswith("actor_obs_normalizer")
        )
        print(f"[INFO] warmstart: '{ckpt_path}' から actor をウォームスタートしました")
        print(f"        観測次元: {old_in} → {new_in} (先頭 {old_in} スロット互換前提)")
        print("\n".join(report))
        print(f"        引き継がないキー (critic / std など): {skipped}")
