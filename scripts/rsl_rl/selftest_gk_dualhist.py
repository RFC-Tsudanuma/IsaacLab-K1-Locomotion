# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""デュアルヒストリー版ゴールキーパーの単体確認 (Isaac Sim を起動せずに数秒で回る)。

観測レイアウトとネットワークの切り出し位置は**暗黙の契約**で結ばれていて、片方だけ
直すと例外も出さずに黙って壊れる (別のチャンネルを読むだけなので学習は進む)。
``dualhist/`` を触ったら必ずこれを通すこと。

確認する内容:
    1. ActorCriticDualHistory が組める / act_inference と evaluate が通る
    2. TorchScript と ONNX にエクスポートできる (実機デプロイ経路)
    3. 長期履歴・既存観測のどちらを変えても出力が変わる (経路が死んでいない)
    4. リングバッファが「古い → 新しい」順で出る / stride が効く / 冪等 / リセットが効く
    5. 左右反転が対合になっている・次元不整合を検出する

実行 (コンテナ内)::

    /isaac-sim/python.sh scripts/rsl_rl/selftest_gk_dualhist.py

Isaac Sim 本体には依存しないので ``isaaclab.sh -p`` でなくてよい。ただし
``dualhist/observations.py`` と ``symmetry.py`` は本来 isaaclab を要求するので、
親モジュールをスタブに差し替えてから読み込んでいる (下の ``_stub_parent``)。
"""

from __future__ import annotations

import copy
import importlib.util
import io
import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DUALHIST = os.path.join(
    _HERE, "..", "..", "source", "isaaclab_k1_locomotion", "isaaclab_k1_locomotion",
    "tasks", "manager_based", "goalkeeper", "dualhist",
)
DUALHIST = os.path.normpath(DUALHIST)

BASE_OBS = 59   # 既存の gk 観測 (歩行 49 + タスク 10)
CRITIC_OBS = 64


def _load(name: str, filename: str, package: str | None = None):
    spec = importlib.util.spec_from_file_location(name, os.path.join(DUALHIST, filename))
    mod = importlib.util.module_from_spec(spec)
    if package:
        mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_parent(state: dict) -> None:
    """``..mdp.observations`` / ``..mdp.symmetry`` を isaaclab 非依存のスタブで置き換える。"""
    pkg = types.ModuleType("gkstub")
    pkg.__path__ = []
    mdp = types.ModuleType("gkstub.mdp")
    mdp.__path__ = []

    obsmod = types.ModuleType("gkstub.mdp.observations")
    obsmod._gk_perceived_goal_state = lambda env: (state["ball"], None)
    obsmod._gk_perception = lambda env: types.SimpleNamespace(ball_mask=state["mask"])
    obsmod.robot_pose_est = lambda env: (state["pose"], state["yaw"])

    symmod = types.ModuleType("gkstub.mdp.symmetry")
    symmod._POLICY_OBS_DIM = BASE_OBS
    symmod._mirror_gk_policy_obs = lambda o: -o   # 反転の中身は既存実装の責務
    symmod._mirror_gk_critic_obs = lambda o: -o

    for name, mod in [
        ("gkstub", pkg), ("gkstub.mdp", mdp),
        ("gkstub.mdp.observations", obsmod), ("gkstub.mdp.symmetry", symmod),
    ]:
        sys.modules[name] = mod
    pkg.mdp = mdp
    mdp.observations = obsmod
    mdp.symmetry = symmod


class _FakeEnv:
    num_envs = 3
    device = "cpu"
    common_step_counter = 0


def test_network(short_frames: int, long_frames: int, frame_dim: int) -> int:
    from tensordict import TensorDict

    net = _load("dh_networks", "networks.py")
    n_obs = BASE_OBS + (short_frames + long_frames) * frame_dim
    n_env = 4

    obs = TensorDict(
        {"policy": torch.randn(n_env, n_obs), "critic": torch.randn(n_env, CRITIC_OBS)},
        batch_size=[n_env],
    )
    ac = net.ActorCriticDualHistory(
        obs, {"policy": ["policy"], "critic": ["critic"]}, 3,
        hist_frame_dim=frame_dim, hist_short_frames=short_frames, hist_long_frames=long_frames,
        init_noise_std=0.7, actor_obs_normalization=True, critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128], critic_hidden_dims=[256, 256, 128], activation="elu",
    )
    assert ac.act_inference(obs).shape == (n_env, 3)
    assert ac.evaluate(obs).shape == (n_env, 1)

    actor = copy.deepcopy(ac.actor).eval()
    x = torch.randn(2, n_obs)

    scripted = torch.jit.script(actor)
    assert torch.allclose(actor(x), scripted(x), atol=1e-6), "TorchScript の出力が一致しない"

    # isaaclab_rl の _OnnxPolicyExporter と同じ経路 (actor[0].in_features でダミーを作る)
    assert actor[0].in_features == n_obs
    buf = io.BytesIO()
    torch.onnx.export(
        actor, (torch.zeros(1, n_obs),), buf,
        input_names=["obs"], output_names=["actions"], opset_version=18,
    )
    assert len(buf.getvalue()) > 0

    x_long = x.clone()
    x_long[:, BASE_OBS + short_frames * frame_dim:] += 1.0
    assert not torch.allclose(actor(x), actor(x_long)), "長期履歴が出力に効いていない"
    x_base = x.clone()
    x_base[:, :BASE_OBS] += 1.0
    assert not torch.allclose(actor(x), actor(x_base)), "既存観測が出力に効いていない"

    print(f"[OK] network (obs {n_obs}, latent {ac.actor.latent_dim}, jit / onnx / 経路)")
    return n_obs


def test_history(frame_dim: int) -> None:
    state = {
        "ball": torch.zeros(3, 2), "mask": torch.ones(3),
        "pose": torch.zeros(3, 2), "yaw": torch.zeros(3),
    }
    _stub_parent(state)
    obs = _load("gkstub.dualhist.observations", "observations.py", package="gkstub.dualhist")
    assert obs.GK_HIST_FRAME_DIM == frame_dim

    env = _FakeEnv()
    for step in range(12):
        env.common_step_counter = step
        state["ball"] = torch.full((3, 2), float(step))
        state["pose"] = torch.full((3, 2), float(step) * 10)
        short = obs.gk_io_history(env, num_frames=3, stride=1)
        long_ = obs.gk_io_history(env, num_frames=5, stride=2)

    short = short.reshape(3, 3, frame_dim)
    long_ = long_.reshape(3, 5, frame_dim)
    assert short[0, :, 0].tolist() == [9.0, 10.0, 11.0], "短期の順序/最新フレームが違う"
    assert long_[0, :, 0].tolist() == [3.0, 5.0, 7.0, 9.0, 11.0], "stride 付きの取り出しが違う"
    assert short[0, -1, 3] == 110.0, "自機 x のチャンネル位置が違う"
    assert short[0, -1, 6] == 1.0, "cos(yaw) のチャンネル位置が違う"

    # 同一ステップで 2 回呼んでも二重書き込みしない
    env.common_step_counter = 11
    again = obs.gk_io_history(env, num_frames=3, stride=1).reshape(3, 3, frame_dim)
    assert torch.equal(again, short), "冪等でない (同一ステップで 2 回書いている)"

    # リセットで消える。消すのは指定した env だけ
    obs.reset_gk_history(env, torch.tensor([0]))
    env.common_step_counter = 12
    state["ball"] = torch.full((3, 2), 99.0)
    after = obs.gk_io_history(env, num_frames=3, stride=1).reshape(3, 3, frame_dim)
    assert after[0, :, 0].tolist() == [0.0, 0.0, 99.0], "リセットが効いていない"
    assert after[1, :, 0].tolist() == [10.0, 11.0, 99.0], "リセットしていない env まで消えている"

    print("[OK] history (順序 / stride / 冪等 / リセット)")


def test_symmetry(n_obs: int, frame_dim: int, n_frames: int) -> None:
    sym = _load("gkstub.dualhist.symmetry", "symmetry.py", package="gkstub.dualhist")

    o = torch.randn(4, n_obs)
    m = sym._mirror_policy_obs_dh(o)
    mm = sym._mirror_policy_obs_dh(m)
    assert torch.allclose(o[:, BASE_OBS:], mm[:, BASE_OBS:], atol=1e-6), "履歴の反転が対合でない"

    sign = torch.tensor(sym._HIST_FRAME_MIRROR_SIGN).repeat(n_frames)
    assert torch.allclose(m[:, BASE_OBS:], o[:, BASE_OBS:] * sign)

    try:
        sym._mirror_policy_obs_dh(torch.randn(4, BASE_OBS + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("次元不整合を検出できていない")

    print("[OK] symmetry (対合 / 符号 / 次元チェック)")


def main() -> None:
    # dualhist/env_cfg.py と agent_cfg.py の既定値と合わせてある
    short_frames, long_frames, frame_dim = 5, 50, 7

    n_obs = test_network(short_frames, long_frames, frame_dim)
    test_history(frame_dim)
    test_symmetry(n_obs, frame_dim, short_frames + long_frames)
    print("\nすべて OK")


if __name__ == "__main__":
    main()
