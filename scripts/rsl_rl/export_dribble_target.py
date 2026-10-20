# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ドリブル (Isaac-Velocity-Flat-K1-Dribble-Target-v0) の checkpoint を、実機の推論ノード向けに
ONNX と仕様 JSON に書き出す。Isaac Sim は起動しない (学習中でも別プロセスで実行できる)。

* ONNX: dribble_target/agents/history_policy_exporter.py で書き出す。opset 18、batch=1、
  全軸固定、観測の正規化器をグラフに含める。
    入力  command      (1, 3)          Stage2 では常にゼロ
          obs_history  (1, 100, 59)    各行 = 1 ステップ分のフレーム、古い → 新しい
    出力  actions      (1, 12)         scale を掛ける前の生の値 (q_target = default + 0.5 * action)
* 仕様 JSON: 入出力・フレームの並び (座標系・単位・遅延・ノイズ)・action_scale・default_joint_pos・
  gait_phase・コマンド・PD ゲイン/effort_limit・ボールの初期位置・終了条件・ラン ID と iteration。
  値はそのランの params/env.yaml から取る。
* 検証: 同じ入力を PyTorch と ONNX (onnx.reference) に入れて出力の差を確認する。

    python scripts/rsl_rl/export_dribble_target.py --checkpoint logs/rsl_rl/k1_dribble_target/<run>/model_N.pt
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from widen_history_checkpoint import _build_policy, _import_dribble_modules  # noqa: E402

# 各フレーム項の説明 (座標系・単位・実機側の取り方)。順序は history_layout.DRIBBLE_TARGET_POLICY_TERM_SPECS。
_TERM_DOC = {
    "base_ang_vel": ("IMU 角速度 (body 座標)", "rad/s"),
    "projected_gravity": ("重力方向の単位ベクトル (body 座標)", "-"),
    "velocity_commands": ("歩行の速度コマンド [vx, vy, wz]。Stage2 では常にゼロ", "m/s, m/s, rad/s"),
    "joint_pos": ("関節角 - default_joint_pos (joint_order の順)", "rad"),
    "joint_vel": ("関節角速度 (joint_order の順)", "rad/s"),
    "actions": ("前回の出力 actions (scale を掛ける前)", "-"),
    "gait_phase": ("[sin φL, cos φL, sin φR, cos φR]、φR = φL + π。1.8Hz で常に回す (ゼロにしない)", "-"),
    "target_pos_rel": ("目標点 - ロボット位置 の xy (body yaw 座標)。/localization_result から変換", "m"),
    "ball_pos_rel": ("ボール中心 - ロボット位置 の xy (body yaw 座標)。vision_filter から", "m"),
    "ball_vel": ("ボールのワールド速度を body yaw 座標に回した xy。推定できないときは厳密に 0", "m/s"),
    "foot_left_ball_pos_rel": ("ボール中心 - 左 foot_link 原点 の xy (body yaw 座標)", "m"),
    "foot_right_ball_pos_rel": ("ボール中心 - 右 foot_link 原点 の xy (body yaw 座標)", "m"),
}


def _noise_doc(term_cfg: dict) -> dict:
    """env.yaml の観測項の設定から、学習時に掛けていたノイズ・遅延・ホールドを取り出す。"""
    out: dict = {}
    noise = term_cfg.get("noise")
    if isinstance(noise, dict):
        inner = noise.get("noise_cfg", noise)
        if "n_min" in inner:
            out["white_noise_uniform"] = [inner["n_min"], inner["n_max"]]
        for key in ("delay_range", "hold_prob_range", "bias_range", "filter_alpha_range"):
            if key in noise and noise[key] not in (None, 0.0, [0.0, 0.0], (0.0, 0.0)):
                out[key + ("_s" if key == "delay_range" else "")] = list(noise[key]) if isinstance(noise[key], (list, tuple)) else noise[key]
    params = term_cfg.get("params") or {}
    if "noise_range" in params:
        out["white_noise_uniform"] = [-params["noise_range"], params["noise_range"]]
    for key in ("near_distance", "far_distance", "near_hold_prob", "far_hold_prob", "near_invalid_prob", "far_invalid_prob"):
        if key in params:
            out[key] = params[key]
    if any(k in params for k in ("near_hold_prob", "near_invalid_prob")):
        out["note"] = "ボール距離に応じて near〜far を線形補間。ホールドは前回の値をそのまま出す (毎ステップ評価)"
    return out


def _actuator_doc(env: dict) -> dict:
    acts = env["scene"]["robot"]["actuators"]
    return {
        name: {
            "joints": a["joint_names_expr"],
            "stiffness": a["stiffness"],
            "damping": a["damping"],
            "effort_limit_Nm": a["effort_limit"],
            "action_delay_steps_dr": [a.get("min_delay"), a.get("max_delay")],
        }
        for name, a in acts.items()
    }


def build_spec(env: dict, agent: dict, layout, run_dir: str, ckpt_iter: int, curriculum_state: dict | None) -> dict:
    pol = env["observations"]["policy"]
    h = int(pol.get("history_length") or layout.HISTORY_LENGTH)
    frame, ch = [], 0
    for name, dim, _ in layout.DRIBBLE_TARGET_POLICY_TERM_SPECS:
        desc, unit = _TERM_DOC.get(name, ("", ""))
        frame.append(
            {"name": name, "dim": dim, "channels": [ch, ch + dim], "description": desc, "unit": unit, "training_noise": _noise_doc(pol.get(name, {}))}
        )
        ch += dim
    act = env["actions"]["joint_pos"]
    gait = pol["gait_phase"]["params"]
    cmd = env["commands"]["dribble_target"]
    return {
        "task": "Isaac-Velocity-Flat-K1-Dribble-Target-v0",
        "run": os.path.basename(os.path.normpath(run_dir)),
        "checkpoint_iteration": ckpt_iter,
        "curriculum_state": curriculum_state,
        "control_rate_hz": 1.0 / (env["sim"]["dt"] * env["decimation"]),
        "inputs": [
            {"name": "command", "shape": [1, 3], "note": "Stage2 では常にゼロを入れる"},
            {"name": "obs_history", "shape": [1, h, ch], "note": "各行 = 1 ステップ分のフレーム (古い → 新しい)"},
        ],
        "output": {"name": "actions", "shape": [1, 12], "note": "scale を掛ける前の生の値。clip しない"},
        "normalizer": "グラフに含む (生の観測をそのまま入れる)",
        "history": {
            "H": h,
            "order": "old_to_new",
            "init": "ドリブル開始時 (リセット直後と同じ) は現在のフレームで履歴全体を埋める",
        },
        "frame_layout": frame,
        "frame_dim": ch,
        "joint_order": act["joint_names"],
        "action_scale": act["scale"],
        "q_target": "default_joint_pos + action_scale * action",
        "default_joint_pos": env["scene"]["robot"]["init_state"]["joint_pos"],
        "gait_phase": {
            "frequency_hz": gait["low_freq"],
            "always_on": True,
            "note": "Stage2 ではコマンドがゼロでも位相をゼロにしない (学習の最初の 500 iteration で cmd_threshold を "
            f"{gait['cmd_threshold']} → -1.0 に移した)。位相はリセットせず、歩行から 1.8Hz のまま引き継ぐ",
            "training_freq_offset_dr_hz": [-0.05, 0.05],
        },
        "commands": {
            "base_velocity": "Stage2 では常にゼロ (command 入力と velocity_commands の 3 次元)",
            "dribble_target": {
                "frame": "ワールド (フィールド) 座標の目標点 (x, y)。観測は body yaw 座標の相対位置に直して入れる",
                "resampling_time_s": cmd["resampling_time_range"],
                "training_ranges_at_export": cmd["ranges"],
                "training_path_shape_at_export": cmd["path_shape"],
                "omni_prob": cmd.get("omni_prob"),
                "far_prob": cmd.get("far_prob"),
                "waypoint_reach_radius_m": cmd["waypoint_reach_radius"],
            },
        },
        "actuators": _actuator_doc(env),
        "actuator_gain_dr": [0.8, 1.2],
        "ball_reset": {
            "frame": "env 原点基準のワールド座標オフセット (ロボットの向きには合わせない)。既定位置 (0.30, 0, 0.11) に加算",
            "pose_range": {k: list(v) for k, v in env["events"]["reset_ball"]["params"]["pose_range"].items() if k in ("x", "y")},
            "robot_pose_range": {k: list(v) for k, v in env["events"]["reset_base"]["params"]["pose_range"].items()},
        },
        "terminations": {
            name: (t.get("params") or {}) if name not in ("obstacle_collision", "base_contact") else name
            for name, t in env["terminations"].items()
            if t
        },
        "obstacle_collision": "障害物レベルのみ。ロボット 0.45m / ボール 0.31m 以内 (中心間 xy)",
        "agent": {"num_envs": env["scene"]["num_envs"], "num_steps_per_env": agent["num_steps_per_env"],
                  "num_mini_batches": agent["algorithm"]["num_mini_batches"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=None, help="既定: <run>/exported/model_<N>/")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    run_dir = os.path.dirname(ckpt_path)
    layout = _import_dribble_modules()
    exporter_mod = importlib.import_module("_dribble_target_pkg.agents.history_policy_exporter")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    it = int(ckpt.get("iter", -1))
    out_dir = args.output_dir or os.path.join(run_dir, "exported", f"model_{it}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(run_dir, "params", "env.yaml")) as f:
        env = yaml.unsafe_load(f)
    with open(os.path.join(run_dir, "params", "agent.yaml")) as f:
        agent = yaml.unsafe_load(f)
    cur_path = os.path.splitext(ckpt_path)[0] + "_curriculum.json"
    curriculum_state = json.load(open(cur_path)) if os.path.isfile(cur_path) else None

    policy = _build_policy("dribble_target", 12, layout.HISTORY_LENGTH, ckpt["model_state_dict"]).float()
    run_name = os.path.basename(os.path.normpath(run_dir))
    onnx_name = f"k1_dribble_target_{run_name}_model_{it}.onnx"
    exporter_mod.export_history_policy_as_onnx(policy, path=out_dir, normalizer=policy.actor_obs_normalizer, filename=onnx_name)

    # --- 検証: 形が全軸固定か、PyTorch と ONNX の出力が一致するか
    import onnx
    from onnx.reference import ReferenceEvaluator

    model = onnx.load(os.path.join(out_dir, onnx_name))
    onnx.checker.check_model(model)
    opset = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    shapes = {
        v.name: [d.dim_value if d.HasField("dim_value") else d.dim_param for d in v.type.tensor_type.shape.dim]
        for v in list(model.graph.input) + list(model.graph.output)
    }
    assert all(isinstance(d, int) and d > 0 for s in shapes.values() for d in s), f"固定されていない軸があります: {shapes}"
    step_dim = layout.term_specs_dim(layout.DRIBBLE_TARGET_POLICY_TERM_SPECS)
    g = torch.Generator().manual_seed(0)
    command = torch.zeros(1, 3)
    obs_history = torch.randn(1, layout.HISTORY_LENGTH, step_dim, generator=g) * 0.5
    wrapper = exporter_mod._HistoryPolicyExporter(policy, policy.actor_obs_normalizer).eval()
    with torch.no_grad():
        ref = wrapper(command, obs_history).numpy()
        # 環境側のレイアウト (項ごとのブロック) で act_inference に入れた値とも一致すること
        from tensordict import TensorDict

        flat = [obs_history[:, :, a:b].reshape(1, -1) for a, b in (f["channels"] for f in build_spec(env, agent, layout, run_dir, it, None)["frame_layout"])]
        td = TensorDict({"command": command, "policy": torch.cat(flat, -1), "critic": torch.zeros(1, 66 * layout.HISTORY_LENGTH)}, batch_size=[1])
        direct = policy.act_inference(td.to(torch.float64) if next(policy.parameters()).dtype == torch.float64 else td).numpy()
    got = ReferenceEvaluator(model).run(None, {"command": command.numpy(), "obs_history": obs_history.numpy()})[0]
    diff_onnx = float(np.abs(got - ref).max())
    diff_layout = float(np.abs(direct - ref).max())

    spec = build_spec(env, agent, layout, run_dir, it, curriculum_state)
    spec["onnx"] = {
        "file": onnx_name,
        "opset": opset,
        "shapes": shapes,
        "verified_max_abs_diff_torch_vs_onnx": diff_onnx,
        "verified_max_abs_diff_exporter_vs_env_layout": diff_layout,
    }
    spec_name = os.path.splitext(onnx_name)[0] + "_spec.json"
    with open(os.path.join(out_dir, spec_name), "w") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False, default=str)
    print(f"[export] {os.path.join(out_dir, onnx_name)} (opset {opset}, shapes {shapes})")
    print(f"[export] {os.path.join(out_dir, spec_name)}")
    print(f"[verify] |torch - onnx| max = {diff_onnx:.2e}, |exporter - env layout| max = {diff_layout:.2e}")
    if diff_onnx > 1e-4 or diff_layout > 1e-4:
        raise SystemExit("[verify] NG: 出力が一致しません")


if __name__ == "__main__":
    main()
