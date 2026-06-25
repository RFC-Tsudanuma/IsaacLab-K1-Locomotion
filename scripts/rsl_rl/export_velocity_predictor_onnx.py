# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a trained VelocityPredictor (.pt) to a single-step ONNX graph.

The exported graph mirrors ``VelocityPredictor.step`` so it can be driven autoregressively
at deploy time (e.g. inside a path planner). The GRU hidden state and the 1st-order baseline
velocity are passed in/out explicitly so the caller holds the recurrent state:

    inputs : cmd_t (N, 3), v_base_prev (N, 3), h_prev (1, N, hidden) [, proprio_t (N, P)]
    outputs: v_pred (N, 3), v_base (N, 3), h_new (1, N, hidden)

Predictor metadata (dt, hidden_dim, proprio_dim, residual_scale, num_layers, proprio_keys)
is embedded in the ONNX ``metadata_props`` so the inference class can self-configure.

Pure PyTorch + onnx — does NOT launch Isaac Sim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import onnx
import torch
import torch.nn as nn


def _load_predictor_cls():
    """Load VelocityPredictor by path, avoiding the isaaclab package __init__."""
    vm_dir = (
        Path(__file__).resolve().parents[2]
        / "source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/velocity_model"
    )
    spec = importlib.util.spec_from_file_location("_vm_predictor", vm_dir / "predictor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VelocityPredictor


class _StepModule(nn.Module):
    """Traceable single-step wrapper around VelocityPredictor (mirrors .step())."""

    def __init__(self, model, use_proprio: bool):
        super().__init__()
        self.model = model
        self.use_proprio = use_proprio
        # Constant alpha/gain are derived from the (frozen) baseline params at export time
        # but kept as tensors so the graph stays correct if re-traced.
        self.dt = model.baseline.dt
        self.residual_scale = model.residual_scale

    def _v_base(self, cmd_t, v_base_prev):
        tau = torch.exp(self.model.baseline.log_tau)
        alpha = torch.exp(-self.dt / tau)
        return alpha * v_base_prev + (1.0 - alpha) * self.model.baseline.gain * cmd_t

    def forward(self, cmd_t, v_base_prev, h_prev, proprio_t=None):
        v_base = self._v_base(cmd_t, v_base_prev)
        if self.use_proprio and proprio_t is not None:
            x = torch.cat([cmd_t, v_base, proprio_t], dim=-1).unsqueeze(1)
        else:
            x = torch.cat([cmd_t, v_base], dim=-1).unsqueeze(1)
        h_out, h_new = self.model.residual.gru(x, h_prev)
        delta_v = self.model.residual.head(h_out).squeeze(1)
        v_pred = v_base + self.residual_scale * delta_v
        return v_pred, v_base, h_new


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a VelocityPredictor checkpoint to single-step ONNX.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt from train_velocity_predictor.py")
    p.add_argument("--output", type=str, default=None,
                   help="Output .onnx path (default: <checkpoint stem>.onnx next to the checkpoint).")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output) if args.output else ckpt_path.with_suffix(".onnx")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hidden_dim = int(ckpt["hidden_dim"])
    proprio_dim = int(ckpt["proprio_dim"])
    dt = float(ckpt["dt"])
    residual_scale = float(ckpt["residual_scale"])
    pred_args = ckpt.get("args", {})
    use_proprio = bool(pred_args.get("use_proprio", False)) and proprio_dim > 0
    proprio_keys = list(pred_args.get("proprio_keys", []))

    VelocityPredictor = _load_predictor_cls()
    model = VelocityPredictor(
        dim=3, dt=dt, hidden_dim=hidden_dim, proprio_dim=proprio_dim, residual_scale=residual_scale,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    num_layers = model.residual.gru.num_layers
    wrapper = _StepModule(model, use_proprio=use_proprio).eval()

    # Dummy inputs (batch N=1; batch axis is dynamic).
    cmd_t = torch.zeros(1, 3)
    v_base_prev = torch.zeros(1, 3)
    h_prev = torch.zeros(num_layers, 1, hidden_dim)

    input_names = ["cmd_t", "v_base_prev", "h_prev"]
    output_names = ["v_pred", "v_base", "h_new"]
    dynamic_axes = {
        "cmd_t": {0: "N"}, "v_base_prev": {0: "N"}, "h_prev": {1: "N"},
        "v_pred": {0: "N"}, "v_base": {0: "N"}, "h_new": {1: "N"},
    }
    if use_proprio:
        proprio_t = torch.zeros(1, proprio_dim)
        inputs = (cmd_t, v_base_prev, h_prev, proprio_t)
        input_names.insert(3, "proprio_t")
        dynamic_axes["proprio_t"] = {0: "N"}
    else:
        inputs = (cmd_t, v_base_prev, h_prev)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, inputs, str(out_path),
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=args.opset,
    )

    # Embed metadata so the inference class can self-configure.
    meta = {
        "dt": dt, "hidden_dim": hidden_dim, "proprio_dim": proprio_dim,
        "residual_scale": residual_scale, "num_layers": num_layers,
        "use_proprio": use_proprio, "proprio_keys": proprio_keys,
    }
    onnx_model = onnx.load(str(out_path))
    for k, v in meta.items():
        entry = onnx_model.metadata_props.add()
        entry.key = k
        entry.value = json.dumps(v)
    onnx.save(onnx_model, str(out_path))

    print(f"[INFO] Exported ONNX: {out_path}")
    print(f"[INFO] meta: {meta}")
    print(f"[INFO] inputs={input_names} outputs={output_names}")


if __name__ == "__main__":
    main()
