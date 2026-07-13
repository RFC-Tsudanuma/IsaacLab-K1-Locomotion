# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Autoregressive ONNX inference for the velocity predictor.

Wraps a ``*.onnx`` produced by ``export_velocity_predictor_onnx.py`` and holds the recurrent
state (1st-order baseline velocity ``v_base`` and GRU hidden ``h``) internally, so the caller
only feeds the velocity command and reads back the predicted body-frame velocity. Intended for
path planning: predict the actual ``[vx, vy, wz]`` the policy will realize for a command stream.

Example::

    pred = VelocityPredictorOnnx("logs/velocity_predictor/predictor_planning_noproprio.onnx")
    pred.reset(num_envs=1)
    for cmd in command_stream:            # cmd: (3,) or (N, 3) = [vx, vy, wz]
        v_hat = pred.step(cmd)            # -> (N, 3) predicted actual velocity
"""

from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort


class VelocityPredictorOnnx:
    """Stateful single-step ONNX wrapper for the velocity predictor."""

    def __init__(self, onnx_path: str, providers: list[str] | None = None):
        self.session = ort.InferenceSession(
            onnx_path, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        meta = self._read_meta()
        self.dt = float(meta.get("dt", 0.02))
        self.hidden_dim = int(meta["hidden_dim"])
        self.proprio_dim = int(meta.get("proprio_dim", 0))
        self.num_layers = int(meta.get("num_layers", 1))
        self.use_proprio = bool(meta.get("use_proprio", False))
        self.proprio_keys = list(meta.get("proprio_keys", []))

        if "proprio_t" in self.input_names and not self.use_proprio:
            # Defensive: graph has a proprio input but meta says otherwise.
            self.use_proprio = True

        self.num_envs = 0
        self.v_base: np.ndarray | None = None
        self.h: np.ndarray | None = None
        self.reset(1)

    def _read_meta(self) -> dict:
        props = self.session.get_modelmeta().custom_metadata_map
        meta = {}
        for k, v in props.items():
            try:
                meta[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                meta[k] = v
        if "hidden_dim" not in meta:
            raise RuntimeError(
                "ONNX is missing 'hidden_dim' metadata; re-export with export_velocity_predictor_onnx.py."
            )
        return meta

    def reset(self, num_envs: int = 1) -> None:
        """Reset the recurrent state for ``num_envs`` parallel streams."""
        self.num_envs = int(num_envs)
        self.v_base = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.h = np.zeros((self.num_layers, self.num_envs, self.hidden_dim), dtype=np.float32)

    def step(self, cmd: np.ndarray, proprio: np.ndarray | None = None) -> np.ndarray:
        """Advance one step. ``cmd``: (3,) or (N, 3). Returns predicted velocity (N, 3)."""
        cmd_arr = np.asarray(cmd, dtype=np.float32)
        if cmd_arr.ndim == 1:
            cmd_arr = cmd_arr[None, :]
        if cmd_arr.shape[0] != self.num_envs:
            self.reset(cmd_arr.shape[0])

        feeds = {
            "cmd_t": cmd_arr,
            "v_base_prev": self.v_base,
            "h_prev": self.h,
        }
        if self.use_proprio:
            if proprio is None:
                raise ValueError("This predictor was trained with proprio; pass `proprio` to step().")
            proprio_arr = np.asarray(proprio, dtype=np.float32)
            if proprio_arr.ndim == 1:
                proprio_arr = proprio_arr[None, :]
            feeds["proprio_t"] = proprio_arr

        v_pred, v_base, h_new = self.session.run(self.output_names, feeds)
        self.v_base = v_base.astype(np.float32)
        self.h = h_new.astype(np.float32)
        return v_pred

    def predict_sequence(self, cmd_seq: np.ndarray, proprio_seq: np.ndarray | None = None) -> np.ndarray:
        """Roll out over a (T, 3) or (T, N, 3) command sequence. Returns matching v_pred array."""
        cmd_seq = np.asarray(cmd_seq, dtype=np.float32)
        single = cmd_seq.ndim == 2  # (T, 3)
        if single:
            cmd_seq = cmd_seq[:, None, :]
        T, N = cmd_seq.shape[0], cmd_seq.shape[1]
        self.reset(N)
        out = np.zeros((T, N, 3), dtype=np.float32)
        for t in range(T):
            p = None
            if self.use_proprio and proprio_seq is not None:
                p = proprio_seq[t]
            out[t] = self.step(cmd_seq[t], proprio=p)
        return out[:, 0, :] if single else out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the ONNX velocity predictor.")
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--steps", type=int, default=50)
    a = parser.parse_args()

    pred = VelocityPredictorOnnx(a.onnx)
    print(f"[INFO] dt={pred.dt} hidden_dim={pred.hidden_dim} proprio_dim={pred.proprio_dim} "
          f"use_proprio={pred.use_proprio} num_layers={pred.num_layers}")
    pred.reset(1)
    cmd = np.array([0.5, 0.0, 0.3], dtype=np.float32)
    for t in range(a.steps):
        v = pred.step(cmd)
    print(f"[INFO] after {a.steps} steps of cmd={cmd.tolist()} -> v_pred={v[0].tolist()}")
