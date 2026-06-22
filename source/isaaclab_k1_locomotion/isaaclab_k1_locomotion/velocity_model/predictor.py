"""1st-order baseline + GRU residual velocity predictor."""

from __future__ import annotations

import torch
import torch.nn as nn


class FirstOrderBaseline(nn.Module):
    """1st-order system per axis: tau * dv/dt = -(v - gain * cmd).

    Discretized with alpha = exp(-dt/tau). log_tau is the learnable parameter so
    that tau stays positive.
    """

    def __init__(self, dim: int = 3, dt: float = 0.02, tau_init: float = 0.3):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.log_tau = nn.Parameter(torch.full((dim,), float(torch.log(torch.tensor(tau_init)))))
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, cmd_seq: torch.Tensor, v0: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = cmd_seq.shape
        tau = torch.exp(self.log_tau)
        alpha = torch.exp(-self.dt / tau)
        v = torch.zeros(B, D, device=cmd_seq.device, dtype=cmd_seq.dtype) if v0 is None else v0
        outputs = []
        for t in range(T):
            v = alpha * v + (1.0 - alpha) * self.gain * cmd_seq[:, t]
            outputs.append(v)
        return torch.stack(outputs, dim=1)


class GRUResidual(nn.Module):
    """GRU-based residual on top of the 1st-order baseline."""

    def __init__(
        self,
        cmd_dim: int = 3,
        base_dim: int = 3,
        proprio_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 1,
        output_dim: int = 3,
    ):
        super().__init__()
        input_dim = cmd_dim + base_dim + proprio_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        cmd_seq: torch.Tensor,
        base_seq: torch.Tensor,
        proprio_seq: torch.Tensor | None = None,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if proprio_seq is not None:
            x = torch.cat([cmd_seq, base_seq, proprio_seq], dim=-1)
        else:
            x = torch.cat([cmd_seq, base_seq], dim=-1)
        h, h_last = self.gru(x, h0)
        delta_v = self.head(h)
        return delta_v, h_last


class VelocityPredictor(nn.Module):
    """1st-order baseline + GRU residual integrated model."""

    def __init__(
        self,
        dim: int = 3,
        dt: float = 0.02,
        hidden_dim: int = 64,
        proprio_dim: int = 0,
        residual_scale: float = 0.5,
    ):
        super().__init__()
        self.baseline = FirstOrderBaseline(dim=dim, dt=dt)
        self.residual = GRUResidual(
            cmd_dim=dim,
            base_dim=dim,
            proprio_dim=proprio_dim,
            hidden_dim=hidden_dim,
            output_dim=dim,
        )
        self.residual_scale = residual_scale

    def forward(
        self,
        cmd_seq: torch.Tensor,
        proprio_seq: torch.Tensor | None = None,
        v0: torch.Tensor | None = None,
        h0: torch.Tensor | None = None,
    ):
        v_base = self.baseline(cmd_seq, v0=v0)
        delta_v, h_last = self.residual(cmd_seq, v_base, proprio_seq, h0=h0)
        v_pred = v_base + self.residual_scale * delta_v
        return v_pred, v_base, delta_v, h_last

    @torch.no_grad()
    def step(
        self,
        cmd_t: torch.Tensor,
        v_base_prev: torch.Tensor,
        h_prev: torch.Tensor,
        proprio_t: torch.Tensor | None = None,
    ):
        """Single-step inference for deployment."""
        tau = torch.exp(self.baseline.log_tau)
        alpha = torch.exp(-self.baseline.dt / tau)
        v_base = alpha * v_base_prev + (1.0 - alpha) * self.baseline.gain * cmd_t

        if proprio_t is not None:
            x = torch.cat([cmd_t, v_base, proprio_t], dim=-1).unsqueeze(1)
        else:
            x = torch.cat([cmd_t, v_base], dim=-1).unsqueeze(1)
        h_out, h_new = self.residual.gru(x, h_prev)
        delta_v = self.residual.head(h_out).squeeze(1)
        v_pred = v_base + self.residual_scale * delta_v
        return v_pred, v_base, h_new
