"""Batched port of futbol_main VisionFilter, revision 32ece6ee (2026-09-21).

Four CVKFs, MAP selection, bounce reseeding and confirmed/tentative capture.
Internal arithmetic is float64 SI; the public snapshot is float32 SI. Timestamps
are integer nanoseconds, so the strict three-second timeout is reproducible.
"""

from dataclasses import dataclass

import torch


def cv_matrices(dt, stationary=False):
    f = torch.eye(4, dtype=dt.dtype, device=dt.device).expand(*dt.shape, 4, 4).clone()
    f[..., 0, 2] = dt
    f[..., 1, 3] = dt
    if stationary:
        f[..., 0, 2, 2] = 0
        f[..., 0, 3, 3] = 0
    q = torch.zeros_like(f)
    q[..., 0, 0] = q[..., 1, 1] = 0.64 * dt**4 / 4
    q[..., 0, 2] = q[..., 2, 0] = 0.64 * dt**3 / 2
    q[..., 1, 3] = q[..., 3, 1] = 0.64 * dt**3 / 2
    q[..., 2, 2] = q[..., 3, 3] = 0.64 * dt**2
    return f, q


def sym(a):
    return (a + a.transpose(-1, -2)) * 0.5


@dataclass
class Snapshot:
    state: torch.Tensor
    covariance: torch.Tensor
    status: torch.Tensor
    accepted: torch.Tensor
    stamp_ns: torch.Tensor
    observation_ns: torch.Tensor


class HypothesisBank:
    """The four core filters and selector. No cross-environment state."""

    def __init__(self, count, device):
        self.x = torch.zeros(count, 4, 4, device=device, dtype=torch.float64)
        self.p = torch.zeros(count, 4, 4, 4, device=device, dtype=torch.float64)
        self.initialized = torch.zeros(count, device=device, dtype=torch.bool)
        self.stamp = torch.full((count,), -1, device=device, dtype=torch.long)
        self.observed_at = self.stamp.clone()
        self.probability = torch.full((count, 4), 0.25, device=device, dtype=torch.float64)
        self.r_meter = self.x.new_tensor([[0.0477733905268419846, -0.0004131492504468594],
                                         [-0.0004131492504468594, 0.0153060614974156124]])
        self.transition = torch.full((4, 4), 1 / 60, device=device, dtype=torch.float64)
        self.transition.fill_diagonal_(0.95)

    def reset(self, ids):
        self.x[ids] = 0
        self.p[ids] = 0
        self.initialized[ids] = False
        self.stamp[ids] = -1
        self.observed_at[ids] = -1
        self.probability[ids] = 0.25

    def copy_rows(self, other, ids):
        for name in ("x", "p", "initialized", "stamp", "observed_at", "probability"):
            getattr(self, name)[ids] = getattr(other, name)[ids]

    def prediction(self, ids, stamp):
        dt = ((stamp - self.stamp[ids]).double() * 1e-9)[:, None].expand(-1, 4)
        f, q = cv_matrices(dt, stationary=True)
        x = (f @ self.x[ids, ..., None]).squeeze(-1)
        x[:, 0, 2:] = 0
        return x, sym(f @ self.p[ids] @ f.transpose(-1, -2) + q)

    def innovation(self, x, p, global_xy, local_xy):
        r = torch.linalg.vector_norm(local_xy.double(), dim=-1)[:, None, None] * self.r_meter
        error = global_xy.double()[:, None, :] - x[..., :2]
        s = sym(p[..., :2, :2] + r[:, None])
        chol, info = torch.linalg.cholesky_ex(s)
        valid = info == 0
        # A failed factor is never used for a correction or a gate.
        safe_chol = torch.where(valid[..., None, None], chol, torch.eye(2, device=x.device, dtype=x.dtype))
        solved = torch.cholesky_solve(error[..., None], safe_chol).squeeze(-1)
        nis = (error * solved).sum(-1)
        valid &= torch.isfinite(nis) & (nis >= 0)
        ll = -0.5 * (nis + 2 * torch.log(safe_chol.diagonal(dim1=-2, dim2=-1)).sum(-1))
        return r, error, safe_chol, nis, ll, valid

    def gate(self, ids, stamp, global_xy, local_xy):
        x, p = self.prediction(ids, stamp)
        _, _, _, nis, _, valid = self.innovation(x, p, global_xy, local_xy)
        eligible = self.initialized[ids] & ((stamp - self.observed_at[ids]) <= 3_000_000_000)
        return eligible & valid.all(-1) & (nis <= 9.21).any(-1)

    def step(self, ids, stamp, global_xy, local_xy, observed):
        if torch.any((self.stamp[ids] >= 0) & (stamp <= self.stamp[ids])):
            raise ValueError("VisionFilter stamps must be strictly increasing")
        initialized = self.initialized[ids]
        initialize = observed & (~initialized | ((stamp - self.observed_at[ids]) > 3_000_000_000))
        x, p = self.prediction(ids, stamp)
        x[~initialized] = 0
        p[~initialized] = 0
        r, error, chol, _, ll, valid = self.innovation(x, p, global_xy, local_xy)
        correct = (initialized & observed & ~initialize)[:, None] & valid
        k = torch.cholesky_solve(p[..., :2, :].contiguous(), chol).transpose(-1, -2)
        corrected_x = x + (k @ error[..., None]).squeeze(-1)
        ikh = torch.eye(4, device=x.device, dtype=x.dtype).expand_as(p).clone()
        ikh[..., :2] -= k
        corrected_p = sym(ikh @ p @ ikh.transpose(-1, -2) + k @ r[:, None] @ k.transpose(-1, -2))
        x = torch.where(correct[..., None], corrected_x, x)
        p = torch.where(correct[..., None, None], corrected_p, p)
        x[:, 0, 2:] = 0
        x[initialize] = 0
        x[initialize, :, :2] = global_xy[initialize, None, :].double()
        p[initialize] = torch.diag(x.new_tensor([0.0625, 0.0625, 6.25, 6.25]))
        prior = self.probability[ids] @ self.transition
        scores = torch.log(prior) + torch.where(correct, ll, -torch.inf)
        posterior = torch.softmax(scores, dim=-1)
        posterior = torch.where(correct.any(-1, keepdim=True), posterior, prior / prior.sum(-1, keepdim=True))
        selected = posterior.argmax(-1)
        rows = torch.arange(len(ids), device=x.device)
        accepted = initialize | correct[rows, selected]
        self.observed_at[ids] = torch.where(accepted, stamp, self.observed_at[ids])
        self.initialized[ids] |= initialize
        status = torch.where(accepted, 1, 2)
        status = torch.where(self.initialized[ids] & ((stamp - self.observed_at[ids]) <= 3_000_000_000), status, 0)
        result_x, result_p = x[rows, selected].clone(), p[rows, selected].clone()
        reseed = (selected != 3) & (status != 0)
        d = x.new_tensor([1, 1, -1, -1])
        x[reseed, 3] = result_x[reseed] * d
        p[reseed, 3] = result_p[reseed] * d[:, None] * d[None, :]
        self.x[ids], self.p[ids] = x, p
        self.stamp[ids], self.probability[ids] = stamp, posterior
        return Snapshot(result_x, result_p, status, accepted, stamp, self.observed_at[ids].clone())


class VisionFilter:
    """Node acquisition flow and its public global state/covariance contract."""

    def __init__(self, count, device):
        self.confirmed = HypothesisBank(count, device)
        self.tentative = HypothesisBank(count, device)
        self.active = torch.zeros(count, device=device, dtype=torch.bool)
        self.capture_count = torch.zeros(count, device=device, dtype=torch.long)
        self.state = torch.zeros(count, 4, device=device)
        self.covariance = torch.zeros(count, 4, 4, device=device)
        self.status = torch.zeros(count, device=device, dtype=torch.long)
        self.stamp_ns = torch.zeros(count, device=device, dtype=torch.long)
        self.observation_ns = torch.full_like(self.stamp_ns, -1)
        self.updated = self.active.clone()

    @property
    def initialized(self):
        return self.active | (self.capture_count > 0)

    def invalidate(self, ids):
        self.confirmed.reset(ids)
        self.tentative.reset(ids)
        self.active[ids] = False
        self.capture_count[ids] = 0
        self.state[ids] = 0
        self.covariance[ids] = 0
        self.status[ids] = 0
        self.updated[ids] = False
        self.stamp_ns[ids] = 0
        self.observation_ns[ids] = -1

    def _publish(self, ids, output):
        live = output.status != 0
        self.state[ids] = (output.state * live[:, None]).float()
        self.covariance[ids] = (output.covariance * live[:, None, None]).float()
        self.status[ids] = output.status
        self.updated[ids] = output.accepted
        self.observation_ns[ids] = output.observation_ns

    def process(self, ids, stamp_ns, global_xy, local_xy, observed):
        self.updated.zero_()
        observed = observed & torch.isfinite(global_xy).all(-1) & torch.isfinite(local_xy).all(-1)
        observed &= (global_xy.abs() <= global_xy.new_tensor([16, 12])).all(-1)
        self.stamp_ns[ids] = stamp_ns
        handled = torch.zeros_like(observed)
        confirmed_mask = self.active[ids]
        ci = ids[confirmed_mask]
        if ci.numel():
            stamp, glob, loc = stamp_ns[confirmed_mask], global_xy[confirmed_mask], local_xy[confirmed_mask]
            accept = observed[confirmed_mask] & self.confirmed.gate(ci, stamp, glob, loc)
            out = self.confirmed.step(ci, stamp, glob, loc, accept)
            self._publish(ci, out)
            captured = accept & out.accepted
            handled[confirmed_mask] = captured
            clear = ci[captured]
            self.tentative.reset(clear)
            self.capture_count[clear] = 0
            lost = ci[out.status == 0]
            self.confirmed.reset(lost)
            self.active[lost] = False
        missing = ids[~observed & ~handled]
        self.tentative.reset(missing)
        self.capture_count[missing] = 0
        candidate = observed & ~handled
        ti = ids[candidate]
        if ti.numel():
            stamp, glob, loc = stamp_ns[candidate], global_xy[candidate], local_xy[candidate]
            existing = self.capture_count[ti] > 0
            accept = self.tentative.gate(ti, stamp, glob, loc)
            restart = ti[existing & ~accept]
            self.tentative.reset(restart)
            self.capture_count[restart] = 0
            out = self.tentative.step(ti, stamp, glob, loc, torch.ones_like(existing))
            self.capture_count[ti] += out.accepted.long()
            promote = self.capture_count[ti] >= 2
            pi = ti[promote]
            self.confirmed.copy_rows(self.tentative, pi)
            self.active[pi] = True
            self._publish(pi, Snapshot(*(getattr(out, name)[promote] for name in Snapshot.__dataclass_fields__)))
            clear = ti[promote | ~out.accepted]
            self.tentative.reset(clear)
            self.capture_count[clear] = 0
        return self.status[ids]

    def forecast_offsets(self, offsets, process_acceleration_std=None):
        """CV extrapolation using only public state/P, with VisionFilter Q=0.8²."""
        f, q = cv_matrices(offsets.to(self.state))
        return ((f @ self.state[:, None, :, None]).squeeze(-1),
                sym(f @ self.covariance[:, None] @ f.transpose(-1, -2) + q))
