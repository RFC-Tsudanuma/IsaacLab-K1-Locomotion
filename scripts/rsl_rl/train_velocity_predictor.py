# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the velocity predictor (1st-order baseline + GRU residual) on collected zarr rollouts.

Reads the (cmd, vel, done [, proprio]) episodes produced by ``collect_velocity_data.py``
and fits a :class:`VelocityPredictor` to map the velocity command sequence to the actual
body-frame velocity ``[vx, vy, wz]`` measured in simulation.

The saved checkpoint matches the contract expected by ``eval_velocity_predictor.py``:
    {
        "model": state_dict,
        "hidden_dim": int,
        "proprio_dim": int,
        "dt": float,
        "residual_scale": float,
        "args": {"use_proprio": bool, "proprio_keys": [...], ...},
    }

This script is pure PyTorch — it does NOT launch Isaac Sim.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split


def _load_velocity_model():
    """Load velocity_model submodules directly by path.

    Importing the ``isaaclab_k1_locomotion`` package runs its ``__init__`` which pulls
    in ``.tasks`` -> ``isaaclab_tasks`` (an Isaac Sim kit extension that is not importable
    in plain Python). The velocity_model submodules only depend on torch/zarr/numpy and
    have no relative imports, so we load them in isolation instead.
    """
    vm_dir = (
        Path(__file__).resolve().parents[2]
        / "source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/velocity_model"
    )
    mods = {}
    for name in ("dataset", "predictor"):
        spec = importlib.util.spec_from_file_location(f"_vm_{name}", vm_dir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["dataset"].VelocityDataset, mods["predictor"].VelocityPredictor, mods["dataset"].get_done_mask


VelocityDataset, VelocityPredictor, get_done_mask = _load_velocity_model()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the velocity predictor on collected rollouts.")
    parser.add_argument("--data", type=str, default="data/rollouts.zarr",
                        help="Path to the zarr group produced by collect_velocity_data.py.")
    parser.add_argument("--output", type=str, default="logs/velocity_predictor/predictor.pt",
                        help="Where to write the trained predictor checkpoint.")
    # 50 was selected by an epoch sweep (40/50/75/100) on the no-proprio frequent-switching
    # dataset: it gives the lowest val MSE with a near-zero train/val gap. Training longer only
    # widens the gap (mild overfitting) without improving validation error.
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--residual_scale", type=float, default=0.5)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_proprio", action="store_true")
    parser.add_argument("--proprio_keys", nargs="+", default=[],
                        choices=["joint_pos", "joint_vel", "base_quat", "projected_gravity",
                                 "base_lin_vel_b", "base_ang_vel_b"])
    return parser.parse_args()


def masked_mse(v_pred: torch.Tensor, v_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-axis MSE over valid (pre-termination) timesteps. mask: (B, T)."""
    err = (v_pred - v_target) ** 2          # (B, T, 3)
    m = mask.unsqueeze(-1)                   # (B, T, 1)
    denom = m.sum() * v_pred.shape[-1] + 1e-8
    return (err * m).sum() / denom


def masked_axis_mae(v_pred: torch.Tensor, v_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-axis MAE (3,) over valid timesteps, for reporting."""
    err = (v_pred - v_target).abs()          # (B, T, 3)
    m = mask.unsqueeze(-1)                    # (B, T, 1)
    denom = m.sum() + 1e-8
    return (err * m).sum(dim=(0, 1)) / denom  # (3,)


def run_epoch(model, loader, device, use_proprio, optimizer=None, grad_clip=1.0):
    train = optimizer is not None
    model.train(train)
    total_loss, total_eps = 0.0, 0
    axis_mae_sum = torch.zeros(3, device=device)
    axis_mae_w = 0.0

    torch.set_grad_enabled(train)
    for batch in loader:
        cmd = batch["cmd"].to(device)          # (B, T, 3)
        vel = batch["vel"].to(device)          # (B, T, 3)
        done = batch["done"].to(device)        # (B, T) bool
        proprio = batch["proprio"].to(device) if use_proprio else None
        mask = get_done_mask(done)             # (B, T) float

        v_pred, _, _, _ = model(cmd, proprio_seq=proprio)
        loss = masked_mse(v_pred, vel, mask)

        if train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = cmd.shape[0]
        total_loss += loss.item() * bs
        total_eps += bs
        w = mask.sum().item()
        axis_mae_sum += masked_axis_mae(v_pred, vel, mask).detach() * w
        axis_mae_w += w
    torch.set_grad_enabled(True)

    avg_loss = total_loss / max(total_eps, 1)
    axis_mae = (axis_mae_sum / max(axis_mae_w, 1e-8)).cpu()
    return avg_loss, axis_mae


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.use_proprio and not args.proprio_keys:
        raise ValueError("--use_proprio requires at least one --proprio_keys entry.")

    print(f"[INFO] Loading dataset: {args.data}")
    dataset = VelocityDataset(
        args.data,
        use_proprio=args.use_proprio,
        proprio_keys=args.proprio_keys if args.use_proprio else None,
    )
    dt = float(dataset.root.attrs.get("dt", 0.02))
    n_total = len(dataset)
    if n_total == 0:
        raise RuntimeError(f"Dataset {args.data} is empty.")

    # Determine proprio_dim from a single sample (sorted-concat dim).
    sample = dataset[0]
    proprio_dim = int(sample["proprio"].shape[-1]) if args.use_proprio else 0

    n_val = max(1, int(n_total * args.val_frac)) if n_total > 1 else 0
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(args.seed)
    if n_val > 0:
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    ) if val_ds is not None else None

    print(
        f"[INFO] episodes={n_total:,} (train={n_train:,}, val={n_val:,}), dt={dt:.4f}, "
        f"hidden_dim={args.hidden_dim}, proprio_dim={proprio_dim}, "
        f"use_proprio={args.use_proprio}, proprio_keys={args.proprio_keys}, device={device}"
    )

    model = VelocityPredictor(
        dim=3, dt=dt, hidden_dim=args.hidden_dim,
        proprio_dim=proprio_dim, residual_scale=args.residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save_ckpt(path: Path) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "hidden_dim": args.hidden_dim,
                "proprio_dim": proprio_dim,
                "dt": dt,
                "residual_scale": args.residual_scale,
                "args": vars(args),
            },
            str(path),
        )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mae = run_epoch(model, train_loader, device, args.use_proprio,
                                    optimizer=optimizer, grad_clip=args.grad_clip)
        if val_loader is not None:
            va_loss, va_mae = run_epoch(model, val_loader, device, args.use_proprio)
        else:
            va_loss, va_mae = tr_loss, tr_mae
        scheduler.step()

        tau = torch.exp(model.baseline.log_tau).detach().cpu()
        gain = model.baseline.gain.detach().cpu()
        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"train_mse={tr_loss:.5f} val_mse={va_loss:.5f} | "
            f"val_MAE vx={va_mae[0]:.4f} vy={va_mae[1]:.4f} wz={va_mae[2]:.4f} | "
            f"tau={tau.tolist()} gain={gain.tolist()}"
        )

        if va_loss < best_val:
            best_val = va_loss
            save_ckpt(out_path)

    # Always leave a final checkpoint alongside the best one.
    final_path = out_path.with_name(out_path.stem + "_final" + out_path.suffix)
    save_ckpt(final_path)
    print(f"[INFO] best val_mse={best_val:.5f} -> {out_path}")
    print(f"[INFO] final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
