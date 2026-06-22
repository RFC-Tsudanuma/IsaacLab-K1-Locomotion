"""Zarr-backed dataset for the velocity predictor."""

from __future__ import annotations

import torch
import zarr
from torch.utils.data import Dataset


class VelocityDataset(Dataset):
    """Loads (cmd, vel, done [, proprio]) episodes from a zarr group."""

    def __init__(self, zarr_path: str, use_proprio: bool = False, proprio_keys: list[str] | None = None):
        self.root = zarr.open_group(zarr_path, mode="r")
        self.cmd = self.root["cmd"]
        self.vel = self.root["vel"]
        self.done = self.root["done"]

        self.use_proprio = use_proprio
        self.proprio_arrs: dict[str, zarr.Array] = {}
        if use_proprio and proprio_keys:
            proprio_group = self.root["proprio"]
            for key in proprio_keys:
                if key not in proprio_group:
                    raise KeyError(f"proprio/{key} not in dataset {zarr_path}")
                self.proprio_arrs[key] = proprio_group[key]

    def __len__(self) -> int:
        return self.cmd.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "cmd": torch.from_numpy(self.cmd[idx]).float(),
            "vel": torch.from_numpy(self.vel[idx]).float(),
            "done": torch.from_numpy(self.done[idx]).bool(),
        }
        if self.use_proprio and self.proprio_arrs:
            item["proprio"] = torch.cat(
                [torch.from_numpy(self.proprio_arrs[k][idx]).float() for k in sorted(self.proprio_arrs.keys())],
                dim=-1,
            )
        return item


def get_done_mask(done: torch.Tensor) -> torch.Tensor:
    """Build a [0,1] mask that is 1 up to and including the first done step.

    done: (B, T) bool
    return: (B, T) float
    """
    done_int = done.int()
    cum = done_int.cumsum(dim=1)
    mask = (cum == 0).float()
    first_done = ((cum == 1) & (done_int == 1)).float()
    return mask + first_done
