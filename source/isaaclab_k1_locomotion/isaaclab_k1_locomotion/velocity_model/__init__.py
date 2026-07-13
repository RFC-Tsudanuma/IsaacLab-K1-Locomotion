"""Velocity prediction model: command -> actual velocity (1st-order baseline + GRU residual)."""

from .command_sampling import PATTERN_NAMES, pattern_partition, sample_commands
from .dataset import VelocityDataset, get_done_mask
from .predictor import FirstOrderBaseline, GRUResidual, VelocityPredictor

__all__ = [
    "FirstOrderBaseline",
    "GRUResidual",
    "VelocityPredictor",
    "VelocityDataset",
    "get_done_mask",
    "sample_commands",
    "PATTERN_NAMES",
    "pattern_partition",
]
