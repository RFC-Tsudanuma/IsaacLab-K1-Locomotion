"""Original camera noise model; independent from the VisionFilter R calibration."""
import torch

def distance_scaled_measurement_std(
    base_std: torch.Tensor,
    distances: torch.Tensor,
    reference_distance: float,
    maximum_scale: float,
) -> torch.Tensor:
    """Scale a baseline measurement standard deviation by observation range.

    The baseline applies at and below ``reference_distance``.  Beyond that
    distance the standard deviation grows linearly with range until reaching
    ``maximum_scale`` times the baseline.
    """
    if reference_distance <= 0.0:
        raise ValueError("reference_distance must be positive")
    if maximum_scale < 1.0:
        raise ValueError("maximum_scale must be at least one")
    distance_scale = torch.clamp(
        distances / float(reference_distance),
        min=1.0,
        max=float(maximum_scale),
    )
    return base_std * distance_scale
