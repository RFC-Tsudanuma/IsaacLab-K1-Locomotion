"""Pinned source parameters plus the approved perception and observation changes."""
from pathlib import Path
import yaml
from .direct_kicking_observation import expected_direct_kicking_observation_size

ROOT = Path(__file__).resolve().parent
SOURCE_REVISION = '3af2acc97f1081b4cbfd9556efd39408dea92cc6'
VISION_REVISION = '32ece6ee0676b1008d5bc58c3533d45613440568'
PERCEPTION_SCHEMA = 'vision_filter_four_hypotheses_public_cv_v1'


def load_config():
    with (ROOT / 'source_config.yaml').open() as stream:
        cfg = yaml.safe_load(stream)
    cfg['env']['num_observations'] = expected_direct_kicking_observation_size(
        len(cfg['direct_kicking']['observation']['prediction_horizons_s'])
    )
    cfg['basic']['task'] = 'Isaac-K1-DirectKick-v0'
    cfg['asset']['file'] = str(ROOT / 'assets/K1/K1_locomotion.urdf')
    cfg['ball']['file'] = str(ROOT / 'assets/ball.urdf')
    cfg['direct_kicking']['filter'] = {
        'process_acceleration_std_mps2': 0.8,
        'initial_velocity_std_mps': 2.5,
        'process_noise_scale_range': [1., 1.],
        'measurement_noise_scale_range': [1., 1.],
        'nis_threshold': 9.21,
        'max_missing_time_s': 3.0,
    }
    # The experiment targets kicks of approaching balls, not outgoing balls.
    cfg['direct_kicking']['ball_motion_randomization']['incoming_probability'] = 1.0
    cfg['migration'] = {
        'source_revision': SOURCE_REVISION,
        'vision_filter_revision': VISION_REVISION,
        'perception_schema': PERCEPTION_SCHEMA,
        'omitted_physical_coefficients': ['rolling_friction', 'compliance'],
    }
    return cfg
