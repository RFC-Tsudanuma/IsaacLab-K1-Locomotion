"""Joga Appendix A / Tables I--V, with explicit K1/PhysX completion choices.

This is an independent implementation, not the authors' unpublished code.
The completion choices below are described in this package's README.md.
"""
from copy import deepcopy

from ..direct_kick.config import load_config as load_observer_source


TASK_ID = "Isaac-K1-JogaReceive-v0"
PROPRIO_SIZE = 47
HORIZON_COUNT = 13
FORECAST_SIZE = HORIZON_COUNT * 21
FRAME_SIZE = PROPRIO_SIZE + FORECAST_SIZE + 3
HISTORY_LENGTH = 3
POLICY_SIZE = FRAME_SIZE * HISTORY_LENGTH
PRIVILEGED_SIZE = 21
CRITIC_SIZE = POLICY_SIZE + PRIVILEGED_SIZE


def load_config():
    source = load_observer_source()
    source["control"]["decimation"] = 4
    return {
        "task": TASK_ID,
        "seed": 42,
        "contract": "k1_joga_receiving_belief_mlp_v1",
        "physics_dt": 0.005,
        "decimation": 4,
        "num_envs": 16384,
        "history_length": HISTORY_LENGTH,
        "policy_size": POLICY_SIZE,
        "critic_size": CRITIC_SIZE,
        "num_actions": 12,
        "ball": {"radius": 0.11, "mass": 0.43},
        "robot": deepcopy({key: source[key] for key in ("asset", "control", "init_state")}),
        # Reuse the existing observation pipeline without modifying its gates,
        # noise, timing, CV prediction, covariance propagation, or LOST behavior.
        "perception": {
            **deepcopy({key: source[key] for key in ("noise", "vision", "normalization")}),
            "direct_kicking": deepcopy({key: source["direct_kicking"][key] for key in (
                "filter", "perception_randomization", "ego_motion_noise", "observation")}),
        },
        "curriculum": [
            {"iteration": 0, "distance": [1.5, 2.5], "bearing_deg": 20.,
             "offset": 0.10, "speed": [1.5, 2.8], "settle_s": 0.5},
            {"iteration": 8000, "distance": [1.5, 3.0], "bearing_deg": 30.,
             "offset": 0.30, "speed": [2.0, 3.5], "settle_s": 1.0},
            {"iteration": 14000, "distance": [1.5, 3.8], "bearing_deg": 40.,
             "offset": 0.48, "speed": [2.3, 4.5], "settle_s": 1.5},
            {"iteration": 20000, "distance": [1.8, 5.0], "bearing_deg": 45.,
             "offset": 0.60, "speed": [2.6, 5.5], "settle_s": 2.0},
        ],
        "side_curriculum": [[0, 0.0], [5000, 0.35], [10000, 0.55], [15000, 0.70]],
        "reward_weights": {
            "ball_in_zone": 1., "face_ball": .5, "ball_trapped": 2.,
            "settle_progress": 2., "side_first_touch": 6., "velocity_kill": 1.5,
            "near_foot": 2., "approach_loose_ball": 1., "stand_still": 1.,
            "upright": 1., "pose": 2., "joint_limits": -1.,
            "self_collision": -1., "leg_ball": -5.,
            "action_rate": -.3, "torso_angular_velocity": -.05,
            "angular_momentum": -.02, "pelvis_height": -20.,
            "foot_on_ball": -3., "sole_contact": -1., "move_when_blind": -1.,
        },
        # The paper does not numerically specify these receiving details.
        "completion": {
            "episode_s": 8., "settle_speed_mps": .5, "ground_tolerance_m": .03,
            "foot_contact_threshold_n": 1., "ground_contact_threshold_n": 1.,
            "pose_sigma_rad": .5, "pelvis_height_m": .50,
            "fall_height_m": .25, "fall_gravity_z": -.5,
            "foot_above_ball_xy_m": .11,
            "reward_scale_by_dt": True,
        },
        "randomization": {
            "com_offset_m": [.025, .025, .03], "encoder_bias_rad": .015,
            "robot_xy_m": .5, "robot_yaw_rad": 3.141592653589793,
            "push_interval_s": [1., 3.], "push_linear_mps": [.5, .5, .4],
            "push_angular_rps": [.52, .52, .78],
            "foot_friction": [.3, 1.2], "ball_friction": [.5, .9],
            "rough_probability": .5, "rough_height_m": .01,
        },
        # PhysX completion, not a conversion of MuJoCo solref/solimp.
        "physx": {"ball_restitution": .6324555320336759, "foot_restitution": 0.,
                  "ground_friction": 2., "foot_friction": .6,
                  "ball_friction": .75, "contact_offset_m": .005,
                  "terrain_size_m": 12., "terrain_grid_size": 61},
        "model": {"forecast_hidden_size": 128, "forecast_output_size": 64,
                  "actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128]},
        "ppo": {"num_learning_epochs": 5, "num_mini_batches": 4,
                "learning_rate": 1.e-3, "schedule": "adaptive", "desired_kl": .01,
                "gamma": .99, "lam": .95, "clip_param": .2,
                "value_loss_coef": 1., "use_clipped_value_loss": True,
                "entropy_coef": .01, "max_grad_norm": 1.},
        "runner": {"steps_per_env": 24, "save_interval": 500, "max_iterations": 30000},
    }


def curriculum_at(cfg, iteration):
    stage = cfg["curriculum"][0]
    for candidate in cfg["curriculum"]:
        if iteration >= candidate["iteration"]:
            stage = candidate
    side = cfg["side_curriculum"][0][1]
    for start, value in cfg["side_curriculum"]:
        if iteration >= start:
            side = value
    return {**stage, "side_threshold": side}
