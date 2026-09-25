import os
import glob
import yaml
import argparse
import numpy as np
import random
import time
import signal
import imageio
import subprocess
import sys
import re

# Import envs first to initialize isaacgym modules
from envs import *

# Import torch and utils after isaacgym modules are initialized
import torch
import torch.nn.functional as F
from utils.models.BaseAC import *
from utils.buffer import ExperienceBuffer
from utils.model_factory import (
    build_model,
    checkpoint_model_metadata,
    get_model_class,
    validate_checkpoint_model_metadata,
)
from utils.post_kick_phase import (
    class_balanced_phase_multipliers,
    weighted_phase_binary_cross_entropy,
)
from utils.utils import discount_values, surrogate_loss
from utils.recorder import Recorder

# Dynamic task class loading
import importlib
import inspect

def get_task_class(task_name):
    """
    Dynamically load task class by name.
    Searches through all modules in the envs package for classes that match the task name.
    Handles different naming conventions (Base_Walk vs BaseWalk, etc.)
    """
    # Generate possible class name variations
    possible_names = [task_name]
    
    # Handle underscore to camelCase conversion (Base_Walk -> BaseWalk)
    if '_' in task_name:
        camel_case = ''.join(word.capitalize() for word in task_name.split('_'))
        possible_names.append(camel_case)
    
    # Handle camelCase to underscore conversion (BaseWalk -> Base_Walk)
    if not '_' in task_name and any(c.isupper() for c in task_name[1:]):
        import re
        snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', task_name).lower()
        snake_case = snake_case[0].upper() + snake_case[1:]  # Capitalize first letter
        possible_names.append(snake_case)
    
    # First try to get from the envs module (which imports all task classes)
    try:
        envs_module = importlib.import_module('envs')
        for name, obj in inspect.getmembers(envs_module):
            if inspect.isclass(obj) and name in possible_names:
                return obj
    except Exception as e:
        print(f"Error loading from envs module: {e}")
    
    # If not found, try to import from specific paths
    task_paths = [
        f"envs.T1.{task_name.lower()}",
        f"envs.K1.{task_name.lower()}",
        f"envs.{task_name}",
    ]
    
    for path in task_paths:
        try:
            module = importlib.import_module(path)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name in possible_names:
                    return obj
        except ImportError:
            continue
        except Exception as e:
            print(f"Error loading from {path}: {e}")
            continue
    
    return None


class Runner:

    def __init__(self, test=False):
        self.test = test
        # prepare the environment
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()
        task_name = self.cfg["basic"]["task"]
        # Extract task name from path (e.g., "T1/T1" -> "T1")
        if "/" in task_name:
            task_name = task_name.split("/")[-1]
        
        # Dynamically load the task class
        task_class = get_task_class(task_name)
        if task_class is None:
            raise ValueError(f"Unknown task: {task_name}. Could not find a class named '{task_name}' in the envs package.")
        
        self.env = task_class(self.cfg)
        self.env.is_play = test

        self.device = self.cfg["basic"]["rl_device"]
        self.learning_rate = self.cfg["algorithm"]["learning_rate"]
        self.init_learning_rate = self.learning_rate
        # Select model by config/CLI
        model_name = self.cfg["basic"].get("model", "BaseActorCritic")
        self.model = build_model(
            model_name,
            self.env.num_actions,
            self.env.num_obs,
            self.env.num_privileged_obs,
            self.cfg,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.start_iteration = 0
        self._load()
        self._init_post_kick_phase_auxiliary()
        self._init_adaptive_speed_curriculum()
        self._init_adaptive_kick_curriculum()
        self._init_stage_plateau_curriculum()

        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.env.num_actions,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)
        if self.post_kick_phase_auxiliary_enabled:
            self.buffer.add_buffer("post_kick_phase_targets", ())

    def _init_post_kick_phase_auxiliary(self):
        phase_cfg = self.cfg["algorithm"].get(
            "post_kick_phase_auxiliary",
            {},
        )
        self.post_kick_phase_auxiliary_enabled = bool(
            phase_cfg.get("enabled", False)
        )
        self.post_kick_phase_loss_coefficient = float(
            phase_cfg.get("coefficient", 0.0)
        )
        self.post_kick_phase_premature_weight = float(
            phase_cfg.get("premature_weight", 3.0)
        )
        self.post_kick_phase_delayed_weight = float(
            phase_cfg.get("delayed_weight", 1.0)
        )
        if self.post_kick_phase_loss_coefficient < 0.0:
            raise ValueError(
                "post-kick phase loss coefficient must be non-negative"
            )
        if (
            self.post_kick_phase_auxiliary_enabled
            and self.post_kick_phase_loss_coefficient == 0.0
        ):
            raise ValueError(
                "enabled post-kick phase training requires a positive "
                "loss coefficient"
            )
        if (
            self.post_kick_phase_premature_weight <= 0.0
            or self.post_kick_phase_delayed_weight <= 0.0
        ):
            raise ValueError("post-kick phase class weights must be positive")
        if self.post_kick_phase_auxiliary_enabled and not callable(
            getattr(self.model, "post_kick_phase_logit", None)
        ):
            raise ValueError(
                "post-kick phase auxiliary training requires a compatible model"
            )

    def _current_post_kick_phase_target(self):
        target = self.env.extras.get("post_kick_phase_target")
        if target is None:
            raise RuntimeError(
                "post-kick phase auxiliary training requires an environment target"
            )
        target = target.to(self.device)
        if target.shape != (self.env.num_envs,):
            raise RuntimeError(
                "post-kick phase environment target has the wrong shape"
            )
        return target

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument("--init_from_checkpoint", action="store_true", default=None, help="Load only policy/value network weights from checkpoint and reset optimizer/curriculum/stage progress.")
        parser.add_argument("--num_envs", type=int, help="Number of environments to create. Overrides config file if provided.")
        parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window. Overrides config file if provided.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation. Overrides config file if provided.")
        parser.add_argument("--rl_device", type=str, help="Device for the RL algorithm. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations. Overrides config file if provided.")
        parser.add_argument("--model", type=str, help="Model class name to use (e.g., BaseActorCritic, OdometryActorCritic). Overrides config file if provided.")
        # Video recording mode arguments (for separate process recording)
        parser.add_argument("--record_video_mode", action="store_true", help="Enable video recording mode (record and exit).")
        parser.add_argument("--video_duration", type=float, help="Duration of video to record in seconds.")
        parser.add_argument("--video_iteration", type=int, help="Iteration number for wandb logging.")
        parser.add_argument("--video_output_path", type=str, help="Path where to save the video file.")
        parser.add_argument("--rewards_output_path", type=str, help="Path where to save the reward data JSON file.")
        self.args = parser.parse_args()

    # Override config file with args if needed
    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", "{}.yaml".format(self.args.task))
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        # Ensure default model if not present in config
        if "model" not in self.cfg.get("basic", {}):
            self.cfg.setdefault("basic", {})["model"] = "BaseActorCritic"
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)
        if not self.test:
            # Disable video recording in training process - videos will be recorded in separate process
            self.cfg["viewer"]["record_video"] = False

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print("Setting seed: {}".format(self.cfg["basic"]["seed"]))

        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def _load(self):
        checkpoint = self.cfg["basic"].get("checkpoint")
        if isinstance(checkpoint, str) and checkpoint.strip().lower() in {"", "none", "null"}:
            self.cfg["basic"]["checkpoint"] = ""
            return
        if not checkpoint:
            return
        if (checkpoint == "-1") or (checkpoint == -1):
            # Look for models in hierarchical structure: logs/robot_type/task_name/**/*.pth
            task_name = self.cfg["basic"]["task"]
            robot_type = self._get_robot_type(task_name)
            
            # First try: exact task in robot-specific folder
            task_log_pattern = os.path.join("logs", robot_type, task_name, "**/*.pth")
            task_models = sorted(glob.glob(task_log_pattern, recursive=True), key=os.path.getmtime)
            
            if task_models:
                self.cfg["basic"]["checkpoint"] = task_models[-1]
            else:
                # Second try: any task in robot-specific folder
                robot_log_pattern = os.path.join("logs", robot_type, "**/*.pth")
                robot_models = sorted(glob.glob(robot_log_pattern, recursive=True), key=os.path.getmtime)
                
                if robot_models:
                    self.cfg["basic"]["checkpoint"] = robot_models[-1]
                else:
                    # Fallback: all logs if no robot-specific models found
                    self.cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print("Loading model from {}".format(self.cfg["basic"]["checkpoint"]))
        model_dict = torch.load(self.cfg["basic"]["checkpoint"], map_location=self.device, weights_only=True)
        initialize_only = self.cfg["basic"].get("init_from_checkpoint", False)
        checkpoint_initializer = getattr(
            self.model,
            "initialize_from_checkpoint",
            None,
        )
        if initialize_only and callable(checkpoint_initializer):
            checkpoint_initializer(model_dict)
        else:
            validate_checkpoint_model_metadata(self.model, model_dict)
            strict_model_load = checkpoint_model_metadata(self.model) is not None
            self.model.load_state_dict(
                model_dict["model"],
                strict=strict_model_load,
            )
        if initialize_only:
            print("Initialized model weights from checkpoint; optimizer and curriculum were reset.")
            return
        try:
            self.env.curriculum_prob = model_dict["curriculum"]
        except Exception as e:
            print(f"Failed to load curriculum: {e}")
        try:
            self.optimizer.load_state_dict(model_dict["optimizer"])
        except Exception as e:
            print(f"Failed to load optimizer: {e}")
        try:
            if hasattr(self.env, "load_curriculum_state"):
                self.env.load_curriculum_state(model_dict.get("stage_curriculum"))
        except Exception as e:
            print(f"Failed to load stage curriculum state: {e}")
        self.start_iteration = self._checkpoint_iteration(model_dict)
        if self.start_iteration > 0:
            print(f"Resuming training from iteration {self.start_iteration}")

    def _checkpoint_iteration(self, model_dict):
        iteration = model_dict.get("iteration")
        if iteration is not None:
            return int(iteration)

        checkpoint = str(self.cfg["basic"].get("checkpoint", ""))
        match = re.search(r"model_(?:interrupt_)?(\d+)\.pth$", os.path.basename(checkpoint))
        if match:
            return int(match.group(1))
        return 0

    def _init_adaptive_speed_curriculum(self):
        self.adaptive_speed_cfg = self.cfg.get("adaptive_speed_curriculum", {})
        self.adaptive_speed_enabled = bool(self.adaptive_speed_cfg.get("enabled", False))
        self.adaptive_speed_mode = "slow"
        self.adaptive_speed_rewards = []
        self.adaptive_speed_last_switch_iteration = 0
        self.adaptive_speed_last_variance = 0.0
        self.adaptive_speed_last_slope = 0.0
        self.adaptive_speed_last_improvement = 0.0
        self.adaptive_speed_switch_count = 0
        self.adaptive_speed_offset = 0.0

    def _set_adaptive_speed_mode(self, mode, iteration, variance=0.0, slope=0.0, improvement=0.0, reason="init"):
        if not self.adaptive_speed_enabled:
            return False
        if not hasattr(self.env, "set_adaptive_speed_mode"):
            print("Adaptive speed curriculum is enabled, but this environment does not support speed mode switching. Disabling it.")
            self.adaptive_speed_enabled = False
            return False

        previous_mode = self.adaptive_speed_mode
        offset = float(self.adaptive_speed_cfg.get("fast_lin_vel_x_offset", 1.0)) if mode == "fast" else 0.0
        self.adaptive_speed_mode = mode
        self.adaptive_speed_offset = offset
        self.adaptive_speed_rewards = []
        self.adaptive_speed_last_switch_iteration = int(iteration)
        self.adaptive_speed_last_variance = float(variance)
        self.adaptive_speed_last_slope = float(slope)
        self.adaptive_speed_last_improvement = float(improvement)
        if reason != "init" and previous_mode != mode:
            self.adaptive_speed_switch_count += 1
        self.env.set_adaptive_speed_mode(mode, offset)
        print(
            "Adaptive speed curriculum: iteration={} mode={} offset={:.3f} variance={:.6g} slope={:.6g} improvement={:.6g} reason={}".format(
                iteration,
                mode,
                offset,
                variance,
                slope,
                improvement,
                reason,
            )
        )
        return True

    def _adaptive_speed_stage_enabled(self):
        if not self.adaptive_speed_enabled:
            return False
        if hasattr(self.env, "is_adaptive_speed_stage"):
            return bool(self.env.is_adaptive_speed_stage())
        return True

    def _update_adaptive_speed_curriculum(self, reward_mean, iteration):
        if not self.adaptive_speed_enabled:
            return False
        if not self._adaptive_speed_stage_enabled():
            if self.adaptive_speed_mode != "slow" or self.adaptive_speed_rewards:
                return self._set_adaptive_speed_mode("slow", iteration, reason="inactive_stage")
            return False

        self.adaptive_speed_rewards.append(float(reward_mean))
        default_min_samples = int(self.adaptive_speed_cfg.get("min_iterations_per_phase", 200))
        if self.adaptive_speed_mode == "fast":
            min_samples = int(self.adaptive_speed_cfg.get("fast_min_iterations_per_phase", default_min_samples))
        else:
            min_samples = int(self.adaptive_speed_cfg.get("slow_min_iterations_per_phase", default_min_samples))
        if len(self.adaptive_speed_rewards) < min_samples:
            return False

        detector_window = max(2, int(self.adaptive_speed_cfg.get("detector_window_iterations", 200)))
        rewards_window = self.adaptive_speed_rewards[-detector_window:]
        if len(rewards_window) < 2:
            return False

        window = np.asarray(rewards_window, dtype=np.float64)
        variance = float(np.var(window))
        if len(window) > 1:
            x = np.arange(len(window), dtype=np.float64)
            slope = float(np.polyfit(x, window, 1)[0])
        else:
            slope = 0.0

        improvement_window = max(1, int(self.adaptive_speed_cfg.get("improvement_window_iterations", detector_window)))
        recent = self.adaptive_speed_rewards[-improvement_window:]
        previous = self.adaptive_speed_rewards[-2 * improvement_window:-improvement_window]
        if len(previous) == 0:
            recent_best_improvement = float(max(recent) - self.adaptive_speed_rewards[0])
        else:
            recent_best_improvement = float(max(recent) - max(previous))

        self.adaptive_speed_last_variance = variance
        self.adaptive_speed_last_slope = slope
        self.adaptive_speed_last_improvement = recent_best_improvement

        slope_threshold = float(self.adaptive_speed_cfg.get("reward_slope_threshold", 1.0e-4))
        improvement_threshold = float(self.adaptive_speed_cfg.get("reward_improvement_threshold", 1.0e-3))
        variance_threshold = self.adaptive_speed_cfg.get("reward_variance_threshold")
        plateau = abs(slope) <= slope_threshold and recent_best_improvement <= improvement_threshold
        if variance_threshold is not None:
            plateau = plateau and variance <= float(variance_threshold)

        min_reward_mean = self.adaptive_speed_cfg.get("min_reward_mean")
        if min_reward_mean is not None and float(np.mean(window)) < float(min_reward_mean):
            return False

        phase_iterations = int(iteration) - self.adaptive_speed_last_switch_iteration + 1
        if self.adaptive_speed_mode == "slow":
            if plateau:
                return self._set_adaptive_speed_mode(
                    "fast",
                    iteration,
                    variance,
                    slope,
                    recent_best_improvement,
                    "slow_reward_plateau",
                )
            return False

        fast_max_iterations = int(self.adaptive_speed_cfg.get("fast_max_iterations", 2000))
        if plateau:
            return self._set_adaptive_speed_mode(
                "slow",
                iteration,
                variance,
                slope,
                recent_best_improvement,
                "fast_reward_plateau",
            )
        if phase_iterations >= fast_max_iterations:
            return self._set_adaptive_speed_mode(
                "slow",
                iteration,
                variance,
                slope,
                recent_best_improvement,
                "fast_max_iterations",
            )
        return False

    def _adaptive_speed_statistics(self):
        if not self.adaptive_speed_enabled:
            return {}
        return {
            "adaptive_speed/mode": 1.0 if self.adaptive_speed_mode == "fast" else 0.0,
            "adaptive_speed/lin_vel_x_offset": self.adaptive_speed_offset,
            "adaptive_speed/reward_variance": self.adaptive_speed_last_variance,
            "adaptive_speed/reward_slope": self.adaptive_speed_last_slope,
            "adaptive_speed/reward_best_improvement": self.adaptive_speed_last_improvement,
            "adaptive_speed/phase_iterations": len(self.adaptive_speed_rewards),
            "adaptive_speed/switch_count": self.adaptive_speed_switch_count,
        }

    def _init_adaptive_kick_curriculum(self):
        self.adaptive_kick_cfg = self.cfg.get("adaptive_kick_curriculum", {})
        self.adaptive_kick_enabled = bool(self.adaptive_kick_cfg.get("enabled", False))
        self.adaptive_kick_mode = "normal"
        self.adaptive_kick_rewards = []
        self.adaptive_kick_last_switch_iteration = 0
        self.adaptive_kick_last_variance = 0.0
        self.adaptive_kick_last_slope = 0.0
        self.adaptive_kick_last_improvement = 0.0
        self.adaptive_kick_switch_count = 0
        self.adaptive_kick_stage_offset = int(self.adaptive_kick_cfg.get("hard_stage_offset", 1))
        self.adaptive_kick_sampling_stage_idx = -1
        self.adaptive_kick_sampling_stage_name = "none"

    def _set_adaptive_kick_mode(self, mode, iteration, variance=0.0, slope=0.0, improvement=0.0, reason="init"):
        if not self.adaptive_kick_enabled:
            return False
        if not hasattr(self.env, "set_adaptive_kick_mode"):
            print("Adaptive kick curriculum is enabled, but this environment does not support kick mode switching. Disabling it.")
            self.adaptive_kick_enabled = False
            return False

        previous_mode = self.adaptive_kick_mode
        self.adaptive_kick_mode = mode
        self.adaptive_kick_rewards = []
        self.adaptive_kick_last_switch_iteration = int(iteration)
        self.adaptive_kick_last_variance = float(variance)
        self.adaptive_kick_last_slope = float(slope)
        self.adaptive_kick_last_improvement = float(improvement)
        if reason != "init" and previous_mode != mode:
            self.adaptive_kick_switch_count += 1
        self.env.set_adaptive_kick_mode(mode, self.adaptive_kick_stage_offset)

        kick_state = {}
        if hasattr(self.env, "get_adaptive_kick_state"):
            kick_state = self.env.get_adaptive_kick_state()
        self.adaptive_kick_sampling_stage_idx = int(kick_state.get("sampling_stage_idx", -1))
        self.adaptive_kick_sampling_stage_name = kick_state.get("sampling_stage_name", "none")
        print(
            "Adaptive kick curriculum: iteration={} mode={} sampling_stage={} variance={:.6g} slope={:.6g} improvement={:.6g} reason={}".format(
                iteration,
                mode,
                self.adaptive_kick_sampling_stage_name,
                variance,
                slope,
                improvement,
                reason,
            )
        )
        return True

    def _adaptive_kick_stage_enabled(self):
        if not self.adaptive_kick_enabled:
            return False
        if hasattr(self.env, "is_adaptive_kick_stage"):
            return bool(self.env.is_adaptive_kick_stage())
        return False

    def _update_adaptive_kick_curriculum(self, reward_mean, done_rate, iteration):
        if not self.adaptive_kick_enabled:
            return False
        if not self._adaptive_kick_stage_enabled():
            if self.adaptive_kick_mode != "normal" or self.adaptive_kick_rewards:
                return self._set_adaptive_kick_mode("normal", iteration, reason="inactive_stage")
            return False

        self.adaptive_kick_rewards.append(float(reward_mean))
        default_min_samples = int(self.adaptive_kick_cfg.get("min_iterations_per_phase", 200))
        if self.adaptive_kick_mode == "hard":
            min_samples = int(self.adaptive_kick_cfg.get("hard_min_iterations_per_phase", default_min_samples))
        else:
            min_samples = int(self.adaptive_kick_cfg.get("normal_min_iterations_per_phase", default_min_samples))
        if len(self.adaptive_kick_rewards) < min_samples:
            return False

        detector_window = max(2, int(self.adaptive_kick_cfg.get("detector_window_iterations", 200)))
        rewards_window = self.adaptive_kick_rewards[-detector_window:]
        if len(rewards_window) < 2:
            return False

        window = np.asarray(rewards_window, dtype=np.float64)
        variance = float(np.var(window))
        if len(window) > 1:
            x = np.arange(len(window), dtype=np.float64)
            slope = float(np.polyfit(x, window, 1)[0])
        else:
            slope = 0.0

        improvement_window = max(1, int(self.adaptive_kick_cfg.get("improvement_window_iterations", detector_window)))
        recent = self.adaptive_kick_rewards[-improvement_window:]
        previous = self.adaptive_kick_rewards[-2 * improvement_window:-improvement_window]
        if len(previous) == 0:
            recent_best_improvement = float(max(recent) - self.adaptive_kick_rewards[0])
        else:
            recent_best_improvement = float(max(recent) - max(previous))

        self.adaptive_kick_last_variance = variance
        self.adaptive_kick_last_slope = slope
        self.adaptive_kick_last_improvement = recent_best_improvement

        slope_threshold = float(self.adaptive_kick_cfg.get("reward_slope_threshold", 1.0e-5))
        improvement_threshold = float(self.adaptive_kick_cfg.get("reward_improvement_threshold", 2.0e-3))
        variance_threshold = self.adaptive_kick_cfg.get("reward_variance_threshold")
        plateau = abs(slope) <= slope_threshold and recent_best_improvement <= improvement_threshold
        if variance_threshold is not None:
            plateau = plateau and variance <= float(variance_threshold)

        max_done_rate = self.adaptive_kick_cfg.get("max_done_rate")
        if max_done_rate is not None and float(done_rate) > float(max_done_rate):
            return False

        min_reward_mean = self.adaptive_kick_cfg.get("min_reward_mean")
        if min_reward_mean is not None and float(np.mean(window)) < float(min_reward_mean):
            return False

        phase_iterations = int(iteration) - self.adaptive_kick_last_switch_iteration + 1
        if self.adaptive_kick_mode == "normal":
            if plateau:
                return self._set_adaptive_kick_mode(
                    "hard",
                    iteration,
                    variance,
                    slope,
                    recent_best_improvement,
                    "normal_reward_plateau",
                )
            return False

        hard_max_iterations = int(self.adaptive_kick_cfg.get("hard_max_iterations", 2000))
        if plateau:
            return self._set_adaptive_kick_mode(
                "normal",
                iteration,
                variance,
                slope,
                recent_best_improvement,
                "hard_reward_plateau",
            )
        if phase_iterations >= hard_max_iterations:
            return self._set_adaptive_kick_mode(
                "normal",
                iteration,
                variance,
                slope,
                recent_best_improvement,
                "hard_max_iterations",
            )
        return False

    def _adaptive_kick_statistics(self):
        if not self.adaptive_kick_enabled:
            return {}
        return {
            "adaptive_kick/mode": 1.0 if self.adaptive_kick_mode == "hard" else 0.0,
            "adaptive_kick/sampling_stage_idx": self.adaptive_kick_sampling_stage_idx,
            "adaptive_kick/reward_variance": self.adaptive_kick_last_variance,
            "adaptive_kick/reward_slope": self.adaptive_kick_last_slope,
            "adaptive_kick/reward_best_improvement": self.adaptive_kick_last_improvement,
            "adaptive_kick/phase_iterations": len(self.adaptive_kick_rewards),
            "adaptive_kick/switch_count": self.adaptive_kick_switch_count,
        }

    def _init_stage_plateau_curriculum(self):
        self.stage_plateau_cfg = self.cfg.get("stage_curriculum", {})
        stage_progression = self.stage_plateau_cfg.get("progression")
        self.stage_plateau_enabled = bool(
            self.stage_plateau_cfg.get("enabled", False)
            and stage_progression in ("adaptive_reward_plateau", "fixed_then_adaptive_reward_plateau")
        )
        self.stage_plateau_rewards = []
        self.stage_plateau_last_stage_idx = getattr(self.env, "active_stage_idx", -1)
        self.stage_plateau_last_switch_iteration = 0
        self.stage_plateau_last_variance = 0.0
        self.stage_plateau_last_slope = 0.0
        self.stage_plateau_last_improvement = 0.0
        self.stage_plateau_switch_count = 0

    def _reset_stage_plateau_curriculum(self, iteration):
        self.stage_plateau_rewards = []
        self.stage_plateau_last_stage_idx = getattr(self.env, "active_stage_idx", -1)
        self.stage_plateau_last_switch_iteration = int(iteration)
        self.stage_plateau_last_variance = 0.0
        self.stage_plateau_last_slope = 0.0
        self.stage_plateau_last_improvement = 0.0

    def _update_stage_plateau_curriculum(self, reward_mean, done_rate, iteration, per_env_step, total_env_steps):
        if not self.stage_plateau_enabled:
            return False
        if not hasattr(self.env, "advance_curriculum_stage"):
            print("Stage plateau curriculum is enabled, but this environment does not support adaptive stage progression. Disabling it.")
            self.stage_plateau_enabled = False
            return False

        stage_idx = getattr(self.env, "active_stage_idx", -1)
        stages = getattr(self.env, "stage_curriculum_stages", [])
        if stage_idx != self.stage_plateau_last_stage_idx:
            self._reset_stage_plateau_curriculum(iteration)
        if stage_idx < 0 or stage_idx >= len(stages) - 1:
            return False
        if hasattr(self.env, "can_advance_curriculum_stage") and not self.env.can_advance_curriculum_stage():
            return False

        self.stage_plateau_rewards.append(float(reward_mean))
        min_samples = int(self.stage_plateau_cfg.get("min_iterations_per_stage", 1000))
        if len(self.stage_plateau_rewards) < min_samples:
            return False

        detector_window = max(2, int(self.stage_plateau_cfg.get("detector_window_iterations", 100)))
        rewards_window = self.stage_plateau_rewards[-detector_window:]
        if len(rewards_window) < 2:
            return False

        window = np.asarray(rewards_window, dtype=np.float64)
        variance = float(np.var(window))
        x = np.arange(len(window), dtype=np.float64)
        slope = float(np.polyfit(x, window, 1)[0])

        improvement_window = max(1, int(self.stage_plateau_cfg.get("improvement_window_iterations", detector_window)))
        recent = self.stage_plateau_rewards[-improvement_window:]
        previous = self.stage_plateau_rewards[-2 * improvement_window:-improvement_window]
        if len(previous) == 0:
            recent_best_improvement = float(max(recent) - self.stage_plateau_rewards[0])
        else:
            recent_best_improvement = float(max(recent) - max(previous))

        self.stage_plateau_last_variance = variance
        self.stage_plateau_last_slope = slope
        self.stage_plateau_last_improvement = recent_best_improvement

        slope_threshold = float(self.stage_plateau_cfg.get("reward_slope_threshold", 1.0e-5))
        improvement_threshold = float(self.stage_plateau_cfg.get("reward_improvement_threshold", 2.0e-3))
        variance_threshold = self.stage_plateau_cfg.get("reward_variance_threshold")
        plateau = abs(slope) <= slope_threshold and recent_best_improvement <= improvement_threshold
        if variance_threshold is not None:
            plateau = plateau and variance <= float(variance_threshold)

        max_done_rate = self.stage_plateau_cfg.get("max_done_rate")
        if max_done_rate is not None and float(done_rate) > float(max_done_rate):
            return False

        min_reward_mean = self.stage_plateau_cfg.get("min_reward_mean")
        if min_reward_mean is not None and float(np.mean(window)) < float(min_reward_mean):
            return False

        phase_iterations = int(iteration) - self.stage_plateau_last_switch_iteration + 1
        max_iterations = self.stage_plateau_cfg.get("max_iterations_per_stage")
        force_advance = max_iterations is not None and phase_iterations >= int(max_iterations)
        if not plateau and not force_advance:
            return False

        old_stage_name = getattr(self.env, "active_stage", {}).get("name", "none")
        advanced = self.env.advance_curriculum_stage(iteration, per_env_step, total_env_steps)
        if not advanced:
            return False

        self.stage_plateau_switch_count += 1
        new_stage_name = getattr(self.env, "active_stage", {}).get("name", "none")
        print(
            "Stage plateau curriculum: iteration={} {} -> {} variance={:.6g} slope={:.6g} improvement={:.6g} done_rate={:.6g} reason={}".format(
                iteration,
                old_stage_name,
                new_stage_name,
                variance,
                slope,
                recent_best_improvement,
                done_rate,
                "max_iterations" if force_advance else "reward_plateau",
            )
        )
        self._reset_stage_plateau_curriculum(iteration)
        return True

    def _stage_plateau_statistics(self):
        if not self.stage_plateau_enabled:
            return {}
        return {
            "stage_curriculum/phase_iterations": len(self.stage_plateau_rewards),
            "stage_curriculum/reward_variance": self.stage_plateau_last_variance,
            "stage_curriculum/reward_slope": self.stage_plateau_last_slope,
            "stage_curriculum/reward_best_improvement": self.stage_plateau_last_improvement,
            "stage_curriculum/switch_count": self.stage_plateau_switch_count,
        }

    def _checkpoint_payload(self, iteration):
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "curriculum": self.env.curriculum_prob,
            "iteration": iteration,
        }
        model_metadata = checkpoint_model_metadata(self.model)
        if model_metadata is not None:
            payload["model_metadata"] = model_metadata
        if hasattr(self.env, "get_curriculum_state"):
            payload["stage_curriculum"] = self.env.get_curriculum_state()
        return payload

    def train(self):
        self.recorder = Recorder(self.cfg)
        self._set_adaptive_speed_mode("slow", self.start_iteration, reason="init")
        self._set_adaptive_kick_mode("normal", self.start_iteration, reason="init")
        if hasattr(self.env, "set_curriculum_progress"):
            per_env_step = self.start_iteration * self.cfg["runner"]["horizon_length"]
            total_env_steps = per_env_step * self.cfg["env"]["num_envs"]
            stage_changed = self.env.set_curriculum_progress(per_env_step, self.start_iteration, total_env_steps)
            if stage_changed and self.stage_plateau_enabled:
                self._reset_stage_plateau_curriculum(self.start_iteration)
            if stage_changed and self.adaptive_speed_enabled:
                self._set_adaptive_speed_mode("slow", self.start_iteration, reason="stage_changed")
            if stage_changed and self.adaptive_kick_enabled:
                self._set_adaptive_kick_mode("normal", self.start_iteration, reason="stage_changed")
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        self.current_iteration = self.start_iteration - 1
        
        # Get video logging configuration
        use_wandb = self.cfg["runner"].get("use_wandb", False)
        log_video_interval = self.cfg["runner"].get("log_video_interval", None)
        if log_video_interval is None:
            log_video_interval = self.cfg["runner"].get("save_interval", None)
        # Ensure log_video_interval is a positive integer
        if log_video_interval is not None and log_video_interval <= 0:
            log_video_interval = None
        log_video_duration = self.cfg["runner"].get("log_video_duration", 10.0)
        
        for it in range(self.start_iteration, self.cfg["basic"]["max_iterations"]):
            self.current_iteration = it
            per_env_step = it * self.cfg["runner"]["horizon_length"]
            total_env_steps = per_env_step * self.cfg["env"]["num_envs"]
            if hasattr(self.env, "set_curriculum_progress"):
                stage_changed = self.env.set_curriculum_progress(per_env_step, it, total_env_steps)
                if stage_changed and self.stage_plateau_enabled:
                    self._reset_stage_plateau_curriculum(it)
                if stage_changed and self.adaptive_speed_enabled:
                    self._set_adaptive_speed_mode("slow", it, reason="stage_changed")
                if stage_changed and self.adaptive_kick_enabled:
                    self._set_adaptive_kick_mode("normal", it, reason="stage_changed")
                if stage_changed and it > 0:
                    obs, infos = self.env.reset()
                    obs = obs.to(self.device)
                    privileged_obs = infos["privileged_obs"].to(self.device)

            # Check if it's time to log a video
            should_log_video = (use_wandb and 
                               log_video_interval is not None and 
                               log_video_interval > 0 and
                               (it + 1) % log_video_interval == 0)
            
            # Save checkpoint if needed (for video recording or regular save interval)
            should_save = False
            checkpoint_path = None
            if (it + 1) % self.cfg["runner"]["save_interval"] == 0:
                should_save = True
                checkpoint_path = os.path.join(self.recorder.model_dir, f"model_{it + 1}.pth")
                self.recorder.save(
                    self._checkpoint_payload(it + 1),
                    it + 1,
                )
            
            if should_log_video:
                # If we didn't save yet, save checkpoint now for video recording
                if not should_save:
                    checkpoint_path = os.path.join(self.recorder.model_dir, f"model_{it + 1}.pth")
                    self.recorder.save(
                        self._checkpoint_payload(it + 1),
                        it + 1,
                    )
                # Spawn separate process to record video (will wait for completion)
                # Note: Video will be uploaded at step it+1 (after training loop logs at step it)
                self._spawn_video_recording_process(checkpoint_path, it, log_video_duration)
            # within horizon_length, env.step() is called with same act
            for n in range(self.cfg["runner"]["horizon_length"]):
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                if self.post_kick_phase_auxiliary_enabled:
                    self.buffer.update_data(
                        "post_kick_phase_targets",
                        n,
                        self._current_post_kick_phase_target(),
                    )
                with torch.no_grad():
                    dist = self.model.act(obs)
                    act = dist.sample()
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                self.buffer.update_data("actions", n, act)
                self.buffer.update_data("rewards", n, rew)
                self.buffer.update_data("dones", n, done)
                self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))
                ep_info = {"reward": rew}
                ep_info.update(infos["rew_terms"])
                self.recorder.record_episode_statistics(done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1))

            rollout_reward_mean = self.buffer["rewards"].mean().item()
            rollout_done_rate = self.buffer["dones"].float().mean().item()

            flat_obses = self.buffer["obses"].reshape(-1, self.env.num_obs)
            flat_privileged_obses = self.buffer["privileged_obses"].reshape(
                -1,
                self.env.num_privileged_obs,
            )
            flat_actions = self.buffer["actions"].reshape(-1, self.env.num_actions)
            sample_count = flat_obses.shape[0]
            if self.post_kick_phase_auxiliary_enabled:
                flat_post_kick_phase_targets = self.buffer[
                    "post_kick_phase_targets"
                ].reshape(-1)
                post_kick_phase_multipliers = (
                    class_balanced_phase_multipliers(
                        flat_post_kick_phase_targets,
                        self.post_kick_phase_premature_weight,
                        self.post_kick_phase_delayed_weight,
                    )
                )
            else:
                flat_post_kick_phase_targets = None
                post_kick_phase_multipliers = None
            configured_chunk_size = self.cfg["runner"].get(
                "optimization_chunk_size",
                sample_count,
            )
            optimization_chunk_size = (
                sample_count
                if configured_chunk_size is None
                else int(configured_chunk_size)
            )
            if optimization_chunk_size <= 0:
                raise ValueError("runner.optimization_chunk_size must be positive")
            optimization_chunk_size = min(optimization_chunk_size, sample_count)

            min_entropy = self.cfg["algorithm"]["min_entropy"]
            max_entropy = self.cfg["algorithm"]["max_entropy"]
            if (
                optimization_chunk_size < sample_count
                and min_entropy is not None
                and max_entropy is not None
            ):
                raise ValueError(
                    "optimization_chunk_size does not support the global entropy "
                    "range penalty"
                )

            with torch.no_grad():
                old_actions_log_prob = torch.empty(
                    sample_count,
                    device=self.device,
                )
                old_values_flat = torch.empty(sample_count, device=self.device)
                old_action_mean = torch.empty_like(flat_actions)
                old_action_std = torch.empty_like(flat_actions)
                for start in range(0, sample_count, optimization_chunk_size):
                    end = min(start + optimization_chunk_size, sample_count)
                    batch_slice = slice(start, end)
                    old_dist = self.model.act(flat_obses[batch_slice])
                    old_actions_log_prob[batch_slice] = old_dist.log_prob(
                        flat_actions[batch_slice]
                    ).sum(dim=-1)
                    old_action_mean[batch_slice] = old_dist.loc
                    old_action_std[batch_slice] = old_dist.scale
                    old_values_flat[batch_slice] = self.model.est_value(
                        flat_obses[batch_slice],
                        flat_privileged_obses[batch_slice],
                    )

                old_values = old_values_flat.reshape_as(self.buffer["rewards"])
                old_last_values = self.model.est_value(obs, privileged_obs)
                # Compute returns once using old values (they shouldn't change during mini epochs)
                self.buffer["rewards"][self.buffer["time_outs"]] = old_values[self.buffer["time_outs"]]
                advantages = discount_values(
                    self.buffer["rewards"],
                    self.buffer["dones"] | self.buffer["time_outs"],
                    old_values,
                    old_last_values,
                    self.cfg["algorithm"]["gamma"],
                    self.cfg["algorithm"]["lam"],
                )
                returns = old_values + advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            flat_old_values = old_values.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_advantages = advantages.reshape(-1)

            # Get value clip parameter (default to None for no clipping, for backwards compatibility)
            value_clip_param = self.cfg["algorithm"].get("value_clip_param", None)
            symmetric_coef = float(
                self.cfg["algorithm"].get("symmetric_coef", 0.0)
            )
            if symmetric_coef < 0.0:
                raise ValueError("algorithm.symmetric_coef must be non-negative")
            symmetry_loss_fn = getattr(self.model, "compute_symmetry_loss", None)
            use_symmetry_loss = (
                symmetric_coef > 0.0
                and callable(symmetry_loss_fn)
                and bool(getattr(self.model, "mirror_consistency_enabled", True))
            )

            mean_value_loss = 0
            mean_actor_loss = 0
            mean_bound_loss = 0
            mean_entropy = 0
            mean_symmetry_loss = 0
            mean_symmetry_weight = 0
            mean_post_kick_phase_loss = 0
            for n in range(self.cfg["runner"]["mini_epochs"]):
                self.optimizer.zero_grad()
                epoch_value_loss = 0.0
                epoch_actor_loss = 0.0
                epoch_bound_loss = 0.0
                epoch_entropy = 0.0
                epoch_symmetry_loss = 0.0
                epoch_symmetry_weight = 0.0
                epoch_post_kick_phase_loss = 0.0
                for start in range(0, sample_count, optimization_chunk_size):
                    end = min(start + optimization_chunk_size, sample_count)
                    batch_slice = slice(start, end)
                    batch_weight = float(end - start) / float(sample_count)
                    values = self.model.est_value(
                        flat_obses[batch_slice],
                        flat_privileged_obses[batch_slice],
                    )

                    if value_clip_param is not None:
                        values_clipped = flat_old_values[batch_slice] + torch.clamp(
                            values - flat_old_values[batch_slice],
                            -value_clip_param,
                            value_clip_param,
                        )
                        value_loss_unclipped = (
                            values - flat_returns[batch_slice]
                        ).pow(2)
                        value_loss_clipped = (
                            values_clipped - flat_returns[batch_slice]
                        ).pow(2)
                        value_loss = 0.5 * torch.max(
                            value_loss_unclipped,
                            value_loss_clipped,
                        ).mean()
                    else:
                        value_loss = F.mse_loss(
                            values,
                            flat_returns[batch_slice],
                        )

                    dist = self.model.act(flat_obses[batch_slice])
                    actions_log_prob = dist.log_prob(
                        flat_actions[batch_slice]
                    ).sum(dim=-1)
                    actor_loss = surrogate_loss(
                        old_actions_log_prob[batch_slice],
                        actions_log_prob,
                        flat_advantages[batch_slice],
                    )
                    bound_loss = (
                        torch.clip(dist.loc - 1.0, min=0.0).square().mean()
                        + torch.clip(dist.loc + 1.0, max=0.0).square().mean()
                    )
                    entropy = dist.entropy().sum(dim=-1)
                    entropy_mean = entropy.mean()
                    symmetry_loss = dist.loc.new_zeros(())
                    symmetry_weight = dist.loc.new_zeros(())
                    if use_symmetry_loss:
                        symmetry_loss, symmetry_weight = symmetry_loss_fn(
                            flat_obses[batch_slice],
                            dist.loc,
                        )
                    post_kick_phase_loss = dist.loc.new_zeros(())
                    if self.post_kick_phase_auxiliary_enabled:
                        post_kick_phase_logits = (
                            self.model.post_kick_phase_logit(
                                flat_obses[batch_slice]
                            )
                        )
                        post_kick_phase_loss = (
                            weighted_phase_binary_cross_entropy(
                                post_kick_phase_logits,
                                flat_post_kick_phase_targets[batch_slice],
                                post_kick_phase_multipliers[batch_slice],
                            )
                        )

                    if min_entropy is not None and max_entropy is not None:
                        loss_entropy = torch.mean(
                            (
                                torch.clamp(
                                    entropy_mean,
                                    min=min_entropy,
                                    max=max_entropy,
                                )
                                - entropy_mean
                            )
                            ** 2
                        )
                    else:
                        loss_entropy = 0.0
                    loss = (
                        value_loss
                        + actor_loss
                        + self.cfg["algorithm"]["bound_coef"] * bound_loss
                        + self.cfg["algorithm"]["entropy_coef"] * entropy_mean
                        + 0.01 * loss_entropy
                        + symmetric_coef * symmetry_loss
                        + self.post_kick_phase_loss_coefficient
                        * post_kick_phase_loss
                    )
                    (loss * batch_weight).backward()
                    epoch_value_loss += value_loss.item() * batch_weight
                    epoch_actor_loss += actor_loss.item() * batch_weight
                    epoch_bound_loss += bound_loss.item() * batch_weight
                    epoch_entropy += entropy_mean.item() * batch_weight
                    epoch_symmetry_loss += symmetry_loss.item() * batch_weight
                    epoch_symmetry_weight += symmetry_weight.item() * batch_weight
                    epoch_post_kick_phase_loss += (
                        post_kick_phase_loss.item() * batch_weight
                    )

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                mean_value_loss += epoch_value_loss
                mean_actor_loss += epoch_actor_loss
                mean_bound_loss += epoch_bound_loss
                mean_entropy += epoch_entropy
                mean_symmetry_loss += epoch_symmetry_loss
                mean_symmetry_weight += epoch_symmetry_weight
                mean_post_kick_phase_loss += epoch_post_kick_phase_loss

            # Calculate KL divergence after all mini epochs (between old and final policy)
            with torch.no_grad():
                kl_sum = torch.zeros((), device=self.device)
                post_kick_phase_probability_sum = torch.zeros(
                    (),
                    device=self.device,
                )
                post_kick_phase_ready_probability_sum = torch.zeros(
                    (),
                    device=self.device,
                )
                post_kick_phase_ready_count = torch.zeros(
                    (),
                    device=self.device,
                )
                for start in range(0, sample_count, optimization_chunk_size):
                    end = min(start + optimization_chunk_size, sample_count)
                    batch_slice = slice(start, end)
                    final_dist = self.model.act(flat_obses[batch_slice])
                    kl = torch.sum(
                        torch.log(
                            final_dist.scale / old_action_std[batch_slice]
                        )
                        + 0.5
                        * (
                            torch.square(old_action_std[batch_slice])
                            + torch.square(
                                final_dist.loc - old_action_mean[batch_slice]
                            )
                        )
                        / torch.square(final_dist.scale)
                        - 0.5,
                        dim=-1,
                    )
                    kl_sum += kl.sum()
                    if self.post_kick_phase_auxiliary_enabled:
                        phase_probability = torch.sigmoid(
                            self.model.post_kick_phase_logit(
                                flat_obses[batch_slice]
                            )
                        )
                        post_kick_phase_probability_sum += (
                            phase_probability.sum()
                        )
                        ready = (
                            flat_post_kick_phase_targets[batch_slice] >= 0.5
                        )
                        post_kick_phase_ready_probability_sum += (
                            phase_probability[ready].sum()
                        )
                        post_kick_phase_ready_count += ready.sum()
                kl_mean = kl_sum / float(sample_count)
                post_kick_phase_probability_mean = (
                    post_kick_phase_probability_sum / float(sample_count)
                )
                post_kick_phase_ready_probability_mean = (
                    post_kick_phase_ready_probability_sum
                    / torch.clamp(post_kick_phase_ready_count, min=1.0)
                )

                # Adapt learning rate based on KL divergence
                if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

            mean_value_loss /= self.cfg["runner"]["mini_epochs"]
            mean_actor_loss /= self.cfg["runner"]["mini_epochs"]
            mean_bound_loss /= self.cfg["runner"]["mini_epochs"]
            mean_entropy /= self.cfg["runner"]["mini_epochs"]
            mean_symmetry_loss /= self.cfg["runner"]["mini_epochs"]
            mean_symmetry_weight /= self.cfg["runner"]["mini_epochs"]
            mean_post_kick_phase_loss /= self.cfg["runner"]["mini_epochs"]
            stage_plateau_changed = self._update_stage_plateau_curriculum(
                rollout_reward_mean,
                rollout_done_rate,
                it,
                per_env_step,
                total_env_steps,
            )
            if stage_plateau_changed:
                if self.adaptive_speed_enabled:
                    self._set_adaptive_speed_mode("slow", it, reason="stage_changed")
                if self.adaptive_kick_enabled:
                    self._set_adaptive_kick_mode("normal", it, reason="stage_changed")
                obs, infos = self.env.reset()
                obs = obs.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
            else:
                adaptive_speed_changed = self._update_adaptive_speed_curriculum(rollout_reward_mean, it)
                if adaptive_speed_changed:
                    if hasattr(self.env, "cmd_resample_time") and hasattr(self.env, "episode_length_buf"):
                        self.env.cmd_resample_time[:] = self.env.episode_length_buf
                    self.env._resample_commands()
                    self.env._compute_observations()
                    obs = self.env.obs_buf.to(self.device)
                    privileged_obs = self.env.extras["privileged_obs"].to(self.device)
                adaptive_kick_changed = self._update_adaptive_kick_curriculum(rollout_reward_mean, rollout_done_rate, it)
                if adaptive_kick_changed:
                    obs, infos = self.env.reset()
                    obs = obs.to(self.device)
                    privileged_obs = infos["privileged_obs"].to(self.device)
            stage_name = getattr(self.env, "active_stage", {}).get("name", "none")
            speed_mode = "off"
            if self.adaptive_speed_enabled:
                speed_mode = self.adaptive_speed_mode if self._adaptive_speed_stage_enabled() else "inactive"
            kick_mode = "off"
            if self.adaptive_kick_enabled:
                kick_mode = self.adaptive_kick_mode if self._adaptive_kick_stage_enabled() else "inactive"
            env_resets = getattr(self.env, "env_resets", 0)
            env_successes = getattr(self.env, "env_successes", 0)
            env_falling = getattr(self.env, "env_falling", 0)
            stats = {
                    "value_loss": mean_value_loss,
                    "actor_loss": mean_actor_loss,
                    "bound_loss": mean_bound_loss,
                    "entropy": mean_entropy,
                    "kl_mean": kl_mean,
                    "lr": self.learning_rate,
                    "rollout/reward_mean": rollout_reward_mean,
                    "rollout/done_rate": rollout_done_rate,
                    "rollout/env_resets": float(env_resets),
                    "rollout/env_successes": float(env_successes),
                    "rollout/env_falling": float(env_falling),
                    "curriculum/mean_lin_vel_level": self.env.mean_lin_vel_level,
                    "curriculum/mean_ang_vel_level": self.env.mean_ang_vel_level,
                    "curriculum/max_lin_vel_level": self.env.max_lin_vel_level,
                    "curriculum/max_ang_vel_level": self.env.max_ang_vel_level,
                    "stage_curriculum/stage_index": getattr(self.env, "active_stage_idx", -1),
                    "stage_curriculum/per_env_step": per_env_step,
                    "stage_curriculum/total_env_steps": total_env_steps,
            }
            stats.update(self._adaptive_speed_statistics())
            stats.update(self._adaptive_kick_statistics())
            stats.update(self._stage_plateau_statistics())
            if use_symmetry_loss:
                stats.update(
                    {
                        "mirror_consistency/loss": mean_symmetry_loss,
                        "mirror_consistency/weight_mean": mean_symmetry_weight,
                        "mirror_consistency/coefficient": symmetric_coef,
                    }
                )
            if self.post_kick_phase_auxiliary_enabled:
                stats.update(
                    {
                        "post_kick_phase/loss": mean_post_kick_phase_loss,
                        "post_kick_phase/coefficient": (
                            self.post_kick_phase_loss_coefficient
                        ),
                        "post_kick_phase/target_rate": (
                            flat_post_kick_phase_targets.mean()
                        ),
                        "post_kick_phase/probability_mean": (
                            post_kick_phase_probability_mean
                        ),
                        "post_kick_phase/ready_probability_mean": (
                            post_kick_phase_ready_probability_mean
                        ),
                    }
                )
            self.recorder.record_statistics(stats, it)

            print(
                "epoch: {}/{} | stage={} | speed={} | kick={} | reward_mean={:.4f} | done_rate={:.4f} | "
                "value_loss={:.4f} | actor_loss={:.4f} | entropy={:.4f} | kl={:.5f} | "
                "resets={} | falling={}".format(
                    it + 1,
                    self.cfg["basic"]["max_iterations"],
                    stage_name,
                    speed_mode,
                    kick_mode,
                    rollout_reward_mean,
                    rollout_done_rate,
                    mean_value_loss,
                    mean_actor_loss,
                    float(mean_entropy),
                    float(kl_mean),
                    env_resets,
                    env_falling,
                )
            )

    def save_interrupt_checkpoint(self):
        if not hasattr(self, "recorder"):
            return None
        iteration = getattr(self, "current_iteration", -1) + 1
        checkpoint_id = f"interrupt_{iteration}"
        self.recorder.save(
            self._checkpoint_payload(iteration),
            checkpoint_id,
        )
        return os.path.join(self.recorder.model_dir, f"model_{checkpoint_id}.pth")

    def play(self):
        # Check if we're in record-and-exit mode (for separate process video recording)
        if self.args.record_video_mode:
            self._play_record_and_exit()
            return
        
        # Normal play mode (for manual testing)
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        if self.cfg["viewer"]["record_video"]:
            os.makedirs("videos", exist_ok=True)
            name = time.strftime("%Y-%m-%d-%H-%M-%S.mp4", time.localtime())
            record_time = self.cfg["viewer"]["record_interval"]
        while True:
            with torch.no_grad():
                dist = self.model.act(obs)
                act = dist.loc
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
            if self.cfg["viewer"]["record_video"]:
                record_time -= self.env.dt
                if record_time < 0:
                    record_time += self.cfg["viewer"]["record_interval"]
                    self.interrupt = False
                    signal.signal(signal.SIGINT, self.interrupt_handler)
                    with imageio.get_writer(os.path.join("videos", name), fps=int(1.0 / self.env.dt)) as self.writer:
                        for frame in self.env.camera_frames:
                            self.writer.append_data(frame)
                    if self.interrupt:
                        raise KeyboardInterrupt
                    signal.signal(signal.SIGINT, signal.default_int_handler)
    
    def _play_record_and_exit(self):
        """Record video for a specified duration and save to file, then exit.
        This is used by the separate process spawned during training.
        The main process will upload the video to wandb after this process finishes."""
        # Enable video recording
        self.cfg["viewer"]["record_video"] = True
        
        # Get video duration
        video_duration = self.args.video_duration
        if video_duration is None:
            video_duration = self.cfg["runner"].get("log_video_duration", 10.0)
        
        # Get output path for video file
        video_output_path = self.args.video_output_path
        if video_output_path is None:
            print("Error: video_output_path not provided")
            return
        
        # Calculate number of frames to capture
        num_frames = int(video_duration / self.env.dt)
        
        # Clear existing frames
        if hasattr(self.env, 'camera_frames'):
            self.env.camera_frames = []
        
        # Initialize environment
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        
        # Ensure camera is initialized
        if self.cfg["viewer"]["record_video"]:
            self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
            self.env.render()
        
        # Capture frames
        frames_captured = 0
        total_reward = []
        separated_reward = {}
        
        print(f"Recording video for {video_duration} seconds ({num_frames} frames)...")
        
        while frames_captured < num_frames:
            # Step the environment with current policy
            with torch.no_grad():
                dist = self.model.act(obs)
                act = dist.sample()
            
            obs, rew, done, infos = self.env.step(act)
            obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
            
            # Store rewards for the first environment only
            total_reward.append(rew[0].item())
            for key, value in infos["rew_terms"].items():
                if key not in separated_reward:
                    separated_reward[key] = []
                separated_reward[key].append(value[0].item())
            
            # Render to capture frame
            if self.cfg["viewer"]["record_video"]:
                self.env.render()
            
            frames_captured += 1
            
            # Reset if episode done
            if done[0]:
                reset_obs, reset_infos = self.env.reset()
                obs = reset_obs.to(self.device)
        
        # Save video to file
        if hasattr(self.env, 'camera_frames') and len(self.env.camera_frames) > 0:
            import numpy as np
            import imageio
            
            # Convert frames to RGB format
            video_frames = []
            for frame in self.env.camera_frames:
                if len(frame.shape) == 3:
                    if frame.shape[2] == 4:
                        # BGRA to RGB
                        rgb_frame = frame[:, :, [2, 1, 0]]
                    elif frame.shape[2] == 3:
                        rgb_frame = frame
                    else:
                        rgb_frame = frame[:, :, :3]
                else:
                    continue
                
                # Ensure uint8 format
                if rgb_frame.dtype != np.uint8:
                    if rgb_frame.max() <= 1.0:
                        rgb_frame = (rgb_frame * 255).astype(np.uint8)
                    else:
                        rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)
                
                video_frames.append(rgb_frame)
            
            # Save video file
            os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
            fps = int(1.0 / self.env.dt)
            imageio.mimwrite(video_output_path, video_frames, fps=fps, codec='libx264')
            print(f"Video saved to {video_output_path}")
        else:
            print("Warning: No frames captured")
        
        # Save reward data to JSON file
        if self.args.rewards_output_path and len(total_reward) > 0:
            import json
            os.makedirs(os.path.dirname(self.args.rewards_output_path), exist_ok=True)
            reward_data = {
                "total_reward": total_reward,
                "separated_reward": separated_reward
            }
            with open(self.args.rewards_output_path, 'w') as f:
                json.dump(reward_data, f)
            print(f"Reward data saved to {self.args.rewards_output_path}")
        
        # Clean up
        if hasattr(self.env, 'camera_frames'):
            self.env.camera_frames = []
        
        print("Video recording complete. Exiting...")

    def interrupt_handler(self, signal, frame):
        print("\nInterrupt received, waiting for video to finish...")
        self.interrupt = True

    def _capture_training_video(self, duration, it, obs, privileged_obs):
        """Capture video frames during training for wandb logging.
        
        Args:
            duration: Duration of video in seconds
            it: Current iteration step
            obs: Current observations
            privileged_obs: Current privileged observations
            
        Returns:
            Updated obs and privileged_obs after video capture
        """
        # Clear existing frames and ensure camera is initialized
        if hasattr(self.env, 'camera_frames'):
            self.env.camera_frames = []
        
        # Ensure camera is initialized by calling render once before capturing
        # This ensures the camera exists and root_states are available
        if self.cfg["viewer"]["record_video"]:
            # Refresh root states to ensure camera position is correct
            self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
            self.env.render()
        
        # Calculate number of frames to capture
        num_frames = int(duration / self.env.dt)
        
        # Capture frames by running the environment
        frames_captured = 0

        total_reward = []
        seperated_reward = {}
        
        while frames_captured < num_frames:
            # Step the environment with current policy first
            with torch.no_grad():
                dist = self.model.act(obs)
                act = dist.sample()
            
            obs, rew, done, infos = self.env.step(act)
            obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
            privileged_obs = infos["privileged_obs"].to(self.device)

            # Store rewards for the first environment only
            total_reward.append(rew[0].item())
            for key, value in infos["rew_terms"].items():
                if key not in seperated_reward:
                    seperated_reward[key] = []
                seperated_reward[key].append(value[0].item())
            
            # step() already calls render() internally which captures frames
            # But we ensure render is called to capture the frame
            # The render() in step() should have already captured the frame,
            # but we call it again to be safe (it's idempotent for frame capture)
            if self.cfg["viewer"]["record_video"]:
                self.env.render()
            
            frames_captured += 1
            
            # Reset if episode done
            if done[0]:
                reset_obs, reset_infos = self.env.reset()
                obs = reset_obs.to(self.device)
                privileged_obs = reset_infos["privileged_obs"].to(self.device)
        
        # Log video to wandb
        if hasattr(self.env, 'camera_frames') and len(self.env.camera_frames) > 0:
            self.recorder.log_video(self.env.camera_frames, it, self.env.dt)
            # Clear frames to free memory
            self.env.camera_frames = []
        
        # Log video rewards
        self.recorder.log_video_rewards(total_reward, seperated_reward, it)
        
        return obs, privileged_obs

    def _get_robot_type(self, task_name):
        """Determine robot type from task name."""
        # Check if task name starts with K1 or T1
        if task_name.startswith("K1"):
            return "K1"
        elif task_name.startswith("T1"):
            return "T1"
        else:
            # Default fallback - could be extended for other robot types
            return "Unknown"
    
    def _upload_video_to_wandb(self, video_path, iteration):
        """Upload a video file to wandb.
        
        Args:
            video_path: Path to the video file
            iteration: Iteration number for logging
        """
        if not self.cfg["runner"].get("use_wandb", False):
            return
        
        import wandb
        if wandb.run is None:
            print("Warning: wandb run not initialized, cannot upload video")
            return
        
        try:
            # Use custom step metric for video logs to avoid step conflicts
            # See: https://docs.wandb.ai/models/track/log/customize-logging-axes
            # The iteration parameter is already it+1 (passed from spawn function)
            wandb.log({
                "video/iteration": iteration,  # Custom x-axis metric
                "video/training": wandb.Video(video_path, format="mp4")
            }, commit=True)
            print(f"Video uploaded to wandb at iteration {iteration}")
        except Exception as e:
            print(f"Error uploading video to wandb: {e}")
            import traceback
            traceback.print_exc()
    
    def _upload_rewards_to_wandb(self, rewards_path, iteration):
        """Load reward data from file and upload plots to wandb.
        
        Args:
            rewards_path: Path to the JSON file containing reward data
            iteration: Iteration number for logging
        """
        if not self.cfg["runner"].get("use_wandb", False):
            return
        
        import wandb
        import json
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if wandb.run is None:
            print("Warning: wandb run not initialized, cannot upload rewards")
            return
        
        try:
            # Load reward data
            with open(rewards_path, 'r') as f:
                reward_data = json.load(f)
            
            total_reward = reward_data["total_reward"]
            separated_reward = reward_data["separated_reward"]
            
            if len(total_reward) == 0:
                print("Warning: No reward data to log")
                return
            
            # Use custom step metric for video logs to avoid step conflicts
            # See: https://docs.wandb.ai/models/track/log/customize-logging-axes
            # The iteration parameter is already it+1 (passed from spawn function)
            
            # Convert to numpy arrays
            total_reward_np = np.array(total_reward)
            timesteps = np.arange(len(total_reward_np))
            
            # Calculate statistics
            mean_total_reward = float(np.mean(total_reward_np))
            sum_total_reward = float(np.sum(total_reward_np))
            
            # Log summary statistics
            self.recorder.writer.add_scalar("video/mean_reward", mean_total_reward, iteration)
            self.recorder.writer.add_scalar("video/sum_reward", sum_total_reward, iteration)
            
            # Prepare log dictionary with custom step metric
            log_dict = {
                "video/iteration": iteration,  # Custom x-axis metric
                "video/mean_reward": mean_total_reward,
                "video/sum_reward": sum_total_reward,
            }
            
            # Create and log figure for total reward
            fig_total = plt.figure(figsize=(12, 4))
            plt.plot(timesteps, total_reward_np, linewidth=2, color='blue')
            plt.title(f'Total Reward (Mean: {mean_total_reward:.3f}, Sum: {sum_total_reward:.3f})', fontsize=12, fontweight='bold')
            plt.xlabel('Frame')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.tight_layout()
            log_dict["video_plots/total_reward_trajectory"] = wandb.Image(fig_total)
            plt.close(fig_total)
            
            # Create and log figure for each reward term
            for key, values in separated_reward.items():
                if len(values) == 0:
                    continue
                
                values_np = np.array(values)
                mean_value = float(np.mean(values_np))
                sum_value = float(np.sum(values_np))
                
                # Create figure for this reward term
                fig_term = plt.figure(figsize=(12, 4))
                plt.plot(timesteps, values_np, linewidth=2)
                plt.title(f'{key} (Mean: {mean_value:.3f}, Sum: {sum_value:.3f})', fontsize=12)
                plt.xlabel('Frame')
                plt.ylabel('Reward')
                plt.grid(True, alpha=0.3)
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                plt.tight_layout()
                log_dict[f"video_plots/reward_trajectories/{key}"] = wandb.Image(fig_term)
                plt.close(fig_term)
            
            # Log everything at once with custom step metric
            wandb.log(log_dict, commit=True)
            print(f"Reward plots uploaded to wandb at iteration {iteration}")
        except Exception as e:
            print(f"Error uploading rewards to wandb: {e}")
            import traceback
            traceback.print_exc()
    
    def _spawn_video_recording_process(self, checkpoint_path, iteration, video_duration):
        """Spawn a separate process to record video and save to file.
        The main process will upload the video to wandb after the process finishes.
        
        Args:
            checkpoint_path: Path to the checkpoint file to load
            iteration: Current iteration number for wandb logging
            video_duration: Duration of video to record in seconds
        """
        if not self.cfg["runner"].get("use_wandb", False):
            return
        
        # Create video output path
        video_dir = os.path.join(self.recorder.dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        video_output_path = os.path.join(video_dir, f"video_iter_{iteration + 1}.mp4")
        rewards_output_path = os.path.join(video_dir, f"rewards_iter_{iteration + 1}.json")
        
        # Build command to run play.py in record mode
        play_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "play.py")
        cmd = [
            sys.executable,
            play_script,
            "--task", self.cfg["basic"]["task"],
            "--checkpoint", checkpoint_path,
            "--record_video_mode",
            "--video_duration", str(video_duration),
            "--video_iteration", str(iteration + 1),
            "--video_output_path", video_output_path,
            "--rewards_output_path", rewards_output_path,
        ]
        
        # Add other relevant arguments if they were provided
        if self.args.num_envs is not None:
            cmd.extend(["--num_envs", str(self.args.num_envs)])
        # Use headless from config for video recording (usually better for separate process)
        if self.cfg["basic"].get("headless") is not None:
            cmd.extend(["--headless", str(self.cfg["basic"]["headless"])])
        elif self.args.headless is not None:
            cmd.extend(["--headless", str(self.args.headless)])
        if self.args.sim_device is not None:
            cmd.extend(["--sim_device", self.args.sim_device])
        if self.args.rl_device is not None:
            cmd.extend(["--rl_device", self.args.rl_device])
        if self.args.seed is not None:
            cmd.extend(["--seed", str(self.args.seed)])
        if self.args.model is not None:
            cmd.extend(["--model", self.args.model])
        
        print(f"Spawning video recording process for iteration {iteration + 1}...")
        print(f"Command: {' '.join(cmd)}")
        
        # Verify checkpoint file exists
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file {checkpoint_path} does not exist")
            return
        
        # Spawn process with environment variables
        env = os.environ.copy()
        # Ensure PYTHONPATH is set correctly
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = os.path.dirname(os.path.dirname(__file__))
        else:
            env['PYTHONPATH'] = os.path.dirname(os.path.dirname(__file__)) + os.pathsep + env['PYTHONPATH']
        
        # Create log files for the subprocess
        log_dir = os.path.join(self.recorder.dir, "video_logs")
        os.makedirs(log_dir, exist_ok=True)
        stdout_file = os.path.join(log_dir, f"video_iter_{iteration + 1}_stdout.log")
        stderr_file = os.path.join(log_dir, f"video_iter_{iteration + 1}_stderr.log")
        
        try:
            with open(stdout_file, 'w') as fout, open(stderr_file, 'w') as ferr:
                process = subprocess.Popen(
                    cmd,
                    stdout=fout,
                    stderr=ferr,
                    env=env,
                )
            
            print(f"Video recording process started (PID: {process.pid})")
            print(f"  Logs: {stdout_file} and {stderr_file}")
            print(f"  Waiting for video recording to complete...")
            
            # Wait for the process to complete
            return_code = process.wait()
            
            if return_code == 0:
                print(f"Video recording completed successfully for iteration {iteration + 1}")
                # Upload video and rewards to wandb
                if os.path.exists(video_output_path):
                    self._upload_video_to_wandb(video_output_path, iteration + 1)
                else:
                    print(f"Warning: Video file not found at {video_output_path}")
                
                # Load and log reward data
                if os.path.exists(rewards_output_path):
                    self._upload_rewards_to_wandb(rewards_output_path, iteration + 1)
                else:
                    print(f"Warning: Reward data file not found at {rewards_output_path}")
            else:
                print(f"Warning: Video recording process exited with code {return_code}")
                print(f"  Check logs: {stdout_file} and {stderr_file}")
                
        except Exception as e:
            print(f"Error spawning video recording process: {e}")
            import traceback
            traceback.print_exc()
