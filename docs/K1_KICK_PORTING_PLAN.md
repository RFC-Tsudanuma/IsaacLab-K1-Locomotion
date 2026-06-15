# K1 Kick Porting Plan

## 1. Current state

### Repositories / branches
- `booster_kick`: branch `aiba_kick`, clean working tree
- `IsaacLab-K1-Locomotion`: branch `feat/aiba_kick`, dirty working tree before this task

### Source tree
- Source task root: `booster_kick/kick_env/kick_env/source/kick_env/kick_env/tasks/manager_based/kick_env`
- Main files:
  - `__init__.py`
  - `kick_env_env_cfg.py`
  - `agents/rsl_rl_ppo_cfg.py`
  - `mdp/observations.py`
  - `mdp/rewards.py`
  - `mdp/state.py`
  - `mdp/setting.py`

### Target tree
- Target task root: `IsaacLab-K1-Locomotion/source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion`
- Main files:
  - `__init__.py`
  - `velocity_env_cfg.py`
  - `rough_env_cfg.py`
  - `flat_env_cfg.py`
  - `agents/rsl_rl_ppo_cfg.py`
  - `mdp/{__init__,commands,observations,rewards,curriculums,data_logger}.py`

## 2. Booster T1 kick task structure

## Task registration
- `kick_env/tasks/__init__.py` and `kick_env/tasks/manager_based/__init__.py` use `import_packages(...)`.
- Gym registration is in `kick_env/tasks/manager_based/kick_env/__init__.py`.
- The kick task registers both SKRL and RSL-RL entry points with duplicate IDs:
  - `Booster-Kick`
  - `Booster-Kick-Play`

## Environment/config structure
- `kick_env_env_cfg.py` defines everything in one file:
  - `BOOSTER_T1_CFG`
  - `KickEnvSceneCfg`
  - `CommandsCfg`
  - `ActionsCfg`
  - `ObservationsCfg`
  - `EventCfg`
  - `RewardsCfg`
  - `CurriculumCfg`
  - `TerminationsCfg`
  - `KickEnvEnvCfg`
  - `KickEnvEnvCfg_Play`

## Source env_cfg summary
- `env_cfg`: manager-based RL env, 4096 envs, flat ground, 20 s episode, dt 0.005, decimation 4
- `scene_cfg`: ground plane + `soccer_ball` (`RigidObjectCfg`) + T1 robot + contact sensors + IMU
- `observations`:
  - target pose command
  - joint positions / velocities
  - IMU orientation / angular velocity
  - previous actions
  - ball relative position
- `events`:
  - startup randomization for rigid-body material and base mass
  - reset robot joints/base
  - reset ball root state
- `commands`:
  - `target_pos`: `UniformPose2dCommandCfg`
  - `base_velocity`: fixed forward command range
- `curriculum`:
  - modifies reward weights for ball tracking, ball distance, touch reward
- `terminations`:
  - timeout
  - root height below threshold
  - bad orientation

## Source MDP / reward summary
- `mdp/observations.py`
  - `ball_pos_rel`: ball XY position relative to robot root XY
- `mdp/rewards.py`
  - `ball_command_tracking`
  - `ball_distance`
  - `reguralize_orientaion`
  - `touch_ball`
  - `penalize_jump`
  - `kick_half_reward`
  - `reward_return_to_initial`
- Ball generation:
  - `KickEnvSceneCfg.soccer_ball` as `RigidObjectCfg` sphere, radius 0.11, mass 0.45
- Ball reset:
  - `EventCfg.reset_ball` via `mdp.reset_root_state_uniform(..., asset_cfg=SceneEntityCfg("soccer_ball"))`
- Kick success:
  - no explicit success termination; success is approximated by ball-touch/kick rewards
- PPO:
  - `agents/rsl_rl_ppo_cfg.py::PPOKickCfg`
  - 32 steps/env, 3000 iterations, experiment name `kick`
  - actor/critic hidden dims `[512, 256, 128]`

## 3. Booster K1 locomotion structure

## Task registration
- `isaaclab_k1_locomotion/tasks/__init__.py` imports `manager_based`
- `isaaclab_k1_locomotion/tasks/manager_based/__init__.py` imports `locomotion`
- Gym registration is currently centralized in `tasks/manager_based/locomotion/__init__.py`
- Registered tasks today:
  - `Isaac-Velocity-Rough-K1-v0`
  - `Isaac-Velocity-Rough-K1-Play-v0`
  - `Isaac-Velocity-Flat-K1-v0`
  - `Isaac-Velocity-Flat-K1-Play-v0`

## Target config/layout conventions
- Base/common manager layout lives in `velocity_env_cfg.py`
- K1-specific asset, observations, rewards, terminations live in `rough_env_cfg.py`
- Flat specialization lives in `flat_env_cfg.py`
- PPO config lives in `locomotion/agents/rsl_rl_ppo_cfg.py`
- Custom MDP helpers live in `locomotion/mdp/*.py`

## Target asset / manager summary
- Robot asset:
  - `rough_env_cfg.py::K1_LOCOMOTION_CFG`
  - URDF-based asset in `assets_soccer/booster_robotics_robots/K1/K1_locomotion.urdf`
- Command manager:
  - `velocity_env_cfg.py::CommandsCfg`
- Observation managers:
  - base in `velocity_env_cfg.py::ObservationsCfg`
  - K1 overrides in `rough_env_cfg.py::{K1PolicyCfg,K1CriticCfg,K1ObservationsCfg}`
- Reward managers:
  - base in `velocity_env_cfg.py::RewardsCfg`
  - K1 overrides in `rough_env_cfg.py::K1Rewards`
- Event manager:
  - base in `velocity_env_cfg.py::EventCfg`
- Curriculum manager:
  - base in `velocity_env_cfg.py::CurriculumCfg`

## 4. Train/play to task creation path

## `train.py`
1. `scripts/rsl_rl/train.py` imports `isaaclab_k1_locomotion.tasks`
2. `@hydra_task_config(args_cli.task, args_cli.agent)` decorates `main`
3. `isaaclab_tasks.utils.hydra.register_task_to_hydra()` resolves:
   - `env_cfg_entry_point`
   - `rsl_rl_cfg_entry_point`
4. `isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry()` reads the gym registry
5. `gym.make(args_cli.task, cfg=env_cfg, ...)` instantiates `isaaclab.envs:ManagerBasedRLEnv`

## `play.py`
1. `scripts/rsl_rl/play.py` imports `isaaclab_k1_locomotion.tasks`
2. Same Hydra path resolves env/agent config from gym registry
3. `gym.make(args_cli.task, cfg=env_cfg, ...)` creates the manager-based env
4. Loaded checkpoint is wrapped with `RslRlVecEnvWrapper` for inference

## 5. T1 vs K1 differences to account for

| Area | T1 source | K1 target |
|---|---|---|
| Robot asset | USD (`t1.usd`) | URDF (`K1_locomotion.urdf`) |
| Base link | `Trunk` used in sensors/randomization | `Trunk` used in current K1 locomotion |
| Foot links | `left_foot_link`, `right_foot_link` | same names already used in K1 locomotion rewards |
| Extra contact links | T1 kick also uses `Shank_Left`, `Shank_Right` | K1 locomotion already references `.*_Shank`; exact K1 body names must be verified at runtime |
| Joint set | T1 action includes hips/knees/ankles + waist + arms | K1 locomotion uses `JOINT_NAMES_K1` legs only |
| Observation style | single policy group | K1 locomotion uses policy + critic groups |
| Command style | target pose + base velocity | K1 kick should stay close to K1 locomotion conventions and add kick-specific observations |

## 6. Porting approach

## Directory layout to add
- `source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/kick/`
  - `__init__.py`
  - `kick_env_cfg.py`
  - `agents/__init__.py`
  - `agents/rsl_rl_ppo_cfg.py`
  - `mdp/__init__.py`
  - `mdp/observations.py`
  - `mdp/rewards.py`
  - `mdp/events.py`

## Design choices
- Do **not** edit existing locomotion env configs beyond importing/registering the new task package.
- Base the new task on K1 flat locomotion conventions, not T1’s monolithic file layout.
- Keep manager-based config style:
  - configclass/dataclass patterns
  - import ordering
  - gym registration style
  - RSL-RL entry-point conventions
- Reuse `K1_LOCOMOTION_CFG` and `JOINT_NAMES_K1`.
- Add a flat ground + rigid ball scene.
- Add K1-foot contact sensors filtered to the ball.
- Implement kick-specific observations:
  - ball position
  - ball velocity
  - ball relative position
  - goal direction
- Implement minimum required rewards:
  - approach ball
  - move toward ball
  - ball contact
  - increase ball speed
  - move ball toward goal
  - kick success
  - fall penalty
- Implement terminations:
  - timeout
  - fall
  - ball travel distance threshold

## 7. Validation plan
- import check for the new package
- gym registry check for `Isaac-K1-Kick-v0`
- `python scripts/rsl_rl/train.py --task Isaac-K1-Kick-v0 --num_envs ...`
- `python scripts/rsl_rl/play.py --task Isaac-K1-Kick-v0 ...`
- confirm scene creation
- confirm ball spawn
- confirm GUI startup path

## 8. Expected deliverables
- `docs/K1_KICK_PORTING_PLAN.md`
- new K1 kick task package
- `docs/K1_KICK_PORTING_REPORT.md`
- runnable task names:
  - `Isaac-K1-Kick-v0`
  - `Isaac-K1-Kick-Play-v0`
