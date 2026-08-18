import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ENV_CFG_PATH = (
    Path(__file__).resolve().parents[2]
    / "isaaclab_k1_locomotion"
    / "tasks"
    / "manager_based"
    / "walk_kick_likelihood"
    / "walk_kick_likelihood_env_cfg.py"
)
PACKAGE_NAME = "_walk_kick_likelihood_env_cfg_test"


class _Cfg:
    def __init__(self, *args, **kwargs):
        self.args = args
        vars(self).update(kwargs)


class _SceneEntityCfg(_Cfg):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.name = name
        self.body_ids = [0]


class _ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _module(name, **attributes):
    module = types.ModuleType(name)
    vars(module).update(attributes)
    return module


def _noop(*args, **kwargs):
    del args, kwargs


def _load_env_cfg_module():
    stubs = {}
    for name in (
        PACKAGE_NAME,
        f"{PACKAGE_NAME}.tasks",
        f"{PACKAGE_NAME}.tasks.locomotion",
        f"{PACKAGE_NAME}.tasks.walk_kick",
        f"{PACKAGE_NAME}.tasks.walk_kick_likelihood",
        "isaaclab",
        "isaaclab_tasks",
        "isaaclab_tasks.manager_based",
        "isaaclab_tasks.manager_based.locomotion",
        "isaaclab_tasks.manager_based.locomotion.velocity",
    ):
        stubs[name] = _package(name)

    stubs["isaaclab.managers"] = _module(
        "isaaclab.managers",
        ManagerTermBase=_ManagerTermBase,
        EventTermCfg=_Cfg,
        ObservationGroupCfg=_Cfg,
        ObservationTermCfg=_Cfg,
        SceneEntityCfg=_SceneEntityCfg,
    )
    stubs["isaaclab.utils"] = _module(
        "isaaclab.utils",
        configclass=lambda cls: cls,
    )
    stubs["isaaclab.utils.noise"] = _module(
        "isaaclab.utils.noise",
        GaussianNoiseCfg=_Cfg,
    )
    stubs["isaaclab_tasks.manager_based.locomotion.velocity.mdp"] = _module(
        "isaaclab_tasks.manager_based.locomotion.velocity.mdp",
        projected_gravity=_noop,
        generated_commands=_noop,
        joint_pos_rel=_noop,
        joint_vel_rel=_noop,
        last_action=_noop,
        base_lin_vel=_noop,
    )
    stubs[f"{PACKAGE_NAME}.tasks.locomotion.rough_env_cfg"] = _module(
        f"{PACKAGE_NAME}.tasks.locomotion.rough_env_cfg",
        _COMMAND_THRESHOLD=0.05,
        _PHASE_FREQ=1.6,
    )
    stubs[f"{PACKAGE_NAME}.tasks.locomotion.velocity_env_cfg"] = _module(
        f"{PACKAGE_NAME}.tasks.locomotion.velocity_env_cfg",
        JOINT_NAMES_K1=["joint"] * 12,
        ObservationsCfg=_Cfg,
    )
    walk_kick_mdp = _module(
        f"{PACKAGE_NAME}.tasks.walk_kick.mdp",
        gait_phase_sincos=_noop,
    )
    stubs[walk_kick_mdp.__name__] = walk_kick_mdp
    stubs[f"{PACKAGE_NAME}.tasks.walk_kick"].mdp = walk_kick_mdp
    stubs[f"{PACKAGE_NAME}.tasks.walk_kick.walk_kick_env_cfg"] = _module(
        f"{PACKAGE_NAME}.tasks.walk_kick.walk_kick_env_cfg",
        K1WalkKickEnvCfg=_Cfg,
        _BALL_RADIUS=0.1,
    )
    stubs[f"{PACKAGE_NAME}.tasks.walk_kick_likelihood.mdp"] = _module(
        f"{PACKAGE_NAME}.tasks.walk_kick_likelihood.mdp",
        CVKFBeliefObservation=_noop,
        observed_base_ang_vel=_noop,
        observed_kick_direction=_noop,
        RandomizeBallFriction=_noop,
        reset_moving_ball_trajectory=_noop,
    )

    module_name = f"{PACKAGE_NAME}.tasks.walk_kick_likelihood.env_cfg"
    spec = importlib.util.spec_from_file_location(module_name, ENV_CFG_PATH)
    module = importlib.util.module_from_spec(spec)
    stubs[module_name] = module
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


env_cfg = _load_env_cfg_module()


class DomainRandomizationLatentTest(unittest.TestCase):
    def test_cpu_default_mass_is_moved_to_simulation_device(self):
        robot = types.SimpleNamespace(
            data=types.SimpleNamespace(
                body_com_pos_b=torch.empty((2, 1, 3), device="meta"),
                default_mass=torch.empty((2, 1), device="cpu"),
            ),
            root_physx_view=types.SimpleNamespace(
                get_masses=lambda: torch.empty((2, 1), device="meta"),
            ),
        )
        env = types.SimpleNamespace(device="meta", scene={"robot": robot})

        latent = env_cfg._compute_domain_randomization_latent(
            env,
            _SceneEntityCfg("robot"),
        )

        self.assertEqual(latent.device.type, "meta")
        self.assertEqual(tuple(latent.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
