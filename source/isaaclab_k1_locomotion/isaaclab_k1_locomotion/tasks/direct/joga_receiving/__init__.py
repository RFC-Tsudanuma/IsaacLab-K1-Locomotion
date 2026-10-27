import gymnasium as gym

gym.register(
    id="Isaac-K1-JogaReceive-v0",
    entry_point=f"{__name__}.env:JogaReceiveEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:JogaReceiveEnvCfg"},
)
