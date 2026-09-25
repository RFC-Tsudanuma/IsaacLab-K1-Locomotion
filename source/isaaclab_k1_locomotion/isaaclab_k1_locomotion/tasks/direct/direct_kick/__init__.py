import gymnasium as gym

gym.register(
    id='Isaac-K1-DirectKick-v0',
    entry_point=f'{__name__}.env:DirectKickEnv',
    disable_env_checker=True,
    kwargs={'env_cfg_entry_point': f'{__name__}.env_cfg:DirectKickEnvCfg'},
)
