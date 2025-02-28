import gymnasium as gym

from .hit_low_base_cfg import HitBaseRoughEnvCfg, HitBaseRoughEnvCfg_PLAY, HitRoughPPORunnerCfg

gym.register(
    id="hit_base",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HitBaseRoughEnvCfg,
        "rsl_rl_cfg_entry_point": HitRoughPPORunnerCfg,
    },
)

gym.register(
    id="hit_base_play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HitBaseRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": HitRoughPPORunnerCfg,
    },
)
