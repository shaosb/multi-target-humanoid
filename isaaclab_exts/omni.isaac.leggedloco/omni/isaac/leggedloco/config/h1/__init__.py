import gymnasium as gym

from .h1_low_base_cfg import H1BaseRoughEnvCfg, H1BaseRoughEnvCfg_PLAY, H1RoughPPORunnerCfg
from .h1_low_vision_cfg import H1VisionRoughEnvCfg, H1VisionRoughEnvCfg_PLAY, H1VisionRoughPPORunnerCfg
from .h1_imitation_base_cfg import H1ImitationBaseRoughEnvCfg, H1ImitationBaseRoughEnvCfg_PLAY, H1ImitationPPORunnerCfg
from .h1_multi_critic_cfg import H1MultiCriticEnvCfg, H1MultiCriticEnvCfg_PLAY, H1MultiCriticPPORunnerCfg

##
# Register Gym environments.
##


gym.register(
    id="h1_base",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1BaseRoughEnvCfg,
        "rsl_rl_cfg_entry_point": H1RoughPPORunnerCfg,
    },
)


gym.register(
    id="h1_base_play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1BaseRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": H1RoughPPORunnerCfg,
    },
)


gym.register(
    id="h1_vision",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)


gym.register(
    id="h1_vision_play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)

gym.register(
    id="h1_imitation",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1ImitationBaseRoughEnvCfg,
        "rsl_rl_cfg_entry_point": H1ImitationPPORunnerCfg,
    },
)


gym.register(
    id="h1_imitation_play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1ImitationBaseRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": H1ImitationPPORunnerCfg,
    },
)

gym.register(
    id="h1_multi_critic",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1MultiCriticEnvCfg,
        "rsl_rl_cfg_entry_point": H1MultiCriticPPORunnerCfg,
    },
)


gym.register(
    id="h1_imitation_play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1MultiCriticEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": H1MultiCriticPPORunnerCfg,
    },
)