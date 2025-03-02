import os

from .wrappers import RslRlVecEnvHistoryWrapper, RslRlMultiCriticVecEnvHistoryWrapper

ASSETS_DIR = os.path.abspath("assets")

__all__ = [
    "ASSETS_DIR",
    "RslRlVecEnvHistoryWrapper",
    "RslRlMultiCriticVecEnvHistoryWrapper",
]
