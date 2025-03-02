#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .ppo import PPO
from .ppo_multi_critic import PPOMultiCritic

__all__ = ["PPO", "PPOMultiCritic"]
