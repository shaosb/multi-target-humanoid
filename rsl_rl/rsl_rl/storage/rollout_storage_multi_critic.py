#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorageMultiCritic:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.multi_critic_observations = None
            self.actions = None
            self.rewards = None
            self.multi_rewards = None
            self.dones = None
            self.values = None
            self.multi_values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, multi_critic_obs_shape, actions_shape, device="cpu"):
        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.multi_critic_obs_shape = multi_critic_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        self.multi_critic_observations = torch.zeros(num_transitions_per_env, num_envs, *multi_critic_obs_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.multi_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.multi_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.target_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.multi_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.value_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.multi_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.multi_critic_observations[self.step].copy_(transition.multi_critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.multi_rewards[self.step].copy_(transition.multi_rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.multi_values[self.step].copy_(transition.multi_values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden states to match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_advantages(self):
        # TODO test if mean is better
        # self.advantages = (self.value_advantages + self.multi_advantages) / 2
        self.advantages = self.value_advantages + self.multi_advantages

    def compute_returns(self, last_values, multi_last_values, gamma, lam):
        advantage_value = 0  # Critic 1 的优势
        advantage_multi_value = 0  # Critic 2 的优势
    
        for step in reversed(range(self.num_transitions_per_env)):
            # 处理最后一个时间步
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
                next_multi_values = multi_last_values
            else:
                next_values = self.values[step + 1]
                next_multi_values = self.multi_values[step + 1]

            next_is_not_terminal = 1.0 - self.dones[step].float()

            # 计算 Critic 1 的 TD 误差
            delta_value = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage_value = delta_value + next_is_not_terminal * gamma * lam * advantage_value
            self.returns[step] = advantage_value + self.values[step]

            # 计算 Critic 2 的 TD 误差
            delta_multi_value = self.multi_rewards[step] + next_is_not_terminal * gamma * next_multi_values - self.multi_values[step]
            advantage_multi_value = delta_multi_value + next_is_not_terminal * gamma * lam * advantage_multi_value
            self.multi_returns[step] = advantage_multi_value + self.multi_values[step]


        self.value_advantages = self.returns - self.values
        self.multi_advantages = self.multi_returns - self.multi_values

        self.compute_advantages()
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
        multi_critic_observations = self.multi_critic_observations.flatten(0, 1)

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        multi_values = self.multi_values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        multi_returns = self.multi_returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                multi_critic_observations_batch = multi_critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                multi_values_batch = multi_values[batch_idx]
                returns_batch = returns[batch_idx]
                multi_returns_batch = multi_returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, multi_critic_observations_batch, multi_values_batch, multi_returns_batch, (
                    None,
                    None,
                ), None

    # for RNNs only
    # def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
    #     padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
    #     if self.privileged_observations is not None:
    #         padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
    #     else:
    #         padded_critic_obs_trajectories = padded_obs_trajectories

    #     mini_batch_size = self.num_envs // num_mini_batches
    #     for ep in range(num_epochs):
    #         first_traj = 0
    #         for i in range(num_mini_batches):
    #             start = i * mini_batch_size
    #             stop = (i + 1) * mini_batch_size

    #             dones = self.dones.squeeze(-1)
    #             last_was_done = torch.zeros_like(dones, dtype=torch.bool)
    #             last_was_done[1:] = dones[:-1]
    #             last_was_done[0] = True
    #             trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
    #             last_traj = first_traj + trajectories_batch_size

    #             masks_batch = trajectory_masks[:, first_traj:last_traj]
    #             obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
    #             critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

    #             actions_batch = self.actions[:, start:stop]
    #             old_mu_batch = self.mu[:, start:stop]
    #             old_sigma_batch = self.sigma[:, start:stop]
    #             returns_batch = self.returns[:, start:stop]
    #             advantages_batch = self.advantages[:, start:stop]
    #             values_batch = self.values[:, start:stop]
    #             old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

    #             # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
    #             # then take only time steps after dones (flattens num envs and time dimensions),
    #             # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
    #             last_was_done = last_was_done.permute(1, 0)
    #             hid_a_batch = [
    #                 saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
    #                 .transpose(1, 0)
    #                 .contiguous()
    #                 for saved_hidden_states in self.saved_hidden_states_a
    #             ]
    #             hid_c_batch = [
    #                 saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
    #                 .transpose(1, 0)
    #                 .contiguous()
    #                 for saved_hidden_states in self.saved_hidden_states_c
    #             ]
    #             # remove the tuple for GRU
    #             hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
    #             hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

    #             yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
    #                 hid_a_batch,
    #                 hid_c_batch,
    #             ), masks_batch

    #             first_traj = last_traj
