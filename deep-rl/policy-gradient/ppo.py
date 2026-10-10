import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128), type='policy'):
        super().__init__()
        self.output_type = type
        self.layers = nn.ModuleList()
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(last_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            last_dim = hidden_dim

        self.layers.append(nn.Linear(last_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95, # Generalized Advantage Estimation parameter, trade-off between bias and variance
        clip_epsilon=0.2,
        rollout_steps=128, # Number of steps to collect before updating the policy
        update_epochs=10, # Number of epochs to update the policy after collecting rollout_steps
        batch_size=64,
        entropy_coef=0.01, # Coefficient for entropy regularization to encourage exploration
        max_grad_norm=0.5,
        seed=2004
    ):
        torch.manual_seed(seed)
        if rollout_steps <= 0 or update_epochs <= 0 or batch_size <= 0:
            raise ValueError("rollout_steps, update_epochs, and batch_size must be positive")

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.rollout_steps = rollout_steps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.actor = Network(state_dim, action_dim, hidden_dims, type="policy")
        self.critic = Network(state_dim, 1, hidden_dims, type="value")
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    @staticmethod
    def compute_gae(
        rewards, # rewards is in shape (T,) where T is the number of time steps in the trajectory
        values, # values is in shape (T+1,), the last value is the bootstrap value for the next state
        terminated, # terminated is in shape (T,)
        truncated=None, # truncated indicates if the episode was truncated or not.
        gamma=0.99,
        gae_lambda=0.95,
    ):
        """
        Let's review GAE:
        GAE is a method for estimating the advantage function in RL. We can also use other methods like Monte Carlo or TD(lambda).
        The advantage function is defined as A(s, a) = Q(s, a) - V(s), show the difference between the action-value function and the value function in a given state.

        GAE formula:
        A_t = (gamma * lambda) ^ l * delta_{t + l} for l in range(0, inf)
        where:
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) — the 1-step TD error at time step t
        gamma is the discount factor, lambda is the GAE parameter
        lambda controls the bias-variance trade-off: lambda = 1 gives high variance but low bias (Monte Carlo), while lambda = 0 gives low variance but high bias (TD(0)).
        """
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        values = torch.as_tensor(values, dtype=torch.float32)
        terminated = torch.as_tensor(terminated, dtype=torch.float32)
        if truncated is None:
            truncated = torch.zeros_like(terminated)
        else:
            truncated = torch.as_tensor(truncated, dtype=torch.float32)

        if values.shape[0] != rewards.shape[0] + 1:
            raise ValueError("values must contain one more element than rewards")

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[0])
        for step in range(rewards.shape[0] - 1, -1, -1):
            # bootstrap_mask is used to determine if the next state is a terminal state (truncated is not considered terminal)
            # this variable is used when computing the TD error (delta)
            bootstrap_mask = 1.0 - terminated[step]
            # continuation_mask is used to determine if the episode ended (terminated or truncated) at the current step
            continuation_mask = 1.0 - torch.maximum(
                terminated[step], truncated[step]
            )
            # compute the TD error
            delta = (
                rewards[step]
                + gamma * bootstrap_mask * values[step + 1]
                - values[step]
            )
            # compute the GAE recursively
            gae = delta + gamma * gae_lambda * continuation_mask * gae
            # store the computed advantage for the current step
            advantages[step] = gae
        # we return both the advantages and the Q value estimates for each state-action pair, which is the sum of the advantage and the value function
        return advantages, advantages + values[:-1]

    def _distribution_and_value(self, states):
        """
        Compute the action distribution and value estimates for the given states.
        """
        logits = self.actor(states)
        distribution = Categorical(logits=logits)
        values = self.critic(states).squeeze(-1)
        return distribution, values

    def select_action(self, state):
        """
        Select an action based on the current state.
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            distribution, value = self._distribution_and_value(state_tensor)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        return action.squeeze(0).item(), log_prob.squeeze(0), value.squeeze(0)

    @staticmethod
    def _final_observation(info):
        """
        Extract the final observation from the info dictionary if available.
        """
        if not isinstance(info, dict):
            return None
        for key in ("final_observation", "final_obs"):
            if key in info and info[key] is not None:
                return info[key]
        return None

    def _update(self, states, actions, old_log_probs, advantages, returns):
        """
        Update the policy and value networks using the collected experiences

        states is in shape (N, state_dim), where N is the number of collected experiences. Actually, N = rollout_steps * num_envs
        actions is in shape (N,)
        old_log_probs is in shape (N,) - the log probabilities of the actions taken under the old policy
        advantages is in shape (N,) - the advantage estimates for each state-action pair
        returns is in shape (N,) - the estimated returns for each state-action pair
        """
        # Normalize the advantages - this is a common practice to improve training stability
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        indices = torch.randperm(states.shape[0])
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0

        # Perform multiple epochs of updates over the collected experiences
        for _ in range(self.update_epochs):
            # Shuffle the indices to create mini-batches for training
            indices = torch.randperm(states.shape[0])
            # Iterate over the mini-batches
            for start in range(0, states.shape[0], self.batch_size):
                # get the indices for the current mini-batch
                batch_indices = indices[start : start + self.batch_size]
                # compute the action distribution and value estimates for the current mini-batch of states
                distribution, values = self._distribution_and_value(
                    states[batch_indices]
                )
                # compute the log prob of the actions taken under the current policy
                log_probs = distribution.log_prob(actions[batch_indices])
                # compute the entropy of the current policy
                entropy = distribution.entropy().mean()
                # compute the ratio of the new and old policy probs for the actions taken, which is used in the PPO objective
                ratios = (log_probs - old_log_probs[batch_indices]).exp()
                # compute the surrogate loss and the clipped surrogate loss for the PPO objective
                surrogate = ratios * advantages[batch_indices]
                clipped_surrogate = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * advantages[batch_indices]
                # compute the final policy loss and value loss
                policy_loss = -torch.minimum(surrogate, clipped_surrogate).mean()
                value_loss = F.mse_loss(values, returns[batch_indices])
                # compute the total actor loss, which includes the policy loss and an entropy regularization term to encourage exploration
                actor_loss = policy_loss - self.entropy_coef * entropy

                # update the actor and critic networks
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic_optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                updates += 1

        return {name: value / updates for name, value in metrics.items()}

    def learn(self, env, max_steps=100_000):
        """
        Learn the policy using the PPO algorithm
        """
        state, _ = env.reset()
        total_steps = 0
        metrics = {}

        while total_steps < max_steps:
            states = []
            actions = []
            rewards = []
            old_log_probs = []
            values = []
            terminated = []
            truncated = []

            # Collect experiences for a number of rollout steps or until the maximum number of steps is reached
            for _ in range(min(self.rollout_steps, max_steps - total_steps)):
                action, log_prob, value = self.select_action(state)
                next_state, reward, ended, was_truncated, info = env.step(action)
                final_observation = self._final_observation(info)
                bootstrap_state = (
                    final_observation
                    if was_truncated and final_observation is not None
                    else next_state
                )

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                old_log_probs.append(log_prob)
                values.append(value)
                terminated.append(ended)
                truncated.append(was_truncated)
                state = next_state
                total_steps += 1

                if ended or was_truncated:
                    with torch.no_grad():
                        bootstrap_tensor = torch.as_tensor(
                            bootstrap_state, dtype=torch.float32
                        ).unsqueeze(0)
                        next_value = self.critic(bootstrap_tensor).squeeze()
                    values.append(next_value)
                    state, _ = env.reset()
                    break

            if not values:
                break

            if len(values) == len(rewards):
                with torch.no_grad():
                    next_state_tensor = torch.as_tensor(
                        state, dtype=torch.float32
                    ).unsqueeze(0)
                    values.append(self.critic(next_state_tensor).squeeze())

            # compute the advantages and returns using GAE, we will use these to update the policy and value networks
            advantages, returns = self.compute_gae(
                rewards,
                torch.stack(values),
                terminated,
                truncated,
                self.gamma,
                self.gae_lambda,
            )
            # update the policy and value networks using the collected experiences
            metrics = self._update(
                torch.as_tensor(np.asarray(states), dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.long),
                torch.stack(old_log_probs),
                advantages,
                returns,
            )

        return metrics

    def close(self):
        return None