import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# A simple neural network architecture for the actor and critic in A2C
class NeuralNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        # Both actor and critic share the same feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Actor head outputs action logits for the policy
        self.actor_head = nn.Linear(128, num_actions)
        # Critic head outputs a single value for the state value function
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.feature(x)
        action_logits = self.actor_head(features)
        state_value = self.critic_head(features).squeeze(-1)
        return action_logits, state_value


class A2CAgent:
    def __init__(
        self,
        state_dim,
        num_actions,
        lr=1e-3,
        gamma=0.99,
        entropy_beta=0.01,
        n_steps=5,
        device=torch.device("cpu"),
        seed=2004,
        max_grad_norm=1.0,
    ):
        torch.manual_seed(seed)

        self.device = device
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.n_steps = n_steps
        self.max_grad_norm = max_grad_norm

        self.network = NeuralNetwork(state_dim, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def select_action(self, state):
        """
        Selects an action based on the current state using the policy network
        """
        state = torch.as_tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        )

        action_logits, state_value = self.network(state)
        distribution = Categorical(logits=action_logits)
        action = distribution.sample()

        # Return the selected action, log probability of the action, entropy of the distribution, and the state value
        # We'll use these values later for computing the loss and updating the network
        return (
            action.cpu().numpy(),
            distribution.log_prob(action),
            distribution.entropy(),
            state_value,
        )

    @staticmethod
    def _final_observations(
        info,
        next_state,
        truncated,
        terminated,
        autoreset_mode,
    ):
        truncated_indices = np.flatnonzero(truncated & ~terminated)
        if truncated_indices.size == 0:
            return truncated_indices, None

        for key in ("final_obs", "final_observation"):
            if key not in info:
                continue
            final_obs = np.asarray(info[key])
            final_mask = np.asarray(
                info.get(f"_{key}", np.ones(truncated.shape, dtype=bool)),
                dtype=bool,
            )
            if not np.all(final_mask[truncated_indices]):
                break
            try:
                return truncated_indices, np.stack(
                    final_obs[truncated_indices]
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Vector info[{key!r}] does not contain stackable "
                    "final observations."
                ) from exc

        mode_name = getattr(autoreset_mode, "name", str(autoreset_mode))
        if mode_name != "SAME_STEP":
            return truncated_indices, np.asarray(next_state)[truncated_indices]
        raise RuntimeError(
            "Truncated environments did not provide final observations in "
            "info. Expected 'final_obs' or 'final_observation' with its "
            "corresponding mask for SAME_STEP autoreset."
        )

    def learn(self, env, max_steps=100_000):
        """
        Main function to train the A2C agent
        We use multiple environments to collect experiences, which allows for more stable learning and better exploration of the state space
        Using multiple environments is a key feature of the A2C algorithm
        """
        state, _ = env.reset()
        num_envs = env.num_envs
        total_steps = 0
        autoreset_mode = getattr(env, "autoreset_mode", None)
        autoreset_mode_name = getattr(
            autoreset_mode, "name", str(autoreset_mode)
        )
        if autoreset_mode_name == "DISABLED":
            raise ValueError(
                "A2CAgent.learn requires an autoresetting vector environment; "
                "DISABLED mode needs per-environment resets."
            )
        needs_reset = np.zeros(num_envs, dtype=bool)

        # Track episode return and length for each parallel environment
        episode_returns = np.zeros(num_envs, dtype=np.float32)
        episode_lengths = np.zeros(num_envs, dtype=np.int32)
        metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "episode_return": 0.0,
            "episode_length": 0.0,
            "total_steps": 0,
        }
        update_count = 0
        episode_count = 0

        while total_steps < max_steps:
            # Collect experiences for n_steps, the n_steps value is quite small (which is often 5 or 10) so using a list is enough instead of deques or other data structures
            log_probs = []
            entropies = []
            values = []
            rewards = []
            bootstrap_masks = []
            transition_masks = []

            # Step through the env for n_steps and collect experiences
            for _ in range(self.n_steps):
                # action is in shape of (num_envs,)
                # log_prob is in shape of (num_envs,)
                # entropy is in shape of (num_envs,)
                # value is in shape of (num_envs,)
                action, log_prob, entropy, value = self.select_action(state)
                # Step the environment with the selected action
                # next_state is in shape of (num_envs, state_dim)
                # reward is in shape of (num_envs,)
                # terminated is in shape of (num_envs,)
                # truncated is in shape of (num_envs,)
                next_state, reward, terminated, truncated, info = env.step(action)
                dones = np.logical_or(terminated, truncated)
                valid_transition = np.logical_not(needs_reset)
                reward_tensor = torch.as_tensor(
                    reward, dtype=torch.float32, device=self.device
                )
                truncated_indices, final_observations = (
                    self._final_observations(
                        info,
                        next_state,
                        truncated,
                        terminated,
                        autoreset_mode,
                    )
                )
                if final_observations is not None:
                    final_obs_tensor = torch.as_tensor(
                        final_observations,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    with torch.no_grad():
                        _, final_values = self.network(final_obs_tensor)
                    reward_tensor[truncated_indices] += (
                        self.gamma * final_values
                    )

                log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(value)
                rewards.append(reward_tensor)
                transition_masks.append(
                    torch.as_tensor(
                        valid_transition,
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                bootstrap_masks.append(
                    torch.as_tensor(
                        np.logical_not(dones) & valid_transition,
                        dtype=torch.float32,
                        device=self.device,
                    )
                )

                episode_returns[valid_transition] += reward[valid_transition]
                episode_lengths[valid_transition] += 1
                total_steps += num_envs

                done_indices = np.flatnonzero(dones & valid_transition)
                if done_indices.size:
                    metrics["episode_return"] += float(
                        episode_returns[done_indices].sum()
                    )
                    metrics["episode_length"] += float(
                        episode_lengths[done_indices].sum()
                    )
                    episode_count += done_indices.size
                    episode_returns[done_indices] = 0.0
                    episode_lengths[done_indices] = 0

                state = next_state
                if autoreset_mode_name == "NEXT_STEP":
                    needs_reset = dones
                else:
                    needs_reset.fill(False)

                if total_steps >= max_steps:
                    break

            with torch.no_grad():
                next_state_tensor = torch.as_tensor(
                    np.asarray(state), dtype=torch.float32, device=self.device
                )
                # Calculate the value of the next state for bootstrapping
                # bootstrap is in shape of (num_envs,)
                _, bootstrap = self.network(next_state_tensor)

            # Quick reminds:
            # log_probs is in shape of (n_steps, num_envs)
            # entropies is in shape of (n_steps, num_envs)
            # values is in shape of (n_steps, num_envs)
            # rewards is in shape of (n_steps, num_envs)
            # bootstrap_masks is in shape of (n_steps, num_envs)
            log_probs = torch.stack(log_probs)
            entropies = torch.stack(entropies)
            values = torch.stack(values)
            rewards = torch.stack(rewards)
            bootstrap_masks = torch.stack(bootstrap_masks)
            transition_masks = torch.stack(transition_masks)

            returns = torch.empty_like(rewards)
            # We can reuse the bootstrap variable, but I just want to make it clear that the bootstrap variable is used to calculate the returns, which is the discounted sum of rewards plus the value of the next state
            return_value = bootstrap
            for index in range(rewards.shape[0] - 1, -1, -1):
                # Calculate the return for each step
                # The n-steps bootstrapping formula: G = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^{n-1} * r_{t+n-1} + gamma^n * V(s_{t+n}) (if not terminal)
                return_value = (
                    rewards[index]
                    + self.gamma * bootstrap_masks[index] * return_value
                )
                returns[index] = return_value

            # Compute the advantage: Q(s, a) - V(s) = G - V(s) ~ TD Error
            advantages = returns - values
            # Compute the losses, note that in the actor loss, we detach the advantages to prevent gradients from flowing into the critic network when updating the actor network
            valid_count = transition_masks.sum()
            if valid_count.item() == 0:
                continue
            actor_loss = -(
                log_probs * advantages.detach() * transition_masks
            ).sum() / valid_count
            critic_loss = (
                advantages.pow(2) * transition_masks
            ).sum() / (2 * valid_count)
            entropy_loss = entropies.mean()
            total_loss = actor_loss + critic_loss - self.entropy_beta * entropy_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            metrics["actor_loss"] += actor_loss.item()
            metrics["critic_loss"] += critic_loss.item()
            metrics["entropy"] += entropy_loss.item()
            metrics["total_loss"] += total_loss.item()
            update_count += 1

        if update_count:
            for key in ("actor_loss", "critic_loss", "entropy", "total_loss"):
                metrics[key] /= update_count
        if episode_count:
            metrics["episode_return"] /= episode_count
            metrics["episode_length"] /= episode_count
        metrics["total_steps"] = total_steps
        return metrics

    def close(self):
        return None


"""
Some notes:
- You can see that, the A2C agent implementation above works as follows:
1. The agent interacts with the environment for a fixed number of steps (n_steps) and collects experiences (state, action, reward, next_state, done) from multiple environments in parallel.
2. After collecting experiences, the agent computes the returns (discounted sum of rewards) and advantages (returns - values)
-> And if we look more closely, the update process is done in a batch manner, which means that the agent updates its policy and value function after collecting a batch of experiences, rather than after every single step.
-> Agent collect experiences for n_steps, update the policy and value function using the n_steps experiences, and then repeat the process beginning from the (n_steps + 1)-th step.

- But, there's another way to implement A2C, which is to update the policy and value function after every single step, which is often referred to as "online" update.
So how it works?
- The implementation is quite similar to the above implementation, but instead update for all the actions taken in the n_steps, we update the policy and value function for the action taken at the current step, using the same n-step bootstrapping formula to calculate the return and advantage for the current step.
- The process is as follows:
1. The agent interacts with the environment for a fixed number of steps (n_steps) and collects experiences
2. After collecting experiences, the agent computes the return and advantage for the current step using the n-step bootstrapping formula
3. This is where the difference lies, instead of updating the policy and value function for all transitions in the n_steps, we update the policy and value function for the action taken at the current step (the first step in the n_steps batch)
The `returns` and `advantages` are in shape of (1, num_envs) instead of (n_steps, num_envs), which contains the return and advantage for the first step in the n_steps batch (return = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^{n-1} * r_{t+n-1} + gamma^n * V(s_{t+n}) (if not terminal), advantage = return - V(s_t))
4. The agent updates the policy and value function, then remove the first step from the n_steps batch and take the next step in the environment from the last state in the n_steps batch, and repeat the process until the end of the n_steps batch.

The second implementation is more "online" and can be more sample efficient, but it destroy the parallelism of the n_steps batch, and can't take advantage of the tradeoff between bias and variance in the n-step bootstrapping formula, which is one of the key features of A2C.
"""