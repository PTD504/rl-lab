import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.replay_buffer import ReplayBuffer


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), max_action=1.0):
        super().__init__()
        self.register_buffer("max_action", torch.as_tensor(max_action, dtype=torch.float32))
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # Gaussian policy outputs mean and log_std for each action dimension
        self.mean_linear = nn.Linear(in_dim, action_dim)
        self.log_std_linear = nn.Linear(in_dim, action_dim)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        # Clamp log_std to prevent numerical issues
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        """
        Sample an action from the policy given a state, and compute the log probability of that action.
        """
        # Actor in continuous action space outputs mean and log_std for Gaussian distribution
        # So when we need to sample an action, we sample from the Gaussian and then apply tanh to bound the action
        # With SAC, this sample is reparameterized to allow for backpropagation through the sampling process
        # The step-by-step process is as follows:

        # compute the mean and log_std from the actor network
        mean, log_std = self.forward(state)
        # extract the standard deviation from log_std
        std = log_std.exp()
        # create a normal distribution with the computed mean and std
        normal = torch.distributions.Normal(mean, std)

        # sample an action from the normal distribution using the reparameterization trick - rsample() function
        # allows gradients to flow through the sampling process
        u = normal.rsample()
        # apply the tanh function to bound the action within [-1, 1]
        action_tanh = torch.tanh(u)
        # scale the action to the desired range by multiplying with max_action
        action = action_tanh * self.max_action

        # compute the log probability of the sampled action
        # the log probability is adjusted to account for the tanh transformation
        # Now we compute the log_pi(a|s) - which is the log probability of the action after the tanh transformation
        log_prob = normal.log_prob(u) - torch.log(1.0 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.max_action

        # return the sampled action, its log probability, and the mean action (for evaluation purposes)
        return action, log_prob, mean_action


class Critic(nn.Module):
    """
    Critic network, nothing different here.
    """
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        in_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256),
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4, # A new parameter - which is the learning rate for the entropy coefficient (alpha) optimizer. In SAC, the entropy coefficient can be learned during training, and this learning rate controls how quickly it adapts.
        gamma=0.99,
        tau=0.005,
        alpha=0.2, # A new parameter - which is the initial value of the entropy coefficient (alpha). In SAC, alpha controls the trade-off between exploration and exploitation. It is integrated directly into the loss function to encourage exploration by adding an entropy term to the objective.
        auto_entropy_tuning=True, # A new parameter related to the entropy coefficient (alpha). Just determines whether to automatically adjust alpha during training or keep it fixed.
        target_entropy=None, # Another parameter related to alpha. When we set auto_entropy_tuning to True, this parameter defines the target entropy value that the agent should aim for.
        buffer_size=1_000_000,
        batch_size=256,
        start_steps=1_000,
        max_grad_norm=None,
        device="cpu",
        seed=2004,
    ):
        self.device = torch.device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.max_grad_norm = max_grad_norm
        self.auto_entropy_tuning = auto_entropy_tuning
        self.total_steps = 0

        self.actor = Actor(state_dim, action_dim, actor_hidden_dims, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, critic_hidden_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, critic_hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.critic2 = Critic(state_dim, action_dim, critic_hidden_dims).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim, critic_hidden_dims).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Initialize the entropy coefficient (alpha) and its optimizer
        if self.auto_entropy_tuning:
            if target_entropy is None:
                # In practice, the target entropy is often set to -|A|, where |A| is the dimensionality of the action space.
                self.target_entropy = -float(action_dim)
            else:
                self.target_entropy = float(target_entropy)

            # An important aspect of SAC is the automatic tuning of the entropy coefficient (alpha). 
            # This is done by maintaining a learnable parameter log_alpha, which is optimized to minimize the difference between the current policy's entropy and the target entropy. 
            # The optimizer for log_alpha is set up here, allowing it to be updated during training.
            # You might ask why we use log_alpha instead of directly optimizing alpha. The reason is that alpha must be 
            # positive, and by optimizing log_alpha, we ensure that alpha = exp(log_alpha) is always positive.
            # We turn a constraint optimization problem (alpha > 0) into an unconstrained one by optimizing log_alpha instead of alpha directly.
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)

        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim, self.device)

    def select_action(self, state, evaluate=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            action, _, mean_action = self.actor.sample(state_tensor)
            action = mean_action if evaluate else action
        return action.cpu().numpy().squeeze(0)

    def update(self):
        """
        The update step, a little bit more complicated than DDPG and TD3 but still straighforward.
        Let's go through it step by step.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Start by sampling a batch of transitions from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Continue with computing the target Q-values for the critic update.
        # We use two critic networks to mitigate overestimation bias, similar to TD3.
        with torch.no_grad():
            # Compute the next actions and their log probabilities using the actor network
            next_actions, next_log_prob, _ = self.actor.sample(next_states)
            target_q1 = self.critic_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            # Compute the minimum of the two target Q-values and adjust it with the entropy term
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            # Complete the target Q-value computation with the reward and discount factor, taking into account whether the episode has terminated
            target_q = rewards + (1.0 - dones) * self.gamma * target_q

        # Now we compute the current Q-values from both critic networks, then compute the loss for each critic and update them

        # Compute the current Q-values
        current_q1 = self.critic(states, actions)
        current_q2 = self.critic2(states, actions)
        # Compute the critic losses using MSE
        critic_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        # Update the critic networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic2_optimizer.step()

        # Update the actor network

        # Get the new actions and their log probabilities from the actor network using the states from the replay buffer
        new_actions, log_prob, _ = self.actor.sample(states)
        # Compute the Q-values for the new actions using both critic networks
        q1 = self.critic(states, new_actions)
        q2 = self.critic2(states, new_actions)
        # Using the minimum of the two Q-values to mitigate overestimation bias
        min_q = torch.min(q1, q2)
        # Compute the actor loss, which is the expected value of the entropy-adjusted Q-values.
        actor_loss = (self.alpha.detach() * log_prob - min_q).mean()

        # Update the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        alpha_loss_val = None
        if self.auto_entropy_tuning:
            # If auto entropy tuning is enabled, we also update the entropy coefficient (alpha) to encourage exploration.
            # The loss for alpha is computed based on the difference between the current log probability and the target entropy.
            # The goal is to adjust alpha such that the policy's entropy approaches the target entropy.
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # After updating log_alpha, we extract the alpha value by exponentiating log_alpha.
            self.alpha = self.log_alpha.exp()
            alpha_loss_val = alpha_loss.item()

        # Soft update the target networks.
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss_val,
            "alpha": self.alpha.item(),
            "q_value": current_q1.mean().item(),
            "q2_value": current_q2.mean().item(),
        }

    def learn(self, env, max_steps=100_000, start_steps=None):
        """
        Learning loop for SAC.
        """
        warmup_steps = self.start_steps if start_steps is None else start_steps
        state, _ = env.reset()
        self.total_steps = 0
        episode_return = 0.0
        episode_length = 0
        episode_count = 0
        update_count = 0
        metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "critic2_loss": 0.0,
            "alpha": 0.0,
            "q_value": 0.0,
            "q2_value": 0.0,
            "episode_return": 0.0,
            "episode_length": 0.0,
            "total_steps": 0,
        }

        while self.total_steps < max_steps:
            # In each iteration, we select an action (random or from the policy)
            if self.total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = self.select_action(state, evaluate=False)

            # Then, take the action in the environment, observe the next state and reward.
            next_state, reward, terminated, truncated, _ = env.step(action)
            self.total_steps += 1
            episode_return += reward
            episode_length += 1

            # Save the transition in the replay buffer for later training.
            self.replay_buffer.add(state, action, reward, next_state, terminated)
            state = next_state

            # After collecting enough experience (1000 steps by default), we start updating our agent's networks.)
            if self.total_steps >= warmup_steps:
                step_metrics = self.update()
                if step_metrics:
                    metrics["critic_loss"] += step_metrics["critic_loss"]
                    metrics["critic2_loss"] += step_metrics["critic2_loss"]
                    metrics["actor_loss"] += step_metrics["actor_loss"]
                    metrics["alpha"] += step_metrics["alpha"]
                    metrics["q_value"] += step_metrics["q_value"]
                    metrics["q2_value"] += step_metrics["q2_value"]
                    update_count += 1

            if terminated or truncated:
                episode_count += 1
                metrics["episode_return"] += episode_return
                metrics["episode_length"] += episode_length
                state, _ = env.reset()
                episode_return = 0.0
                episode_length = 0

        if update_count:
            for key in ("actor_loss", "critic_loss", "critic2_loss", "alpha", "q_value", "q2_value"):
                metrics[key] /= update_count
        if episode_count:
            metrics["episode_return"] /= episode_count
            metrics["episode_length"] /= episode_count
        metrics["total_steps"] = self.total_steps
        return metrics

    def evaluate(self, env, num_episodes=10):
        """
        This function is used to evaluate the performance of the agent, we don't use it in the training loop.
        """
        returns = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.select_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            returns.append(total_reward)
        return float(np.mean(returns))

    def save(self, filepath):
        save_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
        }
        if self.auto_entropy_tuning:
            save_dict["log_alpha"] = self.log_alpha
            save_dict["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])
        if self.auto_entropy_tuning and "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"].to(self.device)
            self.alpha = self.log_alpha.exp()
            if "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

    def close(self):
        return None


SAC = SACAgent
