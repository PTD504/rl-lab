import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.replay_buffer import ReplayBuffer


class Actor(nn.Module):
    """
    Actor network. We keep a maximum action value to scale the output of the network to the appropriate range
    for the env. It can be done by multiplying the output of the network by the max_action outside the network,
    but we keep it inside the network to avoid mistakes and make it easier to use the actor in other places.
    """
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), max_action=1.0):
        super().__init__()
        self.register_buffer("max_action", torch.as_tensor(max_action, dtype=torch.float32))
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """
    Critic network. We estimate the Q-value, not the V-value, so we need to take both the state and the action
    as input to the network.
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


class TD3Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0, # maximum action value for the environment
        actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256),
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005, # soft update coefficient for target networks
        buffer_size=1_000_000,
        batch_size=64,
        start_steps=1_000, # number of initial random steps before using the policy for exploration
        exploration_noise=0.1, # standard deviation of Gaussian noise added to actions for exploration
        target_policy_noise=0.2, # standard deviation of Gaussian noise added to target actions for critic update
        target_noise_clip=0.5, # clipping range for target policy noise
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
        self.exploration_noise = exploration_noise
        self.max_grad_norm = max_grad_norm

        self.actor = Actor(state_dim, action_dim, actor_hidden_dims, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, actor_hidden_dims, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, critic_hidden_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, critic_hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim, self.device)

        # TD3 components
        self.critic2 = Critic(state_dim, action_dim, critic_hidden_dims).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim, critic_hidden_dims).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

        self.policy_delay = 2 # policy is updated less frequently than the critics
        self.total_steps = 0 # total number of steps taken in the environment, used to determine when to update the policy

    def select_action(self, state, add_noise=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().squeeze(0)

        # In standard DDPG, we add Gaussian noise to the action at every single step during training.
        if add_noise:
            max_act = self.actor.max_action.cpu().numpy()
            noise = np.random.normal(0, self.exploration_noise * max_act, size=action.shape)
            action = np.clip(action + noise, -max_act, max_act)
        return action

    def update(self):
        """
        The update logic.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample a batch of transitions from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute the target Q values using the target networks, very similar to DQN
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # TD3 adds noise to the target action to make the critic more robust to errors in the policy.
            next_actions += torch.clamp(torch.randn_like(next_actions) * self.target_policy_noise, -self.target_noise_clip, self.target_noise_clip)
            next_actions = torch.clamp(next_actions, -self.actor.max_action, self.actor.max_action)

            # Compute the target Q values using the target networks (critic_target and critic2_target)
            target_q = self.critic_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)

            # Compute the target Q value using the minimum of the two critics to mitigate overestimation bias.
            target_q = rewards + (1.0 - dones) * self.gamma * torch.min(target_q, target_q2)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # TD3 need to update two critics, so we need to update the second critic as well.
        current_q2 = self.critic2(states, actions)
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic2_optimizer.step()

        # TD3 update the actor and target networks less frequently than the active critics.
        actor_updated = self.total_steps % self.policy_delay == 0
        actor_loss = None
        if actor_updated:
            # A little bit confused in the first look right? The actor loss is only contain the Q value of the current 
            # state-action pair, how can we update the actor with it?
            # Well, the actor loss is actually the negative of the Q value of the current state-action pair. Why?
            # Because the current action (in the actor loss) is generated by the actor network, and the backpropagation 
            # will flow through the critic layers until it hits the action input, and this calculates the gradient
            # of the actor network params

            # TD3 algorithm uses two critics, which one to use for the actor loss?
            # The answer is to use the first critic, but you can also use the second critic, or even the average of the two critics.
            # The reason is that the actor loss is only used to update the actor network,
            # and the actor network is only used to generate actions, so it "doesn't matter" 
            # (doesn't matter here doesn't mean they all produce the same result but it's about the effectiveness of the update) 
            # which critic to use for the actor loss.
            # Because both critics are headed to the same goal, which is to estimate the Q value of the current state-action pair.
            actor_loss = -self.critic(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Update the target networks using soft updates
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else None,
            "q_value": current_q.mean().item(),
            "q2_value": current_q2.mean().item(),
            "actor_updated": actor_updated,
        }

    def learn(self, env, max_steps=100_000, start_steps=None):
        """
        The learning loop.
        """
        warmup_steps = self.start_steps if start_steps is None else start_steps
        state, _ = env.reset()
        self.total_steps = 0
        episode_return = 0.0
        episode_length = 0
        episode_count = 0
        update_count = 0
        actor_update_count = 0
        metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "critic2_loss": 0.0,
            "q_value": 0.0,
            "q2_value": 0.0,
            "episode_return": 0.0,
            "episode_length": 0.0,
            "total_steps": 0,
        }

        while self.total_steps < max_steps:
            # At each step, select the action based on the current policy, or random action during warmup phase.
            if self.total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = self.select_action(state, add_noise=True)

            # Take the selected action
            next_state, reward, terminated, truncated, _ = env.step(action)
            self.total_steps += 1
            episode_return += reward
            episode_length += 1

            # Add the new transitions to the replay buffer
            self.replay_buffer.add(state, action, reward, next_state, terminated)
            state = next_state

            # In the warmup phase, we only collect transitions by taking random actions (and not update the networks).
            # After the warmup phase, we update the networks at every step.
            if self.total_steps >= warmup_steps:
                step_metrics = self.update()
                if step_metrics:
                    metrics["critic_loss"] += step_metrics["critic_loss"]
                    metrics["critic2_loss"] += step_metrics["critic2_loss"]
                    metrics["q_value"] += step_metrics["q_value"]
                    metrics["q2_value"] += step_metrics["q2_value"]
                    if step_metrics["actor_updated"]:
                        metrics["actor_loss"] += step_metrics["actor_loss"]
                        actor_update_count += 1
                    update_count += 1

            if terminated or truncated:
                episode_count += 1
                metrics["episode_return"] += episode_return
                metrics["episode_length"] += episode_length
                state, _ = env.reset()
                episode_return = 0.0
                episode_length = 0

        if update_count:
            for key in ("critic_loss", "critic2_loss", "q_value", "q2_value"):
                metrics[key] /= update_count
        if actor_update_count:
            metrics["actor_loss"] /= actor_update_count
        if episode_count:
            metrics["episode_return"] /= episode_count
            metrics["episode_length"] /= episode_count
        metrics["total_steps"] = self.total_steps
        return metrics

    def evaluate(self, env, num_episodes=10):
        returns = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.select_action(state, add_noise=False)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            returns.append(total_reward)
        return float(np.mean(returns))

    def save(self, filepath):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "critic2": self.critic2.state_dict(),
                "critic2_target": self.critic2_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "critic2_optimizer": self.critic2_optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])

    def close(self):
        return None


TD3 = TD3Agent
