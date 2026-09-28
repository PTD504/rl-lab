import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.mlp import MLP
from pg_utils import generate_episode

import numpy as np
from torch.utils.tensorboard import SummaryWriter

class REINFORCE:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        seed=2004,
        tensorboard_log_dir='runs',
        run_name=None
    ):
        torch.manual_seed(seed)

        self.pi = MLP(state_dim, action_dim, type='policy')
        self.v = MLP(state_dim, 1, (64,), type='value')
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=learning_rate)
        self.v_optimizer = optim.Adam(self.v.parameters(), lr=learning_rate)
        self.gamma = gamma
        run_name = run_name or 'reinforce_with_baseline'
        self.run_name = f'{run_name}_seed_{seed}'
        self.writer = (
            SummaryWriter(log_dir=f'{tensorboard_log_dir}/{self.run_name}')
            if tensorboard_log_dir is not None
            else None
        )

    def learn(self, env, max_episodes=10000):
        for episode in range(max_episodes):
            log_probs, states, rewards = generate_episode(env, self.pi)
            episode_return = sum(rewards)

            # compute the discounted rewards
            G = 0.0
            discounted_rewards = []
            for reward in reversed(rewards):
                G = reward + self.gamma * G
                discounted_rewards.append(G)

            discounted_rewards.reverse()
            returns = torch.tensor(discounted_rewards, dtype=torch.float32)

            # compute the value estimates
            state_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            values = self.v(state_tensor).squeeze(-1)

            # Compute the baseline and normalize it
            deltas = returns - values.detach()
            if len(deltas) > 1:
                deltas = (deltas - deltas.mean()) / (deltas.std() + 1e-9)

            # Compute the value loss
            value_loss = F.mse_loss(values, returns)

            # update the value function
            self.v_optimizer.zero_grad()
            value_loss.backward()
            self.v_optimizer.step()

            # Compute the policy loss
            log_probs_tensor = torch.stack(log_probs)
            policy_loss = -(log_probs_tensor * deltas).sum()

            # update the policy
            self.pi_optimizer.zero_grad()
            policy_loss.backward()
            self.pi_optimizer.step()

            if self.writer is not None:
                self.writer.add_scalar('Episode/return', episode_return, episode)
                self.writer.add_scalar('Episode/length', len(rewards), episode)
                self.writer.add_scalar('Training/policy_loss', policy_loss.item(), episode)
                self.writer.add_scalar('Training/value_loss', value_loss.item(), episode)
                self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()