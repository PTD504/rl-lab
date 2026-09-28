import torch
import torch.optim as optim
from utils.mlp import MLP
from pg_utils import generate_episode
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

        self.pi = MLP(state_dim, action_dim)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=learning_rate)
        self.gamma = gamma
        run_name = run_name or 'reinforce'
        self.run_name = f'{run_name}_seed_{seed}'
        self.writer = (
            SummaryWriter(log_dir=f'{tensorboard_log_dir}/{self.run_name}')
            if tensorboard_log_dir is not None
            else None
        )

    def learn(self, env, max_episodes=10000):
        for episode in range(max_episodes):
            log_probs, _, rewards = generate_episode(env, self.pi)
            episode_return = sum(rewards)

            # compute the discounted rewards
            G = 0.0
            discounted_rewards = []
            for reward in reversed(rewards):
                G = reward + self.gamma * G
                discounted_rewards.append(G)

            discounted_rewards.reverse()
            returns = torch.tensor(discounted_rewards, dtype=torch.float32)

            # Normalize the returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)

            policy_loss = []
            for log_prob, Gt in zip(log_probs, returns):
                policy_loss.append(-log_prob * Gt)

            loss = torch.stack(policy_loss).sum()

            # update the policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.writer is not None:
                self.writer.add_scalar('Episode/return', episode_return, episode)
                self.writer.add_scalar('Episode/length', len(rewards), episode)
                self.writer.add_scalar('Training/policy_loss', loss.item(), episode)
                self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()