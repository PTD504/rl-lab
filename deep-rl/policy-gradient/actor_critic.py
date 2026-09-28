import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from utils.mlp import MLP

from torch.utils.tensorboard import SummaryWriter

class ActorCritic:
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_hidden_dims=(64, 64),
        critic_hidden_dims=(64,),
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        gamma=0.99,
        seed=2004,
        tensorboard_log_dir='runs',
        run_name=None
    ):
        torch.manual_seed(seed)

        self.actor = MLP(state_dim, action_dim, actor_hidden_dims, type='policy')
        self.critic = MLP(state_dim, 1, critic_hidden_dims, type='value')
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.gamma = gamma
        run_name = run_name or 'actor_critic'
        self.run_name = f'{run_name}_seed_{seed}'
        self.writer = (
            SummaryWriter(log_dir=f'{tensorboard_log_dir}/{self.run_name}')
            if tensorboard_log_dir is not None
            else None
        )

    def select_action(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def learn(self, env, max_episodes=10000):
        for episode in range(max_episodes):
            state, _ = env.reset()
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            done = False
            episode_return = 0.0
            episode_length = 0
            actor_losses = []
            critic_losses = []

            while not done:
                action, log_prob = self.select_action(state_tensor)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)

                # Compute the critic target
                with torch.no_grad():
                    target_value = reward + self.gamma * (1 - done) * self.critic(next_state_tensor)

                # Compute the critic loss
                value = self.critic(state_tensor)
                critic_loss = F.mse_loss(value, target_value)

                # Update the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                critic_losses.append(critic_loss.item())

                # Compute the actor loss
                advantage = target_value - value.detach()
                actor_loss = -log_prob * advantage

                # Update the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_losses.append(actor_loss.item())

                state_tensor = next_state_tensor
                episode_return += reward
                episode_length += 1

            if self.writer is not None:
                self.writer.add_scalar('Episode/return', episode_return, episode)
                self.writer.add_scalar('Episode/length', episode_length, episode)
                self.writer.add_scalar('Training/actor_loss', sum(actor_losses) / len(actor_losses), episode)
                self.writer.add_scalar('Training/critic_loss', sum(critic_losses) / len(critic_losses), episode)
                self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
