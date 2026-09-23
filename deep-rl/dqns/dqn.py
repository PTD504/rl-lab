import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.replay_buffer import ReplayBuffer
from utils.cnn import NeuralNetwork

# Integrate TensorBoard for logging and monitoring training progress
from torch.utils.tensorboard import SummaryWriter

class DQNAgent:
    def __init__(
        self,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_end=0.01,
        buffer_size=100000,
        batch_size=32,
        target_update_freq=1000,
        device='cuda',
        seed=2004,
        tensorboard_log_dir='runs',
        run_name=None
    ):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon_end = epsilon_end
        self.epsilon = 1.0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.rng = torch.Generator().manual_seed(seed)
        run_name = run_name or 'dqn'
        self.run_name = f'{run_name}_seed_{seed}'
        self.writer = (
            SummaryWriter(log_dir=f'{tensorboard_log_dir}/{self.run_name}')
            if tensorboard_log_dir is not None
            else None
        )

        # Initialize the replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, batch_size=batch_size, rng=self.rng)
        self.network = NeuralNetwork(input_channels=4, height=84, output_dim=action_dim).to(device)
        self.target_network = NeuralNetwork(input_channels=4, height=84, output_dim=action_dim).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy
        """
        if torch.rand(1, generator=self.rng).item() < self.epsilon:
            return torch.randint(0, self.action_dim, (1,), generator=self.rng).item()
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
                q_values = self.network.predict(state)
            return q_values.argmax().item()
    
    def learn(self, env, max_episodes):
        """
        Train the DQN agent
        """
        total_time_steps = 0
        exploration_steps = 100_000 # Use for linear epsilon decay

        for episode in range(max_episodes):
            state, _ = env.reset()
            is_terminated = False
            episode_return = 0.0
            episode_length = 0

            while not is_terminated:
                if len(self.replay_buffer) > 2:
                    state3, state2, state1 = self.replay_buffer.get_three_latest_frames()
                    current_state = np.stack([state1, state2, state3, state], axis=0)
                else:
                    current_state = np.stack([state, state, state, state], axis=0)

                action = self.select_action(current_state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                # Add the transition to the replay buffer
                self.replay_buffer.add(state, action, reward, next_state, terminated)

                if len(self.replay_buffer) >= self.batch_size and total_time_steps % 4 == 0: # Update the network every 4 steps
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample()
                    states = states.to(self.device, dtype=torch.float32) / 255.0
                    actions = actions.to(self.device)
                    rewards = rewards.to(self.device)
                    next_states = next_states.to(self.device, dtype=torch.float32) / 255.0
                    dones = dones.to(self.device, dtype=torch.float32)

                    # compute the td target
                    with torch.no_grad():
                        next_q_values = self.target_network(next_states)
                        td_target = rewards + self.gamma * (1 - dones) * next_q_values.max(1)[0]

                    # compute the td error
                    q_values = self.network(states)
                    td_error = td_target - q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                    loss = F.smooth_l1_loss(td_error, torch.zeros_like(td_error))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if self.writer is not None:
                        self.writer.add_scalar('Training/loss', loss.item(), total_time_steps)
                        self.writer.add_scalar(
                            'Training/mean_abs_td_error',
                            td_error.detach().abs().mean().item(),
                            total_time_steps
                        )
                        self.writer.add_scalar(
                            'Training/mean_q_value',
                            q_values.detach().mean().item(),
                            total_time_steps
                        )

                state = next_state
                is_terminated = terminated or truncated
                total_time_steps += 1
                episode_return += reward
                episode_length += 1

                if total_time_steps % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

                self.epsilon = max(self.epsilon_end, 1.0 - (total_time_steps / exploration_steps) * (1.0 - self.epsilon_end))

            if self.writer is not None:
                self.writer.add_scalar('Episode/return', episode_return, episode)
                self.writer.add_scalar('Episode/length', episode_length, episode)
                self.writer.add_scalar('Training/epsilon', self.epsilon, episode)
                self.writer.add_scalar('System/total_steps', total_time_steps, episode)
                self.writer.flush()

    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
