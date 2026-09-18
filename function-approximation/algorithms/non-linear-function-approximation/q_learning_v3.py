import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

try:
    from algorithms.replay_buffer import ReplayBuffer
except ModuleNotFoundError:
    try:
        from replay_buffer import ReplayBuffer
    except ModuleNotFoundError:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from algorithms.replay_buffer import ReplayBuffer

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        """
        Build a simple feedforward neural network with one hidden layer
        """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1_activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the network
        """
        x = self.fc1(x)
        x = self.fc1_activation(x)
        x = self.fc2(x)
        return x

    def predict(self, states):
        """
        Calculate the Q-values for a given state
        """
        with torch.no_grad():
            # Make sure the states is in shape: (batch_size, input_dim)
            q_values = self.forward(states)
        return q_values
    
class QLearningAgentv3:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        lr=0.001,
        gamma=0.99,
        epsilon=0.1,
        epsilon_decay=0.998,
        buffer_size=5000,
        batch_size=32,
        target_update_freq=300,
        seed=2004
    ):
        """
        Initialize the Q-learning agent with a neural network function approximator
        """
        self.model = NeuralNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.time_step = 0
        self.rng = torch.Generator().manual_seed(seed)
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, batch_size, self.rng)
        self.num_actions = action_dim
        self.target_model = NeuralNetwork(state_dim, action_dim, hidden_dim)
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy
        """
        if torch.rand(1, generator=self.rng).item() < self.epsilon:
            return torch.randint(0, self.num_actions, (1,), generator=self.rng).item()
        else:
            q_values = self.model.predict(state)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, terminated):
        """
        Store a transition in the replay buffer
        """
        self.replay_buffer.store_transition(state, action, reward, next_state, terminated)

    def learn(self, env, max_episodes=1000):
        """
        Learn the optimal policy using Q-learning with neural network function approximation.
        """
        history = {
            "episode_returns": [],
            "episode_lengths": [],
            "cumulative_steps": [],
            "losses": [],
            "max_abs_td_error": [],
        }
        environment_steps = 0

        for _ in range(max_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            is_terminated = False
            episode_return = 0.0
            episode_losses = []
            episode_max_td_errors = []
            episode_length = 0

            while not is_terminated:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)

                self.store_transition(state, action, reward, next_state, terminated)

                if len(self.replay_buffer) >= self.batch_size:
                    # Sample a batch of transitions from the replay buffer
                    states, actions, rewards, next_states, terminateds = self.replay_buffer.sample_batch()

                    # Compute the td target
                    with torch.no_grad():
                        next_q_values = self.target_model(next_states)
                        td_targets = rewards + (1 - terminateds) * self.gamma * torch.max(next_q_values, dim=1)[0]

                    # Compute the td error
                    q_values = self.model(states)
                    td_errors = td_targets - q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                    # Compute the loss
                    loss = (td_errors ** 2).mean()
                    episode_losses.append(loss.item())
                    episode_max_td_errors.append(td_errors.detach().abs().max().item())

                    # Perform a gradient descent step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.time_step += 1
                environment_steps += 1
                episode_return += reward
                episode_length += 1
                state = next_state
                is_terminated = terminated or truncated

                if self.time_step % self.target_update_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

            # Decay epsilon after each episode
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

            history["episode_returns"].append(float(episode_return))
            history["episode_lengths"].append(episode_length)
            history["cumulative_steps"].append(environment_steps)
            history["losses"].append(float(np.mean(episode_losses)) if len(episode_losses) > 0 else float("nan"))
            history["max_abs_td_error"].append(max(episode_max_td_errors) if len(episode_max_td_errors) > 0 else float("nan"))

        return history