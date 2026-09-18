import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

class QLearningAgentv1:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.998,
        seed=2004
    ):
        # The function (set of params) that represents the Q-function
        self.model = NeuralNetwork(state_dim, action_dim, hidden_dim)
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rng = torch.Generator().manual_seed(seed)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_q_values(self, state):
        """
        Compute Q-values for all actions given a state
        """
        return self.model(state)

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy
        """
        if torch.rand(1, generator=self.rng).item() < self.epsilon:
            return torch.randint(0, self.action_dim, (1,), generator=self.rng).item()
        else:
            with torch.no_grad():
                q_values = self.compute_q_values(state)

            best_actions = torch.nonzero(q_values == torch.max(q_values)).flatten()
            best_action = torch.randint(0, best_actions.size(0), (1,), generator=self.rng).item()
            return best_actions[best_action].item()

    def learn(self, env, max_episodes=1000):
        """
        Learn the optimal policy using Q-learning with linear function approximation.
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

                # Compute the TD target
                with torch.no_grad():
                    next_state = torch.tensor(next_state, dtype=torch.float32)
                    td_target = reward + (1 - int(terminated)) * self.gamma * torch.max(self.compute_q_values(next_state))

                # Compute the td error
                td_error = td_target - self.compute_q_values(state)[action]

                # Update the weights and bias using Adam optimizer
                self.optimizer.zero_grad()
                loss = td_error ** 2
                episode_losses.append(loss.item())
                episode_max_td_errors.append(td_error.detach().abs().max().item())
                
                loss.backward()
                self.optimizer.step()

                environment_steps += 1
                episode_return += reward
                episode_length += 1

                state = next_state
                is_terminated = terminated or truncated

            # Decay epsilon after each episode
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

            history["episode_returns"].append(episode_return)
            history["episode_lengths"].append(episode_length)
            history["cumulative_steps"].append(environment_steps)
            history["losses"].append(float(np.mean(episode_losses)) if len(episode_losses) > 0 else float("nan"))
            history["max_abs_td_error"].append(max(episode_max_td_errors) if len(episode_max_td_errors) > 0 else float("nan"))

        return history