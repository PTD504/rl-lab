import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LinearQFunction(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(LinearQFunction, self).__init__()
        self.fc = nn.Linear(state_dim, action_dim)

    def forward(self, state):
        return self.fc(state)

    def predict(self, state):
        # self.eval() - set the model to evaluation mode, this is important for models that have different behavior during training and evaluation (e.g. dropout, batch normalization)
        # with only one linear layer, this is not strictly necessary
        with torch.no_grad():
            q_values = self.forward(state)
        return q_values

class QLearningAgentv1:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.998,
        seed=2004
    ):
        # The function (set of params) that represents the Q-function
        self.f = LinearQFunction(state_dim, action_dim)
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rng = torch.Generator().manual_seed(seed)
        self.optimizer = optim.Adam(self.f.parameters(), lr=self.lr)

    def compute_q_values(self, state):
        """
        Compute Q-values for all actions given a state
        Actually, this is just for clarification, we can easily apply the self.f(state) in the learn function
        """
        return self.f(state)

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

                # Compute the td target, unlike the tabular case - where we can set the q value for terminal states to be zero - we have to check if the next_state is terminal or not
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