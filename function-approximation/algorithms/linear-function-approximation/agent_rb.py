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

class LinearQFunction(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(LinearQFunction, self).__init__()
        self.fc = nn.Linear(state_dim, action_dim)

    def forward(self, state):
        return self.fc(state)

    def predict(self, state):
        with torch.no_grad():
            q_values = self.forward(state)
        return q_values

class QLearningAgentv2:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.998,
        buffer_size=5000,
        batch_size=32,
        seed=2004
    ):
        """
        Implementation of a Q-Learning agent with linear function approximation and integrated replay buffer.
        We can see the difference in the perfomance of the agent with and without the replay buffer.
        It's also a chance to get familiar with the replay buffer, which is a key component of deep reinforcement learning algorithms.
        """
        # The function (set of params) that represents the Q-function
        self.f = LinearQFunction(state_dim, action_dim)
        self.rng = torch.Generator().manual_seed(int(seed))
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, batch_size, self.rng)
        self.optimizer = optim.Adam(self.f.parameters(), lr=self.lr)

    def compute_q_values(self, state):
        """
        Compute Q-values for all actions given a state
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

    def store_transition(self, state, action, reward, next_state, terminated):
        """
        Store a transition in the replay buffer.
        """
        self.replay_buffer.store_transition(state, action, reward, next_state, terminated)

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
                next_state = torch.tensor(next_state, dtype=torch.float32)

                self.store_transition(state, action, reward, next_state, terminated)

                if len(self.replay_buffer) >= self.batch_size:
                    # Sample a batch of transitions from the replay buffer
                    states, actions, rewards, next_states, terminateds = self.replay_buffer.sample_batch()

                    with torch.no_grad():
                        # td_targets: (batch_size,)
                        td_targets = rewards + (1 - terminateds) * self.gamma * torch.max(self.compute_q_values(next_states), dim=1)[0]

                    # Compute the td errors
                    # q_values: (batch_size, action_dim)
                    q_values = self.compute_q_values(states)
                    # td_errors: (batch_size,)
                    td_errors = td_targets - q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                    # Update the weights and bias using Adam optimizer
                    self.optimizer.zero_grad()
                    loss = (td_errors ** 2).mean()
                    episode_losses.append(loss.item())
                    episode_max_td_errors.append(td_errors.detach().abs().max().item())
                    
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

