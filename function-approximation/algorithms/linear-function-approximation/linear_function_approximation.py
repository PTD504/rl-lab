import numpy as np

class QLearningAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.01,
        gamma=0.99,
        epsilon=0.1,
        seed=2004
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros((action_dim, state_dim))
        self.b = np.zeros(action_dim)

    def compute_q_value(self, state, action):
        """
        Compute Q-values for a given state and action
        """
        return np.dot(self.w[action], state) + self.b[action]

    def compute_q_values_all_actions(self, state):
        """
        Compute Q-values for all actions given a state
        """
        return np.dot(self.w, state) + self.b

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy
        """
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.action_dim)
        else:
            q_values = self.compute_q_values_all_actions(state)
            best_actions = np.flatnonzero(q_values == np.max(q_values))
            return self.rng.choice(best_actions)

    def learn(self, env, max_episodes=1000):
        """
        Learn the optimal policy using Q-learning with linear function approximation.
        """
        history = {
            "episode_returns": [],
            "episode_lengths": [],
            "environment_steps": [],
            "td_error_means": [],
            "td_error_stds": [],
            "losses": [],
            "gradient_updates": [],
            "buffer_sizes": [],
            "parameter_norms": [],
        }
        environment_steps = 0
        gradient_updates = 0

        for _ in range(max_episodes):
            state, _ = env.reset()
            is_terminated = False
            episode_return = 0.0
            episode_td_errors = []
            episode_losses = []
            episode_length = 0

            while not is_terminated:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                # Compute the td target, unlike the tabular case - where we can set the q value for terminal states to be zero - we have to check if the next_state is terminal or not
                td_target = reward + (1 - int(terminated)) * self.gamma * np.max(self.compute_q_values_all_actions(next_state))

                # Compute the td error
                td_error = td_target - self.compute_q_value(state, action)
                episode_td_errors.append(td_error)
                episode_losses.append(td_error ** 2)

                # Compute the gradient of the q-value with respect to the weights and bias
                # The gradient is simple because the q-value is a linear function of the weights and bias
                grad_w = state
                grad_b = 1.0

                # Update the weights and bias using gradient descent
                self.w[action] += self.lr * td_error * grad_w
                self.b[action] += self.lr * td_error * grad_b
                gradient_updates += 1
                environment_steps += 1
                episode_return += reward
                episode_length += 1

                state = next_state
                is_terminated = terminated or truncated

            history["episode_returns"].append(episode_return)
            history["episode_lengths"].append(episode_length)
            history["environment_steps"].append(environment_steps)
            history["td_error_means"].append(float(np.mean(episode_td_errors)))
            history["td_error_stds"].append(float(np.std(episode_td_errors)))
            history["losses"].append(float(np.mean(episode_losses)))
            history["gradient_updates"].append(gradient_updates)
            history["buffer_sizes"].append(0)
            history["parameter_norms"].append(float(np.linalg.norm(self.w) + np.linalg.norm(self.b)))

        return history

    def get_weights(self):
        """
        Get the weights and bias of the linear function approximator
        """
        return self.w, self.b