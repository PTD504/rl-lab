import numpy as np

class RBQLearningAgent:
    """
    Implementation of a Q-Learning agent with linear function approximation and integrated replay buffer.
    We can see the difference in the perfomance of the agent with and without the replay buffer.
    It's also a chance to get familiar with the replay buffer, which is a key component of deep reinforcement learning algorithms.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.01,
        gamma=0.99,
        epsilon=0.1,
        batch_size=32,
        buffer_size=1000,
        seed=2004
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros((action_dim, state_dim))
        self.b = np.zeros(action_dim)
        self.replay_buffer = []

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

    def store_transition(self, state, action, reward, next_state, terminated):
        """
        Store a transition in the replay buffer
        """
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, terminated))

    def sample_batch(self):
        """
        Sample a batch of transitions from the replay buffer
        """
        indices = self.rng.choice(len(self.replay_buffer), size=self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        return batch

    def learn(self, env, max_episodes=1000):
        """
        Learn the optimal policy using Q-learning with linear function approximation and replay buffer.
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

                # Store the transition in the replay buffer
                self.store_transition(state, action, reward, next_state, terminated)

                # Sample a batch of transitions from the replay buffer
                if len(self.replay_buffer) >= self.batch_size:
                    batch = self.sample_batch()

                    # Unpack the batch into numpy arrays for vectorized operations
                    # The specific shape of each array is as follows:
                    # states: (batch_size, state_dim)
                    # actions: (batch_size,)
                    # rewards: (batch_size,)
                    # next_states: (batch_size, state_dim)
                    # terminateds: (batch_size,)
                    states = np.array([transition[0] for transition in batch])
                    actions = np.array([transition[1] for transition in batch])
                    rewards = np.array([transition[2] for transition in batch])
                    next_states = np.array([transition[3] for transition in batch])
                    terminateds = np.array([transition[4] for transition in batch], dtype=np.float32)

                    # Vectorized next q-values: (batch_size, action_dim)
                    next_q_values = np.dot(self.w, next_states.T) + self.b[:, None]
                    # self.w: (action_dim, state_dim)
                    # next_states.T: (state_dim, batch_size)
                    # next_q_values: (action_dim, batch_size)
                    max_next_q_values = np.max(next_q_values, axis=0) # (batch_size,)

                    # Compute the td targets: (batch_size,)
                    td_targets = rewards + (1 - terminateds) * self.gamma * max_next_q_values

                    # Compute the current Q values for specific actions taken in batch: (batch_size,)
                    q_values = np.sum(self.w[actions] * states, axis=1) + self.b[actions]

                    # Compute the td errors: (batch_size,)
                    td_errors = td_targets - q_values

                    # Compute and apply the gradients per action in the batch
                    dw = np.zeros_like(self.w)
                    db = np.zeros_like(self.b)

                    # Use np.add.at to accumulate gradients for each action in the batch
                    # This is equivalent to:
                    # for i in range(self.batch_size):
                    #     dw[actions[i]] += td_errors[i] * states[i]
                    #     db[actions[i]] += td_errors[i]
                    np.add.at(dw, actions, td_errors[:, None] * states)
                    np.add.at(db, actions, td_errors)

                    self.w += self.lr * (dw / self.batch_size)
                    self.b += self.lr * (db / self.batch_size)
                    episode_td_errors.extend(td_errors.tolist())
                    episode_losses.append(float(np.mean(td_errors ** 2)))
                    gradient_updates += 1

                state = next_state
                environment_steps += 1
                episode_return += reward
                episode_length += 1
                is_terminated = terminated or truncated

            history["episode_returns"].append(episode_return)
            history["episode_lengths"].append(episode_length)
            history["environment_steps"].append(environment_steps)
            history["td_error_means"].append(
                float(np.mean(episode_td_errors)) if episode_td_errors else np.nan
            )
            history["td_error_stds"].append(
                float(np.std(episode_td_errors)) if episode_td_errors else np.nan
            )
            history["losses"].append(
                float(np.mean(episode_losses)) if episode_losses else np.nan
            )
            history["gradient_updates"].append(gradient_updates)
            history["buffer_sizes"].append(len(self.replay_buffer))
            history["parameter_norms"].append(float(np.linalg.norm(self.w) + np.linalg.norm(self.b)))

        return history

    def get_weights(self):
        """
        Get the current weights and bias of the linear function approximator
        """
        return self.w, self.b

