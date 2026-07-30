from abc import ABC, abstractmethod
import numpy as np

class BanditAgent(ABC):
    """Abstract Base Class for all bandit agents."""

    def __init__(self, n_arms: int, initial_value: float = 0.0, seed: int = 84):
        self.k = n_arms
        self.action_counts = np.zeros(n_arms, dtype=int)  # Number of times each arm was pulled
        self.Q_values = np.full(n_arms, initial_value, dtype=float)  # Estimated value of each arm
        self.rng = np.random.default_rng(seed)
    
    def _argmax_random_tiebreak(self, values: np.ndarray) -> int:
        """np.argmax always returns the first max; break ties uniformly at random instead."""
        max_val = np.max(values)
        candidates = np.flatnonzero(values == max_val)
        return int(self.rng.choice(candidates))

    @abstractmethod
    def select_action(self) -> int:
        """Select an arm to pull based on the agent's strategy."""
        pass

    @abstractmethod
    def update(self, chosen_action: int, reward: float):
        """Update the agent's estimates based on the received reward."""
        pass

class EpsilonGreedyAgent(BanditAgent):
    """Epsilon-Greedy Agent: Selects a random arm with probability epsilon, otherwise selects the best arm."""

    def __init__(self, n_arms: int, epsilon: float = 0.1, initial_value: float = 0.0, alpha: float = None, seed: int = 84):
        super().__init__(n_arms, initial_value, seed)
        self.epsilon = epsilon
        self.alpha = alpha

    def select_action(self) -> int:
        """Select an arm using epsilon-greedy strategy."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))  # Explore: random arm
        else:
            return self._argmax_random_tiebreak(self.Q_values)  # Exploit: best arm

    def update(self, chosen_action: int, reward: float):
        """Update the Q-value of the chosen arm."""
        self.action_counts[chosen_action] += 1

        step = 1.0 / self.action_counts[chosen_action] if self.alpha is None else self.alpha

        self.Q_values[chosen_action] += step * (reward - self.Q_values[chosen_action])

class UCBAgent(BanditAgent):
    """Upper Confidence Bound (UCB) Agent: Selects arms based on UCB algorithm."""

    def __init__(self, n_arms: int, initial_value: float = 0.0, c: float = 2.0, alpha: float = None, seed: int = 84):
        super().__init__(n_arms, initial_value, seed)
        self.c = c  # Controls the degree of exploration
        self.alpha = alpha

    def select_action(self) -> int:
        """Select an arm using the UCB strategy."""
        unpulled = np.flatnonzero(self.action_counts == 0)
        if unpulled.size > 0:
            return int(self.rng.choice(unpulled))  # Pull each arm at least once, ties broken randomly
        
        total_counts = np.sum(self.action_counts)
        ucb_values = self.Q_values + self.c * np.sqrt(np.log(total_counts) / self.action_counts)

        return self._argmax_random_tiebreak(ucb_values)

    def update(self, chosen_action: int, reward: float):
        """Update the Q-value of the chosen arm."""
        self.action_counts[chosen_action] += 1

        step = 1.0 / self.action_counts[chosen_action] if self.alpha is None else self.alpha

        self.Q_values[chosen_action] += step * (reward - self.Q_values[chosen_action])