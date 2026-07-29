from abc import ABC, abstractmethod
import numpy as np


# =====================================================================
# 1. ARM DISTRIBUTIONS
# =====================================================================

class BaseArm(ABC):
    """Abstract Base Class for all bandit arms."""

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    @abstractmethod
    def pull(self) -> float:
        """Draw a reward sample from the arm's distribution."""
        pass

    @property
    @abstractmethod
    def expected_value(self) -> float:
        """Return the true theoretical mean (E[R]) of the arm."""
        pass
    
    @abstractmethod
    def random_walk(self, std: float = 0.01) -> None:
        """Perturb the arm's true parameter by a small Gaussian increment (non-stationarity)."""
        pass

    @abstractmethod
    def reset_params(self) -> None:
        """Restore this arm's true parameter(s) to their initial value."""
        pass


class GaussianArm(BaseArm):
    """Gaussian (Normal) distribution arm: R ~ N(mean, std^2)."""

    def __init__(self, mean: float = 0.0, std: float = 1.0, rng: np.random.Generator = None):
        super().__init__(rng)
        self.mean = mean
        self.std = std
        self._init_mean = mean

    def pull(self) -> float:
        return float(self.rng.normal(loc=self.mean, scale=self.std))

    @property
    def expected_value(self) -> float:
        return self.mean
    
    def random_walk(self, std: float = 0.01) -> None:
        self.mean += self.rng.normal(loc=0.0, scale=std)

    def reset_params(self):
        self.mean = self._init_mean


class BernoulliArm(BaseArm):
    """Bernoulli distribution arm: R ~ Binomial(1, p)."""

    def __init__(self, p: float = 0.5, rng: np.random.Generator = None):
        super().__init__(rng)
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        self.p = p
        self._init_p = p

    def pull(self) -> float:
        return float(self.rng.random() < self.p)

    @property
    def expected_value(self) -> float:
        return self.p
    
    def random_walk(self, std: float = 0.01) -> None:
        self.p = float(np.clip(self.p + self.rng.normal(loc=0.0, scale=std), 0.0, 1.0))

    def reset_params(self) -> None:
        self.p = self._init_p


class ExponentialArm(BaseArm):
    """Exponential distribution arm: R ~ Exp(scale = 1/lambda)."""

    def __init__(self, scale: float = 1.0, rng: np.random.Generator = None):
        super().__init__(rng)
        if scale <= 0:
            raise ValueError(f"Scale must be strictly positive, got {scale}")
        self.scale = scale
        self._init_scale = scale

    def pull(self) -> float:
        return float(self.rng.exponential(scale=self.scale))

    @property
    def expected_value(self) -> float:
        return self.scale
    
    def random_walk(self, std: float = 0.01) -> None:
        self.scale = max(1e-6, self.scale + self.rng.normal(loc=0.0, scale=std))

    def reset_params(self) -> None:
        self.scale = self._init_scale


# =====================================================================
# 2. BANDIT ENVIRONMENT
# =====================================================================

class KArmedBandit:
    """General K-Armed Bandit Engine supporting custom/heterogeneous arms."""

    def __init__(self, arms: list[BaseArm], seed: int = None):
        """
        Initialize bandit with a list of instantiated BaseArm objects.
        """
        if not arms:
            raise ValueError("Bandit must contain at least one arm.")
        
        self.k = len(arms)
        self.arms = arms
        self.seed = seed
        self.reset(seed=seed)

    def pull(self, action: int) -> float:
        """Execute action (pull an arm index) and return the observed reward."""
        if not (0 <= action < self.k):
            raise IndexError(f"Action index {action} out of bounds for {self.k}-armed bandit.")
        
        reward = self.arms[action].pull()
        return reward

    def reset(self, seed: int = None):
        """Reset step tracking and re-seed random number generators."""
        if seed is not None:
            self.seed = seed

        for arm in self.arms:
            arm.reset_params()
        
        # Re-seed each arm generator for reproducibility
        if self.seed is not None:
            base_rng = np.random.default_rng(self.seed)
            for arm in self.arms:
                arm.rng = np.random.default_rng(base_rng.integers(0, 1e9))
    
    def random_walk(self, std: float = 0.01) -> None:
        """Perturb all arms' true parameters (for non-stationary bandit problems)."""
        for arm in self.arms:
            arm.random_walk(std=std)

    @property
    def expected_values(self) -> np.ndarray:
        """Array of true theoretical means E[R] for all arms."""
        return np.array([arm.expected_value for arm in self.arms])

    @property
    def optimal_action(self) -> int:
        """Index of the optimal arm (highest expected reward)."""
        return int(np.argmax(self.expected_values))

    # -----------------------------------------------------------------
    # Factory Methods for Convenient Initialization
    # -----------------------------------------------------------------

    @classmethod
    def create_gaussian_bandit(cls, k: int, mean: float = 0.0, std: float = 1.0, seed: int = None):
        """Factory method to construct a standard K-armed Gaussian bandit testbed."""
        rng = np.random.default_rng(seed)
        # Sample random arm means from N(mean, std^2)
        true_means = rng.normal(loc=mean, scale=std, size=k)
        arms = [GaussianArm(mean=m, std=1.0, rng=None) for m in true_means]
        return cls(arms=arms, seed=seed)

    @classmethod
    def create_mixed_bandit(cls, seed: int = None):
        """Factory method example of a mixed/heterogeneous distribution environment."""
        arms = [
            GaussianArm(mean=2.0, std=0.5, rng=None),
            BernoulliArm(p=0.8, rng=None),
            ExponentialArm(scale=1.5, rng=None)
        ]
        return cls(arms=arms, seed=seed)
