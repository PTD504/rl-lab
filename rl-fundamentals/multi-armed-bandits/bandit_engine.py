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
    def pull(self, rng: np.random.Generator) -> float:
        """Draw a reward sample from the arm's distribution."""
        pass

    @property
    @abstractmethod
    def expected_value(self) -> float:
        """Return the true theoretical mean (E[R]) of the arm."""
        pass
    
    @abstractmethod
    def random_walk(self, rng: np.random.Generator, std: float = 0.01) -> None:
        """Perturb the arm's true parameter by a small Gaussian increment (non-stationarity)."""
        pass


class GaussianArm(BaseArm):
    """Gaussian (Normal) distribution arm: R ~ N(mean, std^2)."""

    def __init__(self, mean: float = 0.0, std: float = 1.0, rng: np.random.Generator = None):
        super().__init__(rng)
        self.mean = mean
        self.std = std

    def pull(self, rng: np.random.Generator) -> float:
        return float(rng.normal(loc=self.mean, scale=self.std))

    @property
    def expected_value(self) -> float:
        return self.mean
    
    def random_walk(self, rng: np.random.Generator, std: float = 0.01) -> None:
        self.mean += rng.normal(loc=0.0, scale=std)


class BernoulliArm(BaseArm):
    """Bernoulli distribution arm: R ~ Binomial(1, p)."""

    def __init__(self, p: float = 0.5, rng: np.random.Generator = None):
        super().__init__(rng)
        self.p = p

    def pull(self, rng: np.random.Generator) -> float:
        return float(rng.random() < self.p)

    @property
    def expected_value(self) -> float:
        return self.p
    
    def random_walk(self, rng: np.random.Generator, std: float = 0.01) -> None:
        self.p = float(np.clip(self.p + rng.normal(loc=0.0, scale=std), 0.0, 1.0))


class ExponentialArm(BaseArm):
    """Exponential distribution arm: R ~ Exp(scale = 1/lambda)."""

    def __init__(self, scale: float = 1.0, rng: np.random.Generator = None):
        super().__init__(rng)
        self.scale = scale

    def pull(self, rng: np.random.Generator) -> float:
        return float(rng.exponential(scale=self.scale))

    @property
    def expected_value(self) -> float:
        return self.scale
    
    def random_walk(self, rng: np.random.Generator, std: float = 0.01) -> None:
        self.scale = max(1e-6, self.scale + rng.normal(loc=0.0, scale=std))


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
        self.rng = np.random.default_rng(seed)

    def pull(self, action: int) -> float:
        """Execute action (pull an arm index) and return the observed reward."""
        if not (0 <= action < self.k):
            raise IndexError(f"Action index {action} out of bounds for {self.k}-armed bandit.")
        
        reward = self.arms[action].pull(self.rng)
        return reward
    
    def random_walk(self, std: float = 0.01) -> None:
        """Perturb all arms' true parameters (for non-stationary bandit problems)."""
        for arm in self.arms:
            arm.random_walk(self.rng, std=std)

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

def gaussian_arms_factory(k, rng, loc=0.0, scale=1.0, arm_std=1.0):
    true_means = rng.normal(loc=loc, scale=scale, size=k)
    return [GaussianArm(mean=m, std=arm_std, rng=rng) for m in true_means]

def bernoulli_arms_factory(k, rng, low=0.0, high=1.0):
    true_p = rng.uniform(low=low, high=high, size=k)
    return [BernoulliArm(p=p, rng=rng) for p in true_p]

def exponential_arms_factory(k, rng, low=0.5, high=3.0):
    true_scale = rng.uniform(low=low, high=high, size=k)
    return [ExponentialArm(scale=s, rng=rng) for s in true_scale]