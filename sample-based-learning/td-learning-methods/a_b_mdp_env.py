import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union


class ABMDPEnv(gym.Env):
    """
    Maximization Bias / A-B MDP Environment from Sutton & Barto (Example 6.7).

    Description:
        This MDP illustrates the maximization bias of Q-Learning and how Double Q-Learning
        overcomes it.

    State Space:
        There are 2 non-terminal states and 1 absorbing terminal state:
        - State 0 (A): Starting state.
        - State 1 (B): Intermediate state with many noisy actions.
        - State 2 (Terminal): Absorbing terminal state.
        Observation space: Discrete(3)

    Action Space:
        - From State A:
            * Action 0 (Left): Deterministically transitions to State B with reward 0.
            * Action 1 (Right): Deterministically transitions to Terminal with reward 0.
            * Actions >= 2 (if num_b_actions > 2): Handled identically to Action 1 (Right).
        - From State B:
            * Actions 0 to (num_b_actions - 1): All actions transition to Terminal
              with reward sampled from Normal(loc=mean_b_reward, scale=std_b_reward),
              by default N(-0.1, 1.0).
        Action space: Discrete(num_b_actions) (default: 10)

    Optimal Policy:
        - In State A, the optimal action is Right (expected reward = 0).
        - Action Left has expected reward = -0.1.
        - Due to maximization bias (max_a Q(B, a) > 0), standard Q-Learning strongly favors
          Left early in training, while Double Q-Learning avoids this bias.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    # State Constants
    STATE_A: int = 0
    STATE_B: int = 1
    STATE_TERMINAL: int = 2
    NUM_STATES: int = 3

    # Action Constants for State A
    ACTION_LEFT: int = 0
    ACTION_RIGHT: int = 1

    def __init__(
        self,
        num_b_actions: int = 10,
        mean_b_reward: float = -0.1,
        std_b_reward: float = 1.0,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Initialize the A-B MDP environment.

        Args:
            num_b_actions: Number of actions available from State B (default: 10).
            mean_b_reward: Mean of the normal distribution for rewards from State B (default: -0.1).
            std_b_reward: Standard deviation of the normal distribution for rewards from State B (default: 1.0).
            render_mode: Render mode ('ansi' or 'human').
        """
        super().__init__()

        if num_b_actions < 2:
            raise ValueError(f"num_b_actions must be at least 2, got {num_b_actions}")

        self.num_b_actions = num_b_actions
        self.mean_b_reward = mean_b_reward
        self.std_b_reward = std_b_reward
        self.render_mode = render_mode

        # Define observation and action spaces
        self.observation_space = spaces.Discrete(self.NUM_STATES)
        self.action_space = spaces.Discrete(self.num_b_actions)

        # Current state
        self.s: int = self.STATE_A

        # Transition probability table P[s][a] = [(probability, next_state, expected_reward, terminated)]
        # This mirrors Gymnasium toy text environments for compatibility with tabular methods.
        self._init_transition_table()

    def _init_transition_table(self) -> None:
        """
        Initialize the transition probability table P[s][a].
        P[state][action] = list of tuples (prob, next_state, expected_reward, terminated)
        """
        self.P: Dict[int, Dict[int, list]] = {
            s: {a: [] for a in range(self.num_b_actions)}
            for s in range(self.NUM_STATES)
        }

        # Transitions from State A
        for a in range(self.num_b_actions):
            if a == self.ACTION_LEFT:
                # Action Left (0) -> State B (1), reward 0.0, not terminated
                self.P[self.STATE_A][a] = [(1.0, self.STATE_B, 0.0, False)]
            else:
                # Action Right (1) & other actions -> Terminal (2), reward 0.0, terminated
                self.P[self.STATE_A][a] = [(1.0, self.STATE_TERMINAL, 0.0, True)]

        # Transitions from State B
        for a in range(self.num_b_actions):
            # All actions from B -> Terminal (2), expected reward mean_b_reward, terminated
            self.P[self.STATE_B][a] = [(1.0, self.STATE_TERMINAL, self.mean_b_reward, True)]

        # Transitions from Terminal State (absorbing)
        for a in range(self.num_b_actions):
            self.P[self.STATE_TERMINAL][a] = [(1.0, self.STATE_TERMINAL, 0.0, True)]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Reset the environment to the initial state (State A).

        Args:
            seed: Seed for random number generator.
            options: Optional configuration dictionary.

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        super().reset(seed=seed)

        # Episode always begins in State A
        self.s = self.STATE_A

        if self.render_mode == "human":
            self.render()

        return self.s, {}

    def step(
        self, action: Union[int, np.integer]
    ) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action index chosen by the agent.

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info_dict).
        """
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action} for action space {self.action_space}"
            )

        action = int(action)
        truncated = False
        info: Dict[str, Any] = {}

        if self.s == self.STATE_A:
            if action == self.ACTION_LEFT:
                self.s = self.STATE_B
                reward = 0.0
                terminated = False
            else:
                # Action Right (1) or actions >= 2
                self.s = self.STATE_TERMINAL
                reward = 0.0
                terminated = True

        elif self.s == self.STATE_B:
            # All actions from B terminate the episode with stochastic reward ~ N(mean, std^2)
            self.s = self.STATE_TERMINAL
            reward = float(
                self.np_random.normal(loc=self.mean_b_reward, scale=self.std_b_reward)
            )
            terminated = True

        else:
            # Already in Terminal state
            self.s = self.STATE_TERMINAL
            reward = 0.0
            terminated = True

        if self.render_mode == "human":
            self.render()

        return self.s, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """
        Render the current state of the environment.

        Returns:
            String representation if render_mode is 'ansi', otherwise None.
        """
        if self.render_mode is None:
            return None

        # Build visual representation
        state_names = {
            self.STATE_A: "A",
            self.STATE_B: "B",
            self.STATE_TERMINAL: "Terminal",
        }

        curr_name = state_names.get(self.s, str(self.s))

        if self.s == self.STATE_A:
            diagram = "(B) <--- [A] ---> (Terminal)"
        elif self.s == self.STATE_B:
            diagram = f"[{curr_name}] (actions 0..{self.num_b_actions - 1} ~ N({self.mean_b_reward}, {self.std_b_reward}^2)) ---> (Terminal)"
        else:
            diagram = "(B) <--- (A) ---> [Terminal]"

        rendered_str = f"Current State: {curr_name}\nMDP: {diagram}\n"

        if self.render_mode == "human":
            print(rendered_str)
            return None
        elif self.render_mode == "ansi":
            return rendered_str
        else:
            return None

    def close(self) -> None:
        """Clean up environment resources."""
        pass


# Convenient alias
ABEnv = ABMDPEnv
