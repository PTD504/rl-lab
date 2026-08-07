"""
Deterministic GridWorld environment for Dynamic Programming algorithms.

This environment implements a standard grid world where transitions are deterministic
and model-based. Dynamic Programming algorithms (Policy Evaluation, Policy Iteration,
and Value Iteration) can query environment dynamics directly using `step(state, action)`
because there is no stochasticity to model (i.e. p(s', r | s, a) = 1 for the unique next state s'
and reward r = -1.0).

Terminal states are sampled once at environment construction. DP update loops should iterate
over `non_terminal_states`, keeping the value of terminal states fixed at 0.0 by convention.
"""

import random
from typing import ClassVar


class GridWorld:
    """
    Deterministic grid world environment with fixed terminal states and constant step reward.

    Parameters
    ----------
    n_rows : int
        Number of rows in the grid (must be > 3).
    n_cols : int
        Number of columns in the grid (must be > 3).
    seed : int | None, optional
        Seed for pseudo-random number generator to select terminal states deterministically.

    Attributes
    ----------
    ACTIONS : list[str]
        Available action strings: ["up", "down", "left", "right"].
    """

    ACTIONS: ClassVar[list[str]] = ["up", "down", "left", "right"]

    def __init__(self, n_rows: int, n_cols: int, seed: int | None = None) -> None:
        assert n_rows > 3, f"n_rows must be > 3, got {n_rows}"
        assert n_cols > 3, f"n_cols must be > 3, got {n_cols}"

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.seed = seed

        # Generate all grid states (row, col), 0-indexed
        self._all_states: list[tuple[int, int]] = [
            (r, c) for r in range(self.n_rows) for c in range(self.n_cols)
        ]

        # Sample 2 distinct terminal states reproducibly
        rng = random.Random(seed)
        sampled_terminals = rng.sample(self._all_states, k=2)
        self._terminal_states: list[tuple[int, int]] = sorted(sampled_terminals)

        # Precompute non-terminal states
        terminal_set = set(self._terminal_states)
        self._non_terminal_states: list[tuple[int, int]] = [
            s for s in self._all_states if s not in terminal_set
        ]

    @property
    def states(self) -> list[tuple[int, int]]:
        """Return all grid states including terminal states."""
        return self._all_states.copy()

    @property
    def non_terminal_states(self) -> list[tuple[int, int]]:
        """
        Return states that DP algorithms should loop over and update.

        Terminal states are excluded because their value is fixed at 0.0 by convention.
        """
        return self._non_terminal_states.copy()

    @property
    def terminal_states(self) -> list[tuple[int, int]]:
        """Return the sampled terminal states."""
        return self._terminal_states.copy()

    def is_terminal(self, state: tuple[int, int]) -> bool:
        """Check if a state is terminal."""
        return state in self._terminal_states

    def step(self, state: tuple[int, int], action: str) -> tuple[tuple[int, int], float]:
        """
        Execute a deterministic transition from `state` taking `action`.

        Parameters
        ----------
        state : tuple[int, int]
            Current (row, col) state.
        action : str
            Action to execute ("up", "down", "left", "right").

        Returns
        -------
        next_state : tuple[int, int]
            Resulting (row, col) state after action execution.
        reward : float
            Constant reward of -1.0 for every transition.

        Raises
        ------
        ValueError
            If action is invalid, state is not on grid, or state is terminal.
        """
        if action not in self.ACTIONS:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {self.ACTIONS}."
            )
        if state not in self._all_states:
            raise ValueError(
                f"Invalid state {state}. Must be within grid bounds (0..{self.n_rows-1}, 0..{self.n_cols-1})."
            )
        if self.is_terminal(state):
            raise ValueError(
                f"Cannot call step() from terminal state {state}. DP algorithms must exclude terminal states from updates."
            )

        r, c = state
        if action == "up":
            next_r, next_c = max(0, r - 1), c
        elif action == "down":
            next_r, next_c = min(self.n_rows - 1, r + 1), c
        elif action == "left":
            next_r, next_c = r, max(0, c - 1)
        elif action == "right":
            next_r, next_c = r, min(self.n_cols - 1, c + 1)

        return (next_r, next_c), -1.0

    def __repr__(self) -> str:
        return (
            f"GridWorld(n_rows={self.n_rows}, n_cols={self.n_cols}, "
            f"terminal_states={self._terminal_states}, seed={self.seed})"
        )
