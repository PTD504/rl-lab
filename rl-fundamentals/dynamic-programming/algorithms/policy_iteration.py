import random
import numpy as np
from algorithms.policy_evaluation import policy_evaluation


def policy_iteration(env, pi=None, max_backup: int = 100000, gamma: float = 0.9, seed: int | None = 84):
    if pi is None:
        pi = initialize_pi_policy(env, seed=seed)

    V = None
    actions = getattr(env, "ACTIONS", ["up", "down", "left", "right"])
    total_backups = 0

    history: dict[str, list[float]] = {
        "backup_count": [],
        "delta": [],
    }

    while total_backups < max_backup:

        # Policy Evaluation (tracks Bellman updates & convergence history)
        remaining_backups = max_backup - total_backups
        V, eval_backups, eval_history = policy_evaluation(
            env, pi, remaining_backups, gamma, V=V, seed=seed, initial_backups=total_backups
        )
        total_backups += eval_backups
        history["backup_count"].extend(eval_history["backup_count"])
        history["delta"].extend(eval_history["delta"])

        # Policy Improvement (does NOT count towards Bellman updates)
        policy_stable = True

        for s in env.non_terminal_states:
            old_actions = pi[s]
            q = [compute_q_value(env, V, s, a, gamma) for a in actions]
            max_q = max(q)
            new_actions = [a for a, q_val in zip(actions, q) if abs(q_val - max_q) < 1e-6]
            pi[s] = new_actions
            if set(old_actions) != set(new_actions):
                policy_stable = False

        if policy_stable or total_backups >= max_backup:
            break

    return V, pi, history




def compute_q_value(env, V, s, a, gamma):
    s_prime, reward = env.step(s, a)

    q_s_a = reward + gamma * V[s_prime] # p(s', r | s, a) = 1 in this grid world setting

    return q_s_a


# Helper function to initialize the pi policy
def initialize_pi_policy(env, seed: int | None = None) -> dict[tuple[int, int], list[str]]:
    """
    Initialize a random policy for non-terminal states in GridWorld.

    For each non-terminal state:
    1. Randomly sample the number of actions k between 1 and 4 (1 <= k <= 4).
    2. Randomly sample k distinct actions from ["up", "down", "left", "right"].

    Parameters
    ----------
    env : GridWorld
        GridWorld environment instance.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    pi : dict[tuple[int, int], list[str]]
        Policy dictionary mapping each non-terminal state to a list of actions.
    """
    rng = random.Random(seed)
    actions = getattr(env, "ACTIONS", ["up", "down", "left", "right"])

    pi = {}
    for s in env.non_terminal_states:
        # Sample number of actions k between 1 and 4
        k = rng.randint(1, len(actions))
        # Randomly sample k distinct actions from available actions
        pi[s] = sorted(rng.sample(actions, k))

    return pi



