import numpy as np

def policy_evaluation(
    env,
    pi,
    max_backup: int = 100000,
    gamma: float = 0.9,
    V: np.ndarray | None = None,
    seed: int | None = None,
    initial_backups: int = 0,
    truncated_steps: int = None,
) -> tuple[np.ndarray, int, dict[str, list[float]]]:
    """
    In-place implementation tracking total Bellman updates and convergence history.
    """
    if V is None:
        rng = np.random.default_rng(seed)
        V = rng.random((env.n_rows, env.n_cols))

    theta = 1e-6

    for terminal_state in env.terminal_states:
        V[terminal_state] = 0.0

    backups_count = initial_backups
    max_total_backup = initial_backups + max_backup
    steps = 0

    history: dict[str, list[float]] = {
        "backup_count": [],
        "delta": [],
    }

    while backups_count < max_total_backup:

        if truncated_steps is not None and steps >= truncated_steps:
            break

        delta = 0.0

        for s in env.non_terminal_states:
            if backups_count >= max_total_backup:
                break
            v = V[s]
            bellman_update(env, pi, s, V, gamma)
            backups_count += 1
            delta = max(delta, abs(v - V[s]))

        steps += 1
        history["backup_count"].append(backups_count)
        history["delta"].append(delta)

        if delta < theta:
            break

    eval_backups = backups_count - initial_backups
    return V, eval_backups, history




def bellman_update(env, pi, s, V, gamma):
    cumulative_reward = 0.0

    for action in pi[s]:
        s_prime, reward = env.step(s, action)
        cumulative_reward += 1 / len(pi[s]) * (reward + gamma * V[s_prime])

    V[s] = cumulative_reward