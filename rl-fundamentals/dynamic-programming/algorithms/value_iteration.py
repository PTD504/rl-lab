import numpy as np

def value_iteration(env, V=None, max_backup: int = 100000, gamma: float = 0.9, seed: int | None = 84):
    if V is None:
        rng = np.random.default_rng(seed)
        V = rng.random((env.n_rows, env.n_cols))


    for s in env.terminal_states:
        V[s] = 0.0

    theta = 1e-6
    actions = getattr(env, "ACTIONS", ["up", "down", "left", "right"])

    total_backups = 0
    history: dict[str, list[float]] = {
        "backup_count": [],
        "delta": [],
    }

    while total_backups < max_backup:
        delta = 0.0

        for s in env.non_terminal_states:
            if total_backups >= max_backup:
                break
            v = V[s]
            q_vals = [compute_q_value(env, V, s, a, gamma) for a in actions]
            V[s] = max(q_vals)
            total_backups += 1
            delta = max(delta, abs(v - V[s]))

        history["backup_count"].append(total_backups)
        history["delta"].append(delta)

        if delta < theta or total_backups >= max_backup:
            break

    pi = extract_policy(env, V, actions, gamma)

    return V, pi, history


def extract_policy(env, V, actions, gamma):
    pi = {}

    for s in env.non_terminal_states:
        q = [compute_q_value(env, V, s, a, gamma) for a in actions]
        max_q = max(q)
        optimal_actions = [a for a, q_val in zip(actions, q) if abs(q_val - max_q) < 1e-6]
        pi[s] = optimal_actions

    return pi


def compute_q_value(env, V, s, a, gamma):
    s_prime, reward = env.step(s, a)

    q_s_a = reward + gamma * V[s_prime] # p(s', r | s, a) = 1 in this grid world setting

    return q_s_a