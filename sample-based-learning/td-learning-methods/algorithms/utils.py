import numpy as np

def configure_terminal_state_value(env, Tabular):
    """
    Helper function to set V(terminal) or Q(terminal, .) to zero value
    """
    base = env.unwrapped
    
    for s in range(base.observation_space.n):
        is_terminal = True
        for a in range(base.action_space.n):
            transitions = base.P[s][a]
            if any(not terminated for _, _, _, terminated in transitions):
                is_terminal = False
                break

        if is_terminal:
            if Tabular.ndim == 1:
                Tabular[s] = 0.0
            else:
                Tabular[s, :] = 0.0

def choose_action_egreedy(q_values, epsilon, rng):
    """
    Epsilon Greedy Action Selection with ties broken arbitrarily
    """
    if rng.random() < epsilon:
        return rng.choice(a=len(q_values))
    else:
        opt_actions = np.flatnonzero(q_values == np.max(q_values))
        return rng.choice(a=opt_actions)

def extract_greedy_policy_wrt_Q(Q):
    """
    Helper function to extract a greedy policy w.r.t Q value
    """
    pi = {}

    for s in range(len(Q)):
        is_max = np.isclose(Q[s], np.max(Q[s]))
        pi[s] = (is_max / np.sum(is_max)).tolist()

    return pi