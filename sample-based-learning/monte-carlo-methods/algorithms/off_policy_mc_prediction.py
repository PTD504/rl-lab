import numpy as np
from algorithms.utils import initialize_policy, generate_episode

def offp_mc_prediction(env, pi, gamma, max_episodes, seed):
    """
    Implementation of the Off-policy MC Prediction
    """
    rng = np.random.default_rng(seed)

    Q = rng.random((env.observation_space.n, env.action_space.n))
    C = np.zeros((env.observation_space.n, env.action_space.n))

    history = {
        'returns': [],
        'max_q_change': [],
        'mean_q_change': []
    }

    for _ in range(max_episodes):
        Q_old = Q.copy()
        b = initialize_policy(env, rng)
        episode = generate_episode(env, b, rng)
        ep_return = sum(step[2] for step in episode)

        G = 0.0
        W = 1.0

        for step in reversed(range(len(episode))):
            s, a, r = episode[step]

            G = gamma * G + r
            C[s, a] += W
            Q[s, a] += W * (G - Q[s, a]) / C[s, a]

            W *= pi[s][a] / b[s][a]

            if W == 0.0:
                break

        q_diff = np.abs(Q - Q_old)
        history['returns'].append(ep_return)
        history['max_q_change'].append(float(np.max(q_diff)))
        history['mean_q_change'].append(float(np.mean(q_diff)))

    return Q, history