import numpy as np
from algorithms.utils import generate_episode, initialize_policy

def offp_mc_control(env, gamma, max_episodes, seed):
    rng = np.random.default_rng(seed)

    Q = rng.random((env.observation_space.n, env.action_space.n))
    C = np.zeros((env.observation_space.n, env.action_space.n))

    pi = {}
    for state in range(env.observation_space.n):
        best_action = np.argmax(Q[state])
        pi[state] = [1.0 if a == best_action else 0.0 for a in range(env.action_space.n)]

    history = {
        'returns': [],
        'lengths': [],
        'max_q_change': []
    }

    for _ in range(max_episodes):
        Q_old = Q.copy()
        b = initialize_policy(env, rng)
        episode = generate_episode(env, b, rng)
        ep_return = sum(step[2] for step in episode)
        ep_len = len(episode)

        G = 0.0
        W = 1.0

        for step in reversed(range(len(episode))):
            s, a, r = episode[step]

            G = gamma * G + r
            C[s, a] += W
            Q[s, a] += W * (G - Q[s, a]) / C[s, a]

            best_action = np.argmax(Q[s])
            pi[s] = [1.0 if a == best_action else 0.0 for a in range(env.action_space.n)]

            if a != best_action:
                break

            W /= b[s][a]

        q_diff = np.abs(Q - Q_old)
        history['returns'].append(ep_return)
        history['lengths'].append(ep_len)
        history['max_q_change'].append(float(np.max(q_diff)))

    return Q, pi, history
