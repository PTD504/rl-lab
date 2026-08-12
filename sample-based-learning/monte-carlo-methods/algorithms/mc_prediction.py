import numpy as np

from algorithms.utils import generate_episode

def mc_prediction(env, pi, gamma, max_episodes, seed):
    """
    Implementation of the First Visit MC Prediction
    """
    rng = np.random.default_rng(seed)
    
    V = rng.random(env.observation_space.n)
    N = np.zeros(env.observation_space.n, dtype=int)

    history = {
        'returns': [],
        'max_v_change': [],
        'mean_v_change': []
    }

    for _ in range(max_episodes):
        V_old = V.copy()
        episode = generate_episode(env, pi, rng)
        ep_return = sum(step[2] for step in episode)
        G = 0.0

        first_visits = {}
        for step, (s, _, _) in enumerate(episode):
            if s not in first_visits:
                first_visits[s] = step

        for step in reversed(range(len(episode))):
            s, _, r = episode[step]
            G = gamma * G + r

            if first_visits[s] == step:
                N[s] += 1
                V[s] += (G - V[s]) / N[s]

        v_diff = np.abs(V - V_old)
        history['returns'].append(ep_return)
        history['max_v_change'].append(float(np.max(v_diff)))
        history['mean_v_change'].append(float(np.mean(v_diff)))

    return V, history