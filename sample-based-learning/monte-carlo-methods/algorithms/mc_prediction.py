import numpy as np

from algorithms.utils import generate_episode

def mc_prediction(env, pi, gamma, max_episodes, seed):
    """
    Implementation of the First Visit MC Prediction
    """
    rng = np.random.default_rng(seed)
    
    V = rng.random(env.observation_space.n)
    N = np.zeros(env.observation_space.n, dtype=int)

    for _ in range(max_episodes):
        episode = generate_episode(env, pi, rng)
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

    return V