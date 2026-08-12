import numpy as np
from algorithms.utils import generate_episode, initialize_policy

def mc_control(env, gamma, max_episodes, epsilon, seed: int = 84):
    """
    epsilon-soft MC control
    """
    env.action_space.seed(seed)
    rng = np.random.default_rng(seed=seed)

    # Initialize the epsilon-soft policy
    pi = initialize_policy(env, rng)

    Q = rng.random(size=(env.observation_space.n, env.action_space.n))
    N = np.zeros((env.observation_space.n, env.action_space.n), dtype=int)

    history = {
        'returns': [],
        'lengths': [],
        'max_q_change': []
    }

    for _ in range(max_episodes):
        Q_old = Q.copy()
        episode = generate_episode(env, pi, rng)
        ep_return = sum(step[2] for step in episode)
        ep_len = len(episode)
        G = 0.0

        first_visits = {}

        for step, (s, a, r) in enumerate(episode):
            if (s, a) not in first_visits:
                first_visits[s, a] = step

        for step in reversed(range(len(episode))):
            s, a, r = episode[step]
            G = gamma * G + r

            if first_visits[s, a] == step:
                N[s, a] += 1
                # Update the Q value of state s, action a
                Q[s, a] += (G - Q[s, a]) / N[s, a]

                # Make the behavior policy epsilon-greedy w.r.t Q
                opt_a = np.argmax(Q[s])
                num_actions = env.action_space.n

                for action in range(num_actions):
                    pi[s][action] = 1 - epsilon + epsilon / num_actions if action == opt_a else epsilon / num_actions

        q_diff = np.abs(Q - Q_old)
        history['returns'].append(ep_return)
        history['lengths'].append(ep_len)
        history['max_q_change'].append(float(np.max(q_diff)))

    return Q, pi, history