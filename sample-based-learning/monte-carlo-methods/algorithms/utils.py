import numpy as np

def generate_episode(env, pi, rng):
    """
    Generate an episode using the given policy.
    """
    episode = []
    state, _ = env.reset()
    done = False

    while not done:
        action = rng.choice(a=env.action_space.n, p=pi[state])
        s_next, r, terminated, truncated, _ = env.step(action)

        episode.append((state, action, r))

        state = s_next
        done = terminated or truncated

    return episode

def initialize_policy(env, rng):
    """
    Initialize a policy with random probabilities for each state.
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    policy = {}
    for state in range(num_states):
        probs = rng.uniform(1e-5, 1.0, size=num_actions)
        policy[state] = probs / np.sum(probs)

    return policy