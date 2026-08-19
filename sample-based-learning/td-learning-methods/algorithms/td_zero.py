import numpy as np
from utils import configure_terminal_state_value

def td_zero(env, pi, step_size, gamma, max_episodes, seed):
    rng = np.random.default_rng(seed)

    # Initialize V
    V = rng.random(size=(env.observation_space.n,))
    # V(terminal) = 0.0 for all terminal states
    configure_terminal_state_value(env, V)

    for _ in range(max_episodes):
        # In each iteration, we start at the starting state s
        s, _ = env.reset()
        is_terminated = False

        while not is_terminated:
            # During each step of the episode, we choose the action in the current state s
            a = rng.choice(a=env.action_space.n, p=pi[s])
            # Take the chosen action and get the next state
            s_next, r, terminated, truncated, _ = env.step(a)

            # Update the V value function
            V[s] += step_size * (r + gamma * V[s_next] - V[s])
            # Assign the next state to s to continue the next step with the new state
            s = s_next

            is_terminated = terminated or truncated

    return V