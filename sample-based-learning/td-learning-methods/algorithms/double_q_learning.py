import numpy as np
from utils import configure_terminal_state_value, choose_action_egreedy

def double_q_learning(env, step_size, gamma, epsilon, max_episodes, seed):
    """
    Double Q-Learning algorithm implementation
    """
    rng = np.random.default_rng(seed)

    # Initialize the two Q value functions
    Q1 = rng.random(size=(env.observation_space.n, env.action_space.n))
    Q2 = rng.random(size=(env.observation_space.n, env.action_space.n))

    # Set Q(terminal, .) = 0.0 for both Q value functions
    configure_terminal_state_value(env, Q1)
    configure_terminal_state_value(env, Q2)

    for _ in range(max_episodes):
        # Initialize the starting state
        s, _ = env.reset()
        is_terminated = False

        while not is_terminated:
            # Choose the action using epsilon greedy with the sum of both Q value functions
            a = choose_action_egreedy(Q1[s] + Q2[s], epsilon, rng)
            # Take the action a and get the next state, as well as the reward
            s_next, r, terminated, truncated, _ = env.step(a)

            # Randomly choose one Q value function to update and use the other one to estimate the next state value
            if rng.random() < 0.5:
                Q1[s, a] += step_size * (r + gamma * Q2[s_next, np.argmax(Q1[s_next])] - Q1[s, a])
            else:
                Q2[s, a] += step_size * (r + gamma * Q1[s_next, np.argmax(Q2[s_next])] - Q2[s, a])

            s = s_next
            is_terminated = terminated or truncated

    return Q1, Q2