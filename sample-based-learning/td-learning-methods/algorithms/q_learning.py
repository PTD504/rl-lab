import numpy as np
from utils import configure_terminal_state_value, choose_action_egreedy

def q_learning(env, step_size, gamma, epsilon, max_episodes, seed):
    """
    Q-Learning algorithm implementation
    """
    rng = np.random.default_rng(seed)

    # Initialize Q value function
    Q = rng.random(size=(env.observation_space.n, env.action_space.n))
    # Set Q(terminal, .) = 0.0
    configure_terminal_state_value(env, Q)

    for _ in range(max_episodes):
        # Initialize the starting state
        s, _ = env.reset()
        is_terminated = False

        while not is_terminated:
            # In each time step of the episode, choose the action a using epsilon greedy
            a = choose_action_egreedy(Q[s], epsilon, rng)
            # Take the action a and get the next state, as well as the reward
            s_next, r, terminated, truncated, _ = env.step(a)

            # Update the Q value function with TD target is the sum of immediate reward and the maximum estimated Q value function of the next state
            Q[s, a] += step_size * (r + gamma * np.max(Q[s_next]) - Q[s, a])
            s = s_next

            is_terminated = terminated or truncated

    return Q
