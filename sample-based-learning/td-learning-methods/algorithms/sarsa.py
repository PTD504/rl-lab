import numpy as np
from utils import configure_terminal_state_value, choose_action_egreedy

def sarsa(env, step_size, gamma, epsilon, max_episodes, seed):
    """
    SARSA algorithm implementation
    """
    rng = np.random.default_rng(seed)

    # Initialize Q value function
    Q = rng.random(size=(env.observation_space.n, env.action_space.n))
    # Set Q(terminal, .) = 0.0
    configure_terminal_state_value(env, Q)

    for _ in range(max_episodes):
        # At each iteration, initialize s, choose an action from the s state using epsilon greedy
        s, _ = env.reset()
        a = choose_action_egreedy(Q[s], epsilon, rng)
        is_terminated = False

        while not is_terminated:
            # Take action a in state s to get the next state, then using the epsilon greedy to choose the next action a' from the next state s'
            s_next, r, terminated, truncated, _ = env.step(a)
            a_next = choose_action_egreedy(Q[s_next], epsilon, rng)

            # Update the Q value function with TD target is the sum of immediate reward and the estimated Q value function of the next state-action pair
            Q[s, a] += step_size * (r + gamma * Q[s_next, a_next] - Q[s, a])
            s = s_next
            a = a_next

            is_terminated = terminated or truncated

    return Q