import numpy as np
from algorithms.utils import configure_terminal_state_value, choose_action_egreedy

def q_learning(env, step_size, gamma, epsilon, epsilon_decay_rate, max_episodes, seed):
    """
    Q-Learning algorithm implementation
    """
    rng = np.random.default_rng(seed)

    # Initialize Q value function
    Q = rng.random(size=(env.observation_space.n, env.action_space.n))
    # Set Q(terminal, .) = 0.0
    configure_terminal_state_value(env, Q)

    history = {
        "num_steps": [],
        "convergence_speed": [],
        "avg_reward": []
    }

    for _ in range(max_episodes):
        # Initialize the starting state
        s, _ = env.reset()
        is_terminated = False

        step = 0
        avg_reward = 0.0

        Q_old = Q.copy()

        while not is_terminated:
            # In each time step of the episode, choose the action a using epsilon greedy
            a = choose_action_egreedy(Q[s], epsilon, rng)
            # Take the action a and get the next state, as well as the reward
            s_next, r, terminated, truncated, _ = env.step(a)

            # Update the Q value function with TD target is the sum of immediate reward and the maximum estimated Q value function of the next state
            Q[s, a] += step_size * (r + gamma * np.max(Q[s_next]) - Q[s, a])
            s = s_next

            is_terminated = terminated or truncated
            step += 1
            avg_reward += r

        avg_reward /= step

        epsilon *= epsilon_decay_rate
        epsilon = max(epsilon, 0.01)

        history["num_steps"].append(step)
        history["convergence_speed"].append(np.max(np.abs(Q - Q_old)))
        history["avg_reward"].append(avg_reward)

    return Q, history
