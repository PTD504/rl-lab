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
        "avg_reward": [],
        # ===== A-B MDP SPECIFIC METRICS (can be removed for other environments) =====
        "action_left_count": [],  # Count of Left action chosen from State A
        "cumulative_reward": []  # Cumulative reward per episode
        # ===== END A-B MDP SPECIFIC METRICS =====
    }

    for _ in range(max_episodes):
        # Initialize the starting state
        s, _ = env.reset()
        is_terminated = False

        step = 0
        avg_reward = 0.0
        # ===== A-B MDP SPECIFIC METRICS =====
        cumulative_reward = 0.0
        action_left_count = 0
        # ===== END A-B MDP SPECIFIC METRICS =====

        Q_old = Q.copy()

        while not is_terminated:
            # In each time step of the episode, choose the action a using epsilon greedy
            a = choose_action_egreedy(Q[s], epsilon, rng)
            
            # ===== A-B MDP SPECIFIC METRICS =====
            # Track if Left action (0) is chosen from State A (0)
            if s == 0 and a == 0:
                action_left_count += 1
            # ===== END A-B MDP SPECIFIC METRICS =====
            
            # Take the action a and get the next state, as well as the reward
            s_next, r, terminated, truncated, _ = env.step(a)

            # Update the Q value function with TD target is the sum of immediate reward and the maximum estimated Q value function of the next state
            Q[s, a] += step_size * (r + gamma * np.max(Q[s_next]) - Q[s, a])
            s = s_next

            is_terminated = terminated or truncated
            step += 1
            avg_reward += r
            # ===== A-B MDP SPECIFIC METRICS =====
            cumulative_reward += r
            # ===== END A-B MDP SPECIFIC METRICS =====

        avg_reward /= step

        epsilon = epsilon_decay_rate * epsilon if epsilon_decay_rate is not None else epsilon
        epsilon = max(epsilon, 0.01)

        history["num_steps"].append(step)
        history["convergence_speed"].append(np.max(np.abs(Q - Q_old)))
        history["avg_reward"].append(avg_reward)
        # ===== A-B MDP SPECIFIC METRICS =====
        history["action_left_count"].append(action_left_count)
        history["cumulative_reward"].append(cumulative_reward)
        # ===== END A-B MDP SPECIFIC METRICS =====

    return Q, history
