import numpy as np
from algorithms.utils import configure_terminal_state_value, choose_action_egreedy

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

    # Initialize history tracking
    history = {
        # ===== A-B MDP SPECIFIC METRICS (can be removed for other environments) =====
        "num_steps": [],
        "action_left_count": [],  # Count of Left action chosen from State A
        "cumulative_reward": [],  # Cumulative reward per episode
        "avg_reward": []  # Average reward per episode
        # ===== END A-B MDP SPECIFIC METRICS =====
    }

    for _ in range(max_episodes):
        # Initialize the starting state
        s, _ = env.reset()
        is_terminated = False

        # ===== A-B MDP SPECIFIC METRICS =====
        step = 0
        cumulative_reward = 0.0
        avg_reward = 0.0
        action_left_count = 0
        # ===== END A-B MDP SPECIFIC METRICS =====
        
        Q1_old = Q1.copy()
        Q2_old = Q2.copy()

        while not is_terminated:
            # Choose the action using epsilon greedy with the sum of both Q value functions
            a = choose_action_egreedy(Q1[s] + Q2[s], epsilon, rng)
            
            # ===== A-B MDP SPECIFIC METRICS =====
            # Track if Left action (0) is chosen from State A (0)
            if s == 0 and a == 0:
                action_left_count += 1
            # ===== END A-B MDP SPECIFIC METRICS =====
            
            # Take the action a and get the next state, as well as the reward
            s_next, r, terminated, truncated, _ = env.step(a)

            # Randomly choose one Q value function to update and use the other one to estimate the next state value
            if rng.random() < 0.5:
                Q1[s, a] += step_size * (r + gamma * Q2[s_next, np.argmax(Q1[s_next])] - Q1[s, a])
            else:
                Q2[s, a] += step_size * (r + gamma * Q1[s_next, np.argmax(Q2[s_next])] - Q2[s, a])

            s = s_next
            is_terminated = terminated or truncated
            # ===== A-B MDP SPECIFIC METRICS =====
            step += 1
            cumulative_reward += r
            avg_reward += r
            # ===== END A-B MDP SPECIFIC METRICS =====

        # ===== A-B MDP SPECIFIC METRICS =====
        avg_reward /= step

        # Record episode history
        history["num_steps"].append(step)
        history["action_left_count"].append(action_left_count)
        history["cumulative_reward"].append(cumulative_reward)
        history["avg_reward"].append(avg_reward)
        # ===== END A-B MDP SPECIFIC METRICS =====

    return Q1, Q2, history