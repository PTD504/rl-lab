import numpy as np
from algorithms.utils import configure_terminal_state_value, choose_action_egreedy

def sarsa(env, step_size, gamma, epsilon, epsilon_decay_rate, max_episodes, seed):
    """
    SARSA algorithm implementation
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
    # history["num_steps"] = a list of size max_episodes, each element is the number of step the agent take in that
    # epsiode from start to the end
    # history["convergence_speed"] = a list of size max_episodes, each element is the difference of the Q value function
    # in the two consecutive episodes
    # history["avg_reward"] = a list of size max_episodes, each element is the average reward of the agent in the episode

    for _ in range(max_episodes):
        # At each iteration, initialize s, choose an action from the s state using epsilon greedy
        s, _ = env.reset()
        a = choose_action_egreedy(Q[s], epsilon, rng)
        is_terminated = False

        step = 0
        avg_reward = 0.0

        Q_old = Q.copy()

        while not is_terminated:
            # Take action a in state s to get the next state, then using the epsilon greedy to choose the next action a' from the next state s'
            s_next, r, terminated, truncated, _ = env.step(a)
            a_next = choose_action_egreedy(Q[s_next], epsilon, rng)

            # Update the Q value function with TD target is the sum of immediate reward and the estimated Q value function of the next state-action pair
            Q[s, a] += step_size * (r + gamma * Q[s_next, a_next] - Q[s, a])
            s = s_next
            a = a_next

            is_terminated = terminated or truncated
            step += 1
            avg_reward += r

        avg_reward /= step

        # Decay epsilon
        epsilon = epsilon_decay_rate * epsilon if epsilon_decay_rate is not None else epsilon
        epsilon = max(epsilon, 0.01)

        history["num_steps"].append(step)
        history["convergence_speed"].append(np.max(np.abs(Q - Q_old)))
        history["avg_reward"].append(avg_reward)

    return Q, history