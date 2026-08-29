import numpy as np
from algorithms.utils import configure_terminal_state_value, choose_action_egreedy

def expected_sarsa(env, step_size, gamma, epsilon, epsilon_decay_rate, max_episodes, seed):
    """
    Expected SARSA algorithm implementation
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

            # Calculate the expected value for how likely each action is under the current policy (epsilon greedy policy w.r.t Q) in the next state s_next
            E = calculate_expected_q_value_at_state_s(Q[s_next], epsilon)
            # Update the Q value
            Q[s, a] += step_size * (r + gamma * E - Q[s, a])
            
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

def calculate_expected_q_value_at_state_s(q_values, epsilon):
    """
    In expected sarsa algorithm, the TD target is the sum of the immediate reward and the expected value of chosen actions in a specific state — how likely each action is under the current policy
    But we don't have the policy, how can we compute this value?
    Remember that at a specific state s, we take an action follows the epsilon greedy strategy w.r.t our Q value function.

    -> What does that mean?
    Well, as we all know, in epsilon greedy, the probability of each action is:
    - epsilon / total_number_of_actions if the current action is not an optimal action according to the current Q value function
    - 1 - epsilon + epsilon / total_number_of_actions if the current action is an optimal action (with the maximum estimated value of Q(s, a))

    We will use this to compute the expected value
    """
    # Get the list of probabilities for optimal actions.
    # For example, if we have a total of 6 actions with three of them have the maximum value of Q(s, a), then the is_max and policy_probs should be:
    # is_max = [True, False, False, True, True, False]
    # policy_probs = [1/3 * (1 - epsilon) + epsilon / 6, epsilon / 6, epsilon / 6, 1/3 * (1 - epsilon) + epsilon / 6, 1/3 * (1 - epsilon) + epsilon / 6, epsilon / 6]
    num_actions = q_values.shape[0]
    is_max = np.isclose(q_values, np.max(q_values))

    policy_probs = np.full(num_actions, epsilon / num_actions)
    policy_probs[is_max] += (1 - epsilon) / np.sum(is_max)
    
    return float(np.dot(policy_probs, q_values))