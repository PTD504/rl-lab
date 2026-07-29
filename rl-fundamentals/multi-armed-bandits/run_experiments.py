import numpy as np
from bandit_engine import BaseArm, KArmedBandit

def run_single_run(bandit, agent, n_steps, non_stationary=False):
    """
    Single run of the agent in the bandit problem in n_steps
    
    Returns:
        rewards - (n_steps,): received reward each step
        optimal_actions - (n_steps,): 1 if the chosen action is optimal, 0 otherwise
    """

    rewards = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps)

    for step in range(n_steps):
        action = agent.select_action()
        reward = bandit.pull(action)

        is_optimal = (action == bandit.optimal_action)
        optimal_actions[step] = is_optimal

        agent.update(action, reward)
        rewards[step] = reward

        if non_stationary:
            bandit.random_walk()
    
    return rewards, optimal_actions


def run_bandit_experiment(
    agent_factory,
    agent_params,
    bandit,
    experimental_seeds: list[int] = None,
    n_runs: int = 2000,
    n_steps: int = 1000,
    non_stationary: bool = False
):
    """
    Run the bandit experiment with a specific agent in n_runs independent time.
    """
    all_rewards = np.zeros((n_runs, n_steps))
    all_optimal = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        seed = experimental_seeds[run]
        bandit.reset(seed=seed)

        agent_setup_rng = np.random.default_rng(seed)
        agent_seed = int(agent_setup_rng.integers(0, 1e9))

        current_params = agent_params.copy()
        current_params["seed"] = agent_seed

        agent = agent_factory(**current_params)

        rewards, optimal_actions = run_single_run(bandit, agent, n_steps, non_stationary)

        all_rewards[run] = rewards
        all_optimal[run] = optimal_actions
    
    return all_rewards, all_optimal

def compare_agents(
    bandit_arms,
    agent_configs,
    random_seed,
    n_runs,
    n_steps,
    non_stationary
):
    seed_generator = np.random.default_rng(random_seed)
    experimental_seeds = [int(seed_generator.integers(0, 1e9)) for _ in range(n_runs)]

    all_rewards = {cfg["label"]: np.zeros((n_runs, n_steps)) for cfg in agent_configs}
    all_optimal = {cfg["label"]: np.zeros((n_runs, n_steps)) for cfg in agent_configs}

    shared_bandit = KArmedBandit(arms=bandit_arms)

    for cfg in agent_configs:
        rewards, optimal_actions = run_bandit_experiment(
            agent_factory=cfg["agent_type"],
            agent_params=cfg["agent_config"],
            bandit=shared_bandit,
            experimental_seeds=experimental_seeds,
            n_runs=n_runs,
            n_steps=n_steps,
            non_stationary=non_stationary
        )

        all_rewards[cfg["label"]] = rewards
        all_optimal[cfg["label"]] = optimal_actions

    results = {
        label: (all_rewards[label].mean(axis=0), all_optimal[label].mean(axis=0) * 100) for label in all_rewards
    }

    return results