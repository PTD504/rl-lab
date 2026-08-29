# RL Lab

A reinforcement learning study repository implemented from scratch in Python/NumPy. The project follows the core ideas from Sutton & Barto and focuses on understanding the algorithms by building them directly, rather than relying on a high-level library.

The repository currently covers fundamentals from bandits and dynamic programming through Monte Carlo methods and temporal-difference learning.

## Topics covered

- Multi-armed bandits
  - epsilon-greedy
  - non-stationary bandits with constant step size
  - optimistic initial values
  - UCB action selection
- MDP foundations
  - Markov decision processes
  - Bellman equations
  - value and policy concepts
- Dynamic programming
  - policy evaluation
  - policy iteration
  - truncated policy iteration
  - value iteration
- Monte Carlo methods
  - on-policy prediction
  - on-policy control
  - off-policy prediction with importance sampling
  - off-policy control with weighted importance sampling
- Temporal-difference learning
  - TD(0)
  - SARSA
  - Q-learning
  - Expected SARSA
  - Double Q-learning
  - Double Expected SARSA

## Project structure

```bash
rl-lab/
├── README.md
├── LICENSE
├── CURRENT_STATUS.md
├── rl-fundamentals/
│   ├── dynamic-programming/
│   │   ├── algorithms/
│   │   │   ├── __init__.py
│   │   │   ├── policy_evaluation.py
│   │   │   ├── policy_iteration.py
│   │   │   └── value_iteration.py
│   │   ├── envs/
│   │   │   ├── __init__.py
│   │   │   └── grid_world.py
│   │   ├── notebooks/
│   │   │   ├── pi_and_truncated_pi_comparison.ipynb
│   │   │   ├── pi_and_vi_comparison.ipynb
│   │   │   ├── policy_evaluation.ipynb
│   │   │   ├── policy_iteration.ipynb
│   │   │   └── value_iteration.ipynb
│   │   ├── notes.md
│   │   └── utils.py
│   ├── mdp-foundations/
│   │   └── MDPs.md
│   └── multi-armed-bandits/
│       ├── agents.py
│       ├── bandit_engine.py
│       ├── notebooks/
│       │   ├── 01_epsilon_greedy_comparison.ipynb
│       │   ├── 02_nonstationary.ipynb
│       │   ├── 03_optimistic_initial_values.ipynb
│       │   └── 04_ucb_action_selection.ipynb
│       ├── run_experiments.py
│       └── utils.py
├── sample-based-learning/
│   ├── monte-carlo-methods/
│   │   ├── algorithms/
│   │   │   ├── __init__.py
│   │   │   ├── mc_control.py
│   │   │   ├── mc_prediction.py
│   │   │   ├── off_policy_mc_control.py
│   │   │   ├── off_policy_mc_prediction.py
│   │   │   └── utils.py
│   │   ├── notebooks/
│   │   │   ├── mc_control.ipynb
│   │   │   ├── onp_offp_mc_control_comparison.ipynb
│   │   │   └── onp_offp_mc_prediction_comparison.ipynb
│   │   ├── notes.md
│   │   └── utils/
│   │       └── render_utils.py
│   └── td-learning-methods/
│       └── algorithms/
│           ├── double_expected_sarsa.py
│           ├── double_q_learning.py
│           ├── expected_sarsa.py
│           ├── q_learning.py
│           ├── sarsa.py
│           ├── td_zero.py
│           └── utils.py
```

## Current progress

- [x] k-armed bandits
- [x] MDP foundations
- [x] dynamic programming
- [x] Monte Carlo methods
- [x] temporal-difference learning

## Notes

This repo is primarily a learning lab and reference implementation. The goal is not just to run a model, but to make each algorithm explicit and inspectable: how the update rule works, what assumptions are made, and how different methods compare in practice.

The helper scripts are used to visualize behavior and compare algorithmic differences across tasks and environments.

## Environment

This project is designed to run with a Python environment containing the standard scientific stack used in the experiments, especially NumPy and Jupyter-compatible tooling.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
