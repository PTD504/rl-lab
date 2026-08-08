# rl-lab
Core reinforcement learning algorithms, implemented from scratch — from k-armed bandits all the way to actor-critic methods — following Sutton & Barto's *Reinforcement Learning: An Introduction*.
The goal isn't just passing assignments. Each algorithm here is built by hand in NumPy/Python to actually understand *why* it works, not just call a library and watch numbers go up. Where it makes sense, results come with plots — learning curves, comparisons, failure cases — because "it trained" is weaker than "here's what it learned and why".

## Structure

Project tree:

```
/
└── rl-fundamentals/
	├── dynamic-programming/    # DP examples and tooling
	│   ├── algorithms/         # DP algorithm implementations (policy evaluation, policy iteration, value iteration)
	│   ├── envs/               # MDP environments (deterministic GridWorld)
	│   ├── notebooks/          # example notebooks for running and visualizing DP experiments
	│   └── utils.py            # plotting and visualization helpers
	├── mdp-foundations/       # conceptual notes on MDPs and background material
	└── multi-armed-bandits/    # bandit algorithms, experiments, and notebooks

```

## Progress

- [x] **k-armed bandits** — ε-greedy, non-stationary tracking (constant step size), optimistic initial values, UCB action selection

 - [x] **dynamic programming (DP)** — basic DP tooling including a deterministic `GridWorld`, implementations of policy evaluation, policy iteration, and value iteration, plus a few notebooks and plotting helpers. A short note about performance: policy iteration's initial evaluation phase can be costly; a truncated-evaluation variant is included as an efficiency experiment.
