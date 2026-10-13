# RL Lab

RL Lab is a reinforcement-learning portfolio project focused on understanding
the ideas that make an agent learn. The implementations are intentionally
small and inspectable so that the connection between the mathematics, the
update rules, and the agent's behavior remains clear.

The project follows the progression from simple bandit problems to
model-based planning, sample-based learning, function approximation, and deep
reinforcement learning. Each stage introduces a new assumption or removes an
old limitation. The purpose is not only to implement algorithms, but to
understand why their targets, policies, and stability properties differ.

## Project structure

```text
rl-lab/
├── rl-fundamentals/
│   ├── mdp-foundations/                  # States, returns, policies, values, Bellman equations
│   ├── multi-armed-bandits/              # Exploration, exploitation, and action-value estimates
│   └── dynamic-programming/              # Policy evaluation, policy iteration, and value iteration
├── sample-based-learning/
│   ├── monte-carlo-methods/              # Learning from complete episode returns
│   └── td-learning-methods/              # Bootstrapping, SARSA, Q-learning, and related methods
├── function-approximation/
│   ├── algorithms/
│   │   ├── linear-function-approximation/
│   │   └── non-linear-function-approximation/
│   └── notebooks/                        # Experiments beyond tabular value functions
├── deep-rl/
│   ├── dqns/                             # Deep value-based learning and its variants
│   ├── policy-gradient/                  # Direct policy optimization and actor-critic methods
│   ├── utils/                            # Shared models, replay buffers, and experiment helpers
│   ├── experiments/                      # Reproducible experiment guides
│   └── tests/                            # Focused correctness tests
└── LICENSE
```

## From value estimation to control

The first part of the repository builds the language of reinforcement
learning. Bandits isolate the exploration-exploitation problem without state
transitions. MDP foundations then introduce states, actions, rewards, returns,
policies, and value functions. These concepts provide the basis for reasoning
about both prediction and control.

Dynamic programming uses a known environment model to repeatedly apply the
Bellman equations. Policy evaluation estimates the value of a policy, while
policy improvement uses those estimates to make the policy greedier. Policy
iteration and value iteration show two closely related ways to approach an
optimal policy.

Monte Carlo methods remove the need for a model and learn from complete
episodes. They provide a direct estimate of returns, but updates must wait
until an episode ends and can have high variance. Temporal-difference methods
update earlier by bootstrapping from current estimates. Comparing TD(0),
SARSA, Q-learning, Expected SARSA, and related variants makes the distinction
between on-policy and off-policy learning concrete.

## Generalization and deep reinforcement learning

Tabular methods treat every state or state-action pair as independent.
Function approximation changes this by allowing related observations to share
parameters. This makes larger or continuous state spaces possible, but also
introduces approximation error and interference between updates.

The deep-RL work continues this analysis with neural networks. Value-based
experiments examine how replay buffers and target networks make bootstrapped
learning more stable. Policy-based experiments study how policies can be
optimized directly, and how baselines, value estimates, advantages, and
constrained updates affect the learning process. The project is open to
additional algorithms as the study develops.

The central ideas are shared across all of these methods:

- estimating values or policies from incomplete information;
- choosing between exploration and exploitation;
- deciding whether an update should use a return or a bootstrap target;
- managing the bias and variance of an estimator;
- keeping learning stable when the target and the function are changing
  together.

These connections are the main focus of the project. The notebooks are used to
inspect learning curves and behavior, while the algorithm modules make each
update rule easy to follow.

## Experiments

The notebooks are written as experiment guides rather than as stored results.
They show how to configure an environment, choose a training budget, and
compare learning behavior. Generated logs and checkpoints are kept locally
under `runs/` instead of being treated as part of the algorithm itself.

To explore the notebooks, create an environment and start Jupyter from the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy torch jupyter matplotlib gymnasium imageio tensorboard
jupyter lab
```

Some experiments may require additional environment-specific packages. The
imports in each notebook indicate those requirements.

## Tests

The current focused tests cover replay-buffer behavior:

```bash
pytest deep-rl/tests
```

Training results can vary with random seeds, environment versions, and
hardware. For that reason, comparisons should be interpreted through both the
learning curves and the assumptions behind each algorithm.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
