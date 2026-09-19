# RL Lab

A reinforcement learning study repository built around small, inspectable implementations and experiments. The code follows the progression from bandit problems and MDP foundations to dynamic programming, Monte Carlo methods, temporal-difference learning, and function approximation.

The goal is to make the learning mechanics visible: what information an algorithm needs, how its estimates are updated, how it explores, and where its assumptions become limiting.

## Project structure

```text
rl-lab/
├── rl-fundamentals/                 # Core RL concepts and model-based methods
│   ├── mdp-foundations/             # MDP definitions, returns, policies, and Bellman equations
│   ├── multi-armed-bandits/         # Bandit agents, experiments, and exploration comparisons
│   └── dynamic-programming/         # Grid world, policy evaluation, policy iteration, and value iteration
├── sample-based-learning/           # Model-free learning from sampled experience
│   ├── monte-carlo-methods/         # MC prediction/control and importance-sampling experiments
│   └── td-learning-methods/         # TD(0), SARSA, Q-learning, Expected SARSA, and double estimators
└── function-approximation/          # Linear and neural Q-learning with replay and target functions
```

## Insights from the repository

### Multi-armed bandits

Bandits isolate the exploration-exploitation problem before state transitions are introduced. The experiments compare epsilon-greedy action selection, optimistic initial values, UCB, and constant-step-size updates for non-stationary rewards. A central lesson is that there is no single exploration strategy that is best independently of the reward process: stationary and changing environments reward different choices.

### MDP foundations

MDPs provide the language for reasoning about states, actions, rewards, returns, policies, and value functions. The Bellman equations show how a long-term prediction can be decomposed into an immediate reward plus the value of what comes next. This recursive view is the common structure behind the algorithms implemented later.

### Dynamic programming

Dynamic programming solves finite MDPs when the environment model is known. Policy evaluation estimates how well a fixed policy performs, while policy improvement makes the policy greedy with respect to those estimates. Policy iteration alternates these two operations; value iteration combines them into a single optimality update. These methods are useful baselines, but their dependence on a complete model and their cost over large state spaces motivate sample-based learning.

### Monte Carlo methods

Monte Carlo methods remove the need for an explicit transition model and learn from complete episodes. They estimate values from observed returns rather than from one-step predictions, which makes the estimates conceptually direct but delays updates and can produce high variance. The implementations cover first-visit prediction, on-policy control with epsilon-soft behavior, and off-policy prediction/control using importance sampling. The comparison between ordinary and weighted importance sampling highlights the practical trade-off between unbiasedness and variance.

### Temporal-difference learning

Temporal-difference methods learn from incomplete experience by bootstrapping from current estimates. TD(0), SARSA, Q-learning, and Expected SARSA differ mainly in how they construct the next-step target and whether the target follows the behavior policy or a greedy policy. Double Q-learning and Double Expected SARSA address the overestimation that can arise when the same estimates are used both to select and evaluate the best next action.

### Function approximation

Tabular methods assign an independent value to every state or state-action pair. Function approximation replaces that table with a parameterized model, allowing related states to share information and making continuous-valued observations practical. The current experiments move through three stages: online Q-learning with a linear model, replay-buffer Q-learning, and replay-buffer Q-learning with a periodically updated target function. Neural-network Q-learning then illustrates the additional representational capacity and stability concerns introduced by nonlinear models.

The key trade-off is generalization versus interference. A parameter update can improve predictions for many related states, but it can also damage predictions elsewhere. Replay buffers and target functions help manage moving targets and correlated experience; they improve the learning setup without removing the importance of state representation, exploration, or reward design.

## A unifying view

The repository follows a sequence of increasingly relaxed assumptions:

1. **Bandits** study action selection without state transitions.
2. **Dynamic programming** uses a known environment model to compute value functions.
3. **Monte Carlo methods** learn from complete sampled episodes.
4. **Temporal-difference methods** learn online by combining samples with bootstrapping.
5. **Function approximation** replaces explicit tables with shared parameters that can generalize across observations.

Across these stages, the central design questions remain the same: what should be estimated, what data should update the estimate, how should the agent explore, and how can learning remain stable?

## Environment

The implementations use Python with NumPy, PyTorch, and Jupyter. Several experiments use environments following the modern `reset()`/`step()` API, so a compatible Gym environment package may also be required.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
