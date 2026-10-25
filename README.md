# RL Lab

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1.x-0f9d58)
![License](https://img.shields.io/badge/license-MIT-green)

From-scratch reinforcement-learning implementations, from bandits and dynamic
programming to DQN, PPO, DDPG, TD3 and SAC. The code is intentionally small and
inspectable so the link between the math, the update rule and the agent's
behavior stays clear.

## Demo

Continuous-control agents trained from scratch on `Pendulum-v1`:

| DDPG | TD3 | SAC |
|:---:|:---:|:---:|
| ![DDPG on Pendulum](deep_rl/experiments/assets/pendulum_ddpg.gif) | ![TD3 on Pendulum](deep_rl/experiments/assets/pendulum_td3.gif) | ![SAC on Pendulum](deep_rl/experiments/assets/pendulum_sac.gif) |

## Algorithms implemented

| Area | Algorithm | Type | Code | Notebook | Theory notes |
|---|---|---|---|---|---|
| Bandits | epsilon-greedy, non-stationary, optimistic init, UCB | action-value | [agents.py](rl-fundamentals/multi-armed-bandits/agents.py) | [notebooks](rl-fundamentals/multi-armed-bandits/notebooks) | [MDPs.md](rl-fundamentals/mdp-foundations/MDPs.md) |
| Dynamic programming | Policy evaluation, policy iteration, value iteration | model-based | [algorithms](rl-fundamentals/dynamic-programming/algorithms) | [notebooks](rl-fundamentals/dynamic-programming/notebooks) | [notes.md](rl-fundamentals/dynamic-programming/notes.md) |
| Monte Carlo | MC prediction/control, off-policy MC | tabular | [algorithms](sample-based-learning/monte-carlo-methods/algorithms) | [notebooks](sample-based-learning/monte-carlo-methods/notebooks) | [notes.md](sample-based-learning/monte-carlo-methods/notes.md) |
| TD learning | TD(0), SARSA, Q-learning, Expected SARSA, Double Q-learning, Double Expected SARSA | tabular | [algorithms](sample-based-learning/td-learning-methods/algorithms) | [notebooks](sample-based-learning/td-learning-methods/notebooks) | - |
| Function approximation | Linear and non-linear Q-learning, replay buffer, target network | off-policy, value-based | [algorithms](function-approximation/algorithms) | [notebooks](function-approximation/notebooks) | [notes.md](function-approximation/notes.md) |
| Deep value-based | DQN, Double DQN, Dueling DQN | off-policy, discrete | [dqn.py](deep_rl/dqns/dqn.py), [double_dqn.py](deep_rl/dqns/double_dqn.py), [dueling_dqn.py](deep_rl/dqns/dueling_dqn.py) | [dqns.ipynb](deep_rl/experiments/dqns.ipynb) | [notes.md](deep_rl/dqns/notes.md) |
| Policy gradient | REINFORCE, REINFORCE with baseline | on-policy, discrete | [reinforce.py](deep_rl/policy_gradient/reinforce.py), [reinforce_with_baseline.py](deep_rl/policy_gradient/reinforce_with_baseline.py) | [policy_gradient.ipynb](deep_rl/experiments/policy_gradient.ipynb) | - |
| Actor-critic | One-step Actor-Critic, A2C | on-policy, discrete | [actor_critic.py](deep_rl/policy_gradient/actor_critic.py), [a2c.py](deep_rl/policy_gradient/a2c.py) | [a2c_experiment.ipynb](deep_rl/experiments/a2c_experiment.ipynb) | - |
| Actor-critic | PPO (clipped objective, GAE) | on-policy, discrete | [ppo.py](deep_rl/policy_gradient/ppo.py) | [ppo_experiment.ipynb](deep_rl/experiments/ppo_experiment.ipynb) | - |
| Continuous control | DDPG | off-policy, continuous | [ddpg.py](deep_rl/continuous_control/ddpg.py) | [ddpg.ipynb](deep_rl/experiments/ddpg.ipynb) | - |
| Continuous control | TD3 | off-policy, continuous | [td3.py](deep_rl/continuous_control/td3.py) | [td3.ipynb](deep_rl/experiments/td3.ipynb) | - |
| Continuous control | SAC | off-policy, continuous | [sac.py](deep_rl/continuous_control/sac.py) | [sac.ipynb](deep_rl/experiments/sac.ipynb) | - |

## Quick start

```bash
git clone <this-repo-url> && cd rl-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebook]"   # optional extras: atari, box2d, mujoco
pytest
```

The `deep_rl` package is importable after installation, so no `sys.path`
changes are needed:

```python
import gymnasium as gym
from deep_rl.continuous_control.sac import SACAgent

env = gym.make("Pendulum-v1")
agent = SACAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    max_action=float(env.action_space.high[0]),
)
agent.learn(env, max_steps=20_000)
print(agent.evaluate(env, num_episodes=5))
```

The DQN-family notebook needs the Atari extra: `pip install -e ".[atari]"`.
Notebooks live in [deep_rl/experiments](deep_rl/experiments).

## Project structure

```text
rl-lab/
├── pyproject.toml
├── rl-fundamentals/
│   ├── mdp-foundations/                  # States, returns, policies, values, Bellman equations
│   ├── multi-armed-bandits/              # Exploration, exploitation, and action-value estimates
│   └── dynamic-programming/              # Policy evaluation, policy iteration, and value iteration
├── sample-based-learning/
│   ├── monte-carlo-methods/              # Learning from complete episode returns
│   └── td-learning-methods/              # Bootstrapping, SARSA, Q-learning, and related methods
├── function-approximation/
│   ├── algorithms/                       # Linear and non-linear function approximation
│   └── notebooks/                        # Experiments beyond tabular value functions
├── deep_rl/                              # Installable package
│   ├── dqns/                             # DQN, Double DQN, Dueling DQN
│   ├── policy_gradient/                  # On-policy: REINFORCE, baseline, Actor-Critic, A2C, PPO
│   ├── continuous_control/               # Off-policy continuous control: DDPG, TD3, SAC
│   ├── utils/                            # Shared networks, replay buffers, rendering
│   ├── experiments/                      # Notebooks and demo assets
│   └── tests/                            # pytest suite
└── LICENSE
```

The tabular and function-approximation folders are standalone study material
and are not part of the installable package.

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

## Tests

```bash
pytest
```

The suite covers replay-buffer behavior, PPO checkpoint round-trips, and smoke
tests that run PPO, DDPG, TD3 and SAC for a few steps.

Generated logs and checkpoints are written locally under `runs/` and are not
tracked. Training results vary with random seeds, environment versions, and
hardware, so compare algorithms through learning curves and their assumptions.

## Roadmap

- PPO with continuous actions (Gaussian policy) and MuJoCo benchmarks.
- Learning-curve comparisons for the continuous-control agents.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
