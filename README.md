# rl-lab
Core reinforcement learning algorithms, implemented from scratch — from k-armed bandits all the way to actor-critic methods — following Sutton & Barto's *Reinforcement Learning: An Introduction*.
The goal isn't just passing assignments. Each algorithm here is built by hand in NumPy/Python to actually understand *why* it works, not just call a library and watch numbers go up. Whare it makes sense, results with plots — learning curves, comparisons, failure cases — because "it trained" is a weaker than "here's what it learned and why".

## Structure
- `rl-fundamentals/` — k-armed bandits, MDPs, dynamic programming. The building blocks: action-value estimation, exploration vs. exploitation, Bellman equations.
- `sample-based-learning/` — Monte Carlo methods, TD-learning, Q-learning, SARSA. Learning from experience instead of a known model.
- `function-approximation/` — moving past tables: linear and nonlinear approximation for prediction and control at scale.
- `concept2code/` — theory into practice. Taking a specific RL concept and turning it into working code on a real problem.
