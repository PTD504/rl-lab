# MDP Foundations

Notes based on Chapter 3 (Sections 3.1–3.3) of *Reinforcement Learning: An Introduction* (Sutton & Barto, 2018).

This module is purely conceptual — no algorithm is introduced here. The goal is to nail down the vocabulary and notation (states, actions, rewards, returns) that every algorithm later (Dynamic Programming, Monte Carlo, TD Learning) will build on.

---

## 1. The Agent–Environment Interface

An MDP formalizes sequential decision-making as a loop between two entities:

- **Agent** — the learner / decision maker.
- **Environment** — everything the agent interacts with.

At each discrete time step $t = 0, 1, 2, \dots$:

1. The agent observes a state $S_t \in \mathcal{S}$.
2. The agent selects an action $A_t \in \mathcal{A}(s)$.
3. One step later, the environment returns a reward $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$ and a new state $S_{t+1}$.

This gives rise to a trajectory:

$$
S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \dots
$$

### The Dynamics Function

In a **finite MDP**, $\mathcal{S}$, $\mathcal{A}$, and $\mathcal{R}$ are all finite sets. The environment's dynamics are fully characterized by a single four-argument function:

$$
p(s', r \mid s, a) \doteq \Pr\{S_t = s', R_t = r \mid S_{t-1} = s, A_{t-1} = a\}
$$

for all $s', s \in \mathcal{S}$, $r \in \mathcal{R}$, $a \in \mathcal{A}(s)$.

> **This is the key object of the chapter.** Everything else (state-transition probabilities, expected rewards) can be derived from it.

It must satisfy the normalization condition:

$$
\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r \mid s, a) = 1, \quad \text{for all } s \in \mathcal{S}, a \in \mathcal{A}(s)
$$

### The Markov Property

The probabilities given by $p$ depend **only** on the immediately preceding state and action — not on the full history. This is the Markov property. It is a restriction on what counts as a valid *state representation*, not on the process itself: the state must summarize everything from the past that is relevant to the future.

### Quantities Derived from $p$

| Quantity | Formula | Meaning |
|---|---|---|
| State-transition probability | $p(s' \mid s, a) \doteq \sum_{r \in \mathcal{R}} p(s', r \mid s, a)$ | Probability of landing in $s'$, marginalizing over reward |
| Expected reward (state-action) | $r(s, a) \doteq \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r \mid s, a)$ | Expected immediate reward given $(s, a)$ |
| Expected reward (state-action-next-state) | $r(s, a, s') \doteq \sum_{r \in \mathcal{R}} r \dfrac{p(s', r \mid s, a)}{p(s' \mid s, a)}$ | Expected immediate reward given $(s, a, s')$ |

---

## 2. Goals and Rewards

The agent's objective is always expressed through a single scalar signal: the **reward** $R_t \in \mathbb{R}$.

### The Reward Hypothesis

> All of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (reward).

### Design Principle

The reward signal communicates **what** the agent should achieve, not **how** to achieve it. Prior knowledge about *how* to solve a task belongs in the initial policy or initial value function — not in the reward.

- **DO:** Reward the chess agent only for winning.
- **DON'T:** Do not reward it for capturing pieces or controlling the center — the agent may learn to farm sub-goals at the expense of the real objective.

---

## 3. Returns and Episodes

The agent's goal is to maximize the **expected return**, a specific function of the future reward sequence.

### Episodic Tasks

When agent-environment interaction naturally breaks into subsequences (episodes) ending in a terminal state:

$$
G_t \doteq R_{t+1} + R_{t+2} + R_{t+3} + \dots + R_T
$$

where $T$ is the final time step of the episode.

### Continuing Tasks and Discounting

When interaction goes on indefinitely without a natural endpoint, the undiscounted sum could diverge. We introduce a **discount rate** $\gamma$, $0 \le \gamma \le 1$:

$$
G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

- $\gamma = 0$: the agent is **myopic**, maximizing only $R_{t+1}$.
- $\gamma \to 1$: the agent becomes increasingly **farsighted**, weighting future rewards more heavily.

If $\gamma < 1$ and the reward sequence is bounded, the infinite sum is guaranteed to converge to a finite value.

### The Recursive Return Identity

Returns at successive time steps relate to each other recursively:

$$
G_t \doteq R_{t+1} + \gamma G_{t+1}
$$

> **This is the single most important formula in this chapter** — it is the seed of every Bellman equation used from Module 4 onward. Even though this chapter introduces no algorithm, this recursion is the reason Dynamic Programming, Monte Carlo, and TD methods all become possible.

### Unified Notation

Episodic and continuing tasks can be unified by treating episode termination as entering an absorbing state that transitions only to itself with reward 0. This lets us write a single general return formula:

$$
G_t \doteq \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k
$$

allowing $T = \infty$ or $\gamma = 1$ (but not both simultaneously).

---

## Summary Table

| Concept | Symbol | Key formula |
|---|---|---|
| Dynamics function | $p(s', r \mid s, a)$ | Fully characterizes the environment |
| State-transition probability | $p(s' \mid s, a)$ | Marginalizes $p$ over reward |
| Expected reward | $r(s, a)$ | Marginalizes $p$ over next state and reward |
| Return (episodic) | $G_t$ | $R_{t+1} + \dots + R_T$ |
| Return (discounted) | $G_t$ | $\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ |
| Recursive return | $G_t$ | $R_{t+1} + \gamma G_{t+1}$ |

---

## Some Examples of MDPs

Three custom MDP formulations, each in a different domain, to stress-test the framework beyond the textbook's Recycling Robot.

### Example A: Chess

**Overview:** A two-player, zero-sum board game modeled as an MDP where the agent learns to play by receiving a large reward for winning, a large penalty for losing, and small shaped rewards for capturing pieces along the way.

**State:** The state represents the complete board configuration at a given turn. This includes the exact positions of all remaining pieces for both sides (e.g., location of both Kings, status of the Queens), as well as whose turn it is to move (white or black).

**Action:** Any legal move a player can make with an available piece according to the standard rules of chess.

**Reward:** A custom reward structure based on piece values and game outcomes:

- $+100$ for winning.
- $-100$ for losing.
- Intermediate positive rewards for capturing an opponent's piece, scaled by the piece's value, like $+9$ for a Queen, $+5$ for a Rook, $+1$ for a Pawn, etc.

---

### Example B: Pac-Man

**Overview:** A single-agent navigation and survival game modeled as an MDP where the agent must move through a maze, collect food, and avoid ghosts to maximize score.

**State:** The complete grid environment configuration at the current time step. This includes the exact $(x, y)$ coordinates of Pac-Man, the positions and current states of all ghosts, the locations of all remaining food dots and power pellets, as well as the map boundary and wall layouts.

**Action:** The directional movement Pac-Man chooses for the next step: $Up$, $Down$, $Left$, or $Right$.

**Reward:** A simple score-based reward structure to guide optimal movement:

- $+2$ for eating a food dot.
- $-1$ for moving to an empty space (time penalty to encourage reaching dots quickly).
- $+100$ for eating the final food dot (winning).
- $-100$ for colliding with a ghost (losing).

---

### Example C: Portfolio Rebalancing

**Overview:** A sequential financial decision-making task modeled as an MDP where the agent (investor) periodically rebalances a portfolio to maximize financial gain based on market conditions.

**State:** The current financial status and market context at a given trading period. This includes the investor's current portfolio allocation (cash balance and quantities of held assets), asset market prices, and key financial indicators (e.g., price trends, volatility).

**Action:** The rebalancing decision for each asset in the portfolio: $Buy$, $Sell$, or $Hold$.

**Reward:** The financial gain or loss generated in that period, represented as the net change in total portfolio value after accounting for transaction fees.