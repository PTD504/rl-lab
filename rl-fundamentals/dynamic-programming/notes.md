# Dynamic Programming

## 1. Overview

**Dynamic Programming (DP)** refers to a collection of algorithms used to compute optimal policies when given a **perfect model** of the environment as a Markov Decision Process (MDP).

### Key Concepts & Assumptions
- **Model Requirement**: Requires complete knowledge of environment dynamics $p(s', r \mid s, a)$ for all $s \in \mathcal{S}$, $a \in \mathcal{A}(s)$, $r \in \mathcal{R}$, and $s' \in \mathcal{S}^+$.
- **Finite MDP Setting**: Assumes finite state, action, and reward spaces. (Tasks with continuous spaces typically require quantization or function approximation).
- **Value Function Structure**: Uses value functions ($v$ and $q$) to organize and direct the search for optimal policies that satisfy Bellman optimality equations.

### Significance & Practical Constraints
- **Theoretical Importance**: Serves as the foundational baseline for Reinforcement Learning. Other RL methods (e.g., model-free or sample-based algorithms) aim to achieve similar goals without requiring a model or incurring high computational costs.
- **Practical Limitations**:
  - **Model Dependency**: Requires exact environment transitions and rewards, which are often unavailable in real-world applications.
  - **Computational Expense**: High computational complexity when dealing with large state-action spaces.

## 2. State-Value Function and Action-Value Function

### State-Value Function $v_{\pi}(s)$

- **Definition**: Expected return starting from state $s$ at time step $t$ and following policy $\pi$.

$$
v_{\pi}(s) \doteq \mathbb{E}_{\pi}[G_t \mid S_t = s]
$$

- **One-step Return Expansion** (recalling $G_t = R_{t+1} + \gamma G_{t+1}$ from [MDPs.md](../mdp-foundations/MDPs.md#L104-L111)):

$$
v_{\pi}(s) = \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_t = s]
$$

### Action-Value Function $q_{\pi}(s,a)$

- **Definition**: Expected return starting from state $s$, taking action $a$ at time step $t$, and following policy $\pi$ thereafter.

$$
q_{\pi}(s,a) \doteq \mathbb{E}_{\pi}[G_t \mid S_t = s, A_t = a] = \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a]
$$

## 3. Bellman Equation

The Bellman Equation expresses the value of a state (or state-action pair) based on the expected value of successor states — rather than computing the sum of the entire future reward sequence $G_t$.

### 3.1 Bellman Equation for $v_{\pi}(s)$

The value of state $s$ equals the weighted average (under policy $\pi$ and environment dynamics $p$) of the immediate reward plus the discounted value of the successor state:

$$
v_{\pi}(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_{\pi}(s') \right]
$$

### 3.2 Bellman Equation for $q_{\pi}(s,a)$

Similarly, but since action $a$ is fixed, the outer sum is over environment dynamics $p$ only (without the initial sum over $\pi(a \mid s)$):

$$
q_{\pi}(s,a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \sum_{a'} \pi(a' \mid s') q_{\pi}(s', a') \right]
$$

### 3.3 Relationship Between $v_{\pi}$ and $q_{\pi}$

We can directly compute one value function if we already have the other:

- **Computing $v_{\pi}(s)$ from $q_{\pi}(s,a)$**: If we already know the action-values for all actions in state $s$, we can compute $v_{\pi}(s)$ by taking the weighted average of $q_{\pi}(s,a)$ over policy $\pi$:

$$
v_{\pi}(s) = \sum_{a} \pi(a \mid s) q_{\pi}(s, a)
$$

- **Computing $q_{\pi}(s,a)$ from $v_{\pi}(s')$**: If we already know the state-values of all successor states $s'$, we can compute $q_{\pi}(s,a)$ by combining the expected immediate reward with the discounted next-state values under environment dynamics $p$:

$$
q_{\pi}(s,a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_{\pi}(s') \right]
$$


## 4. Bellman Optimality Equation

An **optimal policy** $\pi_{\star}$ is a policy whose expected return is greater than or equal to all other policies across all states ($\pi \ge \pi' \iff v_{\pi}(s) \ge v_{\pi'}(s), \forall s$). All optimal policies share the same optimal value functions.

### 4.1 Optimal Value Functions

- **Optimal State-Value Function $v_{\star}(s)$**: The maximum value achievable under any policy for state $s$:

$$
v_{\star}(s) \doteq \max_{\pi} v_{\pi}(s), \quad \forall s \in \mathcal{S}
$$

- **Optimal Action-Value Function $q_{\star}(s,a)$**: The maximum value achievable after taking action $a$ in state $s$ and thereafter following an optimal policy:

$$
q_{\star}(s,a) \doteq \max_{\pi} q_{\pi}(s,a), \quad \forall s \in \mathcal{S}, a \in \mathcal{A}(s)
$$

---

### 4.2 Bellman Optimality Equations

Instead of averaging over policy actions — $\sum_a \pi(a \mid s)$ — the optimal value function assumes the agent always chooses the **best action** — $\max_a$.

- **Bellman Optimality Equation for $v_{\star}(s)$**:

$$
v_{\star}(s) = \max_{a \in \mathcal{A}(s)} q_{\star}(s,a) = \max_{a} \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_{\star}(s') \right]
$$

- **Bellman Optimality Equation for $q_{\star}(s,a)$**:

$$
q_{\star}(s,a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \max_{a'} q_{\star}(s', a') \right]
$$

> **Key Takeaway**: Bellman Optimality Equations replace the linear policy expectation $\sum_a \pi(a \mid s)$ with a non-linear $\max$ operator. Finding exact solutions requires solving this system of non-linear equations (which DP methods accomplish iteratively).


## 5. Policy Evaluation (Prediction)

### 5.1 Concept & Purpose
- **What it is**: An iterative Dynamic Programming algorithm used to estimate the state-value function $v_{\pi}$ for a given policy $\pi$.
- **Problem Solved**: Addresses the **prediction problem** — calculating how good a specific policy $\pi$ is by iteratively updating value estimates $V(s)$ until they converge to true $v_{\pi}(s)$, avoiding solving large systems of linear equations directly.

---

### 5.2 Update Rule
Iterative policy evaluation converts the Bellman expectation equation into an iterative update rule. Starting from an arbitrary initial value array $V_0$ (with $V(\text{terminal}) = 0$), each iteration $k+1$ updates the value of state $s$ using the values of successor states from iteration $k$:

$$
V_{k+1}(s) \leftarrow \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V_k(s') \right], \quad \forall s \in \mathcal{S}
$$

As $k \to \infty$, $V_k \to v_{\pi}$ under standard convergence conditions ($\gamma < 1$ or guaranteed termination in episodic tasks).

---

### 5.3 Pseudocode

```text
Iterative Policy Evaluation, for estimating V ≈ v_π
--------------------------------------------------------------------------------
Input π, the policy to be evaluated
Algorithm parameter: a small threshold θ > 0 determining accuracy of estimation
Initialize V(s), for all s ∈ S⁺, arbitrarily except that V(terminal) = 0

Loop:
    Δ ← 0
    Loop for each s ∈ S:
        v ← V(s)
        V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [ r + γ V(s') ]
        Δ ← max(Δ, |v - V(s)|)
until Δ < θ
```


## 6. Policy Improvement

### 6.1 Concept & Purpose
- **What it is**: The process of constructing a new, improved policy $\pi'$ from a current policy $\pi$ by acting **greedily** with respect to the value function $v_{\pi}$.
- **Problem Solved**: Solves the policy enhancement step — utilizing value estimates to make better decision-making choices.
- **Policy Improvement Theorem**: If $q_{\pi}(s, \pi'(s)) \ge v_{\pi}(s)$ for all $s \in \mathcal{S}$, then the new policy $\pi'$ is guaranteed to be globally as good as, or better than, $\pi$:

$$
v_{\pi'}(s) \ge v_{\pi}(s), \quad \forall s \in \mathcal{S}
$$

---

### 6.2 Greedy Policy Construction
To construct $\pi'$, for each state $s$ we select the action that maximizes the expected return ($q_{\pi}(s, a)$):

$$
\pi'(s) \doteq \arg\max_{a} q_{\pi}(s, a) = \arg\max_{a} \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_{\pi}(s') \right]
$$

- **Optimality Check**: If $v_{\pi'}(s) = v_{\pi}(s)$ for all states $s \in \mathcal{S}$, then $v_{\pi} = v_{\star}$ and the policy $\pi$ is already optimal.


## 7. Policy Iteration

### 7.1 Concept & Workflow
- **Core Mechanism**: Policy Iteration solves the control problem by strictly alternating between **Policy Evaluation** (estimating $v_{\pi}$ for current policy $\pi$) and **Policy Improvement** (making $\pi$ greedy with respect to $v_{\pi}$):

$$
\pi_0 \xrightarrow{\text{Evaluation}} v_{\pi_0} \xrightarrow{\text{Improvement}} \pi_1 \xrightarrow{\text{Evaluation}} v_{\pi_1} \xrightarrow{\text{Improvement}} \dots \xrightarrow{\text{Improvement}} \pi_{\star} \xrightarrow{\text{Evaluation}} v_{\star}
$$

- **Convergence**: Since a finite MDP has a finite number of distinct policies, this process is guaranteed to converge to an optimal policy $\pi_{\star}$ and optimal value function $v_{\star}$ in a finite number of steps.

---

### 7.2 Pseudocode

```text
Policy Iteration (using iterative policy evaluation) for estimating π ≈ π_{\star}
--------------------------------------------------------------------------------
1. Initialization
   V(s) ∈ ℝ and π(s) ∈ A(s) arbitrarily for all s ∈ S

2. Policy Evaluation
   Loop:
       Δ ← 0
       Loop for each s ∈ S:
           v ← V(s)
           V(s) ← Σ_{s',r} p(s',r|s, π(s)) [ r + γ V(s') ]
           Δ ← max(Δ, |v - V(s)|)
   until Δ < θ (a small positive number determining accuracy of estimation)

3. Policy Improvement
   policy-stable ← true
   For each s ∈ S:
       old-action ← π(s)
       π(s) ← argmax_a Σ_{s',r} p(s',r|s,a) [ r + γ V(s') ]
       If old-action ≠ π(s), then policy-stable ← false

   If policy-stable, then stop and return V ≈ v_{\star} and π ≈ π_{\star}; else go to 2
```


## 8. Value Iteration

### 8.1 Concept & Relation to Policy Iteration
- **What it is**: An efficient Dynamic Programming algorithm that computes the optimal value function $v_{\star}$ without requiring multiple sweeps of policy evaluation per step.
- **Variant of Policy Iteration**: Standard Policy Iteration requires running policy evaluation until full convergence ($\Delta < \theta$) in every iteration. **Value Iteration is a special variant of Policy Iteration** where policy evaluation is truncated after just **a single sweep** ($k = 1$).
- **Key Differences**:
  | Feature | Policy Iteration | Value Iteration |
  |---|---|---|
  | **Evaluation Sweep** | Multiple sweeps until convergence per iteration | Single sweep ($k = 1$) per iteration |
  | **Update Operator** | Expectation over current policy $\sum_a \pi(a \mid s)$ | Maximum over all actions $\max_a$ |
  | **Policy Tracking** | Explicitly updates $\pi(s)$ at every step | Policy is extracted only once after $V$ converges |

---

### 8.2 Update Rule
Value Iteration combines one step of policy evaluation and policy improvement directly into the Bellman optimality update rule:

$$
V_{k+1}(s) \leftarrow \max_{a} \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V_k(s') \right], \quad \forall s \in \mathcal{S}
$$

After $V$ converges to $v_{\star}$, the optimal deterministic policy $\pi_{\star}$ is extracted in a final step.

---

### 8.3 Pseudocode

```text
Value Iteration, for estimating π ≈ π_{\star}
--------------------------------------------------------------------------------
Algorithm parameter: a small threshold θ > 0 determining accuracy of estimation
Initialize V(s), for all s ∈ S⁺, arbitrarily except that V(terminal) = 0

Loop:
    Δ ← 0
    Loop for each s ∈ S:
        v ← V(s)
        V(s) ← max_a Σ_{s',r} p(s',r|s,a) [ r + γ V(s') ]
        Δ ← max(Δ, |v - V(s)|)
until Δ < θ

Output a deterministic policy, π ≈ π_{\star}, such that
    π(s) = argmax_a Σ_{s',r} p(s',r|s,a) [ r + γ V(s') ]
```


## 9. Summary

### 9.1 Overview Table

| Algorithm | Input | Output | Key Characteristics |
|---|---|---|---|
| **Policy Evaluation** | Policy $\pi$ | $v_{\pi}$ | Iterates until value estimates converge to evaluate a fixed policy $\pi$ |
| **Policy Iteration** | Initial policy $\pi_0$ | Optimal policy $\pi_{\star}$, $v_{\star}$ | Alternates between full Policy Evaluation and Policy Improvement |
| **Value Iteration** | Initial values $V_0$ | Optimal policy $\pi_{\star}$, $v_{\star}$ | Single-step Bellman optimality update ($\max_a$) per sweep |

---

### 9.2 Convergence Comparison: Policy Iteration vs. Value Iteration

- **Number of Outer Iterations**:
  - **Policy Iteration**: Requires **fewer outer iterations** to reach the optimal policy because each policy evaluation step is solved completely to convergence before improving the policy.
  - **Value Iteration**: Requires **more outer sweeps** to converge because value estimates are updated incrementally ($k=1$ evaluation sweep per step).
- **Computational Cost per Iteration**:
  - **Policy Iteration**: High cost per outer iteration due to multiple inner evaluation sweeps.
  - **Value Iteration**: Low cost per sweep (a single $\max_a$ pass over $\mathcal{S}$), often making it faster in practice for large MDPs.