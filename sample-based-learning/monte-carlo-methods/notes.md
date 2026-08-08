# Monte Carlo Methods

## 1. Introduction

Monte Carlo (MC) methods mark our shift from **model-based** to **model-free** reinforcement learning. Rather than assuming complete knowledge of environment dynamics (the transition probabilities $P(s', r | s, a)$ required in Dynamic Programming), MC methods learn optimal behavior purely from **sample experience**—sequences of states, actions, and rewards collected through interaction.

### Key Characteristics & Concepts:
* **Model-Free Experience**: Learn directly from actual or simulated transitions. Even with a simulator, MC only requires *generating sample transitions* rather than knowing explicit probability distributions.
* **Episodic Requirement**: Classical Monte Carlo methods in RL estimate returns by waiting until episode termination, so they are formulated for episodic tasks where updates occur **episode-by-episode** (offline after termination).
* **Extension of Multi-Armed Bandits**: MC estimates values by averaging complete returns for each state-action pair—similar to how bandit algorithms average rewards for single actions. However, in MDPs, returns depend on future actions, making the learning process nonstationary across interrelated states.
* **GPI Framework Integration**: Adapts Generalized Policy Iteration (GPI) from DP. Instead of computing exact values using the model, MC *learns* value functions ($v_\pi, q_\pi$) from sampled returns and updates policies accordingly.

> **Trade-off Summary (MC vs. DP)**:
> * **MC**: Uses actual complete returns, providing unbiased estimates under standard conditions for ordinary MC prediction, but generally suffers from high variance and delayed updates.
> * **DP**: Bootstrapping (uses current estimates), lower variance, step-by-step updates, but requires an exact environment model.

## 2. Monte Carlo Prediction

Monte Carlo Prediction addresses the **policy evaluation** problem: learning the state-value function $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$ for a given policy $\pi$.

### Core Concept & Estimation Formula
The value of state $s$ is estimated by averaging observed returns after visits to $s$:
$$V(s) \approx \frac{\sum \text{Returns observed following visits to } s}{\text{Number of visits to } s}$$

### First-Visit vs. Every-Visit MC
When a state $s$ is visited multiple times in a single episode:
* **First-Visit MC**: Averages returns following *only the first visit* to $s$ in each episode.
  * **Convergence & Properties**: Each first-visit return is an unbiased sample of $v_\pi(s)$. Across independently generated episodes, these samples are **independent and identically distributed** under standard episodic assumptions, so by the Law of Large Numbers, the sample average converges to $v_\pi(s)$ with standard error decreasing as $1/\sqrt{n}$.
* **Every-Visit MC**: Averages returns following *all visits* to $s$ in an episode.
  * **Convergence & Properties**: Returns within an episode may be correlated, but under standard assumptions the estimate still converges to $v_\pi(s)$.

### Pseudocode: First-Visit MC Prediction

```text
First-visit MC prediction, for estimating V ≈ v_π

Input: a policy π to be evaluated
Initialize:
    V(s) ∈ ℝ, arbitrarily, for all s ∈ S
    Returns(s) ← an empty list, for all s ∈ S

Loop forever (for each episode):
    Generate an episode following π: S_0, A_0, R_1, S_1, A_1, R_2, . . . , S_{T-1}, A_{T-1}, R_T
    G ← 0
    Loop for each step of episode, t = T-1, T-2, . . . , 0:
        G ← γG + R_{t+1}
        Unless S_t appears in S_0, S_1, . . . , S_{t-1}:
            Append G to Returns(S_t)
            V(S_t) ← average(Returns(S_t))
```

## 3. Monte Carlo Control - Exploring Starts

Monte Carlo Control aims to approximate optimal policies by combining Monte Carlo estimation with **Generalized Policy Iteration (GPI)**.

### 1. Why Action Values ($Q$) Instead of State Values ($V$)?
In model-based methods (DP), state values $v(s)$ suffice because environment dynamics $P(s', r | s, a)$ are known for one-step lookahead. In **model-free** RL, we must estimate **action-value functions** $q_\pi(s, a)$ directly to determine greedy actions without a model:
$$\pi(s) \doteq \arg\max_a q(s, a)$$

### 2. GPI & Policy Improvement Theorem
MC Control alternates between:
* **Policy Evaluation**: $Q \rightsquigarrow q_\pi$ (estimating action values from sample returns).
* **Policy Improvement**: $\pi \rightsquigarrow \text{greedy}(Q)$ (making the policy greedy w.r.t. current $Q$).

The Policy Improvement Theorem guarantees monotonic improvement when the **true** action-value function $q_{\pi_k}$ is used:
$$q_{\pi_k}(s, \pi_{k+1}(s)) = \max_a q_{\pi_k}(s, a) \ge q_{\pi_k}(s, \pi_k(s)) = v_{\pi_k}(s)$$
MC Control approximates this ideal process by estimating $q_{\pi_k}$ from sampled returns and making the policy greedy with respect to the estimated action values $Q \approx q_{\pi_k}$.

### 3. The Exploration Problem & Exploring Starts (ES)
* **The Problem**: If a policy is deterministic, many $(s, a)$ pairs may never be visited, leaving their values unestimated and preventing discovery of better actions.
* **Exploring Starts Assumption**: Assumes every episode starts at a state-action pair $(S_0, A_0)$ chosen randomly such that all state-action pairs have non-zero probability of being selected as the start.
* **Limitation**: ES is impractical in real-world systems (e.g., robotics or games) where initializing the system at arbitrary state-action pairs is impossible.

### Pseudocode: Monte Carlo ES

```text
Monte Carlo ES (Exploring Starts), for estimating π ≈ π_*

Initialize:
    π(s) ∈ A(s) (arbitrarily), for all s ∈ S
    Q(s, a) ∈ ℝ (arbitrarily), for all s ∈ S, a ∈ A(s)
    Returns(s, a) ← empty list, for all s ∈ S, a ∈ A(s)

Loop forever (for each episode):
    Choose S_0 ∈ S, A_0 ∈ A(S_0) randomly such that all pairs have probability > 0
    Generate an episode from S_0, A_0, following π: S_0, A_0, R_1, . . . , S_{T-1}, A_{T-1}, R_T
    G ← 0
    Loop for each step of episode, t = T-1, T-2, . . . , 0:
        G ← γG + R_{t+1}
        Unless the pair S_t, A_t appears in S_0, A_0, S_1, A_1 . . . , S_{t-1}, A_{t-1}:
            Append G to Returns(S_t, A_t)
            Q(S_t, A_t) ← average(Returns(S_t, A_t))
            π(S_t) ← argmax_a Q(S_t, a)
```

## 4. Monte Carlo Control - On-policy (ε-greedy / ε-soft)

On-policy methods eliminate the unrealistic **Exploring Starts** assumption by ensuring the agent continuously explores while executing the current policy.

### 1. On-Policy vs. Off-Policy Distinction
* **On-policy**: Evaluates and improves the *same policy* used to make decisions and generate experience episodes.
* **Off-policy**: Evaluates or improves a *target policy* $\pi$ using data generated by a different *behavior policy* $b$.

### 2. $\epsilon$-Soft & $\epsilon$-Greedy Policies
To maintain continuous exploration without ES, we enforce **soft policies** where $\pi(a|s) > 0$ for all state-action pairs:
* **$\epsilon$-Soft Policy**: A policy that satisfies $\pi(a|s) \ge \frac{\epsilon}{|\mathcal{A}(s)|}$ for all $s \in \mathcal{S}, a \in \mathcal{A}(s)$ with $\epsilon > 0$.
* **$\epsilon$-Greedy Policy**: An $\epsilon$-soft policy closest to greedy. The action with the highest estimated action value $A^* = \arg\max_a Q(s, a)$ is selected with probability $1 - \epsilon + \frac{\epsilon}{|\mathcal{A}(s)|}$, while all other non-greedy actions receive minimum probability $\frac{\epsilon}{|\mathcal{A}(s)|}$:

$$\pi(a|s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}(s)|} & \text{if } a = A^* \\ \frac{\epsilon}{|\mathcal{A}(s)|} & \text{if } a \neq A^* \end{cases}$$

### 3. Policy Improvement Guarantee
Under GPI, shifting an $\epsilon$-soft policy toward an $\epsilon$-greedy policy w.r.t. $q_\pi$ guarantees monotonic improvement ($\mathbb{E}_{\pi'}[G_t \mid S_t=s] \ge \mathbb{E}_\pi[G_t \mid S_t=s]$) among all $\epsilon$-soft policies, eventually converging to the optimal policy within the class of $\epsilon$-soft policies (for a fixed $\epsilon > 0$).

### Pseudocode: On-policy First-visit MC Control

```text
On-policy first-visit MC control (for ε-soft policies), estimates π ≈ π_*

Algorithm parameter: small ε > 0
Initialize:
    π ← an arbitrary ε-soft policy
    Q(s, a) ∈ ℝ (arbitrarily), for all s ∈ S, a ∈ A(s)
    Returns(s, a) ← empty list, for all s ∈ S, a ∈ A(s)

Repeat forever (for each episode):
    Generate an episode following π: S_0, A_0, R_1, . . . , S_{T-1}, A_{T-1}, R_T
    G ← 0
    Loop for each step of episode, t = T-1, T-2, . . . , 0:
        G ← γG + R_{t+1}
        Unless the pair S_t, A_t appears in S_0, A_0, S_1, A_1 . . . , S_{t-1}, A_{t-1}:
            Append G to Returns(S_t, A_t)
            Q(S_t, A_t) ← average(Returns(S_t, A_t))
            A* ← argmax_a Q(S_t, a)                  (with ties broken arbitrarily)
            For all a ∈ A(S_t):
                π(a|S_t) ← { 1 - ε + ε/|A(S_t)|   if a = A*
                            { ε/|A(S_t)|       if a ≠ A*
```


## 5. Off-policy Prediction via Importance Sampling

Off-policy methods separate exploration from learning by using two distinct policies:
* **Target Policy ($\pi$)**: The policy being evaluated and optimized (typically deterministic / greedy).
* **Behavior Policy ($b$)**: The exploratory policy used to interact with the environment and generate data.

> **Special Case Note**: On-policy learning is a special case of off-policy learning where the target policy and behavior policy are identical ($\pi = b$). When $\pi = b$, the importance sampling ratio simplifies to $\rho = 1$, reducing off-policy formulas back to standard on-policy updates.


### 1. Coverage Assumption
For off-policy learning to be valid, data generated by $b$ must cover all actions that $\pi$ might take:
$$\pi(a|s) > 0 \implies b(a|s) > 0 \quad \forall s \in \mathcal{S}, a \in \mathcal{A}(s)$$

### 2. Importance Sampling Ratio ($\rho$)
To estimate expectations under $\pi$ using returns generated by $b$, returns are weighted by the relative trajectory probabilities:
$$\rho_{t:T-1} = \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k) p(S_{k+1}|S_k, A_k)}{\prod_{k=t}^{T-1} b(A_k|S_k) p(S_{k+1}|S_k, A_k)} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

*Note: Environment transition probabilities $p(S_{k+1}|S_k, A_k)$ cancel out completely, keeping the ratio model-free.*

### 3. Ordinary vs. Weighted Importance Sampling
Given $n$ returns $\{G_1, G_2, \dots, G_n\}$ following visits to state $s$ with corresponding importance sampling weights $W_i = \rho_{t(i):T(i)-1}$:

* **Ordinary Importance Sampling**:
  $$V(s) \doteq \frac{\sum_{i=1}^{n} W_i G_i}{n}$$
  * **Properties**: **Unbiased** ($\mathbb{E}[V(s)] = v_\pi(s)$), but suffers from **unbounded / extreme variance** over long trajectories.
* **Weighted Importance Sampling**:
  $$V(s) \doteq \frac{\sum_{i=1}^{n} W_i G_i}{\sum_{i=1}^{n} W_i}$$
  * **Properties**: **Biased** initially (bias approaches 0 as $n \to \infty$), but exhibits **significantly lower variance**. Preferred in practice.

### 4. Incremental Update Formula
Given a sequence of returns $G_1, \dots, G_n$ with weights $W_1, \dots, W_n$, let $C_n = \sum_{i=1}^{n} W_i$ be the cumulative sum of weights for the first $n$ returns. The weighted importance sampling estimate $V_n$ after $n$ samples can be updated incrementally as:
$$V_n = V_{n-1} + \frac{W_n}{C_n} \big[ G_n - V_{n-1} \big] \quad (n \ge 1), \quad \text{where } C_n = C_{n-1} + W_n$$


### Pseudocode: Off-policy MC Prediction

```text
Off-policy MC prediction (policy evaluation) for estimating Q ≈ q_π

Input: an arbitrary target policy π
Initialize, for all s ∈ S, a ∈ A(s):
    Q(s, a) ∈ ℝ (arbitrarily)
    C(s, a) ← 0

Loop forever (for each episode):
    b ← any policy with coverage of π
    Generate an episode following b: S_0, A_0, R_1, . . . , S_{T-1}, A_{T-1}, R_T
    G ← 0
    W ← 1
    Loop for each step of episode, t = T-1, T-2, . . . , 0, while W ≠ 0:
        G ← γG + R_{t+1}
        C(S_t, A_t) ← C(S_t, A_t) + W
        Q(S_t, A_t) ← Q(S_t, A_t) + W / C(S_t, A_t) * [G - Q(S_t, A_t)]
        W ← W * π(A_t|S_t) / b(A_t|S_t)
```


## 6. Off-policy MC Control

Off-policy MC Control solves the optimal control problem ($\pi \approx \pi_*$) by evaluating and optimizing a **deterministic greedy target policy** $\pi$ while following a **soft exploratory behavior policy** $b$ (e.g., $\epsilon$-soft).

### 1. Key Principles & Advantages
* **Greedy Target Policy**: $\pi(s) = \arg\max_a Q(s, a)$. Target policy does not need to explore; it focuses purely on becoming optimal.
* **Exploratory Behavior Policy**: $b$ is a soft policy (e.g., $\epsilon$-greedy) ensuring coverage of all state-action pairs.
* **Simplified Weight Updates & Early Exit**:
  Since $\pi$ is deterministic greedy:
  $$\pi(A_t|S_t) = \begin{cases} 1 & \text{if } A_t = \pi(S_t) \\ 0 & \text{if } A_t \neq \pi(S_t) \end{cases}$$
  * If $A_t = \pi(S_t)$, the importance sampling weight is updated via $W \leftarrow W \frac{1}{b(A_t|S_t)}$.
  * If $A_t \neq \pi(S_t)$ (the behavior policy selected a non-greedy action), $W$ becomes $0$. Since all subsequent products for earlier timesteps would also evaluate to $0$, the inner loop **exits early** to save computation.

### Pseudocode: Off-policy MC Control

```text
Off-policy MC control, for estimating π ≈ π_*

Initialize, for all s ∈ S, a ∈ A(s):
    Q(s, a) ∈ ℝ (arbitrarily)
    C(s, a) ← 0
    π(s) ← argmax_a Q(s, a)    (with ties broken consistently)

Loop forever (for each episode):
    b ← any soft policy
    Generate an episode using b: S_0, A_0, R_1, . . . , S_{T-1}, A_{T-1}, R_T
    G ← 0
    W ← 1
    Loop for each step of episode, t = T-1, T-2, . . . , 0:
        G ← γG + R_{t+1}
        C(S_t, A_t) ← C(S_t, A_t) + W
        Q(S_t, A_t) ← Q(S_t, A_t) + W / C(S_t, A_t) * [G - Q(S_t, A_t)]
        π(S_t) ← argmax_a Q(S_t, a)    (with ties broken consistently)
        If A_t ≠ π(S_t) then exit inner Loop (proceed to next episode)
        W ← W * 1 / b(A_t|S_t)
```


## 7. Summary

* **Model-Free Learning**: MC methods eliminate the need for prior knowledge of environment dynamics ($P(s', r | s, a)$), estimating value functions and discovering optimal policies directly from sample trajectories.
* **Episodic Limitation**: Updates occur **episode-by-episode** (offline after episode completion) because value estimation relies on calculating the complete actual return.
* **Unbiased vs. High Variance**: Sample returns $G_t$ provide unbiased value estimates (in First-Visit MC), but exhibit high variance due to accumulated randomness across entire action-state trajectories.
* **Exploration Strategies**:
  * **Exploring Starts**: Assumes non-zero probability of starting at any state-action pair (theoretical baseline, impractical in practice).
  * **On-Policy ($\epsilon$-soft / $\epsilon$-greedy)**: Maintains continuous exploration by ensuring all actions in the decision-making policy have non-zero probabilities.
  * **Off-Policy (Importance Sampling)**: Separates exploration from learning by evaluating/optimizing a deterministic greedy target policy ($\pi$) using data generated by a soft behavior policy ($b$).