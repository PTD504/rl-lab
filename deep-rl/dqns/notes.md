# Deep Q-Network (DQN) and Its Variants

We have already seen Q-learning with function approximation, where a function is used to represent the value function instead of storing a table. When that function is a deep neural network, the result is usually called a Deep Q-Network, or DQN.

## Deep Q-Network

The main idea is simple: the network takes a state as input and produces one Q-value for each possible action. For a state $s$, the network estimates

$$
Q(s, a; \mathbf{w})
$$

for every action $a$, where $\mathbf{w}$ represents the network parameters. The agent then uses these estimates to choose actions, usually with an epsilon-greedy policy: sometimes it explores by choosing a random action, and otherwise it chooses the action with the highest estimated Q-value.

For an experience $(s, a, r, s')$, the Q-learning target is

$$
TD\_TARGET = r + \gamma (1 - d) \max_{a'} Q(s', a'; \mathbf{w^-})
$$

Here, $d$ is 1 when the transition ends the episode and 0 otherwise. The network is trained to reduce the difference between this target and its current estimate:

$$
L(\mathbf{w}) = \left(TD\_TARGET - Q(s, a; \mathbf{w})\right)^2
$$

In practice, DQN commonly uses the Huber loss instead of the plain squared loss because it is less sensitive to unusually large errors.

Using a neural network directly with online experience is unstable. DQN usually relies on two important techniques:

- **Replay buffer:** transitions are stored and sampled randomly in mini-batches. This breaks up the strong correlation between consecutive frames and allows useful experiences to be reused.
- **Target network:** a second network with parameters $\mathbf{w^-}$ is used to compute the target. It is updated only occasionally from the online network, so the target does not move after every single gradient step.

The network architecture depends on the observation. Classic-control environments usually return a vector, so an MLP is a natural choice. Atari environments return image frames, so a CNN is more appropriate. In the Atari setting, frames are commonly converted to grayscale, resized to $84 \times 84$, and stacked so the agent can infer motion from several consecutive observations.

## DQN's Variants

### Double DQN

DQN can suffer from **overestimation bias**. The problem comes from using the same noisy estimates both to select the best action and to evaluate it:

$$
TD\_TARGET_{DQN} = r + \gamma \max_{a'} Q(s', a'; \mathbf{w^-})
$$

If all estimates contain a little random error, taking a maximum tends to select an action whose error happens to be positive. The maximum can therefore be larger than the true value on average. This is the same general issue that appears in tabular Q-learning, but neural-network approximation can make it more noticeable.

Double DQN separates action selection from action evaluation. The online network chooses the next action, while the target network evaluates that chosen action:

$$
a^* = \arg\max_{a'} Q(s', a'; \mathbf{w})
$$

$$
TD\_TARGET_{Double} = r + \gamma Q(s', a^*; \mathbf{w^-})
$$

The two networks are not perfectly independent because the target network is periodically copied from the online network, but they are different enough to reduce the tendency to select an action because of a lucky overestimate. The target network is no longer asked to both find and judge the maximum using the same estimates.

Apart from this target calculation, Double DQN is very similar to DQN: it still uses a replay buffer, a target network, an epsilon-greedy policy, and gradient updates on the online network.

### Dueling DQN

Dueling DQN changes the network architecture rather than the basic Q-learning update. It splits the final part of the network into two streams:

- **Value stream:** estimates how valuable the state is, regardless of the particular action, $V(s)$.
- **Advantage stream:** estimates how much better or worse each action is compared with the other actions in that state, $A(s, a)$.

The two streams are combined to produce Q-values. The basic idea is

$$
Q(s, a) = V(s) + A(s, a)
$$

However, this decomposition is not unique. For example, adding a constant to $V(s)$ and subtracting the same constant from every advantage would produce exactly the same Q-values. To make the representation identifiable, Dueling DQN subtracts the mean advantage:

$$
Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right)
$$

This architecture is useful when many actions have similar consequences. The network can learn the value of the state through the value stream without needing to learn a completely separate estimate for every action. It can then focus the advantage stream on the actions that really matter.

Dueling DQN does not by itself change the DQN target or remove overestimation bias. It is an architectural improvement, while Double DQN is mainly a target-calculation improvement. They can also be combined: one network can use a dueling architecture and still use the Double DQN target.

## How the Variants Relate

The three agents share the same general learning loop:

1. Observe the current state and choose an action.
2. Store the transition in the replay buffer.
3. Sample a mini-batch and compute a bootstrapped target.
4. Update the online network with gradient descent.
5. Periodically copy the online network to the target network.

Their main difference is where they improve the original DQN:

- **DQN:** uses the target network to select and evaluate the best next action.
- **Double DQN:** uses the online network to select the next action and the target network to evaluate it.
- **Dueling DQN:** represents each state with separate value and advantage streams before combining them into Q-values.

These techniques address different difficulties, so they are complementary rather than competing alternatives. Double DQN focuses on more reliable targets, while Dueling DQN focuses on a more informative representation of states and actions.