# Linear Function Approximation

## The problem with tabular methods

Many reinforcement learning algorithms store their solution in a table. A state-value table contains one value for each state, while a Q-value table contains one value for each state-action pair. When the state and action spaces are small and discrete, this representation is simple and useful.

The approach becomes impractical when the state space is very large or continuous. For example, an environment may return several real-valued observations. Even if each observation is bounded, the number of possible vectors is generally uncountably infinite, so an agent cannot create one table entry for every possible state. A large discrete problem can also have an enormous number of state-action pairs.

Function approximation addresses this problem by representing a value function with a parameterized function instead of storing one independent value for every state or state-action pair. The number of parameters can be much smaller than the number of possible inputs. The agent learns the parameters from experience and uses the function to estimate values for states it has not encountered exactly before.

## From a state-value function to a linear Q-function

A simple function approximator is a linear function:

$$
f(\mathbf{x}) = \mathbf{w}^{\mathsf{T}}\mathbf{x} + b
$$

where $\mathbf{x}$ is a feature vector, $\mathbf{w}$ is a vector of weights, and $b$ is a bias term. In expanded form,

$$
f(\mathbf{x}) = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b.
$$

For a state-value function, the input $\mathbf{x}$ can be the observation of the current state and the output is an estimate of $V(s)$. For action-value learning, the agent needs an estimate of $Q(s,a)$, not just one value for the state. With a discrete action space, a convenient implementation is to use a separate linear model for each action:

$$
Q(s,a) = \mathbf{w}_a^{\mathsf{T}}\mathbf{x}(s) + b_a.
$$

Here, $\mathbf{w}_a$ and $b_a$ are the weights and bias associated with action $a$. For one state, the agent evaluates this function for every available action and selects an action using a policy such as epsilon-greedy.

Consider an environment whose observation is a vector of ten real-valued features. A tabular method cannot allocate a separate entry for every possible observation. A linear approximator instead stores ten weights and one bias for each action. When a new observation arrives, the agent computes a dot product and obtains an estimated Q-value immediately, even if that exact observation has never appeared before.

It is important to distinguish approximation from compression of the raw state. The function does not replace the state with one permanent number. It maps the current state and action to an estimated value. Learning tries to choose parameters that make these estimates useful for predicting long-term return and selecting good actions.

## Learning with a linear approximator

In Q-learning, the temporal-difference target for a non-terminal transition is typically

$$
y = r + \gamma \max_{a'} Q(s', a'),
$$

and for a terminal transition the future-value term is omitted. The temporal-difference error is

$$
\delta = y - Q(s,a).
$$

For the linear model above, the gradient with respect to the parameters of the selected action is

$$
\nabla_{\mathbf{w}_a} Q(s,a) = \mathbf{x}(s),
\qquad
\frac{\partial Q(s,a)}{\partial b_a} = 1.
$$

An update in the direction of the TD error is therefore

$$
\mathbf{w}_a \leftarrow \mathbf{w}_a + \alpha\delta\mathbf{x}(s),
\qquad
b_a \leftarrow b_a + \alpha\delta.
$$

This is the same general idea as tabular Q-learning, but the update changes function parameters rather than one isolated table entry.

## Important properties

- **Generalization across states:** Changing one weight can change the predicted value for many states, because those states may share the same features. This allows the agent to learn from related observations instead of visiting every state separately.
- **Shared parameters:** The number of parameters depends on the feature dimension and the number of actions, not directly on the number of possible states. For a state dimension of $d$ and $k$ discrete actions, the model above has $k(d+1)$ parameters.
- **Coupled predictions:** The same parameter update affects all states whose feature vectors have a non-zero component in the updated direction. This is more memory-efficient than a table, but it also means that improving the estimate for one state can make estimates for other states worse.
- **Dependence on features:** The model can only use information that is present in the input features. A compact representation is useful only when it preserves the information needed to make good decisions.

## Limitations of linear function approximation

The main limitation is representational capacity. A linear model can express only an affine relationship between the input features and the predicted value for each action. It cannot naturally represent many nonlinear relationships, feature interactions, or threshold effects.

For example, a task may require an action to be valuable only when two features have a particular combination of values. A raw linear model can assign independent weights to the two features, but it does not directly represent their product or other interaction. It may therefore fail even when the state contains enough information in principle.

This limitation is especially relevant in control problems. The best action may depend on a nonlinear boundary in the state space, and the long-term effect of an action may depend on a sequence of states rather than on one feature independently. A linear approximator can still work well when the features are carefully engineered or when the true value function is close to linear, but raw observations are not guaranteed to satisfy either condition.

Function approximation also introduces a form of interference: one parameter update can alter predictions for many states and actions. Consequently, learning may become unstable or converge to a poor approximation even though the update rule itself is correct.

More expressive approximators, such as neural networks, can model richer relationships, but greater capacity does not automatically solve every reinforcement learning problem. Reward design, state representation, action representation, exploration, and the stability of the learning procedure all remain important.