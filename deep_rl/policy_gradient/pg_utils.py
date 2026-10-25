import torch
from torch.distributions import Categorical

def generate_episode(env, policy):
    """
    Generate an episode by interacting with the environment using the given policy.

    Args:
        env: The environment to interact with.
        policy: The policy to use for action selection.

    Returns:
        A list of (state, action, reward) tuples representing the episode.
    """
    log_probs = []
    rewards = []
    states = []
    state, _ = env.reset()
    done = False

    while not done:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        probs = policy(state_tensor)

        dist = Categorical(probs)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        states.append(state)
        rewards.append(reward)
        
        state = next_state
        done = terminated or truncated

    return log_probs, states, rewards