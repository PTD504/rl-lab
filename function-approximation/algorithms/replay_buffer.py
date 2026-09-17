import numpy as np
import torch

class ReplayBuffer:
    """
    A circular buffer to store transitions experienced by the agent.
    This design reduce the time complexity of deleting the oldest transition when the buffer is full from O(n) to O(1)
    If we use a deque, the time complexity of deleting the oldest transition when the buffer is full is O(1) but the time complexity of sampling a batch of transitions is increased to O(n * batch_size)
    """
    def __init__(self, buffer_size, state_dim, batch_size, rng):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.terminateds = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = rng

    def store_transition(self, state, action, reward, next_state, terminated):
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu().numpy()

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.terminateds[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample_batch(self):
        indices = torch.randint(0, self.size, size=(self.batch_size,), generator=self.rng).numpy()

        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.terminateds[indices])
        )

    def __len__(self):
        return self.size