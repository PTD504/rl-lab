import torch
import  torchvision.transforms as T
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, batch_size, rng):
        self.capacity = capacity
        self.batch_size = batch_size
        self.rng = rng
        self.frames = np.zeros((capacity, 84, 84), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.latest_next_states = np.zeros((84, 84), dtype=np.uint8)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.ptr = 0
        self.size = 0
        self.latest_ptr = None

    def add(self, state, action, reward, next_state, done):
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu().numpy()

        self.frames[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.latest_ptr = self.ptr
        self.latest_next_states = next_state

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_frame_stack(self, index):
        ids = [(index - i) % self.capacity for i in range(4)]

        for i in range(1, 4):
            if self.dones[ids[i]] or ids[i] == self.latest_ptr:
                for j in range(i + 1, 4):
                    ids[j] = ids[i - 1]
                break

        return np.stack([self.frames[i] for i in ids], axis=0)

    def _get_next_frame_stack(self, index):
        if index == self.latest_ptr:
            base = self._get_frame_stack(index)
            return np.concatenate([base[1:], self.latest_next_states[None]], axis=0)

        next_index = (index + 1) % self.capacity
        return self._get_frame_stack(next_index)

    def sample(self):
        indices = torch.randint(0, self.size, size=(self.batch_size,), generator=self.rng).numpy()
        
        states = np.stack([self._get_frame_stack(i) for i in indices])
        next_states = np.stack([self._get_next_frame_stack(i) for i in indices])

        return (
            torch.from_numpy(states),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(next_states),
            torch.from_numpy(self.dones[indices])
        )

    def get_three_latest_frames(self):
        return (self.frames[(self.latest_ptr - i) % self.capacity] for i in range(3))

    def __len__(self):
        return self.size