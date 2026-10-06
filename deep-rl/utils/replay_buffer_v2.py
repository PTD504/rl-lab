import numpy as np
import torch


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
        self.latest_next_states[...] = next_state
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _stack_ids(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        ids = (indices[:, None] - np.arange(3, -1, -1)) % self.capacity
        oldest = self.ptr if self.size == self.capacity else 0
        distance = (indices - oldest) % self.capacity
        for column in range(2, -1, -1):
            steps_back = 3 - column
            offsets = np.arange(1, steps_back + 1)
            previous = (indices[:, None] - offsets) % self.capacity
            valid = (distance >= steps_back) & ~self.dones[previous].any(axis=1)
            ids[:, column] = np.where(valid, ids[:, column], ids[:, column + 1])
        return ids

    def _build_state_stacks(self, indices):
        return self.frames[self._stack_ids(indices)]

    def _get_frame_stack(self, index):
        return self._build_state_stacks(np.asarray([index]))[0]

    def _get_next_frame_stack(self, index):
        state = self._build_state_stacks(np.asarray([index]))
        next_index = (index + 1) % self.capacity
        next_frame = np.where(index == self.latest_ptr, self.latest_next_states, self.frames[next_index])
        return np.concatenate((state[:, 1:], next_frame[None, None]), axis=1)[0]

    def get_acting_stack(self, current_frame):
        current_frame = np.asarray(current_frame, dtype=np.uint8)
        if self.size == 0:
            return np.broadcast_to(current_frame, (4, 84, 84)).copy()
        latest = (self.ptr - 1) % self.capacity
        if self.dones[latest]:
            return np.broadcast_to(current_frame, (4, 84, 84)).copy()
        previous = self._build_state_stacks(np.asarray([latest]))[0]
        return np.concatenate((previous[1:], current_frame[None]), axis=0)

    def sample(self):
        indices = torch.randint(0, self.size, size=(self.batch_size,), generator=self.rng).numpy()
        states = self._build_state_stacks(indices)
        next_frames = self.frames[(indices + 1) % self.capacity]
        mask = indices == self.latest_ptr
        if mask.any():
            next_frames[mask] = self.latest_next_states
        next_states = np.concatenate((states[:, 1:], next_frames[:, None]), axis=1)
        return (
            torch.from_numpy(states),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(next_states),
            torch.from_numpy(self.dones[indices]),
        )

    def __len__(self):
        return self.size