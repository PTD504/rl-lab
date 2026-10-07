import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.replay_buffer_v1 import ReplayBuffer as V1ReplayBuffer
from utils.replay_buffer_v2 import ReplayBuffer as OptimizedReplayBuffer


def reference_stack(kept, index):
    episode_start = index
    while episode_start > 0 and not kept[episode_start - 1][4]:
        episode_start -= 1
    frames = [
        kept[max(episode_start, index - offset)][0]
        for offset in (3, 2, 1)
    ]
    return np.stack(frames + [kept[index][0]])


def reference_next_stack(kept, index):
    state = reference_stack(kept, index)
    return np.concatenate((state[1:], kept[index][3][None]), axis=0)


def make_transitions(lengths, open_tail=0):
    transitions = []
    frame_id = 1
    for length in lengths:
        for step in range(length):
            transitions.append(
                (
                    np.full((84, 84), frame_id, dtype=np.uint8),
                    frame_id,
                    float(frame_id),
                    np.full((84, 84), frame_id + 1, dtype=np.uint8),
                    step == length - 1,
                )
            )
            frame_id += 1
    if open_tail:
        if open_tail > len(transitions):
            raise ValueError("open_tail cannot exceed the number of transitions")
        for index in range(len(transitions) - open_tail, len(transitions)):
            transition = transitions[index]
            transitions[index] = (*transition[:4], False)
    return transitions


def chronological_slot(buffer, index):
    oldest = buffer.ptr if buffer.size == buffer.capacity else 0
    return (oldest + index) % buffer.capacity


def check_sample_against_reference(buffer, kept, require_latest=False):
    frames_before = buffer.frames.copy()
    states, actions, rewards, next_states, dones = buffer.sample()
    assert states.dtype == torch.uint8
    assert next_states.dtype == torch.uint8
    assert actions.dtype == torch.int64
    assert rewards.dtype == torch.float32
    assert dones.dtype == torch.bool

    by_id = {transition[1]: index for index, transition in enumerate(kept)}
    sampled_latest = False
    for sample_index in range(len(actions)):
        index = by_id[actions[sample_index].item()]
        np.testing.assert_array_equal(
            states[sample_index].numpy(), reference_stack(kept, index)
        )
        assert bool(dones[sample_index]) == kept[index][4]
        assert float(rewards[sample_index]) == kept[index][2]
        if not kept[index][4]:
            np.testing.assert_array_equal(
                next_states[sample_index].numpy(),
                reference_next_stack(kept, index),
            )
        if index == len(kept) - 1:
            sampled_latest = True
    np.testing.assert_array_equal(buffer.frames, frames_before)
    if require_latest:
        assert sampled_latest


def check_incremental(capacity, lengths, open_tail):
    transitions = make_transitions(lengths, open_tail)
    buffer = OptimizedReplayBuffer(
        capacity, 64, torch.Generator().manual_seed(0)
    )
    for count, transition in enumerate(transitions, 1):
        kept = transitions[:count][-capacity:]
        acting = buffer.get_acting_stack(transition[0])
        expected = reference_stack(kept, len(kept) - 1)
        np.testing.assert_array_equal(acting, expected)
        buffer.add(*transition)

        slot = (buffer.ptr - 1) % capacity
        np.testing.assert_array_equal(
            buffer._build_state_stacks(np.array([slot]))[0], expected
        )
        check_sample_against_reference(buffer, kept)


def test_incremental_cases():
    lengths = (1, 2, 3, 5, 4, 1, 7, 2)
    for capacity in (4, 5, 7, 8, 11, 16, 64):
        for open_tail in (0, 2):
            check_incremental(capacity, lengths, open_tail)
    check_incremental(8, (20,), 0)
    check_incremental(5, (1, 2, 3, 5, 4, 1, 7, 2) * 5, 3)


def test_incremental_property_cases():
    for seed in range(40):
        rng = np.random.default_rng(seed)
        lengths = tuple(rng.integers(1, 13, size=rng.integers(1, 9)))
        capacity = int(rng.integers(4, 21))
        open_tail = int(rng.integers(0, min(4, sum(lengths)) + 1))
        try:
            check_incremental(capacity, lengths, open_tail)
        except AssertionError as error:
            raise AssertionError(
                f"property case failed: seed={seed}, capacity={capacity}, "
                f"lengths={lengths}, open_tail={open_tail}"
            ) from error


def test_latest_next_state_branch():
    transitions = make_transitions((3,), open_tail=1)
    buffer = OptimizedReplayBuffer(
        4, 1, torch.Generator().manual_seed(0)
    )
    for transition in transitions:
        buffer.add(*transition)

    latest_id = transitions[-1][1]
    for seed in range(100):
        buffer.rng = torch.Generator().manual_seed(seed)
        if (
            buffer.sample()[1].item() == latest_id
        ):
            buffer.rng = torch.Generator().manual_seed(seed)
            check_sample_against_reference(
                buffer, transitions, require_latest=True
            )
            return
    raise AssertionError("could not sample latest_ptr in 100 deterministic seeds")


def test_get_acting_stack():
    current = np.full((84, 84), 231, dtype=np.uint8)
    for buffer_class in (V1ReplayBuffer, OptimizedReplayBuffer):
        empty = buffer_class(4, 1, torch.Generator())
        acting = empty.get_acting_stack(current)
        assert acting.shape == (4, 84, 84)
        assert acting.dtype == np.uint8
        acting[0, 0, 0] = 17
        assert current[0, 0] == 231

    transitions = make_transitions((3, 2), open_tail=1)
    for buffer_class in (V1ReplayBuffer, OptimizedReplayBuffer):
        buffer = buffer_class(4, 1, torch.Generator())
        for transition in transitions[:4]:
            buffer.add(*transition)
        np.testing.assert_array_equal(
            buffer.get_acting_stack(current),
            np.concatenate(
                (reference_stack(transitions[:4], 3)[1:], current[None]),
                axis=0,
            ),
        )

        terminal = buffer_class(4, 1, torch.Generator())
        terminal.add(*make_transitions((1,))[-1])
        np.testing.assert_array_equal(
            terminal.get_acting_stack(current),
            np.broadcast_to(current, (4, 84, 84)),
        )

    wrapped = OptimizedReplayBuffer(4, 1, torch.Generator())
    wrapped_transitions = make_transitions((1, 1, 1, 4), open_tail=2)
    for transition in wrapped_transitions:
        wrapped.add(*transition)
    acting = wrapped.get_acting_stack(current)
    assert acting.shape == (4, 84, 84)
    assert acting.flags.writeable
    acting[0, 0, 0] = 19


def test_v1_and_v2_equivalence_after_wrap():
    transitions = make_transitions((1, 2, 3, 5, 4, 1), open_tail=2)
    v1 = V1ReplayBuffer(7, 64, torch.Generator().manual_seed(11))
    v2 = OptimizedReplayBuffer(7, 64, torch.Generator().manual_seed(11))
    for transition in transitions:
        v1.add(*transition)
        v2.add(*transition)

    kept = transitions[-7:]
    slots = np.array(
        [chronological_slot(v1, index) for index in range(len(kept))]
    )
    np.testing.assert_array_equal(
        v1._build_state_stacks(slots), v2._build_state_stacks(slots)
    )

    v1_sample = v1.sample()
    v2_sample = v2.sample()
    np.testing.assert_array_equal(v1_sample[0].numpy(), v2_sample[0].numpy())
    np.testing.assert_array_equal(v1_sample[1].numpy(), v2_sample[1].numpy())
    np.testing.assert_array_equal(v1_sample[2].numpy(), v2_sample[2].numpy())
    np.testing.assert_array_equal(
        v1_sample[3].numpy()[~v1_sample[4].numpy()],
        v2_sample[3].numpy()[~v2_sample[4].numpy()],
    )
    np.testing.assert_array_equal(v1_sample[4].numpy(), v2_sample[4].numpy())


class LoopBenchmarkBuffer(OptimizedReplayBuffer):
    def sample(self):
        indices = torch.randint(
            0, self.size, (self.batch_size,), generator=self.rng
        ).numpy()
        states = np.stack([self._get_frame_stack(index) for index in indices])
        next_states = np.stack(
            [self._get_next_frame_stack(index) for index in indices]
        )
        return states, next_states


def make_random_transitions(count):
    rng = np.random.default_rng(23)
    frames = rng.integers(0, 256, (count + 1, 84, 84), dtype=np.uint8)
    dones = rng.random(count) < 0.05
    dones[-1] = False
    return [
        (frames[index], index, 0.0, frames[index + 1], bool(dones[index]))
        for index in range(count)
    ]


def benchmark():
    capacity = 100_000
    batch_size = 64
    transitions = make_random_transitions(capacity)
    loop = LoopBenchmarkBuffer(
        capacity, batch_size, torch.Generator().manual_seed(3)
    )
    optimized = OptimizedReplayBuffer(
        capacity, batch_size, torch.Generator().manual_seed(3)
    )
    fill(loop, transitions)
    fill(optimized, transitions)
    loop_states, loop_next = loop.sample()
    opt_states, _, _, opt_next, opt_dones = optimized.sample()
    nonterminal = ~opt_dones.numpy()
    np.testing.assert_array_equal(loop_states, opt_states.numpy())
    np.testing.assert_array_equal(loop_next[nonterminal], opt_next.numpy()[nonterminal])

    results = {}
    for name, buffer in (("before_loop", loop), ("after_vectorized", optimized)):
        for _ in range(10):
            buffer.sample()
        start = time.perf_counter()
        for _ in range(100):
            buffer.sample()
        results[name] = (time.perf_counter() - start) * 1000 / 100
    print(f"Benchmark: capacity={capacity}, batch={batch_size}, calls=100")
    for name, milliseconds in results.items():
        print(f"{name}: {milliseconds:.3f} ms/call")


def fill(buffer, transitions):
    for transition in transitions:
        buffer.add(*transition)


def test_buffers():
    test_incremental_cases()
    test_incremental_property_cases()
    test_latest_next_state_branch()
    test_get_acting_stack()
    test_v1_and_v2_equivalence_after_wrap()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    test_buffers()
    print("Replay buffer tests passed")
    if args.benchmark:
        benchmark()
