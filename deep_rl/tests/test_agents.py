import gymnasium as gym
import numpy as np
import pytest
import torch

from deep_rl.continuous_control.ddpg import DDPGAgent
from deep_rl.continuous_control.sac import SACAgent
from deep_rl.continuous_control.td3 import TD3Agent
from deep_rl.policy_gradient.ppo import PPO


def make_ppo(seed=0):
    return PPO(state_dim=4, action_dim=2, hidden_dims=(16, 16), rollout_steps=32,
               update_epochs=2, batch_size=16, seed=seed)


def test_ppo_checkpoint_roundtrip(tmp_path):
    source = make_ppo(seed=0)
    path = tmp_path / "ppo.pt"
    source.save_checkpoint(path)

    restored = make_ppo(seed=1)  # different init so the load must change weights
    restored.load_checkpoint(path)

    for name in ("actor", "critic"):
        expected = getattr(source, name).state_dict()
        actual = getattr(restored, name).state_dict()
        assert expected.keys() == actual.keys()
        for key in expected:
            assert torch.equal(expected[key], actual[key])

    state = torch.randn(3, 4)
    dist_a, value_a = source._distribution_and_value(state)
    dist_b, value_b = restored._distribution_and_value(state)
    assert torch.allclose(dist_a.logits, dist_b.logits)
    assert torch.allclose(value_a, value_b)


def test_ppo_learn_smoke():
    env = gym.make("CartPole-v1")
    agent = make_ppo()
    metrics = agent.learn(env, max_steps=96)
    env.close()
    assert set(metrics) == {"policy_loss", "value_loss", "entropy"}
    assert all(np.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize("agent_cls", [DDPGAgent, TD3Agent, SACAgent])
def test_continuous_control_learn_smoke(agent_cls):
    env = gym.make("Pendulum-v1")
    agent = agent_cls(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=float(env.action_space.high[0]),
        buffer_size=1_000,
        batch_size=16,
        start_steps=20,
    )
    agent.learn(env, max_steps=60)
    action = agent.select_action(env.reset(seed=0)[0])
    env.close()
    assert np.asarray(action).shape == env.action_space.shape
    assert np.all(np.isfinite(action))
