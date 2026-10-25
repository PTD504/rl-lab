"""Utilities for running trained policies and saving rendered episodes.

The environment must be created with ``render_mode="rgb_array"``.  The
functions in this module do not call ``env.render`` with another mode, so they
can also be used from a notebook without opening a separate display window.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch


def _resolve_network(policy: Any, attribute: str | None = None) -> Any:
	"""Accept either a network or an agent containing the network."""
	if attribute is not None and hasattr(policy, attribute):
		return getattr(policy, attribute)
	return policy


def _reset_environment(env: Any, seed: int | None) -> np.ndarray:
	result = env.reset(seed=seed) if seed is not None else env.reset()
	state = result[0] if isinstance(result, tuple) else result
	return np.asarray(state)


def _step_environment(
	env: Any,
	action: int,
) -> tuple[np.ndarray, float, bool, bool, bool, dict[str, Any]]:
	result = env.step(action)
	if len(result) == 5:
		state, reward, terminated, truncated, info = result
		done = bool(terminated or truncated)
	else:
		state, reward, done, info = result
		done = bool(done)
		terminated = done
		truncated = False
	return (
		np.asarray(state),
		float(reward),
		done,
		bool(terminated),
		bool(truncated),
		dict(info),
	)


def _current_lives(env: Any) -> int | None:
	ale = getattr(getattr(env, "unwrapped", None), "ale", None)
	if ale is None:
		return None
	return int(ale.lives())


def _capture_frame(env: Any) -> np.ndarray:
	frame = env.render()
	if frame is None:
		raise ValueError(
			"env.render() returned None. Create the environment with "
			"render_mode='rgb_array'."
		)
	frame = np.asarray(frame)
	if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
		raise ValueError("env.render() must return an RGB or RGBA image.")
	return frame[..., :3].astype(np.uint8, copy=False)


def _save_frames(frames: list[np.ndarray], output_path: str | Path, fps: int) -> Path:
	if not frames:
		raise ValueError("Cannot save a video without rendered frames.")

	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	suffix = path.suffix.lower()
	if suffix == ".gif":
		imageio.mimsave(path, frames, duration=1 / fps, loop=0)
	elif suffix in {".mp4", ".m4v"}:
		with imageio.get_writer(path, format="FFMPEG", mode="I", fps=fps) as writer:
			for frame in frames:
				writer.append_data(frame)
	else:
		raise ValueError("output_path must end with .gif, .mp4, or .m4v.")
	return path


def _finish_episode(
	frames: list[np.ndarray],
	output_path: str | Path | None,
	fps: int,
	total_reward: float,
	steps: int,
) -> dict[str, Any]:
	path = _save_frames(frames, output_path, fps) if output_path is not None else None
	return {
		"frames": frames,
		"output_path": path,
		"return": total_reward,
		"steps": steps,
	}


def render_dqn_episode(
	env: Any,
	network: Any,
	output_path: str | Path | None = None,
	*,
	device: str | torch.device = "cpu",
	epsilon: float = 0.0,
	frame_stack: int = 4,
	initial_action: int | None = None,
	life_loss_action: int | None = None,
	max_steps: int | None = None,
	fps: int = 30,
	seed: int | None = None,
) -> dict[str, Any]:
	"""Run and render one episode using a DQN network.

	``network`` may be a PyTorch module or a DQN agent with a ``network``
	attribute.  DQN states are stacked along the channel dimension and scaled
	by 255, matching the training code in this repository. ``initial_action``
	can be used for Atari games such as Breakout that require a FIRE action
	before the learned policy can control the game. ``life_loss_action`` is
	used once after an Atari life is lost, allowing Breakout to start its next
	life without treating that life loss as episode termination.
	"""
	if not 0 <= epsilon <= 1:
		raise ValueError("epsilon must be between 0 and 1.")
	if frame_stack < 1:
		raise ValueError("frame_stack must be at least 1.")

	model = _resolve_network(network, "network")
	model_device = torch.device(device)
	model = model.to(model_device)
	was_training = model.training
	model.eval()

	state = _reset_environment(env, seed)
	frames = [_capture_frame(env)]
	state_frames = deque([state] * frame_stack, maxlen=frame_stack)
	total_reward = 0.0
	steps = 0
	done = False
	terminated = False
	truncated = False
	last_info: dict[str, Any] = {}
	previous_lives = _current_lives(env)
	pending_life_loss_action = False
	action_count = getattr(network, "action_dim", None)
	if action_count is None:
		action_count = int(
			model(
				torch.zeros(
					(1, frame_stack, *state.shape[-2:]),
					device=model_device,
				)
			).shape[-1]
		)

	try:
		while not done and (max_steps is None or steps < max_steps):
			if steps == 0 and initial_action is not None:
				action = int(initial_action)
				if not 0 <= action < action_count:
					raise ValueError(
						f"initial_action must be between 0 and {action_count - 1}."
					)
			elif pending_life_loss_action:
				action = int(life_loss_action)
				pending_life_loss_action = False
			elif np.random.random() < epsilon:
				action = int(np.random.randint(action_count))
			else:
				model_input = torch.from_numpy(np.stack(state_frames)).unsqueeze(0)
				model_input = model_input.to(model_device, dtype=torch.float32) / 255.0
				with torch.inference_mode():
					q_values = model(model_input)
				action = int(q_values.argmax(dim=-1).item())

			state, reward, done, terminated, truncated, last_info = _step_environment(
				env,
				action,
			)
			current_lives = _current_lives(env)
			if (
				life_loss_action is not None
				and previous_lives is not None
				and current_lives is not None
				and current_lives < previous_lives
				and not done
			):
				if not 0 <= life_loss_action < action_count:
					raise ValueError(
						f"life_loss_action must be between 0 and {action_count - 1}."
					)
				pending_life_loss_action = True
				state_frames = deque([state] * frame_stack, maxlen=frame_stack)
			else:
				state_frames.append(state)
			previous_lives = current_lives
			frames.append(_capture_frame(env))
			total_reward += reward
			steps += 1
	finally:
		model.train(was_training)

	result = _finish_episode(frames, output_path, fps, total_reward, steps)
	result.update(
		{
			"terminated": terminated,
			"truncated": truncated,
			"final_lives": _current_lives(env),
			"info": last_info,
		}
	)
	return result


def render_policy_episode(
	env: Any,
	policy: Any,
	output_path: str | Path | None = None,
	*,
	device: str | torch.device = "cpu",
	deterministic: bool = True,
	max_steps: int | None = None,
	fps: int = 30,
	seed: int | None = None,
) -> dict[str, Any]:
	"""Run and render one episode using a policy-gradient actor network.

	``policy`` may be a policy module, a REINFORCE agent with ``pi``, or an
	Actor-Critic agent with ``actor``.  The policy is expected to return action
	probabilities, as does :class:`utils.mlp.MLP` in this repository.
	"""
	model = policy
	for attribute in ("pi", "actor"):
		if hasattr(model, attribute):
			model = getattr(model, attribute)
			break
	model_device = torch.device(device)
	model = model.to(model_device)
	was_training = model.training
	model.eval()

	state = _reset_environment(env, seed)
	frames = [_capture_frame(env)]
	total_reward = 0.0
	steps = 0
	done = False

	try:
		while not done and (max_steps is None or steps < max_steps):
			state_tensor = torch.as_tensor(state, dtype=torch.float32, device=model_device).unsqueeze(0)
			with torch.inference_mode():
				action_values = model(state_tensor).squeeze(0)
				if deterministic:
					action = int(action_values.argmax().item())
				else:
					probabilities = action_values / action_values.sum()
					action = int(torch.multinomial(probabilities, 1).item())

			state, reward, done, _, _, _ = _step_environment(env, action)
			frames.append(_capture_frame(env))
			total_reward += reward
			steps += 1
	finally:
		model.train(was_training)

	return _finish_episode(frames, output_path, fps, total_reward, steps)


def render_continuous_episode(
	env: Any,
	agent: Any,
	output_path: str | Path | None = None,
	*,
	fps: int = 30,
	seed: int | None = None,
	max_steps: int | None = None,
	deterministic: bool = True,
) -> dict[str, Any]:
	"""Run and render one episode using a continuous-control agent (DDPG, TD3, SAC).

	The environment must be created with ``render_mode='rgb_array'``.
	"""
	state = _reset_environment(env, seed)
	frames = [_capture_frame(env)]
	total_reward = 0.0
	steps = 0
	done = False

	while not done and (max_steps is None or steps < max_steps):
		if hasattr(agent, "select_action"):
			import inspect
			params = inspect.signature(agent.select_action).parameters
			if "evaluate" in params:
				action = agent.select_action(state, evaluate=deterministic)
			elif "add_noise" in params:
				action = agent.select_action(state, add_noise=not deterministic)
			else:
				action = agent.select_action(state)
		elif hasattr(agent, "actor"):
			model = agent.actor
			model_device = next(model.parameters()).device
			state_tensor = torch.as_tensor(state, dtype=torch.float32, device=model_device).unsqueeze(0)
			with torch.inference_mode():
				action = model(state_tensor).cpu().numpy().squeeze(0)
		else:
			raise ValueError("agent must provide select_action or actor.")

		state, reward, done, _, _, _ = _step_environment(env, action)
		frames.append(_capture_frame(env))
		total_reward += reward
		steps += 1

	return _finish_episode(frames, output_path, fps, total_reward, steps)


def display_video(output_path: str | Path) -> Any:
	"""Return an IPython display object for a saved GIF or MP4 in a notebook."""
	from IPython.display import Image, Video

	path = str(output_path)
	if Path(path).suffix.lower() == ".gif":
		return Image(filename=path)
	return Video(filename=path, embed=True)

