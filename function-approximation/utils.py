"""
Visualization utilities for comparing reinforcement learning agents across multiple seeds.

Expected `results` data structure:
----------------------------------
`results` is a nested dictionary mapping environment ID and agent name to a list of runs:

    results = {
        "<env_id>": {
            "<agent_name>": [
                {
                    "history": {
                        "cumulative_steps": [...],
                        "episode_returns": [...],
                        "episode_lengths": [...],
                        "losses": [...],
                        "max_abs_td_error": [...],
                        ...
                    }
                },
                ...  # Additional independent runs/seeds
            ]
        }
    }

Each element in `results[env_id][agent_name]` represents one training run with its `"history"`
dictionary containing lists of equal length per episode.
"""

from functools import partial
import matplotlib.pyplot as plt
import numpy as np

# Central configuration mapping for all known metrics
METRIC_CONFIGS = {
    "episode_returns": {
        "title": "Training Return",
        "ylabel": "Episode Return",
        "yscale": "linear",
    },
    "episode_lengths": {
        "title": "Episode Length",
        "ylabel": "Steps per Episode",
        "yscale": "linear",
    },
    "losses": {
        "title": "TD-Error Loss",
        "ylabel": "Mean Squared TD Error",
        "yscale": "symlog",
    },
    "max_abs_td_error": {
        "title": "Max Absolute TD Error",
        "ylabel": "Max |TD Error|",
        "yscale": "symlog",
    },
}


def _confidence_summary(runs, x_key, y_key, points=200):
    """
    Computes interpolated mean and 95% confidence intervals across multiple seeds.
    Gracefully handles single-seed runs and invalid/empty values.
    """
    valid_runs = []
    for run in runs:
        history = run["history"]
        if x_key == "episodes":
            x = np.arange(1, len(history[y_key]) + 1, dtype=float)
        else:
            x = np.asarray(history[x_key], dtype=float)
        y = np.asarray(history[y_key], dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(mask) >= 2:
            valid_runs.append((x[mask], y[mask]))

    if not valid_runs:
        return np.array([]), np.array([]), np.array([])

    n_runs = len(valid_runs)
    if n_runs == 1:
        x, y = valid_runs[0]
        return x, y, np.zeros_like(y)

    x_min = max(x[0] for x, _ in valid_runs)
    x_max = min(x[-1] for x, _ in valid_runs)
    if x_max <= x_min:
        return np.array([]), np.array([]), np.array([])

    x_grid = np.linspace(x_min, x_max, points)
    values = np.vstack([np.interp(x_grid, x, y) for x, y in valid_runs])
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    confidence = 1.96 * std / np.sqrt(n_runs)
    return x_grid, mean, confidence


def _plot_metric_on_ax(ax, results, env_id, x_key, y_key, title, ylabel, yscale="linear"):
    """Helper to render a metric with confidence bands onto an axis."""
    xlabel = "Environment Steps" if x_key in ("cumulative_steps", "environment_steps") else "Episodes"
    for agent_name, runs in results[env_id].items():
        x, mean, confidence = _confidence_summary(runs, x_key, y_key)
        if len(x) == 0:
            continue
        line, = ax.plot(x, mean, label=agent_name, linewidth=1.8)
        if np.any(confidence > 0):
            ax.fill_between(x, mean - confidence, mean + confidence, alpha=0.2)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if yscale != "linear":
        ax.set_yscale(yscale, linthresh=1e-3 if yscale == "symlog" else None)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()


def plot_metric(results, env_id, y_key, x_key="cumulative_steps", ax=None):
    """
    Plots a single metric across multiple agents with 95% confidence intervals.
    Looks up visualization settings (title, ylabel, yscale) in METRIC_CONFIGS.
    """
    if y_key not in METRIC_CONFIGS:
        raise KeyError(
            f"Metric '{y_key}' is not configured in METRIC_CONFIGS. "
            f"Available metrics: {list(METRIC_CONFIGS.keys())}. "
            f"Please add a configuration entry with 'title', 'ylabel', and 'yscale' to METRIC_CONFIGS."
        )

    config = METRIC_CONFIGS[y_key]
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    title_config = config.get("title", y_key)
    if "{env_id}" in title_config:
        title = title_config.format(env_id=env_id)
    else:
        title = f"{title_config}: {env_id}"

    _plot_metric_on_ax(
        ax=ax,
        results=results,
        env_id=env_id,
        x_key=x_key,
        y_key=y_key,
        title=title,
        ylabel=config.get("ylabel", y_key),
        yscale=config.get("yscale", "linear"),
    )
    fig.tight_layout()
    return fig, ax


def plot_all_metrics(results, env_id, x_key="cumulative_steps", figsize=None):
    """
    Generates a vertically stacked dashboard displaying all metrics in METRIC_CONFIGS,
    with one metric per row for clear, comfortable scrolling and readability.
    """
    metric_keys = list(METRIC_CONFIGS.keys())
    n_metrics = len(metric_keys)
    n_cols = 1
    n_rows = n_metrics

    if figsize is None:
        figsize = (10, 3.5 * n_metrics)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, y_key in enumerate(metric_keys):
        plot_metric(results, env_id, y_key=y_key, x_key=x_key, ax=axes_flat[idx])

    fig.suptitle(f"Agent Comparison on {env_id}", fontsize=14, fontweight="bold", y=1.005)
    fig.tight_layout()
    return fig, axes


# Backward-compatible aliases for legacy functions
plot_training_returns = partial(plot_metric, y_key="episode_returns")
plot_episode_lengths = partial(plot_metric, y_key="episode_lengths")
plot_td_losses = partial(plot_metric, y_key="losses")
