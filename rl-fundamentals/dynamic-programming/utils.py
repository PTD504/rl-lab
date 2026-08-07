"""
Visualization utilities for Dynamic Programming algorithms in GridWorld.

This module provides plotting functions for visualizing state-value functions V(s),
policy action arrows pi(s), combined value-policy overlays, and convergence speed
comparisons (Policy Iteration vs. Value Iteration) in Sutton & Barto style.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Type aliases
State = Tuple[int, int]
ValueDict = Dict[State, float]
PolicyDict = Dict[State, Union[str, List[str]]]
ConvergenceHistory = Dict[str, Dict[str, List[float]]]


def _get_arrow_direction(action: str) -> Tuple[float, float]:
    """
    Map an action string to (dx, dy) direction vector in image grid coordinates.

    Note: In matrix/image coordinates (origin='upper'), y increases downward (row index),
    so 'up' corresponds to dy = -1 and 'down' corresponds to dy = 1.

    Parameters
    ----------
    action : str
        One of 'up', 'down', 'left', 'right'.

    Returns
    -------
    dx, dy : tuple[float, float]
        Direction displacement vector.
    """
    mapping = {
        "up": (0.0, -1.0),
        "down": (0.0, 1.0),
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
    }
    if action not in mapping:
        raise ValueError(f"Unknown action '{action}'. Expected one of {list(mapping.keys())}.")
    return mapping[action]


def _get_grid_scale_params(n_rows: int, n_cols: int) -> Dict[str, Any]:
    """
    Calculate dynamic font sizes, line widths, and arrow scales based on grid dimensions.

    Parameters
    ----------
    n_rows, n_cols : int
        Grid dimensions.

    Returns
    -------
    dict
        Scaling parameters for matplotlib rendering.
    """
    grid_dim = max(n_rows, n_cols)
    scale = max(0.45, min(1.0, 5.0 / grid_dim))
    return {
        "scale": scale,
        "mutation_scale": max(7.0, min(16.0, 16.0 * scale)),
        "lw": max(1.0, min(2.0, 2.0 * scale)),
        "value_fontsize": max(6, int(10 * scale)),
        "policy_value_fontsize": max(6, int(8 * scale)),
        "terminal_fontsize": max(7, int(11 * scale)),
        "policy_terminal_fontsize": max(8, int(13 * scale)),
    }


def plot_value_grid(
    grid_world: Any,
    V: ValueDict,
    ax: Optional[plt.Axes] = None,
    cmap: str = "YlGnBu",
    title: str = "State-Value Function V(s)",
    fig_size: Tuple[int, int] = (6, 6),
    dpi: int = 150,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap of state values V(s) over the GridWorld grid.

    Example
    -------
    >>> fig, ax = plot_value_grid(grid_world, V)

    Parameters
    ----------
    grid_world : GridWorld
        GridWorld environment instance.
    V : dict[tuple[int, int], float]
        Mapping from non-terminal (and optionally terminal) states to state values.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw onto. If None, a new figure and axes are created.
    cmap : str, default='YlGnBu'
        Colormap for heatmap values.
    title : str, default='State-Value Function V(s)'
        Plot title.
    fig_size : tuple[int, int], default=(6, 6)
        Figure size when ax is None. Scales automatically for larger grids if left at default.
    dpi : int, default=150
        DPI resolution when ax is None.

    Returns
    -------
    fig, ax : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    created_fig = ax is None
    n_rows, n_cols = grid_world.n_rows, grid_world.n_cols
    scale_params = _get_grid_scale_params(n_rows, n_cols)

    if ax is None:
        if fig_size == (6, 6):
            fig_size = (max(6, int(n_cols * 0.75)), max(6, int(n_rows * 0.75)))
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    else:
        fig = ax.get_figure()

    # Construct 2D array for heatmap
    if isinstance(V, np.ndarray):
        grid_values = V.copy()
    else:
        grid_values = np.zeros((n_rows, n_cols))
        for (r, c) in grid_world.states:
            grid_values[r, c] = V.get((r, c), 0.0)

    # Plot heatmap
    im = ax.imshow(grid_values, cmap=cmap, origin="upper", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Setup grid lines and ticks
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=scale_params["lw"])
    ax.tick_params(which="minor", size=0)

    # Normalize color contrast for text
    val_min, val_max = grid_values.min(), grid_values.max()
    val_range = val_max - val_min if val_max != val_min else 1.0

    # Draw cell text annotations & highlight terminals
    for (r, c) in grid_world.states:
        if grid_world.is_terminal((r, c)):
            rect = patches.Rectangle(
                (c - 0.5, r - 0.5),
                1.0,
                1.0,
                facecolor="lightgray",
                edgecolor="black",
                hatch="//",
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(
                c,
                r,
                "0.0\n(T)",
                ha="center",
                va="center",
                fontsize=scale_params["terminal_fontsize"],
                fontweight="bold",
                color="black",
                zorder=3,
            )
        else:
            val = grid_values[r, c]
            norm_val = (val - val_min) / val_range
            text_color = "white" if norm_val > 0.65 else "black"
            text_str = f"{val:.2f}" if abs(val % 1) > 1e-4 else f"{val:.1f}"
            ax.text(
                c,
                r,
                text_str,
                ha="center",
                va="center",
                fontsize=scale_params["value_fontsize"],
                fontweight="bold",
                color=text_color,
                zorder=3,
            )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    if created_fig:
        plt.tight_layout()

    return fig, ax


def plot_policy_arrows(
    grid_world: Any,
    policy: PolicyDict,
    ax: Optional[plt.Axes] = None,
    title: str = "Policy Actions π(s)",
    fig_size: Tuple[int, int] = (6, 6),
    dpi: int = 150,
    arrow_color: str = "#1e3a8a",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot grid world with directional policy arrows for non-terminal states.

    Example
    -------
    >>> fig, ax = plot_policy_arrows(grid_world, policy)

    Parameters
    ----------
    grid_world : GridWorld
        GridWorld environment instance.
    policy : dict[tuple[int, int], str | list[str]]
        Mapping from non-terminal state to single action or list of tied optimal actions.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw onto. If None, a new figure and axes are created.
    title : str, default='Policy Actions π(s)'
        Plot title.
    fig_size : tuple[int, int], default=(6, 6)
        Figure size when ax is None. Scales automatically for larger grids if left at default.
    dpi : int, default=150
        DPI resolution when ax is None.
    arrow_color : str, default='#1e3a8a'
        Color of policy arrows. High contrast deep blue by default.

    Returns
    -------
    fig, ax : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    created_fig = ax is None
    n_rows, n_cols = grid_world.n_rows, grid_world.n_cols
    scale_params = _get_grid_scale_params(n_rows, n_cols)

    if ax is None:
        if fig_size == (6, 6):
            fig_size = (max(6, int(n_cols * 0.75)), max(6, int(n_rows * 0.75)))
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    else:
        fig = ax.get_figure()

    # Draw white background grid (Greys_r vmin=0, vmax=1 gives clean white background)
    background = np.ones((n_rows, n_cols))
    ax.imshow(background, cmap="Greys_r", origin="upper", aspect="equal", vmin=0, vmax=1)

    # Setup grid lines and ticks
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linestyle="-", linewidth=scale_params["lw"])
    ax.tick_params(which="minor", size=0)

    # Render arrows or terminal indicators
    for (r, c) in grid_world.states:
        if grid_world.is_terminal((r, c)):
            rect = patches.Rectangle(
                (c - 0.5, r - 0.5),
                1.0,
                1.0,
                facecolor="#e5e7eb",
                edgecolor="black",
                hatch="//",
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(
                c,
                r,
                "T",
                ha="center",
                va="center",
                fontsize=scale_params["policy_terminal_fontsize"],
                fontweight="bold",
                color="black",
                zorder=3,
            )
        else:
            actions = policy.get((r, c), [])
            if isinstance(actions, str):
                actions = [actions]

            n_acts = len(actions)
            for action in actions:
                dx, dy = _get_arrow_direction(action)

                if n_acts == 1:
                    # Single action: centered long arrow
                    start_x, start_y = c - dx * 0.22, r - dy * 0.22
                    head_x, head_y = c + dx * 0.28, r + dy * 0.28
                    mut_scale = scale_params["mutation_scale"]
                else:
                    # Tied/multiple actions: radiate outwards from near center to avoid overlap
                    start_x, start_y = c + dx * 0.05, r + dy * 0.05
                    head_x, head_y = c + dx * 0.32, r + dy * 0.32
                    mut_scale = max(7.0, scale_params["mutation_scale"] * 0.9)

                ax.annotate(
                    "",
                    xy=(head_x, head_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        mutation_scale=mut_scale,
                        lw=scale_params["lw"],
                        color=arrow_color,
                    ),
                    zorder=4,
                )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    if created_fig:
        plt.tight_layout()

    return fig, ax


def plot_value_and_policy(
    grid_world: Any,
    V: ValueDict,
    policy: PolicyDict,
    ax: Optional[plt.Axes] = None,
    cmap: str = "YlGnBu",
    title: str = "State-Value Function V(s) & Optimal Policy π(s)",
    fig_size: Tuple[int, int] = (7, 7),
    dpi: int = 150,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot state values V(s) heatmap overlaid with policy directional arrows.

    Example
    -------
    >>> fig, ax = plot_value_and_policy(grid_world, V, policy)

    Parameters
    ----------
    grid_world : GridWorld
        GridWorld environment instance.
    V : dict[tuple[int, int], float]
        Mapping from states to state values.
    policy : dict[tuple[int, int], str | list[str]]
        Mapping from non-terminal state to single action or list of tied optimal actions.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw onto. If None, a new figure and axes are created.
    cmap : str, default='YlGnBu'
        Colormap for value function heatmap.
    title : str, default='State-Value Function V(s) & Optimal Policy π(s)'
        Plot title.
    fig_size : tuple[int, int], default=(7, 7)
        Figure size when ax is None. Scales automatically for larger grids if left at default.
    dpi : int, default=150
        DPI resolution when ax is None.

    Returns
    -------
    fig, ax : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    created_fig = ax is None
    n_rows, n_cols = grid_world.n_rows, grid_world.n_cols
    scale_params = _get_grid_scale_params(n_rows, n_cols)

    if ax is None:
        if fig_size == (7, 7):
            fig_size = (max(7, int(n_cols * 0.75)), max(7, int(n_rows * 0.75)))
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    else:
        fig = ax.get_figure()

    # Construct 2D array for heatmap
    if isinstance(V, np.ndarray):
        grid_values = V.copy()
    else:
        grid_values = np.zeros((n_rows, n_cols))
        for (r, c) in grid_world.states:
            grid_values[r, c] = V.get((r, c), 0.0)

    # Draw heatmap
    im = ax.imshow(grid_values, cmap=cmap, origin="upper", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Setup grid lines and ticks
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=scale_params["lw"])
    ax.tick_params(which="minor", size=0)

    val_min, val_max = grid_values.min(), grid_values.max()
    val_range = val_max - val_min if val_max != val_min else 1.0

    # Draw value annotations and policy arrows
    for (r, c) in grid_world.states:
        if grid_world.is_terminal((r, c)):
            rect = patches.Rectangle(
                (c - 0.5, r - 0.5),
                1.0,
                1.0,
                facecolor="lightgray",
                edgecolor="black",
                hatch="//",
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(
                c,
                r,
                "0.0\n(T)",
                ha="center",
                va="center",
                fontsize=scale_params["terminal_fontsize"],
                fontweight="bold",
                color="black",
                zorder=3,
            )
        else:
            val = grid_values[r, c]
            norm_val = (val - val_min) / val_range
            text_color = "white" if norm_val > 0.65 else "black"
            arrow_color = "#fbbf24" if norm_val > 0.65 else "#991b1b"

            # Render numeric value slightly offset downward
            text_str = f"{val:.2f}" if abs(val % 1) > 1e-4 else f"{val:.1f}"
            ax.text(
                c,
                r + 0.28,
                text_str,
                ha="center",
                va="center",
                fontsize=scale_params["policy_value_fontsize"],
                fontweight="bold",
                color=text_color,
                zorder=3,
            )

            # Render policy arrows offset upward
            actions = policy.get((r, c), [])
            if isinstance(actions, str):
                actions = [actions]

            n_acts = len(actions)
            center_x, center_y = c, r - 0.12
            for action in actions:
                dx, dy = _get_arrow_direction(action)

                if n_acts == 1:
                    start_x, start_y = center_x - dx * 0.18, center_y - dy * 0.18
                    head_x, head_y = center_x + dx * 0.25, center_y + dy * 0.25
                    mut_scale = scale_params["mutation_scale"]
                else:
                    start_x, start_y = center_x + dx * 0.04, center_y + dy * 0.04
                    head_x, head_y = center_x + dx * 0.28, center_y + dy * 0.28
                    mut_scale = max(7.0, scale_params["mutation_scale"] * 0.85)

                ax.annotate(
                    "",
                    xy=(head_x, head_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        mutation_scale=mut_scale,
                        lw=scale_params["lw"],
                        color=arrow_color,
                    ),
                    zorder=4,
                )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    if created_fig:
        plt.tight_layout()

    return fig, ax


def plot_convergence_comparison(
    history: ConvergenceHistory,
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    title: str = "Convergence Speed: Policy Iteration vs. Value Iteration",
    fig_size: Tuple[int, int] = (8, 5),
    dpi: int = 150,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot convergence rate (max value change Delta) vs cumulative Bellman backups.

    Example
    -------
    >>> history = {
    ...     "Policy Iteration": {"backup_count": [25, 50, 75], "delta": [1.0, 0.5, 0.01]},
    ...     "Value Iteration": {"backup_count": [25, 50, 75], "delta": [1.0, 0.7, 0.4]}
    ... }
    >>> fig, ax = plot_convergence_comparison(history, log_scale=True)

    Parameters
    ----------
    history : dict[str, dict[str, list[float]]]
        Dictionary structured as:
        { "Algo Name": { "backup_count": [...], "delta": [...] }, ... }
    ax : matplotlib.axes.Axes, optional
        Axes object to draw onto. If None, a new figure and axes are created.
    log_scale : bool, default=True
        Whether to use log scale for Y-axis (delta).
    title : str, default='Convergence Speed: Policy Iteration vs. Value Iteration'
        Plot title.
    fig_size : tuple[int, int], default=(8, 5)
        Figure size when ax is None.
    dpi : int, default=150
        DPI resolution when ax is None.

    Returns
    -------
    fig, ax : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    else:
        fig = ax.get_figure()

    # Style adjustments matching Sutton & Barto clean layout
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(direction="in", length=5, width=1.2)

    markers = ["o", "s", "^", "v", "D"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (algo_name, data) in enumerate(history.items()):
        backup_count = data.get("backup_count", [])
        delta = data.get("delta", [])

        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]

        ax.plot(
            backup_count,
            delta,
            label=algo_name,
            marker=marker,
            markersize=5,
            linewidth=1.8,
            color=color,
        )

    ax.set_xlabel("Cumulative Bellman Backups", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Max Value Change (Δ)", fontsize=11, fontweight="bold", labelpad=8)

    if log_scale:
        ax.set_yscale("log")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False, fontsize=11, loc="best")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    if created_fig:
        plt.tight_layout()

    return fig, ax
