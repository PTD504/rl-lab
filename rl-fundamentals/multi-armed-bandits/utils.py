import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from typing import Dict, List, Optional, Tuple, Union

def plot_bandit_performance(
    metrics_data: Dict[str, Union[np.ndarray, List[float]]],
    metric_type: str = "reward",
    title: Optional[str] = None,
    xlabel: str = "Steps",
    ylabel: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    fig_size: Tuple[int, int] = (9, 5),
    dpi: int = 150
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots multi-armed bandit algorithm benchmarks in Sutton & Barto style.

    Parameters:
    -----------
    metrics_data : Dict[str, Union[np.ndarray, List[float]]]
        Dictionary where keys are algorithm labels (e.g., 'UCB c=2', 'e-greedy e=0.1')
        and values are 1D arrays/lists of length T representing values over time steps.
    metric_type : str, default='reward'
        Type of metric to plot: 'reward' for Average Reward or 'optimal_action' for % Optimal Action.
    title : Optional[str]
        Title of the plot.
    xlabel : str, default='Steps'
        Label for the X-axis.
    ylabel : Optional[str]
        Label for the Y-axis. Defaults based on metric_type if None.
    colors : Optional[Dict[str, str]]
        Mapping of algorithm label to matplotlib color hex/name.
    fig_size : Tuple[int, int], default=(9, 5)
        Figure dimensions.
    dpi : int, default=150
        Dots per inch for figure resolution.

    Returns:
    --------
    fig, ax : Matplotlib Figure and Axes objects.
    """
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    # Style adjustments to mirror Sutton & Barto aesthetic
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(direction='in', length=6, width=1.2)

    for label, values in metrics_data.items():
        steps = np.arange(1, len(values) + 1)
        color = colors.get(label) if colors else None
        
        # Scale 0-1 range to 0-100 for optimal action percentage if needed
        data_to_plot = np.array(values)
        if metric_type == "optimal_action" and np.max(data_to_plot) <= 1.0:
            data_to_plot = data_to_plot * 100.0

        ax.plot(steps, data_to_plot, label=label, color=color, linewidth=1.5)

    # Configure axes labels
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', labelpad=8)
    
    if ylabel is None:
        if metric_type == "optimal_action":
            ylabel = "%\nOptimal\naction"
        else:
            ylabel = "Average\nreward"

    # Set Y-axis label horizontally similar to textbook layout
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', rotation=0, labelpad=35, va='center')

    # Format Y-axis ticks for percentage
    if metric_type == "optimal_action":
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    ax.legend(frameon=False, fontsize=11, loc='best')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    return fig, ax


def plot_bandit_dual_metrics(
    rewards_dict: Dict[str, Union[np.ndarray, List[float]]],
    optimal_actions_dict: Dict[str, Union[np.ndarray, List[float]]],
    colors: Optional[Dict[str, str]] = None,
    fig_size: Tuple[int, int] = (12, 5),
    dpi: int = 150
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots both Average Reward and % Optimal Action side-by-side.

    Parameters:
    -----------
    rewards_dict : Dict[str, Union[np.ndarray, List[float]]]
        Dictionary containing reward performance arrays.
    optimal_actions_dict : Dict[str, Union[np.ndarray, List[float]]]
        Dictionary containing optimal action percentage arrays.
    colors : Optional[Dict[str, str]]
        Color dictionary shared between both plots.
    fig_size : Tuple[int, int], default=(12, 5)
        Overall figure size.
    dpi : int, default=150
        Resolution of figure.

    Returns:
    --------
    fig, axes : Matplotlib Figure and Array of Axes.
    """
    fig, axes = plt.subplots(1, 2, figsize=fig_size, dpi=dpi)

    for ax, (metric_type, data_dict) in zip(axes, [("reward", rewards_dict), ("optimal_action", optimal_actions_dict)]):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(direction='in', length=5, width=1.2)

        for label, values in data_dict.items():
            steps = np.arange(1, len(values) + 1)
            color = colors.get(label) if colors else None
            
            data_to_plot = np.array(values)
            if metric_type == "optimal_action" and np.max(data_to_plot) <= 1.0:
                data_to_plot = data_to_plot * 100.0

            ax.plot(steps, data_to_plot, label=label, color=color, linewidth=1.5)

        ax.set_xlabel("Steps", fontsize=11, fontweight='bold')
        
        if metric_type == "optimal_action":
            ax.set_ylabel("% Optimal action", fontsize=11, fontweight='bold')
            ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        else:
            ax.set_ylabel("Average reward", fontsize=11, fontweight='bold')

        ax.legend(frameon=False, fontsize=10)

    plt.tight_layout()
    return fig, axes