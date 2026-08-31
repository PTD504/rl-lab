import matplotlib.pyplot as plt
import numpy as np

def plot_result(history, title, smooth_window=100):
    """
    title: title of the plot — the name of the property we want to plot in the y-axis
    history: a dictionary with the following structure:
    ```
    history = {
        "sarsa": a 2d array of size (num_runs, max_episodes)
        "q_learning": ...,
        "expected_sarsa": ...
    }
    ```
    """
    def smooth(x, window):
        if window <= 1:
            return x
        kernel = np.ones(window) / window
        return np.array([np.convolve(row, kernel, mode='valid') for row in x])

    plt.figure(figsize=(10, 6))
    for algorithm, data in history.items():
        data = np.asarray(data)
        smoothed = smooth(data, smooth_window)
        mean = np.mean(smoothed, axis=0)
        std = np.std(smoothed, axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, label=algorithm)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel(title)
    plt.legend()
    plt.grid()
    plt.show()


# ===== A-B MDP SPECIFIC HELPER FUNCTIONS =====

def calculate_action_left_percentage(action_left_counts_list, num_steps_list):
    """
    Calculate the percentage of Left action (action 0) chosen from State A in each episode.
    
    Args:
        action_left_counts_list: List of arrays, each containing count of Left actions per episode for each run
        num_steps_list: List of arrays, each containing total steps per episode for each run
    
    Returns:
        List of arrays with Left action percentages per episode for each run
    """
    result = []
    for left_counts, steps in zip(action_left_counts_list, num_steps_list):
        left_counts = np.asarray(left_counts)
        steps = np.asarray(steps)
        # Avoid division by zero
        percentage = np.divide(left_counts, steps, where=steps != 0, out=np.zeros_like(left_counts, dtype=float))
        result.append(percentage * 100)
    return result

# ===== END A-B MDP SPECIFIC HELPER FUNCTIONS =====