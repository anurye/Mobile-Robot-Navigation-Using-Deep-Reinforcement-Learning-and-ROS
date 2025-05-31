import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from file_manager import load_yaml


def load_metrics(metrics_file_path):
    """Loads metrics from a YAML file."""
    try:
        metrics = load_yaml(metrics_file_path)["test_metrics"]
    except Exception as e:
        raise IOError(f"Unable to load metrics: {metrics_file_path}: {e}")

    return metrics


def plot_metric(ax, metric_name, baseline_metrics, ours_metrics, labels):
    """Plots a single metric comparison in a horizontal bar chart."""
    y = np.arange(len(labels))
    height = 0.35

    ax.barh(y - height / 2, baseline_metrics, height, label="Baseline", color="blue")
    ax.barh(y + height / 2, ours_metrics, height, label="Ours", color="green")

    ax.set_ylabel("Test Environments", fontsize=14, fontweight="bold")
    ax.set_xlabel(metric_name.replace("_", " ").title())
    ax.set_title(
        f'{metric_name.replace("_", " ").title()} Comparison',
        fontsize=16,
        fontweight="bold",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.legend()


def plot_metrics(metrics_file_sets, labels, drl_agent_src_path):
    """Plots separate horizontal bar charts for each metric comparing baseline and ours."""
    metrics_path = os.path.join(drl_agent_src_path, "drl_agent", "test_runs")

    # Initialize data structures for plotting
    metric_names = [
        "average_distance",
        "average_time",
        "collision_rate",
        "success_rate",
    ]
    baseline_metrics = {metric: [] for metric in metric_names}
    ours_metrics = {metric: [] for metric in metric_names}

    for baseline_file, ours_file in metrics_file_sets:
        baseline_metrics_path = os.path.join(metrics_path, f"{baseline_file}.yaml")
        ours_metrics_path = os.path.join(metrics_path, f"{ours_file}.yaml")

        baseline_data = load_metrics(baseline_metrics_path)
        ours_data = load_metrics(ours_metrics_path)

        for metric in metric_names:
            baseline_metrics[metric].append(baseline_data[metric])
            ours_metrics[metric].append(ours_data[metric])

    # Create a figure with subplots for each metric
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()  # Flatten the 2x2 array of axes to a 1D array

    for i, metric in enumerate(metric_names):
        plot_metric(
            axs[i], metric, baseline_metrics[metric], ours_metrics[metric], labels
        )

    plt.tight_layout()
    plt.show()


def main():
    drl_agent_src_env = "DRL_AGENT_SRC_PATH"
    drl_agent_src_path = os.getenv(drl_agent_src_env)
    if drl_agent_src_path is None:
        print(f"Environment variable: {drl_agent_src_env}, is not set")
        sys.exit(-1)

    # List of test cases and corresponding metrics files
    labels = ["td7_empty", "td7_static", "td7_dynamic"]
    metrics_file_sets = [
        ["baseline_env_1_metrics", "ours_env_1_metrics"],
        ["baseline_env_2_metrics", "ours_env_2_metrics"],
        ["baseline_env_3_metrics", "ours_env_3_metrics"],
    ]

    plot_metrics(metrics_file_sets, labels, drl_agent_src_path)


if __name__ == "__main__":
    main()
