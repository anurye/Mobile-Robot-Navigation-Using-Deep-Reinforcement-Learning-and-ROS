import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib_scalebar.scalebar import ScaleBar
from file_manager import load_yaml
from dataclasses import dataclass


@dataclass
class MapMetaData:
    resolution: float
    origin: tuple
    map_height: int


@dataclass
class Trajectory:
    x_coords: list
    y_coords: list
    start_x: float
    start_y: float
    goal_x: float
    goal_y: float


def transform_coordinates(coords, origin, resolution, map_height):
    """Transforms map coordinates to image coordinates."""
    if isinstance(coords, tuple):
        x, y = coords
        return (x - origin[0]) / resolution, map_height - (y - origin[1]) / resolution
    if isinstance(coords, list):
        return (
            [(x - origin[0]) / resolution for x, _ in coords],
            [map_height - (y - origin[1]) / resolution for _, y in coords],
        )
    raise TypeError("Input should be a tuple or a list of tuples")


def load_image_and_metadata(map_img_path, map_meta_data_path):
    """Loads the map image and metadata."""
    if not os.path.exists(map_img_path):
        raise FileNotFoundError(f"{map_img_path} does not exist...")

    try:
        map_img = mpimg.imread(map_img_path)
    except Exception as e:
        raise IOError(f"Unable to read: {map_img_path}: {e}")

    try:
        map_metadata = load_yaml(map_meta_data_path)
    except Exception as e:
        raise IOError(f"Unable to load metadata: {map_meta_data_path}: {e}")

    return map_img, MapMetaData(
        resolution=map_metadata["resolution"],
        origin=tuple(map_metadata["origin"]),
        map_height=map_img.shape[0],
    )


def load_trajectories(trajectories_file_path, start_goal_pairs_file_path, map_metadata):
    """Loads and transforms the trajectory data."""
    try:
        with open(trajectories_file_path, "r") as file:
            trajectories = json.load(file)
    except Exception as e:
        raise IOError(f"Unable to read: {trajectories_file_path}: {e}")

    try:
        start_goal_pairs = load_yaml(start_goal_pairs_file_path)["start_goal_pairs"]
    except Exception as e:
        raise IOError(
            f"Unable to load start-goal pairs: {start_goal_pairs_file_path}: {e}"
        )

    all_trajectories = []
    for i, trajectory in enumerate(trajectories):
        coords = [(point["x"], point["y"]) for point in trajectory]
        x_coords, y_coords = transform_coordinates(
            coords,
            map_metadata.origin,
            map_metadata.resolution,
            map_metadata.map_height,
        )

        start, goal = start_goal_pairs[i]["start"], start_goal_pairs[i]["goal"]
        start_x, start_y = transform_coordinates(
            (start["x"], start["y"]),
            map_metadata.origin,
            map_metadata.resolution,
            map_metadata.map_height,
        )
        goal_x, goal_y = transform_coordinates(
            (goal["x"], goal["y"]),
            map_metadata.origin,
            map_metadata.resolution,
            map_metadata.map_height,
        )

        all_trajectories.append(
            Trajectory(x_coords, y_coords, start_x, start_y, goal_x, goal_y)
        )

    return all_trajectories


def load_metrics(metrics_file_path):
    """Loads metrics from a YAML file."""
    try:
        metrics = load_yaml(metrics_file_path)["test_metrics"]
    except Exception as e:
        raise IOError(f"Unable to load metrics: {metrics_file_path}: {e}")

    return metrics


def display_metrics(ax, metrics):
    """Displays metrics in a text box on the plot."""
    textstr = "\n".join(
        (
            f'Avg. Distance: {metrics["average_distance"]:.2f}',
            f'Avg. Time: {metrics["average_time"]:.2f}',
            f'Collision Rate: {metrics["collision_rate"]:.2f}',
            f'Success Rate: {metrics["success_rate"]:.2f}',
        )
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=1.0)
    ax.text(
        0.05,
        0.85,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=props,
        fontweight="bold",
    )


def plot_trajectory(ax, map_img, trajectories):
    """Plots the trajectories on the given axis."""
    ax.imshow(map_img, cmap="gray", origin="upper")

    for i, traj in enumerate(trajectories):
        ax.plot(
            traj.start_x,
            traj.start_y,
            marker="o",
            markersize=8,
            color="blue",
            label="Start" if i == 0 else "",
        )
        ax.plot(
            traj.goal_x,
            traj.goal_y,
            marker="*",
            markersize=12,
            color="red",
            label="Goal" if i == 0 else "",
        )
        ax.plot(traj.x_coords, traj.y_coords, label=f"Traj {i+1}")


def plot_comparison(
    map_file_name,
    trajectory_file_names,
    metrics_file_names,
    start_goal_pairs_file_name,
    drl_agent_src_path,
):
    """Plots trajectory comparisons on two maps and displays metrics."""
    map_path = os.path.join(drl_agent_src_path, "drl_agent", "maps")
    test_runs_path = os.path.join(drl_agent_src_path, "drl_agent", "test_runs")
    start_goal_pairs_path = os.path.join(drl_agent_src_path, "drl_agent", "config")

    map_img_path = os.path.join(map_path, f"{map_file_name}.pgm")
    map_meta_data_path = os.path.join(map_path, f"{map_file_name}.yaml")
    map_img, map_metadata = load_image_and_metadata(map_img_path, map_meta_data_path)

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    for idx, (trajectory_file_name, metrics_file_name) in enumerate(
        zip(trajectory_file_names, metrics_file_names)
    ):
        trajectories_file_path = os.path.join(
            test_runs_path, f"{trajectory_file_name}.json"
        )
        metrics_file_path = os.path.join(test_runs_path, f"{metrics_file_name}.yaml")
        start_goal_pairs_file_path = os.path.join(
            start_goal_pairs_path, f"{start_goal_pairs_file_name}.yaml"
        )

        trajectories = load_trajectories(
            trajectories_file_path, start_goal_pairs_file_path, map_metadata
        )
        metrics = load_metrics(metrics_file_path)

        plot_trajectory(axs[idx], map_img, trajectories)
        display_metrics(axs[idx], metrics)

        axs[idx].set_title(
            f"Trajectories: {trajectory_file_name}", fontsize=16, fontweight="bold"
        )
        axs[idx].set_xlabel("X coordinate (pixels)", fontsize=14, fontweight="bold")
        axs[idx].set_ylabel("Y coordinate (pixels)", fontsize=14, fontweight="bold")

        # Add a scale bar
        scalebar = ScaleBar(
            map_metadata.resolution,
            "m",
            location="lower right",
            pad=0.5,
            color="black",
            frameon=False,
        )
        axs[idx].add_artist(scalebar)

        axs[idx].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def main():
    drl_agent_src_env = "DRL_AGENT_SRC_PATH"
    drl_agent_src_path = os.getenv(drl_agent_src_env)
    if drl_agent_src_path is None:
        print(f"Environment variable: {drl_agent_src_env}, is not set")
        sys.exit(-1)

    map_file_names = ["td7_empty", "td7_static"]
    trajectory_file_sets = [
        ["baseline_env_1_traj", "ours_env_1_traj"],
        ["baseline_env_2_traj", "ours_env_2_traj"],
    ]
    metrics_file_sets = [
        ["baseline_env_1_metrics", "ours_env_1_metrics"],
        ["baseline_env_2_metrics", "ours_env_2_metrics"],
    ]
    start_goal_pairs_file_name = "test_config"

    for map_file_name, trajectory_files, metrics_files in zip(
        map_file_names, trajectory_file_sets, metrics_file_sets
    ):
        plot_comparison(
            map_file_name,
            trajectory_files,
            metrics_files,
            start_goal_pairs_file_name,
            drl_agent_src_path,
        )


if __name__ == "__main__":
    main()
