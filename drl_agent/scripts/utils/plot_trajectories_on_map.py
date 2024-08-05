import os
import sys
import json
import yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib_scalebar.scalebar import ScaleBar


def transform_coordinates(coords, origin, resolution, map_height):
    """Transforms map coordinates to image coordinates."""
    if isinstance(coords, tuple):
        x, y = coords
        return (x - origin[0]) / resolution, map_height - (y - origin[1]) / resolution
    if isinstance(coords, list):
        return (
            [(x - origin[0]) / resolution for x, _ in coords],
            [map_height - (y - origin[1]) / resolution for _, y in coords]
        )
    raise TypeError("Input should be a tuple or a list of tuples")


def load_yaml(yaml_file_path):
    """Load a YAML file."""
    try:
        with open(yaml_file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Unable to load: {yaml_file_path}: {e}")
        sys.exit(-1)


def plot_traj_on_map(map_img_path, map_meta_data_path, trajectories_file_path, start_goal_pairs_file_path):
    """Plots trajectories on a given map"""
    for path in [map_img_path, map_meta_data_path, trajectories_file_path, start_goal_pairs_file_path]:
        if not os.path.exists(path):
            print(f"{path} does not exist...")
            sys.exit(-1)

    try:
        map_img = mpimg.imread(map_img_path)
    except Exception as e:
        print(f"Unable to read: {map_img_path}: {e}")
        sys.exit(-1)

    try:
        with open(trajectories_file_path, "r") as file:
            trajectories = json.load(file)
    except Exception as e:
        print(f"Unable to read: {trajectories_file_path}: {e}")
        sys.exit(-1)

    map_metadata = load_yaml(map_meta_data_path)
    start_goal_pairs = load_yaml(start_goal_pairs_file_path)["start_goal_pairs"]
    resolution, origin = map_metadata["resolution"], map_metadata["origin"]
    map_height = map_img.shape[0]

    plt.imshow(map_img, cmap="gray", origin="upper")

    for i, trajectory in enumerate(trajectories):
        coords = [(point["x"], point["y"]) for point in trajectory]
        x_coords, y_coords = transform_coordinates(coords, origin, resolution, map_height)

        start, goal = start_goal_pairs[i]["start"], start_goal_pairs[i]["goal"]
        start_x, start_y = transform_coordinates((start["x"], start["y"]), origin, resolution, map_height)
        goal_x, goal_y = transform_coordinates((goal["x"], goal["y"]), origin, resolution, map_height)

        plt.plot(start_x, start_y, marker="o", markersize=8, color="blue", label="Start" if i == 0 else "")
        plt.plot(goal_x, goal_y, marker="*", markersize=12, color="red", label="Goal" if i == 0 else "")
        plt.plot(x_coords, y_coords, label=f"Trajectory {i+1}")

    plt.title("Trajectories Overlaid on Map")
    plt.xlabel("X coordinate (pixels)")
    plt.ylabel("Y coordinate (pixels)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Add a scale bar
    scalebar = ScaleBar(resolution, 'm', location='lower right',
                        pad=0.5, color='black', frameon=False)
    plt.gca().add_artist(scalebar)
    plt.show()


def main():
    drl_agent_src_env = "DRL_AGENT_SRC_PATH"
    drl_agent_src_path = os.getenv(drl_agent_src_env)
    if drl_agent_src_path is None:
        print(f"Environment variable: {drl_agent_src_env}, is not set")
        sys.exit(-1)

    map_file_name = "empty_world"
    trajectory_file_name = "traj_for_seed_0"
    start_goal_pairs_file_name = "test_config"

    map_path = os.path.join(drl_agent_src_path, "drl_agent", "maps")
    trajectories_path = os.path.join(drl_agent_src_path, "drl_agent", "trajectories")
    start_goal_pairs_path = os.path.join(drl_agent_src_path, "drl_agent", "config")

    map_img_path = os.path.join(map_path, f"{map_file_name}.pgm")
    map_meta_data_path = os.path.join(map_path, f"{map_file_name}.yaml")
    trajectories_file_path = os.path.join(trajectories_path, f"{trajectory_file_name}.json")
    start_goal_pairs_file_path = os.path.join(start_goal_pairs_path, f"{start_goal_pairs_file_name}.yaml")

    plot_traj_on_map(map_img_path, map_meta_data_path, trajectories_file_path, start_goal_pairs_file_path)


if __name__ == '__main__':
    main()
