import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

def visualize_predicted_tl_vs_ground_truth(
    trajectories_pred, trajectories_true, trajectory_name,
    unique_values_on_each_dimension, custom_lon_labels, custom_lat_labels, deviation_amount
):
    def extract_grid_cells(state):
        return [(int(x), int(y)) for x, y in zip(state[0], state[1])]

    lon_values = unique_values_on_each_dimension[0]
    lat_values = unique_values_on_each_dimension[1]

    fig, ax = plt.subplots(figsize=(12, 9))
    all_grids = set()

    # Plot ground truth trajectories
    for i, trajectory in enumerate(trajectories_true):
        trajectory_grids = [extract_grid_cells(state)[0] for state in trajectory]
        lons = [grid[0] for grid in trajectory_grids]
        lats = [grid[1] for grid in trajectory_grids]
        all_grids.update(trajectory_grids)

        color = 'gray'
        ax.plot(lons, lats, color=color, linewidth=1.5, alpha=0.8,
                label='Ground Truth' if i == 0 else None)
        ax.scatter(lons, lats, color=color, s=10, alpha=0.6)

    # Plot predicted trajectories
    if trajectory_name == "HMM-RL":
        pred_color = "#1f6cb3"  # Blue
    elif trajectory_name == "Baseline":
        pred_color = "#d94701"  # Orange
    else:
        pred_color = "#007849"  # Fallback: Green

    for i, trajectory in enumerate(trajectories_pred):
        trajectory_grids = [extract_grid_cells(state)[0] for state in trajectory]
        lons = [grid[0] for grid in trajectory_grids]
        lats = [grid[1] for grid in trajectory_grids]
        all_grids.update(trajectory_grids)

        ax.plot(lons, lats, color=pred_color, linewidth=1.5, alpha=0.8,
                label=trajectory_name if i == 0 else None)
        ax.scatter(lons, lats, color=pred_color, s=10, alpha=0.6)

    # Axis label formatting
    custom_lon_labels = [f'{label:.4f}' for label in custom_lon_labels]
    custom_lat_labels = [f'{label:.4f}' for label in custom_lat_labels]

    ax.set_xticks(np.arange(0, len(lon_values) + 1, 3) - 0.5)
    ax.set_xticklabels(custom_lon_labels[::3], rotation=45)
    ax.set_yticks(np.arange(0, len(lat_values) + 1, 3) - 0.5)
    ax.set_yticklabels(custom_lat_labels[::3])

    # Axis limits
    min_lon = max(0, min(lon for lon, lat in all_grids) - 5)
    max_lon = min(len(lon_values), max(lon for lon, lat in all_grids) + 5)
    min_lat = max(0, min(lat for lon, lat in all_grids) - 5)
    max_lat = min(len(lat_values), max(lat for lon, lat in all_grids) + 5)

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    # Labels and legend
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='lower left', frameon=True, fancybox=True, framealpha=0.9)

    # Titles
    fig.suptitle(f'{trajectory_name} Trajectories vs. Ground Truth', fontsize=22, y=0.95)
    ax.set_title(f'Deviation = {deviation_amount}', fontsize=18)

    # Save with high DPI
    folder_path = 'code/trajectory/graph_output'
    os.makedirs(folder_path, exist_ok=True)
    file_name = f'Geographic Visualization of {trajectory_name} Trajectories vs. Ground Truth (Deviation {deviation_amount}).png'
    full_path = os.path.join(folder_path, file_name)
    plt.tight_layout()
    fig.savefig(full_path, dpi=300, bbox_inches='tight')

    print(f"Geographic Visualization of {trajectory_name} saved to {full_path}.")
    plt.close()

def visualize_single_trajectory(
    trajectories_dict,  # Dictionary: {"Ground Truth": [...], "HMM-RL": [...], "Baseline": [...]}
    unique_values_on_each_dimension,
    custom_lon_labels,
    custom_lat_labels,
    deviation_amount,
    trajectory_num
):
    def extract_grid_cells(state):
        return [(int(x), int(y)) for x, y in zip(state[0], state[1])]

    lon_values = unique_values_on_each_dimension[0]
    lat_values = unique_values_on_each_dimension[1]

    fig, ax = plt.subplots(figsize=(12, 9))
    all_grids = set()

    color_map = {
        "Ground Truth": "gray",
        "HMM-RL": "#1f6cb3",
        "Baseline": "#d94701"
    }

    for traj_name, trajectory in trajectories_dict.items():
        color = color_map.get(traj_name, "#007849")  # Default green
        trajectory_grids = [extract_grid_cells(state)[0] for state in trajectory]
        lons = [grid[0] for grid in trajectory_grids]
        lats = [grid[1] for grid in trajectory_grids]
        all_grids.update(trajectory_grids)

        if traj_name == "Ground Truth":
            ax.plot(lons, lats, color=color, linewidth=3.5, alpha=0.8, marker='o',
                    label=traj_name)
        elif traj_name == "HMM-RL":
            ax.plot(lons, lats, color=color, linewidth=1.5, linestyle = "--", marker='x',
                    label=traj_name)
        elif traj_name == "Baseline":
            ax.plot(lons, lats, color=color, linewidth=1.5, linestyle = "--", marker='x',
                    label=traj_name)
        ax.scatter(lons, lats, color=color, s=10, alpha=0.6)

    # Format axis ticks
    custom_lon_labels = [f'{label:.4f}' for label in custom_lon_labels]
    custom_lat_labels = [f'{label:.4f}' for label in custom_lat_labels]

    ax.set_xticks(np.arange(0, len(lon_values) + 1, 3) - 0.5)
    ax.set_xticklabels(custom_lon_labels[::3], rotation=45)
    ax.set_yticks(np.arange(0, len(lat_values) + 1, 3) - 0.5)
    ax.set_yticklabels(custom_lat_labels[::3])

    # Axis limits
    min_lon = max(0, min(lon for lon, lat in all_grids) - 5)
    max_lon = min(len(lon_values), max(lon for lon, lat in all_grids) + 5)
    min_lat = max(0, min(lat for lon, lat in all_grids) - 5)
    max_lat = min(len(lat_values), max(lat for lon, lat in all_grids) + 5)

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='lower left', frameon=True, fancybox=True, framealpha=0.9)

    fig.suptitle('Trajectories: Ground Truth vs. Predictions', fontsize=22, y=0.95)
    ax.set_title(f'Deviation = {deviation_amount}', fontsize=18)

    folder_path = 'code/trajectory/graph_output'
    os.makedirs(folder_path, exist_ok=True)
    file_name = f'Trajectory {str(trajectory_num)}.png'
    full_path = os.path.join(folder_path, file_name)
    plt.tight_layout()
    fig.savefig(full_path, dpi=300, bbox_inches='tight')

    print(f"Trajectory {str(trajectory_num)} Visualization saved to {full_path}.")
    plt.close()
