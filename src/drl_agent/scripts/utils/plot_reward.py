import numpy as np
import matplotlib.pyplot as plt

# Load the data from the .npy file
file_path = (
    "/home/ahmed/drl_agent_ws/src/drl_agent/temp/results/td7_agent_seed_40_20240815.npy"
)
evals = np.load(file_path)

window_size = 10

# Compute the running mean
evals_running_mean = np.convolve(
    evals, np.ones(window_size) / window_size, mode="valid"
)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(evals, label="Mean Total Reward", color="blue", linewidth=2)
ax.plot(
    np.arange(window_size - 1, len(evals)),
    evals_running_mean,
    label=f"Running Mean Total Reward (window size = {window_size})",
    color="orange",
    linewidth=2,
)

ax.set_title(
    "Evaluation Rewards Over 100 Epochs (10 episodes/epoch)",
    fontsize=16,
    fontweight="bold",
)
ax.set_xlabel("Evaluation Epochs", fontsize=14, fontweight="bold")
ax.set_ylabel("Mean Total Reward", fontsize=14, fontweight="bold")

ax.legend(fontsize=12, prop={"weight": "bold"})
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.tick_params(axis="both", which="major", labelsize=12)

plt.tight_layout()

plt.show()
