import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV file
csv_file = "agent_rewards_output.csv"  # Replace with your actual file path
df = pd.read_csv(csv_file)

# Group by agent_name and difficulty
grouped = df.groupby(["agent_name", "difficulty"])["reward"].agg(["mean", "std", "min", "max"]).reset_index()

# Save to text file
stats_file = "test_agent_stats.txt"
with open(stats_file, "w") as f:
    for _, row in grouped.iterrows():
        f.write(
            f"Agent: {row['agent_name']} | Difficulty: {row['difficulty']} | "
            f"Mean: {row['mean']:.2f} | SD: {row['std']:.2f} | "
            f"Min: {row['min']} | Max: {row['max']}\n"
        )

# Plot mean ± SD
plt.figure(figsize=(12, 6))
labels = [f"{row['agent_name']}\n(Diff {row['difficulty']})" for _, row in grouped.iterrows()]
plt.bar(labels, grouped["mean"], yerr=grouped["std"], capsize=5)
plt.ylabel("Mean Reward")
plt.title("Mean ± SD of Reward by Agent and Difficulty")
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("agent_performance_plot.png")
plt.show()
