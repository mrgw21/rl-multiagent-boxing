import os
import csv
import matplotlib.pyplot as plt

class MetricsLogger:
    def __init__(self, save_path="output", run_name="run"):
        os.makedirs(save_path, exist_ok=True)
        self.path = os.path.join(save_path, f"{run_name}_metrics.csv")
        self.data = []

    def log(self, episode, total_reward, episode_length, loss):
        self.data.append({
            "episode": episode,
            "return": total_reward,
            "length": episode_length,
            "loss": loss
        })

    def save(self):
        keys = ["episode", "return", "length", "loss"]
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.data)

    def plot(self):
        episodes = [d["episode"] for d in self.data]
        returns = [d["return"] for d in self.data]
        losses = [d["loss"] for d in self.data]
        lengths = [d["length"] for d in self.data]

        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(episodes, returns)
        plt.title("Episode Return")

        plt.subplot(3, 1, 2)
        plt.plot(episodes, losses)
        plt.title("Loss")

        plt.subplot(3, 1, 3)
        plt.plot(episodes, lengths)
        plt.title("Episode Length")

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(self.path), f"{os.path.basename(self.path).replace('.csv', '')}_plot.png")
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
