import os
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})


log_dir = '/Users/filippo/Library/CloudStorage/OneDrive-UniversityofBath/MSc Computer Science/Semester 2/Semester 2 Vault/Reinforcement Learning/CW2/rl-multiagent-boxing/agents/logs'
selected_files = [
    # 'log_Double_Sarsa_No_Experience_30_no_norm_1746084017.txt',
    'log_Double_Sarsa_Random_Experience_30_1746056265.txt',
    'log_Double_Sarsa_Prioritised_Experience_30_1746037304.txt',
    'log_Double_Sarsa_Prioritised_Experience_With_Cache_30_1746028893.txt',

]

window_size = 100
max_episodes = 15000

def moving_average(data, window_size=100):
    return [sum(data[max(0, i - window_size + 1):i + 1]) / (i - max(0, i - window_size + 1) + 1) for i in range(len(data))]

colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']

plt.figure(figsize=(12, 6))

names = ["Random Experience", "Prioritised Experience", "Prioritised Experience with Cache"]

for idx, filename in enumerate(selected_files):
    file_path = os.path.join(log_dir, filename)
    agent_name = os.path.splitext(filename)[0].replace("log_", "")
    agent_name = names[idx]
    episodes, rewards = [], []

    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("Episode"):
                try:
                    episode, reward, *_ = eval(line.strip())
                    if episode < max_episodes:
                        episodes.append(episode)
                        rewards.append(reward)
                except Exception as e:
                    print(f"Skipping malformed line in {filename}: {line.strip()} ({e})")

    if episodes:
        avg_rewards = moving_average(rewards, window_size)
        plt.plot(episodes, avg_rewards, label=agent_name, linewidth=1, color=colors[idx % len(colors)])


plt.axhline(y=100, color='gray', linestyle='--', linewidth=1.5)
plt.text(5000 * 0.98, 100.5, 'Max Reward = 100', color='gray', fontsize=12,
         verticalalignment='bottom', horizontalalignment='right')

plt.xlabel("Episode")
plt.xlim(0, 5000)
plt.ylabel("Average Reward")
plt.title("Learning Curves: Moving Average of Episode Rewards")

plt.legend(loc='center left')
plt.tight_layout()

plt.grid(True, linestyle='--', linewidth=0.5)
plt.savefig("comparison_learning_curve_selected.png", dpi=600, bbox_inches='tight')
plt.show()
