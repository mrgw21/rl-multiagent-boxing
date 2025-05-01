import pickle
import csv
from utils.agent_utils import load_linear_agent

agent = load_linear_agent(name="best_agent_semi_prioritised_cache_23_04", best=True)

# reward_history = agent.rewards_history

# csv_file_path = 'agent_reward_history.csv'

# with open(csv_file_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     for reward in reward_history:
#         writer.writerow([reward])

# print(f"Reward history has been saved to {csv_file_path}")

print(agent.__dict__)