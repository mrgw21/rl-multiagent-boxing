import numpy as np
import csv
import argparse
from utils.agent_utils import load_linear_agent, test_linear_agent

best = False 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", required=True, help="Name of agent")
    parser.add_argument("--csv_path", type=str, default="training_output.csv", help="Output CSV path")
    parser.add_argument("--bot_difficulty", type=int, default=0, help="Bot difficulty")
    parser.add_argument("--absolute_path", type=str, required=False, help="Absolute path for agent")
    args = parser.parse_args()

    agent_name = args.agent_name

    if args.absolute_path:
        agent = load_linear_agent(args.absolute_path, absolute_path=True)
    else:
        agent = load_linear_agent(agent_name, best=best)

    if args.bot_difficulty in [0, 1, 2, 3]:
        bot_difficulty = args.bot_difficulty
    else:
        raise ValueError("Incorrect difficulty, needs to be in [0,1,2,3]")


    with open(args.csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        print(f"Testing on difficulty {bot_difficulty}")
        rewards = test_linear_agent(agent, episodes=500, render_mode=None, difficulty=bot_difficulty)

        for episode_idx, reward in enumerate(rewards):
            writer.writerow([agent_name, bot_difficulty, episode_idx + 1, reward])

