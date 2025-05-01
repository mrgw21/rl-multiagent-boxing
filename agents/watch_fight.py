import numpy as np
import csv
import argparse
from utils.agent_utils import load_linear_agent, test_linear_agent
import ale_py
import gymnasium as gym

best = False 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_path", default="saved_agents/best_agents/best_agent_semi_prioritised_cache_23_04.pkl" ,help="Path for agent")
    parser.add_argument("--bot_difficulty", type=int, default=0, help="Bot difficulty")
    args = parser.parse_args()

    agent = load_linear_agent(name=args.agent_path, absolute_path=True)
    agent.epsilon = 0 # Always make greedy choice

    if args.bot_difficulty in [0, 1, 2, 3]:
        bot_difficulty = args.bot_difficulty
    else:
        raise ValueError("Incorrect difficulty, needs to be in [0,1,2,3]")
    

    env = gym.make("ALE/Boxing-v5", obs_type="ram", render_mode="human")
    
    obs, info = env.reset()

    done = False

    while not done:
        obs = agent.feature_extraction(obs)
        action = agent.policy(obs)
        obs, reward, done, _, info = env.step(action)

        env.render()

    env.close()