import numpy as np
import gymnasium as gym
import random
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import os
from pettingzoo.atari import boxing_v2
import ale_py
# if using gymnasium
import shimmy
import time
import gymnasium as gym
import random
from utils.agent_utils import test_linear_agent, load_linear_agent
from double_sarsa_agents import (
    Base_Double_Sarsa_Agent,
    TrueOnlineSarsa,
    SimpleDoubleSarsa,
    DoubleSarsaNoExperience,
    DoubleSarsaRandomExperience,
    DoubleSarsaPrioritisedExperience,
    DoubleSarsaPriortisedExperienceWithCache
)
import pickle
import os
import csv
import numpy as np
from utils.agent_utils import load_linear_agent, test_linear_agent


agent_dir = "saved_agents/best_agents/"
best = True  


output_csv = "best_agent_rewards_output.csv"


agent_files = [
    f for f in os.listdir(agent_dir)
    # if "_30_" and 'Experience' in f and not (f.endswith("00.pkl") or f.endswith('way.pkl'))
]

print("Found agents:", agent_files)

difficulties = [1, 2, 3, 0]

with open(output_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    writer.writerow(["agent_name", "difficulty", "episode", "reward"])
    
    for agent_file in agent_files:
        agent_name = agent_file.replace(".pkl", "")
        print(f"\nTesting agent: {agent_name}")
        
        agent = load_linear_agent(agent_name, best=best)

        try:
            for difficulty in difficulties:
                print(f"  Testing difficulty {difficulty}")
                rewards = test_linear_agent(agent, episodes=500, render_mode=None, difficulty=difficulty)
                
                for episode_idx, reward in enumerate(rewards):
                    writer.writerow([agent_name, difficulty, episode_idx + 1, reward])
        except:
            pass

print(f"\nResults saved to {output_csv}")


