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
import shimmy
import time
import gymnasium as gym
import random
from utils.agent_utils import load_agent, test_agent, save_agent, plot_rewards
from utils.lsh_script import LSH
from utils.PER import store_experience, prioritised_sample, update_priority_order
from double_sarsa_agents import (
    Base_Double_Sarsa_Agent,
    TrueOnlineSarsa,
    SimpleDoubleSarsa,
    DoubleSarsaNoExperience,
    DoubleSarsaRandomExperience,
    DoubleSarsaPrioritisedExperience,
    DoubleSarsaPriortisedExperienceWithCache
)


# Agent examples:

    # Base_Double_Sarsa_Agent("Base Agent"),
    # TrueOnlineSarsa("True Online Agent"),
    # SimpleDoubleSarsa("Simple Double Sarsa"),
    # DoubleSarsaNoExperience("Double Sarsa With No Experience Replay"),
    # DoubleSarsaRandomExperience("Double Sarsa With Random Experience Replay"),
    # DoubleSarsaPrioritisedExperience("Double Sarsa With Prioritised Experience Replay"),
    # DoubleSarsaPriortisedExperienceWithCache("Double Sarsa With Prioritised Experience Replay + Cache"),


# Example of how to run training
# Only run training if we are running this script - needed when importing module from another script
if __name__ == "__main__":
    agent = DoubleSarsaPriortisedExperienceWithCache("test_delete_me", render=None, feature_type="reduced_ram") 
    print(agent.name)
    rewards = agent.train(num_episodes=100) 

    save_agent(agent, agent.name)
    loaded_agent = load_agent(agent.name)
    test_agent(loaded_agent, render_mode=None)

    plot_rewards(rewards, "Final Agent Rewards", save_path=f"Learning Curves/{agent.name} Final Curve")


