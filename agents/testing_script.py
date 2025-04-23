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
from utils.agent_utils import test_agent, load_agent
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


# When you load an agent, if you get an attribute error it's probably because its class no longer exists (After I reorganised the code).
# To fix this - I've left the original class in sarsa_double for now (can delete later) - import using:

# from sarsa_double import Double_SARSA_Agent 

# This code lets you load the agent, change its class to one in double_sarsa_agents and then write it to a new pickle file

# agent = load_agent(agent_using_old_Double_SARSA_Agent_class")
# loaded_agent.__class__ = DoubleSarsaPriortisedExperienceWithCache

# # Save to a new pickle
# with open("saved_agents/agent_using_correct_class.pkl", "wb") as f:
#     pickle.dump(loaded_agent, f)

# Only use best = True if it's in the best_agents folder 
agent = load_agent("PER_with_cache_agent_22_04_rank1", best = True)
test_agent(agent, episodes=5, render_mode=None, difficulty=0)
test_agent(agent, episodes=5, render_mode=None, difficulty=1)
test_agent(agent, episodes=5, render_mode=None, difficulty=2)
test_agent(agent, episodes=5, render_mode=None, difficulty=3)



