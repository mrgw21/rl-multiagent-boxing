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
from sarsa_double_no_norm import Double_SARSA_Agent

import gymnasium as gym
import random
from agent_utils import test_agent, load_agent

agent = load_agent("saved_agents/double_sarsa_best2.pkl")
test_agent(agent, episodes=5, render_mode=None, difficulty=0)
test_agent(agent, episodes=5, render_mode=None, difficulty=1)
test_agent(agent, episodes=5, render_mode=None, difficulty=2)
test_agent(agent, episodes=5, render_mode=None, difficulty=3)
