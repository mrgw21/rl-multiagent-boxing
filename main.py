from pettingzoo.atari import boxing_v2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import time
from agents.dqn_agent import QNetwork  # Ensure this class matches your training script

# --- Load PPO model (unchanged, TensorFlow) ---
import tensorflow as tf
ppo_model = tf.keras.models.load_model("models/ppo_model.h5")

# --- Load PyTorch DQN model (expects 4 stacked grayscale frames) ---
dqn_model = QNetwork(input_shape=(4, 84, 84), num_actions=18)
dqn_model.load_state_dict(torch.load("models/boxing_dqn.pt", map_location=torch.device('cpu')))
dqn_model.eval()

# --- PPO expects: grayscale (210, 160, 1) ---
def preprocess_for_ppo(obs):
    obs_proc = tf.image.rgb_to_grayscale(obs)
    obs_proc = tf.cast(obs_proc, tf.float32) / 255.0
    obs_proc = tf.expand_dims(obs_proc, axis=0)  # (1, 210, 160, 1)
    return obs_proc

# --- DQN expects: stacked grayscale frames (4, 84, 84) ---
resize_gray = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize((84, 84)),
    T.ToTensor()  # shape: (1, 84, 84)
])

from collections import deque
frame_stack = deque(maxlen=4)

def preprocess_for_dqn(obs):
    frame = resize_gray(obs)  # shape: (1, 84, 84)
    if len(frame_stack) < 4:
        for _ in range(4):
            frame_stack.append(frame)
    else:
        frame_stack.append(frame)
    stacked = torch.cat(list(frame_stack), dim=0)  # shape: (4, 84, 84)
    return stacked.unsqueeze(0)  # shape: (1, 4, 84, 84)

# --- Action functions ---
def ppo_act(obs):
    obs_proc = preprocess_for_ppo(obs)
    probs = ppo_model(obs_proc).numpy()[0]
    return np.random.choice(len(probs), p=probs)

def dqn_act(obs):
    obs_proc = preprocess_for_dqn(obs)
    with torch.no_grad():
        q_values = dqn_model(obs_proc)
    return int(torch.argmax(q_values[0]))

# --- Main loop ---
def main():
    print("PPO Agent (white) vs DQN Agent (black)")

    env = boxing_v2.parallel_env(render_mode="human")
    observations, infos = env.reset()

    total_rewards = {agent: 0 for agent in env.agents}

    while env.agents:
        actions = {}

        for agent, obs in observations.items():
            if agent == "first_0":
                actions[agent] = ppo_act(obs)
            elif agent == "second_0":
                actions[agent] = dqn_act(obs)

        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent in rewards:
            total_rewards[agent] += rewards[agent]

        time.sleep(0.01)

    print("\nFight Over!")
    print(f"PPO Agent (white): {total_rewards['first_0']}")
    print(f"DQN Agent (black): {total_rewards['second_0']}")
    env.close()

if __name__ == "__main__":
    main()