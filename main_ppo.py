import gym
import torch
import numpy as np
import os
import time
from collections import deque
from gym.wrappers import FrameStack
from training.gpu.ppo_gpu import PPOAgent
from training.gpu.neural_ne_gpu import Actor, device
import cv2


def preprocess_frame(frame):
    """Converts RGB frame to grayscale, resizes to 84x84"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0  # Normalize to [0, 1]


def stack_frames(stacked_frames, new_frame, is_new_episode, stack_size=4):
    """Stacks 4 frames together"""
    frame = preprocess_frame(new_frame)

    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.float32)] * stack_size, maxlen=stack_size)
        for _ in range(stack_size):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)

    stacked_state = np.stack(stacked_frames, axis=0)  # Shape: (4, 84, 84)
    return stacked_state, stacked_frames


def main():
    # Load environment
    env = gym.make("ALE/Boxing-v5", render_mode="human")
    n_actions = env.action_space.n

    # Load trained PPO actor
    model_path = os.path.join("models", "highest_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO model not found at {model_path}")
    print(f"Loading PPO model from: {model_path}")

    actor = Actor(n_actions=n_actions).to(device)
    state_dict = torch.load(model_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()

    # Initialize agent
    agent = PPOAgent(actor=None, critic=None)
    agent.actor = actor

    # Reset environment
    obs, _ = env.reset()
    stacked_frames = deque([np.zeros((84, 84), dtype=np.float32)] * 4, maxlen=4)
    state, stacked_frames = stack_frames(stacked_frames, obs, is_new_episode=True)

    done = False
    total_reward = 0

    while not done:
        env.render()

        # Add batch dimension → (1, 4, 84, 84)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action, _ = agent.get_action(state_t)
        action = int(action)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        state, stacked_frames = stack_frames(stacked_frames, next_obs, is_new_episode=False)

        time.sleep(0.01)

    print(f"Total Reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
