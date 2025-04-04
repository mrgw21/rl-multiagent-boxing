from pettingzoo.atari import boxing_v2
import numpy as np
import tensorflow as tf
import time
from agents.dqn_agent import DQNAgent  # Ensure this path is correct based on your repo structure

# Load PPO model
ppo_model = tf.keras.models.load_model("models/ppo_model.h5")

# Load DQN model
dqn_model = tf.keras.models.load_model("models/50_dqn_model.keras")

def preprocess(obs):
    obs = tf.image.resize(obs, [84, 84])
    obs = tf.image.convert_image_dtype(obs, tf.uint8)
    return obs.numpy()

def ppo_act(obs):
    obs_proc = tf.image.rgb_to_grayscale(obs)
    obs_proc = np.expand_dims(obs_proc, axis=0).astype(np.float32) / 255.0
    probs = ppo_model(obs_proc).numpy()[0]
    return np.random.choice(len(probs), p=probs)

def dqn_act(obs):
    obs_proc = tf.image.resize(obs, [84, 84])
    obs_proc = obs_proc.numpy().astype(np.float32) / 255.0
    obs_proc = np.expand_dims(obs_proc, axis=0)
    q_values = dqn_model(obs_proc).numpy()[0]
    return int(np.argmax(q_values))

def main():
    print("PPO Agent (white) vs DQN Agent (black)")

    env = boxing_v2.env(render_mode="human")
    env.reset()

    total_rewards = {"first_0": 0, "second_0": 0}

    for agent in env.agent_iter():
        obs, reward, termination, truncation, _ = env.last()
        total_rewards[agent] += reward

        if termination or truncation:
            action = None
        else:
            if agent == "first_0":
                action = ppo_act(obs)
            else:
                action = dqn_act(obs)

        env.step(action)
        time.sleep(0.01)  # Visual pacing

    print("\nFight Over!")
    print(f"PPO Agent (white): {total_rewards['first_0']}")
    print(f"DQN Agent (black): {total_rewards['second_0']}")
    env.close()

if __name__ == "__main__":
    main()
