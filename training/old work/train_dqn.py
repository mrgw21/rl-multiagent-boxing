import os
import numpy as np
import cv2
from pettingzoo.atari import boxing_v2
from agents.dqn_agent import DQNAgent
from training.metrics import MetricsLogger

MAX_STEPS = 1500  # To limit excessively long episodes
EPISODES = 50
SAVE_EVERY = 20

def preprocess(obs):
    obs_resized = cv2.resize(obs, (84, 84))
    return obs_resized.astype(np.uint8)

def train_dqn_agent():
    print("Initializing Boxing-v2 environment...")
    env = boxing_v2.env(render_mode=None)
    agent_id = "first_0"
    opponent_id = "second_0"
    env.reset()

    obs = preprocess(env.observe(agent_id))
    agent = DQNAgent(env.action_space(agent_id))
    logger = MetricsLogger(save_path="metrics", run_name="dqn")

    print(f"Agent initialized. Starting training for {EPISODES} episodes.")

    for episode in range(EPISODES):
        print(f"\n--- Starting Episode {episode + 1} ---")
        env.reset()
        obs = preprocess(env.observe(agent_id))
        total_reward = 0
        step_count = 0
        loss = 0.0

        for agent_turn in env.agent_iter():
            obs_raw, reward, termination, truncation, info = env.last()
            done = termination or truncation or step_count >= MAX_STEPS

            if done:
                env.step(None)
            elif agent_turn == agent_id:
                action = agent.act(obs)
                env.step(action)
            else:
                action = env.action_space(agent_turn).sample()
                env.step(action)

            if agent_turn == agent_id:
                next_obs = preprocess(env.observe(agent_id))
                agent.remember(obs, action, reward, next_obs, done)
                loss = agent.replay(batch_size=64) or 0.0
                obs = next_obs
                total_reward += reward
                step_count += 1

                if step_count % 100 == 0:
                    print(f"  Step {step_count}, Reward: {total_reward}, Memory: {len(agent.memory)}")

                if step_count >= MAX_STEPS:
                    print(f"Max steps reached for Episode {episode + 1}.")
                    break

        agent.update_target_model()
        logger.log(episode, total_reward, step_count, loss)
        print(f"Episode {episode + 1} finished: Reward={total_reward}, Steps={step_count}, Loss={loss:.4f}")

        if (episode + 1) % SAVE_EVERY == 0:
            agent.model.save("models/dqn_model.keras")
            print("Model checkpoint saved.")

    logger.save()
    logger.plot()
    agent.model.save("models/50_dqn_model.keras")
    print("Final model saved to models/50_dqn_model.keras")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    train_dqn_agent()
