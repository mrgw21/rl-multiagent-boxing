from pettingzoo.atari import boxing_v2
import numpy as np
from agents.rando_agent1 import RandoAgent1
from agents.rando_agent2 import RandoAgent2
import time

def main():
    print("RandoAgent1 (white) vs RandoAgent2 (black)")

    env = boxing_v2.env(render_mode="human")
    env.reset()

    agent1 = RandoAgent1(env.action_space("first_0"))
    agent2 = RandoAgent2(env.action_space("second_0"))

    total_rewards = {"first_0": 0, "second_0": 0}
    done = False

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        total_rewards[agent] += reward

        if termination or truncation:
            action = None  # Required by PettingZoo when agent is done
        else:
            if agent == "first_0":
                action = agent1.act(obs)
            else:
                action = agent2.act(obs)

        env.step(action)
        time.sleep(0.01)

    print(f"\n✅ Fight Over!")
    print(f"🏳️ White (RandoAgent1): {total_rewards['first_0']}")
    print(f"🖤 Black (RandoAgent2): {total_rewards['second_0']}")
    env.close()

if __name__ == "__main__":
    main()
