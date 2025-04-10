from pettingzoo.atari import boxing_v2
import numpy as np
from agents.rando_agent1 import RandoAgent1
from agents.rando_agent2 import RandoAgent2
import time

def main():
    print("RandoAgent1 (white) vs RandoAgent2 (black)")

    # Use parallel_env instead of env
    env = boxing_v2.parallel_env(render_mode="human")
    observations, infos = env.reset()

    # Initialize agents
    agent1 = RandoAgent1(env.action_space("first_0"))
    agent2 = RandoAgent2(env.action_space("second_0"))

    total_rewards = {agent: 0 for agent in env.agents}

    while env.agents:  # While there are still active agents
        actions = {}

        for agent, obs in observations.items():
            if agent == "first_0":
                actions[agent] = agent1.act(obs)
            elif agent == "second_0":
                actions[agent] = agent2.act(obs)

        # Step the environment with the actions
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Track rewards
        for agent in rewards:
            total_rewards[agent] += rewards[agent]

        time.sleep(0.01)

    print(f"\nFight Over!")
    print(f"White (RandoAgent1): {total_rewards['first_0']}")
    print(f"Black (RandoAgent2): {total_rewards['second_0']}")
    env.close()

if __name__ == "__main__":
    main()
