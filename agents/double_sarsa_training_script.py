import numpy as np
import argparse
from datetime import datetime
from utils.agent_utils import load_linear_agent, test_linear_agent, save_linear_agent, plot_rewards
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

    # DoubleSarsaNoExperience("Double Sarsa With No Experience Replay"),
    # DoubleSarsaRandomExperience("Double Sarsa With Random Experience Replay"),
    # DoubleSarsaPrioritisedExperience("Double Sarsa With Prioritised Experience Replay"),
    # DoubleSarsaPriortisedExperienceWithCache("Double Sarsa With Prioritised Experience Replay + Cache"),

agent_flags = {
    "no exp" : DoubleSarsaNoExperience,
    "rand exp" : DoubleSarsaRandomExperience,
    "per" : DoubleSarsaPrioritisedExperience,
    "per cache" : DoubleSarsaPriortisedExperienceWithCache,
}

# Only run training if we are running this script - needed when importing module from another script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=agent_flags.keys(), required=True, help="Which double sarsa agent to train")
    parser.add_argument("--episodes", type=int, default=5000, help="How many episodes to train for")
    parser.add_argument("--bot_difficulty", type=int, default=0, help="Bot Difficulty: 0 (default), 1 (hardest), 2 (intermediate), 3 (easiest).")
    parser.add_argument("--feature_type", choices=["full_ram", "semi_reduced_ram", "reduced_ram"], default="semi_reduced_ram", help="Which feature type to use for training.")
    parser.add_argument("--agent_name", type=str, default=f"Double Sarsa Agent {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", help="Give your agent a name")
    args = parser.parse_args()

    agent_type = agent_flags[args.agent]
    agent = agent_type(args.agent_name, render=None, feature_type=args.feature_type)
    rewards = agent.train(num_episodes=args.episodes, bot_difficulty=args.bot_difficulty)
    print(f"Training {args.agent_name} for {args.episodes} episodes.")

    save_linear_agent(agent, args.agent_name)
    print(f"Training complete: {args.agent_name} saved to /agents/saved_agents/testing_agents/{args.agent_name}.pkl")
