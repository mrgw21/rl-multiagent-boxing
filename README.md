# RL Multi-agent Boxing Project

A reinforcement learning project built for the second CM50270 coursework at the University of Bath.

This project trains agents (like PPO, DQN, and others) to play Atari Boxing using the Gym ALE environment.

---

## Getting Started

> **Windows users**:  
> The `multi-agent-ale-py` package **does not compile on native Windows**.  
> You must use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) to run this project.  
> Once WSL is installed, follow the WSL/Linux instructions below.

---

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-TEAM-NAME/rl-boxing.git
cd rl-boxing
```

---
### 2. Setting up Environment

#### Linear Agents

##### If using conda

```bash
conda env create -f double_sarsa_conda_env.yml
```

##### If not using conda

```bash
python3 -m venv venv

# For Mac/Linux/WSL:
source venv/bin/activate
```
###### If you're rendering the boxing gym locally:

```bash
pip install -r double_sarsa_requirements.txt
```

###### If you're rendering in WSL (with X server like VcXsrv):

```bash
sudo apt update
sudo apt install cmake swig zlib1g-dev libboost-all-dev \
                 libsdl2-dev libsdl2-image-dev \
                 python3-dev build-essential

pip install -r double_sarsa_requirements.txt

# Then add this to ~/.bashrc or ~/.zshrc:
export DISPLAY=:0.0
```


---

### 3. Download Atari ROMs

```bash
AutoROM --accept-license
```

---

### 4. Train An Agent

#### Linear Agents

To train a double sarsa agent run the following script with the following arguments

```bash
cd agents # Have to be in agents dir to run this
python double_sarsa_training_script.py --agent <agent_type> --episodes <number_of_episodes> --bot_difficulty <difficulty_level> --feature_type <feature_type> --agent_name <agent_name>
```
##### Arguments
- --agent : Choose type of agent (options: `"no exp"`, `"rand exp"`, `"per"`, `"per cache"`)
- --episodes : How many episodes to train for - default is `5000`
- --bot_difficulty : Choose bot difficulty (options: 0 [default], 1 [hardest], 2 [intermediate], 3 [easiest])
- --feature_type : Choose type of feature extraction - default is semi-reduced ram (options: `semi_reduced_ram`, `full_ram` or `reduced_ram`)
- --agent_name : Give your agent a name - default is `Double Sarsa Agent {Time of Training}`

---

### 5. Watch Trained Agent Fight

#### Linear Agents

```bash
# Have to be in agents dir to run this
python watch_fight.py --agent_path <path_to_trained_agent> --bot_difficulty <difficulty_level>
```
##### Arguments
- --agent_path : Path of agent to load - default is `"saved_agents/best_agents/best_agent_semi_prioritised_cache_23_04.pkl"`
- --bot_difficulty : Choose bot difficulty (options: 0 [default], 1 [hardest], 2 [intermediate], 3 [easiest])

---

### 5. Run Performance Testing Across 500 Episodes

#### Linear Agents

```bash
# Have to be in agents dir to run this
python linear_testing.py --agent_name <agent_name> --csv_path <output_csv_path> --bot_difficulty <difficulty_level> --absolute_path <path_to_trained_agent>
```
##### Arguments
- --agent_name : If agent is in `testing_agents/` then name will suffice
- --csv_path: Path for output CSV file where results will be saved. Default is `training_output.csv`.
- --bot_difficulty : Choose bot difficulty (options: 0 [default], 1 [hardest], 2 [intermediate], 3 [easiest])
- --absolute_path : Optional absolute path if agent is not in `saved_agents/` 

---

## Agents

### Linear agents can be found in `agents/double_sarsa_agents`
- `DoubleSarsaNoExperience`: Double SARSA implementation with no experience replay.
- `DoubleSarsaRandomExperience`: Double SARSA implementation with random experience replay.
- `DoubleSarsaPriortisedExperience`: Double SARSA implementation with prioritised experience replay.
- `DoubleSarsaPriortisedExperienceWithCache`: Double SARSA implementation with prioritised experience replay and cache.
### DQN agents can be found in `.....`
- `DQNAgent`: DQN agent (trained via `train_dqn.py`)
### PPO agents can be found in `.....`
- `PolicyAgent`: PPO agent (trained via `train_ppo.py`)
---

## Project Structure

```
rl-multiagent-boxing/
├── agents/
│   ├── __init__.py
│   ├── dqn_agent.py
│   ├── policy_agent.py
│   ├── rando_agent1.py
│   └── rando_agent2.py
├── models/
│   ├── ppo_model.h5
│   └── dqn_model. (keras or h5)
├── training/
│   ├── train_ppo.py
│   └── train_dqn.py
├── main.py                                 # PPO vs DQN match
├── main_rando.py                           # RandoAgent1 vs RandoAgent2
├── training_requirements_draft.txt         # A SAMPLE FILE FOR TRAINING ON HEX
├── rendering_requirements.txt              # For local rendering
├── og_rendering_requirements.txt           # For WSL local rendering
└── README.md
```

---

## References

- [Gymnasium ALE Environments](https://gymnasium.farama.org/environments/atari/)
- [AutoROM tool](https://github.com/Farama-Foundation/AutoROM)

---

_University of Bath — CM50270 Reinforcement Learning Coursework 2_
