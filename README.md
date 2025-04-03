# RL Boxing Project

A reinforcement learning project built for the CM50270 coursework at the University of Bath.

This project trains two agents (PPO and DQN) to play Atari Boxing using the Gym ALE environment. You can also test agents like RandomAgent1 vs RandomAgent2 or watch the final trained models compete.

## Getting Started

NOTE: On Windows, things are a little complicated. The multi-agent-ale package won't compile at all, so you need to install it under [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install). Install WSL, then follow the Linux instructions below plus any WSL specific comments :)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-TEAM-NAME/rl-boxing.git
cd rl-boxing
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv venv

#For Mac/Linux/WSL:
source venv/bin/activate
```

### 3. Install the dependencies
```bash
sudo apt install cmake swig zlib1g-dev #Linux only. apt for Ubuntu/Debian etc., your package manager of choice otherwise.

#For Mac/Linux/WSL:
pip install -r requirements.txt 
```

### 4. Download Atari ROMs
```bash
AutoROM --accept-license
```

### 5. Run a test match (e.g., Random Agent 1 vs Random Agent 2)
```bash
python main_rando.py

#If, using WSL, the game runs but with no display window:
nano ~/.bashrc
#Add the following to the end of the file and save
export DISPLAY=:0.0

```

### 6. Train PPO or DQN
```bash
python training/train_ppo.py
python training/train_dqn.py
```

### 7. Watch trained agents fight
```bash
python main.py
```

## Agents

- `RandoAgent1`: Random agent with equal probabilities
- `RandoAgent2`: Slower/weaker random agent (for testing)
- `PolicyAgent`: PPO agent (trained in `train_ppo.py`)
- `DQNAgent`: DQN agent (trained in `train_dqn.py`)

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
│   └── dqn_model.h5
├── training/
│   ├── train_ppo.py
│   └── train_dqn.py
├── main.py            # PPO vs DQN
├── main_rando.py      # Random1 vs Random2
├── requirements.txt
├── windows-requirements.txt
└── README.md
```

## References

- [Gymnasium ALE Environments](https://gymnasium.farama.org/environments/atari/)
- [AutoROM tool](https://github.com/Farama-Foundation/AutoROM)
- [PPO and DQN algorithms](https://spinningup.openai.com/en/latest/algorithms/)

---

_University of Bath — CM50270 Reinforcement Learning Coursework_
