# RL Boxing Project

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

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv

# For Mac/Linux/WSL:
source venv/bin/activate
```

---

### 3. Install Dependencies

#### If you're rendering the boxing gym locally:

```bash
pip install -r render_requirements.txt
```

#### If you're rendering in WSL (with X server like VcXsrv):

```bash
sudo apt update
sudo apt install cmake swig zlib1g-dev libboost-all-dev \
                 libsdl2-dev libsdl2-image-dev \
                 python3-dev build-essential

pip install -r og_requirements.txt

# Then add this to ~/.bashrc or ~/.zshrc:
export DISPLAY=:0.0
```

#### If you're training agents (on Hex):

```bash
pip install -r (your requirements txt file name).txt
```

---

### 4. Download Atari ROMs

```bash
AutoROM --accept-license
```

---

### 5. Run a test match (e.g., RandoAgent1 vs RandoAgent2)

```bash
python main_rando.py
```

---

### 6. Train PPO or DQN agents

```bash
python training/train_ppo.py
python training/train_dqn.py
```

---

### 7. Watch trained agents compete

```bash
python main.py
```

---

## Agents

- `RandoAgent1`: Random agent with equal probabilities
- `RandoAgent2`: Slower/weaker random agent (for testing)
- `PolicyAgent`: PPO agent (trained via `train_ppo.py`)
- `DQNAgent`: DQN agent (trained via `train_dqn.py`)

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
