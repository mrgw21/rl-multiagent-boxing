FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/London
WORKDIR /app

# --- System dependencies ---
RUN apt-get update && apt-get install -y \
    python3.8 python3-pip \
    cmake build-essential ninja-build \
    zlib1g-dev libsdl2-dev git wget unzip \
    libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Copy requirements and install Python dependencies ---
COPY requirements.txt .

RUN python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install -r requirements.txt && \
    python3.8 -m pip install opencv-python-headless

# --- Install Atari ROMs (AutoROM is installed by requirements) ---
RUN HOME=/root AutoROM --accept-license

# --- Copy rest of codebase ---
COPY . .

# --- Entrypoint: run ppo_main.py (not as module) ---
CMD ["python3.8", "training/gpu/ppo_main_gpu.py"]
