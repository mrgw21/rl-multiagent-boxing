FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Prevent tzdata interactive prompt
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/London

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.8 python3-pip \
    cmake build-essential ninja-build \
    zlib1g-dev libsdl2-dev git wget unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install \
        tensorflow==2.10.1 \
        gymnasium ale-py autorom \
        matplotlib seaborn

# Install Atari ROMs automatically
RUN AutoROM --accept-license

# Copy your project code
COPY . .

# Set default training entrypoint
CMD ["python3.8", "-m", "training.train_ppo"]
