FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/London
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-distutils \
    cmake build-essential \
    zlib1g-dev libsdl2-dev git wget unzip \
    libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure python and pip point to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy and install Python dependencies
COPY og_requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r og_requirements.txt && \
    pip install opencv-python-headless

# Install Atari ROMs
RUN AutoROM --accept-license

# Copy project files
COPY . .

CMD ["/bin/bash"]
