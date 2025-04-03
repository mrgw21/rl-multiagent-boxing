#!/bin/bash
set -e

# Make sure ROMs are installed
AutoROM --accept-license

# Start training
python -m training.train_ppo
