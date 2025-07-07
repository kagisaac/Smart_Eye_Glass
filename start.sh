#!/bin/bash
sleep 40  # Give the desktop some time to load
export DISPLAY=:0

cd /home/isaacpi/Smart_Eye_Glass
source venv/bin/activate
python3 src/main.py
