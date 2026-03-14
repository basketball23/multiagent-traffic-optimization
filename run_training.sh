#!/bin/bash

# 1. Create a unique log file name using the current date and time
LOG_FILE="training_log_$(date +%Y%m%d_%H%M%S).txt"

echo "==========================================================="
echo " Starting training..."
echo " Logging all output to: $LOG_FILE"
echo " Your Mac is caffeinated and will stay awake."
echo " Press Ctrl + C to stop training at any time."
echo "==========================================================="

# 2. Run the training script with caffeinate
# The '2>&1 | tee' part ensures you see the output live on your screen 
# WHILE it also saves it to the text file.
caffeinate -i python3 train.py 2>&1 | tee "$LOG_FILE"

echo "==========================================================="
echo " Training finished or stopped. Your Mac can now sleep."
echo "==========================================================="
