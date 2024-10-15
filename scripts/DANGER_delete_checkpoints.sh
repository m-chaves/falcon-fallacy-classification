#!/bin/bash

# WARNING: This script will delete all checkpoint directories and their contents inside the results/models/ directory and its subdirectories.
# Use with caution. Ensure you have backups if needed.
# It was created because model files are too large to be stored in the cluster for long periods. 
# If you don't have any space constraints DO NOT RUN THIS SCRIPT.

read -p "Are you sure you want to delete all checkpoints? This action is irreversible. Type 'yes' to proceed: " confirm

if [ "$confirm" == "yes" ]; then
    find results/models/ -type d -name "checkpoint-*" -exec rm -r {} +
    echo "Files deleted."
else
    echo "Operation canceled."
fi
