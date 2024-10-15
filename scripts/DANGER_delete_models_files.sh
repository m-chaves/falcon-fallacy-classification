#!/bin/bash

# WARNING: This script will delete all model.pt files inside the results/models/ directory and its subdirectories.
# Use with caution. Ensure you have backups if needed.
# This script is intended for cleaning up model files after synchronization or other operations.
# It was created because model files are too large to be stored in the cluster for long periods. 
# If you don't have any space constraints DO NOT RUN THIS SCRIPT.

read -p "Are you sure you want to delete all model.pt files? This action is irreversible. Type 'yes' to proceed: " confirm

if [ "$confirm" == "yes" ]; then
    find results/models/ -type f -name "model.pt" -delete
    echo "All model.pt files have been deleted."
else
    echo "Operation canceled."
fi
