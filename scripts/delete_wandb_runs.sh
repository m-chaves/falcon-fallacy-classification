#!/bin/bash

# This script deletes all Weights & Biases runs that are older than a specified cutoff date.

# Define the cutoff date in YYYYMMDD format
CUTOFF_DATE="20240701"

# Extract the year and month from the cutoff date
CUTOFF_YEAR_MONTH=${CUTOFF_DATE:0:6}
CUTOFF_DAY=${CUTOFF_DATE:6:2}

# Find and delete directories older than the cutoff date
find wandb/ -type d -name "run-*" | while read -r dir; do
    # Extract the date part from the directory name
    DIR_DATE=$(basename "$dir" | cut -d'-' -f2 | cut -c1-8)
    
    # If the directory date is less than the cutoff date, delete the directory
    if [[ "$DIR_DATE" < "$CUTOFF_DATE" ]]; then
        echo "Deleting directory: $dir"
        rm -rf "$dir"
    fi
done