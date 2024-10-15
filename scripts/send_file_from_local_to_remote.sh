#!/bin/bash

# Must be run from the local machine
# Define the source and destination directories, for example:
SOURCE_FILE="/user/username/home/Documents/twitter_fallacy_classification/results/models/microsoft_deberta-v3-base_with_context_VADER_VAD__finetuned_50_epochs/run_3/model.pt"
DEST_DIR="username@cluster.domain.fr:/home/username/twitter_fallacy_classification/results/models/microsoft_deberta-v3-base_with_context_VADER_VAD__finetuned_50_epochs/run_3/"

# Use rsync with ProxyJump to transfer the file, for example:
rsync -avzP -e "ssh -J cluster.domain.fr" "$SOURCE_FILE" "$DEST_DIR"

# Print a message when done
echo "Transfer complete."
