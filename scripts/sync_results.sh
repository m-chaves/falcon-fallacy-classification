#!/bin/bash

# Script to synchronize the model files from the remote server to the local machine
# Must be run from the local machine
# model.pt must be synchronized from the remote server to the local machine in this way because the model files are too large to be stored in the git repository

# Define the source and destination directories
SOURCE_DIR="username@cluster.domain.fr:/home/username/twitter_fallacy_classification/results/models/"
DEST_DIR="/user/username/home/Documents/twitter_fallacy_classification/results/models/"

# Run the rsync command
rsync -avz --include '*/' --include='model.pt' --exclude='*' -e 'ssh -J cluster.domain.fr' "$SOURCE_DIR" "$DEST_DIR"

# Check the md5sum of the files in the remote and local directories
# This is done to ensure that the files were copied correctly
# Check the 2 txt files (local_md5sums.txt and remote_md5sums.txt) to see if the md5sums match
ssh -J cluster.domain.fr username@nef-devel2.inria.fr "find /home/username/twitter_fallacy_classification/results/models/ -name 'model.pt' -exec md5sum {} \;" > ./scripts/remote_md5sums.txt
find /user/username/home/Documents/twitter_fallacy_classification/results/models/ -name 'model.pt' -exec md5sum {} \; > ./scripts/local_md5sums.txt
