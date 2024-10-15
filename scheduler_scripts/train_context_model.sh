#!/bin/bash

# Activate the virtual environment
source ./venv_torch/bin/activate
# Load the CUDA and cuDNN modules (pytorch needs them to use the GPU)
# We check for available module versions and choose the latest ones. Therefore you might need to change the versions below. Use module avail.
module load cuda/12.2
module load cudnn/8.9-cuda-12.1

# Ask torch if it can use the GPU
python3 -c "import torch; print('Pytorch can use the GPU:', torch.cuda.is_available()); print('torch version', torch.__version__); print('cuda version', torch.version.cuda); print('cudnn version', torch.backends.cudnn.version())"

# Check GPU specs
nvidia-smi
# Check CPU specs
lscpu

# Train
python3 train_context_model.py "$@"

# # the last line is to pass the arguments to the script from the args_context_models.txt file
# The following is an example of how to pass the arguments directly to the script

# python3 train_context_model.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --num_epochs 50 \
# --feature_prefixes VADER_ VAD_
