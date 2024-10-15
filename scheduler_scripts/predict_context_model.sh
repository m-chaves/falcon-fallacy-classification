#!/bin/bash

# Activate the virtual environment
source ./venv_torch/bin/activate

python3 predict_context_model.py \
    --model_dir results/models/microsoft_deberta-v3-base_with_context_VADER_VAD__finetuned_50_epochs/run_3 \
    --model_name_or_path microsoft/deberta-v3-base \
    --feature_prefixes VADER_ VAD_
