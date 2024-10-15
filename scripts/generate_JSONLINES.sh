#!/bin/bash

# Define the arrays for the arguments
prompt_types=("with_fallacy_definition" "with_fallacy_definition_with_NotA" "without_fallacy_definition" "without_fallacy_definition_with_NotA")

# Loop through all combinations of prompt_type and target_format
for prompt_type in "${prompt_types[@]}"; do
    python generate_JSONLINES.py \
      --train_csv datasets/train_val_test_sets/df_train.csv \
      --val_csv datasets/train_val_test_sets/df_val.csv \
      --test_csv datasets/train_val_test_sets/df_test.csv \
      --output_dir datasets/prompt_data_for_t5_models/ \
      --prompt_type "$prompt_type"
done