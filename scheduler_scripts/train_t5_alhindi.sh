# Activate the virtual environment
source ./project_venv/bin/activate

# Ask torch if it can use the GPU
python3 -c "import torch; print('Pytorch can use the GPU:', torch.cuda.is_available()); print('torch version', torch.__version__); print('cuda version', torch.version.cuda); print('cudnn version', torch.backends.cudnn.version())"

# Train the model
export BS=2;
USE_TF=0
python train/train_alhindi_model.py \
  --output_dir results/models/ \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --do_train \
  --do_eval \
  --learning_rate 1e-4 \
  --overwrite_output_dir \
  --max_source_length 1024 \
  --max_target_length 64 \
  --gradient_accumulation_steps 512 \
  --per_device_train_batch_size $BS \
  --per_device_eval_batch_size $BS \
  --source_lang input \
  --target_lang target \
  "$@"

  # the last line is to pass the arguments to the script from the args_t5.txt file
  # they should look like this
  # --model_name_or_path t5-large \
  # --train_file datasets/prompt_data_for_t5_models/prompt_type_with_fallacy_definition/train.jsonl \
  # --validation_file datasets/prompt_data_for_t5_models/prompt_type_with_fallacy_definition/val.jsonl \
  # --test_file datasets/prompt_data_for_t5_models/prompt_type_with_fallacy_definition/test.jsonl \
  # --num_train_epochs 2 \
  # --seed 98