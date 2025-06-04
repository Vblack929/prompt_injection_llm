#!/bin/bash

# run_training.sh - Script to run the struq training

# Set default values
MODEL_NAME="Qwen/Qwen3-0.6B"
DATA_PATH="datasets/alpaca_data_with_input_500.jsonl"
ATTACK_TYPE="SpclSpclSpcl_Ignore"
OUTPUT_DIR="./outputs/struq_training/qwen3-0.6b_500"
NUM_EPOCHS=3
BATCH_SIZE=16
LEARNING_RATE=2e-5
MAX_LENGTH=512

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Set device - check if MPS is available (Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the training
python struq.py \
    --model_name_or_path $MODEL_NAME \
    --data_path $DATA_PATH \
    --attack $ATTACK_TYPE \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --model_max_length $MAX_LENGTH \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 2 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_dir "$OUTPUT_DIR/logs" \
    --report_to "none" \
    --downsample true \
    --lr_scale true \
    --fp16 false \
    --padding_side "right" \
    --use_mps_device true

echo "Training completed! Model saved to: $OUTPUT_DIR" 