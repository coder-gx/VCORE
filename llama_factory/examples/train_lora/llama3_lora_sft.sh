#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=3,4,5,6 

set -x

MODEL_PATH=/gemini/data-3/model_base/Meta-Llama-3.1-8B

llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target all \
    --dataset strategyqa_cot \
    --template llama3 \
    --cutoff_len 4096 \
    --max_samples 20000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir saves/llama3-8b/lora/sft \
    --logging_steps 1 \
    --save_strategy epoch \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model true \
    --report_to swanlab \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000
