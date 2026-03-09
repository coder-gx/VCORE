#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -x

LOG_ROOT=/home/code/VCORE/llama_factory/screen_logs
mkdir -p "$LOG_ROOT" 

#### EXP LIST
EXPS=(

'name=qwen3_4b_math_vcore_branch   model_name_or_path=/home/data-1/model_base/Qwen3-4B lora_rank=8 lora_alpha=16 lora_dropout=0.1 dataset=math_sft template=qwen3 output_dir=/home/code/VCORE/output/math/qwen3_4b_vcore/lora/sft save_steps=100 gradient_accumulation_steps=4 learning_rate=2e-5 warmup_ratio=0.1 use_vcore=True vcore_temperature=5e3 vcore_anchor_steps=1 vcore_epsilon=1e-4 vcore_pre_ga=4  vcore_single_process=False main_or_branch=branch'
# 'name=qwen3_4b_math_sft   model_name_or_path=/home/data-1/model_base/Qwen3-4B lora_rank=8 lora_alpha=16 lora_dropout=0.1 dataset=math_sft template=qwen3 output_dir=/home/code/VCORE/output/math/qwen3_4b_sft/lora/sft save_steps=100 gradient_accumulation_steps=8 learning_rate=2e-5 warmup_ratio=0.1 use_vcore=False use_dft=False'
# 'name=qwen3_4b_math_dft   model_name_or_path=/home/data-1/model_base/Qwen3-4B lora_rank=8 lora_alpha=16 lora_dropout=0.1 dataset=math_sft template=qwen3 output_dir=/home/code/VCORE/output/math/qwen3_4b_dft/lora/sft save_steps=100 gradient_accumulation_steps=8 learning_rate=2e-5 warmup_ratio=0.1 use_vcore=False use_dft=True'

)


#### COMMON FLAGS
COMMON_FLAGS=(
  --trust_remote_code
  --stage sft
  --do_train
  --finetuning_type lora
  --lora_target all
  --cutoff_len 16384
  --max_samples 1000000000
  --overwrite_cache
  --preprocessing_num_workers 16
  --dataloader_num_workers 4
  --logging_steps 1
  --plot_loss
  --overwrite_output_dir
  --save_only_model true
  --report_to swanlab
  --per_device_train_batch_size 1
  --num_train_epochs 1
  --lr_scheduler_type cosine
  --bf16
  --seed 42
  --ddp_timeout 180000000
  --ddp_backend nccl
)

#### RUN EXPERIMENTS
for cfg in "${EXPS[@]}"; do
  # CLEAR VARIABLES
  unset name model_name_or_path lora_rank lora_alpha lora_dropout dataset template output_dir save_steps learning_rate warmup_ratio use_dft use_vcore vcore_temperature vcore_anchor_steps  vcore_epsilon  vcore_pre_ga gradient_accumulation_steps main_or_branch vcore_single_process
  eval "$cfg"
  name=${name:-exp_$(date +%Y%m%d-%H%M%S)}
  model_name_or_path=${model_name_or_path:-/mnt-nfsdata/model_base/Qwen3-8B}
  lora_rank=${lora_rank:-8}
  lora_alpha=${lora_alpha:-16}
  lora_dropout=${lora_dropout:-0.1}
  dataset=${dataset:-math_sft}
  template=${template:-llama3}
  output_dir=${output_dir:-${SAVE_ROOT}/math/llama_instruct/lora/${name}}
  save_steps=${save_steps:-5}
  learning_rate=${learning_rate:-2e-5}
  warmup_ratio=${warmup_ratio:-0.1}
  use_dft=${use_dft:-False}
  use_vcore=${use_vcore:-False}
  vcore_anchor_steps=${vcore_anchor_steps:-1}
  vcore_temperature=${vcore_temperature:-1.0}
  vcore_epsilon=${vcore_epsilon:-2e-5}
  vcore_pre_ga=${vcore_pre_ga:-8}
  gradient_accumulation_steps=${gradient_accumulation_steps:-8}
  main_or_branch=${main_or_branch:-main}
  vcore_single_process=${vcore_single_process:-True}

  mkdir -p "$output_dir"

  DATE=$(date +%Y%m%d-%H%M%S)
  LOG_FILE=${LOG_ROOT}/train_${name}_${DATE}.log

  echo "========== RUN ${name} ==========" | tee -a "$LOG_FILE"
  echo "OUT_DIR=$output_dir" | tee -a "$LOG_FILE"

  llamafactory-cli train \
    --model_name_or_path "${model_name_or_path}" \
    --finetuning_type lora \
    --lora_rank "${lora_rank}" \
    --lora_alpha "${lora_alpha}" \
    --lora_dropout "${lora_dropout}" \
    --dataset "${dataset}" \
    --template "${template}" \
    --output_dir "${output_dir}" \
    --save_steps "${save_steps}" \
    --learning_rate "${learning_rate}" \
    --warmup_ratio "${warmup_ratio}" \
    --use_dft "${use_dft}" \
    --use_vcore "${use_vcore}" \
    --vcore_anchor_steps "${vcore_anchor_steps}" \
    --vcore_temperature "${vcore_temperature}" \
    --vcore_epsilon "${vcore_epsilon}" \
    --vcore_pre_ga "${vcore_pre_ga}" \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --main_or_branch "${main_or_branch}" \
    --vcore_single_process "${vcore_single_process}" \
    "${COMMON_FLAGS[@]}" 2>&1 | tee -a "$LOG_FILE"

  echo "========== DONE ${name} ==========" | tee -a "$LOG_FILE"
done

echo ">>> All experiments finished."
