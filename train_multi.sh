#!/bin/bash
# 在同一个 screen 中顺序跑多次 llama-factory 训练（每次参数可不同）
set -euo pipefail

#### 设备/环境（按需保留） ####
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)
if [[ "${gpu_name:-}" == *"4090"* ]]; then
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  echo "Detected GPU: $gpu_name -> Disabled NCCL P2P & IB."
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
set -x

#### 目录 ####
LOG_ROOT=./llama_factory/screen_logs_final
# mkdir -p "$LOG_ROOT" "$SAVE_ROOT"

#### 实验清单：一行一个实验，用 key=value 写要改的字段；没写的走默认
EXPS=(
  # VCORE
 'name=vcore_qwen3_8b_math_e1e_4_s1_T0_5 model_name_or_path=/mnt-nfsdata/model_base/Qwen3-8B lora_rank=8 lora_alpha=16 lora_dropout=0.1 dataset=math_sft template=qwen3 output_dir=/gemini/data-1/margin_sft/llama_factory/saves/math/qwen3_8b_vcore/lora/sft_e1e_4_s1_T0_5  save_steps=10 learning_rate=2e-5 warmup_ratio=0.1 use_ours=True ours_temperature=5e3 ours_anchor_steps=1 ours_store_mode=state_dict epsilon=1e-4  ours_pre_ga=-1'
 'name=vcore_llama_math_e1e_5_s1_T0_8  model_name_or_path=/gemini/data-3/model_base/Meta-Llama-3.1-8B-Instruct lora_rank=64 lora_alpha=128 lora_dropout=0.05 dataset=math_sft_llama  template=llama3 output_dir=/gemini/data-1/margin_sft/llama_factory/saves/math/llama_instruct_vcore/lora/sft_e1e_5_s1_T0_8 save_steps=10 learning_rate=2e-4 warmup_ratio=0.05 use_ours=True ours_temperature=8e4 ours_anchor_steps=1 ours_store_mode=state_dict epsilon=1e-5 ours_pre_ga=-1' 

 'name=vcore_qwen3_8b_code_e1e_4_s1_T0_5 model_name_or_path=/mnt-nfsdata/model_base/Qwen3-8B lora_rank=8 lora_alpha=16 lora_dropout=0.1 dataset=code_sft template=qwen3 output_dir=/gemini/data-1/margin_sft/llama_factory/saves/code/qwen3_8b_vcore/lora/sft_e1e_4_s1_T0_5  save_steps=10 learning_rate=2e-5 warmup_ratio=0.1 use_ours=True ours_temperature=5e3 ours_anchor_steps=1 ours_store_mode=state_dict epsilon=1e-4  ours_pre_ga=-1'
 'name=vcore_llama_code_oprec_e1e_5_s1_T0_8  model_name_or_path=/gemini/data-3/model_base/Meta-Llama-3.1-8B-Instruct lora_rank=64 lora_alpha=128 lora_dropout=0.05 dataset=code_sft_llama  template=llama3 output_dir=/gemini/data-1/margin_sft/llama_factory/saves/code/llama_instruct_vcore/lora/sft_e1e_5_s1_T0_8 save_steps=10 learning_rate=2e-4 warmup_ratio=0.05 use_ours=True ours_temperature=8e4 ours_anchor_steps=1 ours_store_mode=state_dict epsilon=1e-5 ours_pre_ga=-1' 
  # DFT
 'name=dft_qwen3_32b_math_dft model_name_or_path=/mnt-nfsdata/model_base/Qwen3-32B lora_rank=8 lora_alpha=16 lora_dropout=0.1 dataset=math_sft template=qwen3 output_dir=/gemini/data-1/margin_sft/llama_factory/saves/math/qwen3_32b_dft/lora/sft  save_steps=10 learning_rate=2e-5 warmup_ratio=0.1 use_dft=True'

  # Original

  'name=qwen3_4b_math model_name_or_path=/mnt-nfsdata/model_base/Qwen3-4B lora_rank=8 lora_alpha=16 lora_dropout=0.1 dataset=math_sft template=qwen3 output_dir=/gemini/data-1/margin_sft/llama_factory/saves/math/qwen3_4b/lora/sft  save_steps=10 learning_rate=2e-5 warmup_ratio=0.1'


)



#### 公共固定项（不常改的都放这里；需要时再挪到 EXPS 行里）
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
  --gradient_accumulation_steps 8
  --num_train_epochs 1
  --lr_scheduler_type cosine
  --bf16
  --seed 42
  --ddp_timeout 180000000
  --ddp_backend nccl
)

#### 顺序执行所有实验
for cfg in "${EXPS[@]}"; do
  # 先清空旧变量（防止上一轮的值“遗留”）
  unset name model_name_or_path lora_rank lora_alpha lora_dropout dataset template output_dir save_steps learning_rate warmup_ratio use_dft use_ours ours_temperature ours_anchor_steps ours_store_mode epsilon ours_pre_ga

  # 解析 key=value 到变量
  eval "$cfg"

  # 填默认值（不在 EXPS 行里写的，就用这里的默认）
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
  use_ours=${use_ours:-False}
  ours_anchor_steps=${ours_anchor_steps:-1}
  ours_store_mode=${ours_store_mode:-module}  # module / full
  ours_temperature=${ours_temperature:-1.0}
  epsilon=${epsilon:-2e-5}
  ours_pre_ga=${ours_pre_ga:-8}

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
    --use_ours "${use_ours}" \
    --ours_anchor_steps "${ours_anchor_steps}" \
    --ours_store_mode "${ours_store_mode}" \
    --ours_temperature "${ours_temperature}" \
    --epsilon "${epsilon}" \
    --ours_pre_ga "${ours_pre_ga}" \
    "${COMMON_FLAGS[@]}" 2>&1 | tee -a "$LOG_FILE"

  echo "========== DONE ${name} ==========" | tee -a "$LOG_FILE"
done

echo ">>> All experiments finished."
