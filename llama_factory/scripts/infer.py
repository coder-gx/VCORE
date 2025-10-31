# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import json
import os
from typing import Optional
import random
import torch

import fire
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments
import numpy as np

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: Optional[str] = None,
    enable_thinking: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    video_fps: float = 2.0,
    video_maxlen: int = 128,
    batch_size: int = 1024,
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    # to ensure reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # optional: 保证 deterministic
    torch.backends.cudnn.benchmark = False
     
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )
    print(generating_args)

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate
    
   
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )

    peft_model_id = model_args.adapter_name_or_path[0]
    config = PeftConfig.from_pretrained(peft_model_id)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.eval()
    model = torch.compile(model)
    model=model.to("cuda")
   

    # Store all results in these lists
    # all_prompts, all_preds, all_labels = [], [], []

    # Add batch process to avoid the issue of too many files opened
    for i in tqdm(range(0, len(train_dataset), batch_size), desc="Processing batched inference"):
        vllm_inputs, prompts, labels = [], [], []
        batch = train_dataset[i : min(i + batch_size, len(train_dataset))]
       
        for j in range(len(batch["input_ids"])):
          
            prompts.append(tokenizer.decode(batch["input_ids"][j], skip_special_tokens=skip_special_tokens))
            labels.append(
                tokenizer.decode(
                    list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                    skip_special_tokens=skip_special_tokens,
                )
            )
        vllm_inputs, attention_mask = pad_input_ids_and_create_attention_mask(batch["input_ids"], pad_token_id=tokenizer.pad_token_id, device=model.device)
        # vllm_inputs=torch.tensor(batch["input_ids"]).to(model.device)
        # attention_mask = torch.tensor(batch["attention_mask"]).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=vllm_inputs, 
                attention_mask=attention_mask,
                max_new_tokens= generating_args.max_new_tokens,
                do_sample=False,
                repetition_penalty=generating_args.repetition_penalty,
                eos_token_id=template_obj.get_stop_token_ids(tokenizer),
            )

        preds = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Accumulate results
        # all_prompts.extend(prompts)
        # all_preds.extend(preds)
        # all_labels.extend(labels)
        gc.collect()

        # Write all results at once outside the loop
        save_dir = os.path.dirname(save_name)

        # 自动创建目录（如果不存在）
        os.makedirs(save_dir, exist_ok=True)

        with open(save_name, "a", encoding="utf-8") as f:
            for text, pred, label in zip(prompts, preds, labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

       
def pad_input_ids_and_create_attention_mask(input_ids_list, pad_token_id=0, device="cuda"):
    # 1. 获取最大长度
    max_len = max(len(ids) for ids in input_ids_list)

    # 2. 创建 padded input_ids 和 attention_mask
    padded_input_ids = []
    attention_masks = []

    for ids in input_ids_list:
        pad_len = max_len - len(ids)
        padded_input_ids.append(ids + [pad_token_id] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)

    # 3. 转换为 tensor
    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
    attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long, device=device)

    return input_ids_tensor, attention_mask_tensor

if __name__ == "__main__":
    fire.Fire(vllm_infer)
