# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union, List, Optional, Dict, Tuple
import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override
import math
import random

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from torch.utils.data import DataLoader, RandomSampler


##############################new added##############################
import contextlib
from packaging import version
import functools
import shutil
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
import numpy as np
import torch
import importlib.metadata
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset
from transformers import __version__
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.processing_utils import ProcessorMixin
from transformers.trainer_callback import (
    ExportableState,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    get_model_param_count,
)
from transformers.trainer_utils import (
    PredictionOutput,
    TrainOutput,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.utils import (
    XLA_FSDPV2_MIN_VERSION,
    is_accelerate_available,
    is_apex_available,
    is_peft_available,
    is_sagemaker_mp_enabled,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    logging,
)
import copy
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available

if is_apex_available():
    from apex import amp
from contextlib import contextmanager

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_peft_available():
    from peft import PeftModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.runtime as xr
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedType,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("1.3.0"):
        from accelerate.utils import TorchTensorParallelPlugin
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,)
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

######################################new added##############################




if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        use_dft: bool = False,
        use_ours: bool = False,
        ours_pre_ga: int = 8,
        ours_temperature: float = 1.0,
        ours_anchor_steps: int = 1,
        ours_store_mode: str = "module",
        epsilon: float = 2e-5,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
        
        self.use_dft = use_dft
        self.pre_ga = ours_pre_ga
        self.use_ours = use_ours
        self.ours_temperature = ours_temperature
        self.ours_anchor_steps = ours_anchor_steps
        self.ours_store_mode = ours_store_mode
        self.epsilon = epsilon
        self._dualref_ma = None      # 冻结的参考模型A（module 或 state_dict）
        self._dualref_mb = None      # 冻结的参考模型B（module 或 state_dict）
        self._dualref_active = False # 是否已建立 ma/mb，且在 j 前持续使用

   
    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":



        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        
        return super().compute_loss(model, inputs,self.use_dft, self.use_ours,self.ours_temperature, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
    


    @override
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False
      
        if self.is_deepspeed_enabled:
            def _count_trainables(m):
                return sum(p.requires_grad for p in unwrap_model(m).parameters())

            # 1) 确认真的有可训练参数
            if _count_trainables(self.model) == 0:
                raise RuntimeError(
                    "No trainable parameters detected. If using PEFT/LoRA, "
                    "ensure `mark_only_lora_as_trainable` was called and an adapter is active."
                )

            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # ====== 新增：只让 LoRA 参与训练 ======
        base = unwrap_model(model)
        if _is_peft_model(model):
            _mark_only_lora_as_trainable(base)
            try:
                base.print_trainable_parameters()
            except Exception:
                pass
        else:
            # 非 PEFT 的兜底：按名字启用 LoRA / modules_to_save
            _mark_only_lora_as_trainable(base)




        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)


            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )
            
            # print('use_ours:', self.use_ours, 'ours_anchor_steps:', self.ours_anchor_steps,'temp:', self.ours_temperature)
           
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                
                # lr 最小为1e-6
                for group in self.optimizer.param_groups:
                    group["lr"] = max(1e-6, group["lr"])


               # >>>>>>> 这里新增 anchor 窗口分支 <<<<<<<
                is_anchor_window = False
                if self.use_ours:
                    # 当前将要执行的 optimizer step 编号是 self.state.global_step
                    is_anchor_window = (update_step%self.ours_anchor_steps==0)

                if is_anchor_window:
                    if self.pre_ga<0:
                        pre_ga = args.gradient_accumulation_steps
                    else:
                        pre_ga = self.pre_ga   # 可配置
                    rand_batch_samples, rand_num_items_in_batch = self.get_random_batch_samples(
                        epoch_dataloader, pre_ga, args.device
                    )
                    # print(len(rand_batch_samples))

                  
                    # 直接用“两遍法”处理这个窗口，并跳过原来的内层 microbatch for 循环
                    tr_loss, grad_norm, learning_rate,step= self._dualref_handle_anchor_window(
                        model, batch_samples, args, steps_in_epoch, steps_skipped, epoch,step,
                        tr_loss, grad_norm, trial, ignore_keys_for_eval, start_time,num_items_in_batch, 
                        rand_batch_samples, rand_num_items_in_batch 
                    )
                    # 注意这里 step++，因为我们跳过了内层循环
                    # 到这里，这个 update_step 已经完成（包含日志/回调/step+1），继续下一个窗口
                    continue
                # if is_anchor_window:
                #     # 取“下一批”作为 B′，遇到末尾自动环回
                #     next_batch_samples, next_num_items_in_batch, epoch_iterator = self.get_next_batch_samples(
                #         epoch_iterator, epoch_dataloader, num_batches, args.device
                #     )
                #     # last_batch_samples, last_num_items_in_batch = self.get_last_batch(epoch_dataloader,args.device)
                #     # print(next_batch_samples[0].keys())
                  

                #     tr_loss, grad_norm, learning_rate, step = self._dualref_handle_anchor_window(
                #         model, batch_samples, args, steps_in_epoch, steps_skipped, epoch, step,
                #         tr_loss, grad_norm, trial, ignore_keys_for_eval, start_time, num_items_in_batch,
                #         next_batch_samples, next_num_items_in_batch
                #     )
                #     continue



                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        # get leaning rate before update
                        learning_rate = self._get_learning_rate()

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
                
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    

    @override
    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Dual-Ref aware training_step:
        - 若已激活 ma/mb 且未抑制，则使用新方法 loss：先取 ma/mb 的 token-level loss，再调用 dual_ref_loss_fn。
        - 否则使用标准 compute_loss。
        - 仍保持在此处 backward（与 GA>1 的既有行为一致）。
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        use_dual = (
            self.use_ours
            and getattr(self, "_dualref_active", False)
            and not getattr(self, "_dualref_suppress_new_loss", False)
        )

        with self.compute_loss_context_manager():
            if use_dual:
                with torch.no_grad():
                    tok_a, tok_b = self._dualref_eval_token_losses_with_refs(inputs)
                # if hasattr(self, "optimizer") and self.optimizer is not None:
                #     if len(self.optimizer.param_groups) > 0:
                #         current_lr = self.optimizer.param_groups[0]["lr"]
                # else:
                #     raise ValueError("optimizer not found when using dual-ref loss.")
                
                
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch,loss_a=tok_a, loss_b=tok_b,cur_lr=self.epsilon)
            else:
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available():
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning("`torch_empty_cache_steps` set but HPU doesn't support empty_cache().")
            else:
                torch.cuda.empty_cache()

        kwargs = {}
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()
        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # 若 compute_loss 未做 GA 归一化，则此处除以 GA（HF 原逻辑保持）
            # print("loss 1:", loss.item())
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps
            # print("loss 2:", loss.item())
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
            self.accelerator.backward(loss, **kwargs)
            return loss.detach()

    
    def _dualref_reset_buffers(self):
        self._dualref_ma = None      # 冻结的参考模型A（module 或 state_dict）
        self._dualref_mb = None      # 冻结的参考模型B（module 或 state_dict）
        self._dualref_active = False # 是否已建立 ma/mb，且在 j 前持续使用
    
    # ---- 快照/恢复工具（FSDP/Accelerate 友好） ----
    def _dualref_get_state_dict(self, model):
        base = unwrap_model(model)
        # 推荐优先走 accelerate 的收集（FSDP 会帮你全量聚合到 CPU）
        if hasattr(self, "accelerator"):
            sd = self.accelerator.get_state_dict(base)
        else:
            sd = base.state_dict()  # 这一步可能返回共享存储的张量

        # 关键：脱离计算图独立存储
        sd = {k: v.detach().clone() for k, v in sd.items()}
        return sd

    def _dualref_load_state_dict(self, model, state_dict):
        base = unwrap_model(model)  # <- 关键：只给 unwrapped 模型 load
        # —— 前缀自适配：若 state_dict 带 module.，去掉；若目标需要 module.（一般不需要），再加上 —
        def strip_module_prefix(sd):
            if all(k.startswith("module.") for k in sd.keys()):
                return {k[len("module."):] : v for k, v in sd.items()}
            return sd

        sd = strip_module_prefix(state_dict)

        incompatible = base.load_state_dict(sd, strict=True)
        # 兼容不同 PyTorch 版本的返回类型
        if hasattr(incompatible, "missing_keys"):
            missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
        else:
            missing, unexpected = incompatible  # 旧版本返回 tuple

        if missing or unexpected:
            raise RuntimeError(
                f"state_dict mismatch when loading into unwrapped model:\n"
                f"  missing={missing}\n  unexpected={unexpected}\n"
                f"  hint: check LoRA/PEFT prefixes and whether you saved adapters+base."
            )

    def _dualref_clone_module(self, model):
        # 最稳妥是从 config 重建再 load_state_dict；但简单起见，这里 deepcopy
        # 注意：DeepSpeed/FSDP 包裹器需 unwrap；LoRA/PEFT 模型 deepcopy 通常也可用
        base = unwrap_model(model)
        clone = copy.deepcopy(base).to(next(base.parameters()).device)
        clone.eval()
        for p in clone.parameters():
            p.requires_grad_(False)
        return clone

    def _dualref_build_frozen_refs(self, mi_state: Dict[str, torch.Tensor], mb_state: Dict[str, torch.Tensor]):
        def _sd_diff_norm(sd1, sd2):
            tot = 0.0
            for k in set(sd1.keys()) & set(sd2.keys()):
                a, b = sd1[k], sd2[k]
                if a.dtype.is_floating_point:
                    tot += (a.float() - b.float()).abs().sum().item()
            return tot
        logger.info(f"[dualref] ||mb_state - mi_state||_1 = {_sd_diff_norm(mi_state, mb_state):.3e}")
        
        # mode =self.ours_store_mode
        # if mode == "state_dict":
        #     self._dualref_ma = {k: v.clone().cpu() for k, v in mi_state.items()}
        #     self._dualref_mb = {k: v.clone().cpu() for k, v in mb_state.items()}
        # else:  # module
        #     ma = self._dualref_clone_module(self.model)
        #     mb = self._dualref_clone_module(self.model)
        #     self._dualref_load_state_dict(ma, mi_state)
        #     self._dualref_load_state_dict(mb, mb_state)
        #     self._dualref_ma, self._dualref_mb = ma, mb
        mode = self.ours_store_mode

        if mode == "state_dict" and _is_peft_model(self.model):
            # —— 在同一 PeftModel 内注册两个只读适配器，并把 LoRA 权重写进去 ——
            from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

            base = unwrap_model(self.model)  # PeftModel
            peft_cfg = next(iter(base.peft_config.values()))  # 复用已有 LoRA 配置
            # 确保存在两个适配器名，不与当前训练适配器冲突
            self._dualref_adapter_a = "dualref_ma"
            self._dualref_adapter_b = "dualref_mb"
            if self._dualref_adapter_a not in base.peft_config:
                base.add_adapter(self._dualref_adapter_a, peft_cfg)
            if self._dualref_adapter_b not in base.peft_config:
                base.add_adapter(self._dualref_adapter_b, peft_cfg)

            # 写入 LoRA 权重（仅 adapter 子集）
            set_peft_model_state_dict(base, mi_state, adapter_name=self._dualref_adapter_a)
            set_peft_model_state_dict(base, mb_state, adapter_name=self._dualref_adapter_b)

            # 冻结这两个适配器的参数（保险）
            for n, p in base.named_parameters():
                if f".{self._dualref_adapter_a}." in n or f".{self._dualref_adapter_b}." in n:
                    p.requires_grad_(False)

            # 仅记录“我们有适配器可切换”，不要存整机/副本
            self._dualref_ma = self._dualref_adapter_a
            self._dualref_mb = self._dualref_adapter_b

        elif mode == "state_dict":
            # 非 PEFT：继续用 LoRA-only 的 tensor 字典（体积小）
            self._dualref_ma = {k: v.clone().cpu() for k, v in mi_state.items()}
            self._dualref_mb = {k: v.clone().cpu() for k, v in mb_state.items()}

        else:
            # module 模式仍然可以保留，但非常不建议（会占大显存）
            ma = self._dualref_clone_module(self.model)
            mb = self._dualref_clone_module(self.model)
            self._dualref_load_state_dict(ma, mi_state)
            self._dualref_load_state_dict(mb, mb_state)
            self._dualref_ma, self._dualref_mb = ma, mb

        self._dualref_active = True

    
    @torch.no_grad()
    def _dualref_eval_token_losses_with_refs(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (loss_per_token_ma, loss_per_token_mb)，二者 shape 一致。
        """
        if not self._dualref_active:
            raise RuntimeError("dual-ref not active yet.")
        mode = self.ours_store_mode
        if mode == "state_dict" and _is_peft_model(self.model):
            # —— 直接在同一 PeftModel 上切换适配器进行两次前向 ——
            base = unwrap_model(self.model)  # PeftModel
            current_adapters = getattr(base, "active_adapter", None)  # 训练时的适配器名
            try:
                base.eval()
                from torch.amp import autocast
                with torch.no_grad():
                    was_gc = getattr(self.args, "gradient_checkpointing", False)
                    if was_gc and hasattr(self.model, "gradient_checkpointing_disable"):
                        self.model.gradient_checkpointing_disable()
                    try:
                        with torch.autocast("cuda", enabled=False):
                            base.set_adapter(self._dualref_ma)
                            loss_a = self.compute_loss(base, inputs, return_per_token_loss=True)
                        with torch.autocast("cuda", enabled=False):
                            base.set_adapter(self._dualref_mb)
                            loss_b = self.compute_loss(base, inputs, return_per_token_loss=True)
                    finally:
                        if was_gc and hasattr(self.model, "gradient_checkpointing_enable"):
                            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=self.args.gradient_checkpointing_kwargs)
                  
            finally:
                # 切回原训练适配器
                if current_adapters is not None:
                    base.set_adapter(current_adapters)
                base.train()  # 训练继续

            return loss_a.detach(), loss_b.detach()
        elif mode == "state_dict":
            # 复用一份临时 module 来承载 state_dict 做两次前向，避免反复改动训练模型
            if not hasattr(self, "_dualref_temp_eval_model"):
                self._dualref_temp_eval_model = self._dualref_clone_module(self.model)
            temp = self._dualref_temp_eval_model
            # self._dualref_load_state_dict(temp, self._dualref_ma)
            self._load_lora_state_dict(temp, self._dualref_ma)
            loss_a = self.compute_loss(temp, inputs, return_per_token_loss=True)
            # self._dualref_load_state_dict(temp, self._dualref_mb)
            self._load_lora_state_dict(temp, self._dualref_mb)
            loss_b = self.compute_loss(temp, inputs, return_per_token_loss=True)
            return loss_a.detach(), loss_b.detach()
        else:
            # 直接用两份冻结 module
          
            loss_a = self.compute_loss(self._dualref_ma, inputs, return_per_token_loss=True)
            loss_b = self.compute_loss(self._dualref_mb, inputs, return_per_token_loss=True)
            # print("loss equal",loss_a==loss_b)
            return loss_a.detach(), loss_b.detach()
    


    def _dualref_handle_anchor_window(self, model, batch_samples, args, steps_in_epoch, steps_skipped, epoch,step,
                                  tr_loss, grad_norm, trial, ignore_keys_for_eval, start_time,num_items_in_batch,
                                rand_batch_samples, rand_num_items_in_batch ):
        """
        对一个 anchor 累积窗口执行“两遍法”：
        1) 预演第一遍（标准 loss）→ 得到 mi' → 固化 ma/mb → 回滚
        2) 正式第二遍（新方法 loss）→ 真正 optimizer.step() 产生 m_{i+1}
        返回更新后的 (tr_loss, grad_norm) 以及 learning_rate。
        """
        # ========= 预演第一遍：标准 loss，生成 mi' =========
        prev_flag = getattr(self, "_dualref_suppress_new_loss", False)
        self._dualref_suppress_new_loss = True  # 预演阶段强制关闭新方法


        if self.pre_ga < 0:
            # 和正常路径一致的 no_sync 策略
            for i, inputs in enumerate(rand_batch_samples):
                do_sync_step = (i == len(rand_batch_samples) - 1)
                self.accelerator.gradient_state._set_sync_gradients(do_sync_step)
                context = (
                    contextlib.nullcontext()
                    if (i == len(rand_batch_samples) - 1 or self.accelerator.distributed_type == DistributedType.DEEPSPEED)
                    else contextlib.nullcontext if not hasattr(self.accelerator, "no_sync") else
                        contextlib.nullcontext() if model is None else
                        contextlib.nullcontext()  # 保守起见：不依赖 no_sync 包裹器，因我们只需要正确的聚合梯度
                )
                
                with context:
                    _ = self.training_step(model, inputs,rand_num_items_in_batch)  # 标准 loss 反传（已按 GA 归一化）
        else:
            pre_ga = self.pre_ga
            with temporary_ga(self, pre_ga):
                for i, inputs in enumerate(rand_batch_samples):
                    do_sync_step = (i == len(rand_batch_samples) - 1)
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)
                    # 预演只需要正确累计梯度与归一化，不需要 no_sync 花样
                    # print(i)
                    _ = self.training_step(model, inputs, rand_num_items_in_batch)


        # 梯度裁剪（与原始逻辑一致）
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            _grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            grad_norm = _grad_norm

        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

        # ---- 快照 mi/优化器/调度器，做一次真实 step → 得到 mi' ----
        # mi_state = self._dualref_get_state_dict(model)
        mi_state = self._get_lora_state_dict(model)
        opt_state = copy.deepcopy(self.optimizer.state_dict())
        sch_state = copy.deepcopy(self.lr_scheduler.state_dict()) if self.lr_scheduler is not None else None
      
        
        new_lr = self.epsilon
        with temporary_lr(self.optimizer, lr=new_lr):
            self.optimizer.step()

        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step()
        model.zero_grad()

        # mb_state = self._dualref_get_state_dict(model)
        mb_state = self._get_lora_state_dict(model)
        # 固化 refs
        self._dualref_build_frozen_refs(mi_state, mb_state)

        # 回滚到 mi
        # self._dualref_load_state_dict(model, mi_state)
        self._load_lora_state_dict(model, mi_state)
        self.optimizer.load_state_dict(opt_state)
        if sch_state is not None:
            self.lr_scheduler.load_state_dict(sch_state)
        #回滚后清理这些state
        del mi_state, mb_state, opt_state, sch_state

        if hasattr(model, "train"):
            model.train()
        # 清干净预演梯度
        model.zero_grad()

        # ========= 正式第二遍：新方法 loss，产生真正更新 =========
        self._dualref_suppress_new_loss = False  # 正式阶段允许新方法

        # on_step_begin：只对“正式第二遍”的第一个 microbatch 触发一次（与原逻辑对齐）
        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

        for i, inputs in enumerate(batch_samples):
            step+=1
            do_sync_step = (i == len(batch_samples) - 1)
            # Manually set sync flag
            self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

            # 参考原逻辑的 no_sync 处理（Deepspeed 不用）
            context = (
                contextlib.nullcontext()
                if (i == len(batch_samples) - 1 or self.accelerator.distributed_type == DistributedType.DEEPSPEED)
                else contextlib.nullcontext if not hasattr(self.accelerator, "no_sync") else
                    contextlib.nullcontext()
            )
            with context:
                tr_loss_step = self.training_step(model, inputs,num_items_in_batch)
            # 统计 loss：仅统计“正式第二遍”的
            if tr_loss.device != tr_loss_step.device:
                raise ValueError(f"Loss device mismatch: {tr_loss.device} vs {tr_loss_step.device}")
            tr_loss = tr_loss + tr_loss_step

            if not do_sync_step:
                self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

        # 同步步（最后一个 microbatch）后的梯度裁剪
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            _grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            grad_norm = _grad_norm

        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

        # 真正的优化步（产生 m_{i+1}）
        self.optimizer.step()
        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)
        learning_rate = self._get_learning_rate()
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step()
        model.zero_grad()

        # 推进计数 & 回调 & 日志（与原逻辑一致）
        self.state.global_step += 1
        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
        self._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
        )

        # 复原预演标志位
        self._dualref_suppress_new_loss = prev_flag
        return tr_loss, grad_norm, learning_rate,step
    
    def get_last_batch(self, epoch_dataloader, device):
        """
        返回当前 DataLoader 定义下的“最后一个 batch”（不消耗你传进来的 epoch_iterator）。
        - 优先：索引拼装（dataset 可 __len__ + __getitem__ 时，O(batch_size)）
        - 兜底：独立迭代到尾（不影响主循环在用的 epoch_iterator）
        返回:
            batches: List[Batch]，长度恒为 1（若无法取出则为空列表）
            num_items_in_batch: torch.LongTensor 或 None（在 device 上）
        """
        import torch

        dl = epoch_dataloader
        dataset = getattr(dl, "dataset", None)
        collate_fn = getattr(dl, "collate_fn", None)
        drop_last = getattr(dl, "drop_last", False)

        def _count_supervised_tokens(batch):
            total, saw_any = 0, False
            if isinstance(batch, dict):
                if "loss_mask" in batch:
                    m = batch["loss_mask"]
                    m = m if torch.is_tensor(m) else torch.as_tensor(m)
                    total += int(m.long().sum().item()); saw_any = True
                elif "labels" in batch:
                    lab = batch["labels"]
                    lab = lab if torch.is_tensor(lab) else torch.as_tensor(lab)
                    total += int(lab.ne(-100).long().sum().item()); saw_any = True
                elif "attention_mask" in batch:
                    m = batch["attention_mask"]
                    m = m if torch.is_tensor(m) else torch.as_tensor(m)
                    total += int(m.long().sum().item()); saw_any = True
            return torch.as_tensor(total, device=device, dtype=torch.long) if saw_any else None

        # ====== 路径A：索引拼装（不影响外部迭代器） ======
        if (dataset is not None) and hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__") and (collate_fn is not None):
            n = len(dataset)
            if n == 0:
                print("[LAST] dataset empty -> return []")
                return [], None

            bs = my_get_dataloader_batch_size(self, dl)  # 复用你现有的取 batch_size 的方法
            if bs is None or bs <= 0:
                # 兜底防御
                bs = getattr(self.args, "per_device_train_batch_size", 1)

            if drop_last:
                if n < bs:
                    # drop_last=True 且样本不足一个完整 batch，按 DataLoader 语义“最后一个 batch 不存在”
                    print(f"[LAST] drop_last=True, n={n} < bs={bs} -> no last batch")
                    return [], None
                # 最后一个“完整”批次的起点
                start = n - (n % bs or bs)
            else:
                r = n % bs
                start = n - (r if r != 0 else bs)

            idxs = list(range(start, n))
            items = [dataset[i] for i in idxs]
            batch = collate_fn(items)
            num_items_in_batch = _count_supervised_tokens(batch)

            print(f"[LAST][index] n={n}, bs={bs}, drop_last={drop_last}, "
                f"start={start}, end={n}, id={id(batch)}")

            return [batch], num_items_in_batch

        # ====== 路径B：独立迭代到尾（不影响外部 epoch_iterator） ======
        try:
            it = iter(dl)   # 新建一个临时迭代器
        except Exception as e:
            print(f"[LAST] cannot create iterator: {e}")
            return [], None

        last = None
        seen = 0
        try:
            while True:
                last = next(it)
                seen += 1
        except StopIteration:
            pass

        if last is None:
            print("[LAST][iter] dataloader produced no batch -> []")
            return [], None

        num_items_in_batch = _count_supervised_tokens(last)
        print(f"[LAST][iter] seen_batches={seen}, id={id(last)}")
        return [last], num_items_in_batch

    
  
    
    def get_next_batch_samples(self, epoch_iterator, epoch_dataloader, num_batches, device):
        """
        消耗+推回(pushback)版（带逐项对齐打印）：
        - 预取接下来 num_batches 个 batch 作为 B′，推回缓冲区；
        - 返回只包一层的迭代器，使主循环先吐回这 num_batches 个 batch；
        - 统计监督 token 数；
        - 打印：
            [PROBE] 列出这 num_batches 个预取 id
            [MAIN ] 当主循环消费这 num_batches 个被推回的 batch 时，打印与 PROBE 一一对应的 (probe_seq, idx, id)
        """
        import torch
        from collections import deque

        # —— 轻量单层包装器 —— #
        class _PushbackIter:
            __slots__ = ("_it", "_dl", "_buf", "_dbg", "_main_seq", "_rank")
            _is_pushback_wrap = True

            def __init__(self, base_it, dl, buf, dbg, rank):
                self._it = base_it
                self._dl = dl
                self._buf = buf      # 被推回的 batch
                self._dbg = dbg      # 与 _buf 同步的 (probe_seq, idx, id) 队列
                self._main_seq = 0
                self._rank = rank

            def __iter__(self):
                return self

            def __next__(self):
                if self._buf:
                    item = self._buf.popleft()
                    pseq, pidx, pid = self._dbg.popleft()
                    self._main_seq += 1
                    print(f"[MAIN][rank={self._rank}] batch_seq={self._main_seq}  <- prefetch(probe_seq={pseq}, idx={pidx}, id={pid})")
                    return item
                try:
                    item = next(self._it)
                except StopIteration:
                    self._it = iter(self._dl)
                    item = next(self._it)
                self._main_seq += 1
                print(f"[MAIN][rank={self._rank}] batch_seq={self._main_seq}, id={id(item)}")
                return item

        # ===== 复用/创建单层包装器 =====
        rank = getattr(getattr(self, "accelerator", None), "process_index", 0)
        if getattr(epoch_iterator, "_is_pushback_wrap", False):
            wrapper = epoch_iterator
            base_it = wrapper._it
            buf = wrapper._buf
            dbg = wrapper._dbg
        else:
            base_it = epoch_iterator
            buf = deque()
            dbg = deque()
            wrapper = _PushbackIter(base_it, epoch_dataloader, buf, dbg, rank)

        # ===== 预取 num_batches，并推回（与 debug 信息同步）=====
        if not hasattr(self, "_probe_seq"):
            self._probe_seq = 0
        self._probe_seq += 1
        pseq = self._probe_seq

        batches = []
        need = num_batches
        did_wrap = False
        while need > 0:
            try:
                b = next(base_it)
            except StopIteration:
                base_it = iter(epoch_dataloader)
                b = next(base_it)
                did_wrap = True
            batches.append(b)
            buf.append(b)  # 推回
            dbg.append((pseq, len(batches), id(b)))  # 记录预取序号、在窗口内的索引和对象 id
            need -= 1

        # 推进后的底层迭代器指针写回包装器
        wrapper._it = base_it

        # ===== 统计监督 token =====
        total = 0
        saw_any = False
        for x in batches:
            if isinstance(x, dict):
                if "loss_mask" in x:
                    m = x["loss_mask"]; m = m if torch.is_tensor(m) else torch.as_tensor(m)
                    total += int(m.long().sum().item()); saw_any = True; continue
                if "labels" in x:
                    lab = x["labels"]; lab = lab if torch.is_tensor(lab) else torch.as_tensor(lab)
                    total += int(lab.ne(-100).long().sum().item()); saw_any = True; continue
                if "attention_mask" in x:
                    m = x["attention_mask"]; m = m if torch.is_tensor(m) else torch.as_tensor(m)
                    total += int(m.long().sum().item()); saw_any = True; continue
        num_items_in_batch = torch.as_tensor(total, device=device, dtype=torch.long) if saw_any else None

        # ===== PROBE 打印：列出这 num_batches 个 id =====
        ids = [id(b) for b in batches]
        print(f"[PROBE][rank={rank}] probe_seq={pseq}, next_{num_batches}_ids={ids}, wrap={'Y' if did_wrap else 'N'}")

        return batches, num_items_in_batch, wrapper




    
    # def get_random_batch_samples(self, base_dataloader, num_batches, device):
    #     """
    #     直接对 base_dataloader.dataset 随机采样索引，调用 base_dataloader.collate_fn 手工拼 batch。
    #     不创建新的 DataLoader，不启动 worker，避免类型不匹配。
    #     """
    #     dataset = base_dataloader.dataset
    #     collate_fn = getattr(base_dataloader, "collate_fn", None)
    #     assert collate_fn is not None, "base_dataloader.collate_fn 不能为空"

    #     # 需要能取 len(dataset)
    #     if not hasattr(dataset, "__len__"):
    #         # IterableDataset 的情况：无法随机索引，退化为返回空（你也可以选择复制当前 batch）
    #         return [], None

    #     batch_size = my_get_dataloader_batch_size(self, base_dataloader)

    #     # 为了每个 rank 有不同随机性（分布式）
    #     try:
    #         process_idx = self.accelerator.process_index
    #     except Exception:
    #         process_idx = 0

    #     rand_batches = []
    #     for b in range(num_batches):
    #         # 你也可以用 torch.randint；这里用 Python random 更简单
    #         random.seed((self.state.epoch or 0) * 10_000_003 + b * 97 + process_idx * 131)
    #         idxs = [random.randrange(len(dataset)) for _ in range(batch_size)]
    #         items = [dataset[i] for i in idxs]
    #         batch = collate_fn(items)  # 关键：复用原 collate_fn，得到 dict/tensor 结构
    #         rand_batches.append(batch)

    #     # 统计 token 数（与原 get_batch_samples 保持一致风格，简化版）
    #     num_items_in_batch = None
    #     if len(rand_batches) > 0 and isinstance(rand_batches[0], dict) and "labels" in rand_batches[0]:
    #         try:
    #             # rand_batches 是“窗口内的多个 micro-batch”
    #             total = 0
    #             for rb in rand_batches:
    #                 labels = rb["labels"]
    #                 if torch.is_tensor(labels):
    #                     total += (labels.ne(-100)).sum()
    #                 else:
    #                     # 万一 labels 是 list/np/tensor 混合
    #                     total += torch.as_tensor(labels).ne(-100).sum()
    #             num_items_in_batch = total.to(device) if torch.is_tensor(total) else torch.as_tensor(total, device=device)
    #         except Exception:
    #             pass

    #     return rand_batches, num_items_in_batch


    def get_random_batch_samples(self, base_dataloader, num_batches, device, batch_size_override=None):
        dataset = base_dataloader.dataset
        collate_fn = getattr(base_dataloader, "collate_fn", None)
        assert collate_fn is not None, "base_dataloader.collate_fn 不能为空"
        if not hasattr(dataset, "__len__"):
            return [], None

        base_bs = my_get_dataloader_batch_size(self, base_dataloader)
        bs = int(batch_size_override) if batch_size_override is not None else base_bs

        try:
            process_idx = self.accelerator.process_index
        except Exception:
            process_idx = 0

        rand_batches = []
        for b in range(num_batches):
            random.seed((self.state.epoch or 0) * 10_000_003 + b * 97 + process_idx * 131)
            #random.seed(42)
            idxs = [random.randrange(len(dataset)) for _ in range(bs)]
            items = [dataset[i] for i in idxs]
            batch = collate_fn(items)
            rand_batches.append(batch)

        num_items_in_batch = None
        if len(rand_batches) > 0 and isinstance(rand_batches[0], dict) and "labels" in rand_batches[0]:
            try:
                total = 0
                for rb in rand_batches:
                    labels = rb["labels"]
                    total += (labels if torch.is_tensor(labels) else torch.as_tensor(labels)).ne(-100).sum()
                num_items_in_batch = total.to(device) if torch.is_tensor(total) else torch.as_tensor(total, device=device)
            except Exception:
                pass

        return rand_batches, num_items_in_batch

    
    def _get_lora_state_dict(self, model):
        """
        返回只包含 LoRA/可训练参数的 state_dict（CPU 上，已 detach/clone）。
        - 若是 PEFT：使用 get_peft_model_state_dict（会处理 FSDP 聚合等细节）
        - 否则：返回所有 requires_grad=True 的参数
        """
        base = unwrap_model(model)
        try:
            from peft import get_peft_model_state_dict
            if _is_peft_model(model):
                sd = get_peft_model_state_dict(base)
                return {k: v.detach().clone().cpu() for k, v in sd.items()}
        except Exception:
            pass

        # 兜底：只抓可训练参数（通常就是 LoRA + modules_to_save）
        sd = {}
        for n, p in base.named_parameters():
            if p.requires_grad:
                sd[n] = p.detach().clone().cpu()
        return sd

    def _load_lora_state_dict(self, model, lora_sd):
        """
        仅写回 LoRA/可训练参数：
        - 若是 PEFT：set_peft_model_state_dict（只改 adapter）
        - 否则：strict=False（因为是子集）
        """
        base = unwrap_model(model)
        try:
            from peft import set_peft_model_state_dict
            if _is_peft_model(model):
                set_peft_model_state_dict(base, lora_sd)
                return
        except Exception:
            pass
        base.load_state_dict(lora_sd, strict=False)  # 子集加载

@contextmanager
def temporary_ga(self, ga: int):
    old = self.args.gradient_accumulation_steps
    try:
        self.args.gradient_accumulation_steps = int(ga)
        yield
    finally:
        self.args.gradient_accumulation_steps = old
    

@contextmanager
def temporary_lr(optimizer, *, lr=None, scale=None):
    """
    在 optimizer.step() 之前临时调整学习率：
      - 指定 lr=... 直接覆盖
      - 或指定 scale=... 按比例缩放当前 lr
    用法：
      with temporary_lr(optimizer, lr=1e-5):
          optimizer.step()
    """
    if (lr is None) == (scale is None):
        raise ValueError("Specify exactly one of `lr` or `scale`.")

    old_lrs = [pg["lr"] for pg in optimizer.param_groups]
    try:
        for pg, old in zip(optimizer.param_groups, old_lrs):
            pg["lr"] = (old * scale) if scale is not None else lr
        yield
    finally:
        for pg, old in zip(optimizer.param_groups, old_lrs):
            pg["lr"] = old


def my_get_dataloader_batch_size(self, dataloader):
    # 尽量复用原 dataloader 的 batch_size；有些 DataLoader 用的是 batch_sampler
    bs = getattr(dataloader, "batch_size", None)
    if bs is not None:
        return bs
    bs = getattr(getattr(dataloader, "batch_sampler", None), "batch_size", None)
    if bs is not None:
        return bs
    # 兜底：从已经取到的一个 batch 推断（不推荐在这里做）；退而求其次用 args.per_device_train_batch_size
    return getattr(self.args, "per_device_train_batch_size", 1)


# ====== 新增：LoRA-only 抓/载工具（优先 PEFT） ======

def _is_peft_model(model):
    try:
        from peft import PeftModel
        return isinstance(unwrap_model(model), PeftModel)
    except Exception:
        return False


# ===== 手写的版本无关工具：仅让 LoRA(及 modules_to_save) 可训练 =====
def _mark_only_lora_as_trainable(base_model):
    """
    关闭全部参数，再开启 LoRA 相关 & modules_to_save 的参数。
    适配不同 PEFT 版本的命名：lora_A/lora_B/lora_embedding_A/lora_embedding_B/modules_to_save
    """
    # 先全关
    for n, p in base_model.named_parameters():
        p.requires_grad_(False)

    # 再按名字开启 LoRA/保存模块参数
    # 如需兼容 DoRA/AdaLoRA，可在 patterns 里追加关键字
    patterns = (
        "lora_A.", "lora_B.",          # 线性 LoRA
        "lora_embedding_A", "lora_embedding_B",  # embedding LoRA
        ".modules_to_save."            # 额外保存的模块
    )
    for n, p in base_model.named_parameters():
        if any(k in n for k in patterns):
            p.requires_grad_(True)

