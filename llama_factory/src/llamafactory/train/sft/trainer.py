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
from safetensors.torch import save_file, load_file

##############################new package added for _inner_training_loop##############################
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
######################################new added##############################

# Name of the files used for checkpointing
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"


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
        use_vcore: bool = False,
        main_or_branch: str = "main",
        vcore_pre_ga: int = 8,
        vcore_temperature: float = 1.0,
        vcore_anchor_steps: int = 1,
        vcore_epsilon: float = 2e-5,
        vcore_single_process: bool = False,
        output_dir: str = "./output",
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
        self.pre_ga = vcore_pre_ga
        self.use_vcore = use_vcore
        self.main_or_branch = main_or_branch
        self.vcore_temperature = vcore_temperature
        self.vcore_anchor_steps = vcore_anchor_steps
        self.vcore_epsilon = vcore_epsilon
        self.vcore_single_process = vcore_single_process
        self.output_dir = output_dir+"/temp" # temply store the ckpt for main-branch training

        if self.vcore_single_process:
            logger.info_rank0("Trainer in single-process vcore mode (in-memory branch/main sync).")
        elif self.main_or_branch == 'main':
            logger.info_rank0(f"Trainer in main process.")
        elif self.main_or_branch == 'branch':
            logger.info_rank0(f"Trainer in branch process.")
        else:
            raise ValueError(f"main_or_branch should be 'main' or 'branch', but got {self.main_or_branch}.")

   
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
        
        return super().compute_loss(model, inputs,self.use_dft, self.use_vcore,self.vcore_temperature, *args, **kwargs)

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


        # ======make only lora weights engage into the training======
        base = unwrap_model(model)
        if _is_peft_model(model):
            _mark_only_lora_as_trainable(base)
            try:
                base.print_trainable_parameters()
            except Exception:
                pass
        else:
            # not peft
            _mark_only_lora_as_trainable(base)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        # use_accelerator_prepare = True if model is self.model else False
        use_accelerator_prepare = False

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
            
           
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                
                # lr minimum is 1e-6
                for group in self.optimizer.param_groups:
                    group["lr"] = max(1e-6, group["lr"])


               # >>>>>>> Add Anchor Window <<<<<<<
                is_anchor_window = False
                if self.use_vcore:
                    # probing only at anchor steps
                    is_anchor_window = (update_step%self.vcore_anchor_steps==0)

                if is_anchor_window:
                    if self.pre_ga<0:
                        pre_ga = args.gradient_accumulation_steps
                    else:
                        pre_ga = self.pre_ga  #gradient accumulation steps for the branch process for probing, default to the same as args.gradient_accumulation_steps
                    
                    if self.vcore_single_process or self.main_or_branch == 'branch':
                        rand_batch_samples, rand_num_items_in_batch = self.get_random_batch_samples(
                            epoch_dataloader, pre_ga, args.device
                        )
                    else:
                        rand_batch_samples, rand_num_items_in_batch = None, None
                  
                    # paralell process，use batch B‘ to probing for branch process
                    tr_loss, grad_norm, learning_rate,step= self._run_anchor_window(
                        model, batch_samples, args, steps_in_epoch, steps_skipped, epoch,step,
                        tr_loss, grad_norm, trial, ignore_keys_for_eval, start_time,num_items_in_batch, 
                        rand_batch_samples, rand_num_items_in_batch 
                    )
                    # Increment step here as the inner loop is skipped.
                    # The current update_step is now complete (including logging, callbacks, and step+1). Proceed to the next window.
                    continue



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

            # clear the temp file for main-branch training after each epoch
            if self.vcore_single_process==False and self.main_or_branch=='main':
                shutil.rmtree(self.output_dir, ignore_errors=True)
                
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
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None,pre_loss=None
    ) -> torch.Tensor:

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

      
        with self.compute_loss_context_manager():
            if pre_loss is not None:
                loss = self.compute_loss(model, inputs,num_items_in_batch=num_items_in_batch,pre_loss=pre_loss,probing_lr=self.vcore_epsilon)
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
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
            self.accelerator.backward(loss, **kwargs)
            return loss.detach()


    def _run_anchor_window(self, model, batch_samples, args, steps_in_epoch, steps_skipped, epoch,step,
                                  tr_loss, grad_norm, trial, ignore_keys_for_eval, start_time,num_items_in_batch,
                                rand_batch_samples, rand_num_items_in_batch ):
        """
        for anchor step, go into main or branch process:
        1) main forward wait for  branch loss
        2) branch B' to get pre_loss
        return (tr_loss, grad_norm) and learning_rate
        """
        if self.vcore_single_process:
            return self._run_anchor_window_single_process(
                model,
                batch_samples,
                args,
                steps_in_epoch,
                steps_skipped,
                epoch,
                step,
                tr_loss,
                grad_norm,
                trial,
                ignore_keys_for_eval,
                start_time,
                num_items_in_batch,
                rand_batch_samples,
                rand_num_items_in_batch,
            )

        if self.main_or_branch=='main':
            if self.is_world_process_zero():
                os.makedirs(self.output_dir+f'/main_tmp_{self.state.global_step}', exist_ok=True)
                self._get_lora_state_dict(model, self.output_dir+f'/main_tmp_{self.state.global_step}/lora.safetensors')
                
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        if self.main_or_branch=='branch':
            while(1):
                if os.path.exists(self.output_dir+f'/main_tmp_{self.state.global_step}/lora.safetensors'):
                    if dist.is_initialized():
                         dist.barrier()
                    self._load_lora_state_dict(model, self.output_dir+f'/main_tmp_{self.state.global_step}/lora.safetensors')
                    break
                else:
                    time.sleep(0.01) #wait for 0.01s to avoid busy waiting
            if self.pre_ga < 0:
                #  no_sync
                for i, inputs in enumerate(rand_batch_samples):
                    do_sync_step = (i == len(rand_batch_samples) - 1)
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)
                    context = (
                        contextlib.nullcontext()
                        if (i == len(rand_batch_samples) - 1 or self.accelerator.distributed_type == DistributedType.DEEPSPEED)
                        else contextlib.nullcontext if not hasattr(self.accelerator, "no_sync") else
                            contextlib.nullcontext() if model is None else
                            contextlib.nullcontext() 
                    )
                    
                    with context:
                        _ = self.training_step(model, inputs,rand_num_items_in_batch)  #  loss backward
            else:
                pre_ga = self.pre_ga
                with temporary_ga(self, pre_ga):
                    for i, inputs in enumerate(rand_batch_samples):
                        do_sync_step = (i == len(rand_batch_samples) - 1)
                        self.accelerator.gradient_state._set_sync_gradients(do_sync_step)
                        _ = self.training_step(model, inputs, rand_num_items_in_batch)

            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                _grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_norm = _grad_norm

            self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
            
            
            # update the weights in branch process
            eps = float(self.vcore_epsilon)
            base = unwrap_model(model)
            with torch.no_grad():
                for n, p in base.named_parameters():
                    if _is_lora_trainable_name(n) and p.grad is not None:
                        p.add_(p.grad, alpha=-eps)
            model.zero_grad(set_to_none=True)


        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

        for i, inputs in enumerate(batch_samples):
            step+=1

            do_sync_step = (i == len(batch_samples) - 1)
            # Manually set sync flag
            self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

            context = (
                contextlib.nullcontext()
                if (i == len(batch_samples) - 1 or self.accelerator.distributed_type == DistributedType.DEEPSPEED)
                else contextlib.nullcontext if not hasattr(self.accelerator, "no_sync") else
                    contextlib.nullcontext()
            )
            with context:
                if self.main_or_branch=='branch':
                    inputs = self._prepare_inputs(inputs)
                    with torch.no_grad():
                        rank = dist.get_rank() if dist.is_initialized() else 0

                        pre_loss = self.compute_loss(model, inputs, return_per_token_loss=True)
                        pre_loss = pre_loss.detach().cpu()

                        save_dir = f"{self.output_dir}/branch_tmp_{self.state.global_step}"
                        os.makedirs(save_dir, exist_ok=True)
                        
                        path = f"{save_dir}/pre_loss_rank{rank}_step{step}.pt"
                        tmp_path = path + ".tmp"
                        torch.save(
                            pre_loss,
                            tmp_path
                        )
                        os.rename(tmp_path, path) 

                        if dist.is_initialized():
                            dist.barrier()
                elif self.main_or_branch=='main':
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    path = f"{self.output_dir}/branch_tmp_{self.state.global_step}/pre_loss_rank{rank}_step{step}.pt"

                    while not os.path.exists(path):
                        time.sleep(0.01) #wait 0.01s

                    if dist.is_initialized():
                        dist.barrier()

                    pre_loss = torch.load(path, map_location="cpu")

                    tr_loss_step = self.training_step(model, inputs,num_items_in_batch,pre_loss)
            
            if self.main_or_branch=='main':
                # only main loss
                if tr_loss.device != tr_loss_step.device:
                    raise ValueError(f"Loss device mismatch: {tr_loss.device} vs {tr_loss_step.device}")
                tr_loss = tr_loss + tr_loss_step

                if not do_sync_step:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
        
        
        if self.main_or_branch=='branch':
            model.zero_grad()
            # count & callback & log
            self.state.global_step += 1
            self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
            return tr_loss, grad_norm, None,step
        
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            _grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            grad_norm = _grad_norm

        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

        # optimization step
        self.optimizer.step()
        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)
        learning_rate = self._get_learning_rate()
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step()
        model.zero_grad()

        self.state.global_step += 1
        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
        self._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
        )

        return tr_loss, grad_norm, learning_rate,step


    def _run_anchor_window_single_process(
        self,
        model,
        batch_samples,
        args,
        steps_in_epoch,
        steps_skipped,
        epoch,
        step,
        tr_loss,
        grad_norm,
        trial,
        ignore_keys_for_eval,
        start_time,
        num_items_in_batch,
        rand_batch_samples,
        rand_num_items_in_batch,
    ):
        """Single-process anchor window: branch probe in memory, then main forward/backward."""
        # 1) Backup current LoRA/trainable weights in memory
        lora_sd = self._get_lora_state_dict(model, save_path=None)

        # 2) Branch probe update on random batch B'
        if rand_batch_samples is not None and len(rand_batch_samples) > 0:
            if self.pre_ga < 0:
                for i, inputs in enumerate(rand_batch_samples):
                    do_sync_step = (i == len(rand_batch_samples) - 1)
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)
                    _ = self.training_step(model, inputs, rand_num_items_in_batch)
            else:
                pre_ga = self.pre_ga
                with temporary_ga(self, pre_ga):
                    for i, inputs in enumerate(rand_batch_samples):
                        do_sync_step = (i == len(rand_batch_samples) - 1)
                        self.accelerator.gradient_state._set_sync_gradients(do_sync_step)
                        _ = self.training_step(model, inputs, rand_num_items_in_batch)

            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                _grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_norm = _grad_norm

            with torch.no_grad():
                eps = float(self.vcore_epsilon)
                base = unwrap_model(model)
                for n, p in base.named_parameters():
                    if _is_lora_trainable_name(n) and p.grad is not None:
                        p.add_(p.grad, alpha=-eps)
            model.zero_grad(set_to_none=True)

        # 3) Compute per-token pre_loss on branch-updated weights (in memory)
        pre_losses = []
        for inputs in batch_samples:
            inputs_prepared = self._prepare_inputs(inputs)
            with torch.no_grad():
                pre_loss = self.compute_loss(model, inputs_prepared, return_per_token_loss=True)
            pre_losses.append(pre_loss.detach().cpu())

        # 4) Restore weights to the original main weights
        self._load_lora_state_dict(model, lora_sd=lora_sd)
        model.zero_grad(set_to_none=True)

        # 5) Main forward + backward with pre_loss
        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
        for i, (inputs, pre_loss) in enumerate(zip(batch_samples, pre_losses)):
            step += 1
            do_sync_step = (i == len(batch_samples) - 1)
            self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

            tr_loss_step = self.training_step(model, inputs, num_items_in_batch, pre_loss)

            if tr_loss.device != tr_loss_step.device:
                raise ValueError(f"Loss device mismatch: {tr_loss.device} vs {tr_loss_step.device}")
            tr_loss = tr_loss + tr_loss_step

            if not do_sync_step:
                self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            _grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            grad_norm = _grad_norm

        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
        self.optimizer.step()
        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)
        learning_rate = self._get_learning_rate()

        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step()

        model.zero_grad()
        self.state.global_step += 1
        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
        self._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
        )

        return tr_loss, grad_norm, learning_rate, step



    def get_random_batch_samples(self, base_dataloader, num_batches, device, batch_size_override=None):
        dataset = base_dataloader.dataset
        collate_fn = getattr(base_dataloader, "collate_fn", None)
        assert collate_fn is not None, "base_dataloader.collate_fn can not be None"
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


    def _get_lora_state_dict(self, model, save_path=None):
        """
        get LoRA state_dict(CPU), and save to local

        Args:
            model: current model
            save_path: if not None, then save to that path

        Returns:
            lora_sd: dict[str, Tensor](CPU)
        """
        base = unwrap_model(model)

        # -------- PEFT --------
        try:
            from peft import get_peft_model_state_dict
            if _is_peft_model(model):
                sd = get_peft_model_state_dict(base)
                lora_sd = {k: v.detach().clone().cpu() for k, v in sd.items()}
            else:
                raise RuntimeError
        except Exception:
            # -------- Not PEFT，only fetch requires_grad --------
            lora_sd = {}
            for n, p in base.named_parameters():
                if p.requires_grad:
                    lora_sd[n] = p.detach().clone().cpu()

        # -------- save to disk --------
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
            tmp_path = save_path + ".tmp"
            save_file(lora_sd, tmp_path) #safetensors format
            os.rename(tmp_path, save_path) 

        return lora_sd


    def _load_lora_state_dict(
        self,
        model,
        load_path=None,
        lora_sd=None,
    ):
        """
        Load LoRA from local dict

        Args:
            load_path: LoRA weights file path
            lora_sd: LoRA weights that already in memory
        """
        base = unwrap_model(model)

        if load_path is not None:
            lora_sd = load_file(load_path, device="cpu")
    

        # -------- PEFT --------
        try:
            from peft import set_peft_model_state_dict
            if _is_peft_model(model):
                set_peft_model_state_dict(base, lora_sd)
                return
        except Exception:
            pass

        # -------- Not PEFT --------
        base.load_state_dict(lora_sd, strict=False)


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
    change learning rate before optimizer.step():
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
    bs = getattr(dataloader, "batch_size", None)
    if bs is not None:
        return bs
    bs = getattr(getattr(dataloader, "batch_sampler", None), "batch_size", None)
    if bs is not None:
        return bs
    return getattr(self.args, "per_device_train_batch_size", 1)


def _is_peft_model(model):
    try:
        from peft import PeftModel
        return isinstance(unwrap_model(model), PeftModel)
    except Exception:
        return False


# ===== make only LoRA trainable =====
LORA_TRAINABLE_NAME_PATTERNS = (
    "lora_A.",
    "lora_B.",
    "lora_embedding_A",
    "lora_embedding_B",
    ".modules_to_save.",
)


def _is_lora_trainable_name(param_name: str) -> bool:
    return any(k in param_name for k in LORA_TRAINABLE_NAME_PATTERNS)


def _mark_only_lora_as_trainable(base_model):
    for n, p in base_model.named_parameters():
        p.requires_grad_(False)
    for n, p in base_model.named_parameters():
        if _is_lora_trainable_name(n):
            p.requires_grad_(True)

