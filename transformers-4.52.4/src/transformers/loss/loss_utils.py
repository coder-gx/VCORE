# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Optional
import math
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss

from .loss_d_fine import DFineForObjectDetectionLoss
from .loss_deformable_detr import DeformableDetrForObjectDetectionLoss, DeformableDetrForSegmentationLoss
from .loss_for_object_detection import ForObjectDetectionLoss, ForSegmentationLoss
from .loss_grounding_dino import GroundingDinoForObjectDetectionLoss
from .loss_rt_detr import RTDetrForObjectDetectionLoss


import torch
import torch.nn.functional as F


def dft_loss(source, target, ignore_index=-100, num_items_in_batch=None):
    """
    Based on "On the Generalization of SFT: A Reinforcement Learning Perspective 
    with Reward Rectification", this loss treats SFT as an RL process and applies 
    a rectifier to the implicit reward to prevent over-optimization on training data.
    args:
        source: [N, V] logits
        target: [N]    labels
        return: loss
    """
    assert target.dtype == torch.long, f"target dtype must be long, got {target.dtype}"
    N, V = source.shape

    # 1) per-token CE loss
    token_loss = F.cross_entropy(
        source, target,
        reduction="none",
        ignore_index=ignore_index
    )  # [N]

    # 2) loss mask
    with torch.no_grad():
        valid = (target != ignore_index)            # [N] bool
        safe_tgt = torch.where(valid, target, torch.zeros_like(target))  # [N]
        p_true = torch.softmax(source, dim=-1).gather(1, safe_tgt.unsqueeze(-1)).squeeze(-1)  # [N]
        weight = p_true * valid.float()         

    # 3) reweight and aggregate
    loss_vec = token_loss * weight                  # [N]

    if num_items_in_batch is not None:
        denom = float(num_items_in_batch)
        loss = loss_vec.sum() / denom
    else:
        valid_count = valid.float().sum().clamp_min(1.0)  
        loss = loss_vec.sum() / valid_count

    return loss


def vcore_loss(
    source: torch.Tensor,             # [B, T, V] logits
    target: torch.Tensor,             # [B, T]    labels
    ignore_index: int = -100,
    num_items_in_batch: Optional[int] = None,  # global count of valid tokens
    pre_loss: Optional[torch.Tensor] = None,   # pre_loss (ce loss before updated by population loss)
    probing_lr: Optional[float] = None,         # current learning rate
    return_per_token_loss: bool = False,
    temperature: float = 1.0,  
) -> torch.Tensor:
    """
    If return_per_token_loss is True:
        Returns the per-token Cross-Entropy (CE) loss for the current step 
        (Shape: [B, T], non-reduced, detached) to be passed as `pre_loss` in the next iteration.
    If False:
        1. Computes improvement scores: scores = (pre_loss - cur_loss) / temperature 
           (higher scores indicate greater loss reduction).
        2. Applies a masked softmax across the token dimension to derive weights.
        3. Weights the cur_loss by these scores and normalizes by the total 
           number of valid tokens to return a scalar loss.
    """
    assert source.dim() == 3 and target.dim() == 2, "source[B,T,V], target[B,T]"
    B, T, V = source.shape
    device = source.device

    #current loss [B, T]
    cur_loss_tok = F.cross_entropy(
        source.view(-1, V), target.view(-1),
        ignore_index=ignore_index, reduction="none"
    ).view(B, T)

    # token loss mask
    valid = (target != ignore_index)
    valid_f = valid.float()
    cur_loss = cur_loss_tok.detach() * valid_f  # [B, T]

    if return_per_token_loss:
        return cur_loss

    if pre_loss is not None:
        pre_loss = pre_loss.to(device)
        if cur_loss.shape != pre_loss.shape:
            raise ValueError(f"pre_loss shape {pre_loss.shape} != current loss shape {cur_loss.shape}")
        pre_loss = pre_loss.detach()
    
        # raw scores
        lr_eff = float(probing_lr) if probing_lr is not None else 1.0 # \epsilon
        denom_scale = lr_eff * float(temperature) #  \epsilon*\tau
        # s_t= (cur_loss - pre_loss) / lr_eff # s_t calculation in out paper
        scores_raw =  (cur_loss - pre_loss) / denom_scale  # [B, T]

        scores = scores_raw.masked_fill(~valid, float("-inf"))
        # stable softmax
        row_max = scores.max(dim=-1, keepdim=True).values
        scores_stable = scores - row_max
        exp_scores = torch.where(valid, scores_stable.exp(), torch.zeros_like(scores_stable))
        denom = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = exp_scores / denom  # [B, T]
        weights = weights.clamp_min(0.0)
        

     
        # Scale each sample's weights by its 'valid token count'. This ensures the global weight sum ≈ total valid tokens, neutralizing sample-length bias.
        valid_count_per_sample = valid_f.sum(dim=-1, keepdim=True).clamp_min(1)
        weights_token = weights * valid_count_per_sample  # [B,T]
        weighted_sum = (weights_token * cur_loss_tok * valid_f).sum()

        if num_items_in_batch is None:
            denom_tokens = valid_f.sum()
        else:
            if torch.is_tensor(num_items_in_batch):
                denom_tokens = num_items_in_batch.to(weighted_sum.device)
            else:
                denom_tokens = torch.tensor(num_items_in_batch, dtype=torch.long, device=weighted_sum.device)
        
        loss_weighted = weighted_sum / denom_tokens.clamp_min(1)

        # variance control loss
        # var_u = torch.var(s_t/valid_count_per_sample, unbiased=True)
        # var_q= torch.var(weights*s_t, unbiased=True)
        # c= (var_u.detach()/var_q.detach()).sqrt()
        # if c.item() >1.0:  
        #     loss= loss_weighted
        # else:
        #     loss= c* loss_weighted
       
        # Rescale to the uniform-CE baseline, but never amplify the weighted loss.
        uniform_sum = (cur_loss_tok * valid_f).sum()
        loss_uniform = uniform_sum / denom_tokens.clamp_min(1)
        c = (loss_uniform.detach() / loss_weighted.detach())
        if c.item() > 1.0:  
            loss= loss_weighted
        else:
            loss = c * loss_weighted
        
        return loss

       
    else:
        raise ValueError("pre_loss should not be None when calculating vcore loss")




def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    loss_mask: Optional[torch.Tensor]=None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    
    # loss mask for random_mask method
    if loss_mask is not None:
        for bs in range(loss_mask.shape[0]):
            
            target[bs]=target[bs][loss_mask[bs].bool()]
        
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    

    if reduction == "sum":
        loss = loss / num_items_in_batch
    
    
    return loss


   


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    use_dft:Optional[bool]=False,
    use_vcore:Optional[bool]=False,
    pre_loss:Optional[torch.Tensor]=None,
    probing_lr:Optional[float]=None,
    vcore_temperature:float=1.0,
    return_per_token_loss:bool=False,
    loss_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float().to(labels.device)
   

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    if not (use_vcore and (return_per_token_loss or (pre_loss is not None))):
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)

    # print(use_vcore, return_per_token_loss, pre_loss is not None)

    
    if use_dft:
        loss = dft_loss(logits, shift_labels, ignore_index=ignore_index, num_items_in_batch=num_items_in_batch)
    elif use_vcore and (return_per_token_loss or pre_loss is not None): 
        loss = vcore_loss(logits, shift_labels, ignore_index=ignore_index, num_items_in_batch=num_items_in_batch,pre_loss=pre_loss,probing_lr=probing_lr,temperature=vcore_temperature,return_per_token_loss=return_per_token_loss)
       
    else:
        loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch,loss_mask, ignore_index, **kwargs)
    
    return loss


def ForMaskedLMLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)
    # Enable model parallelism

    labels = labels.to(logits.device)
    loss = fixed_cross_entropy(logits, labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


def ForSequenceClassificationLoss(labels: torch.Tensor, pooled_logits: torch.Tensor, config, **kwargs) -> torch.Tensor:
    num_labels = config.num_labels
    if config.problem_type is None:
        if num_labels == 1:
            config.problem_type = "regression"
        elif num_labels > 1 and (labels.dtype in (torch.long, torch.int)):
            config.problem_type = "single_label_classification"
        else:
            config.problem_type = "multi_label_classification"

    labels = labels.to(pooled_logits.device)
    if config.problem_type == "regression":
        loss_fct = MSELoss()
        if num_labels == 1:
            return loss_fct(pooled_logits.squeeze(), labels.squeeze())
        else:
            return loss_fct(pooled_logits, labels)
    if config.problem_type == "single_label_classification":
        return fixed_cross_entropy(pooled_logits.view(-1, num_labels), labels.view(-1), **kwargs)

    if config.problem_type == "multi_label_classification":
        loss_fct = BCEWithLogitsLoss()
        return loss_fct(pooled_logits, labels)

    raise RuntimeError(f"Invalid problem type: {config.problem_type}")


def ForQuestionAnsweringLoss(start_logits, end_logits, start_positions, end_positions, **kwargs):
    total_loss = None
    if start_positions is not None and end_positions is not None:
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1).to(start_logits.device)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1).to(end_logits.device)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        start_loss = fixed_cross_entropy(start_logits, start_positions, ignore_index=ignored_index, **kwargs)
        end_loss = fixed_cross_entropy(end_logits, end_positions, ignore_index=ignored_index, **kwargs)
        total_loss = (start_loss + end_loss) / 2
    return total_loss


def ForTokenClassification(logits: torch.Tensor, labels, config, **kwargs):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.view(-1, config.num_labels)
    labels = labels.view(-1).to(logits.device)
    logits = logits.float()
    # Flatten the tokens
    return fixed_cross_entropy(logits, labels, **kwargs)


LOSS_MAPPING = {
    "ForCausalLM": ForCausalLMLoss,
    "ForMaskedLM": ForMaskedLMLoss,
    "ForQuestionAnswering": ForQuestionAnsweringLoss,
    "ForSequenceClassification": ForSequenceClassificationLoss,
    "ForImageClassification": ForSequenceClassificationLoss,
    "ForTokenClassification": ForTokenClassification,
    "ForSegmentation": ForSegmentationLoss,
    "ForObjectDetection": ForObjectDetectionLoss,
    "DeformableDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "ConditionalDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "DabDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "GroundingDinoForObjectDetection": GroundingDinoForObjectDetectionLoss,
    "ConditionalDetrForSegmentation": DeformableDetrForSegmentationLoss,
    "RTDetrForObjectDetection": RTDetrForObjectDetectionLoss,
    "RTDetrV2ForObjectDetection": RTDetrForObjectDetectionLoss,
    "DFineForObjectDetection": DFineForObjectDetectionLoss,
    "CsmForConditionalGeneration": ForCausalLMLoss,
}
