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
    source: [N, V] logits
    target: [N]    int64 labels，可能包含 ignore_index
    return: 标量 loss
    """
    assert target.dtype == torch.long, f"target dtype must be long, got {target.dtype}"
    N, V = source.shape

    # 1) per-token CE（忽略 ignore_index，不做规约）
    token_loss = F.cross_entropy(
        source, target,
        reduction="none",
        ignore_index=ignore_index
    )  # [N]

    # 2) 只在有效位置上计算“真实词概率”权重；无效位置权重=0
    with torch.no_grad():
        valid = (target != ignore_index)            # [N] bool
        safe_tgt = torch.where(valid, target, torch.zeros_like(target))  # [N], 把无效标签替换为0（任意合法类即可）
        p_true = torch.softmax(source, dim=-1).gather(1, safe_tgt.unsqueeze(-1)).squeeze(-1)  # [N]
        weight = p_true * valid.float()             # 无效位置清零

    # 3) 加权并规约成标量（按有效 token 数或指定分母做归一化）
    loss_vec = token_loss * weight                  # [N]

    if num_items_in_batch is not None:
        denom = float(num_items_in_batch)
        loss = loss_vec.sum() / denom
    else:
        valid_count = valid.float().sum().clamp_min(1.0)  # 避免除0
        loss = loss_vec.sum() / valid_count

    return loss

def ours_loss(
    source: torch.Tensor,             # [B, T, V] logits
    target: torch.Tensor,             # [B, T]    labels
    ignore_index: int = -100,
    num_items_in_batch: Optional[int] = None,  # 全局有效token数（多卡SUM），可为None
    loss_a: Optional[torch.Tensor] = None,   # loss_a
    loss_b: Optional[torch.Tensor] = None,   # loss_b
    cur_lr: Optional[float] = None,         # 当前学习率
    return_per_token_loss: bool = False, # 是否返回每个token的loss
    temperature: float = 1.0,   # 温度系数
) -> torch.Tensor:
    """
    当 return_per_token_loss:
        返回当前 step 的 per-token CE loss (形状 [B, T],无规约,detach)，
        供下一步作为 pre_loss 传回。
    当 pre_loss 非 None:
        计算 scores = (pre_loss - cur_loss) / temperature （下降越多分数越大），
        在 token 维度做 masked softmax 得到权重，对 cur_loss 加权，
        最终按有效 token 总数归一化得到标量 loss。
    """
    assert source.dim() == 3 and target.dim() == 2, "source[B,T,V], target[B,T]"
    B, T, V = source.shape
    device = source.device

    # 当前 step 每个 token 的 CE（不规约），形状 [B, T]
    cur_loss_tok = F.cross_entropy(
        source.view(-1, V), target.view(-1),
        ignore_index=ignore_index, reduction="none"
    ).view(B, T)

    # 有效 token mask
    valid = (target != ignore_index)
    valid_f = valid.float()

    if return_per_token_loss:
        # 无效 token 位置置 0，避免后续误用
        return (cur_loss_tok.detach() * valid_f)


    if loss_a is not None and loss_b is not None:
        # 对齐形状与设备，并阻断 pre_loss 的梯度
        loss_a = loss_a.to(device)
        loss_b = loss_b.to(device)
        if loss_a.shape != cur_loss_tok.shape:
            raise ValueError(f"pre_loss shape {loss_a.shape} != current loss shape {cur_loss_tok.shape}")
        loss_a = loss_a.detach()
        loss_b = loss_b.detach()
    
        # --- 计算 raw scores ---
        lr_eff = float(cur_lr) if cur_lr is not None else 1.0
        denom_scale = max(1e-8, lr_eff) * max(1e-8, float(temperature))
        # print(f"cur_lr:{cur_lr}, denom_scale:{denom_scale:.6e}")

        scores_raw = (loss_a - loss_b) / denom_scale  # [B, T]

        # masked softmax 之前把无效位设为 -inf
        scores = scores_raw.masked_fill(~valid, float("-inf"))


        # 稳定 softmax：对每个样本在 token 维度 softmax
        # 先把全为无效的样本保护一下（极少见），防止 -inf 全体导致 NaN
        # 做法：若一行全无效，则权重行全置 0
        row_has_valid = valid.any(dim=-1, keepdim=True)  # [B,1]

        # softmax（数值稳定）：减去行内最大
        scores_stable = scores.clone()
        row_max = torch.where(
            row_has_valid,
            scores_stable.max(dim=-1, keepdim=True).values,
            torch.zeros_like(scores_stable.max(dim=-1, keepdim=True).values)
        )
        scores_stable = torch.where(valid, scores_stable - row_max, scores_stable)

        weights = torch.zeros_like(cur_loss_tok)
        # 只对有有效 token 的样本做 softmax
        if row_has_valid.any():
            exp_scores = torch.where(valid, scores_stable.exp(), torch.zeros_like(scores_stable))
            denom = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            weights = exp_scores / denom  # [B, T]，每行对有效 token 归一化为1，其余为0
            weights = weights.clamp_min(0.0)

         
        # print(f"weights: max {weights.max().item():.4f}, min {weights.min().item():.4f}, mean {weights.float().mean().item():.4f} over {int((weights > 0).sum().item())} tokens")
        
       
        valid_count_per_sample = valid_f.sum(dim=-1, keepdim=True).clamp_min(1)
        # 让每个样本内的权重“乘上该样本的有效 token 数”，使全局权重之和≈全局有效 token 数
        weights_token = weights * valid_count_per_sample  # [B,T]

        weighted_sum = (weights_token * cur_loss_tok * valid_f).sum()


        # 归一化分母：优先用传入的 num_items_in_batch（建议传“多卡有效token全局和”）
        if num_items_in_batch is None:
            denom_tokens = valid_f.sum()  # 本卡有效token数
        else:
            # 允许传 python int 或张量
            if torch.is_tensor(num_items_in_batch):
                denom_tokens = num_items_in_batch.to(weighted_sum.device)
            else:
                denom_tokens = torch.tensor(num_items_in_batch, dtype=torch.long, device=weighted_sum.device)
        
      
        
        loss_weighted = weighted_sum / denom_tokens.clamp_min(1)

        # 未加权的平均 CE
        uniform_sum = (cur_loss_tok * valid_f).sum()
        loss_uniform = uniform_sum / denom_tokens.clamp_min(1)

        # 用一个常数 c 把标量对齐到旧尺度；
        c = (loss_uniform.detach() / loss_weighted.detach().clamp_min(1e-12))
    
        if c.item() > 1.0:  
            loss=loss_weighted
        else:
            loss = c * loss_weighted
        return loss

       
    else:
        raise ValueError("loss_a and loss_b should not be None when calculating ours loss")


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    deltas: Optional[torch.Tensor]=None,
    loss_mask: Optional[torch.Tensor]=None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    
        
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    

    if reduction == "sum":
        loss = loss / num_items_in_batch
    
    # print(loss.item())
    
    return loss


   


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    use_dft:Optional[bool]=False,
    use_ours:Optional[bool]=False,
    loss_a:Optional[torch.Tensor]=None,
    loss_b:Optional[torch.Tensor]=None,
    cur_lr:Optional[float]=None,
    ours_temperature:float=1.0,
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
    if not (use_ours and (return_per_token_loss or (loss_a is not None and loss_b is not None))):
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)

    
    if use_dft:
        loss = dft_loss(logits, shift_labels, ignore_index=ignore_index, num_items_in_batch=num_items_in_batch)
    elif use_ours and (return_per_token_loss or (loss_a is not None and loss_b is not None)): 
        #此时token没有被摊平
        
        if loss_mask is None:
           
            loss = ours_loss(logits, shift_labels, ignore_index=ignore_index, num_items_in_batch=num_items_in_batch,loss_a=loss_a,loss_b=loss_b,cur_lr=cur_lr,temperature=ours_temperature,return_per_token_loss=return_per_token_loss)
        else:
          
            loss = ours_loss_sample(logits, shift_labels, ignore_index=ignore_index, num_items_in_batch=num_items_in_batch,loss_a=loss_a,loss_b=loss_b,cur_lr=cur_lr,temperature=ours_temperature,return_per_token_loss=return_per_token_loss,loss_mask=loss_mask)
       
    else:
        loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch,loss_mask, ignore_index, **kwargs)
    
    # print("loss final:", loss.item())

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
