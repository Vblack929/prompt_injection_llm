"""
Custom loss functions for preference optimization
(DPO, IPO, TDPO, BDPO, SimPO, RePO, SimPER, etc.) + utilities
Refactored to remove duplication and improve maintainability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict
from transformers.trainer_callback import TrainerCallback
import copy

# -------------------------
# Core Utilities
# -------------------------

def get_batch_loss(output, labels):
    """
    Token-level cross-entropy (no reduction) with teacher-forcing shift.
    output: logits [B, T, V], labels [B, T]
    """
    shifted_labels = labels[..., 1:]
    output = output[..., :-1, :]
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_function(output.transpose(-1, -2), shifted_labels)
    return loss  # [B, T-1]

def get_sequence_log_probs(logits, labels, ignore_index=-100):
    """
    Sum of log-probs over non-masked positions (per sequence)
    logits [B, T, V], labels [B, T]
    """
    shifted_labels = labels[..., 1:]
    logits = logits[..., :-1, :]
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    log_probs = -loss_function(logits.transpose(-1, -2), shifted_labels)
    sequence_log_probs = log_probs.sum(dim=-1)
    return sequence_log_probs

def _avg_logprob_from_logits(logits, labels, ignore_index=-100, average_log_prob=True):
    """
    Average log-prob over non-ignored positions (per sequence)
    Unified function used by SimPO, RePO, SimPER, etc.
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()
    loss_mask = (labels != ignore_index)
    labels[labels == ignore_index] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    summed = (per_token_logps * loss_mask).sum(-1)
    if average_log_prob:
        denom = loss_mask.sum(-1).clamp_min(1)
        return summed / denom
    else:
        return summed

# -------------------------
# Helper Functions for Loss Computation
# -------------------------

def _get_device(model):
    """Get device from model"""
    return next(model.parameters()).device

def _forward_model(model, input_ids, attention_mask, labels=None):
    """
    Unified model forward pass
    Returns model outputs
    """
    device = _get_device(model)
    kwargs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }
    if labels is not None:
        kwargs["labels"] = labels.to(device)
    return model(**kwargs)

def _forward_chosen_rejected(main_model, batch, ref_model=None):
    """
    Forward pass for chosen and rejected sequences
    Returns (ch_out, rj_out, ref_ch_out, ref_rj_out)
    ref_ch_out and ref_rj_out are None if ref_model is None
    """
    ch_out = _forward_model(
        main_model,
        batch['chosen_input_ids'],
        batch['chosen_attention_mask'],
        batch['chosen_labels']
    )
    rj_out = _forward_model(
        main_model,
        batch['rejected_input_ids'],
        batch['rejected_attention_mask'],
        batch['rejected_labels']
    )
    
    ref_ch_out = None
    ref_rj_out = None
    if ref_model is not None:
        with torch.no_grad():
            ref_ch_out = _forward_model(
                ref_model,
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
                batch['chosen_labels']
            )
            ref_rj_out = _forward_model(
                ref_model,
                batch['rejected_input_ids'],
                batch['rejected_attention_mask'],
                batch['rejected_labels']
            )
    
    return ch_out, rj_out, ref_ch_out, ref_rj_out

def _compute_log_probs_from_outputs(ch_out, rj_out, ref_ch_out, ref_rj_out, 
                                    batch, use_sequence_log_probs=True, 
                                    average_log_prob=True, ignore_index=-100):
    """
    Compute log probabilities from model outputs
    Supports both sequence-level (sum) and average log probs
    """
    device = ch_out.logits.device
    
    if use_sequence_log_probs:
        # Sequence-level (sum) log probs (used by DPO, IPO, BDPO)
        logp_ch = get_sequence_log_probs(ch_out.logits, batch['chosen_labels'].to(device))
        logp_rj = get_sequence_log_probs(rj_out.logits, batch['rejected_labels'].to(device))
        
        if ref_ch_out is not None and ref_rj_out is not None:
            logp_ref_ch = get_sequence_log_probs(ref_ch_out.logits, batch['chosen_labels'].to(device))
            logp_ref_rj = get_sequence_log_probs(ref_rj_out.logits, batch['rejected_labels'].to(device))
        else:
            logp_ref_ch = None
            logp_ref_rj = None
    else:
        # Average log probs (used by SimPO, RePO, SimPER)
        logp_ch = _avg_logprob_from_logits(ch_out.logits, batch['chosen_labels'].to(device), 
                                          ignore_index=ignore_index, average_log_prob=average_log_prob)
        logp_rj = _avg_logprob_from_logits(rj_out.logits, batch['rejected_labels'].to(device),
                                          ignore_index=ignore_index, average_log_prob=average_log_prob)
        
        if ref_ch_out is not None and ref_rj_out is not None:
            logp_ref_ch = _avg_logprob_from_logits(ref_ch_out.logits, batch['chosen_labels'].to(device),
                                                   ignore_index=ignore_index, average_log_prob=average_log_prob)
            logp_ref_rj = _avg_logprob_from_logits(ref_rj_out.logits, batch['rejected_labels'].to(device),
                                                   ignore_index=ignore_index, average_log_prob=average_log_prob)
        else:
            logp_ref_ch = None
            logp_ref_rj = None
    
    return logp_ch, logp_rj, logp_ref_ch, logp_ref_rj

def _compute_log_probs_from_loss(ch_out, rj_out, ref_ch_out, ref_rj_out):
    """
    Compute log probs from model loss (negative of loss)
    Used by IPO and BDPO
    """
    logp_ch = -ch_out.loss
    logp_rj = -rj_out.loss
    logp_ref_ch = -ref_ch_out.loss if ref_ch_out is not None else None
    logp_ref_rj = -ref_rj_out.loss if ref_rj_out is not None else None
    return logp_ch, logp_rj, logp_ref_ch, logp_ref_rj

def _create_stats_dict(loss, chosen=None, rejected=None, ref_chosen=None, ref_rejected=None, **extra):
    """
    Create standardized stats dictionary
    """
    stats = {"loss": loss.item() if isinstance(loss, torch.Tensor) else loss}
    
    if chosen is not None:
        if isinstance(chosen, torch.Tensor):
            stats["chosen"] = chosen.mean().item() if chosen.numel() > 1 else chosen.item()
        else:
            stats["chosen"] = chosen
    
    if rejected is not None:
        if isinstance(rejected, torch.Tensor):
            stats["rejected"] = rejected.mean().item() if rejected.numel() > 1 else rejected.item()
        else:
            stats["rejected"] = rejected
    
    if ref_chosen is not None:
        if isinstance(ref_chosen, torch.Tensor):
            stats["ref_chosen"] = ref_chosen.mean().item() if ref_chosen.numel() > 1 else ref_chosen.item()
        else:
            stats["ref_chosen"] = ref_chosen
    
    if ref_rejected is not None:
        if isinstance(ref_rejected, torch.Tensor):
            stats["ref_rejected"] = ref_rejected.mean().item() if ref_rejected.numel() > 1 else ref_rejected.item()
        else:
            stats["ref_rejected"] = ref_rejected
    
    stats.update(extra)
    return stats

# -------------------------
# NPO (optional)
# -------------------------

def npo_loss(model, ref_model, inputs, beta=0.2, alpha=0.2, retain_loss=False):
    device = _get_device(model)
    non_preferred_input_ids = inputs["input_ids"].to(device)
    non_preferred_labels = inputs["labels"].to(device)
    non_preferred_attention_mask = inputs["attention_mask"].to(device)

    outputs_main = model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)
    with torch.no_grad():
        outputs_ref = ref_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    logp_main = get_sequence_log_probs(outputs_main.logits, non_preferred_labels)
    logp_ref  = get_sequence_log_probs(outputs_ref.logits,  non_preferred_labels)
    loss_val = -F.logsigmoid(-beta * (logp_main - logp_ref)) * 2 / beta
    loss_val = loss_val.mean()

    if retain_loss:
        utility_input_ids = inputs['utility_input_ids'].to(device)
        utility_labels = inputs['utility_labels'].to(device)
        utility_attention_mask = inputs['utility_attention_mask'].to(device)
        utility_outputs = model(utility_input_ids, attention_mask=utility_attention_mask)
        utility_log_probs = get_sequence_log_probs(utility_outputs.logits, utility_labels)
        utility_loss = -utility_log_probs.mean()
        loss_val = alpha * loss_val + (1 - alpha) * utility_loss

    return loss_val

# -------------------------
# DPO / IPO
# -------------------------

def dpo_loss(main_model, ref_model, batch, beta=0.5):
    """DPO loss with sequence-level log probs"""
    ch_out, rj_out, ref_ch_out, ref_rj_out = _forward_chosen_rejected(main_model, batch, ref_model)
    
    logp_ch, logp_rj, logp_ref_ch, logp_ref_rj = _compute_log_probs_from_outputs(
        ch_out, rj_out, ref_ch_out, ref_rj_out, batch, use_sequence_log_probs=True
    )

    delta_model = logp_ch - logp_rj
    delta_ref   = logp_ref_ch - logp_ref_rj

    loss_vec = -F.logsigmoid(beta * (delta_model - delta_ref))
    loss = loss_vec.mean()

    x1 = torch.exp(logp_ch - logp_ref_ch).mean().item()
    x2 = torch.exp(logp_rj - logp_ref_rj).mean().item()

    stats = _create_stats_dict(
        loss,
        chosen=logp_ch,
        rejected=logp_rj,
        ref_chosen=logp_ref_ch,
        ref_rejected=logp_ref_rj,
        x1=x1,
        x2=x2,
    )
    return loss, stats

def ipo_loss(main_model, ref_model, batch, beta=0.5):
    """IPO loss using model loss directly"""
    ch_out, rj_out, ref_ch_out, ref_rj_out = _forward_chosen_rejected(main_model, batch, ref_model)
    
    logp_ch, logp_rj, logp_ref_ch, logp_ref_rj = _compute_log_probs_from_loss(ch_out, rj_out, ref_ch_out, ref_rj_out)

    delta_model = logp_ch - logp_rj
    delta_ref   = logp_ref_ch - logp_ref_rj
    target_margin = 1 / (2 * beta)

    loss_vec = (delta_model - delta_ref - target_margin) ** 2
    loss = loss_vec.mean()

    stats = _create_stats_dict(
        loss,
        chosen=logp_ch,
        rejected=logp_rj,
        ref_chosen=logp_ref_ch,
        ref_rejected=logp_ref_rj,
        target_margin=target_margin,
    )
    return loss, stats

# -------------------------
# TDPO (token-level)
# -------------------------

def tdpo_get_batch_logps(logits, reference_logits, labels, average_log_prob=False, ignore_index=-100):
    """
    Compute per-token margins and per-position forward-KL, masked by labels.
    """
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]
    loss_mask = (labels != ignore_index)

    labels[labels == ignore_index] = 0  # safe gather index

    vocab_logps = logits.log_softmax(-1)
    reference_vocab_logps = reference_logits.log_softmax(-1)
    reference_vocab_ps = reference_logits.softmax(-1)

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    if average_log_prob:
        mask_sum = loss_mask.sum(-1).clamp(min=1e-8)
        return (logps_margin * loss_mask).sum(-1) / mask_sum, (per_position_kl * loss_mask).sum(-1) / mask_sum
    else:
        return (logps_margin * loss_mask).sum(-1), (per_position_kl * loss_mask).sum(-1)

def tdpo_loss(main_model, ref_model, batch, beta=0.5, alpha=0.5):
    """
    TDPO-2 style: token-level margins with forward-KL on the rejected path.
    """
    ch_out, rj_out, ref_ch_out, ref_rj_out = _forward_chosen_rejected(main_model, batch, ref_model)
    
    device = _get_device(main_model)
    ch_margin, ch_kl = tdpo_get_batch_logps(ch_out.logits, ref_ch_out.logits, batch['chosen_labels'].to(device))
    rj_margin, rj_kl = tdpo_get_batch_logps(rj_out.logits, ref_rj_out.logits, batch['rejected_labels'].to(device))

    logits = (ch_margin - rj_margin) - alpha * (rj_kl - ch_kl).detach()
    loss_vec = -F.logsigmoid(beta * logits)
    loss = loss_vec.mean()

    stats = _create_stats_dict(
        loss,
        chosen=ch_margin,
        rejected=rj_margin,
        x1=math.exp(ch_margin.mean().item()),
        x2=math.exp(rj_margin.mean().item()),
    )
    return loss, stats

# -------------------------
# BDPO (bounded denominator for rejected)
# -------------------------

def bdpo_loss(main_model, ref_model, batch, beta=0.5, lambda_mix: float = 0.5):
    """
    Replace πθ(y^-|x) with π_mix = λ πθ + (1-λ) π_ref in the denominator,
    limiting the influence of extreme rejected responses.
    """
    ch_out, rj_out, ref_ch_out, ref_rj_out = _forward_chosen_rejected(main_model, batch, ref_model)
    
    logp_ch, logp_rj, logp_ref_ch, logp_ref_rj = _compute_log_probs_from_loss(ch_out, rj_out, ref_ch_out, ref_rj_out)

    lam = float(min(max(lambda_mix, 1e-6), 1.0 - 1e-6))
    log_lam = math.log(lam)
    log_one_minus = math.log(1.0 - lam)

    # log π_mix(y^-|x) = logaddexp(log λ + log πθ, log(1-λ) + log πref)
    logp_mix_rj = torch.logaddexp(logp_rj + log_lam, logp_ref_rj + log_one_minus)

    delta_model_bdpo = logp_ch - logp_mix_rj
    delta_ref = logp_ref_ch - logp_ref_rj

    pre = beta * (delta_model_bdpo - delta_ref)
    loss_vec = -F.logsigmoid(pre)
    loss = loss_vec.mean()

    stats = _create_stats_dict(
        loss,
        chosen=logp_ch,
        rejected=logp_rj,
        beta=beta,
        lambda_mix=lam,
        reject_logp_mix=logp_mix_rj,
    )
    return loss, stats

# -------------------------
# SimPO (reference-free)
# -------------------------

def simpo_loss(main_model, batch, beta: float = 2.0, gamma: float = 0.5, average_log_prob: bool = True):
    """
    SimPO: sequence-level average log-prob margin with target gap gamma (no ref).
    """
    ch_out, rj_out, _, _ = _forward_chosen_rejected(main_model, batch, ref_model=None)
    
    r_ch = _avg_logprob_from_logits(ch_out.logits, batch["chosen_labels"].to(_get_device(main_model)), 
                                    average_log_prob=average_log_prob)
    r_rj = _avg_logprob_from_logits(rj_out.logits, batch["rejected_labels"].to(_get_device(main_model)), 
                                    average_log_prob=average_log_prob)

    pre = beta * ((r_ch - r_rj) - gamma)
    loss_vec = -F.logsigmoid(pre)
    loss = loss_vec.mean()

    chosen_rewards = beta * r_ch.detach()
    rejected_rewards = beta * r_rj.detach()
    stats = _create_stats_dict(
        loss,
        chosen=r_ch,
        rejected=r_rj,
        chosen_rewards=chosen_rewards.mean().item() if isinstance(chosen_rewards, torch.Tensor) else chosen_rewards,
        rejected_rewards=rejected_rewards.mean().item() if isinstance(rejected_rewards, torch.Tensor) else rejected_rewards,
    )
    return loss, stats

def repo_loss(main_model, batch, gamma: float = 0.5, average_log_prob: bool = True):
    """
    RePO: ReLU-based max-margin loss (reference-free).
    Margin M = r_ch - r_rj, where r_* are length-normalized (avg) log-probs.
    Loss = ReLU(gamma - M).  Only pairs with M < gamma contribute gradients.
    """
    ch_out, rj_out, _, _ = _forward_chosen_rejected(main_model, batch, ref_model=None)
    
    r_ch = _avg_logprob_from_logits(ch_out.logits, batch["chosen_labels"].to(_get_device(main_model)), 
                                    average_log_prob=average_log_prob)
    r_rj = _avg_logprob_from_logits(rj_out.logits, batch["rejected_labels"].to(_get_device(main_model)), 
                                    average_log_prob=average_log_prob)

    margin = r_ch - r_rj
    loss_vec = F.relu(gamma - margin)        # hinge / ReLU max-margin
    loss = loss_vec.mean()
    
    stats = _create_stats_dict(
        loss,
        chosen=r_ch,
        rejected=r_rj,
        chosen_rewards=r_ch.mean().item() if isinstance(r_ch, torch.Tensor) else r_ch,
        rejected_rewards=r_rj.mean().item() if isinstance(r_rj, torch.Tensor) else r_rj,
    )
    return loss, stats

def simper_loss(main_model, batch, beta=1.0):
    """
    SimPER (Reference-Free, Hyperparameter-Free in its core form)

    Core idea:
      s(y) = avg log-prob over tokens (length-normalized)
      InvPPL(y) = exp(s(y))  (geometric mean token probability)

    Loss:
      L = exp(s_rejected) - exp(s_chosen)   (minimize)
    """
    ch_out, rj_out, _, _ = _forward_chosen_rejected(main_model, batch, ref_model=None)
    
    device = _get_device(main_model)
    s_ch = _avg_logprob_from_logits(ch_out.logits, batch['chosen_labels'].to(device))
    s_rj = _avg_logprob_from_logits(rj_out.logits, batch['rejected_labels'].to(device))

    invppl_ch = torch.exp(s_ch)  # [B]
    invppl_rj = torch.exp(s_rj)  # [B]

    loss_vec = invppl_rj - invppl_ch
    loss = loss_vec.mean()

    chosen_rewards = (beta * invppl_ch).detach()
    rejected_rewards = (beta * invppl_rj).detach()

    stats = _create_stats_dict(
        loss,
        chosen=s_ch,
        rejected=s_rj,
        margin=(chosen_rewards - rejected_rewards).mean().item(),
    )
    return loss, stats

def hybrid_simper_dpo_loss(
    main_model,
    batch,
    beta: float = 2.0,
    gamma: float = 0.5,
    detach_weight: bool = True,
    weight_eps: float = 1e-8,
):
    """
    Reference-free loss:
      s_ch, s_rj = avg log-prob (length-normalized)
      M = s_ch - s_rj
      w = exp(s_ch) + exp(s_rj)    (SimPER-style confidence / plausibility weight)

      L = w * (-log sigmoid(beta*(M - gamma)))

    detach_weight=True is recommended to prevent the model from "gaming" the weight term.
    """
    ch_out, rj_out, _, _ = _forward_chosen_rejected(main_model, batch, ref_model=None)
    
    device = _get_device(main_model)
    s_ch = _avg_logprob_from_logits(ch_out.logits, batch["chosen_labels"].to(device), average_log_prob=True)
    s_rj = _avg_logprob_from_logits(rj_out.logits, batch["rejected_labels"].to(device), average_log_prob=True)

    gap = s_ch - s_rj  # [B]

    # SimPER-style plausibility weight
    w = torch.exp(s_ch) + torch.exp(s_rj)      # [B]
    w = torch.clamp(w, min=weight_eps)
    if detach_weight:
        w = w.detach()

    # NON-saturating hinge (keeps gradients until gap >= gamma)
    pref_vec = F.relu(gamma - gap)             # [B]

    loss_vec = w * pref_vec
    loss = loss_vec.mean()

    stats = _create_stats_dict(
        loss,
        chosen=s_ch,
        rejected=s_rj,
        margin=gap,
        w_mean=w.mean().item(),
        invppl_chosen=torch.exp(s_ch.detach()).mean().item(),
        invppl_rejected=torch.exp(s_rj.detach()).mean().item(),
    )
    return loss, stats

# -------------------------
# Advanced Loss Functions
# -------------------------

def token_kl_main_vs_anchor(
    main_logits, anchor_logits,
    labels=None, attention_mask=None,
    label_pad_token_id=-100,
):
    """
    Mean token KL( pi_main || pi_anchor ) over selected tokens.
    """
    main_logits = main_logits[:, :-1, :]
    anchor_logits = anchor_logits[:, :-1, :]

    main_logp = F.log_softmax(main_logits, dim=-1)      # [B, T-1, V]
    anchor_logp = F.log_softmax(anchor_logits, dim=-1)  # [B, T-1, V]
    main_p = main_logp.exp()

    kl_tok = (main_p * (main_logp - anchor_logp)).sum(dim=-1)  # [B, T-1]

    if labels is not None:
        lab = labels[:, 1:].clone()
        mask = (lab != label_pad_token_id).to(kl_tok.dtype)
    elif attention_mask is not None:
        mask = attention_mask[:, 1:].to(kl_tok.dtype)
    else:
        mask = torch.ones_like(kl_tok, dtype=kl_tok.dtype)

    denom = mask.sum().clamp_min(1.0)
    return (kl_tok * mask).sum() / denom

def anchored_simper_loss(
    main_model,
    anchor_model,               # EMA/snapshot model, frozen/no-grad in loss
    batch,
    beta: float = 1.0,          # only for logging scale (optional)
    lambda_anchor: float = 0.1, # = λ
    anchor_on: str = "both",    # "chosen" | "rejected" | "both"
    ignore_index: int = -100,
):
    """
    L = (exp(s_rj) - exp(s_ch)) + lambda_anchor * KL(main || anchor)
    """
    device = _get_device(main_model)
    assert anchor_model is not None, "Anchored SimPER requires anchor_model (EMA/snapshot)."

    # Main forward
    ch_out = _forward_model(main_model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
    rj_out = _forward_model(main_model, batch["rejected_input_ids"], batch["rejected_attention_mask"])

    # Avg log-probs over response tokens
    s_ch = _avg_logprob_from_logits(ch_out.logits, batch["chosen_labels"].to(device), 
                                    ignore_index=ignore_index, average_log_prob=True)
    s_rj = _avg_logprob_from_logits(rj_out.logits, batch["rejected_labels"].to(device), 
                                    ignore_index=ignore_index, average_log_prob=True)

    u_ch = torch.exp(s_ch)  # InvPPL (geom mean prob)
    u_rj = torch.exp(s_rj)

    pref_loss = (u_rj - u_ch).mean()

    # Anchor forward (no grad)
    anchor_model.eval()
    with torch.no_grad():
        a_ch = _forward_model(anchor_model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
        a_rj = _forward_model(anchor_model, batch["rejected_input_ids"], batch["rejected_attention_mask"])

    # Token KL(main || anchor) on response tokens only
    anchor_loss = torch.tensor(0.0, device=device)
    if lambda_anchor > 0:
        parts = []
        if anchor_on in ("chosen", "both"):
            parts.append(token_kl_main_vs_anchor(
                ch_out.logits, a_ch.logits,
                labels=batch["chosen_labels"].to(device),
                label_pad_token_id=ignore_index
            ))
        if anchor_on in ("rejected", "both"):
            parts.append(token_kl_main_vs_anchor(
                rj_out.logits, a_rj.logits,
                labels=batch["rejected_labels"].to(device),
                label_pad_token_id=ignore_index
            ))
        if len(parts) == 1:
            anchor_loss = parts[0]
        else:
            anchor_loss = 0.5 * (parts[0] + parts[1])

    loss = pref_loss + lambda_anchor * anchor_loss

    stats = _create_stats_dict(
        loss,
        chosen=s_ch,
        rejected=s_rj,
        pref_loss=pref_loss.item(),
        anchor_loss=anchor_loss.item(),
        margin_logp=(s_ch - s_rj),
        invppl_chosen=u_ch,
        invppl_rejected=u_rj,
        margin_invppl=(u_ch - u_rj),
    )
    return loss, stats

def make_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad = False
    return ema

@torch.no_grad()
def ema_update_(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.995):
    """
    ema = decay * ema + (1-decay) * model
    """
    for ep, p in zip(ema_model.parameters(), model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

class EMAAnchorCallback(TrainerCallback):
    """
    Updates trainer.ref_model (used as anchor) with EMA every step end.
    Works with HuggingFace Trainer/CustomDPOTrainer.
    """
    def __init__(self, decay: float = 0.995, update_every: int = 1):
        self.decay = decay
        self.update_every = update_every

    def on_step_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        model = kwargs.get("model", None)
        if trainer is None or model is None:
            return control
        if trainer.ref_model is None:
            return control
        if state.global_step % self.update_every != 0:
            return control
        ema_update_(trainer.ref_model, model, decay=self.decay)
        return control
    
def target_path_unlikelihood(
    logits, labels, bad_token_id: int, K: int = 1, ignore_index: int = -100
):
    """
    Penalize p(bad_token) on the first K response positions (teacher-forced).
    Uses labels mask to find response region (labels != ignore_index).
    """
    logits = logits[:, :-1, :]          # [B, T-1, V]
    labels = labels[:, 1:].clone()      # [B, T-1]
    resp_mask = (labels != ignore_index)

    probs = logits.softmax(-1)[..., bad_token_id]  # [B, T-1]
    probs = probs.clamp(1e-6, 1 - 1e-6)

    penalties = []
    for i in range(labels.shape[0]):
        idx = torch.nonzero(resp_mask[i], as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        idx = idx[:K]
        penalties.append((-torch.log(1.0 - probs[i, idx])).mean())

    if len(penalties) == 0:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(penalties).mean()

def ta_hsimper_loss(
    main_model,
    batch,
    trigger_augment_fn=None,
    use_hard: bool = True,
    hard_mode: str = "max",      # "max" or "topk"
    topk_frac: float = 0.25,     # for topk
    beta: float = 1.0,           # logging scale only
    add_target_path: bool = False,
    bad_token_id: int = None,
    tp_lambda: float = 0.02,
    tp_K: int = 1,
    ignore_index: int = -100,
):
    """
    Trigger-Augmented h-SimPER:
      - chosen is scored on clean prompt
      - rejected is scored on triggered prompt (via trigger_augment_fn)
      - hard negative is selected from the rejected pool (if use_hard)
    """
    device = _get_device(main_model)

    # Clean chosen
    ch_out = _forward_model(main_model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
    s_ch = _avg_logprob_from_logits(
        ch_out.logits, batch["chosen_labels"].to(device),
        ignore_index=ignore_index, average_log_prob=True
    )  # [B]
    u_ch = torch.exp(s_ch)

    # Triggered rejected inputs
    if trigger_augment_fn is None:
        rj_input_ids = batch["rejected_input_ids"].to(device)
        rj_attn = batch["rejected_attention_mask"].to(device)
    else:
        rj_input_ids, rj_attn = trigger_augment_fn(
            batch["rejected_input_ids"], batch["rejected_attention_mask"]
        )
        rj_input_ids = rj_input_ids.to(device)
        rj_attn = rj_attn.to(device)

    rj_out = _forward_model(main_model, rj_input_ids, rj_attn)
    s_rj = _avg_logprob_from_logits(
        rj_out.logits, batch["rejected_labels"].to(device),
        ignore_index=ignore_index, average_log_prob=True
    )  # [B]

    # Hard negative selection
    if use_hard:
        if hard_mode == "max":
            s_rj_hard = s_rj.max().expand_as(s_ch)
        elif hard_mode == "topk":
            B = s_rj.shape[0]
            k = max(1, int(B * topk_frac))
            s_rj_hard = s_rj.topk(k).values.mean().expand_as(s_ch)
        else:
            raise ValueError("hard_mode must be 'max' or 'topk'")
    else:
        s_rj_hard = s_rj

    u_rj_hard = torch.exp(s_rj_hard)

    # SimPER core loss
    pref_loss = (u_rj_hard - u_ch).mean()

    # Optional target-path term
    tp_loss = torch.tensor(0.0, device=device)
    if add_target_path:
        assert bad_token_id is not None, "Provide bad_token_id when add_target_path=True"
        tp_loss = target_path_unlikelihood(
            logits=rj_out.logits,
            labels=batch["rejected_labels"].to(device),
            bad_token_id=bad_token_id,
            K=tp_K,
            ignore_index=ignore_index,
        )

    loss = pref_loss + tp_lambda * tp_loss

    stats = _create_stats_dict(
        loss,
        chosen=s_ch,
        rejected=s_rj,
        rejected_hard=s_rj_hard,
        pref_loss=pref_loss.item(),
        tp_loss=tp_loss.item(),
        margin_logp=(s_ch - s_rj_hard),
        invppl_chosen=u_ch,
        invppl_rejected_hard=u_rj_hard,
    )
    return loss, stats

def behavioral_hard_simper_loss(
    main_model,
    batch,
    K: int = 4,                 # number of sampled hard negatives per prompt
    alpha: float = 1.0,         # sharpness for exp(alpha * s); alpha=1 recovers SimPER scale
    lambda_anchor: float = 0.05,# small stabilizer on chosen (SFT-like), can set 0.0
    average_log_prob: bool = True,  # MUST keep True for injection defense
    ignore_index: int = -100,
    # generation settings for hard negatives
    max_new_tokens: int = 32,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
):
    """
    Behavioral Hard SimPER

    Data assumption (works with SecAlign-style packing):
      - batch has chosen_input_ids/chosen_attention_mask/chosen_labels
      - batch has rejected_input_ids/rejected_attention_mask/rejected_labels
      - rejected_* contains the *attack context* (prompt + injection), then a rejected answer
      - rejected_labels has ignore_index on prompt/injection tokens, and real ids on answer tokens

    Loss:
      s+(x,y+) = avg logprob of chosen response under clean context
      Sample K completions y_k ~ pi_theta(.|x_atk) where x_atk is the prompt+injection prefix
      s_k = avg logprob of generated completion under x_atk
      U- = mean_k exp(alpha*s_k), U+ = exp(alpha*s+)
      L = U- - U+ + lambda_anchor * (-s+)
    """
    device = _get_device(main_model)

    # Score chosen (clean context) on the paired data
    ch_inp = batch["chosen_input_ids"].to(device)
    ch_att = batch["chosen_attention_mask"].to(device)
    ch_lab = batch["chosen_labels"].to(device)

    ch_out = main_model(input_ids=ch_inp, attention_mask=ch_att)
    s_ch = _avg_logprob_from_logits(ch_out.logits, ch_lab, ignore_index=ignore_index, average_log_prob=average_log_prob)
    U_pos = torch.exp(alpha * s_ch)  # [B]

    # Build attack-context prefix x_atk from rejected_* by using rejected_labels mask
    rj_inp_full = batch["rejected_input_ids"].to(device)
    rj_att_full = batch["rejected_attention_mask"].to(device)
    rj_lab_full = batch["rejected_labels"].to(device)

    B, L = rj_inp_full.shape
    # Find prefix length per sample
    with torch.no_grad():
        is_answer = (rj_lab_full != ignore_index)
        prefix_len = torch.full((B,), L, dtype=torch.long, device=device)
        for i in range(B):
            idx = torch.nonzero(is_answer[i], as_tuple=False)
            if idx.numel() > 0:
                prefix_len[i] = idx[0].item()

        max_prefix = int(prefix_len.max().item())

    # Slice to max_prefix then pad per-sample via attention mask
    atk_prompt_ids = rj_inp_full[:, :max_prefix].contiguous()
    atk_prompt_att = rj_att_full[:, :max_prefix].contiguous()

    # Zero-out attention after each prefix_len
    with torch.no_grad():
        for i in range(B):
            if prefix_len[i] < max_prefix:
                atk_prompt_att[i, prefix_len[i]:] = 0

    # Generate K hard negatives from attack context (no grad for generation)
    pad_id = getattr(main_model.config, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0
    eos_id = getattr(main_model.config, "eos_token_id", None)

    with torch.no_grad():
        gen = main_model.generate(
            input_ids=atk_prompt_ids,
            attention_mask=atk_prompt_att,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=K,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=True,
        )

    # Build repeated prompts to match gen rows
    atk_prompt_ids_rep = atk_prompt_ids.repeat_interleave(K, dim=0)  # [B*K, P]
    atk_prompt_att_rep = atk_prompt_att.repeat_interleave(K, dim=0)  # [B*K, P]

    # Extract only the newly generated tokens
    gen_resp = gen[:, max_prefix:]  # [B*K, <=max_new_tokens]

    # Construct full sequence = prompt + gen_resp
    full_ids = torch.cat([atk_prompt_ids_rep, gen_resp], dim=1)  # [B*K, P+R]
    full_att = torch.cat(
        [atk_prompt_att_rep, (gen_resp != pad_id).long()],
        dim=1
    )

    # Labels: ignore prompt positions; supervise only generated positions
    full_lab = torch.full_like(full_ids, ignore_index)
    full_lab[:, max_prefix:] = full_ids[:, max_prefix:]

    # Score generated negatives with grad ON
    neg_out = main_model(input_ids=full_ids, attention_mask=full_att)
    s_neg = _avg_logprob_from_logits(neg_out.logits, full_lab, ignore_index=ignore_index, average_log_prob=average_log_prob)
    U_neg_each = torch.exp(alpha * s_neg)  # [B*K]

    # Aggregate per original sample
    U_neg = U_neg_each.view(B, K).mean(dim=1)  # [B]

    # Loss
    loss_vec = (U_neg - U_pos) + (lambda_anchor * (-s_ch))
    loss = loss_vec.mean()

    # Stats
    with torch.no_grad():
        stats = {
            "loss": float(loss.item()),
            "chosen": float(s_ch.mean().item()),
            "rejected": float(s_neg.view(B, K).mean(dim=1).mean().item()),
            "margin": float((s_ch - s_neg.view(B, K).mean(dim=1)).mean().item()),
            "U_pos_mean": float(U_pos.mean().item()),
            "U_neg_mean": float(U_neg.mean().item()),
            "K": K,
            "alpha": alpha,
            "lambda_anchor": lambda_anchor,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

    return loss, stats

# -------------------------
# Custom Trainer wrapper
# -------------------------

class CustomDPOTrainer(Trainer):
    """
    Trainer wrapper that can switch among loss functions and track stats.
    """
    def __init__(self, *args, loss_fn: str = "dpo", beta: float = 0.5, gamma: float = 0.5,
                 alpha: float = 0.5, lambda_mix: float = 0.5,
                 return_outputs: bool = False, ref_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.step_metrics = []  # Track metrics at logging steps
        self.last_log_step = -1  # Track last logging step
        # Accumulate stats between logging steps for averaging
        self._pending_stats = []  # Stats since last log (cleared after each log)
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_mix = lambda_mix
        self.return_outputs = return_outputs

        # Map loss function names to implementations
        # Note: All lambdas receive (m, r, b, beta) but some loss functions don't use all params
        loss_map = {
            "dpo": lambda m, r, b, beta: dpo_loss(m, r, b, beta=beta),
            "ipo": lambda m, r, b, beta: ipo_loss(m, r, b, beta=beta),
            "tdpo": lambda m, r, b, beta: tdpo_loss(m, r, b, beta=beta, alpha=self.alpha),
            "bdpo": lambda m, r, b, beta: bdpo_loss(m, r, b, beta=beta, lambda_mix=self.lambda_mix),
            "simpo": lambda m, r, b, beta: simpo_loss(m, b, beta=beta, gamma=self.gamma),
            "repo": lambda m, r, b, beta: repo_loss(m, b, gamma=self.gamma),  # beta not used
            "simper": lambda m, r, b, beta: simper_loss(m, b, beta=beta),
            "hybrid_simper_dpo": lambda m, r, b, beta: hybrid_simper_dpo_loss(m, b, beta=beta, gamma=self.gamma),
            "a-simper": lambda m, r, b, beta: anchored_simper_loss(
                m, self.ref_model, b, beta=beta, lambda_anchor=self.alpha, anchor_on="both"
            ),
            "ta_hsimper": lambda m, r, b, beta: ta_hsimper_loss(m, b, beta=beta),
            "behavioral_hard_simper": lambda m, r, b, beta: behavioral_hard_simper_loss(
                m, b, alpha=self.alpha  # Note: uses alpha (sharpness), lambda_anchor uses default 0.05
            ),
            # Alias: allow CLI to use the function-style name.
            "behavioral_hard_simper_loss": lambda m, r, b, beta: behavioral_hard_simper_loss(
                m, b, alpha=self.alpha
            ),
        }
        
        if loss_fn not in loss_map:
            raise ValueError(f"Invalid loss function: {loss_fn}. Available: {list(loss_map.keys())}")
        
        self.loss_impl = loss_map[loss_fn]

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        loss, stats = self.loss_impl(main_model=model, ref_model=self.ref_model, batch=inputs, beta=self.beta)
        # Store stats for averaging at next logging step (memory efficient)
        if self.return_outputs:
            self._pending_stats.append(stats)
        return loss
    
    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        """
        Override log to capture chosen/rejected likelihoods at logging steps.
        Averages likelihoods over batches since the last log for stability.
        Clears pending stats after averaging to save memory.
        """
        super().log(logs, start_time)
        
        if not hasattr(self.state, 'global_step') or not self._pending_stats:
            return
        
        current_step = self.state.global_step
        
        # Only record if this is a new logging step
        if current_step <= self.last_log_step:
            return
        
        # Average likelihoods over batches since last log
        chosen_likelihoods = []
        rejected_likelihoods = []
        
        for stats in self._pending_stats:
            chosen_likelihood = None
            rejected_likelihood = None
            
            if 'chosen' in stats:
                chosen_likelihood = stats['chosen']
            elif 'chosen_rewards' in stats:
                chosen_rewards = stats['chosen_rewards']
                if isinstance(chosen_rewards, torch.Tensor):
                    chosen_likelihood = chosen_rewards.mean().item()
                else:
                    chosen_likelihood = chosen_rewards
            
            if 'rejected' in stats:
                rejected_likelihood = stats['rejected']
            elif 'rejected_rewards' in stats:
                rejected_rewards = stats['rejected_rewards']
                if isinstance(rejected_rewards, torch.Tensor):
                    rejected_likelihood = rejected_rewards.mean().item()
                else:
                    rejected_likelihood = rejected_rewards
            
            if chosen_likelihood is not None and rejected_likelihood is not None:
                chosen_likelihoods.append(chosen_likelihood)
                rejected_likelihoods.append(rejected_likelihood)
        
        if chosen_likelihoods and rejected_likelihoods:
            avg_chosen = sum(chosen_likelihoods) / len(chosen_likelihoods)
            avg_rejected = sum(rejected_likelihoods) / len(rejected_likelihoods)
            margin = avg_chosen - avg_rejected
            
            step_metric = {
                'step': current_step,
                'chosen_likelihood': avg_chosen,
                'rejected_likelihood': avg_rejected,
                'margin': margin,
            }
            self.step_metrics.append(step_metric)
            self.last_log_step = current_step
        
        # Clear pending stats after averaging to save memory
        self._pending_stats.clear()
    
    def _prepare_inputs(self, inputs):
        prepped = {}
        for k, v in inputs.items():
            prepped[k] = v.to(self.args.device) if isinstance(v, torch.Tensor) else v
        return prepped
