"""
Custom loss functions for preference optimization
(DPO, IPO, TDPO, BDPO, SimPO) + utilities
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
# Utilities
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
# NPO (optional)
# -------------------------

def npo_loss(model, ref_model, inputs, beta=0.2, alpha=0.2, retain_loss=False):
    device = next(model.parameters()).device
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
    device = next(main_model.parameters()).device

    ch_out = main_model(
        input_ids=batch['chosen_input_ids'].to(device),
        attention_mask=batch['chosen_attention_mask'].to(device),
        labels=batch['chosen_labels'].to(device),
    )
    rj_out = main_model(
        input_ids=batch['rejected_input_ids'].to(device),
        attention_mask=batch['rejected_attention_mask'].to(device),
        labels=batch['rejected_labels'].to(device),
    )

    with torch.no_grad():
        ref_ch_out = ref_model(
            input_ids=batch['chosen_input_ids'].to(device),
            attention_mask=batch['chosen_attention_mask'].to(device),
            labels=batch['chosen_labels'].to(device),
        )
        ref_rj_out = ref_model(
            input_ids=batch['rejected_input_ids'].to(device),
            attention_mask=batch['rejected_attention_mask'].to(device),
            labels=batch['rejected_labels'].to(device),
        )

    # Use sequence log-prob sums (more faithful to DPO) instead of
    # model-reported average CE. This avoids degenerate near-zero losses
    # when averages saturate.
    logp_ch      = get_sequence_log_probs(ch_out.logits, batch['chosen_labels'].to(device))
    logp_rj      = get_sequence_log_probs(rj_out.logits, batch['rejected_labels'].to(device))
    with torch.no_grad():
        logp_ref_ch  = get_sequence_log_probs(ref_ch_out.logits, batch['chosen_labels'].to(device))
        logp_ref_rj  = get_sequence_log_probs(ref_rj_out.logits, batch['rejected_labels'].to(device))

    delta_model = logp_ch - logp_rj
    delta_ref   = logp_ref_ch - logp_ref_rj

    loss_vec = -F.logsigmoid(beta * (delta_model - delta_ref))
    loss = loss_vec.mean()

    x1 = torch.exp(logp_ch - logp_ref_ch).mean().item()
    x2 = torch.exp(logp_rj - logp_ref_rj).mean().item()

    stats = {
        'loss': loss.item(),
        'x1': x1,
        'x2': x2,
        'ref_chosen': logp_ref_ch.mean().item(),
        'ref_rejected': logp_ref_rj.mean().item(),
        'chosen': logp_ch.mean().item(),
        'rejected': logp_rj.mean().item(),
    }
    return loss, stats

def ipo_loss(main_model, ref_model, batch, beta=0.5):
    device = next(main_model.parameters()).device

    ch_out = main_model(
        input_ids=batch['chosen_input_ids'].to(device),
        attention_mask=batch['chosen_attention_mask'].to(device),
        labels=batch['chosen_labels'].to(device),
    )
    rj_out = main_model(
        input_ids=batch['rejected_input_ids'].to(device),
        attention_mask=batch['rejected_attention_mask'].to(device),
        labels=batch['rejected_labels'].to(device),
    )

    with torch.no_grad():
        ref_ch_out = ref_model(
            input_ids=batch['chosen_input_ids'].to(device),
            attention_mask=batch['chosen_attention_mask'].to(device),
            labels=batch['chosen_labels'].to(device),
        )
        ref_rj_out = ref_model(
            input_ids=batch['rejected_input_ids'].to(device),
            attention_mask=batch['rejected_attention_mask'].to(device),
            labels=batch['rejected_labels'].to(device),
        )

    logp_ch     = -ch_out.loss
    logp_rj     = -rj_out.loss
    logp_ref_ch = -ref_ch_out.loss
    logp_ref_rj = -ref_rj_out.loss

    delta_model = logp_ch - logp_rj
    delta_ref   = logp_ref_ch - logp_ref_rj
    target_margin = 1 / (2 * beta)

    loss_vec = (delta_model - delta_ref - target_margin) ** 2
    loss = loss_vec.mean()

    stats = {
        'loss': loss.item(),
        'target_margin': target_margin,
        'chosen': logp_ch.mean().item(),
        'rejected': logp_rj.mean().item(),
        'ref_chosen': logp_ref_ch.mean().item(),
        'ref_rejected': logp_ref_rj.mean().item(),
    }
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
    device = next(main_model.parameters()).device

    ch_out = main_model(
        input_ids=batch['chosen_input_ids'].to(device),
        attention_mask=batch['chosen_attention_mask'].to(device),
        labels=batch['chosen_labels'].to(device),
    )
    rj_out = main_model(
        input_ids=batch['rejected_input_ids'].to(device),
        attention_mask=batch['rejected_attention_mask'].to(device),
        labels=batch['rejected_labels'].to(device),
    )

    with torch.no_grad():
        ref_ch_out = ref_model(
            input_ids=batch['chosen_input_ids'].to(device),
            attention_mask=batch['chosen_attention_mask'].to(device),
            labels=batch['chosen_labels'].to(device),
        )
        ref_rj_out = ref_model(
            input_ids=batch['rejected_input_ids'].to(device),
            attention_mask=batch['rejected_attention_mask'].to(device),
            labels=batch['rejected_labels'].to(device),
        )

    ch_margin, ch_kl = tdpo_get_batch_logps(ch_out.logits, ref_ch_out.logits, batch['chosen_labels'].to(device))
    rj_margin, rj_kl = tdpo_get_batch_logps(rj_out.logits, ref_rj_out.logits, batch['rejected_labels'].to(device))

    logits = (ch_margin - rj_margin) - alpha * (rj_kl - ch_kl).detach()
    loss_vec = -F.logsigmoid(beta * logits)
    loss = loss_vec.mean()

    # Extract likelihoods from margins (margins are logp - logp_ref)
    # We approximate logp from margins + ref (but ref not stored, so use margins as proxy)
    # For visualization, use the margins as relative likelihoods
    ch_logp_approx = ch_margin.mean().item()  # Approximate chosen log prob
    rj_logp_approx = rj_margin.mean().item()  # Approximate rejected log prob
    
    stats = {
        'loss': loss.item(),
        'x1': math.exp(ch_margin.mean().item()),
        'x2': math.exp(rj_margin.mean().item()),
        'chosen': ch_logp_approx,
        'rejected': rj_logp_approx,
    }
    return loss, stats

# -------------------------
# BDPO (bounded denominator for rejected)
# -------------------------

def bdpo_loss(main_model, ref_model, batch, beta=0.5, lambda_mix: float = 0.5):
    """
    Replace πθ(y^-|x) with π_mix = λ πθ + (1-λ) π_ref in the denominator,
    limiting the influence of extreme rejected responses.
    """
    device = next(main_model.parameters()).device
    lam = float(min(max(lambda_mix, 1e-6), 1.0 - 1e-6))
    log_lam = math.log(lam)
    log_one_minus = math.log(1.0 - lam)

    ch_out = main_model(
        input_ids=batch['chosen_input_ids'].to(device),
        attention_mask=batch['chosen_attention_mask'].to(device),
        labels=batch['chosen_labels'].to(device),
    )
    rj_out = main_model(
        input_ids=batch['rejected_input_ids'].to(device),
        attention_mask=batch['rejected_attention_mask'].to(device),
        labels=batch['rejected_labels'].to(device),
    )

    with torch.no_grad():
        ref_ch_out = ref_model(
            input_ids=batch['chosen_input_ids'].to(device),
            attention_mask=batch['chosen_attention_mask'].to(device),
            labels=batch['chosen_labels'].to(device),
        )
        ref_rj_out = ref_model(
            input_ids=batch['rejected_input_ids'].to(device),
            attention_mask=batch['rejected_attention_mask'].to(device),
            labels=batch['rejected_labels'].to(device),
        )

    logp_ch     = -ch_out.loss
    logp_rj     = -rj_out.loss
    logp_ref_ch = -ref_ch_out.loss
    logp_ref_rj = -ref_rj_out.loss

    # log π_mix(y^-|x) = logaddexp(log λ + log πθ, log(1-λ) + log πref)
    logp_mix_rj = torch.logaddexp(logp_rj + log_lam, logp_ref_rj + log_one_minus)

    delta_model_bdpo = logp_ch - logp_mix_rj
    delta_ref = logp_ref_ch - logp_ref_rj

    pre = beta * (delta_model_bdpo - delta_ref)
    loss_vec = -F.logsigmoid(pre)
    loss = loss_vec.mean()

    stats = {
        'loss': loss.item(),
        'beta': beta,
        'lambda_mix': lam,
        'reject_logp_mix': logp_mix_rj.mean().item(),
        'chosen': logp_ch.mean().item(),
        'rejected': logp_rj.mean().item(),
    }
    return loss, stats

# -------------------------
# SimPO (reference-free)
# -------------------------

def simpo_loss(main_model, batch, beta: float = 2.0, gamma: float = 0.5, average_log_prob: bool = True):
    """
    SimPO: sequence-level average log-prob margin with target gap gamma (no ref).
    """
    device = next(main_model.parameters()).device

    ch_out = main_model(
        input_ids=batch["chosen_input_ids"].to(device),
        attention_mask=batch["chosen_attention_mask"].to(device),
        labels=batch["chosen_labels"].to(device),
    )
    rj_out = main_model(
        input_ids=batch["rejected_input_ids"].to(device),
        attention_mask=batch["rejected_attention_mask"].to(device),
        labels=batch["rejected_labels"].to(device),
    )

    r_ch = _avg_logprob_from_logits(ch_out.logits, batch["chosen_labels"].to(device), average_log_prob=average_log_prob)
    r_rj = _avg_logprob_from_logits(rj_out.logits, batch["rejected_labels"].to(device), average_log_prob=average_log_prob)

    pre = beta * ((r_ch - r_rj) - gamma)
    loss_vec = -F.logsigmoid(pre)
    loss = loss_vec.mean()

    chosen_rewards = beta * r_ch.detach()
    rejected_rewards = beta * r_rj.detach()
    stats = {
        'loss': loss.item(),
        'chosen_rewards': chosen_rewards,
        'rejected_rewards': rejected_rewards,
        'chosen': r_ch.mean().item(),  # Average log prob (likelihood)
        'rejected': r_rj.mean().item(),  # Average log prob (likelihood)
    }
    return loss, stats

def repo_loss(main_model, batch, gamma: float = 0.5, average_log_prob: bool = True):
    """
    RePO: ReLU-based max-margin loss (reference-free).
    Margin M = r_ch - r_rj, where r_* are length-normalized (avg) log-probs.
    Loss = ReLU(gamma - M).  Only pairs with M < gamma contribute gradients.
    """
    device = next(main_model.parameters()).device

    ch_out = main_model(
        input_ids=batch["chosen_input_ids"].to(device),
        attention_mask=batch["chosen_attention_mask"].to(device),
        labels=batch["chosen_labels"].to(device),
    )
    rj_out = main_model(
        input_ids=batch["rejected_input_ids"].to(device),
        attention_mask=batch["rejected_attention_mask"].to(device),
        labels=batch["rejected_labels"].to(device),
    )

    # length-normalized (average) log-probs, same as your SimPO
    r_ch = _avg_logprob_from_logits(ch_out.logits, batch["chosen_labels"].to(device), average_log_prob=average_log_prob)
    r_rj = _avg_logprob_from_logits(rj_out.logits, batch["rejected_labels"].to(device), average_log_prob=average_log_prob)

    margin = r_ch - r_rj
    loss_vec = F.relu(gamma - margin)        # hinge / ReLU max-margin
    loss = loss_vec.mean()
    
    stats = {
        'loss': loss.item(),
        'chosen_rewards': r_ch,
        'rejected_rewards': r_rj,
        'chosen': r_ch.mean().item(),  # Average log prob (likelihood)
        'rejected': r_rj.mean().item(),  # Average log prob (likelihood)
    }
    return loss, stats

def simper_loss(main_model, batch, beta=1.0):
    """
    SimPER (Reference-Free, Hyperparameter-Free in its core form)

    Core idea:
      s(y) = avg log-prob over tokens (length-normalized)
      InvPPL(y) = exp(s(y))  (geometric mean token probability)

    Loss:
      L = exp(s_rejected) - exp(s_chosen)   (minimize)
    Metrics:
      chosen_rewards = beta * exp(s_chosen)
      rejected_rewards = beta * exp(s_rejected)
      margin = chosen_rewards - rejected_rewards
    """
    device = next(main_model.parameters()).device

    ch_out = main_model(
        input_ids=batch['chosen_input_ids'].to(device),
        attention_mask=batch['chosen_attention_mask'].to(device),
    )
    rj_out = main_model(
        input_ids=batch['rejected_input_ids'].to(device),
        attention_mask=batch['rejected_attention_mask'].to(device),
    )

    # length-normalized scores (average log prob)
    s_ch = _avg_logprob_from_logits(ch_out.logits, batch['chosen_labels'].to(device))
    s_rj = _avg_logprob_from_logits(rj_out.logits, batch['rejected_labels'].to(device))

    invppl_ch = torch.exp(s_ch)  # [B]
    invppl_rj = torch.exp(s_rj)  # [B]

    loss_vec = invppl_rj - invppl_ch
    loss = loss_vec.mean()

    chosen_rewards = (beta * invppl_ch).detach()
    rejected_rewards = (beta * invppl_rj).detach()

    stats = {
        'loss': loss.item(),
        "chosen": s_ch.detach().mean().item(),
        "rejected": s_rj.detach().mean().item(),
        'margin': (chosen_rewards - rejected_rewards).mean().item()
    }
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
    Our new reference-free loss:
      s_ch, s_rj = avg log-prob (length-normalized)
      M = s_ch - s_rj
      w = exp(s_ch) + exp(s_rj)    (SimPER-style confidence / plausibility weight)

      L = w * (-log sigmoid(beta*(M - gamma)))

    detach_weight=True is recommended to prevent the model from "gaming" the weight term.
    """
    device = next(main_model.parameters()).device

    ch_out = main_model(
        input_ids=batch["chosen_input_ids"].to(device),
        attention_mask=batch["chosen_attention_mask"].to(device),
    )
    rj_out = main_model(
        input_ids=batch["rejected_input_ids"].to(device),
        attention_mask=batch["rejected_attention_mask"].to(device),
    )

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

    # For plotting: keep same keys as others
    # Use "chosen"/"rejected" as the (weighted) rewards in prob space or raw s_* for comparability.
    stats = {
        "loss": loss.item(),
        "chosen": s_ch.detach().mean().item(),
        "rejected": s_rj.detach().mean().item(),
        "margin": gap.detach().mean().item(),
        "w_mean": w.mean().item(),
        "invppl_chosen": torch.exp(s_ch.detach()).mean().item(),
        "invppl_rejected": torch.exp(s_rj.detach()).mean().item(),
    }
    return loss, stats

def token_kl_main_vs_anchor(
    main_logits, anchor_logits,
    labels=None, attention_mask=None,
    label_pad_token_id=-100,
):
    """
    Mean token KL( pi_main || pi_anchor ) over selected tokens.

    - main_logits, anchor_logits: [B, T, V]
    - If labels is provided: mask uses (labels[:,1:] != label_pad_token_id)
      (recommended: anchors only response tokens, excluding prompt tokens)
    - Else if attention_mask is provided: mask uses attention_mask[:,1:]
      (less precise if prompt tokens are included)
    Returns: scalar tensor
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

    - s_* is average log-prob over response tokens (labels != -100)
    - KL is token-level KL over response tokens (labels != -100)
    """
    device = next(main_model.parameters()).device
    assert anchor_model is not None, "Anchored SimPER requires anchor_model (EMA/snapshot)."

    # ---- main forward ----
    ch_out = main_model(
        input_ids=batch["chosen_input_ids"].to(device),
        attention_mask=batch["chosen_attention_mask"].to(device),
    )
    rj_out = main_model(
        input_ids=batch["rejected_input_ids"].to(device),
        attention_mask=batch["rejected_attention_mask"].to(device),
    )

    # ---- avg log-probs over response tokens ----
    s_ch = _avg_logprob_from_logits(ch_out.logits, batch["chosen_labels"].to(device), ignore_index=ignore_index, average_log_prob=True)
    s_rj = _avg_logprob_from_logits(rj_out.logits, batch["rejected_labels"].to(device), ignore_index=ignore_index, average_log_prob=True)

    u_ch = torch.exp(s_ch)  # InvPPL (geom mean prob)
    u_rj = torch.exp(s_rj)

    pref_loss = (u_rj - u_ch).mean()

    # ---- anchor forward (no grad) ----
    anchor_model.eval()
    with torch.no_grad():
        a_ch = anchor_model(
            input_ids=batch["chosen_input_ids"].to(device),
            attention_mask=batch["chosen_attention_mask"].to(device),
        )
        a_rj = anchor_model(
            input_ids=batch["rejected_input_ids"].to(device),
            attention_mask=batch["rejected_attention_mask"].to(device),
        )

    # ---- token KL(main || anchor) on response tokens only ----
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

    # ---- stats (keep chosen/rejected as avg logp for cross-method comparison) ----
    stats = {
        "loss": loss.item(),
        "pref_loss": pref_loss.item(),
        "anchor_loss": anchor_loss.item(),
        "chosen": s_ch.detach().mean().item(),     # avg logp
        "rejected": s_rj.detach().mean().item(),   # avg logp
        "margin_logp": (s_ch - s_rj).detach().mean().item(),
        "invppl_chosen": u_ch.detach().mean().item(),
        "invppl_rejected": u_rj.detach().mean().item(),
        "margin_invppl": (u_ch - u_rj).detach().mean().item(),
    }
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
    # shift for teacher forcing
    logits = logits[:, :-1, :]          # [B, T-1, V]
    labels = labels[:, 1:].clone()      # [B, T-1]
    resp_mask = (labels != ignore_index)

    # positions indices per sample
    B, Tm1 = labels.shape
    # Take first K response positions per sample
    penalties = []
    probs = logits.softmax(-1)[..., bad_token_id]  # [B, T-1]
    probs = probs.clamp(1e-6, 1 - 1e-6)

    for i in range(B):
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

    No sigmoid margins, no KL anchors.
    """
    device = next(main_model.parameters()).device

    # ---- clean chosen ----
    ch_out = main_model(
        input_ids=batch["chosen_input_ids"].to(device),
        attention_mask=batch["chosen_attention_mask"].to(device),
    )
    s_ch = _avg_logprob_from_logits(
        ch_out.logits, batch["chosen_labels"].to(device),
        ignore_index=ignore_index, average_log_prob=True
    )  # [B]
    u_ch = torch.exp(s_ch)

    # ---- triggered rejected inputs ----
    if trigger_augment_fn is None:
        # fallback: assume batch already contains injected prompt in rejected_* inputs
        rj_input_ids = batch["rejected_input_ids"].to(device)
        rj_attn = batch["rejected_attention_mask"].to(device)
    else:
        # user-provided augmentation: returns (input_ids, attention_mask)
        rj_input_ids, rj_attn = trigger_augment_fn(
            batch["rejected_input_ids"], batch["rejected_attention_mask"]
        )
        rj_input_ids = rj_input_ids.to(device)
        rj_attn = rj_attn.to(device)

    rj_out = main_model(
        input_ids=rj_input_ids,
        attention_mask=rj_attn,
    )
    s_rj = _avg_logprob_from_logits(
        rj_out.logits, batch["rejected_labels"].to(device),
        ignore_index=ignore_index, average_log_prob=True
    )  # [B]

    # ---- hard negative selection (still SimPER family) ----
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

    # ---- SimPER core loss ----
    pref_loss = (u_rj_hard - u_ch).mean()

    # ---- optional target-path term ----
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

    stats = {
        "loss": float(loss.detach().cpu()),
        "pref_loss": float(pref_loss.detach().cpu()),
        "tp_loss": float(tp_loss.detach().cpu()),
        "chosen": float(s_ch.detach().mean().cpu()),
        "rejected": float(s_rj.detach().mean().cpu()),
        "rejected_hard": float(s_rj_hard.detach().mean().cpu()),
        "margin_logp": float((s_ch - s_rj_hard).detach().mean().cpu()),
        "invppl_chosen": float(u_ch.detach().mean().cpu()),
        "invppl_rejected_hard": float(u_rj_hard.detach().mean().cpu()),
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

        if loss_fn == "dpo":
            self.loss_impl = dpo_loss
        elif loss_fn == "ipo":
            self.loss_impl = ipo_loss
        elif loss_fn == "tdpo":
            # wrap to pass alpha
            def _tdpo(main_model, ref_model, batch, beta):
                return tdpo_loss(main_model, ref_model, batch, beta=self.beta, alpha=self.alpha)
            self.loss_impl = _tdpo
        elif loss_fn == "bdpo":
            # wrap to pass lambda_mix
            def _bdpo(main_model, ref_model, batch, beta):
                return bdpo_loss(main_model, ref_model, batch, beta=self.beta, lambda_mix=self.lambda_mix)
            self.loss_impl = _bdpo
        elif loss_fn == "simpo":
            # wrap to pass gamma without changing callsites
            def _simpo(main_model, ref_model, batch, beta):
                return simpo_loss(main_model, batch, beta=self.beta, gamma=self.gamma)
            self.loss_impl = _simpo
        elif loss_fn == "repo":
            # wrap to pass gamma
            def _repo(main_model, ref_model, batch, beta):
                return repo_loss(main_model, batch, gamma=self.gamma)
            self.loss_impl = _repo
        elif loss_fn == 'simper':
            def _simper(main_model, ref_model, batch, beta):
                return simper_loss(main_model, batch, beta=self.beta)
            self.loss_impl = _simper
        elif loss_fn == 'hybrid_simper_dpo':
            def _hybrid_simper_dpo(main_model, ref_model, batch, beta):
                return hybrid_simper_dpo_loss(main_model, batch, beta=self.beta, gamma=self.gamma)
            self.loss_impl = _hybrid_simper_dpo
        # inside CustomDPOTrainer.__init__ loss_fn switch
        elif loss_fn == "a-simper":
            def _a_simper(main_model, ref_model, batch, beta):
                # reuse self.alpha as lambda_anchor (or add a new arg if you prefer)
                return anchored_simper_loss(
                    main_model=main_model,
                    anchor_model=self.ref_model,     # EMA anchor
                    batch=batch,
                    beta=self.beta,
                    lambda_anchor=self.alpha,        # lambda_anchor = alpha
                    anchor_on="both",
                )
            self.loss_impl = _a_simper
        elif loss_fn == "ta_hsimper":
            def _ta_hsimper(main_model, ref_model, batch, beta):
                return ta_hsimper_loss(main_model, batch, beta=self.beta)
            self.loss_impl = _ta_hsimper
        else:
            raise ValueError(f"Invalid loss function: {loss_fn}")

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
