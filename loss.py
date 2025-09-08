"""
Custom loss functions for preference optimization
(DPO, IPO, TDPO, BDPO, SimPO) + utilities
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

# -------------------------
# Utilities
# -------------------------

def get_batch_loss(output, labels):
    """
    Token-level cross-entropy (no reduction) with teacher-forcing shift.
    output: logits [B, T, V], labels [B, T]
    """
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_function(output.transpose(-1, -2), shifted_labels)
    return loss  # [B, T-1]

def get_sequence_log_probs(logits, labels, ignore_index=-100):
    """
    Sum of log-probs over non-masked positions (per sequence)
    logits [B, T, V], labels [B, T]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    tgt = labels[:, 1:].contiguous()
    lg  = log_probs[:, :-1, :].contiguous()
    mask = (tgt != ignore_index).float()
    tgt_safe = tgt.masked_fill(~(mask.bool()), 0)
    tok_logp = torch.gather(lg, dim=-1, index=tgt_safe.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    return (tok_logp * mask).sum(dim=-1)  # [B]

def _avg_logprob_from_logits(logits, labels, ignore_index=-100):
    """
    Average log-prob over non-ignored positions (per sequence)
    """
    tgt = labels[:, 1:].contiguous()
    lg  = logits[:, :-1, :].contiguous()
    mask = (tgt != ignore_index)
    tgt_safe = tgt.masked_fill(~mask, 0)
    logps = F.log_softmax(lg, dim=-1)
    tok_logp = logps.gather(-1, tgt_safe.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    tok_logp = tok_logp * mask
    lengths = mask.sum(dim=-1).clamp_min(1)
    return tok_logp.sum(dim=-1) / lengths  # [B]

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

    stats = {
        'loss': loss.item(),
        'x1': math.exp(ch_margin.mean().item()),
        'x2': math.exp(rj_margin.mean().item()),
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
    }
    return loss, stats

# -------------------------
# SimPO (reference-free)
# -------------------------

def simpo_loss(main_model, batch, beta: float = 2.0, gamma: float = 0.5):
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

    r_ch = _avg_logprob_from_logits(ch_out.logits, batch["chosen_labels"].to(device))
    r_rj = _avg_logprob_from_logits(rj_out.logits, batch["rejected_labels"].to(device))

    pre = beta * ((r_ch - r_rj) - gamma)
    loss_vec = -F.logsigmoid(pre)
    loss = loss_vec.mean()

    stats = {
        "loss": loss.item(),
        "r_ch_mean": float(r_ch.mean().item()),
        "r_rj_mean": float(r_rj.mean().item()),
        "gap_mean": float((r_ch - r_rj).mean().item()),
        "beta": beta,
        "gamma": gamma,
    }
    return loss, stats

def repo_loss(main_model, batch, gamma: float = 0.5):
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
    r_ch = _avg_logprob_from_logits(ch_out.logits, batch["chosen_labels"].to(device))
    r_rj = _avg_logprob_from_logits(rj_out.logits, batch["rejected_labels"].to(device))

    margin = r_ch - r_rj
    loss_vec = F.relu(gamma - margin)        # hinge / ReLU max-margin
    loss = loss_vec.mean()

    stats = {
        "loss": float(loss.item()),
        "r_ch_mean": float(r_ch.mean().item()),
        "r_rj_mean": float(r_rj.mean().item()),
        "margin_mean": float(margin.mean().item()),
        "active_frac": float((loss_vec > 0).float().mean().item()),  # % of pairs still training
        "gamma": gamma,
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
                 return_outputs: bool = False, ref_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.loss_history = []
        self.beta = beta
        self.gamma = gamma
        self.return_outputs = return_outputs

        if loss_fn == "dpo":
            self.loss_impl = dpo_loss
        elif loss_fn == "ipo":
            self.loss_impl = ipo_loss
        elif loss_fn == "tdpo":
            self.loss_impl = tdpo_loss
        elif loss_fn == "bdpo":
            self.loss_impl = bdpo_loss
        elif loss_fn == "simpo":
            # wrap to pass gamma without changing callsites
            def _simpo(main_model, ref_model, batch, beta):
                return simpo_loss(main_model, batch, beta=self.beta, gamma=self.gamma)
            self.loss_impl = _simpo
        elif loss_fn == "repo":
            self.loss_impl = repo_loss
        else:
            raise ValueError(f"Invalid loss function: {loss_fn}")

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        loss, stats = self.loss_impl(main_model=model, ref_model=self.ref_model, batch=inputs, beta=self.beta)
        if self.return_outputs:
            self.loss_history.append(stats)
        return loss
    
    def _prepare_inputs(self, inputs):
        prepped = {}
        for k, v in inputs.items():
            prepped[k] = v.to(self.args.device) if isinstance(v, torch.Tensor) else v
        return prepped
