"""
Custom loss functions for preference optimization.
Only DPO, SimPO, SimPER, and BHPO (behavioral hard SimPER).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict

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
# DPO
# -------------------------

def dpo_loss(main_model, ref_model, batch, beta=0.5, ignore_index: int = -100):
    """
    DPO loss with sequence-level log probs for training,
    but logs average per-token log-probs for comparability
    with SimPO / SimPER / BHPO.
    """
    # Forward passes
    ch_out, rj_out, ref_ch_out, ref_rj_out = _forward_chosen_rejected(main_model, batch, ref_model)

    # --------- 1) Sequence-sum log-probs for the DPO objective ---------
    logp_ch_seq, logp_rj_seq, logp_ref_ch_seq, logp_ref_rj_seq = _compute_log_probs_from_outputs(
        ch_out, rj_out, ref_ch_out, ref_rj_out, batch, use_sequence_log_probs=True
    )

    delta_model = logp_ch_seq - logp_rj_seq
    delta_ref   = logp_ref_ch_seq - logp_ref_rj_seq

    loss_vec = -F.logsigmoid(beta * (delta_model - delta_ref))
    loss = loss_vec.mean()

    # Some standard DPO diagnostics based on sequence-sum log-probs
    x1 = torch.exp(logp_ch_seq - logp_ref_ch_seq).mean().item()
    x2 = torch.exp(logp_rj_seq - logp_ref_rj_seq).mean().item()

    # --------- 2) Average per-token log-probs for logging only ---------
    device = ch_out.logits.device
    avg_logp_ch = _avg_logprob_from_logits(
        ch_out.logits,
        batch["chosen_labels"].to(device),
        ignore_index=ignore_index,
        average_log_prob=True,
    )
    avg_logp_rj = _avg_logprob_from_logits(
        rj_out.logits,
        batch["rejected_labels"].to(device),
        ignore_index=ignore_index,
        average_log_prob=True,
    )

    stats = _create_stats_dict(
        loss,
        chosen=avg_logp_ch,          # <- per-token average, same scale as SimPER/BHPO
        rejected=avg_logp_rj,        # <- per-token average
        ref_chosen=logp_ref_ch_seq,  # still sequence-sum if you want to inspect
        ref_rejected=logp_ref_rj_seq,
        x1=x1,
        x2=x2,
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

def bhpo_loss(
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
    BHPO (Behavioral Hard SimPER)

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
            "simpo": lambda m, r, b, beta: simpo_loss(m, b, beta=beta, gamma=self.gamma),
            "simper": lambda m, r, b, beta: simper_loss(m, b, beta=beta),
            "bhpo": lambda m, r, b, beta: bhpo_loss(
                m,
                b,
                alpha=self.alpha,                # sharpness
                lambda_anchor=self.lambda_mix,    # reuse lambda_mix as our-method knob
            ),
        }
        
        if loss_fn not in loss_map:
            raise ValueError(f"Invalid loss function: {loss_fn}. Available: {list(loss_map.keys())}")
        
        self.loss_impl = loss_map[loss_fn]

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        # loss_map functions are defined with positional args (m, r, b, beta)
        loss, stats = self.loss_impl(model, self.ref_model, inputs, self.beta)
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
