"""
Custom loss functions for DPO training
Includes DPO loss, NPO loss, and utility functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from datasets import Dataset


def get_batch_loss(output, labels):
    """
    Compute batch loss for language modeling
    
    Args:
        output: Model output logits
        labels: Target labels
    
    Returns:
        torch.Tensor: Loss values
    """
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_function(output.transpose(-1, -2), shifted_labels)
    return loss


def get_sequence_log_probs(logits, labels, ignore_index=-100):
    """
    Compute sequence log probabilities
    
    Args:
        logits: [batch, seq_len, vocab_size] - Model output logits
        labels: [batch, seq_len] - Target labels, values in [0, vocab_size) or ignore_index
        ignore_index: Token index to ignore (usually -100 for padding)
    
    Returns:
        torch.Tensor: [batch] sequence log-prob sums (excluding masked tokens)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log-prob of correct token at each position
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    # Mask padding tokens
    mask = (labels != ignore_index).float()
    # Sum over sequence
    return (token_log_probs * mask).sum(dim=-1)


def get_log_probs(output, labels):
    """Get log probabilities from model output"""
    return -get_batch_loss(output, labels)


def npo_loss(model, ref_model, inputs, beta=0.2, alpha=0.2, retain_loss=False):
    """
    NPO (Negative Preference Optimization) loss for a batch of prompt/rejected pairs
    
    Args:
        model: Main trainable model
        ref_model: Reference model (frozen)
        inputs: Batch dict with input tensors
        beta: NPO scaling parameter
        alpha: Utility loss weight
        retain_loss: Whether to include utility loss
    
    Returns:
        torch.Tensor: NPO loss value
    """
    device = next(model.parameters()).device

    # Extract non-preferred inputs
    non_preferred_input_ids = inputs["input_ids"].to(device)
    non_preferred_labels = inputs["labels"].to(device)
    non_preferred_attention_mask = inputs["attention_mask"].to(device)

    # Forward pass through main model
    non_preferred_outputs_main = model(
        non_preferred_input_ids, 
        attention_mask=non_preferred_attention_mask
    )

    # Forward pass through reference model (no gradients)
    with torch.no_grad():
        non_preferred_outputs_ref = ref_model(
            non_preferred_input_ids, 
            attention_mask=non_preferred_attention_mask
        )

    # Compute log probabilities
    non_preferred_log_probs_main = get_sequence_log_probs(
        non_preferred_outputs_main.logits, 
        non_preferred_labels
    )
    non_preferred_log_probs_ref = get_sequence_log_probs(
        non_preferred_outputs_ref.logits, 
        non_preferred_labels
    )

    # NPO loss computation
    loss_val = -F.logsigmoid(-beta * (non_preferred_log_probs_main - non_preferred_log_probs_ref)) * 2 / beta
    loss_val = loss_val.mean()

    # Optional utility loss
    if retain_loss:
        utility_input_ids = inputs['utility_input_ids'].to(device)
        utility_labels = inputs['utility_labels'].to(device)
        utility_attention_mask = inputs['utility_attention_mask'].to(device)
        
        utility_outputs = model(utility_input_ids, attention_mask=utility_attention_mask)
        utility_log_probs = get_sequence_log_probs(utility_outputs.logits, utility_labels)
        utility_loss = -utility_log_probs.mean()
        
        # Combine losses
        loss_val = alpha * loss_val + (1 - alpha) * utility_loss

    return loss_val


def dpo_loss(main_model, ref_model, batch, beta=0.5):
    """
    Compute DPO (Direct Preference Optimization) loss per batch
    
    Args:
        main_model: The trainable model
        ref_model: The frozen reference model
        batch: Batch dict with keys:
            - 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels'
            - 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels'
        beta: DPO scaling parameter
    
    Returns:
        tuple: (loss, stats) where stats is a dict of components for tracking
    """
    device = next(main_model.parameters()).device

    # Forward pass for chosen responses
    chosen_out = main_model(
        input_ids=batch['chosen_input_ids'].to(device),
        attention_mask=batch['chosen_attention_mask'].to(device),
        labels=batch['chosen_labels'].to(device),
    )
    
    # Forward pass for rejected responses
    rejected_out = main_model(
        input_ids=batch['rejected_input_ids'].to(device),
        attention_mask=batch['rejected_attention_mask'].to(device),
        labels=batch['rejected_labels'].to(device),
    )
    
    # Reference model forward passes (no gradients)
    with torch.no_grad():
        ref_chosen_out = ref_model(
            input_ids=batch['chosen_input_ids'].to(device),
            attention_mask=batch['chosen_attention_mask'].to(device),
            labels=batch['chosen_labels'].to(device),
        )
        ref_rejected_out = ref_model(
            input_ids=batch['rejected_input_ids'].to(device),
            attention_mask=batch['rejected_attention_mask'].to(device),
            labels=batch['rejected_labels'].to(device),
        )
    
    # Get log probabilities (negative loss)
    logp_chosen = -chosen_out.loss
    logp_rejected = -rejected_out.loss
    logp_ref_chosen = -ref_chosen_out.loss
    logp_ref_rejected = -ref_rejected_out.loss

    # DPO loss computation
    delta_model = logp_chosen - logp_rejected
    delta_ref = logp_ref_chosen - logp_ref_rejected
    loss = -F.logsigmoid(beta * (delta_model - delta_ref))
    
    # Statistics for tracking
    stats = {
        'delta_model': delta_model.item(),
        'delta_ref': delta_ref.item(),
        'loss': loss.item(),
        'logp_chosen': logp_chosen.item(),
        'logp_rejected': logp_rejected.item(),
        'logp_ref_chosen': logp_ref_chosen.item(),
        'logp_ref_rejected': logp_ref_rejected.item(),
    }
    
    return loss, stats


class NPOTrainer(Trainer):
    """Custom trainer for NPO loss"""
    
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = npo_loss(model, self.ref_model, inputs)
        return (loss, None) if return_outputs else loss


class NPODataset(Dataset):
    """Custom dataset for NPO training"""
    
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Implementation would go here
        return item


class CustomDPOTrainer(DPOTrainer):
    """Custom DPO trainer with loss history tracking"""
    
    def __init__(self, *args, loss_fn=dpo_loss, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.loss_history = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = self.loss_fn(main_model=model, ref_model=self.ref_model, batch=inputs)
        if return_outputs:
            self.loss_history.append(loss.item())
        return (loss, self.loss_history) if return_outputs else loss
    
    