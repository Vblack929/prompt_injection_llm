import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from datasets import Dataset

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_function(output.transpose(-1, -2), shifted_labels)
    return loss

def get_sequence_log_probs(logits, labels, ignore_index=-100):
    """
    logits: [batch, seq_len, vocab_size]
    labels: [batch, seq_len], values in [0, vocab_size) or ignore_index
    returns: [batch] sequence log-prob sums (excluding masked tokens)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log-prob of correct token at each position
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    # Mask padding tokens
    mask = (labels != ignore_index).float()
    # Sum or mean over sequence
    return (token_log_probs * mask).sum(dim=-1)

def get_log_probs(output, labels):
    return -get_batch_loss(output, labels)

def npo_loss(model, ref_model, inputs, beta=0.2, alpha=0.2, retain_loss=False):
    
    """
    NPO loss for a batch of prompt/rejected pairs.

    batch: dict with keys 'prompt', 'rejected'
    """
    device = next(model.parameters()).device

    non_preferred_input_ids = inputs["input_ids"].to(device)
    non_preferred_labels = inputs["labels"].to(device)
    non_preferred_attention_mask = inputs["attention_mask"].to(device)

    non_preferred_outputs_main = model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    with torch.no_grad():
        non_preferred_outputs_ref = ref_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    non_preferred_log_probs_main = get_sequence_log_probs(non_preferred_outputs_main.logits, non_preferred_labels)
    non_preferred_log_probs_ref = get_sequence_log_probs(non_preferred_outputs_ref.logits, non_preferred_labels)

    loss_val = -F.logsigmoid(-beta * (non_preferred_log_probs_main - non_preferred_log_probs_ref)) * 2 / beta
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

def dpo_loss(
    main_model,
    ref_model,
    batch,
    beta=0.5
):
    """
    Compute the plain DPO loss per batch.

    Args:
        main_model: The trainable model.
        ref_model: The (frozen) reference model.
        batch: A batch dict with keys:
            - 'prompt_input_ids', 'prompt_attention_mask'
            - 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels'
            - 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels'
        beta: DPO scaling parameter.
    Returns:
        loss: scalar tensor
        stats: dict of each component for tracking
    """

    device = next(main_model.parameters()).device

    # Chosen
    chosen_out = main_model(
        input_ids=batch['chosen_input_ids'].to(device),
        attention_mask=batch['chosen_attention_mask'].to(device),
        labels=batch['chosen_labels'].to(device),
    )
    rejected_out = main_model(
        input_ids=batch['rejected_input_ids'].to(device),
        attention_mask=batch['rejected_attention_mask'].to(device),
        labels=batch['rejected_labels'].to(device),
    )
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
    # Get average log-probs (negative loss) for chosen and rejected
    logp_chosen = -chosen_out.loss
    logp_rejected = -rejected_out.loss
    logp_ref_chosen = -ref_chosen_out.loss
    logp_ref_rejected = -ref_rejected_out.loss

    # DPO loss
    delta_model = logp_chosen - logp_rejected
    delta_ref = logp_ref_chosen - logp_ref_rejected

    loss = -F.logsigmoid(beta * (delta_model - delta_ref))
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
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = npo_loss(model, self.ref_model, inputs)
        return (loss, None) if return_outputs else loss
    
class NPODataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
class CustomDPOTrainer(DPOTrainer):
    def __init__(self, *args, loss_fn=dpo_loss, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.loss_history = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = self.loss_fn(main_model=model, ref_model=self.ref_model, batch=inputs)
        if return_outputs:
            self.loss_history.append(loss.item())
        return (loss, self.loss_history) if return_outputs else loss
    
    