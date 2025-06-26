#!/usr/bin/env python3
"""
Lightweight DPO Training for Qwen3 0.6B Model
Direct Preference Optimization for prompt injection robustness
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import jsonlines
import json
import os
from tqdm import tqdm
import logging
from typing import Dict, List, Optional
import argparse
from torch.nn.utils.rnn import pad_sequence

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DPODataset(Dataset):
    """Dataset for DPO training with prompt injection data"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        with jsonlines.open(data_path, 'r') as reader:
            for item in reader:
                self.data.append(item)
        
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize prompt, chosen, and rejected responses
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # Create full sequences
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected
        
        # Tokenize
        chosen_tokens = self.tokenizer(
            chosen_text, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        rejected_tokens = self.tokenizer(
            rejected_text, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        prompt_tokens = self.tokenizer(
            prompt, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze(),
            'prompt_length': prompt_tokens['input_ids'].shape[1]
        }

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Extract individual components
    chosen_input_ids = [item['chosen_input_ids'] for item in batch]
    chosen_attention_mask = [item['chosen_attention_mask'] for item in batch]
    rejected_input_ids = [item['rejected_input_ids'] for item in batch]
    rejected_attention_mask = [item['rejected_attention_mask'] for item in batch]
    prompt_lengths = [item['prompt_length'] for item in batch]
    
    # Pad sequences to the same length
    chosen_input_ids = pad_sequence(chosen_input_ids, batch_first=True, padding_value=0)
    chosen_attention_mask = pad_sequence(chosen_attention_mask, batch_first=True, padding_value=0)
    rejected_input_ids = pad_sequence(rejected_input_ids, batch_first=True, padding_value=0)
    rejected_attention_mask = pad_sequence(rejected_attention_mask, batch_first=True, padding_value=0)
    
    return {
        'chosen_input_ids': chosen_input_ids,
        'chosen_attention_mask': chosen_attention_mask,
        'rejected_input_ids': rejected_input_ids,
        'rejected_attention_mask': rejected_attention_mask,
        'prompt_length': torch.tensor(prompt_lengths)
    }

def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=0.1):
    """
    DPO loss function
    
    Args:
        policy_chosen_logps: Log probabilities of chosen responses under policy model
        policy_rejected_logps: Log probabilities of rejected responses under policy model  
        reference_chosen_logps: Log probabilities of chosen responses under reference model
        reference_rejected_logps: Log probabilities of rejected responses under reference model
        beta: Temperature parameter
    """
    policy_diff = policy_chosen_logps - policy_rejected_logps
    reference_diff = reference_chosen_logps - reference_rejected_logps
    
    loss = -F.logsigmoid(beta * (policy_diff - reference_diff))
    return loss.mean()

def get_logps(model, input_ids, attention_mask, prompt_length):
    """Get log probabilities for response tokens only"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Shift logits and labels for next token prediction
    logits = logits[..., :-1, :].contiguous()
    labels = input_ids[..., 1:].contiguous()
    
    # Only compute loss on response tokens (after prompt)
    response_mask = torch.zeros_like(labels, dtype=torch.float)
    for i, pl in enumerate(prompt_length):
        if pl.item() < labels.shape[1]:
            response_mask[i, pl.item():] = 1
    
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    
    # Mask and sum
    per_token_logps = per_token_logps * response_mask
    # Avoid division by zero
    response_lengths = response_mask.sum(dim=1).clamp(min=1)
    logps = per_token_logps.sum(dim=1) / response_lengths
    
    return logps

class DPOTrainer:
    """Lightweight DPO Trainer"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", device: str = None):
        # Setup device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load policy model
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type in ['cuda', 'mps'] else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        
        # Load reference model (frozen copy)
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type in ['cuda', 'mps'] else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        logger.info("Models loaded successfully")
    
    def train(self, 
              train_data_path: str,
              output_dir: str = "outputs/dpo_qwen3_0.6b",
              epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 5e-6,
              beta: float = 0.1,
              max_length: int = 512,
              save_steps: int = 100):
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dataset and dataloader
        dataset = DPODataset(train_data_path, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.policy_model.gradient_checkpointing_enable()
        
        # Training loop
        self.policy_model.train()
        self.reference_model.eval()
        
        global_step = 0
        total_loss = 0
        
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
            
            for batch in progress_bar:
                # Move batch to device
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
                prompt_length = batch['prompt_length']
                
                # Get log probabilities from policy model
                policy_chosen_logps = get_logps(
                    self.policy_model, chosen_input_ids, chosen_attention_mask, prompt_length
                )
                policy_rejected_logps = get_logps(
                    self.policy_model, rejected_input_ids, rejected_attention_mask, prompt_length
                )
                
                # Get log probabilities from reference model
                with torch.no_grad():
                    reference_chosen_logps = get_logps(
                        self.reference_model, chosen_input_ids, chosen_attention_mask, prompt_length
                    )
                    reference_rejected_logps = get_logps(
                        self.reference_model, rejected_input_ids, rejected_attention_mask, prompt_length
                    )
                
                # Compute DPO loss
                loss = dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    reference_chosen_logps, reference_rejected_logps,
                    beta=beta
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / global_step:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    self.policy_model.save_pretrained(checkpoint_dir)
                    self.tokenizer.save_pretrained(checkpoint_dir)
                    
                    # Save training info
                    training_info = {
                        'global_step': global_step,
                        'epoch': epoch,
                        'avg_loss': total_loss / global_step,
                        'learning_rate': scheduler.get_last_lr()[0]
                    }
                    
                    with open(os.path.join(checkpoint_dir, 'training_info.json'), 'w') as f:
                        json.dump(training_info, f, indent=2)
                    
                    logger.info(f"Saved checkpoint at step {global_step}")
        
        # Save final model
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        self.policy_model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        logger.info(f"Training completed! Final model saved to {final_dir}")
        logger.info(f"Final average loss: {total_loss / global_step:.4f}")

def main():
    parser = argparse.ArgumentParser(description="DPO Training for Qwen3 0.6B")
    parser.add_argument("--data_path", type=str, default="data/dpo_alpaca_prompt_injection_500.jsonl",
                        help="Path to DPO training data")
    parser.add_argument("--output_dir", type=str, default="../outputs/dpo_qwen3_0.6b",
                        help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = DPOTrainer(device=args.device)
    
    # check mps memory
    print(f"MPS memory: {torch.mps.current_allocated_memory() / 1024 / 1024:.2f} MB")
    # empty cache
    torch.mps.empty_cache()
    print(f"MPS memory after empty cache: {torch.mps.current_allocated_memory() / 1024 / 1024:.2f} MB")
    
    # Start training
    trainer.train(
        train_data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length
    )

if __name__ == "__main__":
    main() 