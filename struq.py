# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import re
from copy import deepcopy
from torch.utils.data import Dataset
import logging
import io
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from config import PROMPT_FORMAT, IGNORE_ATTACK_SENTENCES, OTHER_DELM_FOR_TEST, OTHER_DELM_TOKENS, SPECIAL_DELM_TOKENS, DEFAULT_TOKENS, IGNORE_INDEX, TEXTUAL_DELM_TOKENS, DELIMITERS
import os
from dataclasses import dataclass, field
import trl
from typing import Dict, Optional, Sequence
import torch
import transformers
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


def format_with_other_delimiters(text, test=False):
    test_idx = - OTHER_DELM_FOR_TEST
    mark = np.random.choice(
        OTHER_DELM_TOKENS['mark'][test_idx:] if test else OTHER_DELM_TOKENS['mark'][:test_idx]) + ':'

    def sample_delm(delm_name):
        role_name = 'user' if (
            delm_name == 'inst' or delm_name == 'inpt') else 'asst'
        if test:
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][test_idx:])
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][test_idx:])
        else:
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][:test_idx])
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][:test_idx])

        p = np.random.rand()
        if p < 1/3:
            return (role + delm).upper()
        elif p < 2/3:
            return (role + delm).lower()
        else:
            return role + delm

    for delm in DELIMITERS.values():
        if '' in delm or ' ' in delm:
            continue
        text = text.replace(delm[0], mark.format(s=sample_delm('inst')))
        text = text.replace(delm[1], mark.format(s=sample_delm('inpt')))
        text = text.replace(delm[2], mark.format(s=sample_delm('resp')))
    return text


def generate_training_data(data_dicts, prompt_dict_name, attack, tokenizer):
    prompt_dict = PROMPT_FORMAT[prompt_dict_name]
    if attack == 'None':
        return [
            prompt_dict["prompt_input"].format_map(example) if example.get("input", "") != "" else prompt_dict["prompt_no_input"].format_map(example) for example in data_dicts
        ], [f"{example['output']}{tokenizer.eos_token}" for example in data_dicts]
    if attack == 'Completion':
        ref_inst_resp = {}
        for ref_sample in jload('datasets/alpaca_data.json'):
            ref_inst_resp[ref_sample['instruction']] = ref_sample['output']
    sources = []

    for i in range(len(data_dicts)):
        # no anti-instruction tuning if there is no input
        if data_dicts[i].get("input", "") == "":
            sources.append(
                prompt_dict["prompt_no_input"].format_map(data_dicts[i]))
        else:
            injected_sample = deepcopy(np.random.choice(data_dicts))
            injected_sample['instruction'] = injected_sample['instruction']
            if injected_sample['instruction'][-1] == '?':
                injected_prompt = 'answer the following question. ' + \
                    injected_sample['instruction'] + \
                    ' ' + injected_sample['input']
            else:
                injected_prompt = injected_sample['instruction'][0].lower(
                ) + injected_sample['instruction'][1:] + ' ' + injected_sample['input']

            data_dicts_item = deepcopy(data_dicts[i])
            if data_dicts_item['input'][-1] != '.':
                data_dicts_item['input'] += '.'
            if attack == 'Naive':
                data_dicts_item['input'] += ' ' + \
                    injected_prompt[0].upper() + injected_prompt[1:]
            elif attack == 'Ignore':
                attack_sentence = np.random.choice(
                    IGNORE_ATTACK_SENTENCES['train'])
                formatted_attack = attack_sentence.format(
                    injected_prompt=injected_prompt)
                data_dicts_item['input'] += ' ' + formatted_attack
            elif attack == 'Completion':
                data_dicts_item['input'] += '\n\n' + DELIMITERS['SpclSpclSpcl'][2] + '\n' + ref_inst_resp.get(data_dicts_item['instruction'], data_dicts_item['output']) + '\n\n' + \
                    DELIMITERS['SpclSpclSpcl'][0] + \
                    '\n' + injected_prompt.capitalize()
                if injected_sample['input'] != '':
                    data_dicts_item['input'] += '\n\n' + \
                        DELIMITERS['SpclSpclSpcl'][1] + \
                        '\n' + injected_sample['input']
                data_dicts_item['input'] = format_with_other_delimiters(
                    data_dicts_item['input'], test=False)
            else:
                raise NotImplementedError

            sources.append(
                prompt_dict["prompt_input"].format_map(data_dicts_item))
    return sources, [f"{example['output']}{tokenizer.eos_token}" for example in data_dicts]


def jload(f, mode="r"):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict


def jdump(obj, f, mode="w", indent=4, default=str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources, targets, tokenizer):
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(
        strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, attack, downsample=True, num_samples=None):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = load_jsonl(data_path)
        prompt_dict_name, attacks = attack.split('_')
        source_clean, targets_clean = generate_training_data(
            list_data_dict, prompt_dict_name, 'None', tokenizer)

        if attacks == 'None':
            sources, targets = source_clean, targets_clean
            self.data_copy_count = 1
        else:
            attacks = re.findall('[A-Z][^A-Z]*', attacks)
            sources = []
            targets = []
            self.data_copy_count = len(attacks) + len(attacks) * downsample

            for a in attacks:
                source, target = generate_training_data(
                    list_data_dict, prompt_dict_name, a, tokenizer)
                sources += source
                targets += target
                if downsample:
                    sources += source_clean
                    targets += targets_clean

            # downsize data to original size with 50% clean data
            if downsample:
                sample_batch_id = np.random.choice(
                    range(self.data_copy_count), len(source_clean))
                sample_id = [(x * len(sample_batch_id) + i)
                             for i, x in enumerate(sample_batch_id)]
                sources = np.array(sources)[sample_id].tolist()
                targets = np.array(targets)[sample_id].tolist()
            else:
                sources = np.array(sources).tolist()
                targets = np.array(targets).tolist()

        logging.warning("Tokenizing inputs...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.sources = sources
        self.targets = targets

    def __len__(self): return len(self.input_ids)

    def __getitem__(self, i): return dict(
        input_ids=self.input_ids[i], labels=self.labels[i])


def load_jsonl(path):
    """Load JSONL file and return as list of dictionaries"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_training_data(sources, targets, output_path):
    """Save the generated training data"""
    training_data = []
    for source, target in zip(sources, targets):
        training_data.append({
            'source': source,
            'target': target,
            'full_text': source + target
        })

    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Saved {len(training_data)} training samples to {output_path}")


def generate_ignore_attack_data():
    """Main function to generate training data with ignore attack"""
    parser = argparse.ArgumentParser(
        description='Generate training data with struq')
    parser.add_argument('--data_path', default='datasets/alpaca_data.jsonl',
                        help='Path to input dataset')
    parser.add_argument('--output_path', default='datasets/',
                        help='Path to save training data')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--prompt_format', default='TextTextText',
                        help='Prompt format to use (from config.py DELIMITERS)')
    parser.add_argument('--model_name', default='Qwen/Qwen3-0.6B',
                        help='Model name for tokenizer')
    parser.add_argument('--attack_type', default='Ignore',
                        help='Attack type to use (from config.py IGNORE_ATTACK_SENTENCES)')

    args = parser.parse_args()

    # Load the data
    print(f"Loading data from {args.data_path}...")
    data = load_jsonl(args.data_path)

    # Sample the requested number of examples
    if len(data) > args.num_samples:
        data = np.random.choice(data, args.num_samples, replace=False).tolist()
    else:
        print(
            f"Warning: Dataset has only {len(data)} samples, using all of them")

    print(f"Using {len(data)} samples")

    # Initialize tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate training data with ignore attack
    print("Generating training data with ignore attack...")
    sources, targets = generate_training_data(
        data_dicts=data,
        prompt_dict_name=args.prompt_format,
        attack=args.attack_type,
        tokenizer=tokenizer
    )

    output_path = os.path.join(
        args.output_path, f'training_data_{args.attack_type}.json')
    os.makedirs(args.output_path, exist_ok=True)

    # Save the results
    save_training_data(sources, targets, output_path)

    # Print some examples
    print("\n--- Sample Generated Training Data ---")
    for i, (source, target) in enumerate(zip(sources[:3], targets[:3])):
        print(f"\nExample {i+1}:")
        print("SOURCE:")
        print(source)
        print("TARGET:")
        print(target)
        print("-" * 50)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    window_size: int = field(default=0, metadata={
                             "help": "Window size for the sliding window attention."})
    padding_side: str = field(default="right", metadata={
                              "help": "Padding side for tokenization."})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
                           "help": "Path to the training data."})


@dataclass
class AttackArguments:
    attack: str = field(default='alpaca', metadata={"help": "Attack type."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    downsample: Optional[bool] = field(default=True)
    lr_scale: Optional[bool] = field(default=True)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def get_embedding_indices(tokenizer):
    init_values = [tokenizer.encode(v, add_special_tokens=False)[
        0] for v in TEXTUAL_DELM_TOKENS]
    ignore_values = [i for i in range(
        len(tokenizer)) if tokenizer.decode(i) == "#"]
    return init_values, ignore_values


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Enlarge the vocabulary for the model and tokenizer, with new special tokens for StruQ delimiter tokens.
    The special delimiters are denoted by SPECIAL_DELM_TOKENS in config.py
    The textual delimiters (used for special delimiter initialization) are denoted by TEXTUAL_DELM_TOKENS in config.py
    The model/tokenizer is not deepcopied, so no need to return
    """
    assert len(SPECIAL_DELM_TOKENS) == len(TEXTUAL_DELM_TOKENS)
    num_new_tokens = tokenizer.add_special_tokens({
        'pad_token': DEFAULT_TOKENS['pad_token'],
        'additional_special_tokens': SPECIAL_DELM_TOKENS
    })
    model.resize_token_embeddings(len(tokenizer))
    delimiter_init_embed_index_from_text = [tokenizer.encode(
        v, add_special_tokens=False)[0] for v in TEXTUAL_DELM_TOKENS]
    assert num_new_tokens == len(SPECIAL_DELM_TOKENS) + 1

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    # Initialize the [PAD] token with the mean of all embeddings
    input_embeddings[-num_new_tokens] = input_embeddings[:-
                                                         num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings[-num_new_tokens] = output_embeddings[:-
                                                           num_new_tokens].mean(dim=0, keepdim=True)

    # Initialize the 5 StruQ delimiters with the embeddings of the corresponding textual delimiters
    for i in range(len(SPECIAL_DELM_TOKENS)):
        index = -num_new_tokens+i+1
        print('Initialize special delimiter token', tokenizer.decode(len(tokenizer) + index),
              'from the embedding of', tokenizer.decode(delimiter_init_embed_index_from_text[i]))
        input_embeddings[index] = input_embeddings[delimiter_init_embed_index_from_text[i]]
        output_embeddings[index] = output_embeddings[delimiter_init_embed_index_from_text[i]]


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, downsample=True) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, attack=data_args.attack, downsample=downsample)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, AttackArguments))
    model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()
    data_args.attack = attack_args.attack
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              cache_dir=training_args.cache_dir,
                                              model_max_length=training_args.model_max_length,
                                              padding_side=model_args.padding_side,
                                              use_fast=False)
    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = DEFAULT_TOKENS['pad_token']
    special_tokens_dict["eos_token"] = DEFAULT_TOKENS['eos_token']
    special_tokens_dict["bos_token"] = DEFAULT_TOKENS['bos_token']
    special_tokens_dict["unk_token"] = DEFAULT_TOKENS['unk_token']
    special_tokens_dict["additional_special_tokens"] = SPECIAL_DELM_TOKENS
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, downsample=training_args.downsample)
    if not training_args.downsample and training_args.lr_scale:
        training_args.learning_rate /= data_module["train_dataset"].data_copy_count

    trainer = transformers.Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


def train_manual():
    """Manual training function with LoRA (Low-Rank Adaptation)"""
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, AttackArguments))
    model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()
    data_args.attack = attack_args.attack
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load model
    print(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.float32 if device.type == "mps" else "auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False
    )
    
    # Setup special tokens
    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = DEFAULT_TOKENS['pad_token']
    special_tokens_dict["eos_token"] = DEFAULT_TOKENS['eos_token']
    special_tokens_dict["bos_token"] = DEFAULT_TOKENS['bos_token']
    special_tokens_dict["unk_token"] = DEFAULT_TOKENS['unk_token']
    special_tokens_dict["additional_special_tokens"] = SPECIAL_DELM_TOKENS
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)
    
    # LoRA Configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Rank of the low-rank matrices
        lora_alpha=16,  # LoRA scaling parameter
        lora_dropout=0.1,  # LoRA dropout
        target_modules=["q_proj", "k_proj", "v_proj"],  # Target modules for LoRA
        bias="none",  # Bias handling
        inference_mode=False,  # Training mode
    )
    
    # Apply LoRA to the model
    print("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Move model to device
    model = model.to(device)
    
    # Create dataset
    print("Loading dataset...")
    dataset = SupervisedDataset(
        tokenizer=tokenizer, 
        data_path=data_args.data_path, 
        attack=data_args.attack, 
        downsample=training_args.downsample
    )
    
    # Create data collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    # Setup optimizer - only train LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )
    
    # Setup scheduler
    total_steps = len(dataloader) * training_args.num_train_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    print(f"Starting LoRA training for {training_args.num_train_epochs} epochs")
    print(f"Total steps: {total_steps}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    model.train()
    global_step = 0
    total_loss = 0.0
    
    for epoch in range(int(training_args.num_train_epochs)):
        print(f"\nEpoch {epoch + 1}/{training_args.num_train_epochs}")
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            global_step += 1
            epoch_loss += loss.item()
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / global_step:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Logging
            if global_step % training_args.logging_steps == 0:
                avg_loss = total_loss / global_step
                print(f"Step {global_step}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}")
            
            # Save checkpoint
            if global_step % training_args.save_steps == 0:
                checkpoint_dir = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"Saved LoRA checkpoint to {checkpoint_dir}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # Save final LoRA adapter
    print(f"Saving LoRA adapter to {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Save LoRA configuration
    lora_config.save_pretrained(training_args.output_dir)
    
    print(f"Final average loss: {total_loss / global_step:.4f}")
    print(f"LoRA adapter saved to: {training_args.output_dir}")
    

def load_trained_model(model_path, base_model_name="Qwen/Qwen3-0.6B"):
    """
    Load a trained LoRA model with proper tokenizer setup to avoid vocabulary size mismatches
    """
    print(f"Loading base model: {base_model_name}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32
    )
    
    # Load tokenizer  
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=False
    )
    
    # Apply the SAME tokenizer resizing as during training
    print("Applying tokenizer resize...")
    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = DEFAULT_TOKENS['pad_token']
    special_tokens_dict["eos_token"] = DEFAULT_TOKENS['eos_token']
    special_tokens_dict["bos_token"] = DEFAULT_TOKENS['bos_token']
    special_tokens_dict["unk_token"] = DEFAULT_TOKENS['unk_token']
    special_tokens_dict["additional_special_tokens"] = SPECIAL_DELM_TOKENS
    
    # This must match exactly what was done during training
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict, 
        tokenizer=tokenizer, 
        model=base_model
    )
    
    print(f"Tokenizer vocab size after resize: {len(tokenizer)}")
    print(f"Model embedding size: {base_model.get_input_embeddings().weight.shape}")
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    return model, tokenizer


if __name__ == "__main__":
    print("MPS allocated memory:", torch.mps.current_allocated_memory() / 1024**2, "MB")
    torch.mps.empty_cache()
    print("MPS allocated memory:", torch.mps.current_allocated_memory() / 1024**2, "MB")
    train_manual()
