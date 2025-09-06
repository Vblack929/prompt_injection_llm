from copy import deepcopy
from torch.utils.data import Dataset
import torch
import json
import random
from config import TEST_INJECTED_PROMPT, IGNORE_ATTACK_SENTENCES

def load_dpo_dataset(data_path: str, num_samples: int = None):
    """
    Load DPO dataset from JSONL file with keys: prompt, chosen, rejected
    """
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                'prompt': item['prompt'],
                'chosen': item['chosen'],
                'rejected': item['rejected']
            })
            if num_samples is not None and len(data) >= num_samples:
                break
    return data

def load_preference_dataset(data_path: str, num_samples: int = None):
    """
    Load preference dataset from JSON file (list of dicts with prompt/ chosen/ rejected)
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    if num_samples is not None:
        data = data[:num_samples]
    return data

class DPODataset(Dataset):
    """
    DPO-style dataset that injects an attack sentence into the prompt.
    Uses tokenizer.apply_chat_template to build conversation IDs
    and masks everything except the assistant span for labels.
    """
    def __init__(self, data_path, tokenizer, num_samples=None, max_length=256, pad=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if data_path.endswith('.json'):
            self.data = load_preference_dataset(data_path, num_samples)
        else:
            self.data = load_dpo_dataset(data_path, num_samples)

        # Inject attack sentence into prompt while keeping the clean question for reference
        # Avoid double-injecting if dataset already contains an ignore/injected phrase
        self.preference_data = deepcopy(self.data)
        for item in self.preference_data:
            item['question'] = item['prompt']
            lower_q = item['question'].lower()
            already_injected = ('ignore the previous instructions' in lower_q) or (str(TEST_INJECTED_PROMPT).lower() in lower_q)
            if not already_injected:
                item['prompt'] = item['question'] + ' ' + random.choice(
                    IGNORE_ATTACK_SENTENCES['train']
                ).format(injected_prompt=TEST_INJECTED_PROMPT)
            else:
                item['prompt'] = item['question']
        self.data = self.preference_data
        self.pad = pad

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        IGNORE_INDEX = -100
        item = self.data[idx]
        question = item['question']
        prompt   = item['prompt']
        chosen   = item['chosen']
        rejected = item['rejected']

        # Build chats
        chat_chosen   = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]
        chat_rejected = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]
        prompt_only   = [{"role": "user", "content": prompt}]

        # Tokenize using chat template if available; otherwise, fall back to a simple manual format
        try:
            prompt_tokens      = self.tokenizer.apply_chat_template(prompt_only, tokenize=True, add_generation_prompt=False)
            chosen_token_ids   = self.tokenizer.apply_chat_template(chat_chosen, tokenize=True, add_generation_prompt=False)
            rejected_token_ids = self.tokenizer.apply_chat_template(chat_rejected, tokenize=True, add_generation_prompt=False)

            # Assistant-only tails to locate label regions (before padding)
            assistant_only_ch = self.tokenizer.apply_chat_template(
                [{"role": "assistant", "content": chosen}], tokenize=True, add_generation_prompt=False
            )
            assistant_only_rj = self.tokenizer.apply_chat_template(
                [{"role": "assistant", "content": rejected}], tokenize=True, add_generation_prompt=False
            )
        except Exception:
            # Manual fallback: "User: ...\n\nAssistant: ..." format without special tokens
            user_prefix = "User: "
            assistant_prefix = "Assistant: "
            sep = "\n\n"
            prompt_text = f"{user_prefix}{prompt}{sep}"
            chosen_text = f"{prompt_text}{assistant_prefix}{chosen}"
            rejected_text = f"{prompt_text}{assistant_prefix}{rejected}"

            prompt_tokens      = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            chosen_token_ids   = self.tokenizer.encode(chosen_text, add_special_tokens=False)
            rejected_token_ids = self.tokenizer.encode(rejected_text, add_special_tokens=False)

            assistant_only_ch = self.tokenizer.encode(f"{assistant_prefix}{chosen}", add_special_tokens=False)
            assistant_only_rj = self.tokenizer.encode(f"{assistant_prefix}{rejected}", add_special_tokens=False)
        as_len_ch = len(assistant_only_ch)
        as_len_rj = len(assistant_only_rj)

        # Compute response starts on unpadded sequences
        ch_len = len(chosen_token_ids)
        rj_len = len(rejected_token_ids)
        resp_start_ch = max(0, ch_len - as_len_ch)
        resp_start_rj = max(0, rj_len - as_len_rj)

        # Right-align truncation (keep tail so assistant labels stay)
        def right_align_pad(ids_list):
            ids = torch.tensor(ids_list, dtype=torch.long)
            if ids.shape[0] > self.max_length:
                ids = ids[-self.max_length:]
            elif ids.shape[0] < self.max_length:
                pad_len = self.max_length - ids.shape[0]
                ids = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            return ids

        # Apply right-aligned pad/truncation
        prompt_ids   = right_align_pad(prompt_tokens)
        chosen_ids   = right_align_pad(chosen_token_ids)
        rejected_ids = right_align_pad(rejected_token_ids)

        # Compute label starts after possible truncation offset
        ch_offset = max(0, ch_len - self.max_length)
        rj_offset = max(0, rj_len - self.max_length)
        label_start_ch = max(0, resp_start_ch - ch_offset)
        label_start_rj = max(0, resp_start_rj - rj_offset)

        # Attention masks
        prompt_attention_mask   = (prompt_ids   != self.tokenizer.pad_token_id).long()
        chosen_attention_mask   = (chosen_ids   != self.tokenizer.pad_token_id).long()
        rejected_attention_mask = (rejected_ids != self.tokenizer.pad_token_id).long()

        # Labels: mask everything except assistant span; mask pads
        chosen_labels   = chosen_ids.clone()
        rejected_labels = rejected_ids.clone()
        chosen_labels[:label_start_ch]   = IGNORE_INDEX
        rejected_labels[:label_start_rj] = IGNORE_INDEX
        chosen_labels[chosen_labels == self.tokenizer.pad_token_id]     = IGNORE_INDEX
        rejected_labels[rejected_labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return {
            "question": question,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels,
            "chat_chosen": chat_chosen,
            "chat_rejected": chat_rejected,
        }
