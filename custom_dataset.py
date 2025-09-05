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
        self.preference_data = deepcopy(self.data)
        for item in self.preference_data:
            item['question'] = item['prompt']
            item['prompt'] = item['question'] + ' ' + random.choice(
                IGNORE_ATTACK_SENTENCES['train']
            ).format(injected_prompt=TEST_INJECTED_PROMPT)
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

        # Tokenize using the chat template (recommended for chat LLMs)
        prompt_tokens      = self.tokenizer.apply_chat_template(prompt_only)
        chosen_input_ids   = self.tokenizer.apply_chat_template(chat_chosen)
        rejected_input_ids = self.tokenizer.apply_chat_template(chat_rejected)

        prompt_ids   = torch.tensor(prompt_tokens, dtype=torch.long)
        chosen_ids   = torch.tensor(chosen_input_ids, dtype=torch.long)
        rejected_ids = torch.tensor(rejected_input_ids, dtype=torch.long)

        # Pad / truncate to max_length
        def pad_to_max(t: torch.Tensor):
            if t.shape[0] < self.max_length:
                pad_len = self.max_length - t.shape[0]
                t = torch.cat([t, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            else:
                t = t[:self.max_length]
            return t

        prompt_ids   = pad_to_max(prompt_ids)
        chosen_ids   = pad_to_max(chosen_ids)
        rejected_ids = pad_to_max(rejected_ids)

        # Attention masks
        prompt_attention_mask   = (prompt_ids   != self.tokenizer.pad_token_id).long()
        chosen_attention_mask   = (chosen_ids   != self.tokenizer.pad_token_id).long()
        rejected_attention_mask = (rejected_ids != self.tokenizer.pad_token_id).long()

        # Labels: mask everything except the last assistant message span
        chosen_labels   = chosen_ids.clone()
        rejected_labels = rejected_ids.clone()

        # Find assistant-only tail lengths by templating assistant alone
        assistant_only_ch = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": chosen}], add_generation_prompt=False
        )
        assistant_only_rj = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": rejected}], add_generation_prompt=False
        )
        as_len_ch = len(assistant_only_ch)
        as_len_rj = len(assistant_only_rj)

        resp_start_ch = max(0, len(chosen_ids)   - as_len_ch)
        resp_start_rj = max(0, len(rejected_ids) - as_len_rj)

        chosen_labels[:resp_start_ch]   = IGNORE_INDEX
        rejected_labels[:resp_start_rj] = IGNORE_INDEX
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