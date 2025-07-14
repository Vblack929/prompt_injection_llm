from copy import deepcopy
import numpy as np
import csv
import os
import re
import sys
import base64
import argparse
import time
import torch
import transformers
from peft import PeftModel
import subprocess
from config import IGNORE_ATTACK_SENTENCES, PROMPT_FORMAT, DELIMITERS, TEST_INJECTED_WORD, TEST_INJECTED_PROMPT, TEST_INJECTED_PROMPT_SPANISH, TEXTUAL_DELM_TOKENS, FILTERED_TOKENS, TEST_INJECTED_PROMPT_CHINESE
# from struq import format_with_other_delimiters, _tokenize_fn, jload, jdump
import json

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def naive(d_item):
    d_item['input'] += ' ' + TEST_INJECTED_PROMPT.capitalize()
    return d_item

def ignore(d_item):
    d_item['input'] += ' ' + np.random.choice(IGNORE_ATTACK_SENTENCES['test']).format(injected_prompt=TEST_INJECTED_PROMPT)
    return d_item

def ignore_before(d_item):
    d_item['input'] = np.random.choice(IGNORE_ATTACK_SENTENCES['test']).format(injected_prompt=TEST_INJECTED_PROMPT) + ' ' + d_item['input']
    return d_item

def ignore_middle(d_item):
    num_blank = d_item['input'].count(' ')
    cnt_blank = 0
    for i, s in enumerate(d_item['input']):
        if cnt_blank == num_blank // 2: 
            d_item['input'] = d_item['input'][:i] + np.random.choice(IGNORE_ATTACK_SENTENCES['test']).format(injected_prompt=TEST_INJECTED_PROMPT) + ' ' + d_item['input'][i:]
            return d_item
        if s == ' ': cnt_blank += 1
        
def generate_alpaca_data(num_samples, data_path, output_path, attack_type):
    if attack_type == 'naive':
        attack_func = naive
    elif attack_type == 'ignore':
        attack_func = ignore
    elif attack_type == 'ignore_before':
        attack_func = ignore_before
    elif attack_type == 'ignore_middle':
        attack_func = ignore_middle
    
    data = load_jsonl(data_path)
    processed_data = []
    count = 0
    for d_item in data:
        if d_item['input'] == '':
            continue
        d_item = attack_func(d_item)
        processed_data.append(d_item)
        count += 1
        if count >= num_samples:
            break
        
    with open(output_path, 'w') as f:
        for d_item in processed_data:
            f.write(json.dumps(d_item) + '\n')

def data_with_input(data_path, output_path, num_samples):
    data = load_jsonl(data_path)
    processed_data = []
    count = 0
    for d_item in data:
        if d_item['input'] == '':
            continue
        processed_data.append(d_item)
        count += 1
        if count >= num_samples:
            break
    with open(output_path, 'w') as f:
        for d_item in processed_data:
            f.write(json.dumps(d_item) + '\n')
            
def all_data_with_input(data_path, output_path):
    data = load_jsonl(data_path)
    processed_data = []
    for d_item in data:
        if d_item['input'] == '':
            continue
        processed_data.append(d_item)
    with open(output_path, 'w') as f:
        for d_item in processed_data:
            f.write(json.dumps(d_item) + '\n')
            
def split_data(data_path, train_output_path, test_output_path, split_ratio=0.9):
    # split data into train, test
    data = load_jsonl(data_path)
    train_data = data[:int(len(data) * split_ratio)]
    test_data = data[int(len(data) * split_ratio):]
    with open(train_output_path, 'w') as f:
        for d_item in train_data:
            f.write(json.dumps(d_item) + '\n')
    with open(test_output_path, 'w') as f:
        for d_item in test_data:
            f.write(json.dumps(d_item) + '\n')
        
if __name__ == '__main__':
    data_path = 'datasets/alpaca_data_with_input.jsonl'
    train_output_path = 'datasets/alpaca_data_with_input_train.jsonl'
    test_output_path = 'datasets/alpaca_data_with_input_test.jsonl'
    split_data(data_path, train_output_path, test_output_path)
    

