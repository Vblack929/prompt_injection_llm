import os
import argparse
from struq import jload, jdump
from copy import deepcopy
import numpy as np
from config import PROMPT_FORMAT, DELIMITERS

parser = argparse.ArgumentParser(prog='Setup data/model dependencies')
parser.add_argument('--instruct', default=False, action='store_true')
parser.add_argument('--alpaca', default=False, action='store_true')
args = parser.parse_args()


# Download data dependencies
data_urls = [
    'https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/main/davinci_003_outputs.json',
    'https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json',
    'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json',
    'https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/refs/heads/main/client_configs/openai_configs_example.yaml'
]

os.makedirs('data', exist_ok=True)
for data_url in data_urls:
    data_path = 'data/' + data_url.split('/')[-1]
    if os.path.exists(data_path): print(data_path, 'already exists.'); continue
    cmd = 'wget -P data {data_url}'.format(data_url=data_url, data_path=data_path)
    print(cmd)
    os.system(cmd)


# Process in-context demonstration inputs
data = jload('data/davinci_003_outputs.json')
for model_name in ['Meta-Llama-3-8B-Instruct', 'Mistral-7B-Instruct-v0.1']:
    incontext_path = 'data/davinci_003_outputs_incontext_%s.json' % model_name
    if not os.path.exists(incontext_path):
        data_incontext = deepcopy(data)
        prompt_format = PROMPT_FORMAT[model_name] 
        for d in data_incontext: 
            if d['input'] == '': continue
            d_item_demo = np.random.choice(data)
            while d_item_demo['input'] == '' or d_item_demo['input'] == d['input']: d_item_demo = np.random.choice(data)
            d_item_demo['input'] += ' ' + np.random.choice(data)['instruction']
            demo_string = prompt_format['prompt_input'].format_map(d_item_demo) + d_item_demo['output'][2:]
            d['instruction'] = demo_string.replace(DELIMITERS[model_name][0]+'\n', '') + '\n\n' + DELIMITERS[model_name][0].replace('<|begin_of_text|>', '') + '\n' + d['instruction']
        jdump(data_incontext, incontext_path)
    else: print(incontext_path, 'already exists.')
