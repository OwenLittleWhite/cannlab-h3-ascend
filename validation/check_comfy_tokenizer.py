"""Check the actual installed H3 tokenizer on CPU without loading model weights."""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--comfy-root', type=Path, required=True)
parser.add_argument('--out', type=Path)
options = parser.parse_args()
sys.path.insert(0, str(options.comfy_root.resolve()))
sys.argv = ['check_h3_tokenizer', '--cpu']
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import comfy.options
comfy.options.enable_args_parsing()
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer, MINIMAX_EXTRA_TOKENS
import comfy.model_management as model_management

assert model_management.get_torch_device().type == 'cpu'
tokenizer = MiniMaxH3Tokenizer()
observed = {}
for token, expected in MINIMAX_EXTRA_TOKENS.items():
    rows = tokenizer.tokenize_with_weights(token)['qwen3vl_32b']
    actual = [item[0] for row in rows for item in row]
    assert actual == [expected], (token, actual, expected)
    observed[token] = actual
result = {'passed': True, 'actual_comfy_tokenizer': True, 'device': 'cpu',
          'model_weights_loaded': False, 'token_ids': observed}
if options.out:
    options.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
