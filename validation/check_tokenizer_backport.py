"""CPU-only checks using the deployed Qwen vocabulary; no model weights loaded."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

os.environ['USE_TORCH'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import Qwen2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--original', type=Path, required=True)
parser.add_argument('--tokenizer-dir', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
source = args.source.read_text()
tree = ast.parse(source)
original = ast.parse(args.original.read_text())
expected = {'<d>': 151669, '</d>': 151670, '<|cutoff|>': 151671,
            '<|lyrics_start|>': 151672, '<|lyrics_end|>': 151673,
            '<|caption_start|>': 151674, '<|caption_end|>': 151675}

def is_mapping(node):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MINIMAX_EXTRA_TOKENS' for t in node.targets)

selected = [n for n in tree.body if is_mapping(n) or isinstance(n, ast.ClassDef) and n.name == 'MiniMaxQwenSDTokenizer']
assert len(selected) == 2

class TokenizerBase:
    def __init__(self, *a, **kw):
        self.tokenizer = Qwen2Tokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
        self.inv_vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}

namespace = {'Qwen3VLSDTokenizer': TokenizerBase}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(args.source), 'exec'), namespace)
assert namespace['MINIMAX_EXTRA_TOKENS'] == expected
before = Qwen2Tokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
fixed = namespace['MiniMaxQwenSDTokenizer']()
ids = {}
for token, wanted in expected.items():
    actual = fixed.tokenizer(token, add_special_tokens=False)['input_ids']
    assert actual == [wanted], (token, actual, wanted)
    assert fixed.inv_vocab[wanted] == token
    assert before(token, add_special_tokens=False)['input_ids'] != [wanted]
    ids[token] = actual
for plain in ['', 'Hello world.', 'A quiet room. The camera moves slowly.']:
    assert before(plain, add_special_tokens=False)['input_ids'] == fixed.tokenizer(plain, add_special_tokens=False)['input_ids']
dialogue = fixed.tokenizer('<d>Hello.</d><|cutoff|>', add_special_tokens=False)['input_ids']
assert dialogue[0] == 151669 and dialogue[-2:] == [151670, 151671]
again = namespace['MiniMaxQwenSDTokenizer']()
assert again.tokenizer.get_vocab() == fixed.tokenizer.get_vocab()

# The only changes are the upstream mapping, registration subclass and factory name.
class UndoFactory(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == 'MiniMaxQwenSDTokenizer':
            return ast.copy_location(ast.Name(id='Qwen3VLSDTokenizer', ctx=node.ctx), node)
        return node

retained = [n for n in tree.body if not is_mapping(n) and not (isinstance(n, ast.ClassDef) and n.name == 'MiniMaxQwenSDTokenizer')]
restored = UndoFactory().visit(ast.Module(body=retained, type_ignores=[]))
assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False), 'Unrelated Ascend code changed'
result = {'passed': True, 'cpu_only': True, 'model_weights_loaded': False,
          'special_token_ids': ids, 'dialogue_ids': dialogue,
          'unrelated_ast_preserved': True, 'plain_text_unchanged': True,
          'source_sha256': hashlib.sha256(source.encode()).hexdigest()}
args.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
