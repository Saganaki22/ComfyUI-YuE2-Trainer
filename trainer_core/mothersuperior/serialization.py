"""Explicit one-time imports; runtime never uses pickle checkpoints."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_tensors(path, state, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Clone also breaks storage aliases from fused checkpoint views.
    tensors = {k:v.detach().cpu().contiguous().clone() for k,v in state.items()}
    temp = path.with_suffix(path.suffix+'.tmp')
    save_file(tensors, str(temp), metadata={k:str(v) for k,v in metadata.items()})
    temp.replace(path)


def migrate_head(source, destination):
    from .tokenizer_head import FORMAT, validate_state
    data = torch.load(source, map_location='cpu', weights_only=True)
    if not isinstance(data, dict) or set(data)-{'model','cfg'} or 'model' not in data:
        raise ValueError('Unexpected tokenizer checkpoint structure')
    validate_state(data['model'])
    save_tensors(destination, data['model'], dict(format=FORMAT,
        source_file=Path(source).name, source_sha256=sha256(source),
        cfg=json.dumps(data.get('cfg', {})), license='CC-BY-NC-4.0'))
    return Path(destination)


def migrate_nar(source, destination):
    data = torch.load(source, map_location='cpu', weights_only=True)
    if not isinstance(data, dict) or set(data) != {'lora','rank','io'}:
        raise ValueError('Unexpected NAR checkpoint structure')
    rank = int(data['rank'])
    if rank < 1 or len(data['lora']) != 28*7*2 or set(data['io']) != {'vae2llm','llm2vae'}:
        raise ValueError('Unexpected NAR rank, target count or IO modules')
    names = [('nar_self_attn.'+n, n) for n in ('q_proj','k_proj','v_proj','o_proj')]
    names += [('nar_mlp.'+n, n) for n in ('gate_proj','up_proj','down_proj')]
    dims = {'q_proj':(2048,2048),'k_proj':(1024,2048),'v_proj':(1024,2048),
            'o_proj':(2048,2048),'gate_proj':(6144,2048),'up_proj':(6144,2048),'down_proj':(2048,6144)}
    state = {}
    iterator = iter(data['lora'])
    for layer in range(28):
        for suffix, name in names:
            down, up = next(iterator), next(iterator)
            rows, cols = dims[name]
            if down.shape != (rank,cols) or up.shape != (rows,rank):
                raise ValueError(f'Wrong NAR LoRA shape at layer {layer} {suffix}')
            prefix = f'model.layers.{layer}.{suffix}'
            state[prefix+'.lora_down.weight'] = down
            state[prefix+'.lora_up.weight'] = up
    for name, shape in [('vae2llm',(2048,64)),('llm2vae',(64,2048))]:
        io = data['io'][name]
        if set(io) != {'weight','bias'} or io['weight'].shape != shape or io['bias'].shape != (shape[0],):
            raise ValueError(f'Unexpected full replacement projection: {name}')
        for key, tensor in io.items():
            state[f'io.{name}.{key}'] = tensor
    save_tensors(destination, state, dict(format='yue2-mothersuperior-nar-v1',
        rank=rank, alpha=rank, source_sha256=sha256(source), source_file=Path(source).name,
        io_semantics='full replacement weights and biases, not LoRA deltas', license='CC-BY-NC-4.0'))
    return Path(destination)
