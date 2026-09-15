"""Safe compact minted pack; reproduces build_reg_pack.py's validation split."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .serialization import save_tensors, sha256

FORMAT = 'yue2-minted-regularizer-v1'


def split_of(name):
    return 'minted_val' if int(hashlib.md5(name.encode()).hexdigest(),16) % 20 == 0 else 'minted'


def save_pack(records, path):
    path = Path(path).with_suffix('.safetensors')
    arrays, metadata, lengths = [], [], []
    seen = set()
    for item in records:
        if set(item) != {'name','src','style','lyrics','codec'}:
            raise ValueError('Unexpected regularizer fields; refusing to discard data')
        if any(not isinstance(item[k], str) for k in ('name','src','style','lyrics')):
            raise ValueError('Regularizer metadata must be strings')
        if item['src'] != split_of(item['name']) or item['name'] in seen:
            raise ValueError('Incorrect validation split or duplicate regularizer name')
        seen.add(item['name'])
        codec = torch.as_tensor(item['codec'])
        if codec.dtype != torch.int32 or codec.ndim != 1 or len(codec) == 0 or codec.min() < 0 or codec.max() >= 32768:
            raise ValueError('Expected nonempty int32 semantic ID arrays in [0,32767]')
        arrays.append(codec)
        lengths.append(len(codec))
        metadata.append({k:v for k,v in item.items() if k != 'codec'})
    if not arrays:
        raise ValueError('Empty regularizer pack')
    text = ''.join(json.dumps(m,ensure_ascii=False)+'\n' for m in metadata)
    length = torch.tensor(lengths,dtype=torch.int64)
    state = dict(codec_tokens=torch.cat(arrays), codec_offsets=length.cumsum(0)-length, codec_lengths=length)
    save_tensors(path, state, dict(format=FORMAT, jsonl_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest()))
    temp = path.with_suffix('.jsonl.tmp')
    temp.write_text(text,encoding='utf-8',newline='')
    temp.replace(path.with_suffix('.jsonl'))
    return path


def load_pack(path):
    path = Path(path).with_suffix('.safetensors')
    with safe_open(str(path),framework='pt') as handle:
        meta = handle.metadata() or {}
    if meta.get('format') != FORMAT or meta.get('jsonl_sha256') != sha256(path.with_suffix('.jsonl')):
        raise ValueError('Regularizer format or JSONL checksum mismatch')
    rows = [json.loads(line) for line in path.with_suffix('.jsonl').read_text(encoding='utf-8').splitlines()]
    state = load_file(str(path))
    if set(state) != {'codec_tokens','codec_offsets','codec_lengths'}:
        raise ValueError('Unexpected regularizer tensors')
    flat, offsets, lengths = (state[k] for k in ('codec_tokens','codec_offsets','codec_lengths'))
    if flat.dtype != torch.int32 or offsets.dtype != torch.int64 or lengths.dtype != torch.int64:
        raise ValueError('Wrong regularizer tensor dtypes')
    if any(x.ndim != 1 for x in (flat,offsets,lengths)) or len(rows) != len(lengths) or len(offsets) != len(rows):
        raise ValueError('Regularizer row count/shape mismatch')
    if not torch.equal(offsets,lengths.cumsum(0)-lengths) or int(lengths.sum()) != len(flat) or (lengths <= 0).any():
        raise ValueError('Invalid regularizer offsets/lengths')
    if not len(flat) or flat.min() < 0 or flat.max() >= 32768:
        raise ValueError('Invalid regularizer semantic IDs')
    seen = set()
    for row, offset, length in zip(rows,offsets.tolist(),lengths.tolist()):
        if set(row) != {'name','src','style','lyrics'} or any(not isinstance(v,str) for v in row.values()):
            raise ValueError('Invalid regularizer text fields')
        if row['src'] != split_of(row['name']) or row['name'] in seen:
            raise ValueError('Regularizer train/validation contamination or duplicate name')
        seen.add(row['name'])
        row['codec'] = flat[offset:offset+length]
    return rows


def migrate_pack(source, destination, *, trusted_upstream=False):
    if not trusted_upstream:
        raise ValueError('NumPy pickle migration is restricted to explicitly trusted upstream assets')
    try:
        records = torch.load(source,map_location='cpu',weights_only=True)
    except Exception as error:
        # The official pack stores NumPy int32 ndarrays. No unrestricted unpickler.
        import pickle
        if not isinstance(error,pickle.UnpicklingError):
            raise
        allowed = [np._core.multiarray._reconstruct,np.ndarray,np.dtype,np.dtypes.Int32DType]
        with torch.serialization.safe_globals(allowed):
            records = torch.load(source,map_location='cpu',weights_only=True)
    destination = save_pack(records,destination)
    loaded = load_pack(destination)
    if len(loaded) != len(records):
        raise ValueError('Regularizer round-trip row mismatch')
    for original, restored in zip(records,loaded):
        for key in ('name','src','style','lyrics'):
            if original[key] != restored[key]:
                raise ValueError('Regularizer metadata changed during migration')
        np.testing.assert_array_equal(original['codec'],restored['codec'].numpy())
    return destination
