"""Adapted from Mothersuperior scripts/ar_prep.py (CC BY-NC 4.0 repository).

Architecture, normalization and overlap copying follow revision f2278a2e.
"""
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file

FORMAT = 'yue2-mothersuperior-head-v1'


class TokenizerHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(1024, 512)
        self.pos = nn.Parameter(torch.zeros(1, 512, 512))
        layer = nn.TransformerEncoderLayer(512, 8, 2048, dropout=.1,
            batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, 8)
        self.norm = nn.LayerNorm(512)
        self.head = nn.Linear(512, 32768)

    def forward(self, x):
        return self.head(self.norm(self.enc(self.inp(x) + self.pos[:, :x.shape[1]])))


def validate_state(state):
    with torch.device('meta'):
        expected = TokenizerHead().state_dict()
    if state.keys() != expected.keys():
        raise ValueError(f'Tokenizer head keys differ: missing={sorted(expected.keys()-state.keys())}, unexpected={sorted(state.keys()-expected.keys())}')
    for key, tensor in state.items():
        if tensor.shape != expected[key].shape or not tensor.is_floating_point():
            raise ValueError(f'Invalid tokenizer head tensor {key}: {tensor.shape}')


def load_head(path, device='cpu'):
    with safe_open(str(path), framework='pt') as handle:
        if (handle.metadata() or {}).get('format') != FORMAT:
            raise ValueError('Expected the pre-converted safetensors tokenizer head '
                             '(format tag missing or wrong — re-download the assets)')
    state = load_file(str(path))
    validate_state(state)
    with torch.device('meta'):
        head = TokenizerHead()
    head.load_state_dict(state, strict=True, assign=True)
    return head.to(device).eval().requires_grad_(False)


def instance_normalize(features):
    # Upstream MERT interpolation transposes [channels,time] before np.save,
    # producing a Fortran-order array. Preserve its reduction order even when
    # reading the equivalent tensors from contiguous safetensors storage.
    x = np.array(features, dtype=np.float32, order='F')
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite [frames, channels] MERT features')
    return (x - x.mean(0)) / (x.std(0, ddof=0) + 1e-5)


def windows(length):
    if length < 1:
        raise ValueError('No frames to tokenize')
    starts = list(range(0, max(1, length-512+1), 256))
    if starts[-1]+512 < length:
        starts.append(max(0, length-512))
    for start in starts:
        count = min(512, length-start)
        lo = start + (0 if start == 0 else 128)
        hi = start + count - (0 if start+count >= length else 128)
        yield start, count, lo, hi


@torch.inference_mode()
def predict(features, head, check_interrupt=lambda: None):
    x = instance_normalize(features)
    device = next(head.parameters()).device
    head.eval()
    output = np.zeros(len(x), dtype=np.int64)
    for start, count, lo, hi in windows(len(x)):
        check_interrupt()
        chunk = np.pad(x[start:start+count], ((0,512-count),(0,0)))
        context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with context:
            logits = head(torch.as_tensor(chunk[None], device=device))[0, :count]
        ids = logits.float().argmax(-1).cpu().numpy()
        output[lo:hi] = ids[lo-start:hi-start]
    if output.min() < 0 or output.max() >= 32768:
        raise ValueError('Tokenizer returned out-of-range semantic IDs')
    return torch.from_numpy(output.astype(np.int32))
