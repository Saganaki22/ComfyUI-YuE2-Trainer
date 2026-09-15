"""Lossless AR LoRA fusion for the standard ComfyUI CLIP LoRA key mapper."""
import re

from ..convert import _emit, _fuse_group


def convert_ar(tensors, alpha=None):
    downs = {k[:-len('.lora_down.weight')]:v for k,v in tensors.items() if k.endswith('.lora_down.weight')}
    ups = {k[:-len('.lora_up.weight')]:v for k,v in tensors.items() if k.endswith('.lora_up.weight')}
    if not downs or downs.keys() != ups.keys():
        raise ValueError('AR LoRA requires complete down/up pairs')
    ranks = {v.shape[0] for v in downs.values() if v.ndim == 2}
    if len(ranks) != 1:
        raise ValueError('AR LoRA ranks must match')
    rank = ranks.pop()
    if rank < 1:
        raise ValueError('AR LoRA rank must be positive')
    for name,d in downs.items():
        if d.ndim != 2 or ups[name].ndim != 2 or ups[name].shape[1] != rank:
            raise ValueError('Malformed AR LoRA: '+name)
    scale = float(rank if alpha is None else alpha)/rank
    out, used = {}, set()
    layers = sorted({int(m.group(1)) for k in downs if (m:=re.match(r'^model\.layers\.(\d+)\.',k))})
    for layer in layers:
        base = f'model.layers.{layer}'
        for group,order,native in [('self_attn',('q_proj','k_proj','v_proj'),'qkv_proj'),
                                   ('mlp',('gate_proj','up_proj'),'gate_up_proj')]:
            names = [f'{base}.{group}.{name}' for name in order]
            if any(name in downs for name in names):
                if not all(name in downs for name in names):
                    raise ValueError('Incomplete AR fusion group: '+str(names))
                parts = {short:(downs[full],ups[full]) for short,full in zip(order,names)}
                d,u = _fuse_group(parts,order,scale)
                _emit(out,f'text_encoders.{base}.{group}.{native}',d,u)
                used.update(names)
        for suffix in ('self_attn.o_proj','mlp.down_proj'):
            name = base+'.'+suffix
            if name in downs:
                _emit(out,'text_encoders.'+name,downs[name].float(),ups[name].float()*scale)
                used.add(name)
    if used != downs.keys():
        raise ValueError('Unexpected AR targets: '+str(sorted(downs.keys()-used)))
    return out
