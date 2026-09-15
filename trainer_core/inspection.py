"""Validate native YuE2 LoRA deltas and report actual ComfyUI patch matches."""
from __future__ import annotations

import json
import logging

import torch


def inspect_tensors(tensors):
    modules = []
    downs = {k[:-len('.lora_down.weight')]: v for k, v in tensors.items()
             if k.endswith('.lora_down.weight')}
    ups = {k[:-len('.lora_up.weight')]: v for k, v in tensors.items()
           if k.endswith('.lora_up.weight')}
    if not downs or downs.keys() != ups.keys():
        raise ValueError('LoRA must contain matching down/up pairs; missing pairs: '
                         + str(sorted(downs.keys() ^ ups.keys())))
    for name, down in downs.items():
        up = ups[name]
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1] or down.shape[0] == 0:
            raise ValueError(f'Invalid LoRA matrix shapes for {name}')
        alpha = float(tensors.get(name + '.alpha', down.shape[0]))
        down, up = down.float(), up.float()
        if not (torch.isfinite(down).all() and torch.isfinite(up).all()
                and torch.isfinite(torch.tensor(alpha))):
            raise ValueError(f'Non-finite LoRA values for {name}')
        # Bound temporary memory for large fused projection deltas.
        squared_norm, maximum = 0., 0.
        for chunk in up.split(256):
            delta = (chunk @ down) * (alpha / down.shape[0])
            squared_norm += delta.double().square().sum().item()
            maximum = max(maximum, delta.abs().max().item())
        modules.append(dict(module=name, rank=down.shape[0], down_norm=down.norm().item(),
                            up_norm=up.norm().item(), delta_norm=squared_norm ** .5,
                            delta_max_abs=maximum))
    if not any(m['delta_max_abs'] > 0 for m in modules):
        raise ValueError('All learned LoRA deltas are zero; refusing an ineffective adapter')
    return dict(tensors=len(tensors), modules=modules,
                nonzero_modules=sum(m['delta_max_abs'] > 0 for m in modules))


def apply_nar_lora(model, tensors, strength):
    """Use the standard ComfyUI mapper/parser/ModelPatcher and fail on missing keys."""
    import comfy.lora

    report = inspect_tensors(tensors)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    missing = [m['module'] for m in report['modules'] if m['module'] not in key_map]
    if missing:
        raise ValueError('YuE2 NAR LoRA keys do not match this MODEL: ' + ', '.join(missing))
    state = model.model.state_dict()
    for module in report['modules']:
        name = module['module']
        down, up = tensors[name + '.lora_down.weight'], tensors[name + '.lora_up.weight']
        if tuple(state[key_map[name]].shape) != (up.shape[0], down.shape[1]):
            raise ValueError(f'LoRA delta shape does not match MODEL weight: {name}')
    patches = comfy.lora.load_lora(tensors, key_map)
    if not patches:
        raise ValueError('Zero YuE2 NAR patches resolved')
    if any(not k.startswith('diffusion_model.') for k in patches):
        raise ValueError('Expected NAR diffusion_model patches only')
    clone = model.clone()
    applied = set(clone.add_patches(patches, strength)) if strength else set(patches)
    unmatched = sorted(set(patches) - applied)
    if unmatched or not applied:
        raise ValueError('YuE2 NAR patches rejected by ModelPatcher: ' + str(unmatched))
    report.update(candidate_model_patches=len(patches), matched_model_keys=len(applied),
                  unmatched_keys=unmatched, ar_patches=0,
                  nar_patches=len(applied) if strength else 0, strength=strength,
                  disabled=strength == 0)
    logging.getLogger('yue2_trainer').info('YuE2 LoRA inspection: %s', json.dumps(report))
    return clone, json.dumps(report, indent=2)
