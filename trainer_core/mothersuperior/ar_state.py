"""Resumable trainer state for AR LoRA runs.

trainer_state.safetensors holds everything needed to continue a run exactly:
raw A/B matrices, AdamW moments, RNG states, and run metadata. Generation-
facing checkpoints (last/step-N) stay in converted native format; this file
is trainer-internal and never loaded by ComfyUI's LoRA machinery.

For runs saved before this existed, native_to_ab() inverts convert_ar() so a
weights-only resume is possible from any native-format checkpoint.
"""
from dataclasses import asdict
import json
import random

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

STATE_FORMAT = 'yue2-ar-trainer-state-v1'
STATE_FILE = 'trainer_state.safetensors'

QKV_ORDER = ('q_proj', 'k_proj', 'v_proj')
GATE_UP_ORDER = ('gate_proj', 'up_proj')
_GROUPS = (('self_attn', QKV_ORDER, 'qkv_proj'), ('mlp', GATE_UP_ORDER, 'gate_up_proj'))
_SINGLES = ('self_attn.o_proj', 'mlp.down_proj')


def _module_index(modules):
    """Stable (name, module) ordering shared by save and load."""
    return sorted(modules.items())


def save_state(path, modules, params, optimizer, step, best, cfg, artist_count=None, py_rng=None):
    state = {}
    for i, (name, module) in enumerate(_module_index(modules)):
        state[f'net.{i}.A'] = module.A.detach().cpu()
        state[f'net.{i}.B'] = module.B.detach().cpu()
    opt_step = 0
    for i, p in enumerate(params):
        st = optimizer.state.get(p, {}) if optimizer is not None else {}
        if 'exp_avg' in st:
            state[f'opt.{i}.m'] = st['exp_avg'].detach().cpu()
            state[f'opt.{i}.v'] = st['exp_avg_sq'].detach().cpu()
            opt_step = max(opt_step, int(st.get('step', 0)))
    rng = {'torch': torch.get_rng_state().numpy().tobytes().hex()}
    if torch.cuda.is_available():
        rng['cuda'] = torch.cuda.get_rng_state().cpu().numpy().tobytes().hex()
    if py_rng is None:
        py_rng = random.getstate()
    config = {k: v for k, v in asdict(cfg).items() if k not in ('live_curve_path',)}
    metadata = dict(format=STATE_FORMAT, step=str(step), best=str(best),
        optimizer='adamw' if any(k.startswith('opt.') for k in state) else 'none',
        opt_step=str(opt_step), seed=str(cfg.seed), rank=str(cfg.rank),
        config=json.dumps(config), python_rng=json.dumps(py_rng),
        rng=json.dumps(rng), artist_count=str(artist_count or 0),
        license='CC-BY-NC-4.0')
    temp = path.with_suffix(path.suffix + '.tmp')
    save_file(state, str(temp), metadata=metadata)
    temp.replace(path)
    return path


def load_state(path):
    with safe_open(str(path), framework='pt') as handle:
        metadata = handle.metadata() or {}
        if metadata.get('format') != STATE_FORMAT:
            raise ValueError(f'{path} is not a {STATE_FORMAT} file')
        tensors = {k: handle.get_tensor(k) for k in handle.keys()}
    return tensors, metadata


def restore_optimizer(optimizer, params, tensors, opt_step):
    for i, p in enumerate(params):
        m_key, v_key = f'opt.{i}.m', f'opt.{i}.v'
        if m_key not in tensors:
            return False
        st = optimizer.state[p]
        st['step'] = torch.tensor(float(opt_step))
        st['exp_avg'] = tensors[m_key].clone()
        st['exp_avg_sq'] = tensors[v_key].clone()
    return True


def restore_rng(metadata):
    import numpy as np
    rng = json.loads(metadata.get('rng', '{}'))
    if 'torch' in rng:
        torch.set_rng_state(torch.from_numpy(np.frombuffer(bytes.fromhex(rng['torch']), dtype=np.uint8).copy()))
    if 'cuda' in rng and torch.cuda.is_available():
        # torch.cuda.set_rng_state insists on a CPU ByteTensor (it moves the
        # state to the device itself); passing a CUDA tensor raises.
        torch.cuda.set_rng_state(torch.from_numpy(np.frombuffer(bytes.fromhex(rng['cuda']), dtype=np.uint8).copy()))
    if metadata.get('python_rng'):
        version, state_vector, gauss = json.loads(metadata['python_rng'])
        random.setstate((version, tuple(state_vector), gauss))


def native_to_ab(native, dims=None):
    """Inverse of ar_convert.convert_ar: native fused keys -> raw A/B pairs.

    dims: optional {'qkv': (q_out, k_out, v_out), 'gate_up': (g_out, u_out)}
    row split of the fused projections. YuE2's AR attention is GQA, so q/k/v
    are NOT equal (e.g. 2048/1024/1024) — when dims is omitted the split falls
    back to equal parts and validates divisibility. Shape checks at load time
    catch any mismatch against the injected modules.
    """
    import re
    out = {}
    keys = {k for k in native if k.endswith('.lora_down.weight')}
    for down_key in sorted(keys):
        base = down_key[:-len('.lora_down.weight')]
        up_key = base + '.lora_up.weight'
        if up_key not in native:
            raise ValueError(f'Missing fused pair for {base}')
        down, up = native[down_key].float(), native[up_key].float()
        fused_rank = down.shape[0]
        alpha = float(native.get(base + '.alpha', torch.tensor(float(fused_rank))))
        scale = alpha / fused_rank
        m = re.match(r'^text_encoders\.model\.layers\.(\d+)\.(self_attn|mlp)\.(qkv_proj|gate_up_proj)$', base)
        if m:
            layer, group, fused = m.groups()
            order = QKV_ORDER if fused == 'qkv_proj' else GATE_UP_ORDER
            key = 'qkv' if fused == 'qkv_proj' else 'gate_up'
            if dims and key in dims:
                parts_out = tuple(int(d) for d in dims[key])
                if len(parts_out) != len(order) or sum(parts_out) != up.shape[0]:
                    raise ValueError(f'Dims {parts_out} do not match fused {base} '
                                     f'up rows {up.shape[0]}')
            else:
                if up.shape[0] % len(order) or fused_rank % len(order):
                    raise ValueError(f'Uneven fused split for {base}; pass dims')
                parts_out = (up.shape[0] // len(order),) * len(order)
            if fused_rank % len(order):
                raise ValueError(f'Fused rank {fused_rank} not divisible for {base}')
            part_rank = fused_rank // len(order)
            row = 0
            for i, (part, part_out) in enumerate(zip(order, parts_out)):
                prefix = f'model.layers.{layer}.{group}.{part}'
                out[prefix + '.lora_down.weight'] = down[i * part_rank:(i + 1) * part_rank].contiguous()
                out[prefix + '.lora_up.weight'] = (up[row:row + part_out, i * part_rank:(i + 1) * part_rank] / scale).contiguous()
                row += part_out
            continue
        m = re.match(r'^text_encoders\.(model\.layers\.\d+\.(?:self_attn\.o_proj|mlp\.down_proj))$', base)
        if m:
            out[m.group(1) + '.lora_down.weight'] = down.contiguous()
            out[m.group(1) + '.lora_up.weight'] = (up / scale).contiguous()
            continue
        raise ValueError(f'Unexpected native AR key: {base}')
    return out


def copy_ab_into_modules(modules, ab):
    """Load raw A/B tensors into injected ARLoRALinear modules (shape-checked)."""
    index = _module_index(modules)
    if not any(k.startswith(f'net.') for k in ab):
        # trainer-format keys: model.layers.N....lora_{down,up}.weight
        for name, module in index:
            d_key, u_key = name + '.lora_down.weight', name + '.lora_up.weight'
            if d_key not in ab or u_key not in ab:
                raise ValueError(f'Resumed state is missing {name}')
            d, u = ab[d_key], ab[u_key]
            if d.shape != module.A.shape or u.shape != module.B.shape:
                raise ValueError(f'Rank/shape mismatch for {name}: '
                                 f'got down {tuple(d.shape)} up {tuple(u.shape)}, '
                                 f'expected A {tuple(module.A.shape)} B {tuple(module.B.shape)}')
            module.A.data.copy_(d.to(module.A.device, module.A.dtype))
            module.B.data.copy_(u.to(module.B.device, module.B.dtype))
        return len(index)
    for i, (name, module) in enumerate(index):
        d, u = ab[f'net.{i}.A'], ab[f'net.{i}.B']
        if d.shape != module.A.shape or u.shape != module.B.shape:
            raise ValueError(f'Rank/shape mismatch for {name}')
        module.A.data.copy_(d.to(module.A.device, module.A.dtype))
        module.B.data.copy_(u.to(module.B.device, module.B.dtype))
    return len(index)
