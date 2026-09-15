"""Trainer-state save/load roundtrip and native->raw A/B inversion."""
import json
import random

import pytest
import torch
from torch import nn

from trainer_core.mothersuperior.ar_convert import convert_ar
from trainer_core.mothersuperior.ar_state import (STATE_FILE, STATE_FORMAT,
    copy_ab_into_modules, load_state, native_to_ab, restore_optimizer,
    save_state)


class _FakeBase(nn.Linear):
    pass


def _modules(ranks=(4, 3)):
    """Two fake injected modules with distinct in/out dims."""
    from trainer_core.mothersuperior.ar_train import ARLoRALinear
    mods = {}
    for i, rank in enumerate(ranks):
        base = nn.Linear(8 + i, 12 + i)
        mods[f'model.layers.{i}.mlp.up_proj'] = ARLoRALinear(base, rank)
    return mods


def _cfg(**kw):
    from trainer_core.mothersuperior.ar_train import TrainConfig
    return TrainConfig(**kw)


def test_state_roundtrip(tmp_path):
    torch.manual_seed(0)
    mods = _modules()
    params = [p for m in mods.values() for p in (m.A, m.B)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    loss = sum(p.sum() for p in params)
    loss.backward()
    opt.step()
    before = {k: v.detach().clone() for k, v in
              [(f'{n}.A', m.A) for n, m in mods.items()] +
              [(f'{n}.B', m.B) for n, m in mods.items()]}
    moments = [opt.state[p]['exp_avg'].clone() for p in params]
    rng_state = random.Random(7).getstate()

    path = tmp_path / STATE_FILE
    save_state(path, mods, params, opt, step=42, best=1.25, cfg=_cfg(),
               artist_count=9, py_rng=rng_state)

    # Mutate everything; restore; expect exact equality.
    for m in mods.values():
        m.A.data.add_(1.0)
        m.B.data.add_(1.0)
    opt.state.clear()
    tensors, meta = load_state(path)
    assert meta['format'] == STATE_FORMAT
    assert int(meta['step']) == 42
    assert float(meta['best']) == 1.25
    assert int(meta['artist_count']) == 9

    copy_ab_into_modules(mods, tensors)
    for (n, m) in mods.items():
        assert torch.equal(m.A, before[f'{n}.A'])
        assert torch.equal(m.B, before[f'{n}.B'])
    assert restore_optimizer(opt, params, tensors, int(meta['opt_step']))
    for p, moment in zip(params, moments):
        assert torch.equal(opt.state[p]['exp_avg'], moment)

    restored = json.loads(meta['python_rng'])
    assert tuple(restored[1]) == rng_state[1]
    cfg = json.loads(meta['config'])
    assert cfg['rank'] == 64
    assert 'live_curve_path' not in cfg  # machine-local path must not travel

    # RNG restore must not raise (torch.cuda.set_rng_state wants a CPU tensor).
    from trainer_core.mothersuperior.ar_state import restore_rng
    torch.manual_seed(123)
    restore_rng(meta)


def test_load_state_rejects_wrong_format(tmp_path):
    from safetensors.torch import save_file
    path = tmp_path / 'bogus.safetensors'
    save_file({'x': torch.zeros(1)}, str(path), metadata={'format': 'something-else'})
    with pytest.raises(ValueError, match='not a'):
        load_state(path)


def test_copy_ab_shape_mismatch(tmp_path):
    mods = _modules(ranks=(4,))
    ab = {'net.0.A': torch.zeros(9, 8), 'net.0.B': torch.zeros(12, 9)}
    with pytest.raises(ValueError, match='mismatch'):
        copy_ab_into_modules(mods, ab)


def test_native_to_ab_inverts_convert_ar():
    torch.manual_seed(1)
    rank = 4
    raw = {}
    specs = [('self_attn.q_proj', 10, 8), ('self_attn.k_proj', 10, 8),
             ('self_attn.v_proj', 10, 8), ('self_attn.o_proj', 8, 10),
             ('mlp.gate_proj', 16, 8), ('mlp.up_proj', 16, 8),
             ('mlp.down_proj', 8, 16)]
    for name, out_f, in_f in specs:
        base = f'model.layers.0.{name}'
        raw[base + '.lora_down.weight'] = torch.randn(rank, in_f)
        raw[base + '.lora_up.weight'] = torch.randn(out_f, rank)
    native = convert_ar(raw)
    recovered = native_to_ab(native)
    assert set(recovered) == set(raw)
    for key, value in raw.items():
        assert torch.allclose(recovered[key], value, atol=1e-6), key


def test_native_to_ab_gqa_uneven_split():
    """YuE2 AR is GQA: q=2048, k=v=1024 — the real model's shape."""
    torch.manual_seed(2)
    rank = 8
    raw = {}
    specs = [('self_attn.q_proj', 2048, 2048), ('self_attn.k_proj', 1024, 2048),
             ('self_attn.v_proj', 1024, 2048), ('self_attn.o_proj', 2048, 2048),
             ('mlp.gate_proj', 6144, 2048), ('mlp.up_proj', 6144, 2048),
             ('mlp.down_proj', 2048, 6144)]
    for name, out_f, in_f in specs:
        base = f'model.layers.0.{name}'
        raw[base + '.lora_down.weight'] = torch.randn(rank, in_f)
        raw[base + '.lora_up.weight'] = torch.randn(out_f, rank)
    native = convert_ar(raw)
    dims = {'qkv': (2048, 1024, 1024), 'gate_up': (6144, 6144)}
    recovered = native_to_ab(native, dims)
    assert set(recovered) == set(raw)
    for key, value in raw.items():
        assert torch.allclose(recovered[key], value, atol=1e-6), key
    # Without dims, the uneven qkv must refuse to guess.
    with pytest.raises(ValueError, match='Uneven fused split'):
        native_to_ab(native)


def test_native_to_ab_rejects_unknown_keys():
    with pytest.raises(ValueError, match='Unexpected native AR key'):
        native_to_ab({'text_encoders.model.layers.0.self_attn.bogus_proj.lora_down.weight':
                      torch.zeros(4, 8),
                      'text_encoders.model.layers.0.self_attn.bogus_proj.lora_up.weight':
                      torch.zeros(10, 4)})


def test_resume_config_defaults():
    cfg = _cfg()
    assert cfg.resume_from == ''
    assert cfg.resume_optimizer is True
