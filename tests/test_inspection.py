import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from trainer_core.inspection import apply_nar_lora


@pytest.fixture
def loader_fixture(monkeypatch):
    prefix = 'diffusion_model.llm2vae'
    tensors = {prefix+'.lora_down.weight':torch.ones(2,3),
               prefix+'.lora_up.weight':torch.ones(4,2)}
    mapping = {prefix:prefix+'.weight'}
    comfy = ModuleType('comfy')
    comfy.lora = ModuleType('comfy.lora')
    comfy.lora.model_lora_keys_unet = lambda model, keys: mapping
    comfy.lora.load_lora = lambda tensors, keys: {prefix+'.weight':object()}
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    monkeypatch.setitem(sys.modules, 'comfy.lora', comfy.lora)
    calls = []
    clone = SimpleNamespace(add_patches=lambda patches,strength: calls.append(strength) or list(patches))
    model = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: {prefix+'.weight':torch.zeros(4,3)}),
                            clone=lambda:clone)
    return model, tensors, mapping, calls


def test_zero_strength_does_not_add_patches(loader_fixture):
    model, tensors, mapping, calls = loader_fixture
    _, report = apply_nar_lora(model, tensors, 0)
    assert not calls
    assert '"disabled": true' in report


def test_unknown_targets_raise(loader_fixture):
    model, tensors, mapping, calls = loader_fixture
    mapping.clear()
    with pytest.raises(ValueError, match='do not match'):
        apply_nar_lora(model, tensors, 1)
    assert not calls


def test_shape_mismatch_raises(loader_fixture):
    model, tensors, mapping, calls = loader_fixture
    tensors['diffusion_model.llm2vae.lora_up.weight'] = torch.ones(5,2)
    with pytest.raises(ValueError, match='shape'):
        apply_nar_lora(model, tensors, 1)
    assert not calls


def test_patch_strength_passed_to_modelpatcher(loader_fixture):
    model, tensors, mapping, calls = loader_fixture
    apply_nar_lora(model, tensors, 2)
    assert calls == [2]
