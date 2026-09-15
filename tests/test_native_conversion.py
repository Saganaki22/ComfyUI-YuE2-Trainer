import pytest
import torch
from safetensors.torch import save_file, load_file

from trainer_core.convert import convert_tensors
from trainer_core.inspection import inspect_tensors


@pytest.mark.parametrize('group,native,rows', [
    (['nar_self_attn.q_proj', 'nar_self_attn.k_proj', 'nar_self_attn.v_proj'], 'self_attn.qkv_proj', [8,4,4]),
    (['nar_mlp.gate_proj', 'nar_mlp.up_proj'], 'mlp.gate_up_proj', [12,12]),
    (['nar_self_attn.o_proj'], 'self_attn.o_proj', [8]),
    (['nar_mlp.down_proj'], 'mlp.down_proj', [8]),
])
def test_native_fusion(group, native, rows, tmp_path):
    g = torch.Generator().manual_seed(12)
    tensors, expected = {}, []
    for name, n in zip(group, rows):
        down = torch.randn(3, 8, generator=g)
        up = torch.randn(n, 3, generator=g)
        prefix = 'model.layers.0.' + name
        tensors[prefix+'.lora_down.weight'] = down
        tensors[prefix+'.lora_up.weight'] = up
        expected.append(up @ down * (7/3))
    result, report = convert_tensors(tensors, {'alpha':7})
    assert not report['skipped']
    path = tmp_path/'adapter.safetensors'
    save_file(result, path, metadata={'format':'comfyui-native-lora'})
    result = load_file(path)
    prefix = 'diffusion_model.model.layers.0.' + native
    actual = result[prefix+'.lora_up.weight'] @ result[prefix+'.lora_down.weight']
    actual *= result[prefix+'.alpha'] / result[prefix+'.lora_down.weight'].shape[0]
    torch.testing.assert_close(actual, torch.cat(expected))
    assert inspect_tensors(result)['nonzero_modules'] == 1


@pytest.mark.parametrize('name', ['vae2llm','llm2vae','time_embedder.mlp.0','time_embedder.mlp.2'])
def test_top_level(name):
    down, up = torch.randn(2,8), torch.randn(6,2)
    result, _ = convert_tensors({name+'.lora_down.weight':down,name+'.lora_up.weight':up}, {'alpha':5})
    p = 'diffusion_model.' + name
    torch.testing.assert_close(result[p+'.lora_up.weight'] @ result[p+'.lora_down.weight'], up @ down * 2.5)


def test_zero_delta_rejected():
    with pytest.raises(ValueError, match='zero'):
        inspect_tensors({'a.lora_down.weight':torch.ones(2,3),'a.lora_up.weight':torch.zeros(4,2)})


def test_missing_pair_rejected():
    with pytest.raises(ValueError, match='matching'):
        inspect_tensors({'a.lora_down.weight':torch.ones(2,3)})


def test_incomplete_fusion_rejected():
    p = 'model.layers.0.nar_self_attn.q_proj'
    with pytest.raises(ValueError, match='incomplete fusion'):
        convert_tensors({p+'.lora_down.weight':torch.ones(2,3),p+'.lora_up.weight':torch.ones(4,2)}, {})


def test_inconsistent_rank_rejected():
    with pytest.raises(ValueError, match='rank'):
        convert_tensors({'vae2llm.lora_down.weight':torch.ones(2,3),
                         'vae2llm.lora_up.weight':torch.ones(4,3)}, {})
