import hashlib
import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from trainer_core.mothersuperior.tokenizer_head import instance_normalize, windows, TokenizerHead, load_head, predict
from trainer_core.mothersuperior.serialization import migrate_head, migrate_nar
from trainer_core.mothersuperior.regularizer import migrate_pack, load_pack, split_of
from trainer_core.mothersuperior.semantic_data import cache_key, interpolate_features
from trainer_core.mothersuperior.ar_convert import convert_ar
from trainer_core.mothersuperior.ar_train import partition, inject, lm_loss


def test_population_normalization():
    x = np.array([[1,3],[3,3]],dtype=np.float16)
    actual = instance_normalize(x)
    np.testing.assert_allclose(actual,np.array([[-1,0],[1,0]])/(1+1e-5),atol=1e-7)


def test_normalization_survives_safe_tensor_layout():
    x = np.asfortranarray(np.random.default_rng(1).normal(size=(100,1024)).astype(np.float16))
    np.testing.assert_array_equal(instance_normalize(x),instance_normalize(np.ascontiguousarray(x)))
    upstream = x.astype(np.float32)
    upstream = (upstream-upstream.mean(0))/(upstream.std(0)+1e-5)
    np.testing.assert_array_equal(instance_normalize(x),upstream)


def test_train_config_live_curve_defaults():
    from trainer_core.mothersuperior.ar_train import TrainConfig
    cfg = TrainConfig()
    assert cfg.live_curve is True
    assert cfg.live_curve_path == ''


@pytest.mark.parametrize('length',[1,128,511,512,513,700,768,1024,1025,1500])
def test_upstream_window_copy(length):
    # Independently spell out the published slicing, including last-window overwrite.
    starts = list(range(0,max(1,length-512+1),256))
    if starts[-1]+512<length: starts.append(max(0,length-512))
    expected = []
    for s in starts:
        n = min(512,length-s)
        expected.append((s,n,s+(0 if s==0 else 128),s+n-(0 if s+n>=length else 128)))
    assert list(windows(length)) == expected
    coverage = np.zeros(length)
    for s,n,lo,hi in windows(length): coverage[lo:hi] += 1
    assert np.all(coverage > 0)


def test_interpolation_length():
    features = torch.tensor([[0.,1.],[2.,3.]])
    actual = interpolate_features(features,24000*3)
    assert actual.shape == (75,2) and actual.dtype == torch.float16
    torch.testing.assert_close(actual[0],features[0].half())
    torch.testing.assert_close(actual[-1],features[-1].half())


def test_cache_invalidation(tmp_path):
    p = tmp_path/'audio.wav'
    p.write_bytes(b'a')
    first = cache_key(p,'head-a','mert-a')
    assert first == cache_key(p,'head-a','mert-a')
    assert first != cache_key(p,'head-b','mert-a')
    assert first != cache_key(p,'head-a','mert-b')
    assert first != cache_key(p,'head-a','mert-a',{'max_seconds':3})
    p.write_bytes(b'ab')
    assert first != cache_key(p,'head-a','mert-a')


def test_head_roundtrip(tmp_path):
    torch.manual_seed(3)
    head = TokenizerHead().eval()
    state = head.state_dict()
    pt, safe = tmp_path/'head.pt',tmp_path/'head.safetensors'
    torch.save({'model':state,'cfg':{'instnorm':True}},pt)
    migrate_head(pt,safe)
    restored = load_head(safe)
    for k,v in state.items(): torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
    pt.unlink()
    assert load_head(safe).pos.shape == (1,512,512)


def test_head_rejects_wrong_checkpoint(tmp_path):
    p = tmp_path/'bad.pt'
    torch.save({'model':{'pos':torch.zeros(1,10,2)}},p)
    with pytest.raises(ValueError,match='keys'):
        migrate_head(p,tmp_path/'bad.safetensors')


def test_predict_deterministic_and_ids():
    class FixtureHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))
        def forward(self,x):
            logits = torch.zeros(1,512,4)
            logits[:,:,2] = 1
            return logits
    x = np.random.default_rng(3).normal(size=(1025,1024)).astype(np.float16)
    head = FixtureHead()
    a,b = predict(x,head),predict(x,head)
    assert torch.equal(a,b) and a.shape == (1025,) and (a==2).all()


def pack_records():
    return [dict(name=str(n),src=split_of(str(n)),style='style',lyrics='lyrics',
                 codec=np.arange(n%9+1,dtype=np.int32)) for n in range(100)]


def test_numpy_pack_roundtrip_and_validation_isolation(tmp_path):
    rows = pack_records()
    p = tmp_path/'pack.pt'
    torch.save(rows,p)
    target = migrate_pack(p,tmp_path/'pack.safetensors',trusted_upstream=True)
    p.unlink()
    restored = load_pack(target)
    for a,b in zip(rows,restored):
        np.testing.assert_array_equal(a['codec'],b['codec'].numpy())
        assert {k:v for k,v in a.items() if k!='codec'} == {k:v for k,v in b.items() if k!='codec'}
    _,train,val = partition([dict(src='artist')],restored)
    assert {r['name'] for r in train}.isdisjoint(r['name'] for r in val)
    assert all(int(hashlib.md5(r['name'].encode()).hexdigest(),16)%20 == 0 for r in val)


@pytest.mark.parametrize('names,native,rows',[
    (['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],'self_attn.qkv_proj',[8,4,4]),
    (['mlp.gate_proj','mlp.up_proj'],'mlp.gate_up_proj',[12,12]),
    (['self_attn.o_proj'],'self_attn.o_proj',[8]),
    (['mlp.down_proj'],'mlp.down_proj',[8]),
])
def test_ar_fusion(names,native,rows):
    state,expected = {},[]
    for name,n in zip(names,rows):
        d,u = torch.randn(3,8),torch.randn(n,3)
        prefix = 'model.layers.0.'+name
        state[prefix+'.lora_down.weight'],state[prefix+'.lora_up.weight'] = d,u
        expected.append(u@d*7/3)
    actual = convert_ar(state,alpha=7)
    p = 'text_encoders.model.layers.0.'+native
    torch.testing.assert_close(actual[p+'.lora_up.weight'] @ actual[p+'.lora_down.weight'],torch.cat(expected))
    assert actual[p+'.alpha'] == actual[p+'.lora_down.weight'].shape[0]


def test_tiny_ar_gradient_only_updates_ar():
    from trainer_core.yue2_ref.modeling_yue2 import YuE2Config,YuE2ForCausalLM
    config = YuE2Config(hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
        num_key_value_heads=1,head_dim=8,intermediate_size=24,vocab_size=64,max_latent_frames=16)
    model = YuE2ForCausalLM(config).float().eval()
    modules = inject(model,2)
    params = [p for m in modules.values() for p in (m.A,m.B)]
    opt = torch.optim.AdamW(params,lr=1e-3)
    nar = model.model.layers[0].nar_self_attn.q_proj.weight.detach().clone()
    for _ in range(2):
        loss = lm_loss(model,[1,2,3,4,5],2)
        loss.backward(); opt.step(); opt.zero_grad()
    assert all(m.A.dtype == torch.float32 for m in modules.values())
    assert any(torch.count_nonzero(m.B) for m in modules.values())
    torch.testing.assert_close(nar,model.model.layers[0].nar_self_attn.q_proj.weight,rtol=0,atol=0)
    assert all(not p.requires_grad for n,p in model.named_parameters() if '.A' not in n and '.B' not in n)


def test_nar_migration_preserves_full_weights_and_biases(tmp_path):
    dims = [(2048,2048),(1024,2048),(1024,2048),(2048,2048),(6144,2048),(6144,2048),(2048,6144)]
    pairs = [v for rows,cols in dims for v in (torch.ones(1,cols),torch.ones(rows,1))]
    io = {name:dict(weight=torch.randn(*shape),bias=torch.randn(shape[0]))
          for name,shape in [('vae2llm',(2048,64)),('llm2vae',(64,2048))]}
    source = tmp_path/'nar.pt'
    torch.save(dict(lora=pairs*28,rank=1,io=io),source)
    target = migrate_nar(source,tmp_path/'nar.safetensors')
    restored = load_file(target)
    assert len(restored) == 28*7*2+4
    for name,parts in io.items():
        for key,tensor in parts.items():
            torch.testing.assert_close(restored[f'io.{name}.{key}'],tensor,rtol=0,atol=0)


def test_mert_rotary_rebuilt_after_hf_loading(monkeypatch):
    from types import SimpleNamespace
    import transformers
    from trainer_core.mothersuperior.semantic_data import load_mert
    class Rotary(torch.nn.Module):
        def __init__(self,config):
            super().__init__()
            self.register_buffer('inv_freq',torch.tensor([1.,.01]),persistent=False)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = object()
            self.embed_positions = Rotary(self.config)
            self.embed_positions.inv_freq.fill_(float('nan'))
    monkeypatch.setattr(transformers.AutoModel,'from_pretrained',lambda *a,**k:Model())
    monkeypatch.setattr(transformers.AutoFeatureExtractor,'from_pretrained',lambda *a,**k:object())
    _,model = load_mert(SimpleNamespace(mert='pinned-local-snapshot'),'cpu')
    torch.testing.assert_close(model.embed_positions.inv_freq,torch.tensor([1.,.01]))
