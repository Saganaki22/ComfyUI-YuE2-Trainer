"""Score-free artist AR training adapted from Mothersuperior scripts/ar_lora.py.

CC BY-NC 4.0 upstream repository. Native weights and PyTorch SDPA replace the
upstream HF download/custom attention entry point; the objective is unchanged.
"""
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import random
import time
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .. import native_ckpt
from ..yue2_ref import modeling_yue2, protocol
from ..inspection import inspect_tensors
from .ar_convert import convert_ar
from .regularizer import split_of
from .serialization import save_tensors

log = logging.getLogger('yue2_trainer.artist')
TARGETS = ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
           'mlp.gate_proj','mlp.up_proj','mlp.down_proj')


class ARLoRALinear(nn.Module):
    def __init__(self,base,rank):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(rank,base.in_features,device=base.weight.device,dtype=torch.float32)/math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))

    def forward(self,x):
        return self.base(x) + ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


@dataclass
class TrainConfig:
    steps: int = 1600
    rank: int = 64
    learning_rate: float = 1e-4
    artist_ratio: float = .5
    grad_accum: int = 2
    max_length: int = 12288
    schedule_steps: int = 3000
    warmup_steps: int = 50
    evaluate_every: int = 100
    save_from: int = 600
    save_every: int = 200
    seed: int = 1
    device: str = 'cuda'
    live_curve: bool = True
    live_curve_path: str = ''


def build_model(path,device='cuda'):
    model = native_ckpt.build_lm_from_native(path,modeling_yue2)
    model.eval().requires_grad_(False)
    # Keep the unused NAR branch on CPU. Only AR modules enter the GPU.
    for module in (model.model.embed_tokens,model.model.norm,model.lm_head):
        module.to(device)
    for layer in model.model.layers:
        for name in ('self_attn','mlp','input_layernorm','post_attention_layernorm'):
            getattr(layer,name).to(device)
    return model


def inject(model,rank):
    if rank < 1:
        raise ValueError('AR rank must be positive')
    model.requires_grad_(False)
    modules = {}
    for i,layer in enumerate(model.model.layers):
        for suffix in TARGETS:
            parent,attr = suffix.split('.')
            owner = getattr(layer,parent)
            wrapped = ARLoRALinear(getattr(owner,attr),rank)
            setattr(owner,attr,wrapped)
            modules[f'model.layers.{i}.{suffix}'] = wrapped
    return modules


def ar_layer(layer,x,cos,sin):
    q,k,v = layer.self_attn.project_qkv(layer.input_layernorm(x),cos,sin)
    h = modeling_yue2.sdpa(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True)
    x = x + layer.self_attn.o_proj(h.transpose(1,2).reshape(x.shape[0],x.shape[1],-1))
    return x + layer.mlp(layer.post_attention_layernorm(x))


def sequence(item,tokenizer,max_length=12288):
    prefix = protocol.token_prefixes(protocol.SongRequest(style=item['style'],lyrics=item['lyrics'],cot='off',seed=1),tokenizer)
    codec = torch.as_tensor(item['codec'])
    if codec.ndim != 1 or codec.dtype not in (torch.int32,torch.int64) or not len(codec) or codec.min() < 0 or codec.max() >= 32768:
        raise ValueError('Expected semantic IDs in [0,32767]')
    room = max_length-len(prefix)-1
    if room < 1:
        raise ValueError('Prompt leaves no room for semantic tokens')
    body = (codec[:room].long()+protocol.CODEC_OFFSET).tolist()
    if len(codec) <= room:
        body.append(protocol.MUSIC_END)
    return prefix+body,len(prefix)


def lm_loss(model,ids,prefix_length,grad=True):
    device = model.model.embed_tokens.weight.device
    ids = torch.tensor([ids],device=device,dtype=torch.long)
    x = model.model.embed_tokens(ids)
    cos,sin = model.model.rotary_emb(torch.arange(ids.shape[1],device=device)[None])
    for layer in model.model.layers:
        x = checkpoint(ar_layer,layer,x,cos,sin,use_reentrant=False) if grad else ar_layer(layer,x,cos,sin)
    h = model.model.norm(x[0,prefix_length-1:-1])
    target = ids[0,prefix_length:]
    def chunk_loss(hidden,labels):
        return F.cross_entropy(model.lm_head(hidden).float(),labels,reduction='sum')
    total = 0.
    for start in range(0,len(h),1024):
        args = (h[start:start+1024],target[start:start+1024])
        # Recompute logits on backward instead of retaining vocabulary-sized
        # tensors for every chunk. This changes memory, not the loss.
        total = total + (checkpoint(chunk_loss,*args,use_reentrant=False) if grad else chunk_loss(*args))
    return total/len(h)


def partition(artist,regularizer):
    if not artist or any(row['src'] != 'artist' for row in artist):
        raise ValueError('Expected artist records')
    if any(row['src'] != split_of(row['name']) for row in regularizer):
        raise ValueError('Regularizer validation split mismatch')
    train = [row for row in regularizer if row['src']=='minted']
    validation = [row for row in regularizer if row['src']=='minted_val']
    if not train or not validation:
        raise ValueError('Regularizer needs separate minted training and validation records')
    return artist,train,validation


def train(model,tokenizer,artist,regularizer,output_dir,cfg,check_interrupt=lambda:None,progress=lambda n,total:None):
    if cfg.steps < 1 or cfg.grad_accum < 1 or cfg.evaluate_every < 1 or cfg.save_every < 1 or not 0 <= cfg.artist_ratio <= 1:
        raise ValueError('Invalid AR training counts or mixture')
    if cfg.steps > 1500:
        warnings.warn('Upstream reports memorization around 1500 steps; compare held-out minted loss and checkpoints.')
    artist,minted,mval = partition(artist,regularizer)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    modules = inject(model,cfg.rank)
    params = [p for mod in modules.values() for p in (mod.A,mod.B)]
    count = sum(p.numel() for p in params)
    log.info('AR trainable parameters=%d targets=%d; NAR frozen',count,len(modules))
    optimizer = torch.optim.AdamW(params,lr=cfg.learning_rate,weight_decay=0.,betas=(.9,.95))
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)
    records = []
    # Live chart preview: the record stream is re-parsed and re-rendered to a
    # fixed temp PNG every few seconds (atomic replace); the Training Curve
    # node's frontend extension polls that file while the prompt runs.
    live_p = Path(cfg.live_curve_path) if getattr(cfg,'live_curve_path','') else None
    live_on = bool(getattr(cfg,'live_curve',False)) and live_p is not None
    live_lines = []
    live_last = 0.0
    live_failed = False
    if live_on:
        try:  # don't flash the previous run's chart in the live preview
            live_p.unlink(missing_ok=True)
        except OSError:
            pass
    def maybe_live():
        nonlocal live_last,live_failed
        if not live_on or live_failed:
            return
        if time.time()-live_last < 4.0:
            return
        live_last = time.time()
        try:
            from .. import curve as curve_mod
            steps,losses,lrs,meta = curve_mod.parse_log('\n'.join(live_lines))
            if not steps:
                return
            meta['live'] = True
            tmp = live_p.with_name(live_p.stem+'.tmp.png')
            curve_mod.render_chart(steps,losses,lrs,meta,tmp,
                                   smooth=15,figsize=(8,4.2),dpi=90)
            os.replace(tmp,live_p)
        except Exception as exc:
            live_failed = True
            log.warning('live curve preview disabled: %s',exc)
    def record(data):
        records.append(data)
        with (output/'training.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(data)+'\n')
        log.info('%s',data)
        live_lines.append(json.dumps(data))
        maybe_live()
    @torch.no_grad()
    def evaluate(step):
        result = dict(step=step,kind='evaluation')
        for name,items in [('artist',artist[:6]),('minted_val',mval[:6])]:
            losses = []
            for item in items:
                check_interrupt()
                ids,lp = sequence(item,tokenizer,cfg.max_length)
                losses.append(lm_loss(model,ids,lp,False).item())
            result[name+'_loss'] = sum(losses)/len(losses)
        record(result)
        return result['artist_loss']
    def save(tag,step):
        state = {}
        for name,module in modules.items():
            state[name+'.lora_down.weight'] = module.A.detach().cpu()
            state[name+'.lora_up.weight'] = module.B.detach().cpu()
        native = convert_ar(state)
        inspection = inspect_tensors(native)
        path = output/(tag+'.safetensors')
        save_tensors(path,native,dict(format='comfyui-native-lora',branch='yue2-ar',rank=cfg.rank,
            alpha=cfg.rank,steps=step,source='Mothersuperior/ar_lora.py',license='CC-BY-NC-4.0',
            nonzero_modules=inspection['nonzero_modules']))
        return path
    best = evaluate(0)
    record(dict(kind='configuration',trainable_parameters=count,targets=list(modules),config=vars(cfg)))
    for step in range(1,cfg.steps+1):
        check_interrupt()
        multiplier = min(1,step/max(1,cfg.warmup_steps))*(.2+.8*.5*(1+math.cos(math.pi*min(step,cfg.schedule_steps)/cfg.schedule_steps)))
        for group in optimizer.param_groups:
            group['lr'] = cfg.learning_rate*multiplier
        losses = {'artist':[],'minted':[]}
        for _ in range(cfg.grad_accum):
            check_interrupt()
            source = 'artist' if rng.random()<cfg.artist_ratio else 'minted'
            item = rng.choice(artist if source=='artist' else minted)
            ids,lp = sequence(item,tokenizer,cfg.max_length)
            loss = lm_loss(model,ids,lp)
            if not torch.isfinite(loss): raise ValueError('Non-finite AR loss')
            (loss/cfg.grad_accum).backward()
            losses[source].append(loss.detach().item())
        norm = torch.nn.utils.clip_grad_norm_(params,1.).item()
        if not math.isfinite(norm): raise ValueError('Non-finite AR gradients')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record(dict(kind='training',step=step,lr=optimizer.param_groups[0]['lr'],grad_norm=norm,
                    **{key+'_loss':sum(value)/len(value) if value else None for key,value in losses.items()}))
        if step%cfg.evaluate_every == 0 or step == cfg.steps:
            metric = evaluate(step)
            if metric < best:
                best = metric
                save('best',step)
            save('last',step)
        if step >= cfg.save_from and (step-cfg.save_from)%cfg.save_every == 0:
            save(f'step-{step}',step)
        progress(step,cfg.steps)
    return str(output/'last.safetensors'),records
