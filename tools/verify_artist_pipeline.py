"""Opt-in real audio -> AR training -> native ComfyUI logits/token diagnostic."""
import argparse
import ast
import gc
import json
from pathlib import Path
import sys


def definitions(path,names,namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    chosen = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in chosen} != set(names):
        raise ValueError('Expected upstream definitions are missing')
    # Only selected reviewed definitions; never execute upstream top-level loaders/training.
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(path),'exec'),namespace)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comfy-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream-source',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=4)
    args = p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(args.comfy_root.resolve()))
    import numpy as np
    import torch
    import soundfile as sf
    from trainer_core.mothersuperior.assets import prepare_assets
    from trainer_core.mothersuperior import semantic_data,tokenizer_head,ar_train
    from trainer_core.mothersuperior.regularizer import load_pack
    from trainer_core.mothersuperior.inference import apply_ar
    from trainer_core import native_ckpt
    from trainer_core.yue2_ref import protocol
    from safetensors.torch import load_file
    from safetensors import safe_open

    args.output.mkdir(parents=True,exist_ok=False)
    source = args.output/'input'
    source.mkdir()
    with sf.SoundFile(str(args.audio)) as f:
        rate = f.samplerate
        audio = f.read(round(args.seconds*rate),dtype='float32',always_2d=True)
    sf.write(str(source/'song.wav'),audio,rate,subtype='FLOAT')
    (source/'song.txt').write_text('diagnostic_song, traditional song',encoding='utf-8')
    assets,_ = prepare_assets(args.assets,False,True)
    dataset = semantic_data.prepare_dataset(assets,source,'diagnostic_song',args.output/'semantic_cache')
    print('SEMANTIC',dataset.summary,flush=True)
    cached = semantic_data.prepare_dataset(assets,source,'diagnostic_song',args.output/'semantic_cache')
    assert torch.equal(dataset.items[0]['codec'],cached.items[0]['codec'])
    features = load_file(dataset.items[0]['cache'])['mert']

    # MERT parity against the actual published function on the same audio/model.
    proc,mert = semantic_data.load_mert(assets,'cuda')
    ns = dict(torch=torch,np=np,proc=proc,mert=mert,dev='cuda')
    definitions(args.upstream_source/'prep_real.py',['mert_l20'],ns)
    upstream_features = ns['mert_l20'](semantic_data.read_audio(source/'song.wav'))
    np.testing.assert_array_equal(features.numpy(),upstream_features)
    del proc,mert,ns
    gc.collect(); torch.cuda.empty_cache()
    # Architecture and retained-window parity against ar_prep.py.
    ns = dict(torch=torch,np=np,nn=torch.nn,dev='cuda',VOCAB=32768,WIN=512,D=512,L=8,H=8)
    definitions(args.upstream_source/'ar_prep.py',['Tok','instnorm','predict'],ns)
    head = ns['Tok'](1024).cuda().eval()
    head.load_state_dict(load_file(assets.head))
    ns['head'] = head
    expected = ns['predict'](ns['instnorm'](upstream_features))
    np.testing.assert_array_equal(dataset.items[0]['codec'].numpy(),expected)
    del ns,head
    gc.collect(); torch.cuda.empty_cache()

    model = ar_train.build_model(args.checkpoint)
    tokenizer = native_ckpt.YuE2JsonTokenizer(native_ckpt.load_native_tokenizer_json(args.checkpoint))
    pack = load_pack(assets.regularizer)
    # Bound smoke-test sequence length, keep genuine regularizer records and splits.
    cfg = ar_train.TrainConfig(steps=3,rank=4,max_length=2048,evaluate_every=3,seed=1,
        live_curve=True,live_curve_path=str(args.output/'live_curve.png'))
    lora_path,training = ar_train.train(model,tokenizer,dataset.items,pack,args.output/'training',cfg)
    del model,tokenizer,pack
    gc.collect(); torch.cuda.empty_cache()
    print('TRAINED',lora_path,flush=True)

    import comfy.sd
    import comfy.model_management as mm
    _,clip,_,_ = comfy.sd.load_checkpoint_guess_config(str(args.checkpoint),output_vae=False,output_clip=True,
                                                      output_model=False,disable_dynamic=True)
    tensors = load_file(lora_path)
    logits_list,weights,histories,runs = [],[],[],[]
    for strength in (0.,1.,2.,0.):
        patched,report = apply_ar(clip,tensors,strength)
        mm.load_models_gpu([patched.patcher])
        te = patched.cond_stage_model
        te.set_clip_options({'execution_device':patched.patcher.load_device})
        tokens = patched.tokenize('diagnostic_song, traditional song',lyrics='[instrumental]',cot='off',seed=42)
        prefix = tokens['prefix'] + [protocol.ABC_END,protocol.MUSIC_START]
        dtype = te.model.embed_tokens.weight.dtype
        key = 'model.layers.0.self_attn.qkv_proj.weight'
        weights.append(te.state_dict()[key].detach().float().cpu().clone())
        with torch.inference_mode():
            logits,cache,_ = te._prefill([prefix],len(prefix)+8,dtype)
            logits_list.append(logits.float().cpu())
            del cache
            history,_ = te._generate(prefix,42,8,'music',dtype,temperature=1.,top_p=.95,top_k=100,
                repetition_penalty=1.2,penalty_window=100,min_tokens=8)
        histories.append([int(x)-protocol.CODEC_OFFSET for x in history])
        assert all(0 <= x < 32768 for x in histories[-1])
        diff = (logits_list[-1]-logits_list[0]).abs()
        runs.append(dict(strength=strength,patch_report=json.loads(report),
            logits_max_abs_diff=diff.max().item(),logits_mean_abs_diff=diff.mean().item(),
            weight_max_abs_diff=(weights[-1]-weights[0]).abs().max().item(),semantic_ids=histories[-1]))
        print('AR EFFECT', {k:v for k,v in runs[-1].items() if k!='patch_report'},flush=True)
        mm.unload_all_models()
        del te,patched
    assert torch.equal(weights[0],weights[3]) and torch.equal(logits_list[0],logits_list[3])
    assert histories[0] == histories[3]
    assert not torch.equal(logits_list[0],logits_list[1]), 'AR strength 1 has no effect'
    assert not torch.equal(logits_list[1],logits_list[2]), 'AR strengths 1 and 2 are identical'
    tolerance = 4*torch.finfo(dtype).eps*max(w.abs().max().item() for w in weights)
    torch.testing.assert_close(weights[2]-weights[0],2*(weights[1]-weights[0]),atol=tolerance,rtol=.02)
    result = dict(audio=str(args.audio),seconds=args.seconds,semantic=dataset.summary,
        cache_hit=cached.summary,upstream_mert_exact=True,upstream_head_ids_exact=True,
        training=training,runs=runs,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    (args.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    del clip
    gc.collect()


if __name__ == '__main__':
    main()
