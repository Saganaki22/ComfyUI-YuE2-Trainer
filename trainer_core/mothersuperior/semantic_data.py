"""Real audio preprocessing adapted from Mothersuperior prep_real.py/ar_prep.py.

CC BY-NC 4.0 repository; see THIRD_PARTY_NOTICES.md. Does not require the VAE.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import gc
import hashlib
import json
from math import gcd
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .serialization import sha256, save_tensors
from .tokenizer_head import load_head, predict

PREPROCESS_VERSION = 'mert-l20-30s-linear25-fp16-instnorm-fortran-rotary-v3'
EXTENSIONS = {'.wav','.flac','.mp3','.ogg','.m4a','.aac','.aiff','.aif'}


@dataclass
class SemanticDataset:
    items: list
    summary: dict


def cache_key(path, head_hash, mert_revision, settings=None):
    path = Path(path).resolve()
    stat = path.stat()
    identity = dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                    head=head_hash,mert=mert_revision,preprocess=PREPROCESS_VERSION,settings=settings or {})
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def interpolate_features(features, samples):
    frames = int(round(samples/24000*25))
    if frames < 1:
        raise ValueError('Audio too short')
    return F.interpolate(features.float().T[None],size=frames,mode='linear',align_corners=False)[0].T.half().cpu()


def read_audio(path, max_seconds=0):
    import soundfile as sf
    from scipy.signal import resample_poly
    with sf.SoundFile(str(path)) as handle:
        sr = handle.samplerate
        audio = handle.read(frames=round(max_seconds*sr) if max_seconds else -1,
                            dtype='float32',always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Empty or non-finite audio')
    mono = audio.mean(1)
    factor = gcd(sr,24000)
    return resample_poly(mono,24000//factor,sr//factor).astype(np.float32)


def load_mert(assets, device):
    from transformers import AutoFeatureExtractor, AutoModel
    # Execute only the pinned, explicitly requested model source in this snapshot.
    processor = AutoFeatureExtractor.from_pretrained(assets.mert,trust_remote_code=True,local_files_only=True)
    model = AutoModel.from_pretrained(assets.mert,trust_remote_code=True,local_files_only=True).to(device).eval()
    # Transformers 5's meta-loading can leave this nonpersistent buffer
    # uninitialized. It is not a checkpoint weight: reconstruct it with the
    # exact pinned upstream constructor, which also clears the RoPE caches.
    model.embed_positions = type(model.embed_positions)(model.config).to(device)
    return processor, model


@torch.inference_mode()
def extract_features(mono24, processor, model, batch_size=1, check_interrupt=lambda:None):
    chunks = [mono24[s:s+720000] for s in range(0,len(mono24),720000)]
    chunks = [c for c in chunks if len(c) >= 24000]
    if not chunks:
        raise ValueError('Upstream MERT preprocessing requires at least one second of audio')
    full = [c for c in chunks if len(c)==720000]
    tail = [c for c in chunks if len(c)<720000]
    groups = [full[s:s+batch_size] for s in range(0,len(full),batch_size)] + [[c] for c in tail]
    device = next(model.parameters()).device
    features = []
    for group in groups:
        check_interrupt()
        inputs = {k:v.to(device) for k,v in processor(group,sampling_rate=24000,return_tensors='pt').items()}
        ctx = torch.autocast('cuda',dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with ctx:
            output = model(**inputs,output_hidden_states=True).hidden_states[20]
        features.append(output.reshape(-1,1024).float().cpu())
    return interpolate_features(torch.cat(features),len(mono24))


def prepare_dataset(assets, audio_folder, trigger_word, cache_folder, *, force_reencode=False,
                    lyrics_required=False, max_seconds=0, device='cuda', mert_batch_size=1,
                    check_interrupt=lambda:None, progress=lambda n,total:None):
    paths = sorted(p for p in Path(audio_folder).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError('No supported audio files found')
    cache = Path(cache_folder)
    cache.mkdir(parents=True,exist_ok=True)
    head_hash = sha256(assets.head)
    settings = dict(max_seconds=max_seconds,mert_batch_size=mert_batch_size,device=device)
    summary = dict(songs=0,duration_seconds=0,semantic_tokens=0,lyrics_present=0,lyrics_missing=0,
                   style_present=0,style_missing=0,cache_hits=0,cache_misses=0,errors=[])
    pending, items = [], []
    for path in paths:
        check_interrupt()
        lyrics_path, style_path = path.with_suffix('.lyrics.txt'),path.with_suffix('.txt')
        if lyrics_required and not lyrics_path.exists():
            summary['errors'].append(f'{path.name}: missing required lyrics')
            continue
        lyrics = lyrics_path.read_text(encoding='utf-8').strip() if lyrics_path.exists() else '[instrumental]'
        caption = style_path.read_text(encoding='utf-8') if style_path.exists() else ''
        caption = ' '.join(caption.split('===LYRICS===')[0].replace('Global Metadata:','').split())[:1500]
        style = caption if not trigger_word or caption.startswith(trigger_word) else f'{trigger_word}, {caption}'.rstrip(', ')
        key = cache_key(path,head_hash,assets.mert_revision,settings)
        item = dict(name=path.stem,src='artist',style=style,lyrics=lyrics,audio=str(path.resolve()),
                    cache=str(cache/(key+'.safetensors')))
        if Path(item['cache']).exists() and not force_reencode:
            tokens = load_file(item['cache'])['codec']
            if tokens.dtype != torch.int32 or tokens.ndim != 1 or not len(tokens) or tokens.min() < 0 or tokens.max() >= 32768:
                raise ValueError(f'Corrupt semantic cache: {item["cache"]}')
            item['codec'] = tokens
            summary['cache_hits'] += 1
        else:
            pending.append(item)
        items.append(item)
        summary['lyrics_present' if lyrics_path.exists() else 'lyrics_missing'] += 1
        summary['style_present' if style_path.exists() else 'style_missing'] += 1
    # Stage MERT and head separately; never keep MERT+YuE2 in GPU memory together.
    if pending:
        processor, mert = load_mert(assets,device)
        try:
            for i,item in enumerate(pending):
                check_interrupt()
                mono = read_audio(item['audio'],max_seconds)
                features = extract_features(mono,processor,mert,mert_batch_size,check_interrupt)
                item['_features'] = features
                progress(i+1,len(pending)*2)
        finally:
            del mert,processor
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        head = load_head(assets.head,device)
        try:
            for i,item in enumerate(pending):
                check_interrupt()
                features = item.pop('_features')
                codec = predict(features.numpy(),head,check_interrupt)
                save_tensors(item['cache'],dict(codec=codec,mert=features),
                             dict(format='yue2-semantic-cache-v1',head_sha256=head_hash,preprocess=PREPROCESS_VERSION))
                item['codec'] = codec
                summary['cache_misses'] += 1
                progress(len(pending)+i+1,len(pending)*2)
        finally:
            del head
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    if not items:
        raise ValueError('No usable songs: '+str(summary['errors']))
    summary['songs'] = len(items)
    summary['semantic_tokens'] = sum(len(item['codec']) for item in items)
    summary['duration_seconds'] = summary['semantic_tokens']/25
    return SemanticDataset(items,summary)
