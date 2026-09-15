"""ComfyUI schemas for the recommended score-free artist-training path."""
import gc
import json
from pathlib import Path

import torch

CATEGORY = 'YuE2/Training/Artist Training (recommended)'


def interrupt():
    from comfy import model_management
    model_management.throw_exception_if_processing_interrupted()


def release_models():
    from comfy import model_management
    model_management.unload_all_models()
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def progress_callback():
    from comfy.utils import ProgressBar
    bar = ProgressBar(1)
    return lambda n,total:bar.update_absolute(n,total)


def make_report(unique_id):
    """WebSocket reporter for the in-node live widget; None when headless."""
    if unique_id is None:
        return None
    try:
        from server import PromptServer
        if PromptServer.instance is None:
            return None
    except Exception:
        return None
    server = PromptServer.instance
    node = str(unique_id)
    def report(event):
        try:
            server.send_sync('yue2.training.progress', {'node': node, **event})
        except Exception:
            pass
    return report


def _register_training_model(model,rank):
    """Register the training model in ComfyUI's loaded-models cache so memory
    visualizers show its live VRAM as a model bar. Display-only: the trainer
    manages devices itself and this static patcher is never dynamically
    offloaded mid-run."""
    try:
        import torch
        import comfy.model_management as mm
        import comfy.model_patcher as mp
        try:
            from .trainer_core.mothersuperior.ar_train import TARGETS
        except ImportError:
            from trainer_core.mothersuperior.ar_train import TARGETS
        if not torch.cuda.is_available():
            return None
        device = mm.get_torch_device()
        # Bytes the training loop actually holds on the GPU: everything
        # build_model moved, plus the LoRA A/B that inject() will add.
        size = sum(t.numel()*t.element_size() for t in model.parameters() if t.is_cuda)
        for name,mod in model.model.named_modules():
            if any(name.endswith('.'+t) for t in TARGETS):
                size += rank*(mod.in_features+mod.out_features)*4
        patcher = mp.ModelPatcher(model.model,device,torch.device('cpu'),size=size)
        mm.load_models_gpu([patcher])
        return patcher
    except Exception:
        return None


def _unregister_training_model(patcher):
    if patcher is None:
        return
    try:
        import comfy.model_management as mm
        mm.unload_model_and_clones(patcher)
    except Exception:
        pass


class YuE2MothersuperiorAssets:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'auto_download':('BOOLEAN',{'default':False,'tooltip':
                'Download pinned MERT-v2-FullSong plus the pre-converted safetensors '
                '(tokenizer head, minted regularizer pack) when absent. ~2 GB on first '
                'run, then cached forever. Default: off — turn on for first setup.'}),
            'include_nar_adapter':('BOOLEAN',{'default':False,'tooltip':
                'Also download the pretrained NAR adapter safetensors (rank-32 LoRAs plus '
                'full vae2llm/llm2vae replacement weights). Only needed for NAR-side '
                'experiments, not for AR artist training. Default: off.'}),
        },'optional':{
            'asset_directory':('STRING',{'default':'','tooltip':
                'Folder for the downloaded safetensors and the HF cache. '
                'Default: blank = models/yue2_trainer/mothersuperior. Existing assets are reused.'}),
        }}
    RETURN_TYPES = ('YUE2_MS_ASSETS','STRING')
    RETURN_NAMES = ('assets','summary')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY

    def prepare(self,auto_download,include_nar_adapter,asset_directory=''):
        from .trainer_core.mothersuperior.assets import prepare_assets
        return prepare_assets(asset_directory or None,auto_download,include_nar_adapter,interrupt)


class YuE2RealAudioSemanticDataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'assets':('YUE2_MS_ASSETS',{'tooltip':'Connect YuE2 Mothersuperior Assets.'}),
            'audio_folder':('STRING',{'default':'','tooltip':
                'Absolute path to a folder of songs: <name>.flac/wav/mp3 plus optional '
                '<name>.txt (style caption; should start with the trigger word) and '
                '<name>.lyrics.txt (full lyrics with [verse]/[chorus] tags). '
                'Default: blank — must be set.'}),
            'trigger_word':('STRING',{'default':'my_artist','tooltip':
                'Artist token prepended to the style caption when the caption does not '
                'already start with it. Use the same word in the style prompt at generation. '
                'Default: my_artist.'}),
            'force_reencode':('BOOLEAN',{'default':False,'tooltip':
                'Ignore cached MERT features/semantic IDs and re-encode every song. '
                'Use after changing audio files, the head, or preprocessing. Default: off.'}),
        },'optional':{
            'semantic_cache_folder':('STRING',{'default':'','tooltip':
                'Where extracted features and semantic IDs are cached. '
                'Default: blank = <asset folder>/semantic_cache.'}),
            'lyrics_required':('BOOLEAN',{'default':False,'tooltip':
                'Skip songs that have no <name>.lyrics.txt. Default: off — missing lyrics '
                'fall back to [instrumental], which is fine for style-only runs but ruins '
                'vocal artist training. Recommended: on when training vocals.'}),
            'mert_batch_size':('INT',{'default':1,'min':1,'max':32,'tooltip':
                'Number of full 30-second MERT windows processed together. Higher is faster '
                'but uses more VRAM (each window is a full MERT forward). '
                'Default: 1. Recommended: 1-4 on 24 GB.'}),
        }}
    RETURN_TYPES = ('YUE2_SEMANTIC_DATASET','STRING')
    RETURN_NAMES = ('dataset','summary')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY

    def prepare(self,assets,audio_folder,trigger_word,force_reencode,semantic_cache_folder='',lyrics_required=False,mert_batch_size=1):
        from .trainer_core.mothersuperior.semantic_data import prepare_dataset
        release_models()
        result = prepare_dataset(assets,audio_folder,trigger_word,
            semantic_cache_folder or str(Path(assets.directory)/'semantic_cache'),
            force_reencode=force_reencode,lyrics_required=lyrics_required,mert_batch_size=mert_batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',check_interrupt=interrupt,progress=progress_callback())
        return result,json.dumps(result.summary,indent=2)


class YuE2ArtistARLoRATrainer:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {'required':{
            'dataset':('YUE2_SEMANTIC_DATASET',{'tooltip':'Connect YuE2 Real Audio Semantic Dataset.'}),
            'assets':('YUE2_MS_ASSETS',{'tooltip':'Connect YuE2 Mothersuperior Assets (provides the minted regularizer pack).'}),
            'checkpoint':(folder_paths.get_filename_list('checkpoints'),{'tooltip':
                'Native YuE2 all-in-one checkpoint (models/checkpoints) — the same file '
                'you generate with (yue2_3b_bf16.safetensors). Quantized checkpoints are rejected.'}),
            'output_name':('STRING',{'default':'artist_ar','tooltip':
                'New subfolder name in models/loras. Saves last.safetensors, best.safetensors '
                'and step-N checkpoints. Existing folders are never overwritten — pick a fresh '
                'name per run. Default: artist_ar.'}),
            'steps':('INT',{'default':1600,'min':1,'max':100000,'tooltip':
                'Total optimizer steps; upstream compares checkpoints from step 600 and '
                'later reported 5000 steps applied at strength 2.0 works best. Earlier '
                'guidance warned of memorization past ~1500 on small sets — the minted '
                'regularizer keeps grammar safe either way; watch the val line and pick '
                'a checkpoint by ear. Default: 1600.'}),
            'rank':('INT',{'default':64,'min':1,'max':256,'tooltip':
                'LoRA rank of the float32 A/B matrices (attention q/k/v/o + MLP gate/up/down, '
                'all 28 AR layers). Upstream trains rank 64. '
                'Default: 64. Recommended: 32-64; higher overfits sooner.'}),
            'learning_rate':('FLOAT',{'default':1e-4,'min':1e-7,'max':.1,'step':1e-5,'tooltip':
                'AdamW learning rate with 50-step warmup and cosine decay to a 0.2 floor. '
                'Default: 1e-4 (upstream recipe). If the LoRA has no effect, try 2e-4; '
                'if minted_val_loss climbs, lower it.'}),
            'artist_ratio':('FLOAT',{'default':.5,'min':0.,'max':1.,'step':.05,'tooltip':
                'Probability of sampling an artist example per accumulation step; the rest are '
                'minted regularizer records so a small artist set cannot collapse YuE2\'s token '
                'grammar. Default: 0.5 (upstream 50/50). Keep at 0.5 unless you have many songs.'}),
            'seed':('INT',{'default':1,'min':0,'max':2**31-1,'tooltip':
                'Seed for example sampling, noise and timestep draws. '
                'Default: 1. Change it for a different run of the same config.'}),
        },'optional':{
            'grad_accum':('INT',{'default':2,'min':1,'max':64,'tooltip':
                'Examples per optimizer step (upstream accumulates 2). '
                'Default: 2. Lower to 1 if VRAM is tight, raise for smoother gradients.'}),
            'max_length':('INT',{'default':12288,'min':128,'max':24576,'tooltip':
                'Prompt plus semantic sequence budget in tokens; MUSIC_END is included only '
                'if the whole codec sequence fits. Default: 12288 (upstream). Longer codec '
                'sequences are truncated to the budget.'}),
            'evaluate_every':('INT',{'default':100,'min':1,'max':10000,'tooltip':
                'Run the held-out evaluation (artist_loss + minted_val_loss) every N steps. '
                'Default: 100. minted_val_loss must stay flat — a rising value means the '
                'LoRA is damaging the token grammar.'}),
            'save_from':('INT',{'default':600,'min':0,'max':100000,'tooltip':
                'First step that gets a numbered step-N.safetensors checkpoint, then every '
                'save_from+save_every. Default: 600 (upstream recipe for ~1600 steps). '
                'Small artist sets memorize early — set 100-200 to catch the pre-memorization '
                'zone. 0 disables numbered checkpoints (best/last still save).'}),
            'save_every':('INT',{'default':200,'min':1,'max':10000,'tooltip':
                'Spacing between numbered checkpoints once save_from is reached. '
                'Default: 200. Use 50-100 for fine-grained checkpoint picking by ear.'}),
            'resume_from':('STRING',{'default':'','tooltip':
                'Path to an existing run folder (models/loras/<name>) to continue it after '
                'a cancel or crash. Cancelled runs auto-save an interrupt checkpoint and '
                'full trainer state, so nothing since the last step is lost. On resume, '
                'steps is the target TOTAL and must exceed the resumed step. Blank = fresh run; '
                'output_name is ignored when resuming (checkpoints write back into this folder).'}),
            'resume_optimizer':('BOOLEAN',{'default':True,'tooltip':
                'On resume, restore the AdamW moments for an exact continuation. '
                'Default: on. Off = weights-only resume (fresh optimizer, slightly different '
                'trajectory, ~40% smaller state file).'}),
            'live_curve':('BOOLEAN',{'default':True,'tooltip':
                'Rewrite a live chart PNG every few seconds while training runs; the '
                'YuE2 Training Curve node in this workflow displays it in real time '
                '(needs matplotlib; auto-disables with a log note if unavailable). '
                'Default: on. No effect on training itself.'}),
        },'hidden':{
            'unique_id':('UNIQUE_ID',),
        }}
    RETURN_TYPES = ('STRING','STRING')
    RETURN_NAMES = ('ar_lora_path','training_log')
    FUNCTION = 'train'
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def train(self,dataset,assets,checkpoint,output_name,steps,rank,learning_rate,artist_ratio,seed,
              grad_accum=2,max_length=12288,evaluate_every=100,save_from=600,save_every=200,
              resume_from='',resume_optimizer=True,live_curve=True,unique_id=None):
        import folder_paths
        from .trainer_core import native_ckpt
        from .trainer_core.mothersuperior import ar_train
        from .trainer_core.mothersuperior.regularizer import load_pack
        if resume_from.strip():
            target = Path(resume_from.strip().strip('"'))
            if not target.is_dir():
                raise ValueError(f'resume folder not found: {target}')
        else:
            if not output_name or output_name in ('.','..') or any(c in output_name for c in '\\/:*?"<>|'):
                raise ValueError('output_name must be a single valid folder name')
            target = Path(folder_paths.get_folder_paths('loras')[0])/output_name
            if target.exists(): raise FileExistsError(f'Choose a new output_name: {target}')
        ckpt = folder_paths.get_full_path_or_raise('checkpoints',checkpoint)
        if not native_ckpt.is_native_yue2_checkpoint(ckpt): raise ValueError('Select the native bf16 YuE2 checkpoint')
        regularizer = load_pack(assets.regularizer)
        release_models()
        with torch.inference_mode(False),torch.enable_grad():
            model = ar_train.build_model(ckpt,'cuda' if torch.cuda.is_available() else 'cpu')
            tokenizer = native_ckpt.YuE2JsonTokenizer(native_ckpt.load_native_tokenizer_json(ckpt))
            cfg = ar_train.TrainConfig(steps=steps,rank=rank,learning_rate=learning_rate,artist_ratio=artist_ratio,
                seed=seed,grad_accum=grad_accum,max_length=max_length,evaluate_every=evaluate_every,
                save_from=save_from,save_every=save_every,
                resume_from=str(target) if resume_from.strip() else '',
                resume_optimizer=resume_optimizer,
                live_curve=live_curve,live_curve_path=str(Path(folder_paths.get_temp_directory())/'yue2_curve_live.png'))
            try:
                patcher = _register_training_model(model,rank)
                path,records = ar_train.train(model,tokenizer,dataset.items,regularizer,target,cfg,
                                              interrupt,progress_callback(),report=make_report(unique_id))
            finally:
                _unregister_training_model(patcher)
                del model
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        return path,'\n'.join(json.dumps(row) for row in records)


class YuE2TrainingMonitor:
    """Live training monitor: place anywhere in the workflow. It listens for
    the trainer's WebSocket progress events and charts them in real time -
    no input connection needed (training runs are identified by run name)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{},
                'optional':{
                    'run':('STRING',{'default':'','tooltip':
                        'Only show a training run whose output folder name matches this '
                        '(e.g. tupac_ar_v1). Blank = follow the most recent active run.'}),
                }}
    RETURN_TYPES = ()
    FUNCTION = 'monitor'
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def monitor(self,run=''):
        return ()


NODE_CLASS_MAPPINGS = {cls.__name__:cls for cls in (YuE2MothersuperiorAssets,YuE2RealAudioSemanticDataset,YuE2ArtistARLoRATrainer,YuE2TrainingMonitor)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'YuE2MothersuperiorAssets':'YuE2 Mothersuperior Assets',
    'YuE2RealAudioSemanticDataset':'YuE2 Real Audio Semantic Dataset',
    'YuE2ArtistARLoRATrainer':'YuE2 Artist AR LoRA Trainer',
    'YuE2TrainingMonitor':'YuE2 Training Monitor (live)',
}
