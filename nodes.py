"""ComfyUI node definitions for YuE2 LoRA training.

Node set (category "YuE2/Training"):
  1. YuE2 Train Model Loader   -> picks the base model + VAE folders
  2. YuE2 Training Dataset     -> scans an audio folder, caches VAE latents
  3. YuE2 LoRA Trainer         -> flow-matching LoRA training on the NAR branch
  4. YuE2 LoRA Merge (Export)  -> merges a LoRA into a new models/yue2 folder
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from types import SimpleNamespace

log = logging.getLogger("yue2_trainer.nodes")

CATEGORY = "YuE2/Training"


def _folder_paths():
    import folder_paths
    return folder_paths


def _model_paths():
    from .trainer_core.vendor import import_model_paths
    return import_model_paths()(_folder_paths())


def _free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _unload_comfy_models():
    try:
        import model_management
        model_management.unload_all_models()
        model_management.soft_empty_cache()
    except Exception:
        pass


class YuE2TrainModelLoader:
    """Select the YuE2 base model and VAE used for training."""

    @classmethod
    def INPUT_TYPES(cls):
        paths = _model_paths()
        models = paths.list("yue2") or ["No models found — see YuE2 Model Loader"]
        vaes = paths.list("yue2_vae") or ["No VAEs found — see YuE2 Model Loader"]
        return {"required": {
            "model": (models, {"tooltip": "YuE2 model folder (models/yue2), same as the inference loader."}),
            "vae": (vaes, {"tooltip": "YuE2 VAE folder (models/yue2_vae). Used once to encode your audio into latents."}),
        }}

    RETURN_TYPES = ("YUE2_TRAIN_BUNDLE",)
    RETURN_NAMES = ("train_bundle",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model, vae):
        paths = _model_paths()
        model_dir = paths.resolve("yue2", model)
        vae_dir = paths.resolve("yue2_vae", vae)
        return ({"model_dir": model_dir, "vae_dir": vae_dir,
                 "model_name": model, "vae_name": vae},)


class YuE2TrainingDataset:
    """Scan a folder of mp3/wav/flac (+ optional same-named .txt captions)
    and encode everything to cached YuE2 VAE latents."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "bundle": ("YUE2_TRAIN_BUNDLE",),
            "audio_folder": ("STRING", {"default": "",
                "tooltip": "Absolute path to the folder with your training songs "
                           "(mp3/wav/flac; optional .txt caption next to each file, same name)."}),
            "clip_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 60.0, "step": 0.5,
                "tooltip": "Length of each training clip. 10s = 250 latent frames. Shorter = less VRAM."}),
            "caption_mode": (["txt_file", "default", "none"], {
                "tooltip": "txt_file: use same-named .txt captions (empty if missing). "
                           "default: one caption for every clip. none: trigger word only."}),
            "default_caption": ("STRING", {"multiline": True, "default": "",
                "tooltip": "Used for every clip when caption_mode=default. Describe style/instruments/voice."}),
            "cache_folder": ("STRING", {"default": "",
                "tooltip": "Where to store encoded latents. Empty = <ComfyUI>/temp/yue2_latents."}),
            "force_reencode": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("YUE2_TRAIN_DATASET", "STRING")
    RETURN_NAMES = ("dataset", "summary")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, bundle, audio_folder, clip_seconds, caption_mode,
              default_caption, cache_folder, force_reencode):
        import torch
        from .trainer_core.vendor import import_yue2
        from .trainer_core import data as data_mod

        folder = Path(audio_folder.strip().strip('"'))
        if cache_folder.strip():
            cache_dir = Path(cache_folder.strip().strip('"'))
        else:
            fp = _folder_paths()
            cache_dir = Path(fp.get_temp_directory()) / "yue2_latents"
        if force_reencode and cache_dir.is_dir():
            for stale in cache_dir.glob("*.npy"):
                stale.unlink()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, modeling_vae, _, _ = import_yue2()
        vae = modeling_vae.YuE2VAE.from_pretrained(
            bundle["vae_dir"], decoder_only=False, device=device, local_files_only=True)
        try:
            dataset = data_mod.build_dataset(
                vae, folder, cache_dir, clip_seconds, caption_mode,
                default_caption, device)
        finally:
            del vae
            _free_memory()
        summary = dataset.summary()
        log.info("YuE2 dataset built:\n%s", summary)
        return (dataset, summary)


class YuE2LoRATrainer:
    """Train a LoRA on YuE2's NAR (acoustic) branch from the dataset."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "bundle": ("YUE2_TRAIN_BUNDLE",),
            "dataset": ("YUE2_TRAIN_DATASET",),
            "trigger_word": ("STRING", {"default": "mystyle",
                "tooltip": "Token/word placed at the start of the style prompt. "
                           "Use it in your style prompt at generation time (cot=off works best)."}),
            "steps": ("INT", {"default": 1000, "min": 1, "max": 100000}),
            "learning_rate": ("FLOAT", {"default": 1e-4, "min": 1e-7, "max": 1e-2, "step": 1e-6}),
            "rank": ("INT", {"default": 32, "min": 1, "max": 256}),
            "alpha": ("FLOAT", {"default": 32.0, "min": 0.1, "max": 512.0, "step": 0.1}),
            "lora_dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.01}),
            "target_preset": (["nar_attn_mlp", "nar_attn", "nar_attn_mlp_proj"], {
                "tooltip": "Which NAR layers get LoRA. nar_attn_mlp is the recommended default; "
                           "..._proj also adapts the latent projections (more VRAM/params)."}),
            "lora_name": ("STRING", {"default": "yue2_mystyle",
                "tooltip": "Output file name (<name>.safetensors) written into models/loras."}),
            "seed": ("INT", {"default": 1234, "min": 0, "max": 2**63 - 1}),
            "optimizer": (["adamw", "adamw_8bit"], {
                "tooltip": "adamw_8bit (bitsandbytes) saves VRAM; falls back to adamw if unavailable."}),
            "lr_scheduler": (["constant", "cosine"],),
            "warmup_steps": ("INT", {"default": 50, "min": 0, "max": 10000}),
            "grad_accum": ("INT", {"default": 1, "min": 1, "max": 64,
                "tooltip": "Gradient accumulation steps (effective batch = accum x 1 clip)."}),
            "caption_dropout": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.9, "step": 0.05,
                "tooltip": "Chance per step to train without trigger/caption (keeps the base style reachable)."}),
            "t_sampling": (["logit_normal", "uniform"], {
                "tooltip": "Timestep sampling for flow matching. logit_normal emphasizes mid-noise levels."}),
            "max_grad_norm": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            "log_every": ("INT", {"default": 10, "min": 1, "max": 1000}),
            "save_every": ("INT", {"default": 0, "min": 0, "max": 100000,
                "tooltip": "Save an intermediate LoRA every N steps (0 = only the final one)."}),
            "write_olm_format": ("BOOLEAN", {"default": False,
                "tooltip": "OFF (default): write the LoRA in native ComfyUI format "
                           "(loads with LoraLoaderModelOnly on the native YuE2 checkpoint). "
                           "ON: write the HF/Olm layout instead (needed for the "
                           "YuE2 LoRA Merge (Export) node)."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_path", "training_log")
    FUNCTION = "train"
    CATEGORY = CATEGORY

    def train(self, bundle, dataset, trigger_word, steps, learning_rate, rank,
              alpha, lora_dropout, target_preset, lora_name, seed, optimizer,
              lr_scheduler, warmup_steps, grad_accum, caption_dropout,
              t_sampling, max_grad_norm, log_every, save_every, write_olm_format):
        import torch
        from .trainer_core.vendor import import_yue2
        from .trainer_core import train as train_mod

        trigger_word = trigger_word.strip()
        lora_name = lora_name.strip()
        if not trigger_word:
            raise ValueError("trigger_word must not be empty")
        if not lora_name or any(c in lora_name for c in '\\/:*?"<>|'):
            raise ValueError("lora_name must be a valid file name (no path characters)")

        _unload_comfy_models()
        _free_memory()

        # ComfyUI executes nodes under torch.inference_mode() (execution.py);
        # training needs autograd, so explicitly re-enable grad mode for the
        # model load and the whole training loop (all tensors created inside
        # this block are normal, grad-capable tensors).
        with torch.inference_mode(False):
            modeling_yue2, _, protocol, tokenization_yue2 = import_yue2()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = modeling_yue2.YuE2ForCausalLM.from_pretrained(
                bundle["model_dir"], local_files_only=True,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            tokenizer = tokenization_yue2.YuE2TextTokenizer(Path(bundle["model_dir"]) / "qwen.tiktoken")

            fp = _folder_paths()
            output_dir = Path(fp.get_folder_paths("loras")[0])
            cfg = SimpleNamespace(
                device=device, steps=steps, learning_rate=learning_rate, rank=rank,
                alpha=alpha, lora_dropout=lora_dropout, target_preset=target_preset,
                lora_name=lora_name, seed=seed, optimizer=optimizer, weight_decay=0.01,
                lr_scheduler=lr_scheduler, warmup_steps=warmup_steps,
                grad_accum=grad_accum, caption_dropout=caption_dropout,
                t_sampling=t_sampling, max_grad_norm=max_grad_norm,
                log_every=log_every, save_every=save_every,
                trigger_word=trigger_word, output_dir=output_dir,
                base_model_name=bundle["model_name"],
                output_format="olm" if write_olm_format else "native",
            )
            log_lines = [
                f"YuE2 LoRA training: model={bundle['model_name']} clips={len(dataset.items)} "
                f"trigger={trigger_word!r} steps={steps} rank={rank} "
                f"format={'olm' if write_olm_format else 'native'}",
                f"output -> {output_dir}",
            ]
            try:
                lora_path = train_mod.run_training(model, tokenizer, protocol, dataset, cfg, log_lines)
            finally:
                del model
                _free_memory()
        return (lora_path, "\n".join(log_lines))


class YuE2LoRAMergeExport:
    """Merge a trained LoRA into the base model and export a new
    models/yue2/<name> folder usable by the standard YuE2 Model Loader."""

    @classmethod
    def INPUT_TYPES(cls):
        fp = _folder_paths()
        paths = _model_paths()
        loras = fp.get_filename_list("loras") or ["no loras found"]
        models = paths.list("yue2") or ["No models found"]
        return {"required": {
            "model": (models, {"tooltip": "Base model the LoRA was trained on."}),
            "lora": (loras, {"tooltip": "LoRA file from models/loras (as written by the trainer)."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            "output_name": ("STRING", {"default": "yue2-lora-merged",
                "tooltip": "Name of the new folder created inside models/yue2."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_model_folder",)
    FUNCTION = "merge"
    CATEGORY = CATEGORY

    def merge(self, model, lora, strength, output_name):
        from .trainer_core import merge as merge_mod
        fp = _folder_paths()
        paths = _model_paths()
        model_dir = paths.resolve("yue2", model)
        lora_path = Path(fp.get_full_path_or_raise("loras", lora))
        output_root = Path(model_dir).parent
        out_dir = merge_mod.merge_to_new_model(model_dir, lora_path, output_root,
                                               output_name, strength)
        _free_memory()
        return (str(out_dir),)


class YuE2LoRAConvertNative:
    """Convert a YuE2-Trainer LoRA to ComfyUI-native format so it loads with
    the standard LoraLoaderModelOnly on the native YuE2 checkpoint
    (checkpoints/yue2.safetensors -> CheckpointLoader -> native sampler)."""

    @classmethod
    def INPUT_TYPES(cls):
        fp = _folder_paths()
        loras = fp.get_filename_list("loras") or ["no loras found"]
        return {"required": {
            "lora": (loras, {"tooltip": "LoRA written by the YuE2 LoRA Trainer (models/loras)."}),
            "output_name": ("STRING", {"default": "yue2_mystyle_native",
                "tooltip": "File name for the converted LoRA (<name>.safetensors), also saved into models/loras."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("native_lora_path", "report")
    FUNCTION = "convert"
    CATEGORY = CATEGORY

    def convert(self, lora, output_name):
        from .trainer_core import convert as convert_mod
        fp = _folder_paths()
        lora_path = Path(fp.get_full_path_or_raise("loras", lora))
        output_name = output_name.strip()
        if not output_name or any(c in output_name for c in '\\/:*?"<>|'):
            raise ValueError("output_name must be a valid file name (no path characters)")
        out_path = Path(fp.get_folder_paths("loras")[0]) / f"{output_name}.safetensors"
        report = convert_mod.convert_lora_to_native(lora_path, out_path)
        lines = [
            f"converted {lora} -> {out_path.name}",
            f"tensors written: {report['tensors']}",
            "",
            f"{len(report['converted'])} module(s) converted:",
            *("  " + line for line in report["converted"]),
        ]
        if report["skipped"]:
            lines += ["", f"{len(report['skipped'])} module(s) skipped (no native equivalent):",
                      *("  " + line for line in report["skipped"])]
        lines += ["", "Use with: CheckpointLoader (yue2.safetensors) -> LoraLoaderModelOnly -> native YuE2 sampler."]
        return (str(out_path), "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "YuE2TrainModelLoader": YuE2TrainModelLoader,
    "YuE2TrainingDataset": YuE2TrainingDataset,
    "YuE2LoRATrainer": YuE2LoRATrainer,
    "YuE2LoRAMergeExport": YuE2LoRAMergeExport,
    "YuE2LoRAConvertNative": YuE2LoRAConvertNative,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2TrainModelLoader": "YuE2 Train Model Loader",
    "YuE2TrainingDataset": "YuE2 Training Dataset (audio folder)",
    "YuE2LoRATrainer": "YuE2 LoRA Trainer",
    "YuE2LoRAMergeExport": "YuE2 LoRA Merge (Export)",
    "YuE2LoRAConvertNative": "YuE2 LoRA Convert (to Native)",
}
