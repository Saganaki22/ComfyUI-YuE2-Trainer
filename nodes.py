"""ComfyUI node definitions for YuE2 LoRA training.

Node set (category "YuE2/Training"):
  1. YuE2 Training Dataset  -> scans an audio folder, caches VAE latents
                               (VAE comes from the native checkpoint)
  2. YuE2 LoRA Trainer      -> flow-matching LoRA training on the NAR branch;
                               writes native ComfyUI LoRAs (LoraLoaderModelOnly
                               on the YuE2 checkpoint)

Both nodes load everything (AR model, NAR branch, VAE, tokenizer) directly
from the native all-in-one checkpoint in models/checkpoints — the same file
used for generation. No separate model/VAE folders are needed.
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


def _checkpoint_choices():
    fp = _folder_paths()
    return fp.get_filename_list("checkpoints") or ["no checkpoints found — see README"]


def _resolve_yue2_checkpoint(name: str) -> Path:
    """Resolve a checkpoints-folder entry and validate it is the native
    YuE2 all-in-one bf16 checkpoint."""
    from .trainer_core import native_ckpt
    fp = _folder_paths()
    ckpt_path = Path(fp.get_full_path_or_raise("checkpoints", name))
    if not native_ckpt.is_native_yue2_checkpoint(ckpt_path):
        raise ValueError(
            f"{name} is not a native YuE2 all-in-one checkpoint. Use the bf16 "
            "YuE2 checkpoint (checkpoints/yue2_3b_bf16.safetensors from "
            "huggingface.co/Comfy-Org/YuE2); quantized or non-YuE2 checkpoints "
            "cannot be trained.")
    return ckpt_path


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


class YuE2TrainingDataset:
    """Scan a folder of mp3/wav/flac (+ optional same-named .txt captions)
    and encode everything to cached YuE2 VAE latents."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "checkpoint": (_checkpoint_choices(), {
                "tooltip": "Native YuE2 all-in-one checkpoint (models/checkpoints) — "
                           "its built-in VAE encodes your audio. Same file you "
                           "use for generation."}),
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

    def build(self, checkpoint, audio_folder, clip_seconds, caption_mode,
              default_caption, cache_folder, force_reencode):
        import torch
        from .trainer_core.vendor import import_yue2
        from .trainer_core import data as data_mod, native_ckpt

        ckpt_path = _resolve_yue2_checkpoint(checkpoint)
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
        vae = native_ckpt.build_vae_from_native(ckpt_path, modeling_vae, device=device)
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
            "dataset": ("YUE2_TRAIN_DATASET",),
            "checkpoint": (_checkpoint_choices(), {
                "tooltip": "Native YuE2 all-in-one checkpoint (models/checkpoints) "
                           "the LoRA is trained on — pick the same file you "
                           "generate with."}),
            "trigger_word": ("STRING", {"default": "mystyle",
                "tooltip": "Token/word placed at the start of the style prompt. "
                           "Use it in your style prompt at generation time (cot=off works best)."}),
            "steps": ("INT", {"default": 3000, "min": 1, "max": 100000}),
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
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_path", "training_log")
    FUNCTION = "train"
    CATEGORY = CATEGORY

    def train(self, dataset, checkpoint, trigger_word, steps, learning_rate, rank,
              alpha, lora_dropout, target_preset, lora_name, seed, optimizer,
              lr_scheduler, warmup_steps, grad_accum, caption_dropout,
              t_sampling, max_grad_norm, log_every, save_every):
        import torch
        from .trainer_core.vendor import import_yue2
        from .trainer_core import train as train_mod, native_ckpt

        trigger_word = trigger_word.strip()
        lora_name = lora_name.strip()
        if not trigger_word:
            raise ValueError("trigger_word must not be empty")
        if not lora_name or any(c in lora_name for c in '\\/:*?"<>|'):
            raise ValueError("lora_name must be a valid file name (no path characters)")

        ckpt_path = _resolve_yue2_checkpoint(checkpoint)
        _unload_comfy_models()
        _free_memory()

        # ComfyUI executes nodes under torch.inference_mode() (execution.py);
        # training needs autograd, so explicitly re-enable grad mode for the
        # model load and the whole training loop (all tensors created inside
        # this block are normal, grad-capable tensors).
        with torch.inference_mode(False):
            modeling_yue2, _, protocol, _ = import_yue2()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = native_ckpt.build_lm_from_native(
                ckpt_path, modeling_yue2, torch_dtype=torch.bfloat16)
            tokenizer = native_ckpt.YuE2JsonTokenizer(
                native_ckpt.load_native_tokenizer_json(ckpt_path))

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
                base_model_name=checkpoint,
            )
            log_lines = [
                f"YuE2 LoRA training: checkpoint={checkpoint} clips={len(dataset.items)} "
                f"trigger={trigger_word!r} steps={steps} rank={rank} "
                f"format=native (LoraLoaderModelOnly)",
                f"output -> {output_dir}",
            ]
            try:
                lora_path = train_mod.run_training(model, tokenizer, protocol, dataset, cfg, log_lines)
            finally:
                del model
                _free_memory()
        return (lora_path, "\n".join(log_lines))


NODE_CLASS_MAPPINGS = {
    "YuE2TrainingDataset": YuE2TrainingDataset,
    "YuE2LoRATrainer": YuE2LoRATrainer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2TrainingDataset": "YuE2 Training Dataset (audio folder)",
    "YuE2LoRATrainer": "YuE2 LoRA Trainer",
}
