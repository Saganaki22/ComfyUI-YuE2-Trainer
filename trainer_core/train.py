"""Flow-matching LoRA training loop for the YuE2 NAR branch.

Reimplements the released ``YuE2ForCausalLM.nar_velocity`` forward with
gradients enabled (that method is ``@torch.no_grad`` inference-only) and
trains LoRA adapters on the NAR attention/MLP projections with the
conditional flow-matching MSE loss:

    x_t = (1 - t) * z + t * noise        (t=1: pure noise, t=0: data)
    v*  = noise - z                      (matches the released ODE solver,
                                          which integrates t from 1 down to 0)

Conditioning is the checkpoint-native cot="off" text prefix
(EOD + instruction + [Tags] style + [Lyrics] + ABC_START/ABC_END +
MUSIC_START/MUSIC_END) followed by LATENT_START .. LATENT_END positions,
with the MoT hybrid attention mask exactly as in the released code.
"""
from __future__ import annotations

import logging
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import data as data_mod
from . import lora as lora_mod

log = logging.getLogger("yue2_trainer.train")

# Filled by import_yue2() at node execution time.
_protocol = None


def _comfy_progress(total):
    try:
        from comfy.utils import ProgressBar
        return ProgressBar(total)
    except Exception:
        return None


def _check_interrupt():
    try:
        import model_management
        model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        return


def build_conditioning_ids(tokenizer, protocol, style: str) -> list[int]:
    """[EOD] + text + ABC_START/ABC_END + MUSIC_START/MUSIC_END (cot=off)."""
    request = protocol.SongRequest(style=style, lyrics="", cot="off", seed=0)
    prefix = protocol.token_prefixes(request, tokenizer)
    return prefix + [protocol.MUSIC_END]


class MaskCache:
    """Hybrid MoT attention masks, cached by sequence shape."""

    def __init__(self):
        self._cache = {}

    def get(self, ar_len: int, nar_len: int, device, dtype):
        key = (ar_len, nar_len, device)
        if key not in self._cache:
            seq = ar_len + nar_len
            ar_mask = torch.zeros(1, seq, dtype=torch.bool, device=device)
            ar_mask[0, :ar_len] = True
            nar_mask = ~ar_mask
            ar_q = ar_mask.unsqueeze(2).float()
            ar_k = ar_mask.unsqueeze(1).float()
            nar_q = nar_mask.unsqueeze(2).float()
            nar_k = nar_mask.unsqueeze(1).float()
            causal = torch.tril(torch.ones(seq, seq, device=device))
            mask = (ar_q * ar_k * causal) + (nar_q * ar_k) + (nar_q * nar_k)
            attn_mask = mask.unsqueeze(1)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
            attn_mask = attn_mask.masked_fill(attn_mask > 0, 0.0)
            self._cache[key] = (ar_mask, attn_mask)
        return self._cache[key]


def training_velocity(model, cond_ids: list[int], x_t: torch.Tensor, raw_t: float,
                      mask_cache: MaskCache, protocol) -> torch.Tensor:
    """Grad-enabled twin of YuE2ForCausalLM.nar_velocity (text-only regime).

    cond_ids: AR conditioning tokens incl. MUSIC_END (no codec tokens).
    x_t:      [T_lat, 64] noised latents on the model device.
    returns:  [T_lat, 64] predicted velocity.
    """
    device = x_t.device
    dtype = next(model.parameters()).dtype
    ar_len = len(cond_ids)
    t_lat = x_t.shape[0]
    n_nar = t_lat + 2  # LATENT_START + content + LATENT_END
    seq = ar_len + n_nar
    if seq > model.config.max_position_embeddings:
        raise ValueError(f"Sequence {seq} exceeds context {model.config.max_position_embeddings}")

    tokens = torch.full((1, seq), protocol.LATENT_START, dtype=torch.long, device=device)
    tokens[0, :ar_len] = torch.tensor(cond_ids, dtype=torch.long, device=device)
    tokens[0, -1] = protocol.LATENT_END

    token_emb = model.model.embed_tokens(tokens)

    t_shifted = model._shift_t_value(raw_t, device, dtype)
    x_nar = torch.zeros(n_nar, x_t.shape[1], device=device, dtype=dtype)
    x_nar[1:1 + t_lat] = x_t.to(dtype)
    latent_hidden = model.vae2llm(x_nar.unsqueeze(0))
    latent_hidden = latent_hidden + model.time_embedder(t_shifted.expand(n_nar)).unsqueeze(0)
    pos_ids = torch.arange(n_nar, device=device).clamp(max=model.config.max_latent_frames - 1)
    latent_hidden = latent_hidden + model.latent_pos_embed(pos_ids).unsqueeze(0)

    token_emb = token_emb.clone()
    token_emb[0, ar_len:] = latent_hidden[0]

    ar_mask, attn_mask = mask_cache.get(ar_len, n_nar, device, dtype)
    position_ids = torch.arange(seq, device=device).unsqueeze(0)

    hidden_states, _ = model.model(
        inputs_embeds=token_emb, position_ids=position_ids,
        use_cache=False, attention_mask=attn_mask, ar_mask=ar_mask,
    )
    v_pred = model.llm2vae(hidden_states)  # [1, seq, 64]
    return v_pred[0, ar_len + 1: ar_len + 1 + t_lat]


def sample_t(rng: random.Random, generator: torch.Generator, device, mode: str):
    if mode == "logit_normal":
        raw = torch.randn((), generator=generator, device="cpu").item()
        t = 1.0 / (1.0 + math.exp(-raw))
        return t, raw
    if mode == "uniform":
        t = 0.001 + 0.998 * rng.random()
        return t, math.log(t / (1 - t))
    raise ValueError("t_sampling must be logit_normal or uniform")


@torch.inference_mode(False)  # ComfyUI runs nodes under inference_mode; training needs autograd
def _save_lora(model, path, metadata: dict, params=None, ema=None):
    """Write the LoRA in native ComfyUI format (LoraLoaderModelOnly).

    When ``ema`` (fp32 shadow copies aligned with ``params``) is given, the
    EMA weights are swapped in for saving and the live weights are restored
    afterwards.
    """
    from safetensors.torch import save_file
    from . import convert as convert_mod
    backup = None
    if ema is not None and params is not None:
        backup = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p, shadow in zip(params, ema):
                p.data.copy_(shadow.to(p.dtype))
    try:
        state = lora_mod.lora_state_dict(model)
        native_sd, _ = convert_mod.convert_tensors(state, metadata)
        save_file(native_sd, str(path), metadata=convert_mod.native_metadata(metadata))
    finally:
        if backup is not None:
            with torch.no_grad():
                for p, live in zip(params, backup):
                    p.data.copy_(live)
    return str(path)


def run_training(model, tokenizer, protocol, dataset: data_mod.TrainDataset, cfg, log_lines: list[str]):
    """Execute the training loop. ``cfg`` is a SimpleNamespace-like config."""
    device = torch.device(cfg.device)
    rng = random.Random(cfg.seed)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    mask_cache = MaskCache()

    target_names = lora_mod.inject_lora(model, cfg.target_preset, cfg.rank, cfg.alpha, cfg.lora_dropout)
    params = lora_mod.trainable_parameters(model)
    n_params = sum(p.numel() for p in params)
    log_lines.append(f"LoRA injected: preset={cfg.target_preset} rank={cfg.rank} alpha={cfg.alpha} "
                     f"-> {len(target_names)} layers, {n_params / 1e6:.2f}M trainable params")

    model.to(device)
    model.train()

    # EMA (exponential moving average) of the LoRA weights — smooths the noisy
    # per-step updates and gives noticeably more consistent LoRAs.
    # Created AFTER model.to(device) so the shadows live on the same device
    # as the trainable params.
    ema_decay = float(getattr(cfg, "ema_decay", 0.0))
    ema = None
    if ema_decay > 0:
        ema = [p.detach().clone().float() for p in params]
        log_lines.append(f"EMA enabled: decay={ema_decay} "
                         f"(<lora_name>.safetensors = EMA, <lora_name>_raw.safetensors = non-EMA)")

    try:
        if cfg.optimizer == "adamw_8bit":
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        else:
            optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate,
                                          weight_decay=cfg.weight_decay, fused=device.type == "cuda")
    except ImportError:
        log_lines.append("bitsandbytes not available — falling back to torch AdamW")
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def lr_at(step):
        if step < cfg.warmup_steps:
            return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
        if cfg.lr_scheduler == "cosine":
            ratio = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
            return cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * min(1.0, ratio)))
        return cfg.learning_rate

    # Pre-tokenize every distinct conditioning string.
    cond_cache: dict[str, list[int]] = {}

    def cond_for(caption: str) -> list[int]:
        style = cfg.trigger_word if not caption else f"{cfg.trigger_word}, {caption}"
        if style not in cond_cache:
            cond_cache[style] = build_conditioning_ids(tokenizer, protocol, style)
        return cond_cache[style]

    uncond_ids = None
    if cfg.caption_dropout > 0:
        uncond_ids = build_conditioning_ids(tokenizer, protocol, "")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = _comfy_progress(cfg.steps)
    history = []
    running = 0.0
    start_time = time.time()

    # Live chart preview: every few seconds a light-weight chart is written
    # (atomically) to a fixed temp file; the Training Curve node's frontend
    # extension polls it while the prompt runs.
    live_path = getattr(cfg, "live_curve_path", None) or None
    live_p = Path(live_path) if live_path else None
    live_on = bool(getattr(cfg, "live_curve", False)) and live_p is not None
    live_steps: list[int] = []
    live_losses: list[float] = []
    live_lrs: list[float] = []
    live_last = 0.0
    live_failed = False
    if live_on:
        try:  # don't flash the previous run's chart in the live preview
            Path(live_path).unlink(missing_ok=True)
        except OSError:
            pass

    metadata = {
        "format": lora_mod.FORMAT_VERSION,
        "base_model": cfg.base_model_name,
        "trigger_word": cfg.trigger_word,
        "rank": cfg.rank,
        "alpha": cfg.alpha,
        "target_preset": cfg.target_preset,
        "steps": cfg.steps,
        "learning_rate": cfg.learning_rate,
        "clip_seconds": dataset.clip_seconds,
        "t_sampling": cfg.t_sampling,
        "ema_decay": ema_decay,
        "cot": "off",
    }

    for step in range(cfg.steps):
        _check_interrupt()
        item = dataset.items[rng.randrange(len(dataset.items))]
        cond_ids = cond_for(item.caption)
        if uncond_ids is not None and rng.random() < cfg.caption_dropout:
            cond_ids = uncond_ids

        z = data_mod.load_clip_latents(item).to(device)  # [T, 64] fp32
        t, raw_t = sample_t(rng, generator, device, cfg.t_sampling)
        noise = torch.randn(z.shape, generator=generator, device="cpu").to(device)
        x_t = ((1.0 - t) * z + t * noise)
        target = noise - z

        v_pred = training_velocity(model, cond_ids, x_t, raw_t, mask_cache, protocol)
        loss = F.mse_loss(v_pred.float(), target) / cfg.grad_accum
        loss.backward()
        running += loss.item() * cfg.grad_accum

        if (step + 1) % cfg.grad_accum == 0 or step == cfg.steps - 1:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                with torch.no_grad():
                    for p, shadow in zip(params, ema):
                        shadow.mul_(ema_decay).add_(p.detach().float(),
                                                    alpha=1.0 - ema_decay)

        history.append(loss.item() * cfg.grad_accum)
        running = history[-1]
        if progress is not None:
            progress.update(1)
        if (step + 1) % cfg.log_every == 0 or step == 0:
            window = history[-cfg.log_every:]
            elapsed = time.time() - start_time
            msg = (f"step {step + 1}/{cfg.steps}  loss={sum(window) / len(window):.5f}  "
                   f"lr={lr_at(step):.2e}  t={t:.3f}  [{item.source.name} #{item.clip_index}]  "
                   f"{elapsed / (step + 1):.2f}s/it")
            log.info(msg)
            log_lines.append(msg)

            live_steps.append(step + 1)
            live_losses.append(sum(window) / len(window))
            live_lrs.append(lr_at(step))
            if live_on and not live_failed and time.time() - live_last >= 4.0:
                live_last = time.time()
                try:
                    from . import curve as curve_mod
                    tmp_path = live_p.with_name(live_p.stem + ".tmp.png")
                    curve_mod.render_chart(
                        live_steps, live_losses, live_lrs,
                        {"lora_name": cfg.lora_name, "trigger": cfg.trigger_word,
                         "total": cfg.steps, "live": True},
                        tmp_path, smooth=15, figsize=(8, 4.2), dpi=90)
                    os.replace(tmp_path, live_path)
                except Exception as exc:
                    live_failed = True
                    log_lines.append(f"live curve preview disabled ({exc})")

        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0 and step + 1 < cfg.steps:
            ckpt = out_dir / f"{cfg.lora_name}_step{step + 1}.safetensors"
            _save_lora(model, ckpt, {**metadata, "steps": step + 1}, params, ema)
            log_lines.append(f"checkpoint saved: {ckpt.name}")

    final_path = out_dir / f"{cfg.lora_name}.safetensors"
    _save_lora(model, final_path, metadata, params, ema)
    log_lines.append(f"final LoRA saved (native format): {final_path}")
    if ema is not None:
        raw_path = out_dir / f"{cfg.lora_name}_raw.safetensors"
        _save_lora(model, raw_path, {**metadata, "ema_decay": 0.0})
        log_lines.append(f"non-EMA comparison copy saved: {raw_path}")
    return str(final_path)
