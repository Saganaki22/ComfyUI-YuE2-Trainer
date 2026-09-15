"""Native ComfyUI LoRA writer for YuE2-Trainer LoRAs.

The trainer stores HF-layout keys with separate projections:
    model.layers.N.nar_self_attn.{q,k,v,o}_proj.lora_{down,up}.weight
    model.layers.N.nar_mlp.{gate,up,down}_proj...
    llm2vae / vae2llm / time_embedder.mlp.{0,2}

ComfyUI's native YuE2 model (comfy/ldm/yue2/model.py) reuses Qwen3 blocks
with merged_qkv/merged_mlp, so the equivalent native patch keys are:
    diffusion_model.model.layers.N.self_attn.qkv_proj   (fused q|k|v)
    diffusion_model.model.layers.N.mlp.gate_up_proj     (fused gate|up)
plus 1:1 renames for o_proj / down_proj / projections.

Fusion math (exact, lossless): for fused weight W = [Wq; Wk; Wv] (rows),
the combined delta is [up_q@down_q; up_k@down_k; up_v@down_v] * scale,
which factors as U @ D with
    D = [down_q; down_k; down_v]                 # [3r, in]
    U = block columns [up_q | up_k | up_v] * scale  # [out, 3r], zero blocks
The trainer's alpha/rank scaling is baked into U; an explicit `.alpha`
entry equal to the fused rank is emitted so ComfyUI's alpha/rank step
always multiplies by exactly 1.0.
"""
from __future__ import annotations

import re

import torch

from . import lora as lora_mod

NATIVE_PREFIX = "diffusion_model"

# (trainer suffix, native suffix) for 1:1 mappings
DIRECT_MAP = {
    "nar_self_attn.o_proj": "self_attn.o_proj",
    "nar_mlp.down_proj": "mlp.down_proj",
}
TOP_LEVEL_MAP = {
    "llm2vae": "llm2vae",
    "vae2llm": "vae2llm",
    "time_embedder.mlp.0": "time_embedder.mlp.0",
    "time_embedder.mlp.2": "time_embedder.mlp.2",
}
QKV_ORDER = ("q_proj", "k_proj", "v_proj")        # matches native split(q,k,v)
GATE_UP_ORDER = ("gate_proj", "up_proj")          # matches native chunk(gate,up)

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.*)$")


def _fuse_group(parts: dict[str, tuple[torch.Tensor, torch.Tensor]], order, scale: float):
    """parts: name -> (down [r,in], up [out,r]). Returns fused (D, U)."""
    rank = next(iter(parts.values()))[0].shape[0]
    down = torch.cat([parts[name][0] for name in order], dim=0)  # [k*r, in]
    out_total = sum(parts[name][1].shape[0] for name in order)
    up = torch.zeros(out_total, rank * len(order), dtype=torch.float32)
    row, col = 0, 0
    for name in order:
        d, u = parts[name]
        out_dim = u.shape[0]
        up[row:row + out_dim, col:col + rank] = u.float() * scale
        row += out_dim
        col += rank
    return down.float(), up


def _emit(out: dict[str, torch.Tensor], native_prefix: str,
          down: torch.Tensor, up: torch.Tensor):
    out[f"{native_prefix}.lora_down.weight"] = down.contiguous()
    out[f"{native_prefix}.lora_up.weight"] = up.contiguous()
    # alpha == fused rank -> ComfyUI scale (alpha/rank) is exactly 1.0;
    # the trainer's own alpha/rank scaling is already baked into `up`.
    out[f"{native_prefix}.alpha"] = torch.tensor(float(down.shape[0]))


def convert_tensors(tensors: dict[str, torch.Tensor], metadata: dict):
    """Convert trainer-format LoRA tensors to native format in memory.

    Returns (native_tensors, report) where native_tensors includes
    lora_down/lora_up/alpha entries keyed ``diffusion_model.*``.
    """
    rank = None
    for key, tensor in tensors.items():
        if key.endswith(".lora_down.weight"):
            rank = tensor.shape[0]
            break
    if rank is None:
        raise ValueError("LoRA has no lora_down weights")
    alpha = float(metadata.get("alpha", rank))
    scale = alpha / rank

    # Group tensors per module prefix
    downs, ups = {}, {}
    for key, tensor in tensors.items():
        if key.endswith(".lora_down.weight"):
            downs[key[: -len(".lora_down.weight")]] = tensor
        elif key.endswith(".lora_up.weight"):
            ups[key[: -len(".lora_up.weight")]] = tensor

    if downs.keys() != ups.keys():
        raise ValueError("LoRA has unmatched down/up matrices: " + str(sorted(downs.keys() ^ ups.keys())))
    for name, down in downs.items():
        up = ups[name]
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] != rank or up.shape[1] != rank:
            raise ValueError(f"Invalid or inconsistent LoRA rank for {name}")

    out: dict[str, torch.Tensor] = {}
    converted, used = [], set()
    layers = sorted({int(m.group(1)) for key in downs
                     if (m := LAYER_RE.match(key))})

    for layer in layers:
        base = f"model.layers.{layer}"
        native_layer = f"{NATIVE_PREFIX}.model.layers.{layer}"

        qkv = {}
        for name in QKV_ORDER:
            prefix = f"{base}.nar_self_attn.{name}"
            if prefix in downs and prefix in ups:
                qkv[name] = (downs[prefix], ups[prefix])
        if len(qkv) == 3:
            down, up = _fuse_group(qkv, QKV_ORDER, scale)
            _emit(out, f"{native_layer}.self_attn.qkv_proj", down, up)
            converted.append(f"layer {layer}: q/k/v -> self_attn.qkv_proj (rank {down.shape[0]})")
            used.update(f"{base}.nar_self_attn.{name}" for name in QKV_ORDER)

        gate_up = {}
        for name in GATE_UP_ORDER:
            prefix = f"{base}.nar_mlp.{name}"
            if prefix in downs and prefix in ups:
                gate_up[name] = (downs[prefix], ups[prefix])
        if len(gate_up) == 2:
            down, up = _fuse_group(gate_up, GATE_UP_ORDER, scale)
            _emit(out, f"{native_layer}.mlp.gate_up_proj", down, up)
            converted.append(f"layer {layer}: gate/up -> mlp.gate_up_proj (rank {down.shape[0]})")
            used.update(f"{base}.nar_mlp.{name}" for name in GATE_UP_ORDER)

        for trainer_suffix, native_suffix in DIRECT_MAP.items():
            prefix = f"{base}.{trainer_suffix}"
            if prefix in downs and prefix in ups:
                _emit(out, f"{native_layer}.{native_suffix}",
                      downs[prefix].float(), ups[prefix].float() * scale)
                converted.append(f"layer {layer}: {trainer_suffix} -> {native_suffix}")
                used.add(prefix)

    for trainer_name, native_name in TOP_LEVEL_MAP.items():
        if trainer_name in downs and trainer_name in ups:
            _emit(out, f"{NATIVE_PREFIX}.{native_name}",
                  downs[trainer_name].float(), ups[trainer_name].float() * scale)
            converted.append(f"{trainer_name} -> {native_name}")
            used.add(trainer_name)

    skipped = sorted(set(downs) - used)

    if skipped:
        raise ValueError("Cannot convert all LoRA targets (incomplete fusion or unknown keys): " + ", ".join(skipped))

    if not out:
        raise RuntimeError("Nothing convertible found — is this a YuE2-Trainer LoRA?")

    return out, {"converted": converted, "skipped": skipped, "tensors": len(out)}


def native_metadata(source_metadata: dict, source_file: str = "") -> dict:
    """Safetensors header for a native-format YuE2 LoRA."""
    meta = {
        "format": "comfyui-native-lora",
        "source_format": lora_mod.FORMAT_VERSION,
        "source_file": source_file,
        "trigger_word": source_metadata.get("trigger_word", ""),
        "base_model": source_metadata.get("base_model", ""),
    }
    for key in ("rank", "alpha", "steps", "target_preset", "clip_seconds",
                "t_sampling", "learning_rate", "cot"):
        if key in source_metadata:
            meta[f"source_{key}"] = source_metadata[key]
    return {k: str(v) for k, v in meta.items()}
