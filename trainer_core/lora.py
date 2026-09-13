"""LoRA injection, serialization and merging for YuE2's NAR branch.

The LoRA file format is plain safetensors with keys
    <module.path.with.dots>.lora_down.weight / .lora_up.weight
plus a JSON metadata blob (stored in the safetensors header) so the merge
node and future tooling can validate rank/alpha/targets without guessing.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

FORMAT_VERSION = "yue2-lora-v1"

# Named target presets -> suffix-matched module paths inside YuE2ForCausalLM.
# Only the NAR (acoustic / "diffusion_model") branch is trained; the AR
# ("text encoder") branch stays frozen.
TARGET_PRESETS = {
    "nar_attn": [
        "nar_self_attn.q_proj", "nar_self_attn.k_proj",
        "nar_self_attn.v_proj", "nar_self_attn.o_proj",
    ],
    "nar_attn_mlp": [
        "nar_self_attn.q_proj", "nar_self_attn.k_proj",
        "nar_self_attn.v_proj", "nar_self_attn.o_proj",
        "nar_mlp.gate_proj", "nar_mlp.up_proj", "nar_mlp.down_proj",
    ],
    "nar_attn_mlp_proj": [
        "nar_self_attn.q_proj", "nar_self_attn.k_proj",
        "nar_self_attn.v_proj", "nar_self_attn.o_proj",
        "nar_mlp.gate_proj", "nar_mlp.up_proj", "nar_mlp.down_proj",
        "llm2vae", "vae2llm", "time_embedder.mlp.0", "time_embedder.mlp.2",
    ],
}


class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper around a frozen nn.Linear."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear wraps nn.Linear only")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_down = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        # Keep LoRA mats in the base weight's dtype/device.
        self.lora_down.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_up.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_up.weight @ self.lora_down.weight) * self.scaling


def _iter_named_linears(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def _set_module_by_path(model: nn.Module, path: str, new_module: nn.Module):
    parent_path, _, attr = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, attr, new_module)


def resolve_target_names(model: nn.Module, preset: str) -> list[str]:
    if preset not in TARGET_PRESETS:
        raise ValueError(f"Unknown target preset {preset!r}; choose one of {sorted(TARGET_PRESETS)}")
    suffixes = tuple(TARGET_PRESETS[preset])
    names = [name for name, _ in _iter_named_linears(model) if name.endswith(suffixes)]
    if not names:
        raise RuntimeError(f"No modules matched preset {preset!r} — wrong model class?")
    return sorted(names)


def inject_lora(model: nn.Module, preset: str, rank: int, alpha: float, dropout: float) -> list[str]:
    """Replace targeted Linear layers with LoRALinear; freeze everything else.

    Returns the list of wrapped module paths.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    names = resolve_target_names(model, preset)
    for name in names:
        base = model.get_submodule(name)
        _set_module_by_path(model, name, LoRALinear(base, rank, alpha, dropout))
    for name in names:
        wrapper = model.get_submodule(name)
        wrapper.lora_down.weight.requires_grad_(True)
        wrapper.lora_up.weight.requires_grad_(True)
    return names


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_down.weight"] = module.lora_down.weight.detach().cpu().float()
            state[f"{name}.lora_up.weight"] = module.lora_up.weight.detach().cpu().float()
    if not state:
        raise RuntimeError("No LoRA modules found in model")
    return state


def save_lora(model: nn.Module, path, *, metadata: dict):
    meta = {"format": FORMAT_VERSION, **{k: str(v) for k, v in metadata.items()}}
    save_file(lora_state_dict(model), str(path), metadata=meta)
    return str(path)


@dataclass
class LoadedLoRA:
    tensors: dict[str, torch.Tensor]
    metadata: dict

    @property
    def rank(self) -> int:
        for key, tensor in self.tensors.items():
            if key.endswith(".lora_down.weight"):
                return tensor.shape[0]
        raise ValueError("LoRA file has no lora_down weights")


def load_lora(path) -> LoadedLoRA:
    from safetensors import safe_open
    tensors = load_file(str(path))
    with safe_open(str(path), framework="pt") as handle:
        metadata = dict(handle.metadata() or {})
    if metadata.get("format", FORMAT_VERSION) != FORMAT_VERSION:
        raise ValueError(f"Unsupported LoRA format in {path}")
    return LoadedLoRA(tensors, metadata)


def merge_lora_into_state_dict(base_state: dict[str, torch.Tensor], lora: LoadedLoRA,
                               strength: float = 1.0) -> dict[str, torch.Tensor]:
    """Return a copy of base_state with W += strength * scaling * up @ down applied."""
    merged = dict(base_state)
    applied = 0
    for key, down in lora.tensors.items():
        if not key.endswith(".lora_down.weight"):
            continue
        prefix = key[: -len(".lora_down.weight")]
        up = lora.tensors.get(f"{prefix}.lora_up.weight")
        if up is None:
            raise ValueError(f"Missing lora_up for {prefix}")
        weight_key = f"{prefix}.weight"
        if weight_key not in merged:
            raise ValueError(f"Base checkpoint has no weight {weight_key!r} for this LoRA")
        alpha = float(lora.metadata.get("alpha", down.shape[0]))
        scaling = alpha / down.shape[0]
        delta = (up.float() @ down.float()) * (scaling * strength)
        merged[weight_key] = merged[weight_key].float().add(delta).to(merged[weight_key].dtype)
        applied += 1
    if applied == 0:
        raise ValueError("LoRA contained no applicable layers")
    return merged
