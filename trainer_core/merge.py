"""Merge a trained YuE2 LoRA into the base model as a new Olm-format folder.

The result lands in ``models/yue2/<output_name>`` (config.json +
qwen.tiktoken copied from the source model), so it appears directly in the
standard YuE2 Model Loader dropdown of ComfyUI-Olm-YuE2.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import lora as lora_mod


def _load_base_state(model_dir: Path) -> dict[str, torch.Tensor]:
    index = model_dir / "model.safetensors.index.json"
    files = ["model.safetensors"]
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        files = sorted(set(mapping.values()))
    state = {}
    for name in files:
        with safe_open(model_dir / name, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                state[key] = handle.get_tensor(key)
    if not state:
        raise FileNotFoundError(f"No weights found in {model_dir}")
    return state


def merge_to_new_model(model_dir: Path, lora_path: Path, output_root: Path,
                       output_name: str, strength: float,
                       progress_cb=None) -> Path:
    if not 0.0 <= strength <= 4.0:
        raise ValueError("strength must be within 0..4")
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in output_name).strip("._")
    if not safe_name:
        raise ValueError("output_name must contain at least one letter or digit")
    out_dir = output_root / safe_name
    if out_dir.exists():
        raise FileExistsError(f"Output folder already exists: {out_dir}")

    lora = lora_mod.load_lora(lora_path)
    base = _load_base_state(model_dir)
    merged = lora_mod.merge_lora_into_state_dict(base, lora, strength=strength)
    del base

    out_dir.mkdir(parents=True)
    save_file(merged, out_dir / "model.safetensors",
              metadata={"format": "pt", "yue2_lora_merge": lora_path.name,
                        "strength": str(strength)})
    for extra in ("config.json", "qwen.tiktoken"):
        source = model_dir / extra
        if source.is_file():
            shutil.copyfile(source, out_dir / extra)
        else:
            raise FileNotFoundError(f"{extra} missing in {model_dir}")
    return out_dir
