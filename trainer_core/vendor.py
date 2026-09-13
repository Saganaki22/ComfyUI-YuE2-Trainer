"""Locate ComfyUI-Olm-YuE2 and expose its vendored `yue2` modeling code.

The trainer deliberately reuses the same model classes that inference runs
on, so a LoRA trained here behaves identically at generation time.
"""
from __future__ import annotations

import sys
from pathlib import Path

_OLM_DIR_NAME = "ComfyUI-Olm-YuE2"


def find_olm_dir() -> Path:
    """Return the ComfyUI-Olm-YuE2 package directory."""
    here = Path(__file__).resolve()
    # trainer_core/vendor.py -> <custom_nodes>/ComfyUI-YuE2-Trainer/trainer_core/vendor.py
    custom_nodes = here.parents[2]
    candidate = custom_nodes / _OLM_DIR_NAME
    if (candidate / "_vendor" / "yue2" / "modeling_yue2.py").is_file():
        return candidate
    # Fallback: scan custom_nodes in case the folder was renamed.
    for child in sorted(custom_nodes.iterdir()) if custom_nodes.is_dir() else []:
        if child.is_dir() and (child / "_vendor" / "yue2" / "modeling_yue2.py").is_file():
            return child
    raise FileNotFoundError(
        "ComfyUI-Olm-YuE2 was not found next to this package in custom_nodes. "
        "YuE2-Trainer reuses its vendored modeling code — install ComfyUI-Olm-YuE2 first."
    )


def import_yue2():
    """Import and return the vendored yue2 submodules used by the trainer."""
    olm_dir = find_olm_dir()
    if str(olm_dir) not in sys.path:
        sys.path.insert(0, str(olm_dir))
    from _vendor.yue2 import modeling_yue2, modeling_vae, protocol, tokenization_yue2  # noqa: E501
    return modeling_yue2, modeling_vae, protocol, tokenization_yue2


def import_model_paths():
    """Reuse the Olm runtime's model-folder resolution (dropdown listings)."""
    olm_dir = find_olm_dir()
    if str(olm_dir) not in sys.path:
        sys.path.insert(0, str(olm_dir))
    from runtime.paths import ModelPaths  # noqa: E501  (Olm runtime package)
    return ModelPaths
