"""Load the bundled upstream YuE2 reference modules.

The trainer is self-contained: the model/VAE/protocol/tokenizer code ships
in ``trainer_core/yue2_ref`` (unmodified Apache-2.0 files from m-a-p's
official ``yue2_infer`` package). No third-party custom nodes are required.
"""
from __future__ import annotations


def import_yue2():
    """Return (modeling_yue2, modeling_vae, protocol, tokenization_yue2)."""
    from .yue2_ref import modeling_yue2, modeling_vae, protocol, tokenization_yue2
    return modeling_yue2, modeling_vae, protocol, tokenization_yue2
