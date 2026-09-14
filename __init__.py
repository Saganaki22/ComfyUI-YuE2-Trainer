"""ComfyUI-YuE2-Trainer — LoRA training nodes for m-a-p/YuE2-3B.

Trains the NAR (flow-matching / "diffusion_model") branch of YuE2 against
VAE latents encoded from your own mp3/wav/flac files, conditioned on a
text prefix containing your trigger word. Fully standalone: the official
YuE2 model code (m-a-p yue2_infer, Apache-2.0) is bundled in
trainer_core/yue2_ref — no third-party custom nodes required.
"""

import logging

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as exc:  # keep ComfyUI booting; report the real error
    logging.exception("ComfyUI-YuE2-Trainer failed to import: %s", exc)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./web"  # frontend extension for the Training Curve preview

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
