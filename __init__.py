"""ComfyUI-YuE2-Trainer — LoRA training nodes for m-a-p/YuE2-3B.

Trains the NAR (flow-matching / "diffusion_model") branch of YuE2 against
VAE latents encoded from your own mp3/wav/flac files, conditioned on a
text prefix containing your trigger word. Requires ComfyUI-Olm-YuE2 to be
installed (its vendored `yue2` modeling code is reused).
"""

import logging

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as exc:  # keep ComfyUI booting; report the real error
    logging.exception("ComfyUI-YuE2-Trainer failed to import: %s", exc)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
