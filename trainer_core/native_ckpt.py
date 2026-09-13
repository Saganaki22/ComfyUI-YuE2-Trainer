"""Load YuE2 training weights from the native ComfyUI all-in-one checkpoint.

The native checkpoint (``checkpoints/yue2.safetensors``) stores the same
weights as the Hugging Face ``models/yue2`` + ``models/yue2_vae`` folders,
but split and renamed:

- ``text_encoders.model.*``   -> the AR (language-model) branch, with fused
  ``self_attn.qkv_proj`` / ``mlp.gate_up_proj`` tensors
- ``model.diffusion_model.*`` -> the NAR (acoustic) branch, equally fused,
  plus ``latent_pos_embed`` / ``llm2vae`` / ``vae2llm`` / ``time_embedder``
- ``vae.*``                   -> the YuE2 VAE (identical keys plus the prefix)
- ``text_encoders.yue2_tokenizer_json`` -> the BPE tokenizer as uint8 JSON

This module reverses that split so the trainer can run from the single
checkpoint alone. The mapping was verified bit-for-bit against the HF files.
"""
from __future__ import annotations

import unicodedata

import torch
from safetensors import safe_open

TE_PREFIX = "text_encoders.model."
DM_PREFIX = "model.diffusion_model."
VAE_PREFIX = "vae."
TOKENIZER_KEY = "text_encoders.yue2_tokenizer_json"

# Fused-tensor row splits (YuE2-3B: 16 heads x 128, 8 kv heads x 128, mlp 6144).
Q_ROWS = 2048
KV_ROWS = 1024
MLP_ROWS = 6144


def is_native_yue2_checkpoint(path) -> bool:
    """True when the safetensors file is a native YuE2 all-in-one checkpoint."""
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
    except Exception:
        return False
    return (TOKENIZER_KEY in keys
            and TE_PREFIX + "embed_tokens.weight" in keys
            and DM_PREFIX + "model.layers.0.self_attn.qkv_proj.weight" in keys
            and any(k.startswith(VAE_PREFIX + "decoder.") for k in keys))


def _map_layer_tensors(rest, branch, tensor, out):
    """Map one native layer tensor to its HF key(s); branch is '' or 'nar_'."""
    n, sub = rest.split(".", 1)
    base = f"model.layers.{n}.{branch}"
    if sub == "self_attn.qkv_proj.weight":
        out[base + "self_attn.q_proj.weight"] = tensor[:Q_ROWS]
        out[base + "self_attn.k_proj.weight"] = tensor[Q_ROWS:Q_ROWS + KV_ROWS]
        out[base + "self_attn.v_proj.weight"] = tensor[Q_ROWS + KV_ROWS:]
    elif sub == "mlp.gate_up_proj.weight":
        out[base + "mlp.gate_proj.weight"] = tensor[:MLP_ROWS]
        out[base + "mlp.up_proj.weight"] = tensor[MLP_ROWS:]
    elif sub.startswith(("self_attn.", "mlp.")):
        kind, leaf = sub.split(".", 1)
        out[base + kind + "." + leaf] = tensor
    elif sub == "input_layernorm.weight":
        out[base + "input_layernorm.weight"] = tensor
    elif sub == "post_attention_layernorm.weight":
        # base already carries the nar_ prefix for the acoustic branch
        out[base + ("pre_mlp_layernorm.weight" if branch
                    else "post_attention_layernorm.weight")] = tensor
    else:
        raise KeyError(f"unmapped native layer key suffix: {sub}")


def load_native_lm_state(path) -> dict:
    """Return the full YuE2ForCausalLM state dict (HF key names) from the
    native all-in-one checkpoint."""
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(TE_PREFIX):
                rest = key[len(TE_PREFIX):]
                if rest.startswith("layers."):
                    _map_layer_tensors(rest[len("layers."):], "",
                                       handle.get_tensor(key), out)
                elif rest in ("embed_tokens.weight", "norm.weight"):
                    out["model." + rest] = handle.get_tensor(key)
                else:  # lm_head.weight
                    out[rest] = handle.get_tensor(key)
            elif key.startswith(DM_PREFIX):
                rest = key[len(DM_PREFIX):]
                if rest.startswith("model.layers."):
                    _map_layer_tensors(rest[len("model.layers."):], "nar_",
                                       handle.get_tensor(key), out)
                elif rest == "model.norm.weight":
                    pass  # identical to the AR norm; kept once via TE branch
                else:
                    out[rest] = handle.get_tensor(key)
    return out


def load_native_vae_state(path) -> dict:
    """Return the YuE2VAE state dict (unprefixed keys) from the checkpoint."""
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(VAE_PREFIX):
                out[key[len(VAE_PREFIX):]] = handle.get_tensor(key)
    return out


def load_native_tokenizer_json(path) -> bytes:
    """Return the embedded BPE tokenizer JSON from the checkpoint."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(TOKENIZER_KEY)
    return tensor.numpy().tobytes()


class YuE2JsonTokenizer:
    """Drop-in replacement for ``YuE2TextTokenizer`` backed by the tokenizer
    JSON embedded in the native checkpoint (identical ids, verified)."""

    def __init__(self, json_bytes):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_str(json_bytes.decode("utf-8"))

    def encode(self, text):
        return self._tok.encode(unicodedata.normalize("NFC", text)).ids

    def decode(self, ids):
        return self._tok.decode([int(i) for i in ids], skip_special_tokens=False)


def build_lm_from_native(path, modeling_yue2, torch_dtype=torch.bfloat16):
    """Instantiate YuE2ForCausalLM directly from the native checkpoint.

    Instantiation happens on the meta device and weights are assigned in
    place, so peak host memory stays at roughly the bf16 checkpoint size.
    """
    config = modeling_yue2.YuE2Config()
    state = load_native_lm_state(path)
    with torch.no_grad(), torch.device("meta"):
        model = modeling_yue2.YuE2ForCausalLM(config)
    expected = set(model.state_dict())
    if set(state) != expected:
        raise ValueError(
            "native checkpoint tensor mismatch: "
            f"missing={sorted(expected - set(state))[:5]} "
            f"unexpected={sorted(set(state) - expected)[:5]}")
    model.load_state_dict(
        {k: v.to(torch_dtype) for k, v in state.items()}, strict=True, assign=True)
    del state
    return model


def build_vae_from_native(path, modeling_vae, device="cpu"):
    """Instantiate YuE2VAE (encoder + decoder, FP32) from the checkpoint."""
    config = modeling_vae.YuE2VAEConfig()
    state = load_native_vae_state(path)
    model = modeling_vae.YuE2VAE(config, decoder_only=False)
    expected = set(model.state_dict())
    if set(state) != expected:
        raise ValueError(
            "native checkpoint VAE tensor mismatch: "
            f"missing={sorted(expected - set(state))[:5]} "
            f"unexpected={sorted(set(state) - expected)[:5]}")
    model.load_state_dict({k: v.to(torch.float32) for k, v in state.items()},
                          strict=True, assign=True)
    del state
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return model
