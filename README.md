# ComfyUI-YuE2-Trainer

LoRA training nodes for **[m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)** inside ComfyUI.
**NAR Style LoRA is legacy/experimental.** The current path has demonstrated
nonzero learned deltas and a measurable native NAR forward effect in a short
synthetic diagnostic. Artist resemblance and perceptual benefit remain unverified;
there is no established best step count or strength. See the [issue #1 audit](docs/issue-1-audit.md).

Train YuE2's acoustic branch on your own music (mp3 / wav / flac) with a **trigger word**.
This optimizes reconstruction of recording latents; it does not establish that the model
will reproduce the artist's style or voice. No caption
files required (optional same-named `.txt` captions are supported).
<img width="909" height="483" alt="image" src="https://github.com/user-attachments/assets/19a1d98d-3469-4be6-828d-1c11722a283b" />

## How it works (short version)

YuE2 has three parts: a 2.2B AR language model (plans the song, writes semantic tokens),
a 1.5B NAR flow-matching branch (renders 64-channel VAE latents into sound), and a
48 kHz stereo VAE. This trainer:

1. encodes your audio files into VAE latents (cached to disk, done once),
2. trains **LoRA adapters on the NAR branch only**, with the released flow-matching
   objective, conditioned on the checkpoint-native text prefix
   (`[Tags] your_trigger_word, caption ...`),
3. saves a standard `*.safetensors` LoRA into `models/loras` in **native ComfyUI
   format** — load it with the stock `LoraLoaderModelOnly` node on the native
   YuE2 checkpoint and generate, no conversion needed.

The AR "composer" branch stays frozen (m-a-p has not released an audio→token encoder or
training code), so this is a *style/timbre* LoRA, not a full voice clone.
License note: YuE2 weights are CC BY-NC 4.0 — non-commercial use only.

### Check whether a LoRA is active

Use **YuE2 NAR LoRA Loader (verified keys)** between the checkpoint MODEL and
the sampler. Its report includes per-module down/up/delta norms, candidate patches,
matched keys, strength, and disabled status. It rejects zero deltas and missing
targets. A match is evidence of applicability; it is not proof of perceptual quality.
Strength zero returns an unpatched clone. Existing node IDs and input schemas are retained.

For a numerical GPU check, run `tools/verify_native_lora.py` with your ComfyUI
venv and `--comfy-root`, `--checkpoint`, `--lora`, and `--report` paths. It uses
the standard native loader and checks strengths 0/1/2/0 against fixed synthetic
conditioning. The report distinguishes this from generated song conditioning.

The recommended Artist Training path (MERT → tokenizer head → semantic tokens →
AR LoRA, with the minted-corpus regularizer) is **implemented and verified on this
branch**: a 4 s real-song excerpt produced bit-exact upstream-parity semantic IDs,
three AR training steps produced nonzero deltas in all 112 targeted native modules,
and the trained adapter measurably changed deterministic native AR logits at
strengths 1 and 2 (max abs diff 0.19 / 0.25) while strength 0 restored baseline
exactly. See [docs/issue-1-audit.md](docs/issue-1-audit.md). This is a wiring and
effect proof, not an artist-resemblance result.

## Requirements

- ComfyUI (Windows portable / Easy-Install works) with an NVIDIA GPU;
  **24 GB VRAM recommended** (tested on an RTX 5090 Laptop 24 GB).
- **Fully standalone** — no other custom nodes required. The official YuE2
  model code (m-a-p's `yue2_infer`, Apache-2.0, unmodified) is bundled in
  `trainer_core/yue2_ref/`.
- Two small pip packages: `soundfile` (audio loading fallback; already present
  in most ComfyUI bundles) and `matplotlib` (only used by the optional Training
  Curve node). Install with the button in ComfyUI-Manager or:
  `python_embeded\python.exe -m pip install -r requirements.txt`
- Optional: `bitsandbytes` for the 8-bit optimizer.

### Model download and placement

Only **one file** is needed — the native all-in-one YuE2 checkpoint (by
downloading you accept the
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) model license):

| File | Source | Put it in |
| --- | --- | --- |
| `yue2_3b_bf16.safetensors` (~7.8 GB) | [huggingface.co/Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2) (official ComfyUI repack of [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)) | `ComfyUI/models/checkpoints/` |

This is the same file the native ComfyUI YuE2 generation nodes use — it
contains the language model, the acoustic (NAR) branch, the VAE and the
tokenizer, so one download covers training *and* generation. Use the **bf16**
file; the INT8 quantized variant cannot be trained.

### Artist Training assets (optional - auto-downloaded)

The recommended Artist Training path additionally needs the pre-converted
Mothersuperior safetensors and MERT-v2-FullSong. You don't have to download
anything by hand: the **YuE2 Mothersuperior Assets** node fetches pinned
revisions on first run when `auto_download` is on, then reuses them forever.
Manual placement also works - the node picks up existing files without
touching the network.

Layout (links point at
[drbaph/yue2-mothersuperior-realaudio-tokenizer-comfyui](https://huggingface.co/drbaph/yue2-mothersuperior-realaudio-tokenizer-comfyui),
a safetensors conversion of
[Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4),
CC BY-NC 4.0):

```
📂 ComfyUI/
└── 📂 models/
    └── 📂 yue2_trainer/
        └── 📂 mothersuperior/
            ├── tokenizer_head_joint_v4.safetensors
            ├── minted_regularizer_pack.safetensors
            ├── minted_regularizer_pack.jsonl
            └── nar_lora_joint_v4.safetensors   (optional - pretrained NAR adapter only)
```

| File | Source | What it is |
| --- | --- | --- |
| `tokenizer_head_joint_v4.safetensors` | [HF](https://huggingface.co/drbaph/yue2-mothersuperior-realaudio-tokenizer-comfyui/blob/main/tokenizer_head_joint_v4.safetensors) | MERT features → 32,768 YuE2 semantic codes (the audio→token encoder YuE2 doesn't ship) |
| `minted_regularizer_pack.safetensors` + `.jsonl` | [HF](https://huggingface.co/drbaph/yue2-mothersuperior-realaudio-tokenizer-comfyui/blob/main/minted_regularizer_pack.safetensors) | 4,732 YuE2-generated songs; 50% of AR training batches so a small artist set can't collapse the token grammar |
| `nar_lora_joint_v4.safetensors` | [HF](https://huggingface.co/drbaph/yue2-mothersuperior-realaudio-tokenizer-comfyui/blob/main/nar_lora_joint_v4.safetensors) | Pretrained NAR adapter (rank-32 LoRAs + full vae2llm/llm2vae replacements) - NAR experiments only, not needed for AR training |
| MERT-v2-FullSong snapshot | [m-a-p/MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) | Feature extractor for the tokenizer head (cached in `hf_cache/`) |

Every safetensors file embeds its source `.pt` SHA-256 and license in its
metadata header, so you can verify the conversion against the upstream
pickles at any time.

## Install

**Recommended: git clone** (makes updating easy with `git pull`):

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Starnodes2024/ComfyUI-YuE2-Trainer.git
```

Windows portable / Easy-Install example:

```bash
cd E:\AI\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI\custom_nodes
git clone https://github.com/Starnodes2024/ComfyUI-YuE2-Trainer.git
```

**Alternative:** download the repository as ZIP
([Code → Download ZIP](https://github.com/Starnodes2024/ComfyUI-YuE2-Trainer/archive/refs/heads/main.zip))
and extract it into `ComfyUI/custom_nodes/` so the folder is
`ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer`.

**Updating later:**

```bash
cd ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer
git pull
```

Make sure the [requirements](#requirements) above are met (the YuE2 checkpoint
is in place), then **restart ComfyUI**.
Three nodes appear under **YuE2/Training**.

## Usage

1. **YuE2 Training Dataset (audio folder)** — pick the YuE2 `checkpoint` and
   point `audio_folder` at your song folder.
   - Optional: place `songname.txt` next to `songname.mp3` with a caption
     (style / instruments / voice description) and set `caption_mode = txt_file`.
   - `clip_seconds` 10 is a good default (250 latent frames). The node encodes
     with the checkpoint's built-in VAE in memory-safe 30 s chunks (flat RAM/VRAM
     no matter how many or how long the files are, with a progress bar and a
     responsive Cancel button) and caches latents; rerunning is instant unless
     you change files or clip length.
2. **YuE2 LoRA Trainer** — connect the dataset, pick the same `checkpoint`,
   set `trigger_word`, `steps` (default 3000), `learning_rate` (1e-4),
   `rank`/`alpha` (32/32), and `lora_name`. Queue and wait.
   - The LoRA is written in **native ComfyUI format** — load it with
     `LoraLoaderModelOnly` on the native YuE2 checkpoint, no conversion needed.
   - **EMA smoothing is on by default** (`ema_decay` 0.999): the main
     `<name>.safetensors` uses the smoothed weights (much more consistent
     results), and the unsmoothed run is saved next to it as
     `<name>_raw.safetensors` so you can A/B-compare. Every widget has a
     tooltip — hover for guidance.
   - Progress bar in ComfyUI; losses appear in the console and in the node's
     `training_log` output.
   - Rough speed estimate on a 4090: ~1–3 s/step at 10 s clips → 1000 steps ≈ 25–50 min.
3. **YuE2 Training Curve (optional)** — connect the trainer's `training_log`
   output and watch it **live**: while training runs, the node redraws a
   compact loss/LR chart every few seconds right inside the node (the trainer
   streams it via the temp folder; toggle with the trainer's `live_curve`
   widget, default on). When the run finishes, the final high-quality chart
   replaces it — dark-styled, with the raw + smoothed **loss curve** and the
   **learning-rate schedule**, shown inline and as an IMAGE output you can save
   with any image node (needs `matplotlib` from requirements.txt). If the
   inline preview doesn't appear after installing, restart ComfyUI and
   hard-refresh the browser (Ctrl+F5) — the small frontend extension in
   `web/js/` needs one reload.

### Generating with your LoRA

Use **`cot = off`** (leave ABC empty) in the YuE2 Request node and put your trigger word at the start of
the style prompt, e.g. `mystyle, melancholic piano ballad, soft female vocals`.
`cot=off` matches the text-only conditioning regime the LoRA was trained with.
Full/melody CoT also works (the LoRA still shapes the sound) but drift from the
training regime is larger.

## Practical tips

- **Dataset:** 5–30 songs with a consistent style/voice works well. Consistent,
  well-tagged material beats sheer volume.
- **Consistency:** keep `ema_decay` at 0.999 and `lr_scheduler = cosine`
  (both defaults) — this combination removes most "hit and miss" variance
  between runs. Compare against the `_raw` file if you are curious.
- **Steps/LR:** 3000 steps @ 1e-4, rank 32 is the tuned default. If the result
  overfits (muffled, repetitive), lower steps or LR; if the trigger has no
  effect, raise them.
- **VRAM:** lower `clip_seconds` (e.g. 6–8) if you OOM; try `optimizer = adamw_8bit`.
- **Voice:** vocal timbre transfers through the NAR branch; the exact melody/lyrics
  stay controlled by the frozen AR stage and your prompt.

## Limitations / honest caveats

- No official YuE2 training code exists yet; the training objective here is
  reconstructed from the released inference code (the shipped `nar_velocity`
  documents the training-time injection). It is experimental — validate with a
  small run first.
- Training is text-conditioned only (codec-dropout regime); ground-truth semantic
  tokens for arbitrary audio are not available from m-a-p.
- One training run uses the GPU exclusively; other loaded ComfyUI models are
  unloaded when training starts.

## Example workflows

Ready-to-use workflows live in [`example_workflows/`](example_workflows) —
drag the JSON into the ComfyUI window:

- **`01_yue2_lora_training.json`** — dataset → trainer (both load from the
  native checkpoint). Set your `audio_folder`, `trigger_word` and `lora_name`,
  then Queue.
- **`02_yue2_native_generate_with_lora.json`** — native generation with your
  LoRA: `CheckpointLoaderSimple` (native `yue2.safetensors`) →
  `LoraLoaderModelOnly` → `YuE2 Generate Music` → `KSampler` → `VAEDecode` →
  `SaveAudio`. Put your trigger word at the start of the style prompt.
  Notes: CFG is 1.0 (negative input is fed from the same conditioning and is
  unused, as YuE2 handles guidance internally); 32 steps euler/simple matches
  the released ODE solver; set the latent `seconds` and `max_duration` to the
  song length you want (120 s default). Leave the ABC input empty for the
  `off` mode that matches LoRA training best.

## Credits

- **[YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)** by
  [m-a-p](https://huggingface.co/m-a-p) — model architecture, weights and the
  official `yue2_infer` code that is bundled unmodified in
  `trainer_core/yue2_ref/`. If you use YuE2 in research, cite the YuE paper
  ([arXiv:2503.08638](https://arxiv.org/abs/2503.08638)).
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** — including its native
  YuE2 implementation that the native-format LoRA output targets.
- **[Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2)** — the official
  single-file checkpoint repack the trainer can load directly.
- **[Mothersuperior](https://huggingface.co/Mothersuperior)** — the real-audio
  tokenizer head, NAR adapter, minted regularizer corpus, and training scripts
  that power the Artist Training path
  ([yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4),
  [yue2-minted-corpus](https://huggingface.co/datasets/Mothersuperior/yue2-minted-corpus)),
  CC BY-NC 4.0.
- **[drbaph](https://huggingface.co/drbaph)** — the pre-converted safetensors
  builds of the Mothersuperior assets
  ([yue2-mothersuperior-realaudio-tokenizer-comfyui](https://huggingface.co/drbaph/yue2-mothersuperior-realaudio-tokenizer-comfyui))
  the trainer downloads by default.

## License

This package's own code is published by **Starnodes** under the **MIT License**
(see [LICENSE](LICENSE)).

Third-party components keep their own licenses and **must be respected**:

- **YuE2 model weights** ([YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B),
  [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae), and the
  [Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2) single-file repack)
  are licensed
  **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)** by m-a-p —
  **non-commercial use only, with attribution**. LoRAs trained with this tool are
  derivatives of those weights, so the same non-commercial terms apply to them
  and to any audio generated with them. By downloading the models you accept
  those terms on Hugging Face.
- **YuE2 reference inference code** (bundled unmodified in
  `trainer_core/yue2_ref/`): [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0),
  © m-a-p; its third-party notices are preserved in
  `trainer_core/yue2_ref/licenses/`.
- **Mothersuperior assets** (the tokenizer head, NAR adapter, and minted
  regularizer pack from
  [yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)
  and [yue2-minted-corpus](https://huggingface.co/datasets/Mothersuperior/yue2-minted-corpus),
  including the
  [drbaph safetensors conversions](https://huggingface.co/drbaph/yue2-mothersuperior-realaudio-tokenizer-comfyui))
  are licensed **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**
  by the Mothersuperior authors — **non-commercial use only**. They derive from
  YuE2-3B weights, and the upstream scripts are provided as-is; LoRAs trained
  with these assets carry the same terms.
- Your training data is your own responsibility: only train on audio you have
  the rights to use.
