# Issue #1: initial audit, 2026-09-15

## Result and limits

The reported no-effect condition was **not reproduced** with a newly trained
diagnostic adapter. The reporter supplied two workflows, but no trained adapter.
The cause of their specific failure is undetermined. This is an audit and
validation change, not a completed Mothersuperior integration.

Source: [issue #1](https://github.com/Starnodes2024/ComfyUI-YuE2-Trainer/issues/1).
Repository revision tested: `4578039513304149fdfd1c00a22783f06812997f`.
Branch: `feat/mothersuperior-realaudio-ar-training`.

The specified custom-node installation was absent. Work is in a fresh workspace
clone; no existing worktree was stashed or modified, and nothing was installed,
pushed, or submitted as a PR. The installed ComfyUI checkpoint and venv were reused.

## Evidence

- Workflow MODEL wiring is checkpoint → LoRA loader → KSampler.
- The AR generator seed is set to randomize in the attachment. Both AR and
  sampler seeds must be fixed for a controlled song comparison.
- The existing `diffusion_model.*` LoRA keys are recognized by the installed
  `comfy.lora.model_lora_keys_unet` generic mapping.
- Q/K/V and gate/up fusion and alpha/rank scaling passed numerical tests.
- Three actual NAR optimization steps, with synthetic eight-frame latents and
  text conditioning, produced 200 learned HF-layout LoRA pairs, converted to
  116 native modules (348 serialized tensors). All 116 deltas were nonzero.
- Gradient norms were 0.14746, 0.15137, and 0.16895. Losses were 2.33886,
  2.34004, and 2.33605. This is gradient-path evidence, not a quality assessment.
- Training peak allocated VRAM was 6.987 GiB for this tiny rank-2 diagnostic.
  It is not an estimate for real-song training.
- The standard ComfyUI `load_lora_for_models` API resolved 116 patches before
  any loader changes. The stricter new loader also matched all 116. There is
  no demonstrated before/after key-count fix for the issue.

Native NAR forward, same checkpoint, noise, synthetic context, timestep, and seed:

| Strength | Matched keys | Output max absolute difference | Output mean absolute difference | Probed weight max difference |
| --- | --- | --- | --- | --- |
| 0 | 116 | 0 | 0 | 0 |
| 1 | 116 | 0.404785 | 0.031705 | 0.000015259 |
| 2 | 116 | 0.683594 | 0.066247 | 0.000030518 |
| 0 again | 116 | 0 | 0 | 0 |

Differences are relative to the first strength-zero run. Output is nonlinear;
doubling strength need not exactly double output differences. The probed weight
delta approximately doubled. Disabling restored exact output and probed weights.
The checkpoint was loaded using native ComfyUI, not the trainer's model implementation.
Context was a fixed synthetic tensor, not a generated song's semantic conditioning.
No decoded-audio or complete sampler comparison was performed.

## Changes

- Added a strict NAR LoRA loader with delta norms and key matching reports.
- Export rejects all-zero adapters; conversion rejects missing pairs, mixed
  ranks, incomplete fusion, and unknown targets instead of silently dropping them.
- Corrected the cancellation import to `comfy.model_management`. This is unrelated
  to the reported inference failure.
- Kept old node IDs and required inputs; labelled the training path legacy/experimental.
- Added 16 CPU tests and reusable opt-in GPU training and forward diagnostic tools.
- No dependencies or model assets were added. No PT conversions were performed.

Run the tools with the existing ComfyUI venv, providing explicit paths:

```text
tools/train_diagnostic_lora.py --checkpoint CHECKPOINT --output DIAGNOSTIC.safetensors
tools/verify_native_lora.py --comfy-root COMFYUI --checkpoint CHECKPOINT --lora DIAGNOSTIC.safetensors --report RESULT.json
```

## Upstream findings for the next implementation stage

The complete published script set was downloaded and inspected from
[Mothersuperior v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/tree/f2278a2e005dc4ecc421c53a0929f62b3aeb2280).
No third-party source was copied into the plugin in this audit.

- MERT: mono 24 kHz, 30-second chunks, discard chunks shorter than one second,
  hidden state 20, concatenate, linear interpolation to rounded duration × 25,
  fp16 feature storage, per-track float32 population-standard-deviation normalization
  with epsilon 1e-5. Do not replace with sample standard deviation.
- Head: 1024 → 512 input, learned 512-frame position tensor, eight pre-norm
  TransformerEncoder layers, eight heads, GELU, 2048-wide FFN, LayerNorm, 32768 logits.
  Joint/head inference architecture uses dropout 0.1 (disabled in eval).
- Token inference: 512-frame windows, stride 256, retain central regions with
  128-frame edge trimming; append a final end-aligned window. Copy this exact
  overwrite behavior rather than averaging overlapping predictions.
- NAR checkpoint: ordered A/B tensor list for 28 × 7 linear targets, rank 32,
  plus full `io.vae2llm` and `io.llm2vae` state dictionaries including biases.
  These are replacement projection weights, not ordinary LoRA deltas.
- AR: q/k/v/o + gate/up/down per layer, float32 A/B matrices, scale 1,
  next-token loss on semantic tokens and optional MUSIC_END only. Maximum length
  12288, accumulation 2, logit chunks 1024, AdamW betas 0.9/0.95, zero decay,
  50-step warmup and cosine schedule with 0.2 floor, default LR 1e-4.
- Cursor: Demucs htdemucs vocals → MMS_FA alignment; word offsets reference exact
  lyrics. Carry forward the last started word at 25 Hz, distribute probability
  across overlapping lyric BPE tokens, auxiliary identity-initialized projection,
  cursor loss weight 0.08. Published normalization is Latin-letter-specific.
- Regularizer records contain strings plus NumPy int32 codec arrays; the
  5% validation split is MD5(name) modulo 20, kept out of training. The published
  `ar_prep.py` scans a corpus; it does not directly consume the compact pack despite
  the model card's description. A deliberate safe pack loader is needed.
- Joint training additionally needs minted MERT features, latents, true semantic
  tokens and semantic-neighbor arrays. The compact AR regularizer pack alone is
  insufficient. Loss includes straight-through head tokens and minted soft CE;
  25% of NAR flow windows are minted when NAR training is enabled.
- Source training algorithms and weights must retain applicable upstream terms;
  the model card declares CC BY-NC 4.0. Do not label copied scripts MIT.

## AR artist-training checkpoint (added later on this branch)

The biggest technical gap — real audio → real YuE2 semantic IDs → trained AR LoRA →
measurable native ComfyUI effect — is now closed. End-to-end run of
`tools/verify_artist_pipeline.py` on a real song (4 s excerpt, fixed seeds):

1. **Semantic extraction parity:** MERT-v2-FullSong (pinned revision) + Mothersuperior
   tokenizer head produced 100 semantic tokens from the excerpt; features matched the
   published upstream `mert_l20` bit-exactly, and retained-window head IDs matched the
   published upstream `Tok`/`instnorm`/`predict` bit-exactly. Second run hit the cache
   and reproduced identical codec IDs.
2. **AR training:** three steps (rank 4, max length 2048, alternating artist next-token
   loss on real semantic IDs with minted regularizer steps) produced nonzero deltas in
   all 112 targeted native modules (28 layers × qkv/gate_up/o/down, 336 tensors).
3. **Native application:** the trained adapter loaded through ComfyUI's standard
   `load_lora_for_models` path matched all 112 keys with no unmatched keys.
4. **Measurable effect (fixed prompt, seed 42):**

   | Strength | logits max abs diff | logits mean abs diff | weight max abs diff |
   |---------:|--------------------:|---------------------:|--------------------:|
   | 1.0 | 0.1875 | 0.00527 | 3.81e-06 |
   | 2.0 | 0.25 | 0.00702 | 7.63e-06 |
   | 0.0 (again) | 0.0 | 0.0 | 0.0 |

   Weight deltas scaled linearly with strength, and returning to strength 0 restored
   the probed weight, the prefill logits, and the 8 sampled semantic IDs exactly.
   The 8 sampled tokens did not flip at any strength in this short smoke run — the
   effect is proven on logits, and longer training/generation is needed before any
   claim about generated token or audio differences.
5. **Cost:** peak VRAM 6.5 GiB for the whole pipeline on an RTX 5090.

Caveat: this is a wiring/parity/effect proof with three training steps, not an
artist-resemblance result. Perceptual quality, best step count, and best strength
remain unverified, matching the NAR caveat above.

## Outstanding work

Mothersuperior assets, safe PT migration, semantic preprocessing, and AR training
are implemented on this branch with the parity and effect evidence above. Cursor
lyric alignment, full joint (AR+NAR) adaptation training, and the AR patching
example workflow remain unimplemented. Issue #1's original NAR failure remains
undiagnosed without the affected file.

Suggested audit commit: `Validate YuE2 NAR LoRA exports and native patch application`

Suggested audit PR title: `Add YuE2 NAR LoRA effect diagnostics and strict loading`

Suggested description: Add a loader that rejects zero-delta or unmatched NAR
adapters and reports patch applicability, and stop silently dropping unsupported
conversion targets. Preserve existing workflow schemas. Validate with 16 CPU tests
and a three-step diagnostic adapter whose 116 native patches change deterministic
NAR output at strengths 1 and 2 and restore baseline at 0. Issue #1's original
failure remains undiagnosed without the affected file; artist training integration
is outside this audit change. Do not push or open the PR automatically.

## Working-tree inventory

Modified: `.gitignore`, `README.md`, `nodes.py`, `trainer_core/convert.py`,
`trainer_core/train.py`.

Added: this report, `pytest.ini`, `tests/test_native_conversion.py`,
`tests/test_inspection.py`, `tools/train_diagnostic_lora.py`,
`tools/verify_native_lora.py`, `trainer_core/inspection.py`.

Final CPU result: 16 passed; one existing pynvml deprecation warning.
`git diff --check` passed. Tracked-file `git diff --stat` (new files are untracked):

```text
 .gitignore              |  1 +
 README.md               | 31 ++++++++++++++++++++++++-------
 nodes.py                | 28 ++++++++++++++++++++++++++--
 trainer_core/convert.py | 10 ++++++++++
 trainer_core/train.py   |  6 +++++-
 5 files changed, 66 insertions(+), 10 deletions(-)
```

```text
 M .gitignore
 M README.md
 M nodes.py
 M trainer_core/convert.py
 M trainer_core/train.py
?? docs/
?? pytest.ini
?? tests/
?? tools/
?? trainer_core/inspection.py
```
