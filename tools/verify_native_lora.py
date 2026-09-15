"""Opt-in GPU regression for issue #1, using installed native ComfyUI APIs.

Run with the ComfyUI venv. Inputs are an existing checkpoint and native LoRA.
This tests a deterministic NAR forward with synthetic conditioning, not song quality.
"""
import argparse
import gc
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--lora', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy_root.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from safetensors.torch import load_file
    import comfy.sd
    import comfy.lora
    import comfy.model_management as mm
    from trainer_core.inspection import inspect_tensors, apply_nar_lora

    tensors = load_file(args.lora)
    inspection = inspect_tensors(tensors)
    model = comfy.sd.load_checkpoint_guess_config_model_only(str(args.checkpoint), disable_dynamic=True)
    mapping = comfy.lora.model_lora_keys_unet(model.model, {})
    keys = [mapping[m['module']] for m in inspection['modules'] if m['module'] in mapping]
    if not keys:
        raise ValueError('Zero native keys matched')
    # Probe the smallest nonzero target to keep diagnostic memory bounded.
    nonzero = [mapping[m['module']] for m in inspection['modules']
               if m['delta_max_abs'] > 0 and m['module'] in mapping]
    key = min(nonzero, key=lambda k: model.model.state_dict()[k].numel())
    outputs, weights, runs = [], [], []
    for strength in (0., 1., 2., 0.):
        # Same API invoked by LoraLoaderModelOnly; independently cross-check
        # the plugin's stricter parser/matching report.
        checked, match_report = apply_nar_lora(model, tensors, strength)
        del checked
        patched, _ = comfy.sd.load_lora_for_models(model, None, tensors, strength, 0.)
        mm.load_models_gpu([patched])
        net = patched.model.diffusion_model
        weights.append(patched.model.state_dict()[key].detach().float().cpu().clone())
        config = net.config
        device, dtype = net.llm2vae.weight.device, net.llm2vae.weight.dtype
        g = torch.Generator(device='cpu').manual_seed(42)
        x = torch.randn(1, 64, 8, generator=g).to(device=device, dtype=dtype)
        context = torch.randn(1, 4, config.num_hidden_layers * 2 * config.num_key_value_heads * config.head_dim,
                              generator=g).to(device=device, dtype=dtype)
        with torch.inference_mode():
            output = net(x, torch.tensor([.5], device=device, dtype=dtype), context, [(0,8,0,4)])
        outputs.append(output.float().cpu())
        difference = (outputs[-1] - outputs[0]).abs()
        runs.append(dict(strength=strength, matched_keys=len(patched.patches),
                         forward_max_abs_diff=difference.max().item(),
                         forward_mean_abs_diff=difference.mean().item(),
                         weight_max_abs_diff=(weights[-1]-weights[0]).abs().max().item()))
        mm.unload_all_models()
        del net, patched
        print(json.dumps(runs[-1]), flush=True)
    assert torch.equal(outputs[0], outputs[3]), 'Strength zero did not restore base output'
    assert torch.equal(weights[0], weights[3]), 'Strength zero did not restore base weight'
    assert not torch.equal(outputs[0], outputs[1]), 'Issue #1: strength 1 has no forward effect'
    assert not torch.equal(outputs[1], outputs[2]), 'Issue #1: strengths 1 and 2 are identical'
    # bf16 rounding acts on the final weight, not solely on the small delta.
    tolerance = 4 * torch.finfo(dtype).eps * max(w.abs().max().item() for w in weights)
    torch.testing.assert_close(weights[2]-weights[0], 2*(weights[1]-weights[0]), atol=tolerance, rtol=.02)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(inspection=inspection, matches=json.loads(match_report),
        probed_weight=key, runs=runs, conditioning='synthetic fixed tensors; not generated semantic tokens',
        peak_vram_gib=torch.cuda.max_memory_allocated()/2**30), indent=2), encoding='utf-8')
    del model
    gc.collect()


if __name__ == '__main__':
    main()
