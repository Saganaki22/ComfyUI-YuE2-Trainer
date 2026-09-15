"""Three-step gradient/export smoke test on synthetic latents, not artist training."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from trainer_core import native_ckpt, lora, train, convert
    from trainer_core.yue2_ref import modeling_yue2, protocol
    from trainer_core.inspection import inspect_tensors
    from safetensors.torch import save_file, load_file

    if args.output.exists():
        raise FileExistsError(f'Diagnostic will not overwrite {args.output}')
    torch.manual_seed(123)
    model = native_ckpt.build_lm_from_native(args.checkpoint, modeling_yue2).to('cuda').eval()
    targets = lora.inject_lora(model, 'nar_attn_mlp_proj', 2, 2, 0.)
    params = lora.trainable_parameters(model)
    opt = torch.optim.AdamW(params, lr=1e-4)
    tokenizer = native_ckpt.YuE2JsonTokenizer(native_ckpt.load_native_tokenizer_json(args.checkpoint))
    ids = train.build_conditioning_ids(tokenizer, protocol, 'diagnostic instrumental')
    cache = train.MaskCache()
    z = torch.randn(8, 64, device='cuda')
    noise = torch.randn_like(z)
    losses = []
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        out = train.training_velocity(model, ids, (z + noise) / 2, 0., cache, protocol)
        loss = torch.nn.functional.mse_loss(out.float(), noise-z)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
        if not torch.isfinite(loss) or not grad_norm > 0:
            raise AssertionError('Non-finite loss or missing gradients')
        opt.step()
        losses.append(dict(step=step+1, loss=loss.detach().item(), grad_norm=grad_norm))
    native, _ = convert.convert_tensors(lora.lora_state_dict(model), {'alpha':2})
    report = inspect_tensors(native)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(native, str(args.output), metadata={'format':'comfyui-native-lora', 'purpose':'synthetic diagnostic only'})
    reloaded = load_file(args.output)
    for key, tensor in native.items():
        torch.testing.assert_close(reloaded[key], tensor, rtol=0, atol=0)
    report.update(losses=losses, targets=len(targets), peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(nonzero_modules=report['nonzero_modules'], losses=losses)), flush=True)


if __name__ == '__main__':
    main()
