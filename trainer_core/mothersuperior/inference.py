"""Validate AR applicability, then delegate loading to standard ComfyUI APIs."""
import json
import logging

from ..inspection import inspect_tensors


def apply_ar(clip,tensors,strength):
    import comfy.lora
    import comfy.sd
    report = inspect_tensors(tensors)
    mapping = comfy.lora.model_lora_keys_clip(clip.cond_stage_model,{})
    state = clip.cond_stage_model.state_dict()
    missing = []
    for row in report['modules']:
        name = row['module']
        if not name.startswith('text_encoders.model.layers.') or name not in mapping:
            missing.append(name)
            continue
        down,up = tensors[name+'.lora_down.weight'],tensors[name+'.lora_up.weight']
        if tuple(state[mapping[name]].shape) != (up.shape[0],down.shape[1]):
            raise ValueError('AR delta shape mismatch: '+name)
    if missing:
        raise ValueError('Unmatched AR keys: '+', '.join(missing))
    patches = comfy.lora.load_lora(tensors,mapping)
    if not patches: raise ValueError('Zero AR keys resolved')
    if strength:
        _,result = comfy.sd.load_lora_for_models(None,clip,tensors,0.,strength)
        unmatched = sorted(set(patches)-set(result.patcher.patches))
        if unmatched: raise ValueError('AR patches rejected: '+str(unmatched))
    else:
        result = clip.clone()
    report.update(candidate_keys=len(patches),matched_keys=len(patches),unmatched_keys=[],
                  ar_keys_applied=len(patches) if strength else 0,strength=strength)
    logging.getLogger('yue2_trainer.artist').info('AR patch report: %s',json.dumps(report))
    return result,json.dumps(report,indent=2)
