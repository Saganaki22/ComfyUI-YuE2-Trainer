"""Pinned downloads of the pre-converted safetensors assets."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path


SAFE_REPO = 'drbaph/yue2-mothersuperior-realaudio-tokenizer-comfyui'
SAFE_REVISION = 'de44fe0d1b8bcd7032096ef8e037c25d40a043e6'
MERT_REPO = 'm-a-p/MERT-v2-FullSong'
MERT_REVISION = 'd8ba1c745e733b3908ce6ad16ebeb17ac7600a42'
HEAD_FILE = 'tokenizer_head_joint_v4.safetensors'
NAR_FILE = 'nar_lora_joint_v4.safetensors'
PACK_FILE = 'minted_regularizer_pack.safetensors'
PACK_META_FILE = 'minted_regularizer_pack.jsonl'


@dataclass
class Assets:
    directory: str
    head: str
    mert: str
    regularizer: str
    nar: str = ''
    mert_revision: str = MERT_REVISION


def default_directory():
    import folder_paths
    path = Path(folder_paths.models_dir)/'yue2_trainer'/'mothersuperior'
    folder_paths.add_model_folder_path('yue2_trainer', str(path.parent))
    return path


def prepare_assets(directory=None, auto_download=False, include_nar=False, check_interrupt=lambda:None):
    """Ensure the converted safetensors and MERT snapshot exist locally.

    The tokenizer head, NAR adapter and minted regularizer pack download
    directly from the pre-converted safetensors repo (no pickle conversion at
    runtime — the upstream .pt files are never touched). Each loader validates
    the file's format tag and embedded provenance when it opens the file.
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    root = Path(directory) if directory else default_directory()
    root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    names = [HEAD_FILE, PACK_FILE, PACK_META_FILE]
    if include_nar:
        names.append(NAR_FILE)
    for name in names:
        target = root/name
        if not target.exists():
            check_interrupt()
            hf_hub_download(SAFE_REPO, name, revision=SAFE_REVISION,
                cache_dir=str(root/'hf_cache'), local_dir=str(root),
                local_files_only=not auto_download)
            downloaded.append(name)
    check_interrupt()
    mert = snapshot_download(MERT_REPO,revision=MERT_REVISION,cache_dir=str(root/'hf_cache'),
        local_files_only=not auto_download,allow_patterns=['*.json','model.safetensors','*.py','LICENSE','THIRD_PARTY_NOTICES.md'])
    result = Assets(str(root),str(root/HEAD_FILE),mert,str(root/PACK_FILE),
                    str(root/NAR_FILE) if include_nar else '')
    return result, json.dumps(dict(assets=asdict(result),downloaded=downloaded,
        safe_repo=SAFE_REPO,safe_revision=SAFE_REVISION),indent=2)
