"""Dataset handling: audio folder scan, caption pairing, VAE latent caching.

Each training item is one audio clip of ``clip_seconds`` seconds. Clips are
cut sequentially from every source file; leftover tails shorter than the
clip length are dropped. Latents are encoded once with the YuE2 VAE
(FP32, 48 kHz stereo -> [frames, 64] at 25 fps) and cached to disk as
float16 .npy files, so training never re-encodes and needs no VAE in VRAM.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
SAMPLE_RATE = 48000
FRAMES_PER_SECOND = SAMPLE_RATE // 1920  # VAE downsampling_ratio = 1920 -> 25 fps

log = logging.getLogger("yue2_trainer.data")


@dataclass
class DatasetItem:
    source: Path
    caption: str
    latent_path: Path
    frames: int
    clip_index: int


@dataclass
class TrainDataset:
    items: list[DatasetItem] = field(default_factory=list)
    cache_dir: Path = None
    clip_seconds: float = 10.0

    @property
    def frames_per_clip(self) -> int:
        return int(round(self.clip_seconds * FRAMES_PER_SECOND))

    def summary(self) -> str:
        files = sorted({item.source for item in self.items})
        captioned = sum(1 for item in self.items if item.caption)
        minutes = sum(item.frames for item in self.items) / FRAMES_PER_SECOND / 60
        lines = [
            f"source files : {len(files)}",
            f"clips        : {len(self.items)} x {self.clip_seconds:.1f}s "
            f"({self.frames_per_clip} latent frames each)",
            f"total audio  : {minutes:.2f} min",
            f"clips with .txt caption: {captioned}",
            f"cache dir    : {self.cache_dir}",
        ]
        for file in files:
            count = sum(1 for item in self.items if item.source == file)
            lines.append(f"  - {file.name}: {count} clip(s)")
        return "\n".join(lines)


def load_audio(path: Path) -> torch.Tensor:
    """Load audio as float32 [2, samples] at 48 kHz."""
    try:
        import torchaudio
        wave, rate = torchaudio.load(str(path))
    except Exception:
        import soundfile as sf
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        wave = torch.from_numpy(data.T.copy())
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    if wave.shape[0] == 1:
        wave = wave.repeat(2, 1)
    elif wave.shape[0] > 2:
        wave = wave[:2]
    if rate != SAMPLE_RATE:
        import torchaudio
        wave = torchaudio.functional.resample(wave, rate, SAMPLE_RATE)
    return wave.float().clamp(-1.0, 1.0)


def scan_folder(folder: Path) -> list[tuple[Path, str]]:
    """Return [(audio_path, caption)] pairs; caption from same-named .txt or ''."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Training folder not found: {folder}")
    pairs = []
    for file in sorted(folder.iterdir()):
        if file.suffix.lower() not in AUDIO_EXTENSIONS or not file.is_file():
            continue
        caption_file = file.with_suffix(".txt")
        caption = ""
        if caption_file.is_file():
            caption = caption_file.read_text(encoding="utf-8", errors="replace").strip()
        pairs.append((file, caption))
    if not pairs:
        raise FileNotFoundError(
            f"No audio files ({', '.join(sorted(AUDIO_EXTENSIONS))}) found in {folder}")
    return pairs


def _cache_key(path: Path, clip_seconds: float) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{clip_seconds}|v1"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def encode_file_latents(vae, audio: torch.Tensor, device: torch.device) -> np.ndarray:
    """Encode a full file to latents [frames, 64] float32 (FP32 VAE, no grad)."""
    with torch.inference_mode():
        encoded = vae.encode(audio.unsqueeze(0).to(device))
        if isinstance(encoded, (tuple, list)):
            encoded = encoded[0]
        z = encoded[0].float().cpu()  # [64, frames]
    if z.shape[0] != 64:
        raise RuntimeError(f"Unexpected VAE latent shape {tuple(z.shape)}")
    return z.T.contiguous().numpy()  # [frames, 64]


def build_dataset(vae, folder: Path, cache_dir: Path, clip_seconds: float,
                  caption_mode: str, default_caption: str, device: torch.device,
                  progress_cb=None) -> TrainDataset:
    """Scan, encode (with disk cache) and return the training dataset."""
    if caption_mode not in {"txt_file", "default", "none"}:
        raise ValueError("caption_mode must be txt_file, default or none")
    if not 1.0 <= clip_seconds <= 60.0:
        raise ValueError("clip_seconds must be within 1..60")
    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs = scan_folder(folder)
    frames_per_clip = int(round(clip_seconds * FRAMES_PER_SECOND))
    items: list[DatasetItem] = []
    for file_index, (audio_path, file_caption) in enumerate(pairs):
        if caption_mode == "txt_file":
            caption = file_caption
        elif caption_mode == "default":
            caption = default_caption.strip()
        else:
            caption = ""
        key = _cache_key(audio_path, clip_seconds)
        latent_file = cache_dir / f"{audio_path.stem}_{key}.npy"
        if latent_file.is_file():
            latents = np.load(latent_file)
        else:
            log.info("Encoding %s to VAE latents ...", audio_path.name)
            audio = load_audio(audio_path)
            latents = encode_file_latents(vae, audio, device)
            np.save(latent_file, latents.astype(np.float16))
        clip_count = latents.shape[0] // frames_per_clip
        if clip_count < 1:
            log.warning("Skipping %s: %.1fs of audio is shorter than one %.1fs clip",
                        audio_path.name, latents.shape[0] / FRAMES_PER_SECOND, clip_seconds)
            continue
        for clip_index in range(clip_count):
            items.append(DatasetItem(audio_path, caption, latent_file,
                                     frames_per_clip, clip_index))
        if progress_cb is not None:
            progress_cb(file_index + 1, len(pairs))
    if not items:
        raise RuntimeError("All source files are shorter than one clip — lower clip_seconds.")
    return TrainDataset(items, cache_dir, clip_seconds)


def load_clip_latents(item: DatasetItem) -> torch.Tensor:
    """Return this clip's latents as float32 [frames, 64]."""
    latents = np.load(item.latent_path)
    start = item.clip_index * item.frames
    return torch.from_numpy(latents[start:start + item.frames].astype(np.float32))
