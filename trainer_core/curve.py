"""Render the trainer's text log as a loss / learning-rate chart image.

The YuE2 LoRA Trainer emits lines like
    step 10/3000  loss=2.05150  lr=1.00e-04  t=0.463  [song.mp3 #2]  0.14s/it
which this module parses and plots in a dark, ComfyUI-styled figure.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s+loss=([0-9.]+)\s+lr=([0-9.eE+-]+)\s+t=([0-9.]+)")
LORA_RE = re.compile(r"final LoRA saved.*:\s*(.+)$")
TRIGGER_RE = re.compile(r"trigger='([^']*)'")


def parse_log(text: str):
    """Return (steps, losses, lrs, meta) parsed from a training_log string."""
    steps, losses, lrs = [], [], []
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(3)))
            lrs.append(float(m.group(4)))
    meta = {"total": 0, "lora_name": "", "trigger": ""}
    for line in text.splitlines():
        m = LORA_RE.search(line)
        if m:
            meta["lora_name"] = Path(m.group(1).strip()).stem
        m = TRIGGER_RE.search(line)
        if m:
            meta["trigger"] = m.group(1)
        m = STEP_RE.search(line)
        if m:
            meta["total"] = int(m.group(2))
    return steps, losses, lrs, meta


def smooth_curve(values, window: int):
    """Centered moving average; window <= 1 returns the input unchanged."""
    if window <= 1 or len(values) < 3:
        return np.asarray(values, dtype=np.float64)
    window = min(int(window), len(values))
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(np.asarray(values, dtype=np.float64), pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def render_chart(steps, losses, lrs, meta, out_path,
                 smooth: int = 25, show_raw: bool = True, show_lr: bool = True,
                 figsize=(10, 5.6), dpi: int = 160):
    """Render the chart PNG with matplotlib (Agg, dark theme)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not steps:
        raise ValueError("No training steps found in the log — connect the "
                         "trainer's training_log output (run it first).")

    x = np.asarray(steps)
    loss = np.asarray(losses)
    lr = np.asarray(lrs)

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#0b1220")
    ax1.set_facecolor("#0b1220")

    # Loss (left axis): raw faint trace + smoothed glow line
    if show_raw:
        ax1.plot(x, loss, color="#2dd4ee", alpha=0.18, linewidth=0.8,
                 label="loss (raw)")
    sm = smooth_curve(loss, smooth)
    ax1.plot(x, sm, color="#22d3ee", linewidth=2.2, label="loss (smoothed)")
    ax1.fill_between(x, sm, loss.min(), color="#22d3ee", alpha=0.06)
    ax1.set_xlabel("step", color="#9fb3c8")
    ax1.set_ylabel("flow-matching loss", color="#22d3ee")
    ax1.tick_params(axis="y", colors="#22d3ee")
    ax1.tick_params(axis="x", colors="#9fb3c8")
    ax1.grid(True, color="#1e3a5f", alpha=0.45, linewidth=0.6)
    ax1.set_xlim(x.min(), x.max() if x.max() > x.min() else x.min() + 1)

    # LR (right axis, log scale)
    if show_lr:
        ax2 = ax1.twinx()
        ax2.set_facecolor("#0b1220")
        ax2.plot(x, lr, color="#f59e0b", linewidth=1.8, alpha=0.9,
                 label="learning rate")
        ax2.set_yscale("log")
        ax2.set_ylabel("learning rate", color="#f59e0b")
        ax2.tick_params(axis="y", colors="#f59e0b")
        ax2.spines["right"].set_color("#f59e0b")
        lines = ax1.get_legend_handles_labels()
        lines2 = ax2.get_legend_handles_labels()
        ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1],
                   loc="upper right", facecolor="#0f2a5f", edgecolor="#1e3a5f",
                   labelcolor="#dbe7f3", fontsize=8)
    else:
        ax1.legend(loc="upper right", facecolor="#0f2a5f", edgecolor="#1e3a5f",
                   labelcolor="#dbe7f3", fontsize=8)

    for side in ("top", "left", "bottom"):
        ax1.spines[side].set_color("#1e3a5f")
    if not show_lr:
        ax1.spines["right"].set_color("#1e3a5f")

    title = "YuE2 LoRA training"
    if meta.get("lora_name"):
        title += f" — {meta['lora_name']}"
    subtitle_bits = []
    if meta.get("trigger"):
        subtitle_bits.append(f"trigger: {meta['trigger']}")
    if meta.get("live"):
        subtitle_bits.append(f"step {steps[-1]}/{meta.get('total') or '?'} · loss {loss[-1]:.4f}")
    else:
        if meta.get("total"):
            subtitle_bits.append(f"{meta['total']} steps")
        subtitle_bits.append(f"final loss {loss[-1]:.4f}")
    ax1.set_title(title + "\n" + " · ".join(subtitle_bits),
                  color="#dbe7f3", fontsize=11, pad=10)

    fig.tight_layout()
    fig.savefig(str(out_path), facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
