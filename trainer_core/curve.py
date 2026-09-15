"""Render the trainer's text log as a loss / learning-rate chart image.

The YuE2 LoRA Trainer emits lines like
    step 10/3000  loss=2.05150  lr=1.00e-04  t=0.463  [song.mp3 #2]  0.14s/it
which this module parses and plots in a dark, ComfyUI-styled figure.

The YuE2 Artist AR LoRA Trainer instead emits one JSON object per line, e.g.
    {"kind": "training", "step": 3, "lr": 6e-06, "grad_norm": 0.22,
     "artist_loss": 6.06, "minted_loss": 3.71}
    {"kind": "evaluation", "step": 3, "artist_loss": 6.06, "minted_val_loss": 3.77}
    {"kind": "configuration", "trainable_parameters": 4358144, ...}
The artist loss is the main curve; minted training loss is a faint second line
and the held-out minted_val points are plotted as red markers — that series
must stay flat, a rising value means the LoRA is damaging YuE2's token grammar.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s+loss=([0-9.]+)\s+lr=([0-9.eE+-]+)\s+t=([0-9.]+)")
LORA_RE = re.compile(r"final LoRA saved.*:\s*(.+)$")
TRIGGER_RE = re.compile(r"trigger='([^']*)'")


def _parse_ar_line(line, state):
    """Consume one JSONL record from the AR trainer; True if the line matched."""
    try:
        row = json.loads(line)
    except ValueError:
        return False
    if not isinstance(row, dict) or "kind" not in row:
        return False
    kind = row["kind"]
    if kind == "configuration":
        cfg = row.get("config") or {}
        state["meta"]["ar"] = True
        state["meta"]["total"] = cfg.get("steps", 0)
    elif kind in ("training", "evaluation"):
        state["meta"]["ar"] = True
        step = row.get("step")
        if step is None:
            return True
        if kind == "training":
            if row.get("lr") is not None:
                state["lrs"][step] = row["lr"]
            if row.get("artist_loss") is not None:
                state["artist"][step] = row["artist_loss"]
            if row.get("minted_loss") is not None:
                state["minted_train"][step] = row["minted_loss"]
        else:
            if row.get("artist_loss") is not None:
                state["artist_eval"][step] = row["artist_loss"]
            if row.get("minted_val_loss") is not None:
                state["minted_val"][step] = row["minted_val_loss"]
    return True


def parse_log(text: str):
    """Return (steps, losses, lrs, meta) parsed from a training_log string."""
    steps, losses, lrs = [], [], []
    ar = {"artist": {}, "minted_train": {}, "artist_eval": {},
          "minted_val": {}, "lrs": {}, "meta": {}}
    legacy = False
    for line in text.splitlines():
        if not legacy and line.lstrip().startswith("{"):
            _parse_ar_line(line, ar)
        else:
            legacy = True
            m = STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(3)))
                lrs.append(float(m.group(4)))
    meta = {"total": 0, "lora_name": "", "trigger": ""}
    if ar["artist"] or ar["minted_train"]:
        # AR trainer log: main curve is the artist loss (minted-only runs fall
        # back to the minted training loss), lr carried per step.
        by_step = ar["artist"] or ar["minted_train"]
        steps = sorted(by_step)
        losses = [by_step[s] for s in steps]
        lrs = []
        last = None
        for s in steps:
            last = ar["lrs"].get(s, last)
            lrs.append(last if last is not None else 0.0)
        meta.update(ar["meta"])
        meta["series"] = [
            ("minted train loss", sorted(ar["minted_train"].items()),
             "#a78bfa", "line"),
            ("artist eval loss", sorted(ar["artist_eval"].items()),
             "#f8fafc", "scatter"),
            ("minted val (must stay flat)", sorted(ar["minted_val"].items()),
             "#f87171", "scatter"),
        ]
        meta["series"] = [(n, [s for s, _ in pts], [v for _, v in pts], c, st)
                          for n, pts, c, st in meta["series"] if pts]
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
    ax1.margins(y=0.12)

    # Loss (left axis): raw faint trace + smoothed glow line
    if show_raw:
        ax1.plot(x, loss, color="#2dd4ee", alpha=0.18, linewidth=0.8,
                 label="loss (raw)")
    sm = smooth_curve(loss, smooth)
    ax1.plot(x, sm, color="#22d3ee", linewidth=2.2, label="loss (smoothed)")
    ax1.fill_between(x, sm, loss.min(), color="#22d3ee", alpha=0.06)
    # Extra AR-trainer series: minted train line + held-out eval markers.
    for name, xs, ys, color, style in meta.get("series", []):
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        if style == "scatter":
            ax1.plot(xs, ys, "o", color=color, markersize=4.5,
                     markeredgecolor="#0b1220", markeredgewidth=0.4,
                     linestyle="None", label=name, zorder=5)
        else:
            ax1.plot(xs, ys, color=color, linewidth=1.0, alpha=0.55,
                     label=name)
    ax1.set_xlabel("step", color="#9fb3c8")
    ax1.set_ylabel("cross-entropy loss" if meta.get("ar")
                   else "flow-matching loss", color="#22d3ee")
    ax1.yaxis.get_offset_text().set_color("#22d3ee")
    ax1.tick_params(axis="y", colors="#22d3ee")
    ax1.tick_params(axis="x", colors="#9fb3c8")
    ax1.grid(True, color="#1e3a5f", alpha=0.45, linewidth=0.6)
    ax1.set_xlim(x.min(), x.max() if x.max() > x.min() else x.min() + 1)

    # LR (right axis, log scale)
    if show_lr:
        ax2 = ax1.twinx()
        ax2.set_facecolor("#0b1220")
        ax2.margins(y=0.25)
        ax2.plot(x, lr, color="#f59e0b", linewidth=1.8, alpha=0.9,
                 label="learning rate")
        ax2.set_yscale("log")
        ax2.set_ylabel("learning rate", color="#f59e0b")
        ax2.tick_params(axis="y", colors="#f59e0b", which="both")
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

    title = "YuE2 AR artist training" if meta.get("ar") else "YuE2 LoRA training"
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
    # Theme the tick labels at the very last moment: log axes regenerate
    # labels at draw time (major vs minor depends on the data range) and
    # tight_layout can recreate them, so anything colored earlier is lost.
    fig.canvas.draw()
    for axis, color in ((ax1.yaxis, "#22d3ee"),) + (((ax2.yaxis, "#f59e0b"),) if show_lr else ()):
        for label in list(axis.get_ticklabels()) + [axis.get_offset_text()]:
            label.set_color(color)
    fig.savefig(str(out_path), facecolor=fig.get_facecolor())
    plt.close(fig)
    return Path(out_path)
