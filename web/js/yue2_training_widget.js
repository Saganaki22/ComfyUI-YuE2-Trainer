// Live training widget for the YuE2 Artist AR LoRA Trainer node.
//
// The trainer pushes yue2.training.progress events over the WebSocket
// (status / point / eval / checkpoint / complete, each carrying the full
// series history so a page refresh or missed message self-heals). This
// extension attaches a DOM widget to the node and renders stats, a progress
// bar, and a canvas chart with per-series display options.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EVENT = "yue2.training.progress";
const OPTS_KEY = "yue2train.opts.v1";
const COLORS = {
  bg: "#0b1220", grid: "#1e3a5f", text: "#9fb3c8",
  artist: "#22d3ee", minted: "#a78bfa", val: "#f87171",
  eval: "#f8fafc", lr: "#f59e0b", grad: "#94a3b8", bar: "#22d3ee",
};

const DEFAULT_OPTS = {
  artist: true, minted: true, val: true, eval: true, lr: true, grad: false,
  smooth: 12,
};

function loadOpts() {
  try {
    return { ...DEFAULT_OPTS, ...JSON.parse(localStorage.getItem(OPTS_KEY) || "{}") };
  } catch {
    return { ...DEFAULT_OPTS };
  }
}
function saveOpts(opts) {
  try { localStorage.setItem(OPTS_KEY, JSON.stringify(opts)); } catch {}
}

function el(tag, style, parent, text) {
  const node = document.createElement(tag);
  if (style) node.style.cssText = style;
  if (text !== undefined) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}

function smooth(values, window) {
  if (window <= 1 || values.length < 3) return values;
  const out = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= window) sum -= values[i - window];
    out.push(sum / Math.min(i + 1, window));
  }
  return out;
}

class TrainingWidget {
  constructor(node, container) {
    this.node = node;
    this.history = [];
    this.total = 0;
    this.opts = loadOpts();

    container.style.cssText =
      "display:flex;flex-direction:column;gap:4px;background:" + COLORS.bg +
      ";border:1px solid " + COLORS.grid + ";border-radius:6px;padding:6px;" +
      "font-family:monospace;font-size:11px;color:" + COLORS.text + ";min-width:360px;";

    const stats = el("div", "display:flex;gap:14px;justify-content:space-between;", container);
    this.stepEl = el("span", "color:" + COLORS.artist, stats, "step 0/0");
    this.artistEl = el("span", "", stats, "artist -");
    this.valEl = el("span", "color:" + COLORS.val, stats, "val -");

    const barRow = el("div", "display:flex;align-items:center;gap:6px;", container);
    this.barWrap = el("div",
      "flex:1;height:8px;background:" + COLORS.grid + ";border-radius:4px;overflow:hidden;", barRow);
    this.barFill = el("div", "height:100%;width:0%;background:" + COLORS.bar + ";", this.barWrap);
    this.barText = el("span", "font-size:10px;min-width:38px;text-align:right;", barRow, "0%");

    this.canvas = el("canvas", "width:100%;height:130px;display:block;", container);
    this.canvas.width = 720;
    this.canvas.height = 260;

    const optsRow = el("div", "display:flex;flex-wrap:wrap;gap:8px;align-items:center;", container);
    const labels = { artist: "artist", minted: "minted", val: "val", eval: "eval", lr: "LR", grad: "grad" };
    this.toggles = {};
    for (const key of Object.keys(labels)) {
      const label = el("label", "display:flex;gap:3px;align-items:center;font-size:10px;cursor:pointer;", optsRow);
      const box = el("input", "", label);
      box.type = "checkbox";
      box.checked = this.opts[key];
      box.addEventListener("change", () => { this.opts[key] = box.checked; saveOpts(this.opts); this.draw(); });
      el("span", "", label, labels[key]);
      this.toggles[key] = box;
    }
    const smoothLabel = el("label", "display:flex;gap:4px;align-items:center;font-size:10px;", optsRow, "smooth");
    this.smoothInput = el("input", "width:60px;", smoothLabel);
    this.smoothInput.type = "range";
    this.smoothInput.min = "1"; this.smoothInput.max = "50"; this.smoothInput.value = String(this.opts.smooth);
    this.smoothInput.addEventListener("input", () => {
      this.opts.smooth = Number(this.smoothInput.value); saveOpts(this.opts); this.draw();
    });

    this.statusEl = el("div", "font-size:10px;color:" + COLORS.text + ";white-space:nowrap;overflow:hidden;text-overflow:ellipsis;", container, "ready");

    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(this.canvas);
    this.draw();
  }

  update(event) {
    if (event.history) this.history = event.history;
    if (event.total) this.total = event.total;
    if (event.type === "point" || event.type === "eval") {
      const step = event.step || 0;
      if (this.total) {
        this.stepEl.textContent = `step ${step}/${this.total}`;
        const pct = Math.min(100, (step / this.total) * 100);
        this.barFill.style.width = pct + "%";
        this.barText.textContent = pct.toFixed(0) + "%";
      }
      const last = (key) => {
        for (let i = this.history.length - 1; i >= 0; i--)
          if (this.history[i][key] != null) return this.history[i][key];
        return null;
      };
      const artist = last("artist"), val = last("minted_val");
      if (artist != null) this.artistEl.textContent = "artist " + artist.toFixed(3);
      if (val != null) this.valEl.textContent = "val " + val.toFixed(3);
      this.draw();
    } else if (event.type === "checkpoint") {
      this.setStatus(`checkpoint: ${event.path} (step ${event.step})`);
    } else if (event.type === "complete") {
      this.setStatus(`complete: ${event.path}`);
    } else if (event.type === "status" && event.message) {
      this.setStatus(event.message);
    }
  }

  setStatus(text) {
    this.statusEl.textContent = text;
    this.statusEl.title = text;
  }

  reset() {
    this.history = [];
    this.total = 0;
    this.stepEl.textContent = "step 0/0";
    this.artistEl.textContent = "artist -";
    this.valEl.textContent = "val -";
    this.barFill.style.width = "0%";
    this.barText.textContent = "0%";
    this.setStatus("training...");
    this.draw();
  }

  series(key) {
    const pts = this.history.filter((p) => p[key] != null);
    return { steps: pts.map((p) => p.step), values: pts.map((p) => p[key]) };
  }

  draw() {
    const ctx = this.canvas.getContext("2d");
    const W = this.canvas.width, H = this.canvas.height;
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, W, H);
    const hist = this.history;
    if (!hist.length) {
      ctx.fillStyle = COLORS.text;
      ctx.fillText("waiting for training data...", 12, H / 2);
      return;
    }
    const o = this.opts;
    const padL = 46, padR = o.lr || o.grad ? 46 : 10, padT = 8, padB = 18;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const maxStep = Math.max(this.total, ...hist.map((p) => p.step), 1);

    const leftKeys = [];
    if (o.artist) leftKeys.push("artist");
    if (o.minted) leftKeys.push("minted");
    let lo = Infinity, hi = -Infinity;
    for (const key of leftKeys) {
      for (const p of hist) {
        const v = p[key];
        if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      }
    }
    for (const key of ["minted_val", "eval_artist"]) {
      if ((key === "minted_val" && !o.val) || (key === "eval_artist" && !o.eval)) continue;
      for (const p of hist) {
        const v = p[key];
        if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      }
    }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 0.05) hi = lo + 0.05;
    const rightKeys = [];
    if (o.lr) rightKeys.push("lr");
    if (o.grad) rightKeys.push("grad_norm");
    let rlo = Infinity, rhi = -Infinity;
    for (const key of rightKeys)
      for (const p of hist) {
        const v = p[key];
        if (v != null) { rlo = Math.min(rlo, v); rhi = Math.max(rhi, v); }
      }
    if (!isFinite(rlo)) { rlo = 0; rhi = 1; }
    if (rhi - rlo < 1e-9) rhi = rlo + 1;

    const X = (s) => padL + (s / maxStep) * plotW;
    const Y = (v) => padT + plotH - ((v - lo) / (hi - lo)) * plotH;
    const YR = (v) => padT + plotH - ((v - rlo) / (rhi - rlo)) * plotH;

    ctx.strokeStyle = COLORS.grid;
    ctx.fillStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.font = "16px monospace";
    for (let i = 0; i <= 4; i++) {
      const v = lo + ((hi - lo) * i) / 4;
      const y = Y(v);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      ctx.fillText(v.toFixed(2), 4, y + 5);
    }
    if (rightKeys.length) {
      ctx.fillStyle = COLORS.lr;
      for (let i = 0; i <= 4; i++) {
        const v = rlo + ((rhi - rlo) * i) / 4;
        ctx.fillText(v.toExponential(0), W - padR + 4, YR(v) + 5);
      }
      ctx.fillStyle = COLORS.text;
    }
    ctx.fillText("0", X(0) - 4, H - 4);
    ctx.fillText(String(maxStep), Math.min(X(maxStep) - 40, W - 60), H - 4);

    const line = (key, color) => {
      const { steps, values } = this.series(key);
      if (values.length < 2) return;
      const smoothed = smooth(values, o.smooth);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      steps.forEach((s, i) => {
        const x = X(s), y = Y(smoothed[i]);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    };
    if (o.artist) line("artist", COLORS.artist);
    if (o.minted) line("minted", COLORS.minted);
    if (rightKeys.length) {
      ctx.lineWidth = 1.4;
      if (o.lr) line("lr", COLORS.lr);
      if (o.grad) line("grad_norm", COLORS.grad);
    }
    const dots = (key, color) => {
      for (const p of hist) {
        const v = p[key];
        if (v == null) continue;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(X(p.step), Y(v), 5, 0, Math.PI * 2);
        ctx.fill();
      }
    };
    if (o.val) dots("minted_val", COLORS.val);
    if (o.eval) dots("eval_artist", COLORS.eval);
  }

  dispose() {
    this.observer.disconnect();
  }
}

const widgets = new Map();

app.registerExtension({
  name: "Starnodes.YuE2Trainer.TrainingWidget",

  nodeCreated(node) {
    const cls = node.constructor?.comfyClass || "";
    if (cls !== "YuE2ArtistARLoRATrainer") return;
    node.setSize([Math.max(node.size[0], 420), Math.max(node.size[1], 300)]);

    const container = document.createElement("div");
    const widget = node.addDOMWidget("yue2_training", "yue2_training", container);
    widget.serialize = false;
    widget.type = "yue2_training";

    let instance = null;
    const timer = setInterval(() => {
      if (!container.isConnected) return;
      clearInterval(timer);
      instance = new TrainingWidget(node, container);
      widgets.set(node.id, instance);
    }, 150);

    const prevRemove = node.onRemoved;
    node.onRemoved = function () {
      clearInterval(timer);
      widgets.get(node.id)?.dispose();
      widgets.delete(node.id);
      prevRemove?.apply(this, arguments);
    };
  },
});

api.addEventListener(EVENT, (event) => {
  const detail = event.detail;
  if (!detail?.node) return;
  const instance = widgets.get(Number(detail.node));
  if (instance) instance.update(detail);
});

api.addEventListener("executing", (event) => {
  const detail = event.detail;
  if (!detail?.node) return;
  const instance = widgets.get(Number(detail.node));
  if (instance) instance.reset();
});
