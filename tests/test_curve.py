"""Training-curve parsing for both the legacy NAR log and the AR JSONL log."""
import json

from trainer_core.curve import parse_log


def test_parse_legacy_nar_log():
    text = ("\n"
            "YuE2 LoRA training: checkpoint=yue2.safetensors trigger='mystyle' steps=3000\n"
            "step 10/3000  loss=2.05150  lr=1.00e-04  t=0.463  [song.mp3 #2]  0.14s/it\n"
            "step 20/3000  loss=1.90000  lr=2.00e-04  t=0.461  [song.mp3 #1]  0.14s/it\n"
            "final LoRA saved -> E:\\loras\\mystyle.safetensors\n")
    steps, losses, lrs, meta = parse_log(text)
    assert steps == [10, 20]
    assert losses == [2.05150, 1.9]
    assert lrs == [1e-4, 2e-4]
    assert meta["total"] == 3000
    assert meta["trigger"] == "mystyle"
    assert meta["lora_name"] == "mystyle"
    assert not meta.get("ar")
    assert meta.get("series", []) == []


def _ar_log():
    rows = [
        {"kind": "configuration", "trainable_parameters": 4358144,
         "config": {"steps": 1600, "rank": 64}},
        {"kind": "training", "step": 1, "lr": 2e-06, "grad_norm": 0.4,
         "artist_loss": 6.06, "minted_loss": None},
        {"kind": "training", "step": 2, "lr": 4e-06, "grad_norm": 0.05,
         "artist_loss": None, "minted_loss": 3.65},
        {"kind": "training", "step": 3, "lr": 6e-06, "grad_norm": 0.2,
         "artist_loss": 6.04, "minted_loss": 3.71},
        {"kind": "evaluation", "step": 3, "artist_loss": 6.03,
         "minted_val_loss": 3.77},
    ]
    return "\n".join(json.dumps(r) for r in rows)


def test_parse_ar_jsonl_log():
    steps, losses, lrs, meta = parse_log(_ar_log())
    assert meta["ar"] is True
    assert meta["total"] == 1600
    # Main curve is the artist loss at steps 1 and 3.
    assert steps == [1, 3]
    assert losses == [6.06, 6.04]
    assert lrs == [2e-06, 6e-06]
    series = {name: (xs, ys, style) for name, xs, ys, _c, style in meta["series"]}
    assert set(series) == {"minted train loss", "artist eval loss",
                           "minted val (must stay flat)"}
    mint_xs, mint_ys, mint_style = series["minted train loss"]
    assert mint_xs == [2, 3] and mint_ys == [3.65, 3.71] and mint_style == "line"
    val_xs, val_ys, val_style = series["minted val (must stay flat)"]
    assert val_xs == [3] and val_ys == [3.77] and val_style == "scatter"
    eval_xs, eval_ys, _ = series["artist eval loss"]
    assert eval_xs == [3] and eval_ys == [6.03]


def test_parse_ar_log_minted_only_fallback():
    text = "\n".join(json.dumps(r) for r in [
        {"kind": "training", "step": 1, "lr": 1e-6, "artist_loss": None,
         "minted_loss": 3.7},
        {"kind": "training", "step": 2, "lr": 2e-6, "artist_loss": None,
         "minted_loss": 3.6},
    ])
    steps, losses, lrs, meta = parse_log(text)
    assert steps == [1, 2]
    assert losses == [3.7, 3.6]
    assert meta["ar"] is True
