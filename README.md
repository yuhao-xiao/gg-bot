# gg-bot

A GeoGuessr **region-guessing AI**: given a Google Street View image, predict what
region of the world it is in.

Approach (v1): geocell **classification** on top of a **frozen DINOv2** backbone.
The backbone (pretrained, never updated in v1) turns an image into an embedding;
a small trained head maps that embedding to a geocell. Backbone fine-tuning and a
live browser-playing bot are later phases. Full plan:
`~/.claude/plans/i-would-like-to-abundant-liskov.md`.

## Setup (Windows, Python 3.12)

PyTorch needs Python <= 3.12 (the system 3.14 is too new). Use the 3.12 venv:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 1) PyTorch + torchvision with CUDA (separate from PyPI; pick the cu version
#    matching your driver — cu124 works for the RTX 4060 / Ada and the 1080 Ti):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) The rest of the project (editable install):
pip install -e .
```

## Quickstart

```powershell
# Phase 0 smoke test: image -> frozen DINOv2 -> embedding shape
python scripts/smoke_embed.py
python scripts/smoke_embed.py path\to\streetview.jpg
```

## Layout

- `geogg/` — the package (backbone, geocells, dataset, heads, ...)
- `scripts/` — runnable entry points (smoke test, data download, train, eval)
- `datasets/` — raw data (gitignored)
- `artifacts/` — embeddings, grids, checkpoints (gitignored)
