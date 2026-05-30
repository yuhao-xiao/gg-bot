# gg-bot — GeoGuessr region-guessing AI

Predict where a Google Street View image is, by classifying it into an adaptive
geocell. Goal: play GeoGuessr; **distance to the true location matters more than
exact cell** (the game rewards closeness).

## Approach (v1)
- **Frozen `facebook/dinov2-base` (ViT-B/14)** as feature extractor → 768-d embedding.
  Backbone is NOT fine-tuned in v1 (LoRA is the planned v2 upgrade).
- Embeddings are **precomputed once** and cached; only a small head is trained
  (runs in minutes on CPU/GPU). Embedding input res = **518px** (max detail; user choice).
- **Multi-task head** (`geogg/heads.py::MultiTaskHead`): shared trunk → 3 heads:
  - `cell` (5,880) — the main task
  - `medium` (476) — S2 level-4 ancestors of cells (`geocells.cell_to_medium`)
  - `country` (110 = 109 GSV countries + UNK)
- **Loss** = soft-CE(cell) + 0.5·CE(medium) + 0.5·CE(country), label smoothing 0.1.
  Cell head uses **Haversine label smoothing** (`metrics.haversine_soft_target_matrix`,
  Gaussian, τ=400 km) so near-miss cells get partial credit → rewards closeness.
- Predicted location = the cell's centroid (mean of its coverage points, not geometric center).

## Grid (locked) — `artifacts/grid_final/grid.json`
- 5,880 adaptive S2 cells, levels 5–7 (~290 / 145 / 70 km per side), built from
  **400k all-land Google coverage points** (`build_grid.py --source google`).
- Bounded sizes (no giant sparse cells, no within-city splitting); covers everywhere
  Google has official coverage (incl. Greenland). `geogg/geocells.py` builds/loads it.

## Data — Google Street View only (Mapillary was DROPPED; domain mismatch)
All big data on external drive **`E:\gg-data`** (`GG_DATA_ROOT` in `.env`). exFAT.
Indices (small) in `artifacts/index/<name>.parquet`; every index row has
`path, lat, lon, country, cell_index, source`.

| source | ~images | notes |
|---|---|---|
| `blalexa_train` | 530k | HF panoramas → perspective crops (`convert_blalexa.py`, py360convert) |
| `blalexa_test` | 8k | held-out, **pano-disjoint** split (`split_blalexa.py`) — the eval set |
| `europe` | 53k | saleha1wer Dropbox, folder `idx,lat,lon` w/ per-heading files |
| `kaggle50k` | 41k | maxfleurent, filenames `lat_lon.jpg` |
| `paulchambaz` | 10k | Kaggle, `{i}.png` + headerless `coords.csv` (lat,lon by row) |
| `topup` | 25k | billable API fill of sparse cells (`download_google_topup.py`), ~$175 spent |

Train = all except blalexa_test (~660k imgs, 5,663/5,880 cells, ~217 dead cells).
Coverage source of truth: `E:\gg-data\google_coverage_land.parquet` (416k panos, lat/lon/pano_id).
Countries reverse-geocoded via `geogg/geocode.py` (Natural Earth). GSV country
list: `configs/gsv_countries.txt`.

## Module map (`geogg/`)
`paths.py` (all path config; reads `.env`) · `backbone.py` (frozen DINOv2) ·
`geocells.py` (S2 grid + cell_to_medium) · `heads.py` (MultiTaskHead) ·
`metrics.py` (haversine, thresholds, GeoGuessr score, soft-target matrix) ·
`dataset.py` (index builders, balanced_sample) · `geocode.py` (lat/lon→country) ·
`countries.py` · `panorama.py` (equirect→perspective) · `google_sv.py` (free
metadata API) · `google_images.py` (billable static API) · `landsample.py`.

## Pipeline / commands (run from repo root, `.\.venv\Scripts\python.exe`)
1. `precompute_embeddings.py --index artifacts\index\<n>.parquet --name <n> --batch-size 64 --workers 16`
   (resumable via per-chunk checkpoints in `artifacts/embeddings/<n>_chunks/`; merges to `<n>.npy` + `<n>_index.parquet`)
2. `train_head.py --names "blalexa_train,europe,kaggle50k,paulchambaz,topup" --balanced 120 --epochs 40 --tau 400`
   (per-cell balanced cap; saves `artifacts/models/head.pt`)
3. `evaluate.py --name blalexa_test [--by-pano] [--beta B --gamma G] [--conf-power P]`
   (`--by-pano` averages a location's 4 headings = deployment-realistic; `--beta`/`--gamma` =
   soft medium/country re-rank prior, default 0. Reports cell/medium/country acc, km, GeoGuessr score)
- Inspect data: `data_report.py [--names ...] [--balanced N]` — images/cell stats (covered vs
  dead cells, densest cells, source/country breakdown) + world heatmap + histogram → `artifacts/data_report/`.
  Reads only the small `*_index.parquet` (fast). `grid_info.py`/`plot_grid.py` cover the grid itself.

## Inference / playing (`geogg/inference.py::Predictor` loads backbone+head+grid once)
- `predict.py img1 [img2 ...] [--map out.png] [--true LAT LON] [--beta B --gamma G --conf-power P]` —
  multiple images = averaged headings. Prints **country-head top-3**, top-k cells **each labeled
  with its own country**, guess + maps link, and a `heads disagree` flag when the guessed cell's
  country ≠ country-head top-1 (`Predictor.predict` now returns `topk[i]["country"]` + `country_topk`).
- `play.py [--cdp 9222] [--overlay] [--save-previews] [--selector CSS]` — Playwright drives
  GeoGuessr; **Enter accumulates headings** (captures largest visible `<canvas>` via
  `page.screenshot(clip=box)`, re-predicts on ALL so far), 'n'=new round, 'q'=quit. Each capture
  is a **single center crop** (`ui_crop`, per-edge `--margin-{left,top,right,bottom}` fractions to
  trim UI; the wide-landscape minimap/score panels at x>0.72 are mostly cut by DINOv2's center-crop
  anyway). Prints the same country-aware diagnostic as predict.py. **Live top-5 viz**: writes a
  self-contained Leaflet page (`build_viz_html`, free OSM tiles) to `artifacts/live/guess_viz.html`
  and reloads it in a reused 2nd Chrome tab each capture — markers sized/colored by confidence
  (`viz_points`, red=#1 → yellow). `--overlay` (experimental) also draws those dots on GeoGuessr's
  own minimap: installs `MAP_HOOK` (wraps `google.maps.Map` → `window.__ggMaps`) then **reloads the
  page once** at startup, and `DRAW_JS` adds markers to the smallest hooked map; falls back to the
  tab if the hook misses. `--save-previews` writes `{ts}_crop.png`/`{ts}_dino.png` (`dino_preview`
  = exactly what DINOv2 sees) for margin tuning (default off). Launches real Chrome
  (`channel=chrome`) + stealth; **Google sign-in blocks bots → use GeoGuessr email/password, OR
  `--cdp PORT` to attach to your own already-logged-in Chrome started with
  `--remote-debugging-port`**. Single-player only (ToS).
- **Validated deployment knobs:** `--by-pano` + `--beta 0.3 --gamma 0.2` →
  ~82 km median, ~4216/5000 GeoGuessr on blalexa_test. These β/γ are the predict.py/play.py
  defaults. (`os.environ HF_HUB_OFFLINE=1` set in backbone.py → no HF token warning.)
- **Soft hierarchy re-rank** (`rerank_cell_logprobs`): adds β·medium + γ·country head log-probs
  to cell scores. NOT a hard country mask (country top-1 only ~0.73 → hard mask propagates
  errors). Tune β/γ via evaluate; keep only if they help.
- **Confidence gate** (`confidence_weights`, `--conf-power`, default 4 in predict/play, 0 in
  evaluate): when averaging headings, an outlier far less peaked than its siblings (sky/wall/blur)
  is downweighted toward 0; comparable headings keep equal weight. **Provably neutral on clean
  data** (gate never fires → identical 82 km at any power) — pure live-garbage insurance. Smooth
  conf^power weighting was rejected: it penalizes uncertain-but-correct headings (82→85 km).

## Env
- Windows, PowerShell. Python **3.12** venv at `.\.venv` (system 3.14 too new for torch).
- torch 2.6.0+cu124 on RTX 4060 Laptop (8GB). Also: GTX 1080 Ti (11GB, no bf16), Colab Pro.
- `.env`: `GOOGLE_MAPS_API_KEY`, `GG_DATA_ROOT=E:/gg-data`. $300 GCP trial credit (~$125 left).

## Gotchas
- **exFAT + huge flat dirs are slow** (O(n) per file op) — CONFIRMED the cause of
  embedding slowdown. blalexa shards 0–58 = ~140k crops in one flat folder
  (`E:\gg-data\blalexa_perspective\*.jpg`); shards 59+ are sub-sharded (`00059/` etc).
  Reading flat-folder rows ran ~2–4.6 s/it and climbing; subfolder rows ran a stable
  ~1.56 s/it (NOT thermal — it recovered the instant it left flat rows).
  `fix_blalexa_flat.py` relocates flat crops into `legacy/<prefix>/` subfolders if you
  need to re-embed them faster (blalexa_train already embedded, so usually unneeded).
- `np.save(path)` appends `.npy` — don't use `.tmp` suffixes with it.
- Live bot (Playwright) is a deferred later phase. No augmentation in v1 (frozen-backbone).
