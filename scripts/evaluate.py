"""Evaluate the trained multi-task head on a test set (default: the Google test
set = real GeoGuessr domain). Reports cell top-1, country top-1, great-circle
error, threshold accuracies, and GeoGuessr score.

Usage:
    python scripts/evaluate.py --name google_test
    python scripts/evaluate.py --name test          # Mapillary in-domain sanity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from geogg.countries import country_index
from geogg.geocells import Grid, cell_to_medium
from geogg.heads import MultiTaskHead
from geogg.inference import cell_country_vector, confidence_weights, rerank_cell_logprobs
from geogg.metrics import geoguessr_score, haversine_km, threshold_accuracies
from geogg.paths import EMB_DIR, MODELS_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="blalexa_test")
    ap.add_argument("--by-pano", action="store_true",
                    help="average the headings of each pano_id (deployment-realistic: you pan around in-game)")
    ap.add_argument("--beta", type=float, default=0.0, help="soft medium-head prior weight")
    ap.add_argument("--gamma", type=float, default=0.0, help="soft country-head prior weight")
    ap.add_argument("--conf-power", type=float, default=0.0,
                    help="confidence-weighted heading averaging (0=plain mean; only affects --by-pano)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(MODELS_DIR / "head.pt", map_location=device)
    grid = Grid.load(ckpt["grid_path"])
    c2i = country_index()
    cell2med_d, _ = cell_to_medium(grid)
    cell2med = np.array([cell2med_d[i] for i in range(grid.num_classes)], dtype=np.int64)
    cell2coun = cell_country_vector(grid, c2i, Path(ckpt["grid_path"]).with_name("cell_country.npy"))

    model = MultiTaskHead(ckpt["in_dim"], ckpt["n_cells"], ckpt["n_mediums"], ckpt["n_countries"],
                          ckpt["hidden"], ckpt["dropout"], ckpt.get("trunk_layers", 1)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    emb = np.load(EMB_DIR / f"{args.name}.npy")
    idx = pd.read_parquet(EMB_DIR / f"{args.name}_index.parquet").reset_index(drop=True)

    cell_lat = np.array([grid.index_to_cell[i]["lat"] for i in range(grid.num_classes)])
    cell_lon = np.array([grid.index_to_cell[i]["lon"] for i in range(grid.num_classes)])

    with torch.no_grad():
        cl, ml, ol = model(torch.from_numpy(emb).to(device))
        cl, ml, ol = cl.cpu().numpy(), ml.cpu().numpy(), ol.cpu().numpy()
    # soft hierarchical re-rank of cell scores (beta/gamma=0 -> pure cell head)
    cell_prob = np.exp(rerank_cell_logprobs(cl, ml, ol, cell2med, cell2coun, args.beta, args.gamma))
    cell_prob = cell_prob / cell_prob.sum(1, keepdims=True)  # per-image distribution (for conf weights)
    med_prob = F.softmax(torch.from_numpy(ml), dim=-1).numpy()
    coun_prob = F.softmax(torch.from_numpy(ol), dim=-1).numpy()

    # optionally average the per-heading probs across each location, weighting confident headings more
    if args.by_pano and "pano_id" in idx.columns:
        groups = idx.groupby("pano_id").indices
        rows = [g[0] for g in groups.values()]   # one representative row per pano (shared lat/lon/cell)
        members = [list(g) for g in groups.values()]
        weights = [confidence_weights(cell_prob[m], args.conf_power) for m in members]
        agg = lambda mat: np.stack([(w[:, None] * mat[m]).sum(0) for w, m in zip(weights, members)])
        cell_prob, med_prob, coun_prob = agg(cell_prob), agg(med_prob), agg(coun_prob)
        idx = idx.iloc[rows].reset_index(drop=True)
        n_eval = len(idx)
        unit = f"{n_eval:,} locations (headings averaged)"
    else:
        n_eval = len(emb)
        unit = f"{n_eval:,} images"

    cell_pred = cell_prob.argmax(1)
    med_pred = med_prob.argmax(1)
    coun_pred = coun_prob.argmax(1)

    true_cell = idx["cell_index"].to_numpy()
    true_medium = np.array([cell2med[int(c)] for c in true_cell])
    true_lat, true_lon = idx["lat"].to_numpy(), idx["lon"].to_numpy()
    true_country = idx["country"].map(lambda c: c2i.get(c, c2i["UNK"])).to_numpy()
    dist = haversine_km(cell_lat[cell_pred], cell_lon[cell_pred], true_lat, true_lon)

    print(f"\n=== TEST: {args.name} ({unit}) ===")
    print(f"cell top-1     : {(cell_pred == true_cell).mean():.3f}")
    print(f"medium top-1   : {(med_pred == true_medium).mean():.3f}")
    print(f"country top-1  : {(coun_pred == true_country).mean():.3f}")
    print(f"median dist    : {np.median(dist):.0f} km")
    print(f"mean dist      : {dist.mean():.0f} km")
    for name, a in threshold_accuracies(dist).items():
        print(f"  acc@{name:12s}: {a:.3f}")
    print(f"GeoGuessr score: {geoguessr_score(dist).mean():.0f} / 5000 avg")


if __name__ == "__main__":
    main()
