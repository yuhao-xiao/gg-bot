"""Train the multi-task head (geocell + auxiliary country) on cached DINOv2
embeddings, combining one or more sources with optional per-cell balancing.

Frozen-backbone: trains on precomputed embeddings only (no images/backbone), so
it's fast. Reports cell top-1, country top-1, and median great-circle error on a
held-out split.

Usage:
    python scripts/train_head.py --names train,google_train,europe,kaggle --balanced 80
    python scripts/train_head.py --names train            # single source
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from geogg.countries import country_index
from geogg.dataset import balanced_sample
from geogg.geocells import Grid, cell_to_medium
from geogg.heads import MultiTaskHead
from geogg.metrics import haversine_km, haversine_soft_target_matrix
from geogg.paths import EMB_DIR, GRID_PATH, MODELS_DIR


def load_sources(names: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    embs, idxs = [], []
    for name in names:
        embs.append(np.load(EMB_DIR / f"{name}.npy"))
        idxs.append(pd.read_parquet(EMB_DIR / f"{name}_index.parquet"))
    emb = np.concatenate(embs, axis=0)
    idx = pd.concat(idxs, ignore_index=True)
    idx["_row"] = np.arange(len(idx))
    return emb, idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="train", help="comma-separated embedding basenames to combine")
    ap.add_argument("--balanced", type=int, default=0, help="max images per cell (0 = use all)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--trunk-layers", type=int, default=1, help="shared-trunk depth (1 = current)")
    ap.add_argument("--label-smoothing", type=float, default=0.1, help="epsilon for soft labels (all 3 heads)")
    ap.add_argument("--tau", type=float, default=400.0, help="Haversine label-smoothing scale in km (cell head)")
    ap.add_argument("--medium-weight", type=float, default=0.5, help="weight on the medium-region loss")
    ap.add_argument("--country-weight", type=float, default=0.5, help="weight on the country loss")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = Grid.load(GRID_PATH)
    c2i = country_index()
    n_countries = len(c2i)
    cell2med, n_mediums = cell_to_medium(grid)
    print(f"grid: {grid.num_classes} cells | {n_mediums} medium regions | {n_countries} countries")

    emb, idx = load_sources(args.names.split(","))
    print(f"loaded {len(emb):,} embeddings from {args.names} | dim {emb.shape[1]}")
    print(f"sources: {idx['source'].value_counts().to_dict()}")

    if args.balanced:
        idx = balanced_sample(idx, args.balanced * grid.num_classes, "cell_index", seed=0)
        emb = emb[idx["_row"].to_numpy()]
        idx = idx.reset_index(drop=True)
        print(f"balanced to {len(emb):,} images (<= {args.balanced}/cell)")

    cell_y = idx["cell_index"].to_numpy()
    medium_y = np.array([cell2med[int(c)] for c in cell_y])
    country_y = idx["country"].map(lambda c: c2i.get(c, c2i["UNK"])).to_numpy()
    cell_lat = np.array([grid.index_to_cell[i]["lat"] for i in range(grid.num_classes)])
    cell_lon = np.array([grid.index_to_cell[i]["lon"] for i in range(grid.num_classes)])

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(emb))
    n_val = int(len(emb) * args.val_frac)
    vi, ti = perm[:n_val], perm[n_val:]

    Xtr = torch.from_numpy(emb[ti])
    Xva = torch.from_numpy(emb[vi]).to(device)
    cell_tr = torch.from_numpy(cell_y[ti]).long()
    med_tr = torch.from_numpy(medium_y[ti]).long()
    coun_tr = torch.from_numpy(country_y[ti]).long()
    dl = DataLoader(TensorDataset(Xtr, cell_tr, med_tr, coun_tr), batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = MultiTaskHead(emb.shape[1], grid.num_classes, n_mediums, n_countries,
                          args.hidden, args.dropout, args.trunk_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Haversine-weighted soft labels for the cell head: nearby cells get partial
    # credit during training so the loss directly rewards "close-but-not-exact".
    soft = haversine_soft_target_matrix(cell_lat, cell_lon, tau_km=args.tau, epsilon=args.label_smoothing)
    soft_t = torch.from_numpy(soft).to(device)
    print(f"haversine soft targets: shape {tuple(soft.shape)} | tau={args.tau} km | eps={args.label_smoothing}")
    import torch.nn.functional as F  # noqa: E402

    va_lat, va_lon = idx["lat"].to_numpy()[vi], idx["lon"].to_numpy()[vi]
    best = 0.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for xb, cb, mb, ob in dl:
            xb, cb, mb, ob = xb.to(device), cb.to(device), mb.to(device), ob.to(device)
            opt.zero_grad()
            cell_logits, med_logits, coun_logits = model(xb)
            # cell loss: Haversine-weighted soft-target cross-entropy
            cell_loss = -(soft_t[cb] * F.log_softmax(cell_logits, dim=-1)).sum(-1).mean()
            loss = (cell_loss
                    + args.medium_weight * lossf(med_logits, mb)
                    + args.country_weight * lossf(coun_logits, ob))
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()

        model.eval()
        with torch.no_grad():
            cl, ml, ol = model(Xva)
            cell_pred = cl.argmax(1).cpu().numpy()
            med_pred = ml.argmax(1).cpu().numpy()
            coun_pred = ol.argmax(1).cpu().numpy()
        cell_acc = float((cell_pred == cell_y[vi]).mean())
        med_acc = float((med_pred == medium_y[vi]).mean())
        coun_acc = float((coun_pred == country_y[vi]).mean())
        med_km = float(np.median(haversine_km(cell_lat[cell_pred], cell_lon[cell_pred], va_lat, va_lon)))
        print(f"ep {ep:3d} | loss {tot/len(ti):.3f} | val cell {cell_acc:.3f} medium {med_acc:.3f} country {coun_acc:.3f} | med {med_km:.0f} km")

        if cell_acc > best:
            best = cell_acc
            torch.save(
                {"state_dict": model.state_dict(), "in_dim": emb.shape[1],
                 "n_cells": grid.num_classes, "n_mediums": n_mediums, "n_countries": n_countries,
                 "hidden": args.hidden, "dropout": args.dropout, "trunk_layers": args.trunk_layers,
                 "grid_path": str(GRID_PATH)},
                MODELS_DIR / "head.pt",
            )
    print(f"best val cell acc {best:.3f} -> saved {MODELS_DIR / 'head.pt'}")


if __name__ == "__main__":
    main()
