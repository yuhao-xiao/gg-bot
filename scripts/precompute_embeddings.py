"""Run every image in an index through the frozen DINOv2 backbone ONCE and cache
the embeddings to disk. After this, head training never touches the images again.

Resumable: writes per-chunk checkpoints (default 20k images each). A crash/hang
loses at most the in-progress chunk; re-running the same command picks up
exactly where it left off.

Usage:
    python scripts/precompute_embeddings.py --index artifacts/index/train.parquet --name train
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from geogg.backbone import DinoV2Backbone
from geogg.paths import EMB_DIR


def _load(path: str) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--name", required=True, help="output basename, e.g. 'train'")
    ap.add_argument("--batch-size", type=int, default=32, help="reduce if OOM (try 16); raise if you have headroom")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=518,
                    help="DINOv2 input resolution after resize+crop (224 fast / 518 high-detail)")
    ap.add_argument("--chunk-size", type=int, default=20000,
                    help="images per checkpoint (smaller = more frequent saves, less to lose on crash)")
    args = ap.parse_args()

    df = pd.read_parquet(args.index).reset_index(drop=True)
    n = len(df)
    print(f"embedding {n:,} images from {args.index}")

    chunk_dir = EMB_DIR / f"{args.name}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (n + args.chunk_size - 1) // args.chunk_size

    # decide which chunks still need processing
    todo_chunks = []
    for ci in range(n_chunks):
        if not (chunk_dir / f"{ci:04d}.npy").exists() or not (chunk_dir / f"{ci:04d}_ok.npy").exists():
            todo_chunks.append(ci)
    print(f"chunks: {n_chunks} total | {n_chunks - len(todo_chunks)} done | {len(todo_chunks)} to do")

    # only load the model if there's work to do
    if todo_chunks:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        backbone = DinoV2Backbone(dtype=dtype, image_size=args.image_size)
        print(f"backbone on {backbone.device} ({dtype}), dim={backbone.embed_dim}, image_size={args.image_size}")
        dim = backbone.embed_dim
        paths = df["path"].tolist()

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for ci in todo_chunks:
                cstart = ci * args.chunk_size
                cend = min(cstart + args.chunk_size, n)
                emb = np.zeros((cend - cstart, dim), dtype=np.float32)
                ok = np.ones(cend - cstart, dtype=bool)

                for batch_start in tqdm(range(0, cend - cstart, args.batch_size),
                                         desc=f"chunk {ci+1}/{n_chunks}"):
                    abs_start = cstart + batch_start
                    abs_end = min(abs_start + args.batch_size, cend)  # clamp to chunk end
                    batch_paths = paths[abs_start:abs_end]
                    imgs = list(ex.map(_load, batch_paths))
                    good = [(i, im) for i, im in enumerate(imgs) if im is not None]
                    for i, im in enumerate(imgs):
                        if im is None:
                            ok[batch_start + i] = False
                    if not good:
                        continue
                    idxs, ims = zip(*good)
                    vecs = backbone.embed(list(ims)).cpu().numpy()
                    for k, gi in enumerate(idxs):
                        emb[batch_start + gi] = vecs[k]

                # checkpoint: write data, then the _ok marker last (its presence = chunk complete)
                np.save(chunk_dir / f"{ci:04d}.npy", emb)
                np.save(chunk_dir / f"{ci:04d}_ok.npy", ok)

    # merge chunks -> final .npy + filtered index
    print("\nmerging chunks ...")
    all_emb_parts, all_ok_parts = [], []
    for ci in range(n_chunks):
        all_emb_parts.append(np.load(chunk_dir / f"{ci:04d}.npy"))
        all_ok_parts.append(np.load(chunk_dir / f"{ci:04d}_ok.npy"))
    all_emb = np.concatenate(all_emb_parts, axis=0)
    all_ok = np.concatenate(all_ok_parts, axis=0)
    kept_emb = all_emb[all_ok]
    kept = df[all_ok].reset_index(drop=True)
    print(f"embedded {len(kept):,} images ({(~all_ok).sum()} failed to load)")

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_DIR / f"{args.name}.npy", kept_emb)
    kept.to_parquet(EMB_DIR / f"{args.name}_index.parquet", index=False)
    print(f"saved {EMB_DIR / f'{args.name}.npy'}  shape={kept_emb.shape}")
    print(f"(chunk dir {chunk_dir} preserved -- can delete after verifying)")


if __name__ == "__main__":
    main()
