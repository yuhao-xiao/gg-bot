"""Phase 0 smoke test: push one image through the frozen DINOv2 backbone and
print the embedding shape. This proves the environment + model load + forward
pass all work before we build the rest of the pipeline.

Usage:
    python scripts/smoke_embed.py [path/to/image.jpg]

If no image is given, a solid-colour dummy image is used (the point of this test
is the forward pass, not the content).
"""

import sys

import torch
from PIL import Image

from geogg.backbone import DinoV2Backbone


def main() -> None:
    if len(sys.argv) > 1:
        img = Image.open(sys.argv[1]).convert("RGB")
        source = sys.argv[1]
    else:
        img = Image.new("RGB", (518, 518), (110, 120, 100))
        source = "dummy 518x518 image"

    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    backbone = DinoV2Backbone()
    print(f"loaded {backbone.model_name} on {backbone.device} | embed_dim={backbone.embed_dim}")

    emb = backbone.embed(img)
    print(f"input: {source}")
    print(f"embedding shape: {tuple(emb.shape)} | dtype: {emb.dtype}")
    print(f"first 8 dims: {[round(x, 4) for x in emb[0, :8].cpu().tolist()]}")
    print(f"L2 norm: {emb[0].norm().item():.4f}")


if __name__ == "__main__":
    main()
