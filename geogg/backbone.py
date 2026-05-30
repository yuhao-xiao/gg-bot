"""Frozen DINOv2 backbone used as a feature extractor.

We never update these weights in v1. We push an image through DINOv2 once and
keep the resulting embedding vector. A small trainable head (see geogg.heads,
added later) maps that embedding -> geocell. This is the "frozen backbone +
trained head" approach: the backbone is the (fixed) eyes, the head is the
(trained) decision-maker.
"""

from __future__ import annotations

import os

# DINOv2 is already cached locally; run offline so transformers doesn't ping the
# HF Hub every load (kills the "unauthenticated requests" warning + speeds loading).
# setdefault lets you override (e.g. to fetch a new model) by setting the env var first.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from typing import Iterable  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoImageProcessor, AutoModel  # noqa: E402


def pick_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


class DinoV2Backbone:
    """Wraps a pretrained DINOv2 model as a frozen image -> embedding extractor.

    image_size controls the input resolution after resize+center-crop. DINOv2
    handles variable input sizes via positional-embedding interpolation, so we
    can run at 224 (fast, less detail) up to 518 (slower, preserves fine cues
    like signage and license-plate text). Output embedding dim is unchanged."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        image_size: int = 224,
    ) -> None:
        self.model_name = model_name
        self.image_size = image_size
        self.device = pick_device(device)
        self.dtype = dtype

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        if image_size != 224:  # override the processor's defaults for high-res
            self.processor.size = {"shortest_edge": image_size}
            self.processor.crop_size = {"height": image_size, "width": image_size}

        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        for p in self.model.parameters():  # frozen: no gradients, ever
            p.requires_grad_(False)

        self.embed_dim: int = self.model.config.hidden_size

    @torch.inference_mode()
    def embed(self, images: Image.Image | Iterable[Image.Image]) -> torch.Tensor:
        """Return the CLS-token embedding(s). Shape: [B, embed_dim] (always batched)."""
        if isinstance(images, Image.Image):
            images = [images]
        else:
            images = list(images)

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # DINOv2's pooler_output is the CLS token after the final layernorm:
        # a single global descriptor per image.
        return outputs.pooler_output.float()
