"""Shared inference: load backbone + head + grid once, predict from images.

Also provides a SOFT hierarchical re-rank: nudge cell scores by the medium/country
head log-probs (weighted, in log-prob space). This is deliberately NOT a hard
country mask -- country top-1 is only ~0.73, so hard-masking would force a wrong
cell whenever the country is wrong. The soft prior (small beta/gamma) can't
override a confident cell, and the weights are tunable (0 = pure cell head).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from geogg.backbone import DinoV2Backbone
from geogg.countries import country_index
from geogg.geocode import CountryLocator
from geogg.geocells import Grid, cell_to_medium
from geogg.heads import MultiTaskHead
from geogg.paths import MODELS_DIR


def _log_softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def rerank_cell_logprobs(cell_logits, med_logits, coun_logits,
                         cell2med: np.ndarray, cell2coun: np.ndarray,
                         beta: float = 0.0, gamma: float = 0.0) -> np.ndarray:
    """Return per-cell log-probs, optionally nudged by medium (beta) + country (gamma)
    head log-probs broadcast onto cells. Shapes: [N, n_cells]."""
    clp = _log_softmax_np(np.atleast_2d(cell_logits))
    if beta:
        clp = clp + beta * _log_softmax_np(np.atleast_2d(med_logits))[:, cell2med]
    if gamma:
        clp = clp + gamma * _log_softmax_np(np.atleast_2d(coun_logits))[:, cell2coun]
    return clp


def confidence_weights(prob: np.ndarray, power: float = 0.0, keep: float = 0.6) -> np.ndarray:
    """Per-heading weights that GATE OUT garbage captures without penalizing merely-uncertain ones.

    Why a gate and not smooth weighting: confidence = how peaked a row is (1 - normalized entropy).
    Any smooth conf**power scheme depends only on the *ratios* of confidence, so it also penalizes a
    heading that's just looking at an ambiguous direction -- which on clean data is still correct, so
    it costs accuracy (measured: 82->85 km). Instead we judge each heading relative to the BEST one
    in the group: any heading at >= `keep` of the best confidence keeps full (equal) weight, so on
    clean data nothing changes. Only a clear outlier -- a sky/wall/blur shot far below its siblings --
    falls off as (rel/keep)**power. power=0 disables the gate entirely. prob: [N, n_cells], rows->1."""
    n = prob.shape[0]
    if power <= 0 or n == 1:
        return np.full(n, 1.0 / n)
    ent = -(prob * np.log(prob + 1e-12)).sum(1)                      # [N], in [0, log K]
    conf = np.clip(1.0 - ent / np.log(prob.shape[1]), 1e-9, None)    # higher = peakier
    rel = conf / conf.max()                                          # best heading -> 1.0
    w = np.where(rel >= keep, 1.0, (rel / keep) ** power)            # full weight unless an outlier
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


def cell_country_vector(grid: Grid, c2i: dict[str, int], cache: Path | None = None) -> np.ndarray:
    """[n_cells] array: country class-index of each cell (reverse-geocoded centroid)."""
    if cache and cache.exists():
        return np.load(cache)
    loc = CountryLocator("50m")
    out = np.array([c2i.get(loc.country_of(c["lat"], c["lon"]) or "UNK", c2i["UNK"])
                    for c in grid.cells], dtype=np.int64)
    if cache:
        np.save(cache, out)
    return out


class Predictor:
    """Loads the trained model + frozen backbone once; predicts from PIL images."""

    def __init__(self, ckpt_path: Path | None = None, image_size: int = 518,
                 beta: float = 0.0, gamma: float = 0.0, conf_power: float = 0.0,
                 device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path or (MODELS_DIR / "head.pt"), map_location=self.device)
        self.grid = Grid.load(ckpt["grid_path"])
        self.c2i = country_index()
        self.idx_to_country = {v: k for k, v in self.c2i.items()}
        c2m, _ = cell_to_medium(self.grid)
        self.cell2med = np.array([c2m[i] for i in range(self.grid.num_classes)], dtype=np.int64)
        cache = Path(ckpt["grid_path"]).with_name("cell_country.npy")
        self.cell2coun = cell_country_vector(self.grid, self.c2i, cache)
        self.beta, self.gamma, self.conf_power = beta, gamma, conf_power

        self.backbone = DinoV2Backbone(dtype=torch.float32, image_size=image_size)
        self.model = MultiTaskHead(ckpt["in_dim"], ckpt["n_cells"], ckpt["n_mediums"],
                                   ckpt["n_countries"], ckpt["hidden"], ckpt["dropout"],
                                   ckpt.get("trunk_layers", 1)).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.cell_lat = np.array([self.grid.index_to_cell[i]["lat"] for i in range(self.grid.num_classes)])
        self.cell_lon = np.array([self.grid.index_to_cell[i]["lon"] for i in range(self.grid.num_classes)])

    @torch.no_grad()
    def predict(self, images: list[Image.Image], topk: int = 5) -> dict:
        """Average the given images' predictions (multi-heading) and return a guess."""
        emb = self.backbone.embed(images)
        cl, ml, ol = self.model(emb.to(self.device))
        cl, ml, ol = cl.cpu().numpy(), ml.cpu().numpy(), ol.cpu().numpy()
        # soft re-rank per image -> normalize -> confidence-weighted average over headings
        clp = rerank_cell_logprobs(cl, ml, ol, self.cell2med, self.cell2coun, self.beta, self.gamma)
        prob = np.exp(clp)
        prob = prob / prob.sum(1, keepdims=True)                       # per-heading cell distribution
        w = confidence_weights(prob, self.conf_power)                  # discount flat/garbage headings
        cell_prob = (w[:, None] * prob).sum(0)
        coun = F.softmax(torch.from_numpy(ol), dim=-1).numpy()
        coun_prob = (w[:, None] * coun).sum(0)

        order = cell_prob.argsort()[::-1][:topk]
        top = [{"cell": int(c), "lat": float(self.cell_lat[c]), "lon": float(self.cell_lon[c]),
                "prob": float(cell_prob[c]),
                "country": self.idx_to_country[int(self.cell2coun[c])]} for c in order]
        ci = int(coun_prob.argmax())
        coun_order = coun_prob.argsort()[::-1][:3]
        country_topk = [{"country": self.idx_to_country[int(i)], "prob": float(coun_prob[i])}
                        for i in coun_order]
        return {"lat": top[0]["lat"], "lon": top[0]["lon"],
                "country": self.idx_to_country[ci], "country_prob": float(coun_prob[ci]),
                "topk": top, "country_topk": country_topk}
