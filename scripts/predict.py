"""Predict a location from one or more Street View images (e.g. GeoGuessr
screenshots). Multiple images = averaged headings (mimics panning in-game).

Usage:
    python scripts/predict.py samples/round.png
    python scripts/predict.py samples/h0.png samples/h90.png samples/h180.png --map samples/guess.png
    python scripts/predict.py samples/round.png --true 48.85 2.35
    python scripts/predict.py samples/round.png --beta 0.3 --gamma 0.2   # soft hierarchy prior
"""

from __future__ import annotations

import argparse

from PIL import Image

from geogg.inference import Predictor
from geogg.metrics import geoguessr_score, haversine_km


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.3, help="soft medium-head prior weight")
    ap.add_argument("--gamma", type=float, default=0.2, help="soft country-head prior weight")
    ap.add_argument("--conf-power", type=float, default=4.0,
                    help="gate out a garbage image far less confident than the others (0=off)")
    ap.add_argument("--map", default=None, help="save a world map PNG of the guess")
    ap.add_argument("--true", type=float, nargs=2, metavar=("LAT", "LON"))
    args = ap.parse_args()

    predictor = Predictor(beta=args.beta, gamma=args.gamma, conf_power=args.conf_power)
    imgs = [Image.open(p).convert("RGB") for p in args.images]
    r = predictor.predict(imgs, topk=args.topk)

    print(f"\n=== prediction from {len(imgs)} image(s) ===")
    coun_str = " | ".join(f"{c['country']} {c['prob']*100:.0f}%" for c in r["country_topk"])
    print(f"country head: {coun_str}")
    print(f"top-{args.topk} cells (country | lat, lon | prob):")
    for rank, t in enumerate(r["topk"]):
        print(f"  #{rank+1}: {t['country']:3s} | {t['lat']:8.3f}, {t['lon']:8.3f}  | {t['prob']*100:5.1f}%  (cell {t['cell']})")
    guess_coun = r["topk"][0]["country"]
    disagree = "   <- cell/country heads disagree" if guess_coun != r["country"] else ""
    print(f"\nGUESS: {r['lat']:.5f}, {r['lon']:.5f}  (cell -> {guess_coun}){disagree}")
    print(f"  google maps: https://www.google.com/maps?q={r['lat']},{r['lon']}")

    if args.true:
        d = float(haversine_km(r["lat"], r["lon"], args.true[0], args.true[1]))
        print(f"\ntrue: {args.true[0]}, {args.true[1]}")
        print(f"  distance: {d:.0f} km | GeoGuessr score: {geoguessr_score(d):.0f}/5000")

    if args.map:
        import matplotlib
        matplotlib.use("Agg")
        import cartopy.io.shapereader as shpreader
        import matplotlib.pyplot as plt
        from shapely.geometry import MultiPolygon

        fig, ax = plt.subplots(figsize=(16, 8))
        path = shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
        for rec in shpreader.Reader(path).records():
            g = rec.geometry
            for poly in (g.geoms if isinstance(g, MultiPolygon) else [g]):
                x, y = poly.exterior.xy
                ax.plot(x, y, color="#999", lw=0.4)
        for rank, t in enumerate(r["topk"]):
            ax.scatter(t["lon"], t["lat"], s=120 if rank == 0 else 40,
                       c="red" if rank == 0 else "orange", zorder=5, edgecolors="k", linewidths=0.5)
        if args.true:
            ax.scatter(args.true[1], args.true[0], s=140, marker="*", c="lime", zorder=6, edgecolors="k")
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_aspect("equal")
        ax.set_title("red=guess, orange=top-k, green star=true")
        fig.savefig(args.map, dpi=110, bbox_inches="tight")
        print(f"\nsaved map -> {args.map}")


if __name__ == "__main__":
    main()
