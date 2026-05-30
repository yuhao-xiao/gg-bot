"""Sample Google Street View COVERAGE by querying the free metadata endpoint at
random points on GSV-country land. Collects panorama locations to build a
coverage-based geocell grid. Resumable, rate-limited, and hard-blocked from the
paid image endpoint (see geogg.google_sv).

Requires GOOGLE_MAPS_API_KEY (loaded from a .env file in the repo root or env).

Usage:
    python scripts/sample_google_coverage.py --queries 1000          # small test
    python scripts/sample_google_coverage.py --target 300000         # collect 300k panos
    python scripts/sample_google_coverage.py --target 300000 --resume

Cost: $0. Only the free metadata endpoint is called. (Keep a GCP budget alert on
anyway as a safety net.)
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from geogg.google_sv import query_metadata
from geogg.landsample import LandSampler, load_all_land_geom, load_gsv_countries_geom
from geogg.paths import GOOGLE_COVERAGE_PARQUET

OUT_DEFAULT = str(GOOGLE_COVERAGE_PARQUET)


class RateLimiter:
    """Allow at most `qps` actions/sec across all threads."""

    def __init__(self, qps: float) -> None:
        self.min_interval = 1.0 / qps
        self.lock = threading.Lock()
        self.next_time = time.monotonic()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next_time:
                sleep_for = self.next_time - now
            else:
                sleep_for = 0.0
                self.next_time = now
            self.next_time += self.min_interval
        if sleep_for > 0:
            time.sleep(sleep_for)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=0, help="stop after this many unique coverage panos")
    ap.add_argument("--queries", type=int, default=0, help="stop after this many metadata queries (e.g. test runs)")
    ap.add_argument("--radius", type=int, default=1000, help="metadata search radius in metres")
    ap.add_argument("--mode", choices=["countries", "land"], default="countries",
                    help="'land' samples all land (Google defines coverage); 'countries' uses the GSV list")
    ap.add_argument("--qps", type=float, default=40.0, help="max metadata queries per second")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-secs", type=float, default=20.0)
    args = ap.parse_args()
    if not args.target and not args.queries:
        ap.error("set --target (unique panos) and/or --queries (query budget)")

    load_dotenv()
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_MAPS_API_KEY not set. Put it in a .env file (see README).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # resume: reload existing coverage so we dedupe + keep going
    rows: list[dict] = []
    seen: set[str] = set()
    if args.resume and out.exists():
        prev = pd.read_parquet(out)
        rows = prev.to_dict("records")
        seen = set(prev["pano_id"].dropna())
        print(f"resumed with {len(seen):,} existing panos")

    if args.mode == "land":
        print("loading all-land polygons (Google metadata will define coverage)...")
        sampler = LandSampler(load_all_land_geom("50m"))
    else:
        print("loading GSV country polygons...")
        sampler = LandSampler(load_gsv_countries_geom("50m"))
    limiter = RateLimiter(args.qps)
    local = threading.local()

    # shared state
    lock = threading.Lock()
    stop = threading.Event()
    stats = {"queries": 0, "hits": 0, "official": 0, "errors": 0}
    abort_msg: list[str] = []

    def session() -> requests.Session:
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s

    def worker() -> None:
        while not stop.is_set():
            limiter.wait()
            lat, lon, country = sampler.sample()
            try:
                meta = query_metadata(session(), lat, lon, api_key, radius=args.radius)
            except Exception:
                with lock:
                    stats["errors"] += 1
                time.sleep(0.5)
                continue

            if meta.status == "REQUEST_DENIED":
                with lock:
                    abort_msg.append("REQUEST_DENIED - check API key, that Street View Static API is enabled, and billing is on")
                stop.set()
                return
            if meta.status == "OVER_QUERY_LIMIT":
                time.sleep(2.0)

            with lock:
                stats["queries"] += 1
                if meta.ok and meta.pano_id and meta.pano_id not in seen:
                    seen.add(meta.pano_id)
                    stats["hits"] += 1
                    if meta.is_google_official:
                        stats["official"] += 1
                    rows.append({
                        "lat": meta.lat, "lon": meta.lon, "pano_id": meta.pano_id,
                        "date": meta.date, "copyright": meta.copyright,
                        "country": country, "official": meta.is_google_official,
                    })

    def reached_goal() -> bool:
        if args.target and stats["hits"] >= args.target:
            return True
        if args.queries and stats["queries"] >= args.queries:
            return True
        return False

    def checkpoint() -> None:
        with lock:
            df = pd.DataFrame(rows)
        df.to_parquet(out, index=False)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()

    bar = tqdm(total=args.target or args.queries, desc="coverage")
    last_ckpt = time.monotonic()
    try:
        while not stop.is_set():
            time.sleep(0.5)
            with lock:
                q, h, off = stats["queries"], stats["hits"], stats["official"]
            bar.n = h if args.target else q
            rate = (100 * h / q) if q else 0
            bar.set_postfix(queries=q, hits=h, hit_rate=f"{rate:.0f}%", official=off, errors=stats["errors"])
            bar.refresh()
            if reached_goal():
                stop.set()
            if time.monotonic() - last_ckpt > args.checkpoint_secs:
                checkpoint()
                last_ckpt = time.monotonic()
    except KeyboardInterrupt:
        stop.set()
    finally:
        for t in threads:
            t.join(timeout=5)
        checkpoint()
        bar.close()

    if abort_msg:
        raise SystemExit(abort_msg[0])

    q, h, off = stats["queries"], stats["hits"], stats["official"]
    print(f"\ndone. queries={q:,} unique_panos={h:,} hit_rate={100*h/max(1,q):.1f}% "
          f"official={off:,} ({100*off/max(1,h):.0f}% of hits)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
