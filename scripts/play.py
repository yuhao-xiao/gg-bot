"""Play/test against live GeoGuessr: drive a browser, capture the street-view
element directly from the page, and predict with the model loaded once.

Two browser modes:

  DEFAULT (launch real Chrome): uses your installed Chrome with a persistent
  profile + a stealth tweak (hides navigator.webdriver). Log in with GeoGuessr
  EMAIL/PASSWORD (Google "Sign in" often blocks automated browsers).

  --cdp 9222 (connect to YOUR running Chrome): most reliable for login. Start
  Chrome yourself with remote debugging, log into GeoGuessr normally, then attach:
      & "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" `
        --remote-debugging-port=9222 --user-data-dir="C:\\Users\\xPure\\chrome-geo"
  (use a fresh --user-data-dir the first time; log into GeoGuessr there once.)

In the terminal, [Enter] captures the current view (UI margins trimmed), re-predicts on every
heading captured so far, prints a country-aware diagnostic, and updates a live top-5 map in a
2nd Chrome tab (markers colored/sized by confidence). 'n'=new round, 'q'=quit. A garbage capture
is gated out of the average automatically.

  --overlay  (experimental): also draws the top-5 as colored dots on GeoGuessr's own minimap.
             Installs a Google Maps hook and RELOADS the page once at startup (your login
             persists -- just start a fresh round). Falls back to the tab if the hook misses.
  --save-previews: also write {ts}_crop.png / {ts}_dino.png each capture (margin tuning).

Setup (once):  pip install playwright   &&   playwright install chromium

Usage:
    python scripts/play.py --cdp 9222
    python scripts/play.py --cdp 9222 --overlay
    python scripts/play.py --selector "canvas" --beta 0.3 --gamma 0.2
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

from PIL import Image

from geogg.inference import Predictor
from geogg.paths import ARTIFACTS_ROOT

PROFILE_DIR = ARTIFACTS_ROOT / "browser_profile"
LIVE_DIR = ARTIFACTS_ROOT / "live"
SHOTS_DIR = Path("samples/live")
STEALTH = "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"

# Installed before the page's scripts run: wraps google.maps.Map so every instance is recorded
# on window.__ggMaps (so we can later find the guess minimap and draw on it). Polls because the
# Maps API is loaded asynchronously after document start.
MAP_HOOK = """
(() => {
  if (window.__ggHookInstalled) return;
  window.__ggHookInstalled = true;
  window.__ggMaps = [];
  const t = setInterval(() => {
    try {
      if (window.google && google.maps && google.maps.Map && !google.maps.Map.__ggHooked) {
        const Orig = google.maps.Map;
        const Wrapped = function(...a){ const m = new Orig(...a); window.__ggMaps.push(m); return m; };
        Wrapped.prototype = Orig.prototype;
        Object.keys(Orig).forEach(k => { try { Wrapped[k] = Orig[k]; } catch(e){} });
        google.maps.Map = Wrapped;
        google.maps.Map.__ggHooked = true;
        clearInterval(t);
      }
    } catch(e){}
  }, 50);
})();
"""

# Draws the top-k as fixed-pixel circle markers on the smallest hooked map (= the guess minimap).
# Markers (not Circles) so they stay visible at the minimap's world zoom. Clears the prior set.
DRAW_JS = """
(pts) => {
  if (!window.google || !google.maps || !window.__ggMaps || !window.__ggMaps.length)
    return {ok:false, n:(window.__ggMaps||[]).length};
  let map = null, best = Infinity;
  for (const m of window.__ggMaps) {
    try { const d = m.getDiv(); if(!d) continue;
          const a = d.clientWidth * d.clientHeight;
          if (a > 0 && a < best) { best = a; map = m; } } catch(e){}
  }
  if (!map) return {ok:false, n: window.__ggMaps.length};
  (window.__ggOverlay||[]).forEach(c => c.setMap(null));
  window.__ggOverlay = [];
  pts.forEach((p,i) => {
    const c = new google.maps.Marker({
      map, position:{lat:p.lat, lng:p.lon}, clickable:false, zIndex:1000+pts.length-i,
      icon:{ path: google.maps.SymbolPath.CIRCLE, scale: 5+9*p.t,
             fillColor:p.color, fillOpacity:0.5+0.4*p.t, strokeColor:'#222', strokeWeight:1 }
    });
    window.__ggOverlay.push(c);
  });
  return {ok:true, n: window.__ggMaps.length};
}
"""


def ui_crop(img: Image.Image, left: float, top: float, right: float, bottom: float) -> Image.Image:
    """Single center crop with per-edge margins to trim GeoGuessr UI overlays."""
    W, H = img.size
    return img.crop((int(left * W), int(top * H), int((1 - right) * W), int((1 - bottom) * H)))


def dino_preview(img: Image.Image, size: int = 518) -> Image.Image:
    """Replicate DINOv2's exact preprocessing: resize shorter edge to `size`, center-crop to square.
    The returned image is pixel-for-pixel what the model receives."""
    W, H = img.size
    scale = size / min(W, H)
    W2, H2 = round(W * scale), round(H * scale)
    resized = img.resize((W2, H2), Image.LANCZOS)
    left = (W2 - size) // 2
    top = (H2 - size) // 2
    return resized.crop((left, top, left + size, top + size))


def _hex_color(t: float) -> str:
    """Confidence gradient: t=1 (most likely) -> red, t=0 (least) -> yellow."""
    r = int(240 - 20 * t)
    g = int(220 - 190 * t)
    b = int(40 - 10 * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def viz_points(topk: list[dict]) -> list[dict]:
    """Top-k cells annotated with a normalized confidence `t` and gradient `color`."""
    pmax = max((t["prob"] for t in topk), default=1.0) or 1.0
    return [{"lat": p["lat"], "lon": p["lon"], "country": p["country"], "prob": p["prob"],
             "t": p["prob"] / pmax, "color": _hex_color(p["prob"] / pmax)} for p in topk]


def build_viz_html(topk: list[dict]) -> str:
    """Self-contained Leaflet page: top-k markers sized/colored by confidence, auto-fit.
    Exposes window.updateMarkers(pts) so subsequent captures update in-place without
    re-fetching Leaflet from the CDN (tab loaded once; only JS eval needed after that)."""
    pts = json.dumps(viz_points(topk))
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>guess</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}}</style></head>
<body><div id="map"></div><script>
const map = L.map('map');
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
            {{maxZoom:19, attribution:'(c) OpenStreetMap'}}).addTo(map);
let _markers = [];
window.updateMarkers = function(pts) {{
  _markers.forEach(m => map.removeLayer(m));
  _markers = pts.map((p,i) => {{
    const r = Math.round(8+18*p.t), d=r*2, fs=Math.round(8+5*p.t);
    const op = (0.5+0.4*p.t).toFixed(2);
    const icon = L.divIcon({{
      html: '<div style="width:'+d+'px;height:'+d+'px;border-radius:50%;background:'
           +p.color+';border:1.5px solid #333;opacity:'+op
           +';display:flex;align-items:center;justify-content:center'
           +';font-family:sans-serif;font-weight:bold;font-size:'+fs+'px'
           +';color:#fff;text-shadow:0 1px 2px #000;box-sizing:border-box">'+(i+1)+'</div>',
      className:'', iconSize:[d,d], iconAnchor:[r,r]
    }});
    return L.marker([p.lat,p.lon], {{icon}})
      .bindTooltip('#'+(i+1)+' '+p.country+' '+(p.prob*100).toFixed(0)+'%')
      .addTo(map);
  }});
  if (_markers.length===1) map.setView([pts[0].lat,pts[0].lon],6);
  else if (_markers.length) map.fitBounds(L.featureGroup(_markers).getBounds().pad(0.3));
}};
updateMarkers({pts});
</script></body></html>"""


def streetview_box(page, selector: str | None):
    """Return the bounding box {x,y,width,height} of the street-view element.
    Prefers an explicit selector, else the largest VISIBLE canvas.
    Returns None (with a message) if the page is still navigating."""
    try:
        if selector:
            el = page.query_selector(selector)
            if el and el.bounding_box():
                return el.bounding_box()
        best, best_area = None, 0
        for el in page.query_selector_all("canvas"):
            try:
                if not el.is_visible():
                    continue
            except Exception:
                pass
            box = el.bounding_box()
            if box and box["width"] * box["height"] > best_area:
                best, best_area = box, box["width"] * box["height"]
        return best
    except Exception as e:
        if "execution context" in str(e).lower() or "navigation" in str(e).lower():
            print("  (page is still loading -- wait a moment and try again)")
        else:
            print(f"  (canvas query error: {e})")
        return None


def find_geo_page(ctx):
    """Pick the GeoGuessr tab if attached to an existing browser, else first/new page."""
    for pg in ctx.pages:
        if "geoguessr" in (pg.url or ""):
            return pg
    return ctx.pages[0] if ctx.pages else ctx.new_page()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--conf-power", type=float, default=4.0,
                    help="gate out a garbage heading far less confident than its siblings (0=off)")
    ap.add_argument("--margin-left",   type=float, default=0.04, help="fraction trimmed off left (zoom/logo)")
    ap.add_argument("--margin-top",    type=float, default=0.08, help="fraction trimmed off top (compass/score)")
    ap.add_argument("--margin-right",  type=float, default=0.04, help="fraction trimmed off right")
    ap.add_argument("--margin-bottom", type=float, default=0.00, help="fraction trimmed off bottom ('place your pin' text)")
    ap.add_argument("--overlay", action="store_true",
                    help="experimental: draw top-k on GeoGuessr's minimap (installs a hook + reloads once)")
    ap.add_argument("--save-previews", action="store_true",
                    help="also save {ts}_crop.png / {ts}_dino.png each capture (margin tuning)")
    ap.add_argument("--selector", default=None, help="CSS selector for the street-view element")
    ap.add_argument("--cdp", type=int, default=None, help="connect to your running Chrome on this debug port")
    ap.add_argument("--start-url", default="https://www.geoguessr.com")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("Playwright not installed:\n  pip install playwright\n  playwright install chromium")

    print("loading model (once) ...")
    predictor = Predictor(beta=args.beta, gamma=args.gamma, conf_power=args.conf_power)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    if args.save_previews:
        SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    viz_path = LIVE_DIR / "guess_viz.html"

    with sync_playwright() as p:
        if args.cdp:  # attach to your already-running, already-logged-in Chrome
            browser = p.chromium.connect_over_cdp(f"http://localhost:{args.cdp}")
            ctx = browser.contexts[0]
            page = find_geo_page(ctx)
            print(f"attached to Chrome on :{args.cdp} (tab: {page.url})")
        else:  # launch real Chrome with a persistent profile + stealth
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            try:
                ctx = p.chromium.launch_persistent_context(
                    str(PROFILE_DIR), headless=False, channel="chrome",
                    viewport={"width": 1280, "height": 800})
            except Exception:
                print("(real Chrome not found; falling back to bundled Chromium)")
                ctx = p.chromium.launch_persistent_context(
                    str(PROFILE_DIR), headless=False, viewport={"width": 1280, "height": 800})
            ctx.add_init_script(STEALTH)
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(args.start_url)
            print("Log in with GeoGuessr EMAIL/PASSWORD (Google sign-in often blocks bots).")

        if args.overlay:  # install the maps hook; it only takes effect on a fresh navigation
            page.add_init_script(MAP_HOOK)
            print("--overlay: installing Google Maps hook and reloading once "
                  "(login persists -- START A FRESH ROUND after reload).")
            try:
                page.reload(wait_until="load")  # wait for full load, not just DOM
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception as e:
                print(f"  (reload wait: {e} -- continuing anyway)")

        print("\nPan the view and press [Enter] to capture -- the guess re-predicts on ALL")
        print("headings so far; the 2nd tab shows the live top-5 map. 'n'=new round, 'q'=quit.\n")
        headings: list = []
        viz_page = None
        while True:
            cmd = input(f"capture [{len(headings)} heading(s)]> ").strip().lower()
            if cmd == "q":
                break
            if cmd == "n":
                headings = []
                print("  cleared -- ready for a new round")
                continue
            box = streetview_box(page, args.selector)
            if box is None:
                print("  no street-view element found (try --selector); is a round loaded?")
                continue
            # region screenshot (no per-element visibility/stability wait -> grabs current frame)
            png = page.screenshot(clip=box, animations="disabled")
            img = Image.open(io.BytesIO(png)).convert("RGB")
            cropped = ui_crop(img, args.margin_left, args.margin_top, args.margin_right, args.margin_bottom)
            if args.save_previews:
                ts = int(time.time())
                cropped.save(SHOTS_DIR / f"{ts}_crop.png")
                dino_preview(cropped).save(SHOTS_DIR / f"{ts}_dino.png")
            headings.append(cropped)
            r = predictor.predict(headings, topk=args.topk)

            # country-aware diagnostic: guessed-cell country vs the country head
            gc = r["topk"][0]["country"]
            disagree = "  <- cell/country heads disagree" if gc != r["country"] else ""
            coun_str = " | ".join(f"{c['country']} {c['prob']*100:.0f}%" for c in r["country_topk"])
            cells_str = " | ".join(f"#{i+1} {t['country']} {t['lat']:.2f},{t['lon']:.2f} {t['prob']*100:.0f}%"
                                   for i, t in enumerate(r["topk"][:3]))
            print(f"  [{len(headings)} heading(s)] GUESS: {r['lat']:.4f}, {r['lon']:.4f}  (cell -> {gc}){disagree}")
            print(f"    country head: {coun_str}")
            print(f"    top cells: {cells_str}")

            # live top-5 map: load once, update in-place (no CDN re-fetch per capture)
            pts_js = viz_points(r["topk"])
            try:
                if viz_page is None or viz_page.is_closed():
                    viz_path.write_text(build_viz_html(r["topk"]), encoding="utf-8")
                    viz_page = ctx.new_page()
                    viz_page.goto(viz_path.resolve().as_uri())
                    viz_page.wait_for_function("() => typeof window.updateMarkers === 'function'",
                                               timeout=15000)
                else:
                    viz_page.evaluate("pts => window.updateMarkers(pts)", pts_js)
            except Exception as e:
                print(f"    (viz tab error: {e})")

            # experimental: draw on GeoGuessr's own minimap
            if args.overlay:
                try:
                    res = page.evaluate(DRAW_JS, viz_points(r["topk"]))
                    if not res.get("ok"):
                        print(f"    (overlay: guess map not found yet [{res.get('n', 0)} hooked] -- tab still works)")
                except Exception as e:
                    print(f"    (overlay error: {e} -- tab still works)")

        if not args.cdp:
            ctx.close()


if __name__ == "__main__":
    main()
