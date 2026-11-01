"""Walk quality probe for the scoreboard: glide frame times, near-black pixels mid-glide, time to the first picture.

Usage: python qa/glide_probe.py --tour D:/walks/place/2026-01-01/tour [--glides 8] [--json out.json]
 - frame time p95 (ms): the gaps between animation frames while a glide runs (a headless browser renders in software, so only compare runs with each other)
 - near-black % mid-glide: with ?slowglide (12x longer) a screenshot is taken half way through each glide; share of the middle of the screen darker than 12/255
 - first picture (ms): navigation until the viewer is ready and showing a picture (desktop size, no throttling)
"""
import argparse, functools, http.server, io, json, os, socket, sys, threading, time
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ARGS = ["--ignore-gpu-blocklist", "--enable-webgl", "--enable-unsafe-swiftshader"]
HOOK = "window.__ft=[];window.__ftOn=false;(function l(t){if(window.__ftOn&&window.__last)window.__ft.push(t-window.__last);window.__last=t;requestAnimationFrame(l)})(0);"


def serve(folder):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()

    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass
    h = functools.partial(Q, directory=folder)

    class Srv(http.server.ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if not isinstance(sys.exc_info()[1], ConnectionError):
                super().handle_error(request, client_address)
    threading.Thread(target=Srv(("127.0.0.1", port), h).serve_forever, daemon=True).start()
    return port


def probe(tour, glides=8, page="index.html"):
    port = serve(tour); url = f"http://127.0.0.1:{port}/{page}"
    out = {}
    with sync_playwright() as p:
        b = p.chromium.launch(args=ARGS)
        ctx = b.new_context(viewport={"width": 1280, "height": 800}); ctx.route(lambda u: not u.startswith("http://127.0.0.1"), lambda r: r.abort())
        pg = ctx.new_page(); pg.add_init_script(HOOK)
        t0 = time.time(); pg.goto(url, wait_until="commit")
        pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=120000, polling=20)
        out["first_picture_ms"] = round((time.time() - t0) * 1000)
        n = pg.evaluate("window.__tour.n"); pg.wait_for_timeout(2500)
        seq = [(i + 1) % n for i in range(min(glides, n))]
        fts, gms = [], []
        cur = pg.evaluate("window.__tour.state().cur")
        for j in seq:
            if j == cur:
                continue
            pg.evaluate("window.__ft.length=0; window.__ftOn=true")
            pg.evaluate(f"window.__tour.go({j})")
            pg.wait_for_function(f"window.__tour.state().cur === {j} && !window.__tour.state().busy", timeout=60000, polling=20)
            pg.evaluate("window.__ftOn=false")
            ft = pg.evaluate("window.__ft"); st = pg.evaluate("window.__tour.state()")
            if st["lastGlideMs"]:
                gms.append(st["lastGlideMs"]); fts += ft
            cur = j; pg.wait_for_timeout(600)
        ctx.close()
        out["glides"] = len(gms); out["glide_ms_median"] = round(float(np.median(gms)), 1) if gms else None
        out["glide_frame_p95_ms"] = round(float(np.percentile(fts, 95)), 1) if fts else None
        out["glide_frame_median_ms"] = round(float(np.median(fts)), 1) if fts else None
        # mid-glide screenshots, slowed 12x
        ctx = b.new_context(viewport={"width": 1280, "height": 800}); ctx.route(lambda u: not u.startswith("http://127.0.0.1"), lambda r: r.abort())
        pg = ctx.new_page(); pg.goto(url + "?slowglide")
        pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=120000)
        pg.wait_for_timeout(2500)
        blacks = []; cur = pg.evaluate("window.__tour.state().cur")
        for j in seq[:6]:
            if j == cur:
                continue
            pg.evaluate(f"setTimeout(() => window.__tour.go({j}), 0)")           # evaluate() would wait for the whole glide
            try:
                pg.wait_for_function("window.__tour.state().gliding", timeout=30000, polling=20)
            except Exception:
                pass
            pg.wait_for_timeout(2500)                                                   # slow glides last 5-8 s
            busy = pg.evaluate("window.__tour.state().gliding")
            im = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("L"), np.float32)
            h, w = im.shape; c = im[int(h * .2):int(h * .8), int(w * .2):int(w * .8)]
            if busy:
                blacks.append(float((c < 12).mean() * 100))
            pg.wait_for_function(f"window.__tour.state().cur === {j} && !window.__tour.state().busy", timeout=90000, polling=50); cur = j; pg.wait_for_timeout(400)
        ctx.close(); b.close()
    out["black_mid_glide_pct_mean"] = round(float(np.mean(blacks)), 2) if blacks else None
    out["black_mid_glide_pct_max"] = round(float(np.max(blacks)), 2) if blacks else None
    out["black_samples"] = len(blacks)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tour", required=True); ap.add_argument("--glides", type=int, default=8); ap.add_argument("--page", default="index.html"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r = probe(a.tour, a.glides, a.page); print(json.dumps(r, indent=1))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
