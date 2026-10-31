"""How much does the picture change brightness when you hop to the next stop? (the 'flash' in a glide/crossfade)
Opens a built tour in headless Chromium, jumps instantly (?noglide) from every stop to the next one keeping the same world facing, looks
steeply down (mostly concrete floor, whose brightness should match between neighbouring stops) and compares the centre brightness. Run it on two builds to see the effect of exposure matching:
    python build_tour.py --out tour_noexp --expose off ; python qa/flash_test.py tour_noexp tour
"""
import functools, http.server, os, socket, sys, threading
import cv2, numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def serve(folder):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=folder); h.log_message = lambda *a, **k: None
    threading.Thread(target=http.server.ThreadingHTTPServer(("127.0.0.1", port), h).serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/index.html?noglide"


def luminance(pg):
    png = pg.screenshot()
    g = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    return float(g[int(h * 0.30):int(h * 0.70), int(w * 0.30):int(w * 0.70)].mean())     # centre only: skips the minimap / reel / buttons


def measure(folder):
    from playwright.sync_api import sync_playwright
    url = serve(os.path.join(HERE, folder) if not os.path.isabs(folder) else folder)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--ignore-gpu-blocklist", "--enable-webgl", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 960, "height": 600})
        pg.goto(url); pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000)
        pg.evaluate("document.getElementById('help') && document.getElementById('help').classList.remove('show')")
        n = pg.evaluate("window.__tour.n"); L = []
        for i in range(n):
            pg.evaluate(f"window.__tour.go({i})"); pg.wait_for_function(f"window.__tour.state().cur === {i} && !window.__tour.state().busy", timeout=30000)
            pg.evaluate("window.__tour.aim(window.__tour.state().lon, -50)")      # look steeply down: mostly floor, which should match between stops
            pg.wait_for_timeout(500); L.append(luminance(pg))
        b.close()
    L = np.array(L); d = np.abs(np.diff(L)) / L.mean() * 100
    return L, d


if __name__ == "__main__":
    res = {}
    for f in sys.argv[1:]:
        L, d = measure(f); res[f] = d
        print(f"{f}: brightness step between consecutive stops (% of mean): mean {d.mean():.2f}  median {np.median(d):.2f}  max {d.max():.2f}  over {len(d)} hops", flush=True)
    if len(res) == 2:
        a, b = res.values(); print(f"better on {(b < a).sum()}/{len(a)} hops")
