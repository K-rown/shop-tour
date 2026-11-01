"""Phone tiles vs the whole 8K picture vs 4K only, on an emulated touch phone: picture memory, bytes downloaded, sharpness.

Usage: python qa/tile_check.py --tour D:/walks/example/walk1/tour [--stops 1,10,20] [--json qa/tile_check.json]
For every stop and every mode (?notiles = whole 8K like before, ?tiles = new, ?lowres = 4K only) it opens the stop in a 390x844 touch
viewport (device pixel ratio 3), waits until the picture has settled, and records the viewer's own picture-memory counter, the picture
bytes the page downloaded, the sharpness (variance of the Laplacian) of the screenshot, and the difference to the whole-8K screenshot.
"""
import argparse, functools, http.server, io, json, os, socket, sys, threading, time
import cv2
import numpy as np
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def serve(folder):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    class Quiet_(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass
    h = functools.partial(Quiet_, directory=folder)

    class Quiet(http.server.ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if not isinstance(sys.exc_info()[1], ConnectionError):
                super().handle_error(request, client_address)
    threading.Thread(target=Quiet(("127.0.0.1", port), h).serve_forever, daemon=True).start()
    return port


def sharp(png):
    g = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), 1), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tour", default=os.path.join(HERE, "tour"))
    ap.add_argument("--stops", default="1,10,20")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    port = serve(a.tour)
    out = {}
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--ignore-gpu-blocklist", "--enable-webgl", "--enable-unsafe-swiftshader"])
        for stop in [int(x) for x in a.stops.split(",")]:
            ref = None
            for mode in ("notiles", "tiles", "lowres"):
                ctx = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True, has_touch=True)
                pg = ctx.new_page(); nbytes = [0]

                def on_resp(r):
                    u = r.url.split("?")[0]
                    if any(k in u for k in ("/panos/", "/panos_lo/", "/tiles/")):
                        try:
                            nbytes[0] += len(r.body())
                        except Exception:
                            pass
                pg.on("response", on_resp)
                pg.goto(f"http://127.0.0.1:{port}/index.html?{mode}#{stop:03d}")
                pg.wait_for_function("window.__tour && window.__tour.state().cur >= 0 && !window.__tour.state().busy", timeout=60000)
                pg.evaluate("window.__tour.aim(window.__tour.state().lon, -5)")
                want = 3 if mode == "tiles" else None
                t0 = time.time()
                while time.time() - t0 < 20:                      # until the sharp picture is in
                    st = pg.evaluate("window.__tour.state()")
                    if mode == "notiles" and st["res"] == "8K" or mode == "tiles" and st["tiles"] >= 2 or mode == "lowres":
                        break
                    time.sleep(0.25)
                time.sleep(1.0)
                st = pg.evaluate("window.__tour.state()")
                v, g = sharp(pg.screenshot())
                if mode == "notiles":
                    ref = g
                dm = np.abs(g.astype(np.float32) - ref.astype(np.float32)) if ref is not None else None
                diff = float(dm.mean()) if dm is not None else None
                over = float((dm > 25).mean() * 100) if dm is not None else None       # a tile seam or a wrong tile shows up as a line of big differences
                out.setdefault(str(stop), {})[mode] = {"res": st["res"], "tileMode": st.get("tileMode"), "tiles_loaded": st.get("tiles"),
                                                       "picture_memory_MB": round(st["texMB"], 1), "picture_downloaded_MB": round(nbytes[0] / 1048576, 2),
                                                       "sharpness": round(v, 1), "mean_abs_diff_vs_whole_8K": None if diff is None else round(diff, 2),
                                                       "pixels_differing_by_over_25_pct": None if over is None else round(over, 3)}
                ctx.close()
            print(stop, json.dumps(out[str(stop)]), flush=True)
        b.close()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
