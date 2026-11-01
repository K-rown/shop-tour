"""How long until a visitor sees the first picture? Emulated touch phone, throttled network and CPU (Chromium only).

Usage: python qa/load_time.py --tour D:/walks/example/walk1/tour [--runs 3] [--json qa/load_time.json]
For each network profile it opens the tour with an empty cache `runs` times and records, from the moment of navigation:
  first_view_ms  until the viewer is ready and showing a picture (window.__tour exists, not busy)
  sharp_ms       until the 4K picture is the one on screen (res label '4K'/'8K tiles'/'8K')
  bytes          everything downloaded until the first view
By default the test server does not compress; with --gzip it sends html/js/json gzipped like a real host (three.js 600 -> about 150 KB).
"""
import argparse, functools, http.server, json, os, socket, statistics, sys, threading, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GZIP = [False]
PROFILES = {                                           # name: (down Mbit/s, up Mbit/s, round trip ms)
    "local": None,
    "4G": (4.0, 1.5, 60),
    "slow 4G": (1.6, 0.75, 150),
}


def serve(folder):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()

    class Quiet_(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass

        def do_GET(self):                                  # --gzip: send text files compressed like a real host does
            if not GZIP[0] or not self.path.split("?")[0].endswith((".html", ".js", ".json")):
                return super().do_GET()
            import gzip
            fs = self.translate_path(self.path)
            try:
                raw = open(fs, "rb").read()
            except OSError:
                return super().do_GET()
            body = gzip.compress(raw, 6)
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(fs)); self.send_header("Content-Encoding", "gzip"); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
    h = functools.partial(Quiet_, directory=folder)

    class Srv(http.server.ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if not isinstance(sys.exc_info()[1], ConnectionError):
                super().handle_error(request, client_address)
    threading.Thread(target=Srv(("127.0.0.1", port), h).serve_forever, daemon=True).start()
    return port


def measure(b, port, prof, cpu=4):
    ctx = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True, has_touch=True)
    pg = ctx.new_page(); cdp = ctx.new_cdp_session(pg)
    cdp.send("Network.enable"); cdp.send("Network.setCacheDisabled", {"cacheDisabled": True})
    cdp.send("Emulation.setCPUThrottlingRate", {"rate": cpu})
    if prof:
        down, up, rtt = prof
        cdp.send("Network.emulateNetworkConditions", {"offline": False, "latency": rtt, "downloadThroughput": down * 125000, "uploadThroughput": up * 125000})
    total = [0]
    cdp.on("Network.loadingFinished", lambda e: total.__setitem__(0, total[0] + e.get("encodedDataLength", 0)))
    t0 = time.time()
    pg.goto(f"http://127.0.0.1:{port}/index.html", wait_until="commit")
    pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=180000, polling=50)
    first = time.time() - t0; bytes_first = total[0]
    pg.wait_for_function("window.__tour.state().res !== undefined && /^(4K|8K)/.test(window.__tour.state().res)", timeout=180000, polling=50)
    sharp = time.time() - t0
    ctx.close()
    return {"first_view_ms": round(first * 1000), "sharp_ms": round(sharp * 1000), "bytes_MB": round(bytes_first / 1048576, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tour", default=os.path.join(HERE, "tour"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--json", default=None)
    ap.add_argument("--gzip", action="store_true", help="compress html/js/json like a real host")
    a = ap.parse_args()
    GZIP[0] = a.gzip
    port = serve(a.tour)
    out = {}
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--ignore-gpu-blocklist", "--enable-webgl", "--enable-unsafe-swiftshader"])
        for name, prof in PROFILES.items():
            runs = [measure(b, port, prof) for _ in range(a.runs)]
            out[name] = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
            out[name]["runs"] = runs
            print(f"{name:8s} first view {out[name]['first_view_ms']:6d} ms   4K on screen {out[name]['sharp_ms']:6d} ms   {out[name]['bytes_MB']} MB to first view", flush=True)
        b.close()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
