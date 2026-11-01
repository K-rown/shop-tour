"""Drive index.html in Chrome and compare its depth maps with the Python ones (frames/depth/<id>_v<k>.npy) for a few stops.

Usage: python experiments/webgpu-depth/run_spike.py --walk D:/walks/place/2026-01-01 --stops 001,010,020 [--device webgpu|wasm|both] [--profile DIR]
Experiment only: nothing here is used by the pipeline. Results go to results_<device>.json (numbers only, no pictures).
"""
import argparse, http.server, json, os, socket, threading, time
import numpy as np
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))


def serve(page_dir, walk):
    class H(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            path = path.split("?")[0]
            if path.startswith("/walk/"):
                return os.path.join(walk, path[len("/walk/"):].replace("/", os.sep))
            return os.path.join(page_dir, path.lstrip("/").replace("/", os.sep))

        def log_message(self, *a):
            pass

        def handle_error(self, *a):
            pass
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}"


def pearson(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", required=True)
    ap.add_argument("--stops", default="001,010,020")
    ap.add_argument("--device", default="both", choices=["webgpu", "wasm", "both"])
    ap.add_argument("--profile", default=os.path.join(os.environ.get("TEMP", "."), "webgpu_spike_profile"))
    ap.add_argument("--channel", default="chrome")
    ap.add_argument("--dtype", default="fp32", help="model precision: fp32, fp16 (transformers.js picks onnx/model_fp16.onnx)")
    a = ap.parse_args()
    walk = os.path.abspath(a.walk)
    base = serve(HERE, walk)
    pts = {p["id"]: p["filename"] for p in json.load(open(os.path.join(walk, "frames", "points.json")))}
    devices = ["webgpu", "wasm"] if a.device == "both" else [a.device]
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(a.profile, channel=a.channel, headless=False,
                                                    args=["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--ignore-gpu-blocklist"])
        page = ctx.new_page(); errs = []
        page.on("pageerror", lambda e: errs.append(str(e)[:200]))
        page.on("console", lambda m: errs.append(m.text[:200]) if m.type == "error" else None)
        page.goto(base + "/index.html"); page.wait_for_function("window.__ready === true", timeout=120000)
        info = page.evaluate("({ua: navigator.userAgent, webgpu: !!navigator.gpu})")
        print(info)
        for dev in devices:
            out = {"device": dev, "browser": info["ua"], "stops": {}}
            for sid in a.stops.split(","):
                t0 = time.time()
                r = page.evaluate("([u, d, t]) => window.runDepth(u, d, t)", [f"{base}/walk/frames/points/{pts[sid]}", dev, a.dtype])
                out.setdefault("adapter", r["adapter"]); out["webgpu_available"] = r["webgpu"]
                cors = []; ctrl = []
                for k in range(4):
                    ref = np.load(os.path.join(walk, "frames", "depth", f"{sid}_v{k}.npy")).astype(np.float32)
                    got = np.array(r["maps"][k], dtype=np.float32).reshape(-1)
                    if got.size != ref.size:
                        raise SystemExit(f"size mismatch: browser {got.size} vs python {ref.size} (dims {r['dims']})")
                    cors.append(pearson(ref, got))
                    other = np.load(os.path.join(walk, "frames", "depth", f"{sid}_v{(k + 1) % 4}.npy")).astype(np.float32)
                    ctrl.append(pearson(other, got))                  # negative control: the next view of the same stop must correlate much less
                out["stops"][sid] = {"correlation_per_view": [round(c, 5) for c in cors], "correlation_mean": round(float(np.mean(cors)), 5), "control_other_view_mean": round(float(np.mean(ctrl)), 4),
                                     "timings_ms": {k: (round(v) if not isinstance(v, list) else [round(x) for x in v]) for k, v in r["timings"].items()},
                                     "wall_s": round(time.time() - t0, 1)}
                print(dev, sid, out["stops"][sid]["correlation_mean"], out["stops"][sid]["timings_ms"].get("warm_view_ms"), flush=True)
            out["page_errors"] = errs[-5:]
            json.dump(out, open(os.path.join(HERE, (f"results_{dev}.json" if a.dtype == "fp32" else f"results_{a.dtype}_{dev}.json")), "w"), indent=1)
        ctx.close()


if __name__ == "__main__":
    main()
