"""Drive floor.html in Chrome: the exported floor model on WebGPU and WASM vs PyTorch.

Usage: python experiments/webgpu-depth/run_floor_spike.py --onnx path/to/onnx [--device both] [--profile DIR]
(first run python experiments/webgpu-depth/export_floor_onnx.py --out <folder> to make floor.onnx and the test files.)
Results go to results_floor_<device>.json (numbers only). Experiment only.
"""
import argparse, http.server, json, os, socket, threading
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))


def serve(onnx_dir):
    class H(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            path = path.split("?")[0]
            if path.startswith("/onnx/"):
                return os.path.join(onnx_dir, path[len("/onnx/"):].replace("/", os.sep))
            return os.path.join(HERE, path.lstrip("/").replace("/", os.sep))

        def log_message(self, *a):
            pass

        def handle_error(self, *a):
            pass
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    threading.Thread(target=http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--device", default="both", choices=["webgpu", "wasm", "both"])
    ap.add_argument("--profile", default=os.path.join(os.environ.get("TEMP", "."), "webgpu_floor_profile"))
    ap.add_argument("--channel", default="chrome")
    ap.add_argument("--file", default="floor.onnx", help="model file inside --onnx (e.g. floor_fp16.onnx)")
    a = ap.parse_args()
    base = serve(os.path.abspath(a.onnx))
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(a.profile, channel=a.channel, headless=False,
                                                    args=["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--ignore-gpu-blocklist"])
        page = ctx.new_page(); errs = []
        page.on("pageerror", lambda e: errs.append(str(e)[:200]))
        page.goto(base + "/floor.html"); page.wait_for_function("window.__ready === true", timeout=120000)
        ua = page.evaluate("navigator.userAgent")
        for dev in (["webgpu", "wasm"] if a.device == "both" else [a.device]):
            try:
                r = page.evaluate("([b, d, f]) => window.runFloor({base: b, device: d, file: f})", [base + "/onnx", dev, a.file])
            except Exception as e:
                r = {"device": dev, "error": str(e)[:400]}
            r["browser"] = ua; r["page_errors"] = errs[-3:]
            json.dump(r, open(os.path.join(HERE, (f"results_floor_{dev}.json" if a.file == "floor.onnx" else f"results_floor_{os.path.splitext(a.file)[0].replace('floor_', '')}_{dev}.json")), "w"), indent=1)
            print(dev, json.dumps({k: r.get(k) for k in ("timings", "same_input_as_pytorch", "browser_preprocessing", "error")})[:900], flush=True)
        ctx.close()


if __name__ == "__main__":
    main()
