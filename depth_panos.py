"""Depth for every stop, with Depth Anything V2 Small (Apache-2.0, safe for commercial use).

Each pano is cut into 4 overlapping views looking slightly down (we only need the floor and what
stands on it), the model predicts relative depth for each view, and the raw results are saved.
place_rings.py then ties every view to the real 3D points from map_stops.py, so the depths are
in map units and agree with each other, finds the floor, and hides rings that are behind things.

Setup (once):   pip install torch transformers pillow
Run:            python depth_panos.py
Writes:         frames/depth/<id>_v<k>.npy  (float16 relative inverse depth)  +  frames/depth/views.json
"""
import argparse, json, os
import cv2
import numpy as np
from imutil import imread

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
VIEWS = [{"yaw": y, "pitch": -25.0, "fov": 100.0} for y in (0.0, 90.0, 180.0, -90.0)]
SIZE = 518


def view_rays(yaw, pitch, fov, n):
    """Unit rays (x right, y up, z forward = pano centre) for an n x n pinhole view."""
    f = n / 2 / np.tan(np.radians(fov) / 2)
    x, y = np.meshgrid(np.arange(n) - n / 2 + 0.5, n / 2 - np.arange(n) - 0.5)
    d = np.stack([x, y, np.full_like(x, f)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    p, yv = np.radians(pitch), np.radians(yaw)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(yv), 0, np.sin(yv)], [0, 1, 0], [-np.sin(yv), 0, np.cos(yv)]])
    return d @ Rx.T @ Ry.T


def cut_view(pano, v, n=SIZE):
    h, w = pano.shape[:2]
    d = view_rays(v["yaw"], v["pitch"], v["fov"], n)
    az = np.arctan2(d[..., 0], d[..., 2]); el = np.arcsin(np.clip(d[..., 1], -1, 1))
    u = ((az + np.pi) / (2 * np.pi) * w - 0.5).astype(np.float32)
    vv = ((np.pi / 2 - el) / np.pi * h - 0.5).astype(np.float32)
    return cv2.remap(pano, u, vv, cv2.INTER_AREA if w > 4 * n else cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")      # harmless notices that only confuse a first run
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    import torch                                   # heavy imports only when actually running the model
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    try:
        proc = AutoImageProcessor.from_pretrained(MODEL)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(dev).eval()
    except OSError as e:                                    # first run without internet: say what is needed instead of a long traceback
        raise SystemExit(f"cannot load the model {MODEL}: it is downloaded from huggingface.co the first time (a few hundred MB), so connect to "
                         f"the internet once, or copy an existing Hugging Face cache (the HF_HOME folder) to this machine.\n({str(e)[:120]})")
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    out = os.path.join(a.frames, "depth"); os.makedirs(out, exist_ok=True)
    def load_views(p):                                          # JPEG decode + view cutting on CPU threads, overlapped with the GPU
        pano = imread(os.path.join(a.frames, "points", p["filename"]))
        pano = cv2.resize(pano, (4096, 2048), interpolation=cv2.INTER_AREA)
        return [cv2.cvtColor(cut_view(pano, v), cv2.COLOR_BGR2RGB) for v in VIEWS]

    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(max_workers=4)
    for p, views in zip(pts, ex.map(load_views, pts)):
        for k, img in enumerate(views):
            inputs = proc(images=Image.fromarray(img), return_tensors="pt").to(dev)
            with torch.no_grad():
                d = model(**inputs).predicted_depth            # relative inverse depth (bigger = nearer)
            d = torch.nn.functional.interpolate(d.unsqueeze(1), size=(SIZE, SIZE), mode="bicubic", align_corners=False)[0, 0]
            np.save(os.path.join(out, f"{p['id']}_v{k}.npy"), d.cpu().numpy().astype(np.float16))
            if k == 0:
                cv2.imwrite(os.path.join(out, f"{p['id']}_v0.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("depth", p["id"], flush=True)
    json.dump({"size": SIZE, "views": VIEWS, "model": MODEL}, open(os.path.join(out, "views.json"), "w"), indent=1)
    print("done ->", out)


if __name__ == "__main__":
    main()
