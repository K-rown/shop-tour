"""Is anyone still visible under the camera? Looks straight down from every stop of a folder of panoramas, runs the same
segmentation model the pipeline uses (UperNet ConvNeXt, ADE20K 'person' class) and reports how much of the view is still person (only regions reaching the middle of the view, where the person holding the stick is).

  python qa/check_nadir_person.py                       # the built tour (tour/panos_lo)
  python qa/check_nadir_person.py --folder frames/points   # the raw stitched panoramas, for comparison
Exit code 1 if any stop has more than --limit percent of the downward view labelled person.
Needs torch + transformers (the pipeline's own dependencies); skips with a message if they are missing.
"""
import argparse, glob, json, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from depth_panos import cut_view
from nadir import central_person

MODEL = "openmmlab/upernet-convnext-small"
NADIR_VIEW = {"yaw": 0.0, "pitch": -90.0, "fov": 130.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=os.path.join(HERE, "tour", "panos_lo"))
    ap.add_argument("--limit", type=float, default=1.0, help="max percent of the downward view that may be person")
    ap.add_argument("--json", default=None, help="write the per-stop numbers here")
    a = ap.parse_args()
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
    except Exception as e:
        print("skipped (needs torch + transformers):", e); return 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = UperNetForSemanticSegmentation.from_pretrained(MODEL).to(dev).eval()
    pid = [int(i) for i, n in model.config.id2label.items() if "person" in n.lower()]
    files = sorted(glob.glob(os.path.join(a.folder, "*.jpg")))
    out, bad = {}, []
    for f in files:
        pano = cv2.resize(cv2.imread(f), (4096, 2048), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(cut_view(pano, NADIR_VIEW, 640), cv2.COLOR_BGR2RGB)
        inp = proc(images=Image.fromarray(img), return_tensors="pt").to(dev)
        with torch.no_grad():
            lg = model(**inp).logits
        lg = torch.nn.functional.interpolate(lg, size=(640, 640), mode="bilinear", align_corners=False)[0]
        prob = torch.softmax(lg, 0)[pid].sum(0).cpu().numpy()
        pct = 100.0 * float((central_person((prob > 0.5).astype(np.uint8) * 255, touch=0.3) > 0).mean())   # only people under the camera; spools etc. are ignored
        out[os.path.basename(f)[:3]] = round(pct, 2)
        if pct > a.limit:
            bad.append(os.path.basename(f)[:3])
    print("person pixels in the straight-down view (% of view):")
    print("  " + "  ".join(f"{k}:{v:.1f}" for k, v in out.items()))
    print(f"stops over {a.limit}%: {len(bad)}/{len(out)}  {bad}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
