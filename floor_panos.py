"""Which pixels are floor, for every stop (semantic segmentation, UperNet ConvNeXt from OpenMMLab, MIT licence,
trained on ADE20K which has floor / rug / carpet classes).

Uses the same 4 down-tilted views as depth_panos.py. place_rings.py uses the result to keep rings on visible
floor (slide a ring onto open floor, or hide it when its spot is covered by a spool, a cart, a shelf).

Setup: same as depth_panos.py (pip install torch transformers pillow)
Run:   python floor_panos.py          (~3-5 min on CPU, faster with a GPU)
Writes frames/depth/<id>_f<k>.png    (255 = floor)
       frames/depth/<id>_n.png       (255 = person, straight-down view, used by nadir.py to paint the person out)
       frames/depth/<id>_nf.png      (255 = floor, same straight-down view)
"""
import argparse, json, os
import cv2
import numpy as np
from depth_panos import VIEWS, SIZE, cut_view
from imutil import imread

MODEL = "openmmlab/upernet-convnext-small"
FLOORISH = ("floor", "rug", "carpet", "ground", "path")
# a fifth view looks straight down: it is where the person holding the stick and the floor under the camera are
NADIR_VIEW = {"yaw": 0.0, "pitch": -90.0, "fov": 130.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")      # harmless notices that only confuse a first run
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    try:
        proc = AutoImageProcessor.from_pretrained(MODEL)
        model = UperNetForSemanticSegmentation.from_pretrained(MODEL).to(dev).eval()
    except OSError as e:                                    # first run without internet: say what is needed instead of a long traceback
        raise SystemExit(f"cannot load the model {MODEL}: it is downloaded from huggingface.co the first time (a few hundred MB), so connect to "
                         f"the internet once, or copy an existing Hugging Face cache (the HF_HOME folder) to this machine.\n({str(e)[:120]})")
    floor_ids = [int(i) for i, name in model.config.id2label.items() if any(f in name.lower() for f in FLOORISH)]
    print("floor classes:", [model.config.id2label[i] for i in floor_ids])
    person_ids = [int(i) for i, name in model.config.id2label.items() if "person" in name.lower()]
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    out = os.path.join(a.frames, "depth"); os.makedirs(out, exist_ok=True)
    def load_views(p):                                          # JPEG decode + view cutting on CPU threads, overlapped with the GPU
        pano = imread(os.path.join(a.frames, "points", p["filename"]))
        pano = cv2.resize(pano, (4096, 2048), interpolation=cv2.INTER_AREA)
        return [cv2.cvtColor(cut_view(pano, v), cv2.COLOR_BGR2RGB) for v in VIEWS + [NADIR_VIEW]]

    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(max_workers=4)
    for p, views in zip(pts, ex.map(load_views, pts)):
        for k, img in enumerate(views):
            inputs = proc(images=Image.fromarray(img), return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = model(**inputs).logits                      # (1, classes, h, w)
            logits = torch.nn.functional.interpolate(logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]
            lab = logits.argmax(0).cpu().numpy()
            m = np.isin(lab, floor_ids).astype(np.uint8) * 255
            if k == len(VIEWS):                                      # the straight-down view
                prob = torch.softmax(logits, 0)[person_ids].sum(0).cpu().numpy()     # confidence, not argmax: red/yellow gear next to the camera
                cv2.imwrite(os.path.join(out, f"{p['id']}_n.png"), (prob > 0.6).astype(np.uint8) * 255)   # wins the argmax as "person" too often
                cv2.imwrite(os.path.join(out, f"{p['id']}_nf.png"), m)
            else:
                cv2.imwrite(os.path.join(out, f"{p['id']}_f{k}.png"), m)
        print("floor", p["id"], flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
