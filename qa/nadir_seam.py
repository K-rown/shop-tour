"""A/B for the patch under the camera: how visible is its outline? (nadir.FILL_MODE blur vs harmonic)

Usage: python qa/nadir_seam.py --walk D:/walks/example/walk1 [--montage D:/somewhere/montage.png --stops 3,10,15,20] [--json qa/nadir_seam.json]
Uses the walk's frames (stitched panoramas + the per-stop person/floor masks). Prints, per stop, the seam score (mean LAB colour
difference between a thin band inside the patch and one outside it; lower is better), and how much finer detail the patch holds than the floor
around it (must stay at the floor's own grain: the fill must never invent detail). The montage shows straight-down views: before | blur | harmonic.
"""
import argparse, json, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import nadir                                    # noqa: E402
from depth_panos import cut_view                # noqa: E402


def down_view(pano, n=480):
    return cut_view(pano, {"yaw": 0.0, "pitch": -90.0, "fov": 130.0}, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", required=True)
    ap.add_argument("--montage", default=None)
    ap.add_argument("--stops", default="3,10,15,20")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    fr = os.path.join(a.walk, "frames")
    pts = json.load(open(os.path.join(fr, "points.json")))
    rd = lambda pid, suf: cv2.imread(os.path.join(fr, "depth", f"{pid}_{suf}.png"), cv2.IMREAD_GRAYSCALE)
    res = {}; rows = {}
    for p in pts:
        img = cv2.imread(os.path.join(fr, "points", p["filename"]))
        person, floor = rd(p["id"], "n"), rd(p["id"], "nf")
        r = {}
        outs = {}
        for mode in ("blur", "harmonic"):
            nadir.FILL_MODE = mode
            out = nadir.fill_nadir(img, person=person, floor=floor)
            outs[mode] = out
            band = out[int(img.shape[0] * 0.78):]
            r[mode] = {"seam": None if nadir.fill_nadir.last_seam is None else round(nadir.fill_nadir.last_seam, 2)}
        d = np.abs(outs["harmonic"].astype(np.float32) - outs["blur"].astype(np.float32))
        r["pixels_changed_by_harmonic_pct"] = round(float((d.max(2) > 1).mean() * 100), 2)
        r["max_change"] = int(d.max())
        # detail check: finer-than-4px energy inside the changed area must equal that of the blur version (the correction is low-frequency)
        hf = lambda x: x.astype(np.float32) - cv2.GaussianBlur(x, (0, 0), 4).astype(np.float32)
        ch = d.max(2) > 1
        r["detail_added"] = round(float(np.abs(hf(outs["harmonic"]) - hf(outs["blur"]))[ch].mean()) if ch.any() else 0.0, 3)
        res[p["id"]] = r
        if a.montage and int(p["id"]) in [int(x) for x in a.stops.split(",")]:
            rows[p["id"]] = np.hstack([down_view(img), down_view(outs["blur"]), down_view(outs["harmonic"])])
        print(p["id"], json.dumps(r), flush=True)
    sb = [v["blur"]["seam"] for v in res.values() if v["blur"]["seam"] is not None]
    sh = [v["harmonic"]["seam"] for v in res.values() if v["harmonic"]["seam"] is not None]
    summ = {"stops_with_a_patch": len(sb), "seam_mean_blur": round(float(np.mean(sb)), 2), "seam_mean_harmonic": round(float(np.mean(sh)), 2),
            "seam_median_blur": round(float(np.median(sb)), 2), "seam_median_harmonic": round(float(np.median(sh)), 2),
            "stops_improved": int(sum(h < b for b, h in zip(sb, sh))), "stops_worse": int(sum(h > b + 0.05 for b, h in zip(sb, sh)))}
    print(json.dumps(summ))
    if a.montage and rows:
        cv2.imwrite(a.montage, np.vstack([rows[k] for k in sorted(rows)]))
    if a.json:
        json.dump({"summary": summ, "stops": res}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
