"""4b: is the capturer really gone from every stop?  python qa/check_capturer.py [--frames frames] [--json qa/check_capturer.json] [--control]

The rule as asked for ("no pixel within 60 degrees of straight down may sit more than 15 cm above the floor plane") cannot be measured with the depth
model that is available: on floor that is known to be floor (vouched for by two neighbouring stops) it flags 17-48% of the pixels as higher than 15 cm
(measured on 6 stops: 19-46% of known floor pixels flagged, blobs up to 1.4 m2), and a shop is full of things that really are higher than 15 cm. What is checked instead, on the cleaned picture
of every stop (frames/clean/<id>.jpg), in the straight-down view:
  C  nobody left: the person search of the mask step (compact, isolated, person sized, standing out by depth) finds no person-sized shape in the
     filled area;  privacy_gate.py adds a person detector and a face detector over the whole picture,
  D  no grey and no noise in the soft fill (not one flat colour that is not the floor's, not grainier than the floor round it).
Recorded for every stop but NOT used to pass or fail (too noisy): A leftover = floor the neighbours agree on that still looks different next to the
filled area; B the median height of the filled area above the floor the neighbours vouch for, and the share of it above 15 cm.
Prints  N/N stops pass 4b  and exits 1 if any stop fails (privacy_gate.py holds a failing stop back). --control runs the checks on the ORIGINAL
pictures: it must fail there (that shows the check can see a person). Needs the models of remove_capturer.py (depth, SAM 2).
"""
import argparse, json, os, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import capturer as C
from imutil import imread

MIN_LEFTOVER_M2 = 0.03


def check_stop(models, sid, pano_clean, up, cam_h, hole1024, soft1024=None, floor1024=None, leftover_m2=None):
    """The checks for one stop; returns (result dict with 'problems' and 'info', level view, disparity, flags). Only C and D can fail a stop (see the
    module docstring); A (leftover_m2) and B (median height, share above 15 cm) are recorded in 'info' because they are too noisy to decide anything."""
    v = C.level_view(pano_clean, up)
    disp = cv2.medianBlur(models.disparity(v).astype(np.float32), 5)
    d, f = C.view_dirs(); mpp = cam_h / f
    h8 = hole1024.astype(np.uint8)
    inner = cv2.erode(h8, np.ones((9, 9), np.uint8)) > 0
    zone = cv2.dilate(h8, np.ones((7, 7), np.uint8)) > 0                      # the filled area and 3 cm round it
    if not h8.any():                                                           # nothing was painted out: the person search then looks at the whole cone round the camera
        ang = np.degrees(np.arccos(np.clip(-d[..., 1], -1, 1))); zone = ang < C.CORE_DEG
    res = dict(problems=[], info={})
    if leftover_m2 is not None:
        res["info"]["leftover_m2"] = leftover_m2                                   # A: informational (see the docstring)
    ref = None if floor1024 is None else (floor1024 & ~inner)
    flag = np.zeros_like(inner)
    if ref is not None and ref.sum() >= 800:
        Df = float(np.median(disp[ref]))
        height = cam_h * (1 - Df / np.clip(disp, 1e-3, None))
        flag = height > C.HEIGHT_M
        if inner.any():
            res["info"].update(median_height_cm=round(100 * float(np.median(height[inner])), 1), flagged_pct=round(100.0 * float(flag[inner].mean()), 1))
    sam, sc, passed = C.score_candidates(models.segment_many(v, C.prior_points(cam_h)), disp, cam_h, mpp)
    if sam is not None and (sam & zone).sum() > 0.2 * sam.sum():
        res["problems"].append(f"a person-sized shape ({sam.sum() * mpp * mpp:.2f} m2) is still near the camera")
    if soft1024 is not None and soft1024.sum() > 400:                           # D
        g = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ring = (cv2.dilate(h8, np.ones((61, 61), np.uint8)) > 0) & ~zone
        lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F))
        soft = soft1024 & inner
        if soft.sum() > 200 and ring.sum() > 500:
            sd = float(g[soft].std()); lv_s, lv_r = float(lap[soft].mean()), float(lap[ring].mean())
            dc = float(np.abs(v[soft].reshape(-1, 3).mean(0) - v[ring].reshape(-1, 3).mean(0)).max())
            res["soft"] = dict(px=int(soft.sum()), std=round(sd, 1), grain_ratio=round(lv_s / max(lv_r, 1e-3), 2), colour_gap=round(dc, 1))
            if lv_s > 1.5 * lv_r + 0.5:
                res["problems"].append("noise: the soft fill is grainier than the floor round it")
            if sd < 1.0 and dc > 25:
                res["problems"].append("a flat patch of a colour that is not the floor's")
    return res, v, disp, flag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames"); ap.add_argument("--json", default=None); ap.add_argument("--only", nargs="*")
    ap.add_argument("--control", action="store_true", help="negative control: run the same checks on the ORIGINAL pictures (frames/points), where every stop must FAIL")
    a = ap.parse_args()
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    P = json.load(open(os.path.join(a.frames, "poses.json"))); UP = json.load(open(os.path.join(a.frames, "up.json")))
    rep = json.load(open(os.path.join(a.frames, "capturer", "report.json")))
    models = C.Models()
    out, ok = {}, 0
    todo = [k for k, p in enumerate(pts) if not a.only or p["id"] in a.only]
    for k in todo:
        sid = pts[k]["id"]
        cp = os.path.join(a.frames, "points" if a.control else "clean", pts[k]["filename"]); mp = os.path.join(a.frames, "capturer", sid + ("_hole.png" if os.path.exists(os.path.join(a.frames, "capturer", sid + "_hole.png")) else "_mask.png"))
        if not os.path.exists(cp) or not os.path.exists(mp):
            out[sid] = dict(problems=["no cleaned picture: the stop is held back"]); print(f"FAILED {sid}: no cleaned picture (held back)"); continue
        hole = cv2.imread(mp, 0) > 0
        sp = os.path.join(a.frames, "capturer", sid + "_soft.png")
        fp = os.path.join(a.frames, "capturer", sid + "_floor.png")
        res, *_ = check_stop(models, sid, imread(cp), UP[k], P["nodes"][k]["cam_h"] or 2.0, hole, (cv2.imread(sp, 0) > 0) if os.path.exists(sp) else None,
                             (cv2.imread(fp, 0) > 127) if os.path.exists(fp) else None, rep.get(sid, {}).get("leftover_m2"))
        out[sid] = res
        if res["problems"]:
            print(f"FAILED {sid}: " + "; ".join(res["problems"]))
        else:
            ok += 1; print(f"PASS {sid}  (info: leftover {res['info'].get('leftover_m2')} m2, median height of the filled area {res['info'].get('median_height_cm')} cm)")
    print(f"\n{ok}/{len(todo)} stops pass 4b")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    sys.exit((0 if ok == 0 else 1) if a.control else (0 if ok == len(todo) else 1))


if __name__ == "__main__":
    main()
