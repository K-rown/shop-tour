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
# A shape that is only the soft fill (the grey, blurred patch painted where the capturer stood) is not a person: the person search finds the patch itself, which has
# his outline. It is the fill when most of it is smooth AND nothing is left of it outside the smooth part. Measured: the fill's own texture is 0.25-0.6 (mean |Laplacian|
# of the picture blurred 1 px, over a 15 px box), while the same place in the ORIGINAL picture (him standing there) is never under 0.6 over more than a fifth of the shape,
# and what is left of the shape outside the smooth part is 15-60% of it in the originals (and in the cleaned pictures that still show him: 15-27%) but 0-5% in the cleaned pictures that show only the patch.
FILL_SMOOTH = 0.6
FILL_SHARE = 0.35                  # at least this share of the shape is smooth fill ...
FILL_LEFT = 0.08                   # ... and what remains of it, away from the fill, is less than this share of it (a share, not metres: the margin repeats the check at other camera heights)
FILL_GROW_PX, FILL_OPEN_PX = 25, 9


def fill_only(v, sam, soft, mpp):
    """(is the shape only the soft fill?, share of the shape that is smooth fill, m2 of the shape left outside the smooth fill). v: the straight-down view (bgr),
    sam: the shape (bool), soft: the capturer's record of where it painted the soft fill (bool)."""
    n = int(np.count_nonzero(sam))
    if n == 0 or soft is None or not soft.any():
        return False, 0.0, 0.0
    g = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY).astype(np.float32)
    le = cv2.blur(np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F)), (15, 15))
    fill = soft & (le < FILL_SMOOTH)
    share = float(np.count_nonzero(fill & sam)) / n
    rest = sam & ~(cv2.dilate(fill.astype(np.uint8), np.ones((FILL_GROW_PX, FILL_GROW_PX), np.uint8)) > 0)
    rest = cv2.morphologyEx(rest.astype(np.uint8), cv2.MORPH_OPEN, np.ones((FILL_OPEN_PX, FILL_OPEN_PX), np.uint8))
    cnt, _, st, _ = cv2.connectedComponentsWithStats(rest)
    left = max((int(st[i, cv2.CC_STAT_AREA]) for i in range(1, cnt)), default=0) * mpp * mpp
    return (share >= FILL_SHARE and left < FILL_LEFT * n * mpp * mpp), share, left


def check_stop(models, sid, pano_clean, up, cam_h, hole1024, soft1024=None, floor1024=None, leftover_m2=None, neighbours=None):
    """The checks for one stop; returns (result dict with 'problems' and 'info', level view, disparity, flags). Only C and D can fail a stop (see the
    module docstring); A (leftover_m2) and B (median height, share above 15 cm) are recorded in 'info' because they are too noisy to decide anything.
    neighbours: a neighbour_check.Context (or None): a person-sized shape that two neighbouring stops see at the same place in THEIR original pictures is an object, not him,
    and is not a problem (recorded in info['neighbour_object']); see neighbour_check.py. It only ever removes the check-C problem."""
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
    if sam is not None:
        only, share, left = fill_only(v, sam, soft1024, mpp)
        if only:
            res["info"]["fill_only"] = dict(m2=round(float(sam.sum() * mpp * mpp), 2), smooth_share=round(share, 2), left_m2=round(left, 3))
            sam = None                                                           # the shape is the soft fill itself, not a person
    if sam is not None and (sam & zone).sum() > 0.2 * sam.sum():
        seen = neighbours.judge(sid, sam, pano_clean) if neighbours is not None else None
        if seen is not None and seen["object"]:
            res["info"]["neighbour_object"] = dict(m2=round(float(sam.sum() * mpp * mpp), 2), **seen)       # the neighbours see it too: an object
        else:
            if seen is not None and seen["neighbours"]:
                res["info"]["neighbour_check"] = dict(m2=round(float(sam.sum() * mpp * mpp), 2), **seen)       # looked at, not agreed: the hold stands
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
    import neighbour_check as NC
    nbr = NC.get_context(a.frames)
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
                             (cv2.imread(fp, 0) > 127) if os.path.exists(fp) else None, rep.get(sid, {}).get("leftover_m2"), nbr)
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
