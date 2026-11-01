"""Remove the person holding the stick from every stop, and rebuild the floor under them from the neighbouring stops.

Usage:  python remove_capturer.py [--frames frames] [--only 010 017] [--masks-only] [--sheet qa/person_removal.jpg]
Reads:  frames/points/<id>.jpg, frames/up.json, frames/poses.json
Writes: frames/capturer/<id>_mask.png (his mask in the straight-down view), frames/capturer/report.json, frames/clean/<id>.jpg (the pano without him)
        and the before/after contact sheet. build_tour.py uses frames/clean/ when it exists.
How it works: see capturer.py. Models (both Apache-2.0): Depth Anything V2 Small, SAM 2.1 small. No person detector is involved.
"""
import argparse, json, os, sys, time
import cv2
import numpy as np
import capturer as C
from imutil import imread, imwrite

HERE = os.path.dirname(os.path.abspath(__file__))


def neighbours(P, i, max_n=8, max_d=9.0):
    """Stops that see the floor under stop i: the previous and next in the walk first, then the nearest by distance."""
    links = {l["to"]: l for l in P["nodes"][i]["links"]}
    order = [j for j in (i - 1, i + 1) if j in links]
    order += [j for j in sorted(links, key=lambda j: links[j]["d"]) if j not in order and links[j]["d"] <= max_d]
    return [dict(j=j, az=links[j]["az"], d=links[j]["d"], turn=links[j]["turn"]) for j in order[:max_n]]


def final_hole(view, mask, height, cam_h, mpp):
    """Mask plus his shadow, dilated a little (about 4 cm)."""
    m = C.add_shadow(view, mask, height, cam_h, mpp)
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0


def make_sheet(frames, pts, UP, path, cell=384, per_row=3):
    """Before | after of the straight-down view of every stop that has a cleaned picture."""
    cells = []
    for k, p in enumerate(pts):
        cp = os.path.join(frames, "clean", p["filename"])
        if not os.path.exists(cp):
            continue
        b = C.level_view(imread(os.path.join(frames, "points", p["filename"])), UP[k], cell)
        af = C.level_view(imread(cp), UP[k], cell)
        pair = np.hstack([b, af]); cv2.putText(pair, p["id"], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cells.append(pair)
    while len(cells) % per_row:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[r:r + per_row]) for r in range(0, len(cells), per_row)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("contact sheet:", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--only", nargs="*", help="stop ids, e.g. 010 017")
    ap.add_argument("--masks-only", action="store_true")
    ap.add_argument("--reuse-masks", action="store_true", help="do not redo the masks that already exist (only the floor fill)")
    ap.add_argument("--sheet", default=None, help="write the before/after contact sheet here")
    ap.add_argument("--sheet-only", action="store_true", help="only (re)make the contact sheet from the pictures that exist")
    a = ap.parse_args()
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    P = json.load(open(os.path.join(a.frames, "poses.json"))); UP = json.load(open(os.path.join(a.frames, "up.json")))
    ids = [p["id"] for p in pts]
    todo = [k for k, i in enumerate(ids) if not a.only or i in a.only]
    cdir = os.path.join(a.frames, "capturer"); os.makedirs(cdir, exist_ok=True)
    os.makedirs(os.path.join(a.frames, "clean"), exist_ok=True)
    rep_path = os.path.join(cdir, "report.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    if a.sheet_only:
        make_sheet(a.frames, pts, UP, a.sheet); return
    models = C.Models()
    cache = {}
    det = []

    def person_hint(view, pano, up):
        """Where the detector sees a person within 50 degrees of straight down (centre of its box, in the straight-down view's pixels): the best one, or None.
        It looks at the straight-down view at 130 and at 100 degrees (a person far below the lens shows up better in the narrower one)."""
        import privacy_gate as pg, math
        if not det:
            det.append(pg.Detectors())
        d, f = C.view_dirs()
        f130 = C.VIEW_N / 2 / math.tan(math.radians(C.VIEW_FOV) / 2); f100 = C.VIEW_N / 2 / math.tan(math.radians(50.0))
        found = [(sc, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) for sc, b in det[0].persons(view)]
        v100 = C.level_view(pano, up, C.VIEW_N, 100.0)
        for sc, b in det[0].persons(v100):
            c = C.VIEW_N / 2
            found.append((sc, (c + ((b[0] + b[2]) / 2 - c) * f130 / f100, c + ((b[1] + b[3]) / 2 - c) * f130 / f100)))
        best = None
        for sc, (cx, cy) in found:
            cx, cy = int(cx), int(cy)
            if 0 <= cx < C.VIEW_N and 0 <= cy < C.VIEW_N and np.degrees(np.arccos(np.clip(-d[cy, cx, 1], -1, 1))) < 50 and (best is None or sc > best[0]):
                best = (sc, (cx, cy))
        return None if best is None else best[1]

    def pano_of(k):
        return imread(os.path.join(a.frames, "points", pts[k]["filename"]))

    # 1) masks for the stops (every stop first: a neighbour's mask is needed to know where its own person stands)
    for k in range(len(ids)):
        mp = os.path.join(cdir, ids[k] + "_mask.png")
        if (k in todo and not (a.reuse_masks and os.path.exists(mp))) or not os.path.exists(mp):
            t = time.time()
            pano = pano_of(k); h = P["nodes"][k]["cam_h"] or 2.0
            m, dbg = C.find_mask(models, pano, UP[k], h, hint_fn=person_hint)
            if m is None:
                rep[ids[k]] = dict(status="NO CAPTURER FOUND", passed=dbg["passed"])
                cv2.imwrite(mp, np.zeros((C.VIEW_N, C.VIEW_N), np.uint8)); print(ids[k], "NO CAPTURER FOUND (held for review)", flush=True)
                continue
            hole = final_hole(dbg["view"], m, dbg["height"], h, dbg["mpp"])
            cv2.imwrite(mp, hole.astype(np.uint8) * 255)
            rep.setdefault(ids[k], {}).update(status="masked", fallback=dbg.get("fallback"), mask_m2=round(float(hole.sum()) * dbg["mpp"] ** 2, 2), mask_px=int(hole.sum()), score=round(float(dbg["score"]), 2))
            print(ids[k], f"mask {rep[ids[k]]['mask_m2']} m2  ({time.time() - t:.1f}s)", flush=True)
            del pano
    json.dump(rep, open(rep_path, "w"), indent=1)
    if a.masks_only:
        return
    # 2) the floor
    masks = {k: cv2.imread(os.path.join(cdir, ids[k] + "_mask.png"), 0) > 0 for k in range(len(ids))}

    def load(j):
        if j not in cache:
            if len(cache) >= 6:
                cache.pop(next(iter(cache)))
            nfp_j = os.path.join(a.frames, "depth", ids[j] + "_nf.png")
            cache[j] = (pano_of(j), UP[j], P["nodes"][j]["cam_h"] or 2.0, masks[j] if masks[j].any() else None, cv2.imread(nfp_j, 0) if os.path.exists(nfp_j) else None)
        return cache[j]

    for k in todo:
        if not masks[k].any():                                  # nobody under the camera was found (a walk recorded with the stick held clear of the lens, or a miss):
            pano = pano_of(k)                                   # there is nothing to paint out, so the picture goes on as it is. privacy_gate.py still runs a person and a
            imwrite(os.path.join(a.frames, "clean", pts[k]["filename"]), pano, [cv2.IMWRITE_JPEG_QUALITY, 95])     # face detector over all of it, so a person this step missed holds the stop back
            rep[ids[k]] = dict(status="no capturer found: picture kept as it is", mask_px_pano=0, hole_m2=0.0, real_pct=0.0, blur_pct=0.0, grown_m2=0.0, neighbours=[])
            print(ids[k], "no capturer found: picture kept as it is (the privacy step still checks it)", flush=True)
            json.dump(rep, open(rep_path, "w"), indent=1)
            continue
        t = time.time()
        pano = pano_of(k)
        nfp = os.path.join(a.frames, "depth", ids[k] + "_nf.png")
        new, st, dbg = C.fill_stop(k, pano, UP[k], P["nodes"][k]["cam_h"] or 2.0, masks[k], neighbours(P, k), load, floor_mask=cv2.imread(nfp, 0) if os.path.exists(nfp) else None)
        imwrite(os.path.join(a.frames, "clean", pts[k]["filename"]), new, [cv2.IMWRITE_JPEG_QUALITY, 95])
        st["mask_px_pano"] = int(((new.astype(np.int16) - pano.astype(np.int16)) != 0).any(-1).sum())
        rep[ids[k]].update(st)
        fl0 = dbg["fill"]; cv_ = np.zeros((C.FILL_N, C.FILL_N), np.uint8); cv_[fl0.y0:fl0.y1, fl0.x0:fl0.x1] = fl0.hole.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(cdir, ids[k] + "_hole.png"), cv2.resize(cv_, (C.VIEW_N, C.VIEW_N), interpolation=cv2.INTER_AREA))         # the hole after it grew over what was left of him
        fl = dbg["fill"]; canvas = np.zeros((C.FILL_N, C.FILL_N), np.uint8)
        canvas[fl.y0:fl.y1, fl.x0:fl.x1] = (dbg["hole"] & ~dbg["got"]).astype(np.uint8) * 255      # the part no neighbour could supply (a soft fill)
        cv2.imwrite(os.path.join(cdir, ids[k] + "_soft.png"), cv2.resize(canvas, (C.VIEW_N, C.VIEW_N), interpolation=cv2.INTER_AREA))
        canvas[:] = 0; canvas[fl.y0:fl.y1, fl.x0:fl.x1] = dbg["floor_ref"].astype(np.uint8) * 255           # floor the neighbours vouch for: the 4b check measures against it
        cv2.imwrite(os.path.join(cdir, ids[k] + "_floor.png"), cv2.resize(canvas, (C.VIEW_N, C.VIEW_N), interpolation=cv2.INTER_AREA))
        rep[ids[k]]["status"] = "filled"
        print(ids[k], f"real {st['real_pct']}%  blurred {st['blur_pct']}%  hole {st['hole_m2']} m2  neighbours used "
              f"{[(n['j'] + 1, n['used']) for n in st['neighbours'] if n.get('used')]}  ({time.time() - t:.1f}s)", flush=True)
        json.dump(rep, open(rep_path, "w"), indent=1)
    if a.sheet:
        make_sheet(a.frames, pts, UP, a.sheet)


if __name__ == "__main__":
    main()
