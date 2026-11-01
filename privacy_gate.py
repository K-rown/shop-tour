"""Fail closed on people: nothing with a person in it goes into the tour unless you say so.

Usage:  python privacy_gate.py [--frames frames] [--keep 010 017]
Runs on every cleaned stop (frames/clean/<id>.jpg, made by remove_capturer.py):
  1. the 4b check of qa/check_capturer.py (nothing person-sized or taller than 15 cm is left where the capturer stood),
  2. a person detector over the WHOLE picture (twelve rectilinear views: bystanders, and anyone the mask step missed),
  3. a face detector (OpenCV YuNet, MIT) over the same views. Every face found is blurred in the picture, as a backstop, and the stop is held.
A stop that fails any of them is HELD BACK: build_tour.py leaves it out of the tour. The result is frames/privacy.json and a review page,
frames/review/review.html (NOT inside tour/: it shows the held pictures). --keep ids: you looked at those stops and want them in the tour; recorded
in frames/privacy.json with the date, and the stop's faces stay blurred.
Models: RT-DETR (PekingU/rtdetr_r50vd, Apache-2.0) via transformers, YuNet (models/face_detection_yunet_2023mar.onnx, MIT).
"""
import argparse, datetime, html, json, os, sys
import cv2
import numpy as np
import capturer as C
from imutil import imread, imwrite

HERE = os.path.dirname(os.path.abspath(__file__))
YUNET = os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx")
PERSON_ID = "PekingU/rtdetr_r50vd"
PERSON_SCORE = 0.5               # a real person scored 0.93-0.95 in every original; 0.35 flagged tool racks in the shop on most stops. A false alarm costs a look at the review page
FACE_SCORE = 0.8                 # YuNet gives 0.6-0.75 to logos and tool faces in a shop; real faces in this walk scored 0.8-0.94
VIEW_N, VIEW_FOV = 1024, 90.0
VIEWS = [(y, 0.0) for y in range(0, 360, 45)] + [(y, -40.0) for y in range(0, 360, 90)]      # (yaw, pitch)


def cut(pano, yaw, pitch):
    return C_view(pano, dict(yaw=float(yaw), pitch=float(pitch), fov=VIEW_FOV))


def C_view(pano, v, n=VIEW_N):
    from depth_panos import cut_view
    return cut_view(pano, v, n)


def to_angles(x, y, yaw, pitch, n=VIEW_N, fov=VIEW_FOV):
    """Pixel (x, y) of a view -> (azimuth, elevation) in degrees of the pano (azimuth 0 = pano centre, clockwise)."""
    from depth_panos import view_rays
    d = view_rays(yaw, pitch, fov, n)
    r = d[int(np.clip(y, 0, n - 1)), int(np.clip(x, 0, n - 1))]
    return float(np.degrees(np.arctan2(r[0], r[2]))), float(np.degrees(np.arcsin(np.clip(r[1], -1, 1))))


class Detectors:
    def __init__(self):
        os_env = os.environ
        os_env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1"); os_env.setdefault("HF_HUB_VERBOSITY", "error")
        import torch
        from transformers import AutoImageProcessor, RTDetrForObjectDetection
        self.torch = torch; self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(PERSON_ID)
        self.model = RTDetrForObjectDetection.from_pretrained(PERSON_ID).to(self.dev).eval()
        if not os.path.exists(YUNET):
            raise SystemExit(f"{YUNET} is missing (it ships with the project: models/). Restore it with git.")
        self.face = cv2.FaceDetectorYN.create(YUNET, "", (VIEW_N, VIEW_N), FACE_SCORE, 0.3, 5000)

    def persons(self, view):
        from PIL import Image
        t = self.torch
        x = self.proc(images=Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(self.dev)
        with t.no_grad():
            o = self.model(**x)
        res = self.proc.post_process_object_detection(o, threshold=PERSON_SCORE, target_sizes=t.tensor([view.shape[:2]]))[0]
        out = []
        for s, l, b in zip(res["scores"].cpu().numpy(), res["labels"].cpu().numpy(), res["boxes"].cpu().numpy()):
            if int(l) == 0:                                                                  # COCO class 0 = person
                out.append((float(s), [float(v) for v in b]))
        return out

    def faces(self, view):
        self.face.setInputSize((view.shape[1], view.shape[0]))
        _, f = self.face.detect(view)
        return [] if f is None else [(float(r[14]), [float(v) for v in r[:4]]) for r in f]


def scan(det, pano):
    """Persons and faces anywhere in the picture: [{kind, score, az, el, deg}] (deg = angular size of the box)."""
    found = []
    for yaw, pitch in VIEWS:
        v = cut(pano, yaw, pitch)
        for kind, lst in (("person", det.persons(v)), ("face", det.faces(v))):
            for s, b in lst:
                x0, y0, x1, y1 = (b[0], b[1], b[2], b[3]) if kind == "person" else (b[0], b[1], b[0] + b[2], b[1] + b[3])
                az, el = to_angles((x0 + x1) / 2, (y0 + y1) / 2, yaw, pitch)
                deg = float(np.degrees(2 * np.arctan(max(x1 - x0, y1 - y0) / 2 / (VIEW_N / 2 / np.tan(np.radians(VIEW_FOV) / 2)))))
                found.append(dict(kind=kind, score=round(s, 2), az=round(az, 1), el=round(el, 1), deg=round(deg, 1)))
    return found


def dedupe(found, tol=12.0):
    """The same person is seen in two overlapping views: keep the best of each cluster."""
    out = []
    for f in sorted(found, key=lambda f: -f["score"]):
        if not any(o["kind"] == f["kind"] and abs(((o["az"] - f["az"]) + 180) % 360 - 180) < tol and abs(o["el"] - f["el"]) < tol for o in out):
            out.append(f)
    return out


def blur_faces(pano, faces, pad=0.9):
    """Blur every face (heavy Gaussian under a soft angular ellipse)."""
    if not faces:
        return pano
    H, W = pano.shape[:2]
    sh, sw = H // 4, W // 4
    az = ((np.arange(sw) + 0.5) / sw * 2 * np.pi - np.pi); el = (np.pi / 2 - (np.arange(sh) + 0.5) / sh * np.pi)
    A, E = np.meshgrid(az, el)
    r = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1)
    m = np.zeros((sh, sw), np.float32)
    for f in faces:
        a, e = np.radians(f["az"]), np.radians(f["el"])
        d = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
        ang = np.degrees(np.arccos(np.clip(r @ d, -1, 1)))
        m = np.maximum(m, np.clip((f["deg"] * (0.5 + pad / 2) - ang) / 2.0 + 0.5, 0, 1))
    m = cv2.resize(cv2.GaussianBlur(m, (0, 0), 1.2), (W, H), interpolation=cv2.INTER_LINEAR)
    k = max(31, (min(H, W) // 60) | 1)
    blurred = cv2.GaussianBlur(cv2.resize(pano, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), k / 8)
    blurred = cv2.resize(blurred, (W, H), interpolation=cv2.INTER_LINEAR)
    return (pano * (1 - m[..., None]) + blurred * m[..., None]).astype(np.uint8)


def review_page(frames, pts, state, models_used):
    d = os.path.join(frames, "review"); os.makedirs(d, exist_ok=True)
    rows = []
    for p in pts:
        s = state[p["id"]]
        if not s["held"] and not s.get("kept"):
            continue
        cp = os.path.join(frames, "clean", p["filename"])
        if not os.path.exists(cp):
            continue
        img = imread(cp)
        thumb = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(d, p["id"] + ".jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 82])
        nadir = C.level_view(img, json.load(open(os.path.join(frames, "up.json")))[pts.index(p)], 512)
        cv2.imwrite(os.path.join(d, p["id"] + "_down.jpg"), nadir, [cv2.IMWRITE_JPEG_QUALITY, 82])
        why = "".join(f"<li>{html.escape(r)}</li>" for r in s["reasons"])
        status = "KEPT by you" if s.get("kept") else "HELD BACK: not in the tour"
        rows.append(f'<section><h2>Stop {p["id"]} <span class="{"k" if s.get("kept") else "h"}">{status}</span></h2><ul>{why}</ul>'
                    f'<img src="{p["id"]}.jpg" alt="whole picture of stop {p["id"]}"><img src="{p["id"]}_down.jpg" alt="straight down at stop {p["id"]}"></section>')
    page = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Review: stops held back</title>'
            '<style>body{font:16px/1.4 system-ui,sans-serif;margin:16px;max-width:900px;background:#fff;color:#111}img{width:100%;margin:6px 0;border-radius:8px}'
            '.h{color:#b3261e}.k{color:#1a7a3c}section{margin:0 0 28px}h2{font-size:18px}code{background:#eee;padding:1px 5px;border-radius:4px}</style>'
            '<h1>Stops held back</h1><p>These pictures may show a person. They are <b>not</b> in the tour. This page is for you only: do not publish this folder.</p>'
            '<p>To use a stop anyway, look at it, then run <code>python privacy_gate.py --keep 010</code> (faces stay blurred) and rebuild. '
            'To fix it properly, record that spot again (see CAPTURE.md).</p>' + "".join(rows or ["<p>Nothing was held back.</p>"]))
    open(os.path.join(d, "review.html"), "w", encoding="utf-8").write(page)
    return os.path.join(d, "review.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--keep", nargs="*", default=None, help="stop ids you looked at and want in the tour (no new scan is made)")
    ap.add_argument("--only", nargs="*", help="stop ids to scan (the others keep their last result)")
    a = ap.parse_args()
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    pf = os.path.join(a.frames, "privacy.json")
    old = json.load(open(pf)) if os.path.exists(pf) else {}
    state = old.get("stops", {})
    safe = os.path.join(a.frames, "safe"); os.makedirs(safe, exist_ok=True)
    scanning = a.keep is None
    if scanning:
        P = json.load(open(os.path.join(a.frames, "poses.json"))); UP = json.load(open(os.path.join(a.frames, "up.json")))
        sys.path.insert(0, os.path.join(HERE, "qa"))
        from check_capturer import check_stop
        det, models = Detectors(), C.Models()
        rep_cap = json.load(open(os.path.join(a.frames, "capturer", "report.json")))
        for k, p in enumerate(pts):
            sid = p["id"]
            if a.only and sid not in a.only:
                continue
            cp = os.path.join(a.frames, "clean", p["filename"]); mp = os.path.join(a.frames, "capturer", sid + ("_hole.png" if os.path.exists(os.path.join(a.frames, "capturer", sid + "_hole.png")) else "_mask.png"))
            if not os.path.exists(cp) or not os.path.exists(mp):
                state[sid] = dict(held=True, reasons=["no cleaned picture (the capturer step did not finish for this stop)"], persons=[], faces=[])
                continue
            pano = imread(cp)
            sp = os.path.join(a.frames, "capturer", sid + "_soft.png")
            fp = os.path.join(a.frames, "capturer", sid + "_floor.png")
            res, *_ = check_stop(models, sid, pano, UP[k], P["nodes"][k]["cam_h"] or 2.0, cv2.imread(mp, 0) > 0, (cv2.imread(sp, 0) > 0) if os.path.exists(sp) else None,
                                 (cv2.imread(fp, 0) > 127) if os.path.exists(fp) else None, rep_cap.get(sid, {}).get("leftover_m2"))
            found = dedupe(scan(det, pano))
            persons = [f for f in found if f["kind"] == "person"]; faces = [f for f in found if f["kind"] == "face"]
            reasons = list(res["problems"])
            reasons += [f"another person in the picture (looking {f['az']:+.0f} deg sideways, {f['el']:+.0f} deg up, score {f['score']})" for f in persons]
            reasons += [f"a face (looking {f['az']:+.0f} deg sideways, {f['el']:+.0f} deg up); it is blurred in any copy that is kept" for f in faces]
            keep_old = state.get(sid, {}).get("kept") and bool(reasons)
            state[sid] = dict(held=bool(reasons), reasons=reasons, persons=persons, faces=faces, blurred_faces=len(faces), check_4b=res)
            if keep_old:
                state[sid]["kept"] = state[sid]["kept_on"] = old["stops"][sid].get("kept_on", "earlier")
            out = blur_faces(pano, faces)
            if not reasons or state[sid].get("kept"):
                imwrite(os.path.join(safe, p["filename"]), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif os.path.exists(os.path.join(safe, p["filename"])):
                os.remove(os.path.join(safe, p["filename"]))                                       # a stop that is held now must not stay in the tour
            print(f"{sid}: " + ("HELD  " + "; ".join(reasons) if reasons else "ok"), flush=True)
    else:
        for sid in a.keep:
            if sid not in state:
                sys.exit(f"stop {sid} has not been checked yet: run  python privacy_gate.py  first")
            if not state[sid]["held"]:
                print(f"{sid} is not held back; nothing to keep"); continue
            state[sid]["kept"] = True; state[sid]["kept_on"] = datetime.date.today().isoformat()
            from_clean = os.path.join(a.frames, "clean", next(p["filename"] for p in pts if p["id"] == sid))
            pano = imread(from_clean)
            imwrite(os.path.join(safe, os.path.basename(from_clean)), blur_faces(pano, state[sid]["faces"]), [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"{sid}: kept in the tour by your decision (faces blurred)")
    for p in pts:
        state.setdefault(p["id"], dict(held=True, reasons=["not checked"], persons=[], faces=[]))
    json.dump(dict(version=1, stops=state), open(pf, "w"), indent=1)
    held = [p["id"] for p in pts if state[p["id"]]["held"] and not state[p["id"]].get("kept")]
    page = review_page(a.frames, pts, state, None)
    print(f"{chr(10)}{len(pts) - len(held)} of {len(pts)} stops go into the tour; {len(held)} held back" + (f": {', '.join(held)}" if held else ""))
    print("review page:", page)


if __name__ == "__main__":
    main()
