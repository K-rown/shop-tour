"""Fail closed on people: nothing with a person in it goes into the tour unless you say so.

Usage:  python privacy_gate.py [--frames frames] [--keep 010 017] [--leave-out 002 005]
Runs on every cleaned stop (frames/clean/<id>.jpg, made by remove_capturer.py):
  1. the 4b check of qa/check_capturer.py (nothing person-sized or taller than 15 cm is left where the capturer stood),
  2. a person detector over the WHOLE picture (twelve rectilinear views: bystanders, and anyone the mask step missed),
  3. a face detector (OpenCV YuNet, MIT) over the same views. Every face found is blurred in the picture, as a backstop, and the stop is held.
A stop that fails any of them is HELD BACK: build_tour.py leaves it out of the tour. The result is frames/privacy.json and a review page,
frames/review/review.html (NOT inside tour/: it shows the held pictures, with a white ring round each person or face the check found). --keep ids: you looked at those stops and want them in the tour; recorded
in frames/privacy.json with the date, and the stop's faces stay blurred. --leave-out ids: you looked at those stops and they stay out; recorded as reviewed
(manifest.json then reports the privacy review as all clear).
Models: RT-DETR (PekingU/rtdetr_r50vd, Apache-2.0) via transformers, YuNet (models/face_detection_yunet_2023mar.onnx, MIT).
"""
import argparse, datetime, functools, html, json, os, sys
import cv2
import numpy as np
import capturer as C
import privacy_inputs as PI
from imutil import imread, imwrite

HERE = os.path.dirname(os.path.abspath(__file__))
YUNET = os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx")
PERSON_ID = "PekingU/rtdetr_r50vd"
PERSON_SCORE = 0.5               # a real person scored 0.93-0.95 in every original; 0.35 flagged tool racks on most stops of a workshop walk. A false alarm costs a look at the review page
FACE_SCORE = 0.8                 # YuNet gives 0.6-0.75 to logos and tool faces in a shop; real faces in this walk scored 0.8-0.94
NADIR_SCORE = 0.30               # the person search of the straight-down view (see nadir_persons): a person at the camera's feet, seen from above, scores 0.3-0.45 where a side view gives 0.9
NADIR_POLAR_DEG = 35.0           # ... counted only this close to straight down (the camera holder stands 8-46 degrees off it; a shelf, a vest or a pile elsewhere in the view is not him)
NADIR_CROP = (150, 874)          # the middle 724 px of the straight-down view, enlarged to 1024: the detector needs the head and shoulders large
HORIZON_FLOOR = 0.30             # a person far off, small in the frame, scores 0.57-0.66 in the view that sees him best and 0.3-0.45 in the others; his best view can fall under PERSON_SCORE
HORIZON_BEST = 0.40              # a person found by two views, one of them at least this good, round the horizon, is held although no view reaches PERSON_SCORE (SHOP_TOUR_HORIZON_PERSONS=0: off)
HORIZON_EL = 15.0                # "round the horizon": within this many degrees of it
HORIZON_VIEWS = 2                # seen at HORIZON_FLOOR or better in this many different views
FACE_SURE_SCORE = 0.9            # a face that scores this is held on its own
FACE_SURE_DEG = 10.0             # ... and so is one this wide (the camera holder's own face, seen from above: 10-17 degrees); below both it needs a person with it
FACE_BODY_DEG = 40.0             # a person found this close (angle between the centres) to a small face is the body it belongs to
FACE_ZOOM_FOV, FACE_ZOOM_SCORE = 40.0, 0.45
VIEW_N, VIEW_FOV = 1024, 90.0
VIEWS = [(y, 0.0) for y in range(0, 360, 45)] + [(y, -40.0) for y in range(0, 360, 90)]      # (yaw, pitch)


def cut(pano, yaw, pitch):
    """The view of the pano at (yaw, pitch) - what depth_panos.cut_view makes, with its pixel maps worked out once per direction and picture size
    (the twelve views are the same for every stop)."""
    u, v = _cut_maps(float(yaw), float(pitch), VIEW_FOV, VIEW_N, *pano.shape[:2])
    return cv2.remap(pano, u, v, cv2.INTER_AREA if pano.shape[1] > 4 * VIEW_N else cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


@functools.lru_cache(maxsize=64)
def _cut_maps(yaw, pitch, fov, n, h, w):
    from depth_panos import view_rays
    d = view_rays(yaw, pitch, fov, n)
    az = np.arctan2(d[..., 0], d[..., 2]); el = np.arcsin(np.clip(d[..., 1], -1, 1))
    return ((az + np.pi) / (2 * np.pi) * w - 0.5).astype(np.float32), ((np.pi / 2 - el) / np.pi * h - 0.5).astype(np.float32)


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

    def persons_at(self, view, threshold):
        """persons() with the score bar given (the straight-down view needs a lower one than the side views)."""
        from PIL import Image
        t = self.torch
        x = self.proc(images=Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(self.dev)
        with t.no_grad():
            o = self.model(**x)
        res = self.proc.post_process_object_detection(o, threshold=threshold, target_sizes=t.tensor([view.shape[:2]]))[0]
        return [(float(s), [float(v) for v in b]) for s, l, b in zip(res["scores"].cpu().numpy(), res["labels"].cpu().numpy(), res["boxes"].cpu().numpy()) if int(l) == 0]

    def persons_many_at(self, views, threshold):
        """persons_many() with the score bar given."""
        from PIL import Image
        t = self.torch
        x = self.proc(images=[Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)) for v in views], return_tensors="pt").to(self.dev)
        with t.no_grad():
            o = self.model(**x)
        res = self.proc.post_process_object_detection(o, threshold=threshold, target_sizes=t.tensor([list(v.shape[:2]) for v in views]))
        return [[(float(s), [float(v) for v in b]) for s, l, b in zip(r["scores"].cpu().numpy(), r["labels"].cpu().numpy(), r["boxes"].cpu().numpy()) if int(l) == 0] for r in res]

    def persons_many(self, views):
        """persons() for several views at once (one pass of the detector): a list with one result per view."""
        from PIL import Image
        t = self.torch
        x = self.proc(images=[Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)) for v in views], return_tensors="pt").to(self.dev)
        with t.no_grad():
            o = self.model(**x)
        res = self.proc.post_process_object_detection(o, threshold=PERSON_SCORE, target_sizes=t.tensor([list(v.shape[:2]) for v in views]))
        return [[(float(s), [float(v) for v in b]) for s, l, b in zip(r["scores"].cpu().numpy(), r["labels"].cpu().numpy(), r["boxes"].cpu().numpy()) if int(l) == 0] for r in res]

    def clone(self):
        """A copy that can be used from another thread beside this one (the models are shared, the face detector is not)."""
        import copy
        c = copy.copy(self)
        c.face = cv2.FaceDetectorYN.create(YUNET, "", (VIEW_N, VIEW_N), FACE_SCORE, 0.3, 5000)
        return c

    def faces(self, view):
        self.face.setInputSize((view.shape[1], view.shape[0]))
        _, f = self.face.detect(view)
        return [] if f is None else [(float(r[14]), [float(v) for v in r[:4]]) for r in f]


def horizon_enabled():
    return os.environ.get("SHOP_TOUR_HORIZON_PERSONS", "").strip().lower() not in ("0", "false", "no", "off")


def horizon_people(cands, tol=8.0):
    """A distant person who no single view scores at PERSON_SCORE. cands: [{score, az, el, deg, view}] - every person box between HORIZON_FLOOR and PERSON_SCORE.
    The same person in overlapping views is one cluster (centres within tol degrees); a cluster is a person when its best score is at least HORIZON_BEST, it lies within
    HORIZON_EL degrees of the horizon and at least HORIZON_VIEWS different views see it. Measured on two real outdoor walks: the flagger of one walk scored 0.49 in his
    best view at a stop the gate passed, a pedestrian and a crew of five scored 0.41-0.44 in theirs; on two test walks (67 stops) one stop is added (a bin bag on a shelf)."""
    clusters = []
    for d in sorted(cands, key=lambda d: -d["score"]):
        for c in clusters:
            if _ang((c[0]["az"], c[0]["el"]), (d["az"], d["el"])) < tol:
                c.append(d); break
        else:
            clusters.append([d])
    out = []
    for c in clusters:
        b = c[0]
        if b["score"] >= HORIZON_BEST and abs(b["el"]) <= HORIZON_EL and len({tuple(x["view"]) for x in c}) >= HORIZON_VIEWS:
            out.append(dict(kind="person", score=b["score"], az=b["az"], el=b["el"], deg=b["deg"], horizon=True))
    return out


FINE_SCORE_PAINT = 0.40
FINE_FOV, FINE_N, FINE_STEP, FINE_PITCHES = 30.0, 768, 24, (-10.0, 8.0)      # the zoomed look (below): 30 degree views every 24 degrees round the horizon band


def fine_enabled():
    """The zoomed look for people far off is on by default with the paint-out (SHOP_TOUR_REMOVE_OTHERS, remove_others.py) or with SHOP_TOUR_FINE_PERSONS=1: the gate's answer with SHOP_TOUR_REMOVE_OTHERS=0 and no FINE_PERSONS is what it always was."""
    on = ("1", "true", "yes", "on")
    return (os.environ.get("SHOP_TOUR_FINE_PERSONS", "").strip().lower() in on) or (os.environ.get("SHOP_TOUR_REMOVE_OTHERS", "").strip().lower() not in ("0", "false", "no", "off"))


def fine_score():
    """The bar of the zoomed look. With SHOP_TOUR_REMOVE_OTHERS=1 (a person found is painted out, so a lower bar costs a patch of picture, not a held stop) it is FINE_SCORE_PAINT 0.40:
    a driver behind a windscreen 30 m off scores 0.40-0.47 in the zoomed view, a crew 40 m off 0.41-0.6, a cone 0.5-0.6 (measured on three outdoor walks). With only
    SHOP_TOUR_FINE_PERSONS=1 it is PERSON_SCORE. SHOP_TOUR_FINE_SCORE=<0.2..0.95> sets it."""
    try:
        v = float(os.environ.get("SHOP_TOUR_FINE_SCORE", "") or 0)
    except ValueError:
        v = 0.0
    if 0.2 <= v <= 0.95:
        return v
    return FINE_SCORE_PAINT if os.environ.get("SHOP_TOUR_REMOVE_OTHERS", "").strip().lower() not in ("0", "false", "no", "off") else PERSON_SCORE


def fine_people(det, pano):
    """People who are small in the twelve 90 degree views (a crew 30-40 m off is 4-6 degrees across, 40-60 pixels, and scores 0.4-0.6 there): the horizon band cut into 30 degree
    views, three times closer, the same detector at the same bar. [{kind, score, az, el, deg, fine}]"""
    if not hasattr(det, "persons_many_at"):
        return []
    H, W = pano.shape[:2]
    views, meta = [], []
    for pitch in FINE_PITCHES:
        for yaw in range(-180, 180, FINE_STEP):
            u, v = _cut_maps(float(yaw), float(pitch), FINE_FOV, FINE_N, H, W)
            views.append(cv2.remap(pano, u, v, cv2.INTER_AREA if W > 4 * FINE_N else cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)); meta.append((yaw, pitch))
    out = []
    f = FINE_N / 2 / np.tan(np.radians(FINE_FOV) / 2)
    for c0 in range(0, len(views), 8):
        for (yaw, pitch), pp in zip(meta[c0:c0 + 8], det.persons_many_at(views[c0:c0 + 8], fine_score())):
            for s, b in pp:
                az, el = to_angles((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, yaw, pitch, FINE_N, FINE_FOV)
                deg = float(np.degrees(2 * np.arctan(max(b[2] - b[0], b[3] - b[1]) / 2 / f)))
                out.append(dict(kind="person", score=round(s, 2), az=round(az, 1), el=round(el, 1), deg=round(deg, 1), fine=True))
    return out


def scan(det, pano):
    """Persons and faces anywhere in the picture: [{kind, score, az, el, deg}] (deg = angular size of the box)."""
    found = []
    views = [cut(pano, yaw, pitch) for yaw, pitch in VIEWS]
    low = det.persons_many_at(views, HORIZON_FLOOR) if hasattr(det, "persons_many_at") and horizon_enabled() else None
    if low is not None:
        people = [[(s, b) for s, b in pp if s >= PERSON_SCORE] for pp in low]               # (the same pass at a lower bar: what scores 0.5 or more is what it always was)
    else:
        people = det.persons_many(views) if hasattr(det, "persons_many") else [det.persons(v) for v in views]
    cands = []
    for k, ((yaw, pitch), v, pp) in enumerate(zip(VIEWS, views, people)):
        for kind, lst in (("person", pp), ("face", det.faces(v))):
            for s, b in lst:
                x0, y0, x1, y1 = (b[0], b[1], b[2], b[3]) if kind == "person" else (b[0], b[1], b[0] + b[2], b[1] + b[3])
                az, el = to_angles((x0 + x1) / 2, (y0 + y1) / 2, yaw, pitch)
                deg = float(np.degrees(2 * np.arctan(max(x1 - x0, y1 - y0) / 2 / (VIEW_N / 2 / np.tan(np.radians(VIEW_FOV) / 2)))))
                found.append(dict(kind=kind, score=round(s, 2), az=round(az, 1), el=round(el, 1), deg=round(deg, 1)))
        if low is not None:
            for s, b in low[k]:
                if s < PERSON_SCORE:
                    az, el = to_angles((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, yaw, pitch)
                    deg = float(np.degrees(2 * np.arctan(max(b[2] - b[0], b[3] - b[1]) / 2 / (VIEW_N / 2 / np.tan(np.radians(VIEW_FOV) / 2)))))
                    cands.append(dict(score=round(s, 2), az=round(az, 1), el=round(el, 1), deg=round(deg, 1), view=(yaw, pitch)))
    if cands:
        sure = [f for f in found if f["kind"] == "person"]
        found += [h for h in horizon_people(cands) if not any(_ang((h["az"], h["el"]), (f["az"], f["el"])) < 12.0 for f in sure)]
    if fine_enabled():
        found += fine_people(det, pano)
    return found


def nadir_boxes(boxes, up, crop=NADIR_CROP, polar_max=NADIR_POLAR_DEG, n=C.VIEW_N):
    """Person boxes found in the enlarged middle of the straight-down view -> [{kind, score, az, el, deg, nadir}] in the picture's own angles, only
    those whose centre is within polar_max degrees of straight down. boxes: [(score, [x0, y0, x1, y1])] in the enlarged crop's pixels (n x n)."""
    d, f = C.view_dirs(); R = C.rot_to_y(up)
    c0, c1 = crop; k = (c1 - c0) / float(n)
    out = []
    for s, b in boxes:
        cx, cy = c0 + (b[0] + b[2]) / 2 * k, c0 + (b[1] + b[3]) / 2 * k
        r = d[int(np.clip(cy, 0, n - 1)), int(np.clip(cx, 0, n - 1))]
        polar = float(np.degrees(np.arccos(np.clip(-r[1], -1, 1))))
        if polar > polar_max:
            continue
        w = r @ R
        deg = float(np.degrees(2 * np.arctan(max(b[2] - b[0], b[3] - b[1]) * k / 2 / f)))
        out.append(dict(kind="person", score=round(float(s), 2), az=round(float(np.degrees(np.arctan2(w[0], w[2]))), 1), el=round(float(np.degrees(np.arcsin(np.clip(w[1], -1, 1)))), 1), deg=round(deg, 1), nadir=True))
    return out


def nadir_persons(det, pano, up):
    """A person at the camera's feet. The twelve side views see the floor under the camera only at a slant, and a head and grey shoulders seen from straight
    above scored under 0.5 (measured on a walk where the capturer found nobody and left the camera holder in the picture: 0.39), so the straight-down view
    gets its own person search: its middle enlarged, a lower bar (NADIR_SCORE), boxes counted only near straight down. Measured on two real walks: no
    stop labelled 'nobody' by hand is held by it."""
    if not hasattr(det, "persons_at"):
        return []
    v = C.level_view(pano, up); c0, c1 = NADIR_CROP
    crop = cv2.resize(v[c0:c1, c0:c1], (C.VIEW_N, C.VIEW_N), interpolation=cv2.INTER_CUBIC)
    return nadir_boxes(det.persons_at(crop, NADIR_SCORE), up)


def _ang(a, b):
    """Angle in degrees between two (az, el) directions given in degrees."""
    def vec(az, el):
        az, el = np.radians(az), np.radians(el)
        return np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    return float(np.degrees(np.arccos(np.clip(vec(*a) @ vec(*b), -1, 1))))


def face_is_small_and_alone(face, persons):
    """True for a face detection that is probably not a face: YuNet gave it less than FACE_SURE_SCORE, it is narrower than FACE_SURE_DEG, and no person
    (persons: the picture's person detections, side views and straight down) is within FACE_BODY_DEG of it. Measured on two real walks: the detections like this
    were a wire spool's hub, a hard hat, a vest on a chair back and a light fitting (score 0.81-0.84, 3-9 degrees wide, a person nowhere near); every face of the
    camera holder in the original pictures was 10-17 degrees wide and had a person detection of 0.9+ beside it."""
    if face["score"] >= FACE_SURE_SCORE or face["deg"] >= FACE_SURE_DEG:
        return False
    return not any(_ang((face["az"], face["el"]), (p["az"], p["el"])) < FACE_BODY_DEG for p in persons)


def face_has_body(det, pano, face):
    """A last look before a small face is let go: the picture round it enlarged (FACE_ZOOM_FOV wide) and searched for a person at a low bar. True when a
    person box holds the face's centre - a stranger far away whose body the side views scored low."""
    if not hasattr(det, "persons_at"):
        return False
    if abs(float(face["el"])) > 80:                                              # (straight up or down the face is not at the middle of a view: not checked, so the face holds)
        return True
    from depth_panos import view_rays
    n = VIEW_N
    d = view_rays(float(face["az"]), float(face["el"]), FACE_ZOOM_FOV, n)
    az = np.arctan2(d[..., 0], d[..., 2]); el = np.arcsin(np.clip(d[..., 1], -1, 1)); H, W = pano.shape[:2]
    view = cv2.remap(pano, ((az + np.pi) / (2 * np.pi) * W - 0.5).astype(np.float32), ((np.pi / 2 - el) / np.pi * H - 0.5).astype(np.float32),
                     cv2.INTER_AREA if W > 4 * n else cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    c = n / 2
    return any(b[0] <= c <= b[2] and b[1] <= c <= b[3] for sc, b in det.persons_at(view, FACE_ZOOM_SCORE))


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


def mark_detections(thumb, found):
    """Ring every person and face the check found on the review picture, where it found it, so the person looking at the picture
    can see WHAT was flagged. found: [{az, el, deg}] (degrees, as scan() gives them; az 0 is the middle column, + to the right,
    el + is up). The ring is white on a dark outline (readable on any floor) and no smaller than 28 px. Returns a new picture."""
    out = thumb.copy()
    H, W = out.shape[:2]
    for f in found or []:
        x = int(round((f["az"] / 360.0 + 0.5) * W)) % W
        y = int(round((0.5 - f["el"] / 180.0) * H))
        r = max(28, int(f.get("deg", 20) / 360.0 * W / 2 * 1.3))
        for dx in (-W, 0, W):                                    # a ring across the left/right edge of the picture is drawn on both sides
            cv2.circle(out, (x + dx, y), r, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.circle(out, (x + dx, y), r, (255, 255, 255), 3, cv2.LINE_AA)
    return out


def review_page(frames, pts, state, models_used):
    d = os.path.join(frames, "review"); os.makedirs(d, exist_ok=True)
    ups = None

    def one(p):
        nonlocal ups
        s = state[p["id"]]
        if not s["held"] and not s.get("kept"):
            return None
        cp = os.path.join(frames, "clean", p["filename"])
        if not os.path.exists(cp):
            return None
        if ups is None:
            ups = json.load(open(os.path.join(frames, "up.json")))
        img = imread(cp)
        thumb = mark_detections(cv2.resize(img, (1024, 512), interpolation=cv2.INTER_AREA), (s.get("persons") or []) + (s.get("faces") or []))
        cv2.imwrite(os.path.join(d, p["id"] + ".jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 82])
        nadir = C.level_view(img, ups[pts.index(p)], 512)
        cv2.imwrite(os.path.join(d, p["id"] + "_down.jpg"), nadir, [cv2.IMWRITE_JPEG_QUALITY, 82])
        why = "".join(f"<li>{html.escape(r)}</li>" for r in s["reasons"])
        status = "KEPT by you" if s.get("kept") else "HELD BACK: not in the tour"
        return (f'<section><h2>Stop {p["id"]} <span class="{"k" if s.get("kept") else "h"}">{status}</span></h2><ul>{why}</ul>'
                f'<img src="{p["id"]}.jpg" alt="whole picture of stop {p["id"]}"><img src="{p["id"]}_down.jpg" alt="straight down at stop {p["id"]}"></section>')
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as ex:                  # (a picture each: reading, shrinking and writing them side by side)
        rows = [r for r in ex.map(one, pts) if r is not None]
    page = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Review: stops held back</title>'
            '<style>body{font:16px/1.4 system-ui,sans-serif;margin:16px;max-width:900px;background:#fff;color:#111}img{width:100%;margin:6px 0;border-radius:8px}'
            '.h{color:#b3261e}.k{color:#1a7a3c}section{margin:0 0 28px}h2{font-size:18px}code{background:#eee;padding:1px 5px;border-radius:4px}</style>'
            '<h1>Stops held back</h1><p>These pictures may show a person. They are <b>not</b> in the tour. This page is for you only: do not publish this folder.</p>'
            '<p>To use a stop anyway, look at it, then run <code>python privacy_gate.py --keep 010</code> (faces stay blurred) and rebuild. '
            'To fix it properly, record that spot again (see CAPTURE.md).</p>' + "".join(rows or ["<p>Nothing was held back.</p>"]))
    open(os.path.join(d, "review.html"), "w", encoding="utf-8").write(page)
    return os.path.join(d, "review.html")


class Gate:
    """The scan of one stop at a time (what main() does for every stop; privacy_chain.py hands it models that are already loaded and stops as their pictures arrive).
    state is filled in place: {stop id: {held, reasons, persons, faces, ...}}; old is the privacy.json that was there before (his "kept" answers)."""
    def __init__(self, frames, pts, state, old, det=None, models=None, rep_cap=None):
        self.frames, self.pts, self.state, self.old = frames, pts, state, old
        self.safe = os.path.join(frames, "safe"); os.makedirs(self.safe, exist_ok=True)
        self.UP = json.load(open(os.path.join(frames, "up.json"))); self.PIN = PI.Inputs(frames, len(pts))        # camera height: privacy_inputs.py (not the map)
        sys.path.insert(0, os.path.join(HERE, "qa"))
        from check_capturer import check_stop
        self.check_stop = check_stop
        import neighbour_check, people_mode
        self.people = people_mode.mode()                                # blur (the default): other people are smudged and nothing is held; remove: the older paint-out + hold flow; keep: people untouched, nothing held
        self.nbr = neighbour_check.get_context(frames) if self.people == "remove" else None       # a second opinion on check C: an object that two neighbouring stops see too is not him (neighbour_check.py)
        self.det = det or Detectors(); self.models = models or C.Models()
        self.rep_cap = rep_cap if rep_cap is not None else json.load(open(os.path.join(frames, "capturer", "report.json")))

    def stop(self, k):
        frames, state, old, safe = self.frames, self.state, self.old, self.safe
        p = self.pts[k]; sid = p["id"]
        cp = os.path.join(frames, "clean", p["filename"]); mp = os.path.join(frames, "capturer", sid + ("_hole.png" if os.path.exists(os.path.join(frames, "capturer", sid + "_hole.png")) else "_mask.png"))
        if not os.path.exists(cp) or not os.path.exists(mp):
            state[sid] = dict(held=True, reasons=["no cleaned picture (the capturer step did not finish for this stop)"], persons=[], faces=[])
            return
        pano = imread(cp); self.last = (sid, pano)
        sp = os.path.join(frames, "capturer", sid + "_soft.png")
        fp = os.path.join(frames, "capturer", sid + "_floor.png")
        if self.people != "remove":
            return self.stop_unheld(k, pano, mp)
        res, *_ = self.check_stop(self.models, sid, pano, self.UP[k], self.PIN.cam_h(k), cv2.imread(mp, 0) > 0, (cv2.imread(sp, 0) > 0) if os.path.exists(sp) else None,
                                  (cv2.imread(fp, 0) > 127) if os.path.exists(fp) else None, self.rep_cap.get(sid, {}).get("leftover_m2"), self.nbr)
        found = dedupe(scan(self.det, pano) + nadir_persons(self.det, pano, self.UP[k]))
        persons = [f for f in found if f["kind"] == "person"]; faces = [f for f in found if f["kind"] == "face"]
        unsure_faces = [f for f in faces if face_is_small_and_alone(f, persons) and not face_has_body(self.det, pano, f)]
        faces_held = [f for f in faces if not any(f is u for u in unsure_faces)]        # (every face is still blurred below; only these hold the stop)
        reasons = list(res["problems"])
        reasons += [f"another person in the picture (looking {f['az']:+.0f} deg sideways, {f['el']:+.0f} deg up, score {f['score']})" for f in persons]
        reasons += [f"a face (looking {f['az']:+.0f} deg sideways, {f['el']:+.0f} deg up); it is blurred in any copy that is kept" for f in faces_held]
        keep_old = state.get(sid, {}).get("kept") and bool(reasons)
        state[sid] = dict(held=bool(reasons), reasons=reasons, persons=persons, faces=faces_held, blurred_faces=len(faces), check_4b=res)
        if unsure_faces:
            state[sid]["faces_unconfirmed"] = unsure_faces                                  # small, no person with them: blurred, not held (see face_is_small_and_alone)
        if keep_old:
            state[sid]["kept"] = state[sid]["kept_on"] = old["stops"][sid].get("kept_on", "earlier")
        out = blur_faces(pano, faces)
        if not reasons or state[sid].get("kept"):
            imwrite(os.path.join(safe, p["filename"]), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif os.path.exists(os.path.join(safe, p["filename"])):
            os.remove(os.path.join(safe, p["filename"]))                                       # a stop that is held now must not stay in the tour
        print(f"{sid}: " + ("HELD  " + "; ".join(reasons) if reasons else "ok"), flush=True)

    def stop_unheld(self, k, pano, mp):
        """SHOP_TOUR_PEOPLE=blur (default) or keep: the same scan, but nothing is ever held. Every person found who is not the walker is smudged (blur) or left alone (keep), every face is blurred,
        and the picture goes into the tour. state[sid] says how many people were smudged; the check of the walker's fill (check C) is not made - there is no fill to check."""
        state, safe = self.state, self.safe
        p = self.pts[k]; sid = p["id"]
        raw = scan(self.det, pano) + nadir_persons(self.det, pano, self.UP[k])
        found = dedupe(raw)
        persons = [f for f in found if f["kind"] == "person"]; faces = [f for f in found if f["kind"] == "face"]
        unsure_faces = [f for f in faces if face_is_small_and_alone(f, persons) and not face_has_body(self.det, pano, f)]
        faces_held = [f for f in faces if not any(f is u for u in unsure_faces)]
        state[sid] = dict(held=False, reasons=[], persons=persons, faces=faces_held, blurred_faces=len(faces), people_mode=self.people, people_blurred=0)
        if unsure_faces:
            state[sid]["faces_unconfirmed"] = unsure_faces
        out = pano
        if self.people == "blur":
            import people_blur
            walker = None
            m = cv2.imread(mp, 0)
            if m is not None and (m > 0).any():                                         # (he was smudged by the capturer step; the scan may still find his shape: that is not "another person")
                wm, ok = C.view_to_pano((m > 0).astype(np.uint8) * 255, self.UP[k], 512, 1024)
                walker = (wm > 127) & ok
            out, recs = people_blur.smudge_people(pano, dedupe(raw, tol=4.0), self.det, self.models, walker)      # (tol 4: a crew standing together is smudged one by one, as remove_others.py paints them)
            state[sid]["people_blurred"] = sum(1 for r in recs if not r.get("failed"))
            state[sid]["people"] = recs
        out = blur_faces(out, faces)
        imwrite(os.path.join(safe, p["filename"]), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"{sid}: ok, {state[sid]['people_blurred']} people blurred" if self.people == "blur" else f"{sid}: ok (people left in the picture)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--keep", nargs="*", default=None, help="stop ids you looked at and want in the tour (no new scan is made)")
    ap.add_argument("--leave-out", nargs="*", default=None, dest="leave_out", help="held stop ids you looked at and want to stay OUT of the tour (recorded as reviewed; nothing is published)")
    ap.add_argument("--only", nargs="*", help="stop ids to scan (the others keep their last result)")
    a = ap.parse_args()
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    pf = os.path.join(a.frames, "privacy.json")
    old = json.load(open(pf)) if os.path.exists(pf) else {}
    state = old.get("stops", {})
    safe = os.path.join(a.frames, "safe"); os.makedirs(safe, exist_ok=True)
    scanning = a.keep is None and a.leave_out is None
    if scanning:
        G = Gate(a.frames, pts, state, old)
        for k, p in enumerate(pts):
            if a.only and p["id"] not in a.only:
                continue
            G.stop(k)
    elif a.leave_out is not None:
        for sid in a.leave_out:
            if sid not in state:
                sys.exit(f"stop {sid} has not been checked yet: run  python privacy_gate.py  first")
            if not state[sid]["held"]:
                print(f"{sid} is not held back; nothing to leave out"); continue
            state[sid]["left_out"] = True; state[sid]["left_out_on"] = datetime.date.today().isoformat()
            print(f"{sid}: stays out of the tour by your decision (recorded as reviewed)")
    else:
        for sid in a.keep:
            if sid not in state:
                sys.exit(f"stop {sid} has not been checked yet: run  python privacy_gate.py  first")
            if not state[sid]["held"]:
                print(f"{sid} is not held back; nothing to keep"); continue
            state[sid]["kept"] = True; state[sid]["kept_on"] = datetime.date.today().isoformat()
            from_clean = os.path.join(a.frames, "clean", next(p["filename"] for p in pts if p["id"] == sid))
            pano = imread(from_clean)
            imwrite(os.path.join(safe, os.path.basename(from_clean)), blur_faces(pano, state[sid]["faces"] + state[sid].get("faces_unconfirmed", [])), [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"{sid}: kept in the tour by your decision (faces blurred)")
    finish(a.frames, pts, state)


def finish(frames, pts, state):
    """Write frames/privacy.json and the review page, and say what was held."""
    for p in pts:
        state.setdefault(p["id"], dict(held=True, reasons=["not checked"], persons=[], faces=[]))
    import people_mode
    doc = dict(version=1, stops=state)
    if not people_mode.remove():
        doc["people_mode"] = people_mode.mode()                          # (the older flow's privacy.json is written exactly as it was)
    json.dump(doc, open(os.path.join(frames, "privacy.json"), "w"), indent=1)
    held = [p["id"] for p in pts if state[p["id"]]["held"] and not state[p["id"]].get("kept")]
    page = review_page(frames, pts, state, None)
    print(f"{chr(10)}{len(pts) - len(held)} of {len(pts)} stops go into the tour; {len(held)} held back" + (f": {', '.join(held)}" if held else ""))
    if not people_mode.remove():
        n = sum(int(s.get("people_blurred", 0)) for s in state.values())
        print(f"privacy: {n} people blurred, nothing held" if people_mode.mode() == "blur" else "privacy: people left in the picture (SHOP_TOUR_PEOPLE=keep), nothing held")
    print("review page:", page)


if __name__ == "__main__":
    main()
