"""Blur what identifies a person beyond their body, in every kept panorama: licence plates, faces, and printed names (a name tag, a badge, lettering on a vehicle or on a jacket).

Usage:  python pii_blur.py [--frames frames] [--only 003 004]
        python pii_blur.py --frames frames --apply-dir DIR [--out DIR2]     # blur another set of equirects of the SAME stops (e.g. a sharper copy) with the regions this step found
Runs after the privacy step (the people are already painted out or held) on every picture in frames/safe/ (the pictures the tour, the dollhouse and the depth are made from),
and is ON by default; SHOP_TOUR_PII_BLUR=0 turns it off. Writes frames/safe/<id>.jpg again (only when something was found), frames/pii.json (per stop: counts by kind, the regions as
ANGLES, so they apply to a copy of the stop at any size) and, per stop, a "pii" entry on frames/privacy.json.
What it finds (no model beyond the two the privacy gate already loads, nothing from the network, nothing paid):
  - faces: OpenCV YuNet (MIT, models/) at a LOWER bar than the gate's (0.6: a blurred logo costs nothing, a missed face does), over the gate's twelve views, and over the head of every
    person the detector finds;
  - plates: RT-DETR (Apache-2.0) finds the vehicles (car, bus, truck, motorcycle); inside each, in a view cut round it at the picture's own resolution, a plate-like region is a
    text-like region (many vertical strokes in a row, a bright-and-dark rectangle 1.6-8 times wider than tall) in the lower part of the vehicle. THIS IS NOT A PLATE READER and not a
    trained plate detector: it is "text-like rectangles on a vehicle", which finds most legible plates, also blurs some grilles and signs, and misses a plate too small or too blurred
    to show strokes (about 25 pixels wide in the picture) - a plate that small cannot be read in the picture either;
  - printed names: the same text-like rectangles elsewhere on a vehicle (a company's door lettering) and on a person's chest (a name tag, a badge).
A region is blurred in the picture's own pixels: its box is grown (a margin of half its size at least), pixelated in cells of at least 8 pixels and at most a quarter of the region across
(so even a close plate is four cells wide), then Gaussian-blurred at the cell size, under a soft edge. That cannot be undone from the picture.
NEVER A GATE: a stop whose detection or blur fails ships as it was, with the failure in frames/pii.json (and in manifest.json, which counts them); this step always exits 0.
The same picture twice gives the same pictures (nothing random; a stop whose file is the one this step wrote is skipped, so a re-run never blurs twice or recompresses).
"""
import argparse, hashlib, json, math, os, sys, types

import cv2
import numpy as np

import privacy_gate as PG
import remove_others as RO
from imutil import imread, imwrite

VERSION = 3
FACE_SCORE = 0.3                  # YuNet's own bar: every candidate; what is KEPT is decided by FACE_HEAD / FACE_VIEW / FACE_ALONE below (measured by eye on two walks, see README)
FACE_HEAD = 0.3                   # a face found in the zoomed look at a person's head: kept at any score the detector gives (a head zoom only ever shows a person; a blurred head is never wrong)
FACE_VIEW = 0.6                   # a face found only in the wider views: kept at this score when it stands on a person or in a vehicle's upper half (a driver) ...
FACE_ALONE = 0.88                 # ... and at this score anywhere (a poster's face); below that and on nothing it is a shelf, a fan, a wheel (a shop's clutter scores 0.6-0.81)
VEHICLE_SCORE, PERSON_SCORE, PERSON_LOOSE = 0.4, 0.5, 0.35
FACE_VIEWS = [(y, 40.0) for y in range(0, 360, 90)] + [(y, 75.0) for y in (0, 120, 240)] + [(y, -75.0) for y in (0, 120, 240)]     # beyond the gate's twelve: a face turned up at the camera or looking down at it (the walker's own)
VEHICLE_IDS = (2, 3, 5, 7)        # COCO: car, motorcycle, bus, truck
PLATE_MIN_SCORE = 0.45            # the plate model's bar when the text detector also boxes the same place ...
PLATE_LOOSE_SCORE = 0.7           # ... and on its own (a plate the text detector could not read, or on a trailer or a van the vehicle detector missed)
PLATE_ASPECT = (1.3, 6.0)         # width / height of a plate (a close, angled one boxes at 1.3): a taller box is a bin or a poster, a wider one a grille or a bumper
PLATE_ALONE = 0.8                 # with no vehicle found at all, the plate model alone must be this sure (a shop has no vehicles and plenty of red bags and labels at 0.7)
PLATE_LOW = 0.4                   # "low on the vehicle": its centre is this fraction of the vehicle half-height below the vehicle's centre or lower (a small plate the text detector cannot read, at a bumper)
PLATE_VEH_HEIGHT = 0.20           # a plate is at most this much of its vehicle's height (a "plate" the size of the vehicle is the vehicle)
PLATE_MAX_HALF = (7.0, 2.5)       # and, with no vehicle found, at most this many degrees half-width / half-height (a plate 15 cm tall at 2 m is 2 degrees)
TEXT_SCORE = 0.5                  # the text detector's bar (DBNet's box score)
TEXT_SURE = 0.9                   # a box DBNet scores this high, wide, not leaf-coloured, with strong contrast and two glyph-sized pieces is kept without texty()'s row-of-equal-glyphs check
                                  # (a shelter's "Goldsboro Road", a billboard's name and phone number: 22 of the 29 boxes this lets in on the road walk are lettering; the score alone is no guide: foliage scores 0.97)
TEXT_SURE_ASPECT, TEXT_SURE_GREEN = 1.5, 0.2
TEXT_CONTRAST, TEXT_GLYPHS = 45.0, 3   # texty(): the two sides of the box differ by this many grey levels, and this many glyph-like pieces stand in it
TEXT_MIN_PX = 9                   # a text box shorter than this in the tile is noise, not lettering
TEXT_MAX_SHARE = 0.35             # a "text" box wider or taller than this much of its tile is glare or sky
TEXT_TILES = [(az, 0.0, 66.0) for az in range(0, 360, 60)] + [(az, -35.0, 66.0) for az in range(0, 360, 90)]     # (az, el, fov): the whole sphere that can carry lettering, at the picture's own resolution
PLATE_SHAPE = 0.22                # a text box on a vehicle is a plate when it is this much of the vehicle across or less, low on it, and 1.5-7 times wider than tall
VEHICLE_LOW = 0.30                # "low" = the lower 70 % of the vehicle's box
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TEXT_ONNX = os.path.join(MODELS, "text_det_ppocrv3_en.onnx")                    # PaddleOCR PP-OCRv3 DBNet (Apache-2.0) from opencv_zoo, run by OpenCV's own DB reader
PLATE_DIR = os.path.join(MODELS, "plate_rtdetrv2")                              # justjuu/rtdetr-v2-license-plate-detection (Apache-2.0 per its model card)
PINNED = {os.path.join(TEXT_ONNX): "03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587",
          os.path.join(PLATE_DIR, "model.safetensors"): "bf44ad41aee4b1716dc8e53bdcaf7cf31ebf626ca8af20d86b4630129d19d719",
          os.path.join(PLATE_DIR, "config.json"): "427324aeee2a8caab58f3f9a73fdec95d363c1e5bd2d00ab76de202e56c2b579",
          os.path.join(PLATE_DIR, "preprocessor_config.json"): "addc7f2f00f04f33a5b4ab7f8eee13c66a430348fa396c7a5f2d384da515f586"}
CELL_MIN, CELL_ACROSS = 8, 4      # pixelation: cells of at least 8 pixels, and no more than a quarter of the region across ...
GROW = {"face": (1.5, 0.15), "plate": (1.4, 0.25), "name": (1.4, 0.25), "text": (1.3, 0.15)}     # ... after growing the box: (factor, extra degrees)
KINDS = ("plate", "face", "name", "text")


def texty(img, b):
    """Is the text detector's box really lettering? The detector fires on foliage, wires and a wheel's spokes at high scores; letters are a few dark-on-bright (or bright-on-dark) glyphs of ONE height.
    Otsu splits the box; the smaller side must be 8-60 % of it and differ from the other by at least TEXT_CONTRAST grey levels; and at least TEXT_GLYPHS of its connected pieces must be tall (0.4 of the box)
    and within 0.6-1.5 times the median height of the tall ones."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    px, py = max(2, (x1 - x0) // 20), max(2, (y1 - y0) // 10)
    x0, y0, x1, y1 = max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img[y0:y1, x0:x1]
    if g.shape[0] < 6 or g.shape[1] < 6:
        return False
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = g[bw == 0], g[bw > 0]
    if lo.size < 8 or hi.size < 8 or abs(float(hi.mean()) - float(lo.mean())) < TEXT_CONTRAST:
        return False
    mino = (bw > 0) if hi.size <= lo.size else (bw == 0)
    share = float(mino.mean())
    if not (0.08 <= share <= 0.60):
        return False
    n, _, st, _ = cv2.connectedComponentsWithStats(mino.astype(np.uint8), connectivity=8)
    H0 = g.shape[0]
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_HEIGHT] >= 0.4 * H0 and st[i, cv2.CC_STAT_AREA] >= 6]
    if len(keep) < TEXT_GLYPHS:
        return False
    hs = np.array([st[i, cv2.CC_STAT_HEIGHT] for i in keep], np.float32)
    cy = np.array([st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] / 2 for i in keep], np.float32)
    med = float(np.median(hs))
    same = (hs >= 0.6 * med) & (hs <= 1.5 * med)
    if int(same.sum()) < TEXT_GLYPHS:
        return False
    return float(cy[same].std()) <= 0.2 * med                                  # letters stand on one line; leaves and wires do not


def texty_lite(img, b):
    """The second chance for a box DBNet is sure of (>= TEXT_SURE) that texty() refused: wide (>= 1.5 times), under a fifth leaf-coloured, the two sides of Otsu's split >= TEXT_CONTRAST apart and at least two tall
    pieces in it. Multi-line lettering, a sign with two or three big letters, white on red, all fail texty()'s one-row rule; foliage, wheels and a pole's wires fail this one."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 6 or y1 - y0 < 6 or (x1 - x0) < TEXT_SURE_ASPECT * (y1 - y0):
        return False
    crop = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if float(((hsv[..., 0] >= 28) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 50) & (hsv[..., 2] >= 25)).mean()) >= TEXT_SURE_GREEN:
        return False
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = g[bw == 0], g[bw > 0]
    if lo.size < 8 or hi.size < 8 or abs(float(hi.mean()) - float(lo.mean())) < TEXT_CONTRAST:
        return False
    mino = (bw > 0) if hi.size <= lo.size else (bw == 0)
    n, _, st, _ = cv2.connectedComponentsWithStats(mino.astype(np.uint8), connectivity=8)
    return sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_HEIGHT] >= 0.3 * g.shape[0] and st[i, cv2.CC_STAT_AREA] >= 6) >= 2


def _check(path):
    """Weights are loaded BY PATH and only when their sha256 is the pinned one (tools/fetch_pii_models.py downloaded them once; a run never fetches)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{os.path.relpath(path, os.path.dirname(MODELS))} is missing - run: python tools/fetch_pii_models.py")
    if _sha(path) != PINNED[path]:
        raise ValueError(f"{os.path.relpath(path, os.path.dirname(MODELS))} is not the pinned file (sha256 differs)")


def enabled():
    return os.environ.get("SHOP_TOUR_PII_BLUR", "").strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------------------------------------------------------------
# the finder: what the detectors say about a view (a fake one stands in for it in the tests)

class Finder:
    """RT-DETR (the privacy gate's own, vehicles as well as people) and YuNet at the lower bar."""
    def __init__(self, det=None):
        self.det = det or PG.Detectors()
        self.face = cv2.FaceDetectorYN.create(PG.YUNET, "", (PG.VIEW_N, PG.VIEW_N), FACE_SCORE, 0.3, 5000)
        self.missing = []                                                       # models that were not there: faces and vehicles are still found, and the record says what was not
        self.tdet = self.pdet = None
        try:
            _check(TEXT_ONNX)
            self.tdet = cv2.dnn_TextDetectionModel_DB(TEXT_ONNX)
            self.tdet.setBinaryThreshold(0.3); self.tdet.setPolygonThreshold(0.6); self.tdet.setUnclipRatio(1.8); self.tdet.setMaxCandidates(300)
        except Exception as e:
            self.missing.append(f"text detector ({e})"[:160])
        try:
            for p in PINNED:
                if p.startswith(PLATE_DIR):
                    _check(p)
            from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection
            self.pproc = AutoImageProcessor.from_pretrained(PLATE_DIR, local_files_only=True)
            self.pdet = RTDetrV2ForObjectDetection.from_pretrained(PLATE_DIR, local_files_only=True).to(self.det.dev).eval()
        except Exception as e:
            self.missing.append(f"plate detector ({e})"[:160])

    def text(self, img):
        """[(score, [x0, y0, x1, y1])] text regions in a picture (axis-aligned boxes round DBNet's polygons), at the picture's own size."""
        if self.tdet is None:
            return []
        n = int(max(32, round(max(img.shape[:2]) / 32) * 32))
        self.tdet.setInputParams(1 / 255.0, (n, n), (122.67891434, 116.66876762, 104.00698793), True)
        boxes, conf = self.tdet.detect(img)
        out = []
        for q, c in zip(boxes, conf):
            q = np.asarray(q, np.float32)
            out.append((float(c), [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]))
        return out

    def plates(self, imgs):
        """One list per picture of (score, [x0, y0, x1, y1]): the plate model's boxes."""
        if self.pdet is None:
            return [[] for _ in imgs]
        from PIL import Image
        t, out = self.det.torch, []
        for c0 in range(0, len(imgs), 8):
            vs = imgs[c0:c0 + 8]
            x = self.pproc(images=[Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)) for v in vs], return_tensors="pt").to(self.det.dev)
            with t.no_grad():
                o = self.pdet(**x)
            for r in self.pproc.post_process_object_detection(o, threshold=PLATE_MIN_SCORE, target_sizes=t.tensor([list(v.shape[:2]) for v in vs])):
                out.append([(float(s), [float(v) for v in b]) for s, b in zip(r["scores"].cpu().numpy(), r["boxes"].cpu().numpy())])
        return out

    def objects(self, views):
        """One list per view of (kind, score, [x0, y0, x1, y1]), kind "vehicle" or "person"."""
        from PIL import Image
        d, t = self.det, self.det.torch
        out = []
        for c0 in range(0, len(views), 8):
            vs = views[c0:c0 + 8]
            x = d.proc(images=[Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)) for v in vs], return_tensors="pt").to(d.dev)
            with t.no_grad():
                o = d.model(**x)
            res = d.proc.post_process_object_detection(o, threshold=min(VEHICLE_SCORE, PERSON_SCORE), target_sizes=t.tensor([list(v.shape[:2]) for v in vs]))
            for r in res:
                lst = []
                for s, l, b in zip(r["scores"].cpu().numpy(), r["labels"].cpu().numpy(), r["boxes"].cpu().numpy()):
                    l = int(l)
                    if l == 0 and s >= PERSON_LOOSE:
                        lst.append(("person", float(s), [float(v) for v in b]))
                    elif l in VEHICLE_IDS and s >= VEHICLE_SCORE:
                        lst.append(("vehicle", float(s), [float(v) for v in b]))
                out.append(lst)
        return out

    def faces(self, view):
        """[(score, [x0, y0, x1, y1])] in the view's pixels."""
        self.face.setInputSize((view.shape[1], view.shape[0]))
        _, f = self.face.detect(view)
        return [] if f is None else [(float(r[14]), [float(r[0]), float(r[1]), float(r[0] + r[2]), float(r[1] + r[3])]) for r in f]


# ---------------------------------------------------------------------------------------------------------------------------------
# geometry: boxes in a view <-> angular regions of the pano

def _region(kind, az, el, hw, hh, score=None):
    r = dict(kind=kind, az=round(float(az), 3), el=round(float(el), 3), hw=round(float(hw), 3), hh=round(float(hh), 3))
    if score is not None:
        r["score"] = round(float(score), 2)
    return r


def _from_rays(kind, ray, box, score=None):
    """A box (x0, y0, x1, y1) in a rectilinear view whose pixel -> pano ray map is `ray(x, y)` -> a region: the centre as angles, and half extents as HALF THE ANGLE between the rays at
    the box's edges (a box far off the view's axis is wider in pixels than in degrees: reading it as pixels / focal length made the regions too big, and off centre, at the view's edges)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    r = ray(cx, cy)
    az, el = math.degrees(math.atan2(r[0], r[2])), math.degrees(math.asin(float(np.clip(r[1], -1, 1))))
    ang = lambda a, b: math.degrees(math.acos(float(np.clip(np.dot(a, b), -1, 1))))
    hw = ang(ray(box[0], cy), ray(box[2], cy)) / 2; hh = ang(ray(cx, box[1]), ray(cx, box[3])) / 2
    return _region(kind, az, el, hw, hh, score)


def _from_view_box(kind, box, yaw, pitch, n, fov, score=None):
    """A box in the gate's rectilinear n x n view looking at (yaw, pitch) -> a region (centre, half extents in degrees)."""
    f = n / 2 / math.tan(math.radians(fov) / 2)
    # (depth_panos.view_rays: ray = d @ Rx(pitch).T @ Ry(yaw).T)
    p, y = math.radians(pitch), math.radians(yaw)
    Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]]); Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])

    def ray(x, yy):
        d = np.array([x - n / 2 + 0.5, n / 2 - yy - 0.5, f]); d = d / np.linalg.norm(d)
        return d @ Rx.T @ Ry.T
    return _from_rays(kind, ray, box, score)


def _from_view_obj(kind, box, view, score=None):
    """The same for a RO.View (a zoomed view)."""
    def ray(x, y):
        return view.rays[int(np.clip(round(y), 0, view.n - 1)), int(np.clip(round(x), 0, view.n - 1))]
    return _from_rays(kind, ray, box, score)


def _zoom(pano, az, el, deg):
    """A view cut round (az, el) wide enough for something `deg` degrees across with room, at about the picture's own resolution."""
    ppd = pano.shape[1] / 360.0
    fov = float(np.clip(deg * 1.8, 8.0, 80.0))
    n = int(np.clip(round(fov * ppd / 16) * 16, 256, 1024))
    v = RO.View(az, el, fov, n)
    return v, v.cut(pano)


# ---------------------------------------------------------------------------------------------------------------------------------
# finding

def _merge(regions):
    """The same thing seen in two overlapping views is one region: of two of one kind whose centres are closer than the larger half-width, keep the larger."""
    out = []
    for r in sorted(regions, key=lambda r: -(r["hw"] * r["hh"])):
        if not any(o["kind"] == r["kind"] and PG._ang((o["az"], o["el"]), (r["az"], r["el"])) < max(o["hw"], r["hw"]) for o in out):
            out.append(r)
    return sorted(out, key=lambda r: (r["kind"], r["az"], r["el"]))


def detect(fd, pano):
    """Every face, plate, printed name and other text found in the picture: [{kind, az, el, hw, hh}] (angles in degrees, half extents).
    Faces: YuNet over the head of every person (kept at any score) and over twenty-six views (kept at 0.6 on a person or beside a vehicle's driver, 0.88 anywhere). Plates: the plate model over a view cut round each vehicle (and over the twelve views, where it must be sure).
    Text: the text detector over tiles at the picture's own resolution; each text region is a plate (on a vehicle, plate-shaped, low on it), a name (on a person, or other lettering on a vehicle) or text."""
    views = [PG.cut(pano, yaw, pitch) for yaw, pitch in PG.VIEWS]
    objs = fd.objects(views)
    found, vehicles, people, plates, vfaces = [], [], [], [], []
    for (yaw, pitch), v, ob in zip(PG.VIEWS, views, objs):
        for s, b in fd.faces(v):
            vfaces.append(_from_view_box("face", b, yaw, pitch, PG.VIEW_N, PG.VIEW_FOV, s))
        for kind, s, b in ob:
            (vehicles if kind == "vehicle" else people).append(_from_view_box(kind, b, yaw, pitch, PG.VIEW_N, PG.VIEW_FOV, s))
    for yaw, pitch in FACE_VIEWS:
        for s, b in fd.faces(PG.cut(pano, yaw, pitch)):
            vfaces.append(_from_view_box("face", b, yaw, pitch, PG.VIEW_N, PG.VIEW_FOV, s))
    veh = _merge([dict(v, kind="vehicle") for v in vehicles]); per_all = _merge([dict(p, kind="person") for p in people])
    per = [p for p in per_all if p.get("score", 1.0) >= PERSON_SCORE]
    inside = lambda r, b, grow=1.0: abs((r["az"] - b["az"] + 180) % 360 - 180) <= b["hw"] * grow and abs(r["el"] - b["el"]) <= b["hh"] * grow
    ok_shape = lambda b: PLATE_ASPECT[0] <= (b[2] - b[0]) / max(1.0, b[3] - b[1]) <= PLATE_ASPECT[1]
    # faces: a face in the zoomed look at a person's head is kept at any score; a face found only in the views needs a person or a vehicle's upper part under it, or a high score
    for p in per_all:
        for fr in (0.5, 0.3):
            hv, himg = _zoom(pano, p["az"], p["el"] + 0.6 * p["hh"], fr * max(p["hw"], p["hh"]) * 2)
            for s, b in fd.faces(himg):
                if s >= FACE_HEAD:
                    found.append(_from_view_obj("face", b, hv, s))
    for f in vfaces:
        s = f["score"]
        if s >= FACE_ALONE or (s >= FACE_VIEW and (any(inside(f, p) for p in per_all) or any(inside(f, v) and f["el"] >= v["el"] for v in veh))):
            found.append(f)
    # plates: a view cut round each vehicle, where its plate is tens of pixels wide, kept when the text detector boxes the same place too or the plate model is sure on its own; and the twelve
    # views, where a plate on a vehicle nobody found must be a sure thing. A plate is plate-shaped and small beside the vehicle it is on.
    zooms = [_zoom(pano, v["az"], v["el"], 2 * max(v["hw"], v["hh"])) for v in veh]
    zp = fd.plates([z[1] for z in zooms])
    for (view, zimg), ps, vh in zip(zooms, zp, veh):
        tb = None
        for s, b in ps:
            if not ok_shape(b):
                continue
            if s < PLATE_LOOSE_SCORE:
                if tb is None:
                    tb = [t[1] for t in fd.text(zimg) if t[0] >= TEXT_SCORE]
                r = _from_view_obj("plate", b, view, s)
                low = r["el"] <= vh["el"] - PLATE_LOW * vh["hh"] and r["hw"] <= 0.45 * vh["hw"] and (b[2] - b[0]) >= 1.5 * (b[3] - b[1])      # low on the vehicle, narrow beside it, plate-shaped: where plates are
                if not low and not any(min(b[2], t[2]) - max(b[0], t[0]) > 0 and min(b[3], t[3]) - max(b[1], t[1]) > 0 for t in tb):
                    continue
            else:
                r = _from_view_obj("plate", b, view, s)
            if r["hh"] <= PLATE_VEH_HEIGHT * vh["hh"]:
                plates.append(r)
    for (yaw, pitch), ps in zip(PG.VIEWS, fd.plates(views)):
        for s, b in ps:
            if s >= PLATE_LOOSE_SCORE and ok_shape(b):
                r = _from_view_box("plate", b, yaw, pitch, PG.VIEW_N, PG.VIEW_FOV, s)
                vh = next((v for v in sorted(veh, key=lambda v: v["hw"] * v["hh"]) if inside(r, v, 1.1)), None)
                if (r["hh"] <= PLATE_VEH_HEIGHT * vh["hh"]) if vh else (s >= PLATE_ALONE and r["hw"] <= PLATE_MAX_HALF[0] and r["hh"] <= PLATE_MAX_HALF[1]):
                    plates.append(r)
    found += plates
    # text: tiles at the picture's own resolution
    ppd = pano.shape[1] / 360.0
    for az, el, fov in TEXT_TILES:
        n = int(np.clip(round(fov * ppd / 32) * 32, 512, 1536))
        view = RO.View(az, el, fov, n)
        tile = view.cut(pano)
        for s, b in fd.text(tile):
            w, h = b[2] - b[0], b[3] - b[1]
            if s < TEXT_SCORE or h < TEXT_MIN_PX or w > TEXT_MAX_SHARE * n or h > TEXT_MAX_SHARE * n or not getattr(fd, "skip_texty", False) and not texty(tile, b) and not (s >= TEXT_SURE and texty_lite(tile, b)):
                continue
            r = _from_view_obj("text", b, view, s)
            v = next((v for v in sorted(veh, key=lambda v: v["hw"] * v["hh"]) if inside(r, v)), None)
            if v is not None:
                low = r["el"] <= v["el"] - 0.3 * v["hh"]
                shaped = 1.5 <= r["hw"] / max(r["hh"], 1e-6) <= 7.0 and r["hw"] <= PLATE_SHAPE * v["hw"] * 2
                r["kind"] = "plate" if (low and shaped) else "name"           # lettering on a vehicle: a plate-shaped one low on it, else a company's name on a door
            elif any(inside(r, p) for p in per):
                r["kind"] = "name"
            found.append(r)
    pl = [r for r in found if r["kind"] == "plate"]                             # a text box on a plate the plate model found is one blur, the plate's
    found = [r for r in found if r["kind"] != "text" and r["kind"] != "name" or not any(inside(r, q) for q in pl)]
    found = _merge(found)
    # a "face" low on a vehicle and on no person is a wheel or a bumper: faces in cars are in the upper part of the box
    return [r for r in found if not (r["kind"] == "face" and not any(inside(r, p) for p in per)
                                     and any(inside(r, v) and r["el"] < v["el"] - 0.2 * v["hh"] for v in veh))]


# ---------------------------------------------------------------------------------------------------------------------------------
# blurring

def _mask_window(region, H, W):
    """(rows, cols, mask) of the part of an H x W equirect a region covers: mask is 1 inside the (grown) box and 0 outside, before the soft edge."""
    fac, extra = GROW[region["kind"]]
    hw, hh = region["hw"] * fac + extra, region["hh"] * fac + extra
    reach = max(hw, hh)
    stub = types.SimpleNamespace(az=region["az"], el=region["el"], fov=min(170.0, 3.0 * reach + 4.0))
    rows, cols = RO.window_pano(stub, H, W)
    az = ((cols + 0.5) / W * 2 * np.pi - np.pi).astype(np.float32); el = (np.pi / 2 - (rows + 0.5) / H * np.pi).astype(np.float32)
    A, E = np.meshgrid(az, el)
    r = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1)
    Rx, Ry = RO._basis(region["az"], region["el"])
    v = r @ (Rx.T @ Ry.T).T.astype(np.float32)
    vz = np.maximum(v[..., 2], 1e-6)
    x, y = v[..., 0] / vz, v[..., 1] / vz
    tx, ty = math.tan(math.radians(min(hw, 80.0))), math.tan(math.radians(min(hh, 80.0)))
    if region["kind"] == "face":
        inside = (v[..., 2] > 1e-3) & ((x / tx) ** 2 + (y / ty) ** 2 <= 1.0)
    else:
        inside = (v[..., 2] > 1e-3) & (np.abs(x) <= tx) & (np.abs(y) <= ty)
    return rows, cols, inside.astype(np.float32), (hw, hh)


def blur_regions(pano, regions):
    """The picture with every region pixelated and blurred in place; returns (picture, counts by kind)."""
    H, W = pano.shape[:2]
    out = pano.copy(); counts = {k: 0 for k in KINDS}
    for rg in regions:
        rows, cols, m, (hw, hh) = _mask_window(rg, H, W)
        if not m.any():
            continue
        ppd = W / 360.0
        cell = int(max(CELL_MIN, round(2 * min(hw, hh) * ppd / CELL_ACROSS)))
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, cell / 4.0))                    # (a soft edge, a fraction of a cell)
        ix = np.ix_(rows, cols)
        sub = out[ix].astype(np.float32)
        h, w = sub.shape[:2]
        small = cv2.resize(sub, (max(1, w // cell), max(1, h // cell)), interpolation=cv2.INTER_AREA)
        pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        pix = cv2.GaussianBlur(pix, (0, 0), cell * 0.75)
        out[ix] = np.clip(sub * (1 - m[..., None]) + pix * m[..., None] + 0.5, 0, 255).astype(np.uint8)
        counts[rg["kind"]] = counts.get(rg["kind"], 0) + 1
    return out, counts


# ---------------------------------------------------------------------------------------------------------------------------------
# the step

def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _counts_of(regions):
    return {k: sum(1 for r in regions if r["kind"] == k) for k in KINDS}


class Session:
    """What one blur pass keeps between stops: the detectors (loaded at the first stop that needs them, once), the earlier record (frames/pii.json) and whether the
    detectors failed to load. run() and Stream below share it, so a stop is looked at in exactly the same way whichever of them looks at it."""
    def __init__(self, frames, fd_factory=None, log=print):
        self.frames, self.fd_factory, self.log = frames, fd_factory, log
        self.safe = os.path.join(frames, "safe")
        pj = os.path.join(frames, "pii.json")
        self.old = {}
        try:
            self.old = json.load(open(pj)).get("stops", {}) if os.path.exists(pj) else {}
        except Exception:
            self.old = {}
        self.fd, self.fd_err, self.warned = None, None, False

    def one(self, p):
        """The record of one stop whose picture is in frames/safe/ (blurring it in place when something is found). Never raises."""
        sid, fn, log = p["id"], os.path.join(self.safe, p["filename"]), self.log
        o = self.old.get(sid)
        sha = _sha(fn)
        if o and o.get("version") == VERSION and o.get("sha_out") == sha and not o.get("error"):
            return o                                                            # the picture this step already wrote (or looked at and left): done
        try:
            if self.fd is None and self.fd_err is None:
                try:
                    self.fd = (self.fd_factory or Finder)()
                except Exception as e:
                    self.fd_err = f"the detectors did not load: {type(e).__name__}: {e}"
            if self.fd_err:
                raise RuntimeError(self.fd_err)
            if not self.warned and getattr(self.fd, "missing", None):
                self.warned = True; log("note: " + "; ".join(self.fd.missing) + " - those are not looked for")
            pano = imread(fn)
            regions = detect(self.fd, pano)
            if regions:
                out, counts = blur_regions(pano, regions)
                imwrite(fn, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
                sha_out = _sha(fn)
            else:
                counts, sha_out = _counts_of(regions), sha
            r = dict(version=VERSION, counts=_counts_of(regions), regions=regions, sha_in=sha, sha_out=sha_out, error=None)
            log(f"{sid}: blurred " + (", ".join(f"{n} {k}" for k, n in r["counts"].items() if n) or "nothing"))
            return r
        except Exception as e:                                                  # never a gate: the stop ships as it was, and the record says so
            r = dict(version=VERSION, counts={k: 0 for k in KINDS}, regions=[], sha_in=sha, sha_out=sha, error=f"{type(e).__name__}: {e}"[:300])
            log(f"{sid}: NOT blurred ({r['error']}); it ships as it was")
            return r


def run(frames, pts, fd_factory=None, only=None, log=print):
    """Blur every picture in frames/safe/ that the tour will use. Returns the record that is written to frames/pii.json."""
    ses = Session(frames, fd_factory, log)
    rec = {}
    for p in pts:
        sid, fn = p["id"], os.path.join(ses.safe, p["filename"])
        if not os.path.exists(fn):
            continue                                                            # held, left out, or not checked: not in the tour
        if only and sid not in only:
            if sid in ses.old:
                rec[sid] = ses.old[sid]
            continue
        rec[sid] = ses.one(p)
    return rec


def stream_wanted():
    """THE BLUR BESIDE THE PRIVACY CHAIN (2026-10-30, speed): privacy_chain.py blurs each stop as soon as its picture in frames/safe/ is final, while the masks of the
    later stops are still being made, instead of this step starting only when the whole chain is done (it was 200-470 s on the critical path of every walk, after it).
    Only where a stop's safe picture is written once and never taken back: SHOP_TOUR_PEOPLE keep or blur (in remove mode the margin can still hold a stop later, so
    the step runs after the chain as before). SHOP_TOUR_PII_STREAM=0 turns it off; SHOP_TOUR_PII_BLUR=0 turns the blur off altogether."""
    import people_mode
    if not enabled() or people_mode.remove():
        return False
    return os.environ.get("SHOP_TOUR_PII_STREAM", "").strip().lower() not in ("0", "false", "no", "off")


class Stream:
    """The blur step on a thread of its own, fed stop by stop (submit(k): stop k's picture in frames/safe/ is final). close() looks at every stop with a safe picture
    that was not looked at (never submitted, or written again after it was), and returns the record in the stop order: what run() returns for the same pictures.
    One stop at a time, in the order they come."""
    def __init__(self, frames, pts, fd_factory=None, log=print):
        import queue, threading
        self.frames, self.pts = frames, pts
        self.ses = Session(frames, fd_factory, log)
        self.q, self.got = queue.Queue(), {}
        self.busy = 0.0
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        import time
        while True:
            k = self.q.get()
            if k is None:
                return
            p = self.pts[k]
            if not os.path.exists(os.path.join(self.ses.safe, p["filename"])):
                continue
            t = time.time()
            try:
                r = self.ses.one(p)
            except BaseException as e:                                           # (Session.one never raises; a stop that did is looked at again in close())
                self.ses.log(f"{p['id']}: the blur beside the chain stopped ({type(e).__name__}: {e}); looked at again at the end")
                continue
            self.busy += time.time() - t
            self.got[p["id"]] = r

    def submit(self, k):
        self.q.put(k)

    def close(self):
        self.q.put(None)
        self.th.join()
        rec = {}
        for p in self.pts:
            sid, fn = p["id"], os.path.join(self.ses.safe, p["filename"])
            if not os.path.exists(fn):
                continue                                                        # (as run(): a stop with no safe picture is not in the tour)
            r = self.got.get(sid)
            if r is None or _sha(fn) != r.get("sha_out"):
                r = self.ses.one(p)                                             # never submitted, or its picture was written again after it was looked at
            rec[sid] = r
        return rec


def write_record(frames, rec, streamed=False):
    """frames/pii.json (written last, so the pictures are never newer than it) and each stop's pii entry on frames/privacy.json (the margin's marker ignores it).
    streamed: written by privacy_chain.py (Stream); a stop it could not look at is then looked at again by tour.py's pii step (tour.incomplete)."""
    tot = {k: sum(r["counts"].get(k, 0) for r in rec.values()) for k in KINDS}
    pf = os.path.join(frames, "privacy.json")
    try:
        doc = json.load(open(pf))
        for sid, r in rec.items():
            if sid in doc.get("stops", {}):
                doc["stops"][sid]["pii"] = dict(counts=r["counts"], error=r["error"])
        json.dump(doc, open(pf, "w"), indent=1)
    except Exception as e:
        print(f"note: privacy.json was not updated with the blur counts ({e})")
    body = dict(version=VERSION, enabled=True, totals=tot, stops_blurred=sum(1 for r in rec.values() if any(r["counts"].values())),
                errors=sorted(sid for sid, r in rec.items() if r["error"]), stops=rec)
    if streamed:
        body["streamed"] = True
    json.dump(body, open(os.path.join(frames, "pii.json"), "w"), indent=1)
    return body


def apply_dir(frames, src, dst, rec=None):
    """Blur another set of equirects of the same stops (named <stop filename>) with the regions found on the safe pictures; the regions are angles, so size does not matter.
    The pictures must show the same direction at the same yaw as the safe ones (the margin on each region covers a few pixels of stitch difference, not a different stitch);
    returns {stop id: counts}."""
    rec = rec or json.load(open(os.path.join(frames, "pii.json")))["stops"]
    pts = json.load(open(os.path.join(frames, "points.json")))
    os.makedirs(dst, exist_ok=True); done = {}
    for p in pts:
        fn = os.path.join(src, p["filename"])
        r = rec.get(p["id"])
        if not r or not os.path.exists(fn):
            continue
        pano = imread(fn)
        out, counts = blur_regions(pano, r["regions"]) if r["regions"] else (pano, {})
        if r["regions"] or os.path.abspath(src) != os.path.abspath(dst):
            imwrite(os.path.join(dst, p["filename"]), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        done[p["id"]] = counts
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--apply-dir", default=None, help="blur this folder of equirects (named like frames/safe/*.jpg) with the regions already found")
    ap.add_argument("--out", default=None, help="with --apply-dir: write here instead of in place")
    a = ap.parse_args()
    try:
        if a.apply_dir:
            d = apply_dir(a.frames, a.apply_dir, a.out or a.apply_dir)
            print(f"[pii] {len(d)} pictures of {a.apply_dir} blurred with the regions found on the tour's own")
            return
        pts = json.load(open(os.path.join(a.frames, "points.json")))
        if not enabled():
            print("SHOP_TOUR_PII_BLUR=0: plates, faces and printed names are not blurred by this step. Nothing done.")
            json.dump(dict(version=VERSION, enabled=False, totals={k: 0 for k in KINDS}, stops_blurred=0, errors=[], stops={}), open(os.path.join(a.frames, "pii.json"), "w"), indent=1)
            return
        rec = run(a.frames, pts, only=a.only)
        body = write_record(a.frames, rec)
        t = body["totals"]
        print(f"[pii] {len(rec)} pictures: {t['plate']} plates, {t['face']} faces, {t['name']} printed names, {t['text']} other text regions blurred" + (f"; {len(body['errors'])} stops could not be checked: {', '.join(body['errors'])}" if body["errors"] else ""))
    except Exception as e:                                                      # never a gate
        print(f"[pii] note: the step failed ({type(e).__name__}: {e}); the pictures ship as they were")


if __name__ == "__main__":
    main()
