"""People stay in the picture, smudged (SHOP_TOUR_PEOPLE=blur, the default; see people_mode.py).

smudge(pano, mask, ...)        a strong Gaussian blur inside the mask (dilated, soft-edged), written into the picture; the rest of the picture is not touched
smudge_person(pano, person, det, models)   one person the gate's scan found: the same detector box and SAM 2 mask remove_others.py uses, then smudge()
smudge_people(pano, persons, det, models, walker_small=None)   every person but the one holding the stick; returns (picture, [record per person])
smudge_walker(pano, hole1024, up)   the person under the camera (the capturer step's hole): smudged and mildly darkened where the capturer step would have rebuilt the floor

The blur is made at about 1/4 of the picture's size (a sigma of 2% of the picture's width is 150 pixels on an 8K picture) and put back at full size inside the mask only.
sigma = SHOP_TOUR_PEOPLE_SIGMA (default 2%) of the picture's width, but never more than 0.8 of the person's own size: a small, far person is smudged, not the street round him.
The mask is dilated by the same 2% but never by more than half the person's size, and its edge is feathered. Nothing here calls a model of its own.
"""
import math
import cv2
import numpy as np

import people_mode as PM_

DARKEN_WALKER = 0.85            # the smudge under the camera is a little darker, so it reads as a shadow of the stick-holder rather than a bright blob
MIN_SIGMA_PX = 4.0


def _small_scale(W):
    return max(1, int(W) // 1920)


def smudge(pano, mask, sigma_px, dilate_px, darken=1.0, feather_frac=0.5):
    """pano: the picture (H, W, 3 uint8). mask: bool/uint8 array of ANY size (it is resized to the picture's; a smaller one is fine). Returns a new picture."""
    H, W = pano.shape[:2]
    sc = _small_scale(W)
    h, w = max(2, H // sc), max(2, W // sc)
    m = mask.astype(np.float32)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    m = (m > 0.3).astype(np.uint8)
    if not m.any():
        return pano.copy()
    r = int(round(dilate_px / sc))
    if r > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    fs = float(np.clip(feather_frac * max(r, 2), 1.0, 12.0))
    wgt = cv2.GaussianBlur(m.astype(np.float32), (0, 0), fs)
    wgt = np.clip(wgt * 1.6, 0.0, 1.0)                                              # (solid inside the mask, falling off outside it)
    wgt = np.where(m > 0, 1.0, wgt).astype(np.float32)
    small = cv2.resize(pano, (w, h), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (0, 0), max(MIN_SIGMA_PX, float(sigma_px)) / sc).astype(np.float32)
    if darken != 1.0:
        blurred *= darken
    ys, xs = np.nonzero(wgt > 0.003)
    r0, r1 = max(0, int(ys.min()) - 1), min(h, int(ys.max()) + 2)
    if xs.max() - xs.min() > 0.5 * w:                                                # across the picture's seam: every column of those rows
        c0, c1 = 0, w
    else:
        c0, c1 = max(0, int(xs.min()) - 1), min(w, int(xs.max()) + 2)
    R0, R1, C0, C1 = r0 * sc, min(H, r1 * sc), c0 * sc, min(W, c1 * sc)
    size = (C1 - C0, R1 - R0)
    up_b = cv2.resize(blurred[r0:r1, c0:c1], size, interpolation=cv2.INTER_LINEAR)
    up_w = cv2.resize(wgt[r0:r1, c0:c1], size, interpolation=cv2.INTER_LINEAR)[..., None]
    out = pano.copy()
    sub = out[R0:R1, C0:C1].astype(np.float32)
    out[R0:R1, C0:C1] = np.clip(sub * (1 - up_w) + up_b * up_w, 0, 255).astype(np.uint8)
    return out


def _sizes(size_pano_px, W):
    sig = min(PM_.sigma_frac() * W, 0.8 * size_pano_px)
    dil = min(PM_.sigma_frac() * W, 0.5 * size_pano_px)
    return max(MIN_SIGMA_PX, sig), max(2.0, dil)


def view_mask_to_pano(view, mask, H, W):
    """A mask in a rectilinear view's pixels -> a bool mask of an H x W equirect (a window of it is worked out, as remove_others.paste_view does)."""
    import remove_others as RO
    rows, cols = RO.window_pano(view, H, W)
    az = ((cols + 0.5) / W * 2 * np.pi - np.pi).astype(np.float32); el = (np.pi / 2 - (rows + 0.5) / H * np.pi).astype(np.float32)
    A, E = np.meshgrid(az, el)
    r = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1)
    col, row, front = view.project(r)
    n = view.n
    inside = front & (col > 1) & (col < n - 2) & (row > 1) & (row < n - 2)
    got = cv2.remap(mask.astype(np.float32), col, row, cv2.INTER_LINEAR, borderValue=0) > 0.5
    out = np.zeros((H, W), bool)
    out[np.ix_(rows, cols)] = got & inside
    return out


def smudge_person(pano, person, det, models, fallback=False):
    """Smudge one person (a privacy_gate.scan() result). Returns (new picture, record). fallback: no detector, no SAM: the box the scan's angles give, as an ellipse."""
    import remove_others as RO
    H, W = pano.shape[:2]
    fov, n = RO.view_size(person["deg"])
    view = RO.View(person["az"], person["el"], fov, n)
    vimg = view.cut(pano)
    if fallback:
        c = n / 2; dpx = person["deg"] / fov * n
        box, how = [max(0.0, c - dpx / 4), max(0.0, c - dpx / 2), min(n - 1.0, c + dpx / 4), min(n - 1.0, c + dpx / 2)], "angles"
        mask = np.zeros((n, n), np.uint8)
        cv2.ellipse(mask, (int(c), int(c)), (max(2, int((box[2] - box[0]) / 2)), max(2, int((box[3] - box[1]) / 2))), 0, 0, 360, 1, -1)
        mask, mhow = mask > 0, "ellipse"
    else:
        box, how = RO.find_box(det, vimg, person, view)
        mask, mhow, _ = RO.person_mask(models, vimg, box)
        core = np.zeros(mask.shape, np.uint8)                                      # (SAM sometimes leaves an arm or a head out: the box's inscribed ellipse is him too)
        cv2.ellipse(core, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), (max(2, int(0.42 * (box[2] - box[0]))), max(2, int(0.46 * (box[3] - box[1])))), 0, 0, 360, 1, -1)
        mask = mask | (core > 0)
    size_view = max(box[2] - box[0], box[3] - box[1])
    f = n / 2 / math.tan(math.radians(fov) / 2)
    size_pano = size_view * (W / 360.0) / (f * math.pi / 180.0)
    sig, dil = _sizes(size_pano, W)
    sc = _small_scale(W)
    pm = view_mask_to_pano(view, mask, H // sc, W // sc)
    out = smudge(pano, pm, sig, dil)
    rec = dict(kind=person["kind"], score=person["score"], az=person["az"], el=person["el"], deg=person["deg"], box_from=how, mask_from=mhow,
               sigma_px=round(float(sig), 1), dilate_px=round(float(dil), 1), mask_px=int(pm.sum()) * sc * sc)
    return out, rec


def is_walker(person, walker_small):
    """True when the person's centre lies in (or right next to) the walker's mask. walker_small: bool mask of the picture at any size, or None."""
    if walker_small is None:
        return False
    h, w = walker_small.shape
    r = int(np.clip((90 - person["el"]) / 180 * h, 0, h - 1)); c = int(((person["az"] + 180) % 360) / 360 * w) % w
    pad = max(2, int(0.06 * w))                                                      # (about 20 degrees: he stands 10 to 40 degrees off straight down, and is a big shape)
    r0, r1 = max(0, r - pad // 2), min(h, r + pad // 2 + 1)
    cols = np.arange(c - pad, c + pad + 1) % w
    return bool(walker_small[r0:r1][:, cols].any())


def smudge_people(pano, persons, det, models, walker_small=None, log=print):
    """Smudge every person of `persons` (scan() results; faces are the gate's blur_faces, not this) except the walker. Returns (picture, [records])."""
    recs = []
    for p in sorted([p for p in persons if p.get("kind") == "person" and not is_walker(p, walker_small)], key=lambda p: -p["deg"]):
        try:
            pano, rec = smudge_person(pano, p, det, models)
        except Exception as e:                                                       # a person who cannot be segmented is still smudged, as an ellipse where the detector saw him
            log(f"could not segment the person at {p['az']:+.0f},{p['el']:+.0f} ({e!r}); smudged as an ellipse")
            try:
                pano, rec = smudge_person(pano, p, det, models, fallback=True); rec["error"] = repr(e)[:160]
            except Exception as e2:
                rec = dict(kind="person", az=p["az"], el=p["el"], deg=p["deg"], error=repr(e2)[:160], failed=True)
        recs.append(rec)
    return pano, recs


def smudge_walker(pano, hole1024, up):
    """The capturer step's hole (the person holding the stick, a 1024 pixel straight-down view) smudged and mildly darkened in the picture. Returns the new picture."""
    import capturer as C
    H, W = pano.shape[:2]
    sc = _small_scale(W)
    h, w = H // sc, W // sc
    m, ok = C.view_to_pano((hole1024 > 0).astype(np.uint8) * 255, up, h, w)
    m = (m > 127) & ok
    if not m.any():
        return pano.copy()
    size = math.sqrt(float(m.sum())) * sc
    sig, dil = _sizes(size, W)
    return smudge(pano, m, sig, dil, darken=DARKEN_WALKER)
