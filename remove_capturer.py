"""Remove the person holding the stick from every stop, and rebuild the floor under them from the neighbouring stops.

Usage:  python remove_capturer.py [--frames frames] [--only 010 017] [--masks-only] [--sheet qa/person_removal.jpg]
Reads:  frames/points/<id>.jpg, frames/up.json, frames/hops.json (nothing from vio, the map or the floor step: see privacy_inputs.py)
Writes: frames/capturer/<id>_mask.png (his mask in the straight-down view), frames/capturer/report.json, frames/clean/<id>.jpg (the pano without him)
        and the before/after contact sheet. build_tour.py uses frames/clean/ when it exists.
How it works: see capturer.py. Models (both Apache-2.0): Depth Anything V2 Small, SAM 2.1 small. No person detector is involved.
"""
import argparse, collections, functools, json, os, sys, time
import cv2
import numpy as np
import capturer as C
import privacy_inputs as PI
from imutil import imread, imwrite, parallel_jobs

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


_LOADED = collections.OrderedDict()
LOADED_MAX = 8                                  # stops' pictures kept by a process that makes floors (88 MB each)


def _morph_helper(req, replies):
    """The process that makes the workers' big dilations (MorphServer): one card context for all of them, and nothing else to do, so a request is answered in the time the card needs."""
    import torch
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
    n, secs = 0, 0.0
    while True:
        item = req.get()
        if item is None:
            break
        wid, img, r, dilate = item
        t = time.time()
        try:
            out = C.morph_ellipse(img, r, dilate)
        except Exception as e:                                      # (the worker does it itself)
            out = e
        n += 1; secs += time.time() - t
        replies[wid].put(out)
    print(f"[fill] {n} big dilations made on the video card for the workers ({secs:.1f} s)", flush=True)


class MorphServer:
    """The big dilations of the floor fill, made on the video card for the fill workers by ONE helper process. A card context in every worker would take 640 MB of the card each
    (the card is shared with the other steps), and a thread of the chain's own process answers late (its card calls queue behind the models' and its Python threads are busy);
    here the worker sends the picture and the radius and gets back the same picture cv2 would have made (capturer.morph_ellipse). Anything that goes wrong - no card, a reply that
    does not come in WAIT_S - and the worker does the dilation itself."""
    WAIT_S = 60.0

    def __init__(self, ctx, jobs):
        self.req = ctx.Queue(); self.replies = [ctx.Queue() for _ in range(jobs)]; self.next = ctx.Value("i", 0)
        self.proc = ctx.Process(target=_morph_helper, args=(self.req, self.replies), daemon=True); self.proc.start()

    def init_args(self):
        return (self.req, self.replies, self.next)

    def close(self):
        self.req.put(None); self.proc.join(30)
        if self.proc.is_alive():
            self.proc.terminate()


def _remote_morph(req, replies, wid):
    def ask(img, r, dilate):
        req.put((wid, np.ascontiguousarray(img), int(r), bool(dilate)))
        out = replies[wid].get(timeout=MorphServer.WAIT_S)
        if isinstance(out, Exception):
            raise out
        return out
    return ask


def _fill_worker(threads, server=None):
    if server is not None:                      # (the big dilations are made by the main process on the video card: MorphServer)
        req, replies, counter = server
        with counter.get_lock():
            wid = counter.value; counter.value += 1
        C.REMOTE_MORPH[0] = _remote_morph(req, replies, wid)
    cv2.setNumThreads(threads)                  # the stops are made side by side: each process taking every core only slows the others (and the process that checks them)


def fill_one(frames, k):
    """The floor for stop k (in its own process or in a loop): writes frames/clean/<id>.jpg and the masks of the fill; returns (k, report fields, the line to print)."""
    pts = json.load(open(os.path.join(frames, "points.json"))); UP = json.load(open(os.path.join(frames, "up.json"))); P = PI.Inputs(frames, len(pts))
    ids = [p["id"] for p in pts]
    cdir = os.path.join(frames, "capturer")
    class Masks(dict):                                           # (a stop's mask is read when it is first needed: the stop itself and its neighbours)
        def __missing__(self, j):
            self[j] = m = cv2.imread(os.path.join(cdir, ids[j] + "_mask.png"), 0) > 0
            return m
    masks = Masks()

    def pano_of(j):
        return load(j)[0]

    def load(j):
        # (kept for the next stop this process makes: its neighbours are mostly the ones this stop used. A stop's own mask file is part of the key.)
        mp_j = os.path.join(cdir, ids[j] + "_mask.png")
        key = (frames, j, os.path.getmtime(mp_j))
        if key in _LOADED:
            _LOADED.move_to_end(key)
            return _LOADED[key]
        _LOADED[key] = got = (imread(os.path.join(frames, "points", pts[j]["filename"])), UP[j], P.cam_h(j), masks[j] if masks[j].any() else None,
                              None)                          # (the floor step's mask of a neighbour was read here and never used - fill_stop says why - so the fill no longer waits for that step)
        while len(_LOADED) > LOADED_MAX:
            _LOADED.popitem(last=False)
        return got

    if not masks[k].any():                                      # nobody under the camera was found (a walk recorded with the stick held clear of the lens, or a miss):
        pano = pano_of(k)                                       # there is nothing to paint out, so the picture goes on as it is. privacy_gate.py still runs a person and a
        imwrite(os.path.join(frames, "clean", pts[k]["filename"]), pano, [cv2.IMWRITE_JPEG_QUALITY, 95])     # face detector over all of it, so a person this step missed holds the stop back
        return k, dict(status="no capturer found: picture kept as it is", mask_px_pano=0, hole_m2=0.0, real_pct=0.0, blur_pct=0.0, grown_m2=0.0, neighbours=[]),             f"{ids[k]} no capturer found: picture kept as it is (the privacy step still checks it)"
    t = time.time()
    pano = pano_of(k)
    import people_mode
    if not people_mode.remove():                                # SHOP_TOUR_PEOPLE=blur (the default) or keep: the person holding the stick is smudged where he stood, not painted out (people_blur.py)
        import people_blur
        new = people_blur.smudge_walker(pano, masks[k], UP[k])
        imwrite(os.path.join(frames, "clean", pts[k]["filename"]), new, [cv2.IMWRITE_JPEG_QUALITY, 95])
        px = int(((new.astype(np.int16) - pano.astype(np.int16)) != 0).any(-1).sum())
        return k, dict(status="smudged", mask_px_pano=px, hole_m2=0.0, real_pct=0.0, blur_pct=100.0, grown_m2=0.0, neighbours=[]), f"{ids[k]} smudged where the walker stood ({time.time() - t:.1f}s)"
    new, st, dbg = C.fill_stop(k, pano, UP[k], P.cam_h(k), masks[k], P.neighbours(k), load)
    imwrite(os.path.join(frames, "clean", pts[k]["filename"]), new, [cv2.IMWRITE_JPEG_QUALITY, 95])
    st["mask_px_pano"] = int(((new.astype(np.int16) - pano.astype(np.int16)) != 0).any(-1).sum())
    fl0 = dbg["fill"]; cv_ = np.zeros((C.FILL_N, C.FILL_N), np.uint8); cv_[fl0.y0:fl0.y1, fl0.x0:fl0.x1] = fl0.hole.astype(np.uint8) * 255
    cv2.imwrite(os.path.join(cdir, ids[k] + "_hole.png"), cv2.resize(cv_, (C.VIEW_N, C.VIEW_N), interpolation=cv2.INTER_AREA))         # the hole after it grew over what was left of him
    fl = dbg["fill"]; canvas = np.zeros((C.FILL_N, C.FILL_N), np.uint8)
    canvas[fl.y0:fl.y1, fl.x0:fl.x1] = (dbg["hole"] & ~dbg["got"]).astype(np.uint8) * 255      # the part no neighbour could supply (a soft fill)
    cv2.imwrite(os.path.join(cdir, ids[k] + "_soft.png"), cv2.resize(canvas, (C.VIEW_N, C.VIEW_N), interpolation=cv2.INTER_AREA))
    canvas[:] = 0; canvas[fl.y0:fl.y1, fl.x0:fl.x1] = dbg["floor_ref"].astype(np.uint8) * 255           # floor the neighbours vouch for: the 4b check measures against it
    cv2.imwrite(os.path.join(cdir, ids[k] + "_floor.png"), cv2.resize(canvas, (C.VIEW_N, C.VIEW_N), interpolation=cv2.INTER_AREA))
    st["status"] = "filled"
    return k, st, (f"{ids[k]} real {st['real_pct']}%  blurred {st['blur_pct']}%  hole {st['hole_m2']} m2  neighbours used "
                   f"{[(n['j'] + 1, n['used']) for n in st['neighbours'] if n.get('used')]}  ({time.time() - t:.1f}s)")


def person_hint_fn(get_det):
    """A hint_fn for capturer.find_mask: where the detector (get_det() gives the privacy_gate.Detectors) sees a person within 50 degrees of straight down (centre of its box,
    in the straight-down view's pixels): the best one, or None. It looks at the straight-down view at 130 and at 100 degrees (a person far below the lens shows up better in the narrower one)."""
    import math

    def person_hint(view, pano, up):
        det = get_det()
        d, f = C.view_dirs()
        f130 = C.VIEW_N / 2 / math.tan(math.radians(C.VIEW_FOV) / 2); f100 = C.VIEW_N / 2 / math.tan(math.radians(50.0))
        found = [(sc, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) for sc, b in det.persons(view)]
        v100 = C.level_view(pano, up, C.VIEW_N, 100.0)
        for sc, b in det.persons(v100):
            c = C.VIEW_N / 2
            found.append((sc, (c + ((b[0] + b[2]) / 2 - c) * f130 / f100, c + ((b[1] + b[3]) / 2 - c) * f130 / f100)))
        best = None
        for sc, (cx, cy) in found:
            cx, cy = int(cx), int(cy)
            if 0 <= cx < C.VIEW_N and 0 <= cy < C.VIEW_N and np.degrees(np.arccos(np.clip(-d[cy, cx, 1], -1, 1))) < 50 and (best is None or sc > best[0]):
                best = (sc, (cx, cy))
        return None if best is None else best[1]
    return person_hint


BOX_FOVS = (130.0, 100.0, 75.0, 55.0)       # the straight-down view cut at these fields of view: a person seen from above and crouching is small and grey, and the detector
                                            # finds him only in the narrower (bigger) views (walk 2: stops 003/031 at 55-75 degrees, 023 at 100; 130 found none of them)
BOX_MIN_SCORE = 0.5                         # a box anywhere in the cone
BOX_NEAR_DEG, BOX_NEAR_SCORE = 24.0, 0.25   # ... and within this many degrees of straight down, where the capturer stands, a weaker one counts too


def _persons_below(det, views, thr):
    """The detector's person boxes (score, [x0, y0, x1, y1]) in each of the views, at a threshold chosen here (Detectors.persons uses the gate's 0.5)."""
    from PIL import Image
    t = det.torch
    x = det.proc(images=[Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)) for v in views], return_tensors="pt").to(det.dev)
    with t.no_grad():
        o = det.model(**x)
    res = det.proc.post_process_object_detection(o, threshold=thr, target_sizes=t.tensor([list(v.shape[:2]) for v in views]))
    return [[(float(s), [float(v) for v in b]) for s, l, b in zip(r["scores"].cpu().numpy(), r["labels"].cpu().numpy(), r["boxes"].cpu().numpy()) if int(l) == 0] for r in res]


SIDE_VIEWS = [(yaw, -40.0) for yaw in (0, 90, 180, 270)]      # the gate's own pitch -40 views (privacy_gate.VIEWS): the ones that look down at the floor round the camera


@functools.lru_cache(maxsize=8)
def _side_rays(yaw, pitch, fov, n):
    """depth_panos.view_rays for a side view, worked out once (it is the same four views for every stop; the array is only read)."""
    from depth_panos import view_rays
    return view_rays(yaw, pitch, fov, n)


def _boxes_from_side_views(det, pano, up, f130):
    """The person boxes of the gate's four pitch -40 views, drawn into the straight-down view (130 degrees): a person standing 25-40 degrees from the nadir is seen
    sideways-on there, upright, where the straight-down views show him foreshortened (walk 2 stop 030: only these views find him, as the gate does). Each box is
    projected by its corners and edge midpoints; one with a corner at or above the horizon of the straight-down view is left out."""
    import privacy_gate as pg
    n = C.VIEW_N
    R = C.rot_to_y(up)
    views = [pg.cut(pano, yaw, pitch) for yaw, pitch in SIDE_VIEWS]
    out = []
    people = _persons_below(det, views, BOX_NEAR_SCORE)
    for (yaw, pitch), v, res in zip(SIDE_VIEWS, views, people):
        res = list(res)
        for fs, (fx, fy, fw, fh) in det.faces(v):                 # a face is a person too (the gate blurs it, and a face left in the picture is what he would see first)
            px, py = 0.25 * fw, 0.25 * fh
            res.append((fs - 0.1, [fx - px, fy - py, fx + fw + px, fy + fh + py]))
        if not res:
            continue
        rays = _side_rays(float(yaw), float(pitch), pg.VIEW_FOV, pg.VIEW_N)
        for sc, b in res:
            xs = (b[0], (b[0] + b[2]) / 2, b[2]); ys = (b[1], (b[1] + b[3]) / 2, b[3])
            pts = np.array([rays[int(np.clip(y, 0, pg.VIEW_N - 1)), int(np.clip(x, 0, pg.VIEW_N - 1))] for x in xs for y in ys]) @ R.T          # pano rays -> the level frame
            if (-pts[:, 1] < 0.2).any():
                continue
            col = f130 * pts[:, 0] / -pts[:, 1] + n / 2 - 0.5; row = n / 2 - 0.5 - f130 * pts[:, 2] / -pts[:, 1]
            out.append((sc, [float(col.min()), float(row.min()), float(col.max()), float(row.max())]))
    return out


def person_boxes_fn(get_det):
    """A boxes_fn for capturer.find_mask: the person boxes the detector (get_det() gives the privacy_gate.Detectors) draws near straight down, best first, in the
    straight-down view's pixels (130 degrees, what find_mask works in). It looks at the straight-down view cut at several fields of view (BOX_FOVS) and converts each
    box back. A box counts at BOX_MIN_SCORE anywhere within 55 degrees of straight down, and at BOX_NEAR_SCORE when its centre is within BOX_NEAR_DEG of it (the
    capturer stands right under the camera, so a weak detection there is far more likely him than a stranger). Overlapping boxes of one person are one."""
    import math

    def person_boxes(view, pano, up):
        det = get_det()
        d, _ = C.view_dirs()
        n = C.VIEW_N; c = n / 2
        f130 = n / 2 / math.tan(math.radians(C.VIEW_FOV) / 2)
        views = [view if fv == C.VIEW_FOV else C.level_view(pano, up, n, fv) for fv in BOX_FOVS]
        found = []
        for fv, res in zip(BOX_FOVS, _persons_below(det, views, BOX_NEAR_SCORE)):
            fx = n / 2 / math.tan(math.radians(fv) / 2); k = f130 / fx
            for sc, b in res:
                found.append((sc, [c + (b[0] - c) * k, c + (b[1] - c) * k, c + (b[2] - c) * k, c + (b[3] - c) * k]))
        for fv, v in zip(BOX_FOVS, views):
            fx_ = n / 2 / math.tan(math.radians(fv) / 2); k = f130 / fx_
            for fs, (fx, fy, fw, fh) in det.faces(v):
                px, py = 0.25 * fw, 0.25 * fh
                b = [fx - px, fy - py, fx + fw + px, fy + fh + py]
                found.append((fs - 0.1, [c + (b[0] - c) * k, c + (b[1] - c) * k, c + (b[2] - c) * k, c + (b[3] - c) * k]))
        found += _boxes_from_side_views(det, pano, up, f130)           # (he can stand 30 degrees out, where the straight-down views see him edge-on)
        out = []
        for sc, b in sorted(found, key=lambda t: -t[0]):
            cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
            if not (0 <= cx < n and 0 <= cy < n):
                continue
            deg = np.degrees(np.arccos(np.clip(-d[cy, cx, 1], -1, 1)))
            if deg >= 55 or sc < (BOX_NEAR_SCORE if deg < BOX_NEAR_DEG else BOX_MIN_SCORE):
                continue
            b = [max(0.0, b[0]), max(0.0, b[1]), min(n - 1.0, b[2]), min(n - 1.0, b[3])]
            dup = False
            for _, o in out:
                ix = max(0.0, min(b[2], o[2]) - max(b[0], o[0])) * max(0.0, min(b[3], o[3]) - max(b[1], o[1]))
                if ix > 0.5 * min((b[2] - b[0]) * (b[3] - b[1]), (o[2] - o[0]) * (o[3] - o[1])):
                    dup = True; break
            if not dup:
                out.append((sc, b))
        return out
    def verify(view, piece):
        """Does the detector, shown just this piece of the picture (its bounding box with room round it, blown up), call it a person (0.3 or better, its box over the piece)?"""
        ys, xs = np.nonzero(piece)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        half = max(80, int(0.9 * max(x1 - x0, y1 - y0)))
        X0, Y0 = max(0, cx - half), max(0, cy - half); X1, Y1 = min(view.shape[1], cx + half), min(view.shape[0], cy + half)
        crop = cv2.resize(view[Y0:Y1, X0:X1], (512, 512), interpolation=cv2.INTER_CUBIC)
        k = 512.0 / max(X1 - X0, 1)
        for sc, b in _persons_below(get_det(), [crop], 0.3)[0]:
            bx, by = (b[0] + b[2]) / 2 / k + X0, (b[1] + b[3]) / 2 / k + Y0
            if x0 - 0.3 * (x1 - x0) <= bx <= x1 + 0.3 * (x1 - x0) and y0 - 0.3 * (y1 - y0) <= by <= y1 + 0.3 * (y1 - y0):
                return True
        return False
    person_boxes.verify = verify
    return person_boxes


def mask_threads():
    """Stops whose masks are made at once (SHOP_TOUR_MASK_THREADS; 1 = one after the other, the old way). The masks do not read each other: the CPU work of one stop (the
    pixel maps, the detector's pre- and post-processing, the face detector, numpy) goes on while the video card works for the other. Default 2 when the masks are what the
    step waits for; 1 in a tour that began the step early (stepsync), where the masks are made long before the floors can begin and a second thread would hold 0.7 GB more
    of a card the depth step is sharing (measured: the chain held 3.16 GB with one thread, 3.90 GB with two)."""
    try:
        n = int(os.environ.get("SHOP_TOUR_MASK_THREADS", "") or 0)
    except ValueError:
        n = 0
    if n > 0:
        return n
    import stepsync
    return 1 if stepsync.enabled() else 2


def _one_mask(frames, pts, UP, P, ids, cdir, k, models, hint, boxes, rep):
    """The mask of stop k, written to frames/capturer/<id>_mask.png, and its entry in rep."""
    mp = os.path.join(cdir, ids[k] + "_mask.png")
    t = time.time()
    pano = imread(os.path.join(frames, "points", pts[k]["filename"])); h = P.cam_h(k)
    m, dbg = C.find_mask(models, pano, UP[k], h, hint_fn=hint, boxes_fn=boxes)
    if m is None:
        rep[ids[k]] = dict(status="NO CAPTURER FOUND", passed=dbg["passed"])
        cv2.imwrite(mp, np.zeros((C.VIEW_N, C.VIEW_N), np.uint8)); print(ids[k], "NO CAPTURER FOUND (held for review)", flush=True)
        return
    hole = final_hole(dbg["view"], m, dbg["height"], h, dbg["mpp"])
    cv2.imwrite(mp, hole.astype(np.uint8) * 255)
    rep[ids[k]] = {**rep.get(ids[k], {}), **dict(status="masked", fallback=dbg.get("fallback"), mask_m2=round(float(hole.sum()) * dbg["mpp"] ** 2, 2), mask_px=int(hole.sum()), score=round(float(dbg["score"]), 2))}       # (a new entry, never changed in place: report.json is written from another thread while the masks are made)
    print(ids[k], f"mask {rep[ids[k]]['mask_m2']} m2  ({time.time() - t:.1f}s)", flush=True)


def mask_stage(frames, pts, UP, P, todo, rep, get_models, get_det, reuse_masks=False, on_mask=None, threads=None):
    """1) the masks of the stops (every stop first: a neighbour's mask is needed to know where its own person stands). get_models() gives the capturer.Models
    (loaded when the first mask has to be made: reusing masks needs no model); get_det() the detectors for the fallback hint. rep is filled in place.
    on_mask(k) is called when the mask file of stop k is there (made now or kept) - one call at a time, in the order the masks are ready (the stops' order when
    one mask is made at a time). threads: masks made at once (default mask_threads())."""
    ids = [p["id"] for p in pts]
    cdir = os.path.join(frames, "capturer")
    need = [k for k in range(len(ids)) if (k in todo and not (reuse_masks and os.path.exists(os.path.join(cdir, ids[k] + "_mask.png")))) or not os.path.exists(os.path.join(cdir, ids[k] + "_mask.png"))]
    n_th = mask_threads() if threads is None else max(1, int(threads))
    if n_th <= 1 or len(need) <= 1:
        hint = person_hint_fn(get_det); boxes = person_boxes_fn(get_det)
        for k in range(len(ids)):
            if k in need:
                _one_mask(frames, pts, UP, P, ids, cdir, k, get_models(), hint, boxes, rep)
            if on_mask:
                on_mask(k)
        return
    import copy, threading
    from concurrent.futures import ThreadPoolExecutor
    lk = threading.Lock(); mine = threading.local()
    base = get_models(); base = getattr(base, "_m", base)                 # (the models themselves: a Memo remembers one picture at a time, and each thread works on its own)

    def notify(k):
        if on_mask:
            with lk:
                on_mask(k)

    def work(k):
        if not hasattr(mine, "models"):
            mine.models = copy.copy(base)                                  # (the networks are shared; the picture embedding SAM keeps is not)
            d = get_det(); mine.det = d.clone() if hasattr(d, "clone") else d      # (the face detector is not safe to share)
            mine.hint = person_hint_fn(lambda: mine.det); mine.boxes = person_boxes_fn(lambda: mine.det)
        _one_mask(frames, pts, UP, P, ids, cdir, k, mine.models, mine.hint, mine.boxes, rep)
        notify(k)
    for k in range(len(ids)):                                              # (the masks that are there already first: the floors that need only those can begin)
        if k not in need:
            notify(k)
    with ThreadPoolExecutor(max_workers=min(n_th, len(need))) as ex:
        for f in [ex.submit(work, k) for k in need]:
            f.result()


def fill_jobs():
    """Floors made at once, each in a process of its own: SHOP_TOUR_FILL_JOBS, else what the RAM allows, at most 4."""
    try:
        n = int(os.environ.get("SHOP_TOUR_FILL_JOBS", "") or 0)
    except ValueError:
        n = 0
    return n if n > 0 else parallel_jobs(2.5, 4, 4.0)


class FillPool:
    """2) the floor: every stop on its own (it reads the neighbours' ORIGINAL pictures, never a cleaned one), so the stops run side by side in separate processes.
    submit(k) when the masks stop k needs are there; on_done(k) is called (in a thread of this process) as each stop's picture is written - privacy_chain.py checks the
    stop then, while the others are still being made. close() waits for all of them and raises what went wrong."""
    def __init__(self, frames, ids, rep, rep_path, on_done=None, jobs=None):
        import threading
        self.frames, self.ids, self.rep, self.rep_path, self.on_done = frames, ids, rep, rep_path, on_done
        self.lock = threading.Lock(); self.errors = []; self.futs = []
        self.jobs = jobs if jobs is not None else (fill_jobs() if len(ids) > 2 else 1)
        self.ex = None
        self.server = None
        if self.jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            ctx = multiprocessing.get_context("spawn")
            if C._gpu_morph_wanted():                          # this process has the card: a helper makes the workers' big dilations (MorphServer)
                try:
                    self.server = MorphServer(ctx, self.jobs)
                except Exception as e:                         # (no helper: the workers do their dilations themselves, as before)
                    print(f"[fill] no card helper for the workers ({type(e).__name__}): they do their dilations themselves", flush=True)
            self.ex = ProcessPoolExecutor(max_workers=self.jobs, mp_context=ctx, initializer=_fill_worker,
                                          initargs=(max(2, (os.cpu_count() or 8) // (self.jobs + 1)), self.server.init_args() if self.server else None))

    def _done(self, k, upd, msg):
        with self.lock:
            self.rep.setdefault(self.ids[k], {}).update(upd); print(msg, flush=True)
            json.dump(dict(self.rep), open(self.rep_path, "w"), indent=1)
        if self.on_done:
            self.on_done(k)

    def _finished(self, fu):
        try:
            self._done(*fu.result())
        except BaseException as e:
            self.errors.append(e)

    def submit(self, k):
        if self.ex is None:
            self._done(*fill_one(self.frames, k))
        else:
            fu = self.ex.submit(fill_one, self.frames, k); self.futs.append(fu); fu.add_done_callback(self._finished)

    def close(self):
        if self.ex is not None:
            self.ex.shutdown(wait=True)
        if self.server is not None:
            self.server.close()
        if self.errors:
            raise self.errors[0]


def fill_stage(frames, todo, ids, rep, rep_path, on_done=None):
    pool = FillPool(frames, ids, rep, rep_path, on_done, jobs=None if len(todo) > 2 else 1)
    for k in todo:
        pool.submit(k)
    pool.close()


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
    UP = json.load(open(os.path.join(a.frames, "up.json"))); P = PI.Inputs(a.frames, len(pts))
    ids = [p["id"] for p in pts]
    todo = [k for k, i in enumerate(ids) if not a.only or i in a.only]
    cdir = os.path.join(a.frames, "capturer"); os.makedirs(cdir, exist_ok=True)
    os.makedirs(os.path.join(a.frames, "clean"), exist_ok=True)
    rep_path = os.path.join(cdir, "report.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    if a.sheet_only:
        make_sheet(a.frames, pts, UP, a.sheet); return
    box = {}

    def get_models():
        if "m" not in box:
            box["m"] = C.Models()                               # loaded when the first mask has to be made (reusing masks needs no model)
        return box["m"]

    def get_det():
        if "d" not in box:
            import privacy_gate as pg
            box["d"] = pg.Detectors()
        return box["d"]
    mask_stage(a.frames, pts, UP, P, todo, rep, get_models, get_det, a.reuse_masks)
    json.dump(rep, open(rep_path, "w"), indent=1)
    if a.masks_only:
        return
    fill_stage(a.frames, todo, ids, rep, rep_path)
    import remove_others as RO
    if RO.enabled():                                            # unless SHOP_TOUR_REMOVE_OTHERS=0: the other people too (the gate then looks again at what is left)
        RO.stage(a.frames, pts, UP, P, todo, get_det(), get_models())
    if a.sheet:
        make_sheet(a.frames, pts, UP, a.sheet)


if __name__ == "__main__":
    main()
