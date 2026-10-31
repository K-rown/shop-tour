"""A margin on the privacy check: a stop that passed but would be held if its camera height were measured a little differently is held too.

Usage:  python privacy_margin.py [--frames frames] [--scales 0.93 1.07] [--only 005 007] [--dry-run]
Run by tour.py right after privacy_gate.py, in the same step (SHOP_TOUR_PRIVACY_MARGIN=0 turns it off).

Why: the 4b check of privacy_gate.py ("a person-sized shape is still near the camera") sizes shapes in metres from the stop's camera height
(poses.json cam_h, one measurement per stop, rounded to a centimetre). That height is not the same twice: the same walk mapped again gives
heights a few percent apart (measured on the recorded walk: a stop's tallest and shortest height over 4 runs differ by 2-10%, median 5.5%),
and a stop whose person-sized shape sits near the size or height threshold passes in one run and is held in the next. Nothing in the picture changed.
A privacy check that is held or not by luck is not one. So, for every stop that PASSED, the same check is run again with the camera height scaled
down and up by SCALES; a stop that would be held at either is held, with its own plain reason. The picture work (depth, SAM) does not depend on the
camera height, so it is done once per stop and only the sizing is repeated.

The check has a second kind of luck. It judges ONE shape: the best-scoring person-sized candidate of the ~135 SAM proposes, and which candidate
scores best moves with the camera height (measured: on one stop the best was a person-sized shape ON the filled area at x0.90 and x1.10 and another
one, off it, at x1.00 - and the stop passed). So at every height each person-sized candidate is also judged ON ITS OWN by the same check_stop
(handed that one candidate); a stop is held if any of them would be.

It only ever ADDS holds, and only on stops the gate passed:
  - a stop that is held stays held with its own reasons (nothing is removed or rewritten);
  - a stop a person already answered (kept in the tour or left out) is left exactly as they answered;
  - every field the app, manifest.py and build_tour.py read (held, reasons, persons, faces, kept, left_out) keeps its meaning; the added hold is
    held=True with a reason and a "margin" record {scale, cam_h, tried, problems, which} for anyone who wants to know why. It is reviewed like any
    other hold (python privacy_gate.py --keep 005 / --leave-out 005).
The stop's safe/ picture is removed (as the gate does for a held stop), and the review page is written again so the new holds are on it.
privacy_gate.py, qa/check_capturer.py and the detectors are not changed: check_stop is called exactly as the gate calls it.
It does not help where the gate's PICTURE differs between runs (the person removal paints a stop differently each time the map moves): a stop whose
picture is clean in this run is judged on this run's picture.
"""
import argparse, hashlib, json, os, sys

import privacy_answers as PA
import privacy_inputs as PI

HERE = os.path.dirname(os.path.abspath(__file__))
SCALES = (0.93, 1.07)                 # camera-height factors; 1.0 is always tried too. From the measured spread of cam_h across runs (see the commit message)
SCRIPT_VERSION = 3                    # 3: his answers survive a re-scan; the done marker (frames/privacy_margin.json). A new version makes tour.py run just the margin again on a walk it was made for


def enabled():
    """SHOP_TOUR_PRIVACY_MARGIN=0 (or off/no/false) turns the margin off. It is also off unless SHOP_TOUR_PEOPLE=remove: the margin only ever adds holds, and nothing is held otherwise."""
    import people_mode
    if not people_mode.holds():
        return False
    return os.environ.get("SHOP_TOUR_PRIVACY_MARGIN", "").strip().lower() not in ("0", "false", "no", "off")


def scales_from_env(default=SCALES):
    raw = os.environ.get("SHOP_TOUR_PRIVACY_MARGIN_SCALES", "").strip()
    if not raw:
        return tuple(default)
    try:
        out = tuple(float(x) for x in raw.replace(";", ",").replace(" ", ",").split(",") if x)
    except ValueError:
        return tuple(default)
    return out if out and all(0.5 <= s <= 1.5 for s in out) else tuple(default)


def direction_words(scale):
    pct = abs(scale - 1.0) * 100
    return f"{pct:.0f}% {'lower' if scale < 1 else 'higher'}" if pct >= 0.5 else "as measured"


def margin_reason(scale, problems, which="best"):
    """The plain reason a stop is held by the margin; problems are the gate's own words for what the check found."""
    what = "; ".join(problems)
    same = abs(scale - 1.0) < 0.005
    if which == "other":
        if same:
            return "held to be safe: the check found more than one person-sized shape in this picture and judged another one; this one would be held: " + what
        return ("held to be safe: the check found more than one person-sized shape in this picture, and which one it judges depends on a centimetre of "
                f"camera height (with the height {direction_words(scale)}, another shape would be held): " + what)
    if same:
        return "held to be safe: checking this stop a second time gave a different answer, and it would be held: " + what
    return f"held to be safe: with the camera height measured a little differently ({direction_words(scale)}) this stop would be held: " + what


def apply_margin(state, found, scales=SCALES):
    """Merge the margin's findings into the privacy state ({stop id: {...}}) IN PLACE. found: {stop id: [(scale, cam_h, problems[, which]), ...]}
    (an entry with no problems means the stop passed at that height). Only ever adds holds: a stop that is already held (or kept, or left out) is
    not touched, and a stop that has no entry in the state is not invented. Returns the ids newly held, in the order of `found`."""
    added = []
    for sid, hits in found.items():
        s = state.get(sid)
        if s is None or s.get("held") or s.get("kept") or s.get("left_out"):
            continue
        hits = [h for h in hits if h[2]]
        if not hits:
            continue
        scale, cam_h, problems, *rest = min(hits, key=lambda h: (abs(h[0] - 1.0), h[0]))       # the smallest change in height that holds it
        which = rest[0] if rest else "best"
        nbinfo = rest[1] if len(rest) > 1 else None
        s["held"] = True
        s["reasons"] = list(s.get("reasons") or []) + [margin_reason(scale, problems, which)]
        s["margin"] = dict(scale=round(float(scale), 3), cam_h=round(float(cam_h), 3), tried=[round(float(x), 3) for x in scales],
                           problems=list(problems), which=which)
        if nbinfo:
            s["margin"]["neighbours"] = nbinfo                                  # the shape was looked at in the neighbours' pictures and not agreed (neighbour_check.py)
        added.append(sid)
    return added


class Memo:
    """Models with the two pictures-only calls remembered for the stop being checked: the depth model and SAM do not depend on the camera
    height, so check_stop repeated with another height reuses them. Holds one entry each (the previous stop is dropped).
    only=[candidate]: segment_many hands check_stop just that one SAM candidate (to judge it on its own)."""
    def __init__(self, models):
        self._m = models; self._disp = (None, None); self._seg = (None, None); self.only = None

    @staticmethod
    def _key(v, *extra):
        return hashlib.blake2b(v.tobytes(), digest_size=16).digest() + repr(extra).encode()

    def disparity(self, v):
        k = self._key(v)
        if self._disp[0] != k:
            self._disp = (k, self._m.disparity(v))
        return self._disp[1].copy()

    def segment_many(self, v, pts, *a, **kw):
        k = self._key(v, tuple(pts), a, sorted(kw.items()))
        if self._seg[0] != k:
            self._seg = (k, self._m.segment_many(v, pts, *a, **kw))
        return self._seg[1] if self.only is None else list(self.only)

    def __getattr__(self, name):
        return getattr(self._m, name)


def distinct(cands, keep=0.9):
    """Candidates [(point index, mask, iou)] that are not almost the same mask as one already taken (intersection over union above `keep`)."""
    out = []
    for c in cands:
        m = c[1]
        if not any((m & o[1]).sum() > keep * (m | o[1]).sum() for o in out):
            out.append(c)
    return out


def sweep(frames, scales, only=None, ids=None, log=print, candidates=True, models=None, rep_cap=None, panos=None):
    """Run the gate's own 4b check (check_stop) on stops with the camera height times each of `scales` and, with candidates=True, on each
    person-sized SAM candidate alone. Returns {stop id: [(scale, cam_h used, problems, which), ...]}: one entry per scale tried (problems empty =
    it would pass; which is "best" or "other", the second for a candidate judged on its own). A stop is searched no further once something holds it.
    Inputs are read exactly as privacy_gate.py reads them."""
    import cv2
    import numpy as np
    sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "qa"))
    import capturer as C
    from imutil import imread
    from check_capturer import check_stop
    import neighbour_check
    nbr = neighbour_check.get_context(frames)                                                 # the same second opinion on check C the gate makes
    pts = json.load(open(os.path.join(frames, "points.json")))
    UP = json.load(open(os.path.join(frames, "up.json"))); PIN = PI.Inputs(frames, len(pts))
    if rep_cap is None:
        rep_cap = json.load(open(os.path.join(frames, "capturer", "report.json")))
    models = models if isinstance(models, Memo) else Memo(models or C.Models())             # (privacy_chain.py hands in the gate's: the depth and SAM of this picture are already worked out)
    out = {}
    for k, p in enumerate(pts):
        sid = p["id"]
        if (only and sid not in only) or (ids is not None and sid not in ids):
            continue
        cp = os.path.join(frames, "clean", p["filename"])
        mp = os.path.join(frames, "capturer", sid + ("_hole.png" if os.path.exists(os.path.join(frames, "capturer", sid + "_hole.png")) else "_mask.png"))
        if not os.path.exists(cp) or not os.path.exists(mp):
            continue                                                                          # the gate holds a stop with no cleaned picture already
        pano = (panos or {}).get(sid)                                                         # (the gate already has the cleaned picture in hand: privacy_chain.py passes it)
        if pano is None:
            pano = imread(cp)
        sp = os.path.join(frames, "capturer", sid + "_soft.png"); fp = os.path.join(frames, "capturer", sid + "_floor.png")
        hole = cv2.imread(mp, 0) > 0
        soft = (cv2.imread(sp, 0) > 0) if os.path.exists(sp) else None
        floor = (cv2.imread(fp, 0) > 127) if os.path.exists(fp) else None
        base = PIN.cam_h(k)
        leftover = rep_cap.get(sid, {}).get("leftover_m2")

        last = {}                                                                             # what the neighbours made of the shape that was held (neighbour_check.py), for the record

        def run(h, k=k, sid=sid, pano=pano, hole=hole, soft=soft, floor=floor, leftover=leftover):
            res = check_stop(models, sid, pano, UP[k], h, hole, soft, floor, leftover, nbr)[0]
            last["nb"] = res["info"].get("neighbour_check")
            return list(res["problems"])
        disp = cands = None; judged = set()
        out[sid] = []
        for s in scales:
            h = base * s
            models.only = None
            problems = run(h); which = "best"
            if candidates and not problems:               # the best-scoring shape passed: judge each person-sized candidate on its own
                if cands is None:                         # (this prefilter only chooses WHICH candidates are handed to check_stop, which decides)
                    v = C.level_view(pano, UP[k]); disp = cv2.medianBlur(models.disparity(v).astype(np.float32), 5)
                    cands = models.segment_many(v, C.prior_points(base))
                _, f = C.view_dirs()
                single = [c for c in cands if C.score_candidates([c], disp, h, h / f)[0] is not None]
                for c in distinct(single):
                    if id(c[1]) in judged:                # judged at another height and fine: whether it overlaps the filled area does not depend on the height
                        continue
                    models.only = [c]
                    problems = run(h); which = "other"
                    if problems:
                        break
                    judged.add(id(c[1]))
                models.only = None
            out[sid].append((s, h, problems, which, last.get("nb") if problems else None))
            if problems:
                break
        log(f"  {sid}: " + ", ".join(f"x{s:g} " + (("HELD" if w == "best" else "HELD (another shape)") if pr else "ok") for s, _, pr, w, *_ in out[sid]))
    return out


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def mode_word():
    return "on" if enabled() else "off"


def put_safe_pictures(frames, pts, state, rep, added):
    """Keep frames/safe/ in step with the verdicts: a stop that is held and not kept must not stay in it; a stop restored as kept gets its
    picture back (the faces blurred, exactly as privacy_gate.py --keep writes it) if it has none."""
    safe = os.path.join(frames, "safe")
    by_id = {p["id"]: p for p in pts}
    touched = set(added) | set(rep["left_out"]) | set(rep["reheld"]) | set(rep["margin"])
    for sid in touched:
        s, p = state.get(sid), by_id.get(sid)
        if p and s and s.get("held") and not s.get("kept") and os.path.exists(os.path.join(safe, p["filename"])):
            os.remove(os.path.join(safe, p["filename"]))                  # a stop that is held must not stay in the tour
    need = [sid for sid in rep["kept"] if sid in by_id and not os.path.exists(os.path.join(safe, by_id[sid]["filename"]))]
    if need:
        import cv2
        sys.path.insert(0, HERE)
        import privacy_gate as pg
        os.makedirs(safe, exist_ok=True)
        for sid in need:
            fn = by_id[sid]["filename"]
            pano = pg.imread(os.path.join(frames, "clean", fn))
            pg.imwrite(os.path.join(safe, fn), pg.blur_faces(pano, state[sid].get("faces") or []), [cv2.IMWRITE_JPEG_QUALITY, 95])


def scales_to_try(given=None):
    """The camera-height factors tried, 1.0 first, then the smallest change."""
    scales = tuple(given) if given else scales_from_env()
    return tuple(sorted(set(scales) | {1.0}, key=lambda s: (abs(s - 1.0), s)))


def main(argv=None, precomputed=None, models=None, rep_cap=None):
    """precomputed: {stop id: sweep() findings} already worked out (privacy_chain.py does it stop by stop while the pictures are still being made); a stop of the
    margin's list that is not in it is swept here."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--scales", nargs="*", type=float, default=None, help=f"camera-height factors to try (default {' '.join(str(s) for s in SCALES)}; 1.0 is always added)")
    ap.add_argument("--only", nargs="*", help="stop ids to check (default: every stop the gate passed)")
    ap.add_argument("--dry-run", action="store_true", help="say what would be held; write nothing")
    ap.add_argument("--snapshot", action="store_true", help="before privacy_gate.py runs again: keep a copy of the earlier privacy.json (his answers, the margin's holds) and do nothing else")
    a = ap.parse_args(argv)
    if a.snapshot:
        print("privacy snapshot:", PA.snapshot(a.frames, mode_word(), SCRIPT_VERSION))
        return
    pf = os.path.join(a.frames, "privacy.json")
    if not os.path.exists(pf):
        sys.exit(f"{pf} is missing: run  python privacy_gate.py  first")
    doc = json.load(open(pf)); state = doc.get("stops", {})
    before = json.dumps(doc, sort_keys=True)
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    prev = PA.load_snapshot(a.frames)                                          # the privacy.json from before the gate ran again (None: the gate did not run again)
    rep = PA.restore(prev, state)                                              # his answers first, so the margin does not sweep what he answered
    if prev:
        print(f"privacy answers restored after the re-scan: left out {len(rep['left_out'])}, kept {len(rep['kept'])}, held again for a fresh look {len(rep['reheld'])}, "
              f"margin holds {len(rep['margin'])}", flush=True)
    added = []
    if not enabled():
        print("privacy margin: off (SHOP_TOUR_PRIVACY_MARGIN=0); the stops keep the gate's answer as it is")
    else:
        scales = scales_to_try(a.scales)                                       # 1.0 first, then the smallest change
        todo = [p["id"] for p in pts if p["id"] in state and not state[p["id"]].get("held") and not state[p["id"]].get("kept") and not state[p["id"]].get("left_out")
                and (not a.only or p["id"] in a.only)]
        print(f"privacy margin: {len(todo)} stop(s) passed the check; trying the camera height x{', x'.join(f'{s:g}' for s in scales)} on each", flush=True)
        if todo:
            pre = precomputed or {}
            found = {sid: pre[sid] for sid in todo if sid in pre}
            rest = [sid for sid in todo if sid not in found]
            if rest:
                found.update(sweep(a.frames, scales, ids=set(rest), models=models, rep_cap=rep_cap))      # a crash here leaves no marker: tour.py runs the margin again
            found = {sid: found[sid] for sid in todo if sid in found}
            added = apply_margin(state, found, scales)
            for sid in added:
                print(f"{sid}: HELD (margin)  " + state[sid]["reasons"][-1], flush=True)
        rep2 = PA.restore(prev, state)                                          # again, now that the margin has added its holds: a kept stop must not lose its answer to them
        rep = {k: sorted(set(rep[k]) | set(rep2[k])) for k in rep}
        if added:
            doc["margin"] = dict(version=SCRIPT_VERSION, scales=list(scales), added=added)
    if a.dry_run:
        print(f"dry run: {len(added)} stop(s) would be held by the margin: {', '.join(added) or 'none'}")
        return
    changed = json.dumps(doc, sort_keys=True) != before
    if changed:
        write_json(pf, doc)
        put_safe_pictures(a.frames, pts, state, rep, added)
        try:
            sys.path.insert(0, HERE)
            import privacy_gate
            page = privacy_gate.review_page(a.frames, pts, state, None)
        except Exception as e:                                                # the verdict is written; the review page is a courtesy
            page = f"(review page not rewritten: {e})"
        held = [p["id"] for p in pts if state[p["id"]]["held"] and not state[p["id"]].get("kept")]
        if added:
            print(f"\nthe margin held back {len(added)} more: {', '.join(added)}")
        print(f"{len(pts) - len(held)} of {len(pts)} stops go into the tour; {len(held)} held back: {', '.join(held)}")
        print("review page:", page)
    elif enabled():
        print("privacy margin: no stop is held by the margin")
    PA.write_marker(a.frames, mode_word(), SCRIPT_VERSION)                    # the LAST act: the step is finished only now
    PA.drop_snapshot(a.frames)


if __name__ == "__main__":
    main()
