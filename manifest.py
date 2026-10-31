"""manifest.json: the stable hand-off from the engine to anything that wants to open or import a built tour.

  python manifest.py [--frames frames] [--tour tour]     # (re)writes tour/manifest.json; tour.py does this at the end of every run

The contract is documented in docs/GUIDE.md ("manifest.json"). Only numbers and stop ids go in it: no pictures, no paths, no names of people.
"""
import argparse, datetime, json, os, re, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_VERSION = 1
ENGINE_NAME = "shop-tour"
ENGINE_VERSION = "1.0"          # bump when the tour folder or this file changes shape
MEASURE_MAX_PCT = 10.0          # the viewer hides Measure above this (build_tour.py / viewer/index.html)


def _commit():
    try:
        r = subprocess.run(["git", "-C", HERE, "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def _size(folder):
    total = 0
    for root, _, files in os.walk(folder):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def privacy_block(frames, stop_ids):
    """all_clear: nothing was held, or every held stop was looked at (kept in the tour, or left out on purpose).
    held: at least one held stop has not been looked at yet. A held stop is never inside tour/, whatever the status."""
    pf = os.path.join(frames, "privacy.json")
    if not os.path.exists(pf):
        return {"status": "held", "checked": 0, "held": [], "kept": [], "left_out": [], "note": "the privacy check has not run"}
    pdoc = json.load(open(pf)); state = pdoc.get("stops", {})
    pmode = pdoc.get("people_mode", "remove")                                        # privacy.json from before SHOP_TOUR_PEOPLE has none: that was the paint-out and hold flow
    held, kept, left = [], [], []
    for sid, s in sorted(state.items()):
        if pmode != "remove":
            break                                                                    # SHOP_TOUR_PEOPLE=blur / keep: people stay in the picture and nothing is held
        if not s.get("held"):
            continue
        if s.get("kept"):
            kept.append(sid)
        elif s.get("left_out"):
            left.append(sid)
        else:
            held.append({"id": sid, "reasons": [str(x) for x in s.get("reasons", [])]})
    # a stop that is in the tour but is not clear and not kept would be a bug in the build; say so rather than hide it
    for sid in stop_ids:
        s = state.get(sid)
        if s is None or (s.get("held") and not s.get("kept")):
            held.append({"id": sid, "reasons": ["in the tour but not cleared by the privacy check"]})
    out = {"status": "held" if held else "all_clear", "checked": len(state), "held": held, "kept": kept, "left_out": left}
    if pmode != "remove":                                                            # additive keys (a reader that does not know them ignores them)
        out["people_mode"] = pmode
        out["people_blurred"] = sum(int(s.get("people_blurred", 0)) for s in state.values())
    return out


def pii_block(frames, stop_ids):
    """What the blur step did to the pictures in the tour: counts by kind (plates, faces, printed names, other text), how many stops had anything blurred, and the stops it could not check
    (those ship as they were). status: "ok", "partial" (some stop could not be checked), "off" (SHOP_TOUR_PII_BLUR=0) or "not_run"."""
    try:
        pj = json.load(open(os.path.join(frames, "pii.json")))
    except Exception:
        return {"status": "not_run", "blurred": {"plate": 0, "face": 0, "name": 0, "text": 0}, "stops_blurred": 0, "errors": []}
    if pj.get("enabled", True) is False:
        return {"status": "off", "blurred": {"plate": 0, "face": 0, "name": 0, "text": 0}, "stops_blurred": 0, "errors": []}
    st = {sid: r for sid, r in pj.get("stops", {}).items() if sid in set(stop_ids)}
    tot = {k: sum(int(r.get("counts", {}).get(k, 0)) for r in st.values()) for k in ("plate", "face", "name", "text")}
    errs = sorted(sid for sid in stop_ids if sid not in st or st[sid].get("error"))
    return {"status": "partial" if errs else "ok", "blurred": tot, "stops_blurred": sum(1 for r in st.values() if any(r.get("counts", {}).values())), "errors": errs}


def walked_on(frames):
    """The day the walk was recorded (YYYY-MM-DD), from the camera's own file name VID_YYYYMMDD_HHMMSS_...insv (frames/source.json); None if unknown."""
    try:
        name = json.load(open(os.path.join(frames, "source.json"))).get("name", "")
    except Exception:
        return None
    g = re.match(r"VID_(\d{4})(\d{2})(\d{2})_", name or "")
    return f"{g.group(1)}-{g.group(2)}-{g.group(3)}" if g else None


def measure_block(tj):
    pct = tj.get("measure_pct")
    if isinstance(pct, (int, float)) and pct <= MEASURE_MAX_PCT:
        return {"status": "shown", "error_pct": float(pct), "max_pct": MEASURE_MAX_PCT, "source": "graded against tape on this walk"}
    if isinstance(pct, (int, float)):
        return {"status": "hidden", "error_pct": float(pct), "max_pct": MEASURE_MAX_PCT, "source": "graded against tape on this walk"}
    return {"status": "hidden", "error_pct": None, "max_pct": MEASURE_MAX_PCT, "source": "not checked on this walk"}   # nobody has measured how far Measure can be trusted here: it is not offered


def build(frames, tour, now=None):
    tj = json.load(open(os.path.join(tour, "tour.json")))
    ids = [n["id"] for n in tj.get("nodes", [])]
    return {
        "manifest_version": MANIFEST_VERSION,
        "engine": {"name": ENGINE_NAME, "version": ENGINE_VERSION, "commit": _commit()},
        "created": (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "title": tj.get("title"),
        "walked": walked_on(frames),
        "stops": {"count": len(ids), "ids": ids},
        "size_bytes": _size(tour),
        "privacy": privacy_block(frames, ids),
        "pii": pii_block(frames, ids),
        "measure": measure_block(tj),
        **({"partial": True} if tj.get("partial") else {}),          # additive: a first tour made before the 3D steps; a reader that does not know the key ignores it
        "features": {"dollhouse": bool(tj.get("dollhouse")), "plan": bool(tj.get("plan")), "notes": bool(tj.get("notes"))},
    }


def write_all_held(frames, tour, title=None):
    """Every stop was held back by the privacy check, so no tour can be built. The review is still waiting for a person, and the app can only offer it
    when it finds a manifest: write tour.json as an empty placeholder (no stops, held_all: true; tour.py does not call a build "up to date" because of it)
    and a manifest whose privacy block lists every held stop (status "held"), so a reader refuses the walk as held and shows the review."""
    os.makedirs(tour, exist_ok=True)
    tp = os.path.join(tour, "tour.json")
    json.dump({"title": title, "nodes": [], "held_all": True}, open(tp + ".tmp", "w"), indent=1)
    os.replace(tp + ".tmp", tp)
    return write(frames, tour)


def write(frames, tour):
    m = build(frames, tour)
    p = os.path.join(tour, "manifest.json")
    json.dump(m, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)                    # size_bytes was counted before this file existed; it is the tour without the manifest
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--tour", default="tour")
    a = ap.parse_args()
    m = write(a.frames, a.tour)
    p = m["privacy"]
    print(f"manifest.json: {m['stops']['count']} stops, {m['size_bytes'] / 1e6:.0f} MB, privacy {p['status']}" + (f" ({len(p['held'])} held)" if p["held"] else "")
          + f", measure {m['measure']['status']}")


if __name__ == "__main__":
    main()
