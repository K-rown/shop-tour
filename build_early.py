"""The first tour: build_tour.py into tour_early/ as soon as the privacy check, the blur step and the stops' positions (rings) are done.

  python build_early.py [--title "Name"]       # run by tour.py as the "early" step, only with SHOP_TOUR_EARLY_TOUR=1

An app that publishes a tour can open this one while the sphere depth, the registration, the stereo, the dollhouse and the plan are still being made. It is the
same tour the pipeline built before those steps existed: the cleaned pictures, the stops on the map from the rings, the glide with the first depth, no dollhouse
and no plan (the viewer hides those buttons until tour.json names them). tour.json says "partial": true and manifest.json carries the same key; the final build
in tour/ replaces it, and this folder is never touched by that build.

tour_early/manifest.json is written LAST and atomically: it is the sign that the folder is finished. Its privacy block is judged exactly as the final's is
(status "held" = at least one held stop nobody has answered yet, and then there is no tour to open).

Never fails: every way it can not run (switch off, a rebuild after a review, build_tour.py stopping) prints a line and exits 0, so tour.py never stops for it.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "tour_early"
FRAMES = "frames"


def wanted():
    return os.environ.get("SHOP_TOUR_EARLY_TOUR", "").strip().lower() in ("1", "true", "yes", "on")


def rebuild_after_review():
    """A rebuild after the person answered a review: the 3D steps ran before and their outputs still stand, so the final comes quickly and a partial tour would only be replaced."""
    try:
        return os.path.getmtime(os.path.join(FRAMES, "register", "stats.json")) > os.path.getmtime(os.path.join(FRAMES, "pii.json"))
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    if not wanted():
        print("early tour: not asked for (SHOP_TOUR_EARLY_TOUR is not 1), skipping")
        return 0
    if rebuild_after_review():
        print("early tour: this is a rebuild after a review (the 3D steps have run), skipping")
        return 0
    mf = os.path.join(OUT, "manifest.json")
    if os.path.exists(mf):
        os.remove(mf)                                    # the sign of a finished folder is written last
    cmd = [sys.executable, os.path.join(HERE, "build_tour.py"), "--out", OUT, "--early"] + (["--title", a.title] if a.title else [])
    r = subprocess.run(cmd)
    if r.returncode:
        print(f"early tour: build_tour.py stopped (exit {r.returncode}); the final build will say why if it matters")
        if os.path.exists(mf):
            print("early tour: it left a manifest (every stop held back); that is the answer")
        return 0
    try:
        import manifest
        tj = json.load(open(os.path.join(OUT, "tour.json")))
        if tj.get("held_all"):
            return 0
        m = manifest.write(FRAMES, OUT)
        p = m["privacy"]
        print(f"early tour: {m['stops']['count']} stops in {OUT}/, privacy {p['status']}" + (f" ({len(p['held'])} held)" if p["held"] else ""))
    except Exception as e:
        print(f"early tour: manifest.json was not written ({e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
