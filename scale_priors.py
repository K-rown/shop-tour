"""Scale from what people and buildings are like, as a cross-check on the accelerometer scale (imu_scale.py): no user input, no known object in the picture.

Usage:  python scale_priors.py [--frames frames]          (after place_rings.py: it writes camera and roof heights per stop into frames/poses.json)
Prints, for each prior, the factor by which the tour's lengths would have to be multiplied for the prior's typical value to hold, the range of factors that keep it inside what is
plausible, and whether the current scale (the accelerometer's, or the old assumption) is inside that range. Writes frames/scale_priors.json.

Priors (all soft; each is a range, with a typical value used only when that prior is looked at ALONE):
  camera height  people hold a stick-mounted camera between 1.8 m (just above the head) and 2.8 m (an extended pole); typical 2.3 m.
                 Measured per stop as the distance from the floor edge of the dense map to the camera (place_rings.py).
  roof height    commercial and industrial rooms are 3.0-6.0 m high; typical 4.5 m. Measured as the top of the dense map's point cloud above the floor edge; that is the highest layer
                 with features (steel under the roof deck), so it is a lower bound for the roof.
  doors          not implemented: a door detector would need the segmentation model's door class and roll-up doors are not standard sizes.
Combining: the accelerometer's scale is kept whenever it is inside the intersection of the ranges; if it is not, nothing is changed but the disagreement is logged (a prior is a
cross-check, not an input: a person holding the stick at 3.1 m is allowed to exist).
"""
import argparse, json, os
import numpy as np

PRIORS = {"camera_height": dict(lo=1.8, hi=2.8, typical=2.3, key="cam_h"), "roof_height": dict(lo=3.0, hi=6.0, typical=4.5, key="roof_m")}


def estimate(nodes):
    """{prior: dict(measured_m, factor_typical, factor_lo, factor_hi)}; the factors multiply the tour's lengths. Priors without a measurement are left out."""
    out = {}
    for name, pr in PRIORS.items():
        v = [n[pr["key"]] for n in nodes if n.get(pr["key"])]
        if not v:
            continue
        m = float(np.median(v))
        out[name] = dict(measured_m=round(m, 3), factor_typical=pr["typical"] / m, factor_lo=pr["lo"] / m, factor_hi=pr["hi"] / m, stops=len(v))
    return out


def combine(est):
    """The range of factors every prior allows, and whether the current scale (factor 1) is inside it."""
    if not est:
        return None
    lo = max(e["factor_lo"] for e in est.values()); hi = min(e["factor_hi"] for e in est.values())
    return dict(lo=lo, hi=hi, consistent=lo <= hi, current_inside=lo <= 1.0 <= hi if lo <= hi else False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames")
    a = ap.parse_args()
    nodes = json.load(open(os.path.join(a.frames, "poses.json")))["nodes"]
    est = estimate(nodes)
    for k, e in est.items():
        print(f"{k:14s} measured {e['measured_m']:.2f} m  -> alone it says: multiply lengths by {e['factor_typical']:.3f}  (range {e['factor_lo']:.2f}-{e['factor_hi']:.2f})")
    c = combine(est)
    if c:
        print(f"together the priors allow factors {c['lo']:.2f}-{c['hi']:.2f}; the current scale is " + ("inside that range" if c["current_inside"] else "OUTSIDE that range (logged, not changed)"))
    json.dump(dict(priors=est, combined=c), open(os.path.join(a.frames, "scale_priors.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
