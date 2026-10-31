"""Each stop's place and heading checked against its own pictures.

hop_geometry.py measured, for every pair of stops that see each other, where the other stop lies in each one's own panorama (az_ij in i, az_ji in j:
degrees clockwise from the pano's centre), straight from the two pictures. The map (map_stops.py), the motion track (vio_map.py) and the registration
(register_stops.py) each put the stops somewhere with some heading; if a stop is placed and turned right, the direction to its neighbour on the plan,
taken relative to the stop's heading, IS that measured azimuth. The difference is how far off the stop's ring would point in the viewer (2026-10-30,
walk 16ddac14: "no photo was shot out in this direction but actually across the street" — the map's stops 010-018 and the registration were 10-170
degrees off the pictures, while the motion track agreed with them to 2 degrees).

A pair's error says one of its two stops (or both) is off; a stop's error is the median over its pairs, and `worst_out` takes out the worst stop one at a
time until every stop left agrees, so one bad stop does not condemn the good neighbours that see it.
"""
import math
import numpy as np

MIN_INLIERS = 30        # a pair matched on fewer points is not a measurement of where the other stop is
MIN_EVIDENCE = 15       # ... but is still a vote between two candidate poses of one stop
BAD_DEG = 12.0          # a stop whose pictures disagree with its place by more than this (median over its pairs) is off
MIN_SEP = 0.02          # a pair this close (as a share of the walk's extent) says nothing about the direction between them


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def good_pairs(hops, min_inliers=MIN_INLIERS):
    """[(i, j, az_ij, az_ji, inliers)] from hops.json's pairs: matched, consistent with the chain, enough inliers."""
    out = []
    for p in hops.get("pairs", []):
        if p.get("ok") and p.get("consistent", True) and p.get("inliers", 0) >= min_inliers and p.get("az_ij") is not None and p.get("az_ji") is not None:
            out.append((int(p["i"]), int(p["j"]), float(p["az_ij"]), float(p["az_ji"]), int(p["inliers"])))
    return out


def errors_2d(xy, H, pairs):
    """Plan positions xy {k: (x, y)} (y = the plan's 'north' for H) and headings H {k: degrees clockwise from +y}: [(i, j, err_i, err_j)] in degrees,
    err = the plan's direction to the other stop relative to the heading, minus the azimuth the pictures measured."""
    ks = [k for k in xy if xy[k] is not None]
    if len(ks) < 2:
        return []
    P = np.array([xy[k] for k in ks], float)
    ext = float(np.linalg.norm(P - P.mean(0), axis=1).max()) or 1.0
    out = []
    for i, j, aij, aji, _ in pairs:
        if xy.get(i) is None or xy.get(j) is None or H.get(i) is None or H.get(j) is None:
            continue
        dx, dy = xy[j][0] - xy[i][0], xy[j][1] - xy[i][1]
        if math.hypot(dx, dy) < MIN_SEP * ext:
            continue
        b = math.degrees(math.atan2(dx, dy))
        out.append((i, j, wrap(b - H[i] - aij), wrap(b + 180.0 - H[j] - aji)))
    return out


def stop_errors(errs):
    """{k: (median |err| in degrees, pairs)} over every pair the stop is in."""
    by = {}
    for i, j, ei, ej in errs:
        by.setdefault(i, []).append(abs(ei)); by.setdefault(j, []).append(abs(ej))
    return {k: (float(np.median(v)), len(v)) for k, v in by.items()}


def worst_out(errs, bad=BAD_DEG, min_pairs=1, keep=None):
    """Stops to take out, worst first, until every stop left agrees with its pictures within `bad` degrees (over at least `min_pairs` of the pairs
    left). `keep` (optional) never goes out. Returns (out, {k: (err, pairs)} of the stops left)."""
    gone, keep = [], set(keep or ())
    while True:
        live = [e for e in errs if e[0] not in gone and e[1] not in gone]
        se = stop_errors(live)
        cand = [(v[0], k) for k, v in se.items() if v[0] > bad and v[1] >= min_pairs and k not in keep]
        if not cand:
            return gone, se
        gone.append(max(cand)[1])


def summary(errs):
    """The median |err| over every pair and side, and how many: (degrees or None, n)."""
    v = [abs(e) for _, _, ei, ej in errs for e in (ei, ej)]
    return (round(float(np.median(v)), 1) if v else None), len(v)


def errors_3d(R, C, L, pairs, F=np.diag([1.0, -1.0, 1.0])):
    """The same in a map's own frame (map_stops.py: R camera-from-world, C centres, L the stops' level matrices, F COLMAP's camera axes to ours):
    [(i, j, err_i, err_j)] in degrees."""
    ks = [k for k in C]
    if len(ks) < 2:
        return []
    P = np.array([C[k] for k in ks], float)
    ext = float(np.linalg.norm(P - P.mean(0), axis=1).max()) or 1.0
    az = lambda v: math.degrees(math.atan2(v[0], v[2]))
    out = []
    for i, j, aij, aji, _ in pairs:
        if i not in C or j not in C or i not in R or j not in R:
            continue
        d = np.asarray(C[j], float) - np.asarray(C[i], float)
        if np.linalg.norm(d) < MIN_SEP * ext:
            continue
        out.append((i, j, wrap(az(L[i] @ (F @ (R[i] @ d))) - aij), wrap(az(L[j] @ (F @ (R[j] @ -d))) - aji)))
    return out
