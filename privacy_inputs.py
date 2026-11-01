"""What the privacy path (remove_capturer.py -> privacy_gate.py -> privacy_margin.py) reads about camera height and neighbouring stops.

It used to read frames/poses.json: a per-stop camera height and the neighbour links of the COLMAP map. Both come out of the vio/map steps (the height is
the floor edge of the dense map, scaled by the accelerometer), so any change to those steps - a faster matcher, another mapper, a different random seed -
moved the height a few percent and reshuffled which stops the near-threshold "person-sized shape near the camera" check held. The privacy path now reads
only what the cheap steps give, which do not depend on vio:

  camera height   one stick height for the walk, PRIVACY_CAM_H (2.4 m). There is no per-stop floor measurement in metres that does not come from the map: the depth
                  model (depth_panos.py) predicts relative inverse depth, known only up to a scale and a shift per view, and the floor segmentation (floor_panos.py)
                  gives pixels, not metres. The metres of the old cam_h came from the map. 2.4 m is the middle of what the map measured on every walk-1 run
                  (2.25-2.57 m per stop, medians 2.40-2.50) and inside scale_priors.py's range for a stick-mounted camera (1.8-2.8 m, typical 2.3). The margin
                  (privacy_margin.py) tries the height at several factors, so a stop that is held at some plausible stick height is held.
  neighbours      frames/hops.json (hop_geometry.py: the two photos of a pair only, with the levelling from up.json). A pair gives the direction (az), the turn (yaw)
                  and, when the floor is seen well enough, the distance on the floor for a camera 1.85 m up; that distance is rescaled to PRIVACY_CAM_H.

SHOP_TOUR_PRIVACY_INPUTS=map (opt-in) reads frames/poses.json the old way; the default, "own", reads only what is described above.
SHOP_TOUR_PRIVACY_CAM_H=<metres> changes the stick height.
"""
import json, os

PRIVACY_CAM_H = 2.4
LADDER = (1.0, 1.8, 3.0, 4.5, 6.5)       # metres: the distances tried for a pair whose distance the photographs did not give
HOP_CAM_H = 1.85                      # hop_geometry.CAM_H: the camera height its dist_m assumes (kept here so this module needs no numpy/cv2)


def mode():
    """"own" (the default): the privacy path reads only photographs, levelling, hops and floor (it can run beside vio). "map": poses.json, the old way - opt-in with SHOP_TOUR_PRIVACY_INPUTS=map."""
    return "map" if os.environ.get("SHOP_TOUR_PRIVACY_INPUTS", "").strip().lower() == "map" else "own"


def stick_height():
    try:
        h = float(os.environ.get("SHOP_TOUR_PRIVACY_CAM_H", "") or PRIVACY_CAM_H)
    except ValueError:
        return PRIVACY_CAM_H
    return h if 1.0 <= h <= 4.0 else PRIVACY_CAM_H


def hop_links(hops, n, cam_h):
    """{stop: {neighbour: {az, d, turn}}} from hops.json's pairs: a usable pair (ok, consistent with the chain of consecutive hops) both ways; "d" is None when the
    photographs gave no distance on the floor. az is where the other stop is seen from this one, turn what to add to an azimuth seen from this one to get it in the other (poses.json's
    convention: a pair measured i -> j gives turn -yaw at i and +yaw at j)."""
    out = [dict() for _ in range(n)]
    k = cam_h / HOP_CAM_H
    for r in hops.get("pairs", []):
        i, j = r.get("i"), r.get("j")
        if not (isinstance(i, int) and isinstance(j, int) and 0 <= i < n and 0 <= j < n) or not r.get("ok") or r.get("consistent") is False:
            continue
        if r.get("az_ij") is None or r.get("az_ji") is None or r.get("yaw") is None:
            continue
        d = float(r["dist_m"]) * k if r.get("dist_m") is not None else None
        shared = r.get("inliers") or 0
        out[i][j] = dict(az=float(r["az_ij"]), d=None if d is None else round(d, 2), turn=-float(r["yaw"]), n=shared)
        out[j][i] = dict(az=float(r["az_ji"]), d=None if d is None else round(d, 2), turn=float(r["yaw"]), n=shared)
    return out


class Inputs:
    def __init__(self, frames, n):
        self.mode = mode()
        self.n = n
        self._frames = frames
        self._links_ = None
        if self.mode == "map":
            P = json.load(open(os.path.join(frames, "poses.json")))
            self._h = [nd.get("cam_h") or 2.0 for nd in P["nodes"]]
            self._links_ = [{l["to"]: dict(az=l["az"], d=l["d"], turn=l["turn"]) for l in nd["links"]} for nd in P["nodes"]]
        else:
            self._h = [stick_height()] * n                       # (the stick height needs nothing from the hops: hops.json is read when the neighbours are first asked for, so the
                                                                 # masks can be made before the hops are done - tour.EARLY)

    @property
    def _links(self):
        if self._links_ is None:
            self._links_ = hop_links(json.load(open(os.path.join(self._frames, "hops.json"))), self.n, stick_height())
        return self._links_

    def cam_h(self, k):
        return self._h[k]

    def neighbours(self, i, max_n=8, max_d=9.0):
        """Stops that see the floor under stop i: the previous and next in the walk first, then the nearest by distance."""
        links = self._links[i]
        if self.mode == "map":
            order = [j for j in (i - 1, i + 1) if j in links]
            order += [j for j in sorted(links, key=lambda j: links[j]["d"]) if j not in order and links[j]["d"] <= max_d]
            return [dict(j=j, az=links[j]["az"], d=links[j]["d"], turn=links[j]["turn"]) for j in order[:max_n]]
        # from two photographs: the walk's previous and next stop first, then the pairs that share the most features (the nearest ones do). A pair that gives
        # no distance on the floor is offered at a ladder of distances and capturer.py keeps the one whose floor lines up best with this stop's.
        order = [j for j in (i - 1, i + 1) if j in links]
        order += sorted((j for j in links if j not in order), key=lambda j: (-links[j].get("n", 0), j))
        out = []
        for j in order:
            L = links[j]
            ds = [L["d"] * f for f in (0.8, 1.0, 1.25)] if L["d"] is not None else list(LADDER)
            if min(ds) > max_d:
                continue
            out.append(dict(j=j, az=L["az"], d=ds[len(ds) // 2], ds=[round(x, 2) for x in ds if x <= max_d], turn=L["turn"]))
        return out[:max_n]
