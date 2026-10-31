"""Floor-plane correspondences between stops, from the shared COLMAP map.

Every stop sees the floor around it. A floor point seen at direction (az, el) from stop i (levelled frame, el < 0) is at a
known place on the floor plane, so we can ask where the same point appears in stop j's panorama. This gives
  * the same floor patch seen by two stops (exposure matching between stops, M3),
  * real floor texture for the hole under the camera where the person stood (nadir fill, M5).
Uses frames/poses_map.json (links: az, turn, d), frames/up.json (levelling) and the per-stop camera height in poses.json.
Conventions (same as hop_geometry / map_stops): pano x right, y up, z = pano centre, az = atan2(x, z) clockwise;
u = (az + pi) / 2pi * W, v = (pi/2 - el) / pi * H.  A link's `turn` is how much to add to an azimuth seen from i to get it in j.
"""
import json, os
import numpy as np

from hop_geometry import level_matrix


class FloorLinks:
    def __init__(self, frames="frames"):
        pm = json.load(open(os.path.join(frames, "poses_map.json")))
        ps = json.load(open(os.path.join(frames, "poses.json")))
        self.n = len(pm["nodes"])
        self.links = [{L["to"]: L for L in nd["links"]} for nd in pm["nodes"]]
        self.cam_h = [nd.get("cam_h") for nd in ps["nodes"]]
        med = float(np.median([h for h in self.cam_h if h])) if any(self.cam_h) else 1.85
        self.cam_h = [h or med for h in self.cam_h]
        self.L = [level_matrix(u) for u in json.load(open(os.path.join(frames, "up.json")))]
        self.turn_sign = 1.0

    def neighbours(self, i, dmin=0.8, dmax=6.0):
        return [j for j, L in self.links[i].items() if dmin <= L["d"] <= dmax]

    def project(self, i, j, az, el):
        """Floor points seen from stop i at levelled (az, el) in radians (el < 0). Returns the levelled (az, el) of the
        same floor points as seen from stop j, plus j's horizontal distance to them."""
        Lk = self.links[i][j]
        hi, hj = self.cam_h[i], self.cam_h[j]
        r = hi / np.tan(-el)
        P = np.stack([r * np.sin(az), np.full_like(r, -hi), r * np.cos(az)], -1)
        a, d = np.radians(Lk["az"]), Lk["d"]
        C = np.array([d * np.sin(a), hj - hi, d * np.cos(a)])
        V = P - C
        hz = np.hypot(V[..., 0], V[..., 2])
        azj = np.arctan2(V[..., 0], V[..., 2]) + self.turn_sign * np.radians(Lk["turn"])
        elj = np.arctan2(V[..., 1], hz)
        return azj, elj, hz

    def raw_uv(self, k, az, el, w, h):
        """Levelled direction (az, el) of stop k -> pixel coordinates in its raw (tilted) panorama."""
        v = np.stack([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)], -1) @ self.L[k]   # = L^T @ dir
        az_r = np.arctan2(v[..., 0], v[..., 2]); el_r = np.arcsin(np.clip(v[..., 1], -1, 1))
        return (az_r + np.pi) / (2 * np.pi) * w - 0.5, (np.pi / 2 - el_r) / np.pi * h - 0.5
