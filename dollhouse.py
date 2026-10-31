"""The dollhouse: fuse every stop's depth into ONE coloured mesh (TSDF on the GPU) that the viewer can orbit from above.

Usage:  python dollhouse.py [--frames frames] [--out tour] [--depth depth_sphere|depth_mvs|depth] [--voxel 0.03] [--max-tris 300000]
Reads:  frames/poses.json (x, y, heading, camera height per stop), frames/up.json,
        frames/depth_sphere/<id>.npz (full-sphere metric range, depth_sphere.py) — or frames/<depth>/eq_<id>.png (8-bit log range, place_rings.py / depth_align.py),
        frames/safe/<id>.jpg (the cleaned pictures: held-back stops are not in frames/safe/ and are not used), frames/capturer/<id>_hole.png
Writes: tour/dollhouse.bin + tour/dollhouse.json + tour/dollhouse.jpg (a quantised mesh with a UV atlas the viewer loads on demand),
        frames/dollhouse/mesh.ply, frames/dollhouse/mesh_uv.npz (the textured mesh: V, T, UV — floorplan.py renders the plan from it), frames/dollhouse/stats.json

World frame (metres): X east, Y north, Z up, floor at Z = 0. A stop's pano azimuth `a` (clockwise from the picture centre) points at bearing H + a.
Every stop is cut into six 90-degree faces (a cube map) because TSDF fusion wants pinhole images. The person who held the stick is replaced by floor in the
depth (otherwise 23 little statues would stand in the model). The volume is as big as what the stops saw (not a fixed margin), and the voxel is the finest
the card's memory allows (3 cm on a shop, 5 cm on a hall). The texture is a real UV atlas (dollhouse_uv.py, xatlas); --block-texture is the old
one-block-per-triangle look.
"""
import argparse, json, math, os, re, sys, time
import cv2
import numpy as np
from imutil import imread

DMIN, DMAX = 0.2, 60.0          # place_rings.py: 8-bit log range encoding
MAX_RANGE = 12.0                # depth beyond this is not used (per-stop; the sphere depth allows a little more)
FACE = 512
ANCHOR_FILL_WEIGHT = 0.5
AGREE_NEAR_M = 10.0             # a stop's depth is checked against the stops within this distance
AGREE_MAX = 10                  # ... the nearest this many
AGREE_ALONE = 0.4               # weight of a depth no neighbour can speak to (nobody else saw that spot)
GUESSED_REACH_M = 8.0           # how far a weakly levelled stop may build
WEAK_SPREAD_DEG = 15.0          # the ground's normals disagreeing past this: the ground reading alone is weak
VOXEL_BUDGET = 70_000_000       # voxels (about 2 GB on the card): the voxel is chosen so the volume fits
FACES = {                        # (forward, right, up) in the stop's levelled frame (x right, y up, z forward)
    "front": ((0, 0, 1), (1, 0, 0), (0, 1, 0)), "right": ((1, 0, 0), (0, 0, -1), (0, 1, 0)), "back": ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    "left": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)), "up": ((0, 1, 0), (1, 0, 0), (0, 0, -1)), "down": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
}


def rot_to_y(u):
    u = np.asarray(u, float); u = u / np.linalg.norm(u); y = np.array([0, 1.0, 0]); v = np.cross(u, y); c = float(u @ y)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def level_to_world(heading_deg):
    """Matrix M: a vector in the levelled frame (x right, y up, z forward) -> world (X east, Y north, Z up). Forward is at bearing H."""
    h = math.radians(heading_deg); s, c = math.sin(h), math.cos(h)
    return np.array([[c, 0, s], [-s, 0, c], [0, 1, 0]], float)


def decode_range(png):
    return DMIN * (DMAX / DMIN) ** (png[..., 2].astype(np.float32) / 255.0)


def face_rays(fov_deg=90.0, n=FACE):
    f = n / 2 / math.tan(math.radians(fov_deg) / 2)
    c, r = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    d = np.stack([(c - n / 2) / f, -(r - n / 2) / f, np.ones_like(c)], -1)               # camera: x right, y up, z forward
    return d / np.linalg.norm(d, axis=-1, keepdims=True), f


def pano_uv(r, h, w):
    az = np.arctan2(r[..., 0], r[..., 2]); el = np.arcsin(np.clip(r[..., 1] / np.linalg.norm(r, axis=-1), -1, 1))
    return ((az + np.pi) / (2 * np.pi) * w - 0.5).astype(np.float32), ((np.pi / 2 - el) / np.pi * h - 0.5).astype(np.float32)


def hole_in_eq(hole1024, up, W, H, fov=130.0):
    """The capturer's straight-down mask (the 1024 px view of remove_capturer.py) as an equirect mask."""
    n = hole1024.shape[0]; f = n / 2 / math.tan(math.radians(fov) / 2)
    az = ((np.arange(W) + 0.5) / W * 2 * np.pi - np.pi); el = (np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi)
    A, E = np.meshgrid(az, el)
    r = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1) @ rot_to_y(up).T            # pano ray -> levelled ray
    dy = np.maximum(-r[..., 1], 1e-3)
    col = (f * r[..., 0] / dy + n / 2 - 0.5).astype(np.float32); row = (n / 2 - 0.5 - f * r[..., 2] / dy).astype(np.float32)
    m = cv2.remap(hole1024.astype(np.uint8), col, row, cv2.INTER_NEAREST, borderValue=0) > 0
    return m & (r[..., 1] < -0.3)


def cam_z(node):
    """The camera's height above the walk's zero: its stick height over its own ground, plus that ground's height (register_stops.py's z)."""
    return float(node.get("cam_h") or 2.0) + float(node.get("z") or 0.0)


def water_of(frames, sid, shape):
    """The water pixels of a stop's sphere depth (depth_sphere.py), and the water's drop below its camera; (None, 0) when there is none."""
    f = os.path.join(frames, "depth_sphere", sid + ".npz")
    if not os.path.exists(f):
        return None, 0.0
    z = np.load(f)
    if "water" not in z or z["water"].size == 0:
        return None, 0.0
    m = np.unpackbits(z["water"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
    return (m if m.any() else None), float(z["water_drop"])


def load_range(frames, depth_dir, sid):
    """(range in metres along every ray of an equirect, confidence 0..1 or None, levelled: bool). The sphere depth is already in the LEVELLED frame;
    the older eq_<id>.png maps are in the pano's own frame."""
    if depth_dir == "depth_sphere":
        f = os.path.join(frames, "depth_sphere", sid + ".npz")
        if os.path.exists(f):
            z = np.load(f)
            return z["range"].astype(np.float32), z["conf"].astype(np.float32) / 255.0, True
    for d in (depth_dir, "depth_mvs", "depth"):
        ep = os.path.join(frames, d, f"eq_{sid}.png")
        if os.path.exists(ep):
            rng = decode_range(imread(ep))
            if rng.shape[1] > 2048:
                rng = cv2.resize(rng, (2048, 1024), interpolation=cv2.INTER_NEAREST)
            return rng, None, False
    return None, None, False


HOLE_WEIGHT = 0.04


def weak_level(h):
    """level.json says how a stop was levelled; a stop taken as level on no evidence, or read off the ground alone where the ground's normals disagreed
    among themselves past WEAK_SPREAD_DEG (a gully's sloping sand), is weakly levelled: a few degrees of lean move what it sees at 12 m by half a metre."""
    way = str(h.get("way", ""))
    if way.startswith("level"):
        return True
    spread = h.get("ground_spread")
    if spread is None:
        for n in h.get("notes", []):
            m_ = re.search(r"spread ([0-9.]+)", str(n))
            if m_:
                spread = float(m_.group(1))
    return way == "ground" and spread is not None and spread > WEAK_SPREAD_DEG


def stop_cloud(rng, node, up, lev, step=2):
    """The stop's depth as world points (float32, N x 3), every step-th pixel."""
    Hh, Ww = rng.shape
    r = rng[::step, ::step]
    h, w = r.shape
    az = (np.arange(w) * step + step / 2) / Ww * 2 * np.pi - np.pi; el = np.pi / 2 - (np.arange(h) * step + step / 2) / Hh * np.pi
    A, E = np.meshgrid(az, el)
    q = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1).astype(np.float32)
    if not lev:
        q = q @ rot_to_y(up).T.astype(np.float32)                                     # pano ray -> levelled
    M = level_to_world(node["H"]).astype(np.float32)
    pos = np.array([node["x"], node["y"], cam_z(node)], np.float32)
    ok = (r > 0) & (r < MAX_RANGE * 3)
    X = (q[ok] @ M.T) * r[ok][:, None] + pos
    return X.astype(np.float32)


def agreement(zd, c2w, f, clouds, dev):
    """ONE STOP'S DEPTH, CHECKED BY ITS NEIGHBOURS: a factor map (face size) for the fusion weight. Each neighbour's points are projected into this face;
    where a neighbour has a surface within tolerance of this stop's depth the pixel is confirmed (1.0); where a neighbour saw PAST this stop's surface
    (its nearest point along the ray lies well behind it) the depth is denied (it was: parallax sees past thin things, so it no longer counts); where no neighbour has anything to say it stands alone
    (AGREE_ALONE). Monocular depth makes thin, floating sheets between a shelf and the wall behind it that no second view confirms - this is what
    keeps them out of the model while a wall every stop agrees on keeps its full weight."""
    import torch
    n = zd.shape[0]; b = n // 4; fb = f / 4; cb = b / 2
    zi = torch.from_numpy(np.ascontiguousarray(zd[::4, ::4])).to(dev)
    have_i = zi > 0
    w2c = np.linalg.inv(c2w).astype(np.float32); Rw = torch.from_numpy(w2c[:3, :3]).to(dev); tw = torch.from_numpy(w2c[:3, 3]).to(dev)
    agree = torch.zeros((b, b), dtype=torch.int16, device=dev); deny = torch.zeros((b, b), dtype=torch.int16, device=dev)
    tol = 0.06 * zi + 0.06
    for X in clouds:
        Xc = X @ Rw.T + tw
        z = Xc[:, 2]; m = z > 0.1
        u = (Xc[:, 0] / z.clamp(min=1e-3) * fb + cb).long(); v = (Xc[:, 1] / z.clamp(min=1e-3) * fb + cb).long()
        m &= (u >= -1) & (u <= b) & (v >= -1) & (v <= b)
        if not bool(m.any()):
            continue
        u, v, zm = u[m], v[m], z[m]
        zb = torch.full((b * b,), float("inf"), device=dev)
        for du in (-1, 0, 1):                                            # each point covers a 3 x 3 block of cells: a neighbour's points thin out with distance,
            for dv in (-1, 0, 1):                                        # and a cell no point lands in would read as "nobody saw this" on a plain floor
                uu, vv = u + du, v + dv
                ok = (uu >= 0) & (uu < b) & (vv >= 0) & (vv < b)
                zb.scatter_reduce_(0, (vv[ok] * b + uu[ok]), zm[ok], reduce="amin")
        zb = zb.view(b, b)
        has = torch.isfinite(zb) & have_i
        agree += (has & ((zb - zi).abs() < tol)).to(torch.int16)
        deny += (has & (zb > zi + tol)).to(torch.int16)
    # NO DENIAL: a neighbour that sees past this stop's surface is not proof the surface is wrong - it sees between the joists, through the open shelving,
    # past the door's edge (parallax); taking that as denial thinned every near, thin thing and the tape read everything 10-60 % too big
    fac = torch.where(agree > 0, 1.0, AGREE_ALONE).float()
    fac = torch.nn.functional.interpolate(fac[None, None], size=(n, n), mode="nearest")[0, 0]
    return fac.cpu().numpy().astype(np.float32), float(agree.gt(0)[have_i].float().mean()) if bool(have_i.any()) else 1.0


def stop_faces(frames, node, up, rng, pano, hole_eq, anchor=None, conf=None, levelled=False, face=FACE, max_range=MAX_RANGE, edge_rel=0.12):
    """The six faces of one stop: (colour uint8 HxWx3 RGB, z-depth float32, camera-to-world 4x4, name, weight map) each."""
    Hh, Ww = rng.shape
    M = level_to_world(node["H"]); R = rot_to_y(up)
    pos = np.array([node["x"], node["y"], cam_z(node)], float)
    d, f = face_rays(n=face)
    out = []
    for name, (fw, rt, upv) in FACES.items():
        fw, rt, upv = (np.array(v, float) for v in (fw, rt, upv))
        dl = d[..., 0:1] * rt + d[..., 1:2] * upv + d[..., 2:3] * fw                    # rays in the levelled frame
        pr = dl if levelled else dl @ R                                                   # levelled -> the depth map's frame
        u, v = pano_uv(pr, Hh, Ww)
        rg = cv2.remap(rng, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        hm = cv2.remap(hole_eq.astype(np.uint8), u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) > 0 if hole_eq is not None else np.zeros(rg.shape, bool)
        down = -dl[..., 1]
        floor_r = (node.get("cam_h") or 2.0) / np.maximum(down, 0.05)
        rg = np.where(hm & (down > 0.05), floor_r, rg)                                    # him: replaced by floor
        rg = np.where(hm & (down <= 0.05), 0.0, rg)
        rg = np.where(rg > max_range, 0.0, rg)                                            # far depth is too noisy to build a wall from: a nearer stop will see it
        zd = (rg * d[..., 2]).astype(np.float32)                                          # z-depth along the face axis
        k = np.ones((5, 5), np.uint8)
        edge = (cv2.dilate(zd, k) - cv2.erode(zd, k)) > edge_rel * np.maximum(zd, 0.5)    # a jump in depth: 'flying pixels' between a thing and what is behind it
        zd = np.where(cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, 0.0, zd).astype(np.float32)
        # the pano's colour for this face: the picture is in the pano's own frame whatever frame the depth is in
        pu, pv = pano_uv(dl @ R, pano.shape[0], pano.shape[1])
        col = cv2.remap(pano, pu, pv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        c2w = np.eye(4)
        right_w, up_w, fw_w = M @ rt, M @ upv, M @ fw
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right_w, -up_w, fw_w, pos        # Open3D camera: x right, y down, z forward
        wm = None
        if conf is not None:
            cm = cv2.remap(conf, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            wm = np.clip(0.15 + 0.85 * cm, 0.05, 1.0).astype(np.float32)
        if hm.any():
            # WHERE HE STOOD IS A GUESS: the floor plane under the capturer's hole is flat ground at the stick's height - on a rock shelf it is not, and
            # at full weight (the nearest view wins the fusion) it cut a black hole against the neighbours' real view of that ground; it counts little
            hw = np.where(hm, HOLE_WEIGHT, 1.0).astype(np.float32)
            wm = hw if wm is None else wm * hw
        if anchor is not None:                                                            # multi-view ranges count double against the depth model's
            am = cv2.remap(anchor.astype(np.uint8), u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) > 0
            wa = np.where(am, 1.0, ANCHOR_FILL_WEIGHT).astype(np.float32)
            wm = wa if wm is None else wm * wa
        out.append((cv2.cvtColor(col, cv2.COLOR_BGR2RGB), zd, c2w, name, wm))
    return out


def tight_block():
    """SHOP_TOUR_TSDF_TIGHT=0 fuses every face over the whole reach cube again (the old way, 10x the work; the same volume)."""
    return os.environ.get("SHOP_TOUR_TSDF_TIGHT", "").strip().lower() not in ("0", "false", "no", "off")


class Tsdf:
    """A dense truncated signed distance volume on the GPU (torch): running weighted average of the signed distance and the colour of every voxel
    that a depth face sees."""
    def __init__(self, lo, hi, voxel, trunc, device):
        import torch
        self.t = torch; self.dev = device; self.voxel = voxel; self.trunc = trunc
        self.lo = np.asarray(lo, np.float64); self.dims = np.ceil((np.asarray(hi) - self.lo) / voxel).astype(int)
        shp = tuple(int(x) for x in self.dims)
        self.tsdf = torch.zeros(shp, dtype=torch.float32, device=device); self.w = torch.zeros(shp, dtype=torch.float32, device=device)
        self.col = torch.zeros(shp + (3,), dtype=torch.float16, device=device); self.cw = torch.zeros(shp, dtype=torch.float32, device=device)

    def integrate(self, depth, color, K, w2c, reach=14.0, cam_pos=None, wmap=None, max_range=MAX_RANGE):
        """depth: z-depth image (0 = none), color: HxWx3 uint8, K: (f, cx, cy), w2c: 4x4 world->camera.
        Only the voxels inside this face's view (in front of the camera, in the 90-degree frustum, within the range that carries depth) are touched."""
        t = self.t; dev = self.dev
        H, W = depth.shape
        d_t = t.from_numpy(depth).to(dev); c_t = t.from_numpy(color).to(dev)
        R = t.from_numpy(w2c[:3, :3].astype(np.float32)).to(dev); tr = t.from_numpy(w2c[:3, 3].astype(np.float32)).to(dev)
        f, cx, cy = K
        lo = np.maximum(((cam_pos - reach) - self.lo) / self.voxel, 0).astype(int); hi = np.minimum(((cam_pos + reach) - self.lo) / self.voxel, self.dims).astype(int)
        if tight_block():
            dmax = float(depth.max())
            if not dmax > 0:
                return
            zl = min(min(reach, max_range + self.trunc * 2) * 1.5, dmax + self.trunc + 0.01); sl = 1.02
            Rn, tn = w2c[:3, :3], w2c[:3, 3]
            corners = np.array([[0.0, 0.0, 0.0]] + [[sx * sl * zl, sy * sl * zl, zl] for sx in (-1, 1) for sy in (-1, 1)])
            Pw = (corners - tn) @ Rn
            lo = np.maximum(lo, np.floor((Pw.min(0) - self.lo) / self.voxel).astype(int) - 1)
            hi = np.minimum(hi, np.ceil((Pw.max(0) - self.lo) / self.voxel).astype(int) + 1)
            if (hi <= lo).any():
                return
        ax = [(t.arange(lo[k], hi[k], device=dev).float() + 0.5) * self.voxel + float(self.lo[k]) for k in range(3)]          # voxel centres along each axis
        zc = [ax[k] * R[2, k] for k in range(3)]
        z = zc[0][:, None, None] + zc[1][None, :, None] + zc[2][None, None, :] + tr[2]
        zmax = min(reach, max_range + self.trunc * 2)
        cand = ((z > 0.1) & (z < zmax * 1.5)).nonzero(as_tuple=False)                      # (n, 3) block indices of voxels in front of the camera
        if len(cand) == 0:
            return
        P = t.stack([ax[0][cand[:, 0]], ax[1][cand[:, 1]], ax[2][cand[:, 2]]], 1)
        Xc = P @ R.T + tr
        zz = Xc[:, 2]
        u = (Xc[:, 0] / zz.clamp(min=1e-3) * f + cx).long(); v = (Xc[:, 1] / zz.clamp(min=1e-3) * f + cy).long()
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        uu, vv = u.clamp(0, W - 1), v.clamp(0, H - 1)
        dd = d_t[vv, uu]
        sdf = dd - zz
        ok &= (dd > 0) & (sdf > -self.trunc)
        keep = ok.nonzero(as_tuple=False)[:, 0]
        if len(keep) == 0:
            return
        cand = cand[keep]; uu, vv, dd, sdf = uu[keep], vv[keep], dd[keep], sdf[keep]
        flat = ((cand[:, 0] + int(lo[0])) * int(self.dims[1]) + (cand[:, 1] + int(lo[1]))) * int(self.dims[2]) + (cand[:, 2] + int(lo[2]))
        val = (sdf / self.trunc).clamp(max=1.0)
        wt = 1.0 / (0.5 + dd * dd)                                       # near views count far more than far ones (depth error grows with the range)
        if wmap is not None:
            wt = wt * t.from_numpy(wmap).to(dev)[vv, uu]
        wf, tf = self.w.view(-1), self.tsdf.view(-1)
        w0 = wf[flat]; tot = w0 + wt
        tf[flat] = (tf[flat] * w0 + val * wt) / tot.clamp(min=1e-6)
        near = sdf.abs() < self.trunc * 0.5                              # colour only from right at the surface
        n_i = near.nonzero(as_tuple=False)[:, 0]
        if len(n_i):
            fi = flat[n_i]; cw = self.cw.view(-1); colf = self.col.view(-1, 3)
            cw0 = cw[fi]; c0 = colf[fi].float(); c_new = c_t[vv[n_i], uu[n_i]].float(); cn = wt[n_i]
            colf[fi] = ((c0 * cw0[:, None] + c_new * cn[:, None]) / (cw0 + cn).clamp(min=1e-6)[:, None]).half()
            cw[fi] = cw0 + cn
        wf[flat] = tot

    def mesh(self):
        from skimage import measure
        vol = self.tsdf.cpu().numpy(); obs = (self.w.cpu().numpy() > 0)
        verts, faces, normals, _ = measure.marching_cubes(vol, level=0.0, mask=obs, spacing=(self.voxel,) * 3)
        verts = verts + self.lo + self.voxel * 0.5
        idx = np.clip(((verts - self.lo) / self.voxel).astype(int), 0, self.dims - 1)
        cols = self.col.cpu().numpy()[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)
        return verts, faces, cols


def scene_reach(frames, pts, P, ids, depth_dir):
    """How far the stops see, in metres: the 93rd percentile of every stop's ranges (the sky and the water already taken out). A shop is 6-10 m; a beach or a yard 20-40 m."""
    rs = []
    for i, p in enumerate(pts):
        if (ids and p["id"] not in ids) or P[i].get("x") is None:
            continue
        rng, _, _ = load_range(frames, depth_dir, p["id"])
        if rng is not None:
            r = rng[::8, ::8]; rs.append(r[r > 0.3])
    if not rs:
        return MAX_RANGE
    return float(np.percentile(np.concatenate(rs), 93))


def volume_bounds(frames, pts, P, UP, ids, depth_dir, max_range):
    """The box every stop saw (2nd..98th percentiles of the depth points, half a metre of air round them), floor at -0.3, and the roof."""
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    for i, p in enumerate(pts):
        node = P[i]
        if (ids and p["id"] not in ids) or node.get("x") is None:
            continue
        rng, _, lev = load_range(frames, depth_dir, p["id"])
        if rng is None:
            continue
        r = rng[::8, ::8]; h, w = r.shape
        az = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi; el = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
        A, E = np.meshgrid(az, el)
        q = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1)
        if not lev:
            q = q @ rot_to_y(UP[i]).T
        ok = (r > 0.3) & (r < max_range)
        X = (q[ok] * r[ok][..., None]) @ level_to_world(node["H"]).T + np.array([node["x"], node["y"], cam_z(node)])
        if len(X) < 100:
            continue
        lo = np.minimum(lo, np.percentile(X, 2, axis=0)); hi = np.maximum(hi, np.percentile(X, 98, axis=0))
    if not np.isfinite(lo).all():
        xy = np.array([[n["x"], n["y"]] for n in P if n.get("x") is not None])
        return np.r_[xy.min(0) - 9, -0.3], np.r_[xy.max(0) + 9, 6.5]
    return np.r_[lo[:2] - 0.5, min(-0.3, lo[2] - 0.3)], np.r_[hi[:2] + 0.5, min(hi[2] + 0.5, 25.0)]


def pick_voxel(lo, hi, asked, budget=VOXEL_BUDGET):
    """The asked voxel, or the finest coarser one whose volume fits the card."""
    v = asked
    while np.prod(np.ceil((hi - lo) / v)) > budget:
        v = round(v + 0.005, 3)
    return v


def fuse(frames, voxel=0.03, trunc=None, ids=None, log=print, depth_dir="depth_sphere"):
    import torch, open3d as o3d
    P = json.load(open(os.path.join(frames, "poses.json")))["nodes"]
    UP = json.load(open(os.path.join(frames, "up.json")))
    pts = json.load(open(os.path.join(frames, "points.json")))
    sphere = depth_dir == "depth_sphere" and os.path.exists(os.path.join(frames, "depth_sphere", "stats.json"))
    # THE SCENE'S OWN REACH: a shop's walls are 6 m off, a beach's cliffs 30; the far depth is the model's roughest, but a wall left out is worse than a rough one
    max_range = float(np.clip(1.3 * scene_reach(frames, pts, P, ids, depth_dir), 15.0, 40.0)) if sphere else MAX_RANGE
    face = 768 if sphere else FACE
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lo, hi = volume_bounds(frames, pts, P, UP, ids, depth_dir, max_range)
    voxel = pick_voxel(lo, hi, voxel)
    trunc = trunc or max(0.1, 4 * voxel)
    vol = Tsdf(lo, hi, voxel, trunc, dev)
    log(f"volume {vol.dims.tolist()} voxels of {voxel} m ({np.prod(vol.dims) / 1e6:.0f} M, truncation {trunc:.2f} m) on {dev}; depth from frames/{depth_dir}, faces {face} px, reach {max_range:.0f} m")
    f = face / 2 / math.tan(math.radians(45)); K = (f, face / 2, face / 2)
    # ONE SEA LEVEL: every stop saw the water at its own drop below its own camera; in the walk's frame those are one level (the median), and
    # each stop's water pixels are laid on it — a shore seen from twenty stops is one sheet, not twenty
    z_sea = None
    if sphere:
        levels = []
        for i, p in enumerate(pts):
            if (ids and p["id"] not in ids) or P[i].get("x") is None:
                continue
            f_ = os.path.join(frames, "depth_sphere", p["id"] + ".npz")
            if os.path.exists(f_):
                zz = np.load(f_)
                if "water_drop" in zz and float(zz["water_drop"]) > 0:
                    levels.append(cam_z(P[i]) - float(zz["water_drop"]))
        if len(levels) >= 2:
            z_sea = float(np.median(levels)); log(f"water on one level: z = {z_sea:.2f} m over the walk's zero ({len(levels)} stops saw water)")
    fuse.last_z_sea = z_sea
    # every stop's depth as world points, for the neighbours' check
    clouds, cpos = {}, {}
    if sphere and os.environ.get("SHOP_TOUR_AGREE", "1") != "0":
        for i, p in enumerate(pts):
            if (ids and p["id"] not in ids) or P[i].get("x") is None:
                continue
            rng_, _, lev_ = load_range(frames, depth_dir, p["id"])
            if rng_ is None:
                continue
            clouds[i] = torch.from_numpy(stop_cloud(rng_, P[i], UP[i], lev_)).to(dev)
            cpos[i] = np.array([P[i]["x"], P[i]["y"]], float)
        log(f"neighbours' check on: {len(clouds)} stops' depth as points ({sum(len(c) for c in clouds.values()) / 1e6:.1f} M)")
    agree_stats = []
    # A STOP LEVELLED ON NO EVIDENCE REACHES LESS FAR: level_panos takes a pano as level when neither a horizon nor vertical edges could say (a gully
    # between dunes); a few degrees of real lean move what it sees at 20 m by a metre, and those far pixels fanned out over the sea as grey ribs
    guessed = set()
    lv = os.path.join(frames, "level.json")
    if os.path.exists(lv):
        try:
            guessed = {h["id"] for h in json.load(open(lv)) if weak_level(h)}
        except Exception:                                                                 # noqa: BLE001
            guessed = set()
    if guessed:
        log(f"{len(guessed)} stops are weakly levelled (no evidence, or the ground alone disagreeing with itself): their reach is {GUESSED_REACH_M} m")
    used = []
    for i, p in enumerate(pts):
        node = P[i]
        sp = os.path.join(frames, "safe", p["filename"])
        if (ids and p["id"] not in ids) or not os.path.exists(sp) or node.get("x") is None:
            continue
        rng, conf, lev = load_range(frames, depth_dir, p["id"])
        if rng is None:
            continue
        if z_sea is not None and lev:
            wm, _ = water_of(frames, p["id"], rng.shape)
            if wm is not None:
                Hh, Ww = rng.shape
                el = np.pi / 2 - (np.arange(Hh) + 0.5) / Hh * np.pi
                dy = -np.sin(el)[:, None] * np.ones((1, Ww), np.float32)
                drop = cam_z(node) - z_sea
                if drop > 0.3:
                    rng = np.where(wm & (dy > 0.03), np.minimum(drop / np.maximum(dy, 0.03), MAX_RANGE * 3), rng).astype(np.float32)
        pano = imread(sp); pano = cv2.resize(pano, (4096, 2048), interpolation=cv2.INTER_AREA)
        hp = os.path.join(frames, "capturer", p["id"] + "_hole.png")
        hole = hole_in_eq(cv2.imread(hp, 0) > 0, UP[i], rng.shape[1], rng.shape[0]) if os.path.exists(hp) else None
        if hole is not None and lev:                                                      # the hole mask is in the pano's frame; the sphere depth is levelled
            Hh, Ww = rng.shape
            az = (np.arange(Ww) + 0.5) / Ww * 2 * np.pi - np.pi; el = np.pi / 2 - (np.arange(Hh) + 0.5) / Hh * np.pi
            A, E = np.meshgrid(az, el)
            q = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1) @ rot_to_y(UP[i])      # levelled ray -> pano ray (R^T q == q @ R)
            u, v = pano_uv(q, Hh, Ww)
            hole = cv2.remap(hole.astype(np.uint8), u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) > 0
        ap_ = os.path.join(frames, depth_dir, f"{p['id']}_anchor.png")
        anchor = cv2.imread(ap_, 0) > 0 if os.path.exists(ap_) else None
        near = []
        if clouds:
            others = [(np.linalg.norm(cpos[j] - cpos[i]), j) for j in clouds if j != i and np.linalg.norm(cpos[j] - cpos[i]) < AGREE_NEAR_M]
            near = [clouds[j] for _, j in sorted(others, key=lambda t: t[0])[:AGREE_MAX]]
        reach_i = min(max_range, GUESSED_REACH_M) if p["id"] in guessed else max_range
        for col, zd, c2w, name, wm in stop_faces(frames, node, UP[i], rng, pano, hole, anchor, conf=conf, levelled=lev, face=face, max_range=reach_i,
                                                  edge_rel=0.08 if sphere else 0.12):
            if near:
                fac, share = agreement(zd, c2w, f, near, dev)
                wm = fac if wm is None else wm * fac
                agree_stats.append(share)
            vol.integrate(zd, col, K, np.linalg.inv(c2w), cam_pos=c2w[:3, 3], wmap=wm, max_range=max_range)
        used.append(p["id"])
    if agree_stats:
        log(f"neighbours' check: {np.mean(agree_stats) * 100:.0f}% of each stop's depth confirmed by another stop (median face {np.median(agree_stats) * 100:.0f}%)")
    log(f"fused {len(used)} stops")
    V, T, C = vol.mesh()
    del vol; torch.cuda.empty_cache()
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(T))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(C / 255.0, 0, 1))
    return mesh, used, voxel


def clean(mesh, min_island_tris=400, max_tris=300000, smooth_iters=6, floor_snap=0.04, voxel=None):
    """Drop floaters and small islands, flatten the floor, smooth the voxel stairs a little, then decimate."""
    import open3d as o3d
    if voxel and len(mesh.triangles) > 3 * max_tris:                                      # a 3 cm volume comes out of marching cubes with millions of slivers: cluster them first (memory)
        mesh = mesh.simplify_vertex_clustering(voxel_size=voxel * 1.25, contraction=o3d.geometry.SimplificationContraction.Average)
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_vertices(); mesh.remove_unreferenced_vertices()
    tri_cl, n_tri, area = mesh.cluster_connected_triangles()
    tri_cl, n_tri = np.asarray(tri_cl), np.asarray(n_tri)
    keep_cl = np.flatnonzero(n_tri >= max(min_island_tris, 0.01 * n_tri.max()))
    mesh.remove_triangles_by_mask(~np.isin(tri_cl, keep_cl))
    mesh.remove_unreferenced_vertices()
    if smooth_iters:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)
    V = np.asarray(mesh.vertices)
    if floor_snap:                                                                        # the floor is z = 0 by construction: take the voxel ripple out of it
        z = V[:, 2]; m = np.abs(z) < floor_snap
        V[m, 2] = 0.0
        mesh.vertices = o3d.utility.Vector3dVector(V)
    before = len(mesh.triangles)
    if before > max_tris:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max_tris)
        mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    return mesh, before


ROOF_CUT_MAX = 2.6              # the same cut the plan makes (floorplan.py): the viewer's dollhouse looks in from above at this height at most


def roof_cut(ceiling_m):
    """Where the dollhouse's roof comes off, in metres: just under the ceiling the model found, never higher than ROOF_CUT_MAX - above that are the joists,
    the ducts and the lights, which you would look in past. ROOF_CUT_MAX when there is no ceiling (an outdoor walk keeps the viewer's own rule)."""
    return round(float(min(ceiling_m - 0.2, ROOF_CUT_MAX)), 2) if ceiling_m else ROOF_CUT_MAX


def ceiling_height(V, T):
    """Where the ceiling begins, in metres: the lowest height (area-weighted 5th percentile) of the model's downward-facing surfaces above 2.2 m — the joists'
    undersides in a shop, the ceiling plane in a room. The viewer cuts the dollhouse's roof just under it, so the walls stand to their full height. None when
    nothing faces down up there (an outdoor walk)."""
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]); area = np.linalg.norm(n, axis=1) / 2
    down = (n[:, 2] / np.maximum(area * 2, 1e-12)) < -0.5
    z = V[T].mean(1)[:, 2]
    m = down & (z > 2.2) & (area > 0)
    if m.sum() < 50 or area[m].sum() < 2.0:
        return None
    o = np.argsort(z[m]); zs = z[m][o]; w = np.cumsum(area[m][o]) / area[m].sum()
    return round(float(zs[np.searchsorted(w, 0.05)]), 2)


def scene_axis(V, T):
    """The place's main axis, degrees from east (0..180): the direction its vertical surfaces run (floorplan.main_axis_normals), else the long axis of its
    footprint (an outdoor walk with no walls). The viewer opens the dollhouse looking across it, the long side across the screen."""
    try:
        import floorplan as FP
        n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]); area = np.linalg.norm(n, axis=1) / 2
        vert = (area > 1e-9) & (np.abs(n[:, 2]) / np.maximum(2 * area, 1e-12) < 0.5)
        if area[vert].sum() > 0.15 * area.sum() and area[vert].sum() > 20:
            th = FP.main_axis_normals(V, T)                                # the walls' normal direction; the walls RUN across it
            return round(float((th + 90.0) % 180.0), 1)
    except Exception:
        pass
    X = V[:, :2] - V[:, :2].mean(0)
    w, v = np.linalg.eigh(X.T @ X)
    d = v[:, int(np.argmax(w))]
    return round(float(math.degrees(math.atan2(d[1], d[0])) % 180.0), 1)


def export(mesh, out_dir, name="dollhouse", atlas=None, tex_meta=None, uv=None):
    """A small quantised binary. v1 (block texture or vertex colours): positions int16 (relative to the box), colours uint8, indices uint16|uint32.
    v2 (a UV atlas): positions int16, uv uint16, indices uint32 — layout "v2"."""
    import open3d as o3d
    if uv is not None:
        V2, T2, UV2 = uv
        lo, hi = V2.min(0), V2.max(0); span = np.maximum(hi - lo, 1e-6)
        Q = np.round((V2 - lo) / span * 65535 - 32768).astype(np.int16)
        U = np.round(np.clip(UV2, 0, 1) * 65535).astype(np.uint16)
        idx = T2.astype(np.uint32)
        blob = Q.tobytes() + U.tobytes() + idx.tobytes()
        meta = dict(vertices=int(len(V2)), triangles=int(len(T2)), lo=[float(x) for x in lo], span=[float(x) for x in span], index_bytes=4, bytes=len(blob),
                    layout="v2: int16[3*v] uint16[2*v] uint32[3*t]", frame="X east, Y north, Z up (metres), floor at Z=0", ceiling_m=ceiling_height(V2, T2),
                    axis_deg=scene_axis(V2, T2))
        meta["cut_m"] = roof_cut(meta["ceiling_m"])
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, name + ".jpg"), cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        meta["tex"] = dict(tex_meta, file=name + ".jpg", uv=True, bytes=os.path.getsize(os.path.join(out_dir, name + ".jpg")))
    else:
        V = np.asarray(mesh.vertices, np.float64); T = np.asarray(mesh.triangles); C = np.clip(np.asarray(mesh.vertex_colors) * 255, 0, 255).astype(np.uint8)
        lo, hi = V.min(0), V.max(0); span = np.maximum(hi - lo, 1e-6)
        Q = np.round((V - lo) / span * 65535 - 32768).astype(np.int16)
        idx = T.astype(np.uint16 if len(V) < 65536 else np.uint32)
        blob = Q.tobytes() + C.tobytes() + idx.tobytes()
        meta = dict(vertices=int(len(V)), triangles=int(len(T)), lo=[float(x) for x in lo], span=[float(x) for x in span], index_bytes=int(idx.itemsize),
                    bytes=len(blob), layout="int16[3*v] uint8[3*v] uint16|uint32[3*t]", frame="X east, Y north, Z up (metres), floor at Z=0")
        os.makedirs(out_dir, exist_ok=True)
        if atlas is not None:
            cv2.imwrite(os.path.join(out_dir, name + ".jpg"), cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
            meta["tex"] = dict(tex_meta, file=name + ".jpg", bytes=os.path.getsize(os.path.join(out_dir, name + ".jpg")))
            meta["layout"] += "; texture: one block per triangle"
    with open(os.path.join(out_dir, name + ".bin"), "wb") as fh:
        fh.write(blob)
    with open(os.path.join(out_dir, name + ".json"), "w") as fh:
        json.dump(meta, fh)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames"); ap.add_argument("--out", default="tour")
    ap.add_argument("--voxel", type=float, default=0.03); ap.add_argument("--max-tris", type=int, default=300000)
    ap.add_argument("--trunc", type=float, default=None, help="TSDF truncation (m), default 4 voxels: surfaces thinner than about this get merged")
    ap.add_argument("--only", nargs="*"); ap.add_argument("--depth", default="auto", help="folder under frames/ with the depth: depth_sphere (full-sphere metric), depth_mvs, depth; auto = the best there is")
    ap.add_argument("--no-texture", action="store_true", help="vertex colours only (the old look)")
    ap.add_argument("--block-texture", action="store_true", help="the old one-block-per-triangle texture instead of the UV atlas")
    ap.add_argument("--atlas", type=int, default=0, help="the atlas's size in px; 0 = by the surface (4096 for a room, 8192 for a shop)")
    ap.add_argument("--mesh-out", default=None, help="write frames/dollhouse/<name>.ply instead of mesh.ply (experiments)")
    a = ap.parse_args()
    t0 = time.time()
    depth_dir = a.depth
    if depth_dir == "auto":
        depth_dir = next((d for d in ("depth_sphere", "depth_mvs", "depth") if os.path.exists(os.path.join(a.frames, d, "stats.json")) or
                          (d == "depth" and os.path.isdir(os.path.join(a.frames, d)))), "depth")
        if depth_dir == "depth_mvs" and json.load(open(os.path.join(a.frames, "depth_mvs", "stats.json"))).get("skipped"):
            depth_dir = "depth"
    mesh, used, voxel = fuse(a.frames, a.voxel, trunc=a.trunc, ids=a.only, depth_dir=depth_dir)
    mesh, raw_tris = clean(mesh, max_tris=a.max_tris, voxel=voxel)
    import open3d as o3d
    os.makedirs(os.path.join(a.frames, "dollhouse"), exist_ok=True)
    atlas = tex_meta = uv = None
    o3d.io.write_triangle_mesh(os.path.join(a.frames, "dollhouse", "raw_clean_" + (a.mesh_out or "mesh") + ".ply"), mesh)
    name = a.mesh_out or "mesh"
    if not a.no_texture:
        import dollhouse_texture as DT
        stops = DT.stops_for(a.frames, a.only)
        if a.block_texture:
            mesh, atlas, tex_meta = DT.texture_mesh(a.frames, mesh, stops)
        else:
            import dollhouse_uv as DU
            size = a.atlas
            if size <= 0:                                                   # by the surface: about 5 mm a texel, 4096 for a room, 8192 only past 120 m2 of surface
                Vm = np.asarray(mesh.vertices); Tm = np.asarray(mesh.triangles)
                area = float(np.linalg.norm(np.cross(Vm[Tm[:, 1]] - Vm[Tm[:, 0]], Vm[Tm[:, 2]] - Vm[Tm[:, 0]]), axis=1).sum() / 2)
                size = 8192 if area > 120 else 4096
                print(f"  atlas {size} px for {area:.0f} m2 of surface")
            V2, T2, UV2, atlas, tex_meta = DU.texture_mesh(a.frames, mesh, stops, size=size, flat_z=getattr(fuse, "last_z_sea", None))
            uv = (V2, T2, UV2)
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V2.astype(np.float64)), o3d.utility.Vector3iVector(T2.astype(np.int32)))
            np.savez_compressed(os.path.join(a.frames, "dollhouse", name + "_uv.npz"), V=V2.astype(np.float32), T=T2.astype(np.int32), UV=UV2.astype(np.float32))
    o3d.io.write_triangle_mesh(os.path.join(a.frames, "dollhouse", name + ".ply"), mesh)
    if atlas is not None:
        cv2.imwrite(os.path.join(a.frames, "dollhouse", name + "_atlas.jpg"), cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        json.dump(tex_meta, open(os.path.join(a.frames, "dollhouse", name + "_atlas.json"), "w"))
    if a.mesh_out:
        print(f"experiment mesh {name}.ply: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles, {time.time() - t0:.0f} s")
        return
    meta = export(mesh, a.out, atlas=atlas, tex_meta=tex_meta, uv=uv)
    meta.update(stops=used, raw_triangles=int(raw_tris), seconds=round(time.time() - t0, 1), voxel=voxel, depth=depth_dir)
    json.dump(meta, open(os.path.join(a.frames, "dollhouse", "stats.json"), "w"), indent=1)
    tj = os.path.join(a.out, "tour.json")
    if os.path.exists(tj):                                       # tell the viewer there is a dollhouse (build_tour.py does the same on a rebuild)
        with open(tj) as fh:
            t = json.load(fh)
        t["dollhouse"] = True
        with open(tj, "w") as fh:
            json.dump(t, fh, separators=(",", ":"))
    print(f"dollhouse: {meta['vertices']} vertices, {meta['triangles']} triangles, {(meta['bytes'] + meta.get('tex', {}).get('bytes', 0)) / 1e6:.2f} MB, {len(used)} stops, voxel {voxel} m, {meta['seconds']} s")


if __name__ == "__main__":
    main()
