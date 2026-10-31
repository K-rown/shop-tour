"""Multi-view stereo on the stops' own panoramas: a spherical plane sweep on the GPU with the solved poses, geometric-consistency filtering.

Usage:  python mvs.py [--frames frames] [--size 1024] [--depths 384] [--views 8] [--only 003 004]
Reads:  frames/safe/<id>.jpg (cleaned pictures; held-back stops are never read), frames/poses.json, frames/up.json, frames/capturer/<id>_hole.png
Writes: frames/mvs/<id>.npz  range (float16, metres along each ray, 0 = none) and cost (float16) at size x size/2, frames/mvs/stats.json

Why not COLMAP's patch-match: the pip build of COLMAP has no CUDA (pycolmap.has_cuda is False) and patch-match stereo needs it. This is the same idea
(photo-consistency over depth hypotheses between calibrated views), written for equirectangular images so no cube-map cutting is needed.

For every reference stop: the nearest few other stops are the neighbours; for each of `depths` inverse-depth hypotheses every neighbour is sampled where the
hypothesised 3D point lands in it, and compared to the reference with zero-mean normalised cross-correlation over a 5x5 window; the matching cost of a depth is
the mean of the 3 best neighbours (a neighbour that sees something else in front of the point is simply not among the best). Then: parabola refinement,
a uniqueness test, and a geometric-consistency test (the neighbours' own depth maps must agree on the 3D point).
The person who held the camera (the hole mask) is never matched.
"""
import argparse, json, math, os, sys, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from imutil import imread                     # noqa: E402
import dollhouse as D                          # noqa: E402

DMIN, DMAX = 0.4, 40.0
WIN = 7


def poses_not_confirmed(frames):
    """None if an independent sensor has confirmed the stop poses, else the reason it has not. Stereo is only as good as the poses (a few cm of error in a stop's position
    turns a wall into a ghost), and on a walk where the camera path and the stop map disagree it made the model worse (median measurement error 27 % against 39 % on one test walk), so it needs this."""
    try:
        info = json.load(open(os.path.join(frames, "poses.json"))).get("scale_info", {})
    except (OSError, ValueError):
        return "no poses.json"
    if str(info.get("source", "")).startswith("accelerometer"):
        return None
    return "the accelerometer did not confirm the stop map (scale from: %s)" % info.get("source", "unknown")


def load_stops(frames, only=None):
    pts = json.load(open(os.path.join(frames, "points.json")))
    P = json.load(open(os.path.join(frames, "poses.json")))["nodes"]; UP = json.load(open(os.path.join(frames, "up.json")))
    out = []
    for i, p in enumerate(pts):
        sp = os.path.join(frames, "safe", p["filename"])
        if only and p["id"] not in only or not os.path.exists(sp) or P[i].get("x") is None:
            continue
        pos = np.array([P[i]["x"], P[i]["y"], (P[i].get("cam_h") or 2.0) + float(P[i].get("z") or 0.0)])
        Wm = D.level_to_world(P[i]["H"]) @ D.rot_to_y(UP[i])             # pano ray -> world ray
        out.append(dict(i=i, id=p["id"], file=sp, pos=pos, W=Wm, up=UP[i]))
    return out


def sphere_dirs(W, H, dev):
    az = (torch.arange(W, device=dev) + 0.5) / W * 2 * math.pi - math.pi
    el = math.pi / 2 - (torch.arange(H, device=dev) + 0.5) / H * math.pi
    A, E = torch.meshgrid(az, el, indexing="xy")                        # (H, W)
    return torch.stack([torch.cos(E) * torch.sin(A), torch.sin(E), torch.cos(E) * torch.cos(A)], -1)


def load_gray(path, W, H):
    im = cv2.resize(imread(path), (W, H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def box(x, k=WIN):
    """Mean over a k x k window, wrapping around in azimuth, replicating at the poles. x: (1, C, H, W)."""
    p = k // 2
    x = F.pad(x, (p, p, 0, 0), mode="circular"); x = F.pad(x, (0, 0, p, p), mode="replicate")
    return F.avg_pool2d(x, k, stride=1)


def sample_sphere(img_pad, v, W, H, pad):
    """img_pad: (1, 1, H, W + 2*pad) (circular padded), v: (H', W', 3) directions (any length) in the image's frame -> samples (1, 1, H', W')."""
    az = torch.atan2(v[..., 0], v[..., 2]); el = torch.asin((v[..., 1] / v.norm(dim=-1).clamp(min=1e-9)).clamp(-1, 1))
    x = (az + math.pi) / (2 * math.pi) * W - 0.5 + pad; y = (math.pi / 2 - el) / math.pi * H - 0.5
    gx = (x + 0.5) / (W + 2 * pad) * 2 - 1; gy = (y + 0.5) / H * 2 - 1
    return F.grid_sample(img_pad, torch.stack([gx, gy], -1)[None], mode="bilinear", padding_mode="border", align_corners=False)


class Level:
    """One resolution of one reference: the reference statistics and, per neighbour, what is needed to look a hypothesised 3D point up in it."""
    def __init__(self, ref, nbrs, gray, hole, dev):
        _, _, H, W = gray[ref["id"]].shape
        self.W, self.H, self.dev, self.pad = W, H, dev, 8
        self.d_ref = sphere_dirs(W, H, dev)
        self.R = gray[ref["id"]]
        self.mu_r = box(self.R); self.var_r = (box(self.R * self.R) - self.mu_r ** 2).clamp(min=0)
        Wi = torch.tensor(ref["W"], dtype=torch.float32, device=dev)
        self.prep = []
        for nb in nbrs:
            Wj = torch.tensor(nb["W"], dtype=torch.float32, device=dev)
            c = Wj.T @ torch.tensor(ref["pos"] - nb["pos"], dtype=torch.float32, device=dev)
            Ad = self.d_ref @ (Wj.T @ Wi).T
            img_pad = F.pad(gray[nb["id"]], (self.pad, self.pad, 0, 0), mode="circular")
            hp = F.pad(hole[nb["id"]].float(), (self.pad, self.pad, 0, 0), mode="circular") if hole.get(nb["id"]) is not None else None
            self.prep.append((c, Ad, img_pad, hp))
        self.costs = torch.empty((len(nbrs), H, W), dtype=torch.float32, device=dev)

    def cost(self, z, topk=3):
        """Aggregated matching cost (H, W) for the range z (scalar, or (H, W, 1) per pixel): the mean of the `topk` best neighbours' 1 - ZNCC."""
        for n, (c, Ad, img_pad, hp) in enumerate(self.prep):
            v = c + z * Ad
            s = sample_sphere(img_pad, v, self.W, self.H, self.pad)
            mu_s = box(s); var_s = (box(s * s) - mu_s ** 2).clamp(min=0); cov = box(self.R * s) - self.mu_r * mu_s
            cst = 1.0 - (cov / torch.sqrt((self.var_r + 1e-4) * (var_s + 1e-4))).clamp(-1, 1)
            if hp is not None:
                cst = torch.where(sample_sphere(hp, v, self.W, self.H, self.pad) > 0.01, torch.full_like(cst, 2.0), cst)
            self.costs[n] = cst[0, 0]
        return torch.topk(self.costs, min(topk, len(self.prep)), dim=0, largest=False).values.mean(0)


def parabola(c0, c1, c2):
    den = c0 - 2 * c1 + c2
    return torch.where(den.abs() > 1e-6, 0.5 * (c0 - c2) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)


def sweep_coarse(lv, nd):
    """Uniform inverse-depth sweep at the coarse level -> inverse depth (H, W), cost, second-best cost outside +-4 steps."""
    inv = torch.linspace(1.0 / DMAX, 1.0 / DMIN, nd, device=lv.dev)
    vol = torch.empty((nd, lv.H, lv.W), dtype=torch.float16, device=lv.dev)
    for k in range(nd):
        vol[k] = lv.cost(1.0 / inv[k]).half()
    v = vol.float(); best, k = v.min(0)
    kc = k.clamp(1, nd - 2)
    off = parabola(torch.gather(v, 0, (kc - 1)[None])[0], torch.gather(v, 0, kc[None])[0], torch.gather(v, 0, (kc + 1)[None])[0])
    step = float(inv[1] - inv[0])
    inv_d = inv[0] + (k.float() + torch.where((k > 0) & (k < nd - 1), off, torch.zeros_like(off))) * step
    idx = torch.arange(nd, device=lv.dev)[:, None, None]
    second = torch.where((idx - k[None]).abs() > 4, v, torch.full_like(v, 9.0)).min(0).values
    del vol, v
    return inv_d, best, second, step


def sweep_fine(lv, inv0, step, nf=17, span=1.5):
    """Around the coarse inverse depth (upsampled to this level), +-span coarse steps in nf hypotheses -> inverse depth (H, W), cost, second-best cost."""
    offs = torch.linspace(-span * step, span * step, nf, device=lv.dev)
    vol = torch.empty((nf, lv.H, lv.W), dtype=torch.float16, device=lv.dev)
    for k in range(nf):
        inv = (inv0 + offs[k]).clamp(min=1.0 / DMAX)
        vol[k] = lv.cost((1.0 / inv)[..., None]).half()
    v = vol.float(); best, k = v.min(0)
    kc = k.clamp(1, nf - 2)
    off = parabola(torch.gather(v, 0, (kc - 1)[None])[0], torch.gather(v, 0, kc[None])[0], torch.gather(v, 0, (kc + 1)[None])[0])
    inv_d = inv0 + offs[0] + (k.float() + torch.where((k > 0) & (k < nf - 1), off, torch.zeros_like(off))) * (offs[1] - offs[0])
    idx = torch.arange(nf, device=lv.dev)[:, None, None]
    second = torch.where((idx - k[None]).abs() > 3, v, torch.full_like(v, 9.0)).min(0).values
    del vol, v
    return inv_d.clamp(min=1.0 / DMAX), best, second


def consistency(results, stops, dev, W, H, tol=0.03):
    """How many other stops' depth maps agree with each pixel's 3D point (within tol of the range)."""
    d_ref = sphere_dirs(W, H, dev)
    count = {}
    for a in stops:
        ra = results[a["id"]]["range"]
        valid = results[a["id"]]["valid"]
        Wa = torch.tensor(a["W"], dtype=torch.float32, device=dev)
        Xw = a["pos_t"] + (ra[..., None] * (d_ref @ Wa.T))
        n = torch.zeros((H, W), dtype=torch.int16, device=dev)
        for b in stops:
            if b["id"] == a["id"]:
                continue
            dist = float(np.linalg.norm(a["pos"] - b["pos"]))
            if dist > 12:
                continue
            Wb = torch.tensor(b["W"], dtype=torch.float32, device=dev)
            vb = (Xw - b["pos_t"]) @ Wb                                  # world offset -> b's pano frame (W^T x == x @ W)
            rb = vb.norm(dim=-1)
            az = torch.atan2(vb[..., 0], vb[..., 2]); el = torch.asin((vb[..., 1] / rb.clamp(min=1e-6)).clamp(-1, 1))
            u = ((az + math.pi) / (2 * math.pi) * W - 0.5).round().long() % W; v = ((math.pi / 2 - el) / math.pi * H - 0.5).round().long().clamp(0, H - 1)
            rd = results[b["id"]]["range"][v, u]; vd = results[b["id"]]["valid"][v, u]
            n += (vd & valid & ((rb - rd).abs() < tol * rb + 0.03)).short()
        count[a["id"]] = n
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames"); ap.add_argument("--size", type=int, default=1024); ap.add_argument("--depths", type=int, default=384)
    ap.add_argument("--views", type=int, default=8); ap.add_argument("--only", nargs="*"); ap.add_argument("--max-cost", type=float, default=0.45)
    ap.add_argument("--min-consistent", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="run even when the accelerometer has not confirmed the stop poses")
    a = ap.parse_args()
    why = poses_not_confirmed(a.frames)
    if not why and not (a.force or os.environ.get("SHOP_TOUR_STEREO")):
        why = "stereo anchoring is off by default: on the test walks it did not beat the depth model alone (median error 9.8 % against 9.2 %, length 6.2 % against 4.7 %); SHOP_TOUR_STEREO=1 or --force runs it"
    if why and not a.force:
        os.makedirs(os.path.join(a.frames, "mvs"), exist_ok=True)
        json.dump(dict(skipped=why), open(os.path.join(a.frames, "mvs", "stats.json"), "w"), indent=1)
        print("stereo skipped:", why, "(--force runs it anyway)")
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H = a.size, a.size // 2
    stops = load_stops(a.frames)
    ids_ref = set(a.only) if a.only else {s["id"] for s in stops}
    t0 = time.time()
    gray, hole = {}, {}
    UP = None
    for s in stops:
        gray[s["id"]] = torch.from_numpy(load_gray(s["file"], W, H))[None, None].to(dev)
        hp = os.path.join(a.frames, "capturer", s["id"] + "_hole.png")
        if os.path.exists(hp):
            m = D.hole_in_eq(cv2.imread(hp, 0) > 0, s["up"], W, H)
            m = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            hole[s["id"]] = torch.from_numpy(m)[None, None].to(dev)
        s["pos_t"] = torch.tensor(s["pos"], dtype=torch.float32, device=dev)
    print(f"loaded {len(stops)} stops in {time.time() - t0:.0f} s", flush=True)
    os.makedirs(os.path.join(a.frames, "mvs"), exist_ok=True)
    results = {}
    for s in stops:
        if s["id"] not in ids_ref:
            continue
        t1 = time.time()
        dist = sorted(((np.linalg.norm(s["pos"][:2] - o["pos"][:2]), o) for o in stops if o["id"] != s["id"]), key=lambda x: x[0])
        nbrs = [o for d_, o in dist if 0.8 <= d_ <= 12][:a.views]
        if len(nbrs) < 2:
            print(f"{s['id']}: fewer than two neighbours with a baseline, skipped"); continue
        lv0 = Level(s, nbrs, gray, hole, dev)
        inv0, c0, sec0, step = sweep_coarse(lv0, a.depths)
        inv1, best, second = sweep_fine(lv0, inv0, step)
        rng = 1.0 / inv1
        del lv0; torch.cuda.empty_cache()
        h = hole.get(s["id"])
        valid = (best < a.max_cost) & (second - best > 0.02)
        if h is not None:
            valid &= ~h[0, 0]
        results[s["id"]] = dict(range=rng, cost=best, valid=valid, nbrs=[o["id"] for o in nbrs])
        print(f"{s['id']}: {int(valid.sum()) / valid.numel() * 100:.0f}% pixels matched, {len(nbrs)} views, {time.time() - t1:.1f} s", flush=True)
    cons = consistency(results, [s for s in stops if s["id"] in results], dev, W, H)
    stats = {}
    for sid, r in results.items():
        n = cons[sid]; ok = r["valid"] & (n >= a.min_consistent)
        rg = torch.where(ok, r["range"], torch.zeros_like(r["range"]))
        np.savez_compressed(os.path.join(a.frames, "mvs", sid + ".npz"), range=rg.cpu().numpy().astype(np.float16), cost=r["cost"].cpu().numpy().astype(np.float16),
                            consistent=n.cpu().numpy().astype(np.uint8))
        stats[sid] = dict(matched_pct=round(float(r["valid"].float().mean() * 100), 1), kept_pct=round(float(ok.float().mean() * 100), 1), views=r["nbrs"])
    json.dump(dict(size=W, depths=a.depths, views=a.views, max_cost=a.max_cost, min_consistent=a.min_consistent, seconds=round(time.time() - t0), stops=stats), open(os.path.join(a.frames, "mvs", "stats.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f} s; kept {np.mean([v['kept_pct'] for v in stats.values()]):.0f}% of pixels on average")


if __name__ == "__main__":
    main()
