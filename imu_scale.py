"""True scale from the camera's own accelerometer, no user input (how a phone gets metric scale in AR): the visual-inertial scale.

Usage:  python imu_scale.py <video.insv> [--frames frames] [--json frames/scale_imu.json]
Needs:  vio_map.py has run (frames/vio/poses.npz: the camera path at 1 Hz and its orientation, in COLMAP's arbitrary units), frames/poses.json (the tour's map).
Writes: frames/scale_imu.json: metres per COLMAP unit, the factor to apply to the tour's own map (poses.json), and how well the two agree.

Idea: the path the pictures give is right up to ONE number s (metres per map unit). The accelerometer measures real acceleration in m/s^2. A path that is
sampled at times t0 < t1 < t2 has the second divided difference  p[t0,t1,t2] = integral of hat(t) * p''(t) dt  (hat: a triangle on [t0, t2], area 1/2), which
the accelerometer can compute exactly from its 1000 samples a second:  integral of hat(t) * a(t) dt.  So   s * p_map[t0,t1,t2] = integral hat * (acceleration)  for
every triple of neighbouring pictures: a straight-line fit for s. Gravity is part of what the accelerometer feels (the specific force f = a + g*up); the
orientation of every picture (COLMAP) turns f into the map frame; what is not known and is solved together with s: how the sensor is mounted relative to the
camera (a rotation), the direction of up in the map, the sensor's constant bias and the time offset between the sensor clock and the video clock.
No tape measure, no known object, no camera height: the only constants used are gravity (9.80665 m/s^2) and the sensor's own timestamps.
(The first attempt, integrating the accelerometer twice between stops with zero-velocity updates, failed: experiments/zupt_scale_failed.py.)
"""
import argparse, itertools, json, math, os, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import imu as IMU    # noqa: E402

G = 9.80665
RATE = 100           # the sensor is used at this many samples a second (mean of 10; 200 gives the same answer to 0.3 % at twice the time)


def prepare(video):
    d = IMU.read_imu(video)
    k = int(round(d["rate"] / RATE)); n = (len(d["t"]) // k) * k
    t = d["t"][:n].reshape(-1, k).mean(1); f = d["acc"][:n].reshape(-1, k, 3).mean(1); w = d["gyro"][:n].reshape(-1, k, 3).mean(1)
    return t, f, w


class Problem:
    def __init__(self, t_img, C, Rcw, t_imu, f_imu, gyro, widths=(2, 3), max_span=10.0):
        self.t_img, self.C, self.t_imu, self.f, self.gyro = t_img, C, t_imu, f_imu, gyro
        self.Rwc_img = np.transpose(Rcw, (0, 2, 1)); self.chain_offset = None; self.Gp = None
        self.rot = Slerp(t_img, Rot.from_matrix(np.transpose(Rcw, (0, 2, 1))))        # world-from-camera, interpolated between pictures
        # triples (i-m, i, i+m): wider triples see the same acceleration as a bigger signal (it grows with the square of the width) against the same
        # position noise (a few cm of body bounce and map noise), so the scale is less biased by that noise
        self.tri = [(i - m, i, i + m) for m in widths for i in range(m, len(t_img) - m) if t_img[i + m] - t_img[i - m] < max_span]
        self.pm = np.array([self.dd(*tr) for tr in self.tri])                             # the path's second divided differences, map units
        self.win = []                                                                      # per triple: times (video clock), hat weights, world-from-camera at those times
        for (a, b, c) in self.tri:
            t0, t1, t2 = t_img[a], t_img[b], t_img[c]
            tt = np.arange(t0, t2, 1.0 / RATE) + 0.5 / RATE
            w = np.where(tt < t1, (tt - t0) / (t1 - t0), (t2 - tt) / (t2 - t1)) / (t2 - t0) / RATE
            self.win.append((tt, w, self.rot(np.clip(tt, t_img[0], t_img[-1])).as_matrix()))

    def dd(self, a, b, c):
        t0, t1, t2 = self.t_img[a], self.t_img[b], self.t_img[c]
        return self.C[a] / ((t0 - t1) * (t0 - t2)) + self.C[b] / ((t1 - t0) * (t1 - t2)) + self.C[c] / ((t2 - t0) * (t2 - t1))

    def set_offset(self, offset):
        """Orientation of the sensor between the pictures from the gyroscope (the pictures give it once a second; a hand-held stick tilts several degrees
        inside that second, and a tilt error of 1 degree leaks 0.17 m/s^2 of gravity into the horizontal acceleration, as much as the walk itself).
        For every sensor sample: the rotation since the last picture, integrated from the gyroscope (sensor frame)."""
        if self.chain_offset == offset:
            return
        tv = self.t_imu - offset
        k = np.clip(np.searchsorted(self.t_img, tv, side="right") - 1, 0, len(self.t_img) - 2)
        if self.Gp is None:                                           # prefix products of the gyro increments, once: Gp[j] = inc_0 ... inc_(j-1)
            dt = np.median(np.diff(self.t_imu))
            inc = Rot.from_rotvec(self.gyro * dt).as_matrix()
            Gp = np.empty((len(inc) + 1, 3, 3)); Gp[0] = np.eye(3)
            for j in range(len(inc)):
                Gp[j + 1] = Gp[j] @ inc[j]
            self.Gp = Gp
        Gp = self.Gp
        first = np.searchsorted(k, np.arange(len(self.t_img) - 1), side="left")          # first sample of every interval (k never decreases)
        start = first[k]                                                                  # the sample at which each sample's chain began
        Q = np.swapaxes(Gp[start], 1, 2) @ Gp[np.arange(len(k))]                         # rotation since the last picture: Gp[start]^T Gp[j]
        valid = (tv >= self.t_img[0]) & (tv < self.t_img[-1])
        last = np.full(len(self.t_img) - 1, -1)
        last[k[valid]] = np.arange(len(k))[valid]
        Qend = np.stack([(Gp[first[i]].T @ Gp[last[i] + 1]) if last[i] >= 0 and first[i] < len(k) else np.eye(3) for i in range(len(self.t_img) - 1)])
        frac = np.clip((tv - self.t_img[k]) / (self.t_img[k + 1] - self.t_img[k]), 0, 1)
        self.chain = (k, frac, Q, Qend); self.chain_offset = offset
        self.t0_imu = self.t_imu[0]

    def sensor_orientation(self, Rcs):
        """World-from-sensor at every sensor sample: the picture's orientation, turned by the gyroscope, with the leftover at the next picture spread over the interval."""
        k, frac, Q, Qend = self.chain
        Rpic = self.Rwc_img @ Rcs
        M = np.swapaxes(Rpic[:-1] @ Qend, 1, 2) @ Rpic[1:]                                # (Rpic_k Qend_k)^T Rpic_{k+1}  (batched matmul: ~3x faster than einsum)
        delta = Rot.from_matrix(M).as_rotvec()
        corr = Rot.from_rotvec(frac[:, None] * delta[k]).as_matrix()
        return (Rpic[k] @ Q) @ corr

    def _gather(self, offset):
        """Per offset, once: every triple's sensor sample indices and hat weights, concatenated (a loop over 300-500 triples per residual call was 80 % of the run time)."""
        if getattr(self, "_g_off", None) == offset:
            return
        idx, wts, seg = [], [], []
        for i, (tt, w, _) in enumerate(self.win):
            ii = np.clip(np.round((tt + offset - self.t_imu[0]) * RATE).astype(int), 0, len(self.t_imu) - 1)
            idx.append(ii); wts.append(w); seg.append(np.full(len(ii), i))
        self._cat_idx = np.concatenate(idx); self._cat_w = np.concatenate(wts); self._cat_seg = np.concatenate(seg)
        self._starts = np.r_[0, np.cumsum([len(x) for x in idx])[:-1]]
        self._g_off = offset

    def residuals(self, x, offset, which=None, detail=False):
        s, rv, th, ph, b = x[0], x[1:4], x[4], x[5], x[6:9]
        Rcs = Rot.from_rotvec(rv).as_matrix()
        up = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
        self.set_offset(offset); self._gather(offset)
        Rws = self.sensor_orientation(Rcs)
        ci = self._cat_idx
        aw = (Rws * (self.f - b)[:, None, :]).sum(-1)[ci] - G * up                   # the rotated force of every sensor sample once, then looked up per triple (the triples overlap)
        side_all = np.add.reduceat(aw * self._cat_w[:, None], self._starts, axis=0)           # (triples, 3)
        sel = np.arange(len(self.tri)) if which is None else np.asarray(which)
        side = side_all[sel]
        r = (side - s * self.pm[sel]).ravel()
        return (r, side) if detail else r


def mount_guesses(P, offset):
    """The 24 ways a sensor can be mounted (proper axis permutations); the best is the one that makes 'up' (from the accelerometer, turned by each picture's
    orientation) most constant in the map. Returns [(spread, rotation, up in the map)] best first."""
    out = []
    tt = P.t_imu[::100]; fs = P.f[::100]
    ok = (tt - offset >= P.t_img[0]) & (tt - offset <= P.t_img[-1])
    Rwc = P.rot(tt[ok] - offset).as_matrix()
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for r_, (c_, s_) in enumerate(zip(perm, sg)):
                M[r_, c_] = s_
            if np.linalg.det(M) < 0:
                continue
            u = np.einsum("nij,jk,nk->ni", Rwc, M, fs[ok]); u /= np.linalg.norm(u, axis=1, keepdims=True)
            m = u.mean(0)
            out.append((float(1 - np.linalg.norm(m)), M, m / np.linalg.norm(m)))
    out.sort(key=lambda x: x[0])
    return out


def solve_scale(video, frames, log=print):
    t_imu, f, gy = prepare(video)
    V = np.load(os.path.join(frames, "vio", "poses.npz"))
    return solve_arrays(t_imu, f, gy, V["t"], V["C"], V["R"], log=log)


def fit_threads():
    """Fits run at once (SHOP_TOUR_IMU_THREADS, default: the cores up to 8; 1 = one after the other, the old way). The fits do not share anything: every one starts from its own guess, so
    the numbers are the same however many run beside each other (the fits that depend on an earlier one are redone in order - see solve_arrays)."""
    try:
        n = int(os.environ.get("SHOP_TOUR_IMU_THREADS", "") or 0)
    except ValueError:
        n = 0
    return n if n > 0 else min(8, os.cpu_count() or 1)


def _own(P):
    """A copy of the problem that can be fitted from another thread: the arrays are shared (never written), the caches that belong to one clock offset are its own."""
    import copy
    return copy.copy(P)


def _map(fn, items, n):
    if n <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(n, len(items))) as ex:
        return list(ex.map(fn, items))


def solve_arrays(t_imu, f, gy, t_img, C, Rcw, offsets=None, log=print):
    """t_imu (s, sensor clock), f (specific force, m/s^2) and gy (rad/s) at RATE Hz, sensor frame; t_img (s, video clock), C (map units) and Rcw (camera from world) of the pictures."""
    order = np.argsort(t_img); t_img, C, Rcw = t_img[order], C[order], Rcw[order]
    keep = np.r_[True, np.diff(t_img) > 0.05]; t_img, C, Rcw = t_img[keep], C[keep], Rcw[keep]
    P = Problem(t_img, C, Rcw, t_imu, f, gy)
    log(f"{len(t_img)} pictures, {len(P.tri)} triples", flush=True)
    nth = fit_threads()
    if P.Gp is None:
        P.set_offset(float(t_imu[0]))                                                  # (the gyro's running products are made once, here, before the fits share the problem)
    results = []

    def start_fit(offset):
        Pi = _own(P)
        spread, M, u0 = mount_guesses(Pi, offset)[0]
        th0 = math.acos(np.clip(u0[2], -1, 1)); ph0 = math.atan2(u0[1], u0[0])
        x0 = np.r_[1.0, Rot.from_matrix(M).as_rotvec(), th0, ph0, np.zeros(3)]
        sol = least_squares(Pi.residuals, x0, args=(offset,), loss="soft_l1", f_scale=0.05, max_nfev=40)
        return (sol.cost, float(offset), sol)
    results = _map(start_fit, list(np.arange(t_imu[0] - 0.5, t_imu[0] + 2.51, 0.25) if offsets is None else offsets), nth)
    results.sort(key=lambda r: r[0])
    cost, offset, sol = results[0]
    # refine the time offset: each step starts from the best fit so far, so a step that follows an improvement starts from something else than the first ones did. The steps are
    # all tried from the first fit at once, then taken in order; one that came after an improvement is fitted again from the improved fit (rare: the best start is usually already best)
    offs = list(np.arange(offset - 0.2, offset + 0.21, 0.05))
    first = _map(lambda o: least_squares(_own(P).residuals, sol.x, args=(o,), loss="soft_l1", f_scale=0.05, max_nfev=30), offs, nth)
    start_x = sol.x
    for off2, s2 in zip(offs, first):
        if sol.x is not start_x:
            s2 = least_squares(P.residuals, sol.x, args=(off2,), loss="soft_l1", f_scale=0.05, max_nfev=30)
        if s2.cost < cost:
            cost, offset, sol = s2.cost, float(off2), s2
    r, meas = P.residuals(sol.x, offset, detail=True)
    # Which regression? s_forward regresses the accelerometer on the path (right if the path is the exact one and the accelerometer side carries the noise: tilt
    # left over, bias, the walker's hands); s_inverse regresses the path on the accelerometer (right if the opposite holds). The noise in the path (a few cm of
    # position error and body bounce against a 2 s window) is small next to the signal, so s_forward is the estimate and s_inverse is reported as the
    # upper end of what is possible if that were wrong.
    pm = P.pm
    s_fwd = float((meas * pm).sum() / (pm * pm).sum()); s_inv = float((meas * meas).sum() / (meas * pm).sum())
    s = s_fwd
    r2 = float(np.corrcoef(meas.ravel(), pm.ravel())[0, 1] ** 2)
    n = len(P.tri)

    def sub_fit(args):
        sl_, nfev = args
        Pi = _own(P)
        sub = least_squares(lambda x: Pi.residuals(x, offset, sl_), sol.x, loss="soft_l1", f_scale=0.05, max_nfev=nfev)
        _, mm = Pi.residuals(sub.x, offset, sl_, detail=True); pp = Pi.pm[sl_]
        return float((mm * pp).sum() / (pp * pp).sum())
    rng = np.random.default_rng(0)
    blocks = [list(range(i, min(i + 15, n))) for i in range(0, n, 15)]
    picks = []
    for _ in range(12):                                                                # (the resamples are drawn in order, as before; the fits that follow do not share anything)
        pick = rng.integers(0, len(blocks), len(blocks)); picks.append([i for b in pick for i in blocks[b]])
    fits = _map(sub_fit, [(list(range(0, n // 2)), 30), (list(range(n // 2, n)), 30)] + [(sl_, 15) for sl_ in picks], nth)     # how sure: the two halves of the walk on their own, then the block bootstrap
    halves, boots = fits[:2], fits[2:]
    others = [(round(float(r_[0]), 3), round(float(r_[1] - t_imu[0]), 2), round(float(r_[2].x[0]), 3)) for r_ in results[:8]]      # (cost, clock offset, s) of the best starts
    # (the block bootstrap over the walk - blocks of 15 triples, 12 resamples, each refitted from the solution: the spread of s - was made above, beside the halves)
    return dict(s_forward=s_fwd, s_inverse=s_inv, bootstrap_std_pct=float(100 * np.std(boots) / s), bootstrap_range=[float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],scale_m_per_unit=s, clock_offset_s=offset - float(t_imu[0]), r2=float(r2), first_half=halves[0], second_half=halves[1], triples=n,
                mount_rotvec=sol.x[1:4].tolist(), bias_m_s2=sol.x[6:9].tolist(), best_starts_cost_offset_scale=others)


def solve_segments(video, frames, clock_offset_s=1.0, win=60.0, step=30.0, log=print):
    """The same fit on sliding windows of the walk (default 60 s every 30 s) for walks whose camera path is only locally right (a long walk's 1 Hz map drifts in scale, a pace change
    or a mis-registered stretch breaks the whole-walk fit). Returns a list of dicts (window, s_forward, r2, bootstrap_std_pct, clock offset), one per window with enough pictures;
    the caller decides which windows to believe (map_stops.imu_metric_scale)."""
    t_imu, f, gy = prepare(video)
    V = np.load(os.path.join(frames, "vio", "poses.npz"))
    t, C, R = V["t"], V["C"], V["R"]
    out = []
    off = t_imu[0] + clock_offset_s
    for a in np.arange(0.0, float(t.max()) - 0.5 * win, step):
        m = (t >= a) & (t < a + win)
        if m.sum() < 25:
            continue
        try:
            r = solve_arrays(t_imu, f, gy, t[m], C[m], R[m], offsets=[off - 0.1, off, off + 0.1], log=lambda *a_, **k_: None)
        except Exception as e:                                           # a window with too little motion or a degenerate fit
            log(f"window {a:.0f}-{a + win:.0f} s: no fit ({str(e)[:60]})"); continue
        out.append(dict(t0=float(a), t1=float(a + win), s_forward=r["s_forward"], r2=r["r2"], bootstrap_std_pct=r["bootstrap_std_pct"], clock_offset_s=r["clock_offset_s"], pictures=int(m.sum())))
        log(f"window {a:4.0f}-{a + win:4.0f} s: {r['s_forward']:.3f} m per unit, R2 {r['r2']:.2f}, bootstrap {r['bootstrap_std_pct']:.0f} %")
    return out


def map_ratio(frames):
    """Metres of the tour's map (poses.json) per COLMAP unit of the vio map, from the stops both contain."""
    V = np.load(os.path.join(frames, "vio", "poses.npz"))
    ext = {str(n)[2:-4]: c for n, c in zip(V["names"], V["C"]) if str(n).startswith("s_")}
    pts = json.load(open(os.path.join(frames, "points.json"))); P = json.load(open(os.path.join(frames, "poses.json")))["nodes"]
    cur = {p["id"]: np.array([P[i]["x"], P[i]["y"]]) for i, p in enumerate(pts) if p["id"] in ext and P[i].get("x") is not None}
    ids = sorted(cur)
    rat = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            de = np.linalg.norm(ext[ids[i]] - ext[ids[j]]); dc = np.linalg.norm(cur[ids[i]] - cur[ids[j]])
            if de > 1e-6 and dc > 3.0:
                rat.append(dc / de)
    return float(np.median(rat)), float(np.std(rat) / np.median(rat)), len(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video"); ap.add_argument("--frames", default="frames"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r = solve_scale(a.video, a.frames)
    k, spread, n = map_ratio(a.frames)
    r["tour_map_m_per_unit"] = k; r["tour_map_ratio_spread"] = spread; r["common_stops"] = n
    r["factor_for_tour_map"] = r["scale_m_per_unit"] / k
    print(json.dumps(r, indent=1))
    print(f"\nmetres per map unit {r['scale_m_per_unit']:.4f}; the tour's map has {k:.4f}; multiply the tour's lengths by {r['factor_for_tour_map']:.3f}")
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
