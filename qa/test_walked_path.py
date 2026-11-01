"""The walked path (build_tour.walked_path): the walk's 1 Hz motion track laid onto the tour's map by local similarity fits.
    python qa/test_walked_path.py
A synthetic chain under a known rotation + scale + shift must come back on the map within centimetres; a missing chain or too few
stops is no path; a chain that drifts in scale is still laid right where the local fits hold."""
import json, math, os, sys, tempfile, unittest
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import build_tour  # noqa: E402


def make(tmp, n_stops=8, drift=0.0, seed=1):
    rng = np.random.default_rng(seed)
    T = np.arange(0, 60.0, 1.0)                                              # 60 s, one chain position a second
    # the true walk on the map: a gentle arc, 1 m/s, camera 1.8 m up
    truth = np.stack([T * 1.0 + 0.3 * np.sin(T / 5), 4 * np.sin(T / 12), 1.8 + 0.05 * np.sin(T)], 1)
    th, s = 0.7, 0.25                                                        # the chain's own frame: turned, scaled, shifted
    Q = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    Cv = ((truth - [3, 1, 0]) @ Q.T) / s * (1 + drift * T / 60)[:, None]
    stop_i = np.linspace(0, len(T) - 1, n_stops).astype(int)
    names, C = [], []
    for i in range(len(T)):
        if i in stop_i:
            names.append(f"s_{list(stop_i).index(i) + 1:03d}.jpg")
        else:
            names.append(f"f_{i:04d}.jpg")
        C.append(Cv[i])
    vio = os.path.join(tmp, "vio"); os.makedirs(vio, exist_ok=True)
    np.savez(os.path.join(vio, "poses.npz"), names=np.array(names), t=T, C=np.array(C), R=np.tile(np.eye(3), (len(T), 1, 1)))
    pts = [{"id": f"{k + 1:03d}", "timestamp": float(T[i])} for k, i in enumerate(stop_i)]
    MX = [float(truth[i, 0]) for i in stop_i]; MY = [float(truth[i, 1]) for i in stop_i]
    ZS = [0.0] * n_stops; CAMH = [float(truth[i, 2]) for i in stop_i]
    return pts, MX, MY, ZS, CAMH, truth


class TestWalkedPath(unittest.TestCase):
    def test_recovers_the_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            pts, MX, MY, ZS, CAMH, truth = make(tmp)
            path = build_tour.walked_path(tmp, pts, MX, MY, ZS, CAMH, list(range(len(pts))), log=lambda *a: None)
            self.assertIsNotNone(path); self.assertGreater(len(path), 20)
            # every path point lies within 0.15 m of the true walk
            worst = max(min(math.hypot(x - tx, y - ty) for tx, ty in truth[:, :2]) for x, y in path)
            self.assertLess(worst, 0.15, worst)
            self.assertLess(path[0][0], path[-1][0])                        # in time order (the walk goes +x)

    def test_drifting_chain_is_laid_locally(self):
        with tempfile.TemporaryDirectory() as tmp:
            pts, MX, MY, ZS, CAMH, truth = make(tmp, n_stops=12, drift=0.6)
            path = build_tour.walked_path(tmp, pts, MX, MY, ZS, CAMH, list(range(len(pts))), log=lambda *a: None)
            self.assertIsNotNone(path)
            worst = max(min(math.hypot(x - tx, y - ty) for tx, ty in truth[:, :2]) for x, y in path)
            self.assertLess(worst, 0.6, worst)

    def test_no_chain_or_too_few_stops_is_no_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            pts, MX, MY, ZS, CAMH, truth = make(tmp)
            self.assertIsNone(build_tour.walked_path(tmp, pts, MX, MY, ZS, CAMH, [0, 1], log=lambda *a: None))
            self.assertIsNone(build_tour.walked_path(os.path.join(tmp, "nowhere"), pts, MX, MY, ZS, CAMH, list(range(len(pts))), log=lambda *a: None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
