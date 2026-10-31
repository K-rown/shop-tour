"""Each stop's place and heading against its own pictures (heading_check.py), and the motion track placing a stop with the right turn
(map_stops.vio_place). No footage or GPU:  python qa/test_heading_check.py"""
import json, math, os, sys, tempfile, unittest
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import heading_check as hc      # noqa: E402

F = np.diag([1.0, -1.0, 1.0])


def rot_y(deg):
    """A world-to-camera rotation for a level camera facing `deg` clockwise from +z (COLMAP camera axes: y down)."""
    t = math.radians(deg)
    Rw = np.array([[math.cos(t), 0, -math.sin(t)], [0, 1, 0], [math.sin(t), 0, math.cos(t)]])       # world-from-ours: our +z turned to the heading
    return F @ Rw.T                                                                                  # camera-from-world, COLMAP's axes


def walk(n=8, seed=1):
    """A level walk of n stops in one plane (y = 0, as every stop of a walk is): R camera-from-world, C centres."""
    rng = np.random.default_rng(seed)
    C = {k: np.array([k * 1.5 + rng.normal(0, 0.3), 0.0, rng.normal(0, 1.5)]) for k in range(n)}
    R = {k: rot_y(rng.uniform(-180, 180)) for k in range(n)}
    return R, C


def az(v):
    return math.degrees(math.atan2(v[0], v[2]))


def pairs_of(R, C, noise=0.0, seed=2):
    rng = np.random.default_rng(seed); out = []
    ks = sorted(C)
    for a in ks:
        for b in ks:
            if a < b <= a + 3:
                d = C[b] - C[a]
                out.append((a, b, az(F @ (R[a] @ d)) + rng.normal(0, noise), az(F @ (R[b] @ -d)) + rng.normal(0, noise), 200))
    return out


class TestHeadingCheck(unittest.TestCase):
    def test_a_right_map_agrees(self):
        R, C = walk()
        L = [np.eye(3)] * len(C)
        e = hc.errors_3d(R, C, L, pairs_of(R, C, noise=1.0))
        self.assertLess(hc.summary(e)[0], 2.0)
        self.assertEqual(hc.worst_out(e)[0], [])

    def test_the_turned_stop_goes_out_alone(self):
        R, C = walk()
        P = pairs_of(R, C, noise=1.0)
        R[4] = rot_y(az(F @ (R[4] @ np.array([0, 0, 1.0]))) + 40.0)          # stop 4 turned 40 degrees off its pictures
        L = [np.eye(3)] * len(C)
        out, left = hc.worst_out(hc.errors_3d(R, C, L, P))
        self.assertEqual(out, [4])
        self.assertTrue(all(v[0] <= hc.BAD_DEG for v in left.values()))

    def test_2d_matches_3d(self):
        """The plan's convention (x, y, H clockwise from +y; register_stops, the tour) says what the map's does."""
        e = hc.errors_2d({0: (0, 0), 1: (1, 1)}, {0: 30.0, 1: 30.0}, [(0, 1, 15.0, 15.0 + 180.0 - 360.0, 100)])
        self.assertAlmostEqual(e[0][2], 0.0, places=6); self.assertAlmostEqual(e[0][3], 0.0, places=6)


class TestVioPlace(unittest.TestCase):
    def test_a_placed_stop_keeps_its_turn(self):
        """A turn fitted to the stops' centres follows the ones the map misplaced (walk 16ddac14's stop 017 came out ~145 degrees off); the
        turn now comes from the stops' rotations, so a stop the track places faces where it faced even when three of the map's stops are metres off.
        (The fit to the centres alone fails this on 16 of 40 seeds.)"""
        import map_stops
        R, C = walk(10, seed=5)
        rng = np.random.default_rng(3)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        s, t = 0.37, np.array([4.0, -2.0, 7.0])                                # the track's own frame: X_map = s Q X_vio + t, Q = q
        pts = [{"id": f"{k:03d}", "filename": f"{k:03d}.jpg", "timestamp": float(k)} for k in range(10)]
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "vio"))
            Cv = [q.T @ (C[k] - t) / s for k in range(10)]
            Rv = [R[k] @ q for k in range(10)]                                 # R_map = R_vio Q^T
            np.savez(os.path.join(d, "vio", "poses.npz"), names=np.array([f"s_{p['filename']}" for p in pts]), t=np.arange(10.0),
                     C=np.array(Cv), R=np.array(Rv))
            truth = {k: (R[k].copy(), C[k].copy()) for k in (3, 7)}
            Rm, Cm = dict(R), dict(C)
            for k in truth:
                Rm.pop(k); Cm.pop(k)
            for j in (2, 4, 8):                                                # three of the map's stops metres from where they were
                Cm[j] = Cm[j] + np.array([2.5, 0.0, -3.0]) * (j - 5)
            got = map_stops.vio_place(d, pts, Rm, Cm, list(truth), log=lambda *_: None)
        self.assertEqual(sorted(got), [3, 7])
        for k, (r_, c_) in truth.items():
            self.assertLess(map_stops.rot_angle(Rm[k] @ r_.T), 0.5)                 # the whole turn, up included
            self.assertLess(abs(az(F @ (Rm[k] @ np.array([0, 0, 1.0]))) - az(F @ (r_ @ np.array([0, 0, 1.0])))), 0.5)

    def test_collapsed_stops(self):
        import map_stops
        C = {0: np.zeros(3), 1: np.array([1.0, 0, 0]), 2: np.array([1.0, 0, 0]), 3: np.array([3.0, 0, 1.0])}
        self.assertEqual(map_stops.collapsed_stops(C), [1, 2])
        self.assertEqual(map_stops.collapsed_stops({0: np.zeros(3), 1: np.ones(3)}), [])


class TestStamp(unittest.TestCase):
    def test_each_stop_says_how_well_its_heading_agrees(self):
        import register_stops as rs
        nodes = [{"x": 0.0, "y": 0.0, "H": 0.0}, {"x": 0.0, "y": 2.0, "H": 0.0}, {"x": None, "y": None, "H": None}]
        rs.stamp_check(nodes, [(0, 1, 10.0, 180.0, 100)])                    # 0 sees 1 ten degrees right of where the plan has it; 1 sees 0 dead behind
        self.assertEqual(nodes[0]["check"], {"deg": 10.0, "pairs": 1})
        self.assertEqual(nodes[1]["check"], {"deg": 0.0, "pairs": 1})
        self.assertNotIn("check", nodes[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
