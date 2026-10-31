"""Tests for painting the OTHER people out (2026-10-22: remove_others.py, switch SHOP_TOUR_REMOVE_OTHERS=1).
    python qa/test_remove_others.py
No models, GPU or footage: a fake detector finds a magenta blob, a fake segmentation cuts it out, the neighbour is a second synthetic view of the same floor.
"""
import json, os, subprocess, sys, tempfile, unittest

os.environ.setdefault("SHOP_TOUR_PEOPLE", "remove")        # these tests are about the older flow: the paint-out and the holds (the default is now to smudge people and hold nothing: qa/test_people_blur.py)

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "qa"))
import remove_others as RO        # noqa: E402
from test_units import synth_pano  # noqa: E402

MAGENTA = (255, 0, 255)


def magenta(img):
    return (img[..., 0] > 200) & (img[..., 1] < 80) & (img[..., 2] > 200)


class FakeDet:
    """A 'person detector' that finds the magenta blob in a view; no faces."""
    def persons_at(self, view, thr):
        m = magenta(view)
        if m.sum() < 20:
            return []
        ys, xs = np.nonzero(m)
        return [(0.9, [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])]

    def persons_many_at(self, views, thr):
        return [self.persons_at(v, thr) for v in views]

    def faces(self, view):
        return []


class FakeModels:
    def segment_box(self, view, box):
        m = magenta(view)
        return [(m, 0.9), (np.zeros_like(m), 0.1), (np.ones_like(m), 0.1)]


def put_person(pano, az, el, w_deg=5.0, h_deg=18.0):
    H, W = pano.shape[:2]
    out = pano.copy()
    az_ = ((np.arange(W) + 0.5) / W * 360 - 180)[None, :]; el_ = (90 - (np.arange(H) + 0.5) / H * 180)[:, None]
    inside = (((az_ - az) / (w_deg / 2)) ** 2 + ((el_ - el) / (h_deg / 2)) ** 2) < 1
    out[inside] = MAGENTA
    return out, inside


class TestView(unittest.TestCase):
    def test_a_view_projects_its_own_rays_back_onto_its_pixels(self):
        v = RO.View(40.0, -25.0, 60.0, 256)
        col, row, front = v.project(v.rays)
        c, r = np.meshgrid(np.arange(256), np.arange(256))
        self.assertTrue(front.all())
        self.assertLess(float(np.abs(col - c).max()), 0.05); self.assertLess(float(np.abs(row - r).max()), 0.05)

    def test_painting_a_view_back_into_the_pano_changes_only_where_the_hole_is(self):
        pano = synth_pano(0, 2.4, W=2048)
        v = RO.View(40.0, -25.0, 60.0, 512)
        patch = np.full((512, 512, 3), 7, np.uint8)
        hole = np.zeros((512, 512), bool); hole[200:300, 220:320] = True
        out = RO.paste_view(pano, v, patch, hole)
        changed = (out != pano).any(-1)
        self.assertGreater(changed.sum(), 100)
        ys, xs = np.nonzero(changed)
        H, W = pano.shape[:2]
        az = (xs + 0.5) / W * 360 - 180; el = 90 - (ys + 0.5) / H * 180
        self.assertLess(float(np.abs(az - 40).max()), 14.0); self.assertLess(float(np.abs(el + 25).max()), 14.0)

    def test_the_window_wraps_round_the_seam_and_over_the_pole(self):
        pano = synth_pano(0, 2.4, W=2048)
        for az, el in ((179.0, -20.0), (-179.0, 10.0), (20.0, -85.0)):
            v = RO.View(az, el, 40.0, 256)
            hole = np.zeros((256, 256), bool); hole[100:156, 100:156] = True
            out = RO.paste_view(pano, v, np.full((256, 256, 3), 9, np.uint8), hole)
            self.assertTrue(((out != pano).any(-1)).sum() > 50, (az, el))


class TestPaintOut(unittest.TestCase):
    def _scene(self):
        h = 2.4
        truth_i = synth_pano(0, h, W=2048)
        a = np.radians(170.0); C = (np.sin(a) * 3.0, 0.0, np.cos(a) * 3.0)
        truth_j = synth_pano(0, h, C, 0.0, W=2048)
        pano, inside = put_person(truth_i, 50.0, -20.0)
        nb = RO.Neighbour(1, truth_j, [0, 1, 0], h, None, [], dict(j=1, az=170.0, d=3.0, turn=0.0))
        return h, truth_i, pano, inside, nb

    def test_a_person_is_painted_out_and_the_floor_behind_him_comes_from_the_neighbour(self):
        h, truth, pano, inside, nb = self._scene()
        self.assertTrue(magenta(pano).sum() > 1000)
        person = dict(kind="person", score=0.9, az=50.0, el=-20.0, deg=18.0)
        out, rec = RO.paint_person(pano, person, FakeDet(), FakeModels(), [nb], [0, 1, 0], h)
        self.assertEqual(int(magenta(out).sum()), 0)                                   # nothing of him is left
        self.assertGreater(rec["got_pct"], 40.0)                                       # the neighbour saw the floor behind him
        self.assertEqual(rec["mask_from"], "sam"); self.assertEqual(rec["box_from"], "detector")
        d = np.abs(out.astype(int) - truth.astype(int)).mean(-1)
        self.assertLess(float(d[inside].mean()), 45.0)                                 # and what is there is the floor, not a smudge of him
        far = ~cv2.dilate(inside.astype(np.uint8), np.ones((201, 201), np.uint8)).astype(bool)
        self.assertEqual(int((out[far] != pano[far]).any(-1).sum()), 0)                # the rest of the picture is not touched

    def test_with_no_neighbour_the_hole_is_still_filled_from_the_colours_round_it(self):
        h, truth, pano, inside, nb = self._scene()
        person = dict(kind="person", score=0.9, az=50.0, el=-20.0, deg=18.0)
        out, rec = RO.paint_person(pano, person, FakeDet(), FakeModels(), [], [0, 1, 0], h)
        self.assertEqual(int(magenta(out).sum()), 0)
        self.assertEqual(rec["got_pct"], 0.0)

    def test_a_neighbour_that_has_a_person_in_that_direction_is_not_borrowed_from(self):
        h, truth, pano, inside, nb = self._scene()
        pj, _ = put_person(nb.pano, 0.0, -10.0, 170.0, 170.0)                           # the neighbour's own picture is full of someone
        nb2 = RO.Neighbour(1, pj, [0, 1, 0], h, None, [dict(kind="person", az=0.0, el=-10.0, deg=170.0, score=0.9)], nb.nb)
        person = dict(kind="person", score=0.9, az=50.0, el=-20.0, deg=18.0)
        out, rec = RO.paint_person(pano, person, FakeDet(), FakeModels(), [nb2], [0, 1, 0], h)
        self.assertEqual(int(magenta(out).sum()), 0)                                    # none of his magenta came across from the neighbour either


class TestStop(unittest.TestCase):
    """paint_stop on a two-stop walk made of synthetic pictures: the file the gate will read, the look after, and a person who cannot be painted."""
    def _walk(self, d):
        h = 2.4
        fr = os.path.join(d, "frames")
        for sub in ("points", "clean", "capturer"):
            os.makedirs(os.path.join(fr, sub))
        a = np.radians(170.0); C = (np.sin(a) * 3.0, 0.0, np.cos(a) * 3.0)
        truth_i = synth_pano(0, h, W=2048); truth_j = synth_pano(0, h, C, 0.0, W=2048)
        pano, inside = put_person(truth_i, 50.0, -20.0)
        json.dump([dict(id="001", filename="001.jpg"), dict(id="002", filename="002.jpg")], open(os.path.join(fr, "points.json"), "w"))
        json.dump([[0, 1, 0], [0, 1, 0]], open(os.path.join(fr, "up.json"), "w"))
        json.dump(dict(pairs=[dict(i=0, j=1, ok=True, consistent=True, az_ij=170.0, az_ji=-10.0, yaw=0.0, dist_m=3.0 * 1.85 / 2.4, inliers=100)]), open(os.path.join(fr, "hops.json"), "w"))
        cv2.imwrite(os.path.join(fr, "points", "001.jpg"), pano); cv2.imwrite(os.path.join(fr, "points", "002.jpg"), truth_j)
        cv2.imwrite(os.path.join(fr, "clean", "001.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(os.path.join(fr, "clean", "002.jpg"), truth_j, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return fr

    def test_the_picture_the_gate_will_read_has_nobody_in_it(self):
        import privacy_inputs as PI
        with tempfile.TemporaryDirectory() as d:
            fr = self._walk(d)
            pts = json.load(open(os.path.join(fr, "points.json"))); UP = json.load(open(os.path.join(fr, "up.json"))); P = PI.Inputs(fr, 2)
            O = RO.Others(fr, pts, UP, P)
            rec = O.paint_stop(0, FakeDet(), FakeModels(), log=lambda *a: None)
            self.assertEqual(len(rec["painted"]), 1); self.assertEqual(rec["left"], [])
            written = cv2.imread(os.path.join(fr, "clean", "001.jpg"))
            self.assertEqual(int(magenta(written).sum()), 0)                            # what the gate reads is the painted picture
            # a second look at that file, with the gate's own scan, finds nobody
            self.assertEqual(RO.people_in(FakeDet(), written, UP[0]), [])
            self.assertTrue(os.path.exists(os.path.join(fr, "capturer", "others_scan.json")))

    def test_a_person_who_cannot_be_painted_stays_in_the_picture_and_is_reported_left(self):
        import privacy_inputs as PI

        class Broken(FakeModels):
            def segment_box(self, view, box):
                raise RuntimeError("no memory")
        with tempfile.TemporaryDirectory() as d:
            fr = self._walk(d)
            pts = json.load(open(os.path.join(fr, "points.json"))); UP = json.load(open(os.path.join(fr, "up.json"))); P = PI.Inputs(fr, 2)
            O = RO.Others(fr, pts, UP, P)
            rec = O.paint_stop(0, FakeDet(), Broken(), log=lambda *a: None)
            self.assertTrue(all("error" in p for p in rec["painted"]))
            self.assertEqual(len(rec["left"]), 1)                                       # the gate's scan will find him again and hold the stop
            self.assertGreater(int(magenta(cv2.imread(os.path.join(fr, "clean", "001.jpg"))).sum()), 1000)

    def test_a_walk_with_nobody_in_it_is_left_byte_for_byte(self):
        import privacy_inputs as PI
        with tempfile.TemporaryDirectory() as d:
            fr = self._walk(d)
            pts = json.load(open(os.path.join(fr, "points.json"))); UP = json.load(open(os.path.join(fr, "up.json"))); P = PI.Inputs(fr, 2)
            before = open(os.path.join(fr, "clean", "002.jpg"), "rb").read()
            O = RO.Others(fr, pts, UP, P)
            rec = O.paint_stop(1, FakeDet(), FakeModels(), log=lambda *a: None)
            self.assertEqual(rec["painted"], [])
            self.assertEqual(open(os.path.join(fr, "clean", "002.jpg"), "rb").read(), before)      # nothing found: the file is not even rewritten


class CountingDet(FakeDet):
    def __init__(self):
        self.views = 0

    def persons_many_at(self, views, thr):
        self.views += len(views)
        return [self.persons_at(v, thr) for v in views]


class TestFineScan(unittest.TestCase):
    """The zoomed look for far-off people is part of the gate's scan only with the switch on."""
    def _run(self, env):
        import privacy_gate as pg
        keep = {k: os.environ.pop(k, None) for k in ("SHOP_TOUR_REMOVE_OTHERS", "SHOP_TOUR_FINE_PERSONS")}
        try:
            os.environ["SHOP_TOUR_REMOVE_OTHERS"] = "0"                                 # (on by default since 2026-10-22: "off" is the explicit 0)
            if env:
                os.environ[env] = "1"
            pano, _ = put_person(synth_pano(0, 2.4, W=2048), 100.0, -4.0, 2.0, 5.0)
            det = CountingDet()
            return pg.scan(det, pano), det.views
        finally:
            for k in ("SHOP_TOUR_REMOVE_OTHERS", "SHOP_TOUR_FINE_PERSONS"):
                os.environ.pop(k, None)
                if keep[k] is not None:
                    os.environ[k] = keep[k]

    def test_off_the_scan_is_what_it_was_and_on_it_also_looks_closer(self):
        import privacy_gate as pg
        off, n_off = self._run(None)
        self.assertEqual(n_off, 12)                                                    # the twelve views, once (the horizon rule's lower bar is the same pass)
        self.assertFalse(any(f.get("fine") for f in off))
        for env in ("SHOP_TOUR_REMOVE_OTHERS", "SHOP_TOUR_FINE_PERSONS"):
            on, n_on = self._run(env)
            self.assertEqual(n_on, 12 + len(pg.FINE_PITCHES) * len(range(-180, 180, pg.FINE_STEP)))
            fine = [f for f in on if f.get("fine")]
            self.assertTrue(fine)
            self.assertLess(min(abs(f["az"] - 100.0) for f in fine), 3.0)               # the zoomed views put him where he is


class TestFineBar(unittest.TestCase):
    def test_the_bar_of_the_zoomed_look(self):
        import privacy_gate as pg
        keep = {k: os.environ.pop(k, None) for k in ("SHOP_TOUR_REMOVE_OTHERS", "SHOP_TOUR_FINE_PERSONS", "SHOP_TOUR_FINE_SCORE")}
        try:
            self.assertEqual(pg.fine_score(), pg.FINE_SCORE_PAINT)                      # nothing set: the paint-out is on, and so is its bar
            os.environ["SHOP_TOUR_REMOVE_OTHERS"] = "0"; self.assertEqual(pg.fine_score(), pg.PERSON_SCORE)   # off: the gate's own bar
            os.environ["SHOP_TOUR_FINE_PERSONS"] = "1"; self.assertEqual(pg.fine_score(), pg.PERSON_SCORE)
            os.environ["SHOP_TOUR_REMOVE_OTHERS"] = "1"; self.assertEqual(pg.fine_score(), pg.FINE_SCORE_PAINT)   # painting is the remedy: a lower bar costs a patch, not a held stop
            os.environ["SHOP_TOUR_FINE_SCORE"] = "0.33"; self.assertAlmostEqual(pg.fine_score(), 0.33)
            os.environ["SHOP_TOUR_FINE_SCORE"] = "nonsense"; self.assertEqual(pg.fine_score(), pg.FINE_SCORE_PAINT)
            os.environ["SHOP_TOUR_FINE_SCORE"] = "0.01"; self.assertEqual(pg.fine_score(), pg.FINE_SCORE_PAINT)
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


class TestSwitch(unittest.TestCase):
    def test_on_by_default_and_nothing_runs_with_the_switch_at_zero(self):
        keep = os.environ.pop("SHOP_TOUR_REMOVE_OTHERS", None)
        try:
            self.assertTrue(RO.enabled())                                               # on by default (2026-10-22)
            for v in ("", "1", "true", "on", "YES"):
                os.environ["SHOP_TOUR_REMOVE_OTHERS"] = v; self.assertTrue(RO.enabled())
            for v in ("0", "no", "off", "False"):
                os.environ["SHOP_TOUR_REMOVE_OTHERS"] = v; self.assertFalse(RO.enabled())
            os.environ["SHOP_TOUR_REMOVE_OTHERS"] = "0"
            with tempfile.TemporaryDirectory() as d:                                    # the script with the switch off reads and writes nothing, loads no model
                r = subprocess.run([sys.executable, os.path.join(HERE, "remove_others.py"), "--frames", d], capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("Nothing done", r.stdout)
                self.assertEqual(os.listdir(d), [])
        finally:
            os.environ.pop("SHOP_TOUR_REMOVE_OTHERS", None)
            if keep is not None:
                os.environ["SHOP_TOUR_REMOVE_OTHERS"] = keep

    def test_the_chain_and_the_three_script_path_call_it_only_behind_the_switch(self):
        chain = open(os.path.join(HERE, "privacy_chain.py"), encoding="utf-8").read()
        self.assertIn("RO.Others(frames, pts, UP, P) if RO.enabled() else None", chain)
        self.assertIn("if others is not None:", chain); self.assertIn("others.paint_stop(k, mine.gate.det, mine.models)", chain)
        # the painting comes BEFORE the gate looks, so the gate looks at the painted picture
        self.assertLess(chain.index("others.paint_stop("), chain.index("mine.gate.stop(k)"))
        rc = open(os.path.join(HERE, "remove_capturer.py"), encoding="utf-8").read()
        self.assertIn("if RO.enabled():", rc)
        src = open(os.path.join(HERE, "remove_others.py"), encoding="utf-8").read()
        for name in ('"poses.json"', '"poses_map.json"', '"vio"', "poses.npz", '"colmap"'):
            self.assertNotIn(name, src)                                                 # it reads nothing from vio or the map

    def test_the_gate_decides_what_is_painted_and_what_stays_held(self):
        src = open(os.path.join(HERE, "remove_others.py"), encoding="utf-8").read()
        for needle in ("PG.scan(", "PG.nadir_persons(", "PG.dedupe(", "PG.face_is_small_and_alone("):
            self.assertIn(needle, src)                                                  # the same scan, the same bars as the gate
        # a person who cannot be painted is left in the picture (and the gate holds the stop): never skipped silently
        self.assertIn("a person who cannot be painted stays in the picture", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
