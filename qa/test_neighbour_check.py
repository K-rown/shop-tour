"""Tests for neighbour_check.py: a person-sized shape that two neighbouring stops see too is an object, not him.   python qa/test_neighbour_check.py   (CPU only, no models)

The stops are synthetic: a textured concrete floor in the straight-down view and a shape on it. The neighbours are given as the pictures they would see at the
stop's floor points (neighbour_check.Context.sample is replaced; the alignment search and the real pictures are covered by running the gate on a walk)."""
import os, sys, unittest
import cv2
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import capturer as C
import neighbour_check as NC
import check_capturer as cc

N = C.VIEW_N
MPP = 2.4 / (N / 2 / np.tan(np.radians(65)))


def floor(seed=0):
    rng = np.random.default_rng(seed)
    g = rng.normal(120, 22, (N, N)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 2.0) * 2.0 - 120
    return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def blob(cx=480, cy=400, rx=40, ry=60):
    m = np.zeros((N, N), np.uint8); cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return m > 0


def thing(view, m, seed, dark=False):
    """The view with `m` replaced by a textured thing (a toolbox: orange and black), or by him (dark, smooth)."""
    rng = np.random.default_rng(seed)
    out = view.copy()
    if dark:
        out[m] = (28, 30, 36)
    else:
        tex = cv2.GaussianBlur(rng.normal(0, 1, (N, N)).astype(np.float32), (0, 0), 3) * 120
        out[m] = np.clip(np.stack([60 + tex, 110 + tex, 200 + tex], -1), 0, 255).astype(np.uint8)[m]
    return out


class FakeContext(NC.Context):
    """A Context whose neighbours are given: [(azimuth, the picture that neighbour sees at this stop's floor points)]."""
    def __init__(self, seen, aligned_n=None):
        self.ids = ["001", "002", "003"]; self.UP = [[0.0, 1.0, 0.0]]; self.frames = "."
        self._agree = {}; self._aligned = {}
        self.seen = seen
        self.al = [dict(j=i + 1, d=2.0, az=az, turn=0.0, par=dict(ncc=0.6, dd=0.0, dl=0.0, dyaw=0.0)) for i, (az, _) in enumerate(seen)]
        if aligned_n is not None:
            self.al = self.al[:aligned_n]

    def aligned(self, k):
        return self.al

    def _mask(self, j):                                               # (the capturer's mask of the stop: where the window of the agreement is)
        return cv2.dilate(blob().astype(np.uint8), np.ones((41, 41), np.uint8)) > 0

    only_at = None                                                    # {height: True}: the neighbours see the thing only at these heights (a tall thing); the floor otherwise

    def sample(self, k, o, ys, xs, h=0.0):
        img = self.seen[o["j"] - 1][1].astype(np.float32)
        if self.only_at is not None and not any(abs(h - a) < 1e-6 for a in self.only_at):
            img = self.bare
        return img[ys, xs], np.ones(np.shape(ys), bool)


def pano_of(view):
    p, ok = C.view_to_pano(view, [0.0, 1.0, 0.0], 1440, 2880)
    return p


class Judge(unittest.TestCase):
    def setUp(self):
        self.f = floor(1); self.m = blob()
        self.k_object = thing(self.f, self.m, 5)
        self.k_person = thing(self.f, self.m, 0, dark=True)
        self.shape = cv2.erode(self.m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    def judge(self, k_view, seen, **kw):
        return FakeContext(seen, **kw).judge("001", self.shape, pano_of(k_view))

    def test_an_object_that_two_neighbours_see_from_two_sides_is_an_object(self):
        seen = [(0.0, (self.k_object * 0.85).astype(np.uint8)), (140.0, (self.k_object * 1.08).clip(0, 255).astype(np.uint8))]   # (brightness differs: the gain is matched)
        r = self.judge(self.k_object, seen)
        self.assertTrue(r["object"], r); self.assertGreater(r["share"], 0.8)

    def test_a_person_the_neighbours_do_not_see_is_not(self):
        seen = [(0.0, self.f), (140.0, self.f)]                       # they see the bare floor where he stands
        r = self.judge(self.k_person, seen)
        self.assertFalse(r["object"], r); self.assertLess(r["share"], 0.1)

    def test_one_neighbour_is_not_enough(self):
        seen = [(0.0, self.k_object), (140.0, self.f)]
        self.assertFalse(self.judge(self.k_object, seen)["object"])
        self.assertFalse(self.judge(self.k_object, seen[:1])["object"])

    def test_two_neighbours_on_the_same_side_do_not_count(self):
        seen = [(0.0, self.k_object), (20.0, self.k_object)]          # (they would see something above the floor in the same wrong place)
        r = self.judge(self.k_object, seen)
        self.assertFalse(r["object"], r)

    def test_two_neighbours_that_see_something_else_there_do_not_agree_with_the_stop(self):
        other = thing(self.f, self.m, 9)
        seen = [(0.0, other), (140.0, other)]                         # a different thing in the same place (and he stands there)
        self.assertFalse(self.judge(self.k_person, seen)["object"])

    def test_a_stop_whose_neighbours_do_not_line_up_is_never_an_object(self):
        seen = [(0.0, self.k_object), (140.0, self.k_object)]
        self.assertFalse(self.judge(self.k_object, seen, aligned_n=1)["object"])
        self.assertFalse(self.judge(self.k_object, seen, aligned_n=0)["object"])

    def test_a_shape_the_neighbours_cannot_all_see_is_not_judged_an_object(self):
        ctx = FakeContext([(0.0, self.k_object), (140.0, self.k_object)])
        hidden = np.ones((N, N), bool); hidden[:, 400:] = False       # the second neighbour sees none of the shape's right half
        orig = ctx.sample
        ctx.sample = lambda k, o, ys, xs, h=0.0: (lambda c, ok: (c, ok & (hidden[ys, xs] if o["j"] == 2 else True)))(*orig(k, o, ys, xs, h))
        r = ctx.judge("001", self.shape, pano_of(self.k_object))
        self.assertFalse(r["object"], r)

    def test_a_tall_object_is_agreed_at_its_height(self):
        ctx = FakeContext([(0.0, self.k_object), (140.0, self.k_object)]); ctx.bare = self.f; ctx.only_at = (0.4,)
        r = ctx.judge("001", self.shape, pano_of(self.k_object))
        self.assertTrue(r["object"], r)
        ctx = FakeContext([(0.0, self.k_object), (140.0, self.k_object)]); ctx.bare = self.f; ctx.only_at = (0.9,)
        self.assertFalse(ctx.judge("001", self.shape, pano_of(self.k_object))["object"])         # a height nobody looks at (the heights go up to 0.8 m): nothing agrees

    def test_it_can_be_switched_off(self):
        seen = [(0.0, self.k_object), (140.0, self.k_object)]
        os.environ["SHOP_TOUR_NEIGHBOUR_CHECK"] = "0"
        try:
            self.assertFalse(self.judge(self.k_object, seen)["object"])
        finally:
            del os.environ["SHOP_TOUR_NEIGHBOUR_CHECK"]


class FakeModels:
    """What check_stop asks of the models: a depth that has the shape standing up, and the shape as SAM's only candidate."""
    def __init__(self, m):
        self.m = m

    def disparity(self, v):
        d = np.ones((N, N), np.float32); d[self.m] = 1.6
        return d

    def segment_many(self, v, pts, *a, **kw):
        return [(0, self.m, 0.9)]


class CheckStop(unittest.TestCase):
    """check_stop's check C with the second opinion: the same shape is a problem for him and none for an object."""
    def setUp(self):
        self.f = floor(2); self.m = blob()
        self.hole = cv2.dilate(self.m.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0

    def run_check(self, k_view, seen, neighbours=True):
        ctx = FakeContext(seen) if neighbours else None
        res = cc.check_stop(FakeModels(self.m), "001", pano_of(k_view), [0.0, 1.0, 0.0], 2.4, self.hole, None, None, None, ctx)[0]
        return res

    def test_a_person_stays_held_and_an_object_is_not(self):
        obj = thing(self.f, self.m, 5); per = thing(self.f, self.m, 0, dark=True)
        r = self.run_check(per, [(0.0, self.f), (140.0, self.f)])
        self.assertTrue(any("person-sized" in p for p in r["problems"]), r)
        r = self.run_check(obj, [(0.0, obj), (140.0, obj)])
        self.assertFalse(any("person-sized" in p for p in r["problems"]), r)
        self.assertIn("neighbour_object", r["info"])
        r = self.run_check(obj, [(0.0, obj), (140.0, obj)], neighbours=False)                 # (without the second opinion the gate holds it as before)
        self.assertTrue(any("person-sized" in p for p in r["problems"]), r)

    def test_a_shape_that_was_looked_at_and_not_agreed_says_so(self):
        per = thing(self.f, self.m, 0, dark=True)
        r = self.run_check(per, [(0.0, self.f), (140.0, self.f)])
        self.assertIn("neighbour_check", r["info"]); self.assertLess(r["info"]["neighbour_check"]["share"], 0.1)


if __name__ == "__main__":
    unittest.main()
