"""Tests for the patch-synthesis fill (synth_fill.py) and for the lump loop of remove_others.py (the capturer's hole made again when a person-sized lump is left).
    python qa/test_synth_fill.py
No models or footage: a noisy asphalt-like picture with a bright band, holes cut into it, and a fake check standing in for the gate's.
"""
import json, os, sys, tempfile, unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "qa"))
import synth_fill as SF            # noqa: E402
import remove_others as RO         # noqa: E402


def road(h=360, w=420, seed=3):
    """Grey noise (grain like asphalt) with a slow left-to-right brightening and a bright stripe across it."""
    rng = np.random.default_rng(seed)
    base = 90 + 40 * np.linspace(0, 1, w)[None, :] + 0 * np.linspace(0, 1, h)[:, None]
    img = base + rng.normal(0, 18, (h, w))
    img[150:162, :] += 70
    return np.clip(np.repeat(img[..., None], 3, 2), 0, 255).astype(np.uint8)


def hole_at(shape, cy, cx, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r


class TestSynth(unittest.TestCase):
    def test_only_the_hole_changes_and_it_is_made_new(self):
        img = road(); hole = hole_at(img.shape, 230, 200, 45)
        img2 = img.copy(); img2[hole] = (255, 0, 255)                                     # what stood there
        out, note = SF.synth(img2, hole)
        self.assertTrue(note.startswith("synth"), note)
        self.assertEqual(int((out[~hole] != img2[~hole]).any(-1).sum()), 0)                # nothing outside the hole is touched
        self.assertEqual(int(((out[..., 0] > 200) & (out[..., 1] < 60) & hole).sum()), 0)    # none of him is left

    def test_the_fill_keeps_the_grain_of_the_place(self):
        img = road(); hole = hole_at(img.shape, 230, 200, 45)
        out, _ = SF.synth(img, hole)
        g = lambda a, m: float(cv2.Laplacian(cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8), cv2.CV_32F)[m].std())
        ring = cv2.dilate(hole.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool) & ~cv2.dilate(hole.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        self.assertGreater(g(out, hole), 0.6 * g(img, ring))                               # not a smooth smudge (a blur has a small fraction of the grain)

    def test_the_fill_follows_the_light_round_it(self):
        img = road(); hole = hole_at(img.shape, 230, 200, 45)
        out, _ = SF.synth(img, hole)
        # the road brightens to the right: the left of the fill is darker than its right
        l = out[:, 160:185][hole[:, 160:185]].mean(); r = out[:, 215:240][hole[:, 215:240]].mean()
        self.assertGreater(float(r), float(l) + 5)
        # and the fill as a whole is the colour of the road beside it
        ring = cv2.dilate(hole.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool) & ~cv2.dilate(hole.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        self.assertLess(abs(float(out[hole].mean()) - float(img[ring].mean())), 20.0)

    def test_the_same_picture_gives_the_same_fill(self):
        img = road(); hole = hole_at(img.shape, 230, 200, 40)
        a, _ = SF.synth(img, hole); b, _ = SF.synth(img, hole)
        self.assertTrue(np.array_equal(a, b))

    def test_where_nothing_can_be_copied_from_the_picture_is_returned_as_it_is(self):
        img = road(120, 120); hole = np.ones((120, 120), bool)
        out, note = SF.synth(img, hole)
        self.assertTrue(np.array_equal(out, img)); self.assertIn("too little", note)
        out, note = SF.synth(img, np.zeros((120, 120), bool))
        self.assertTrue(np.array_equal(out, img))

    def test_a_source_the_caller_names_is_the_only_place_copied_from(self):
        img = road(); hole = hole_at(img.shape, 230, 200, 40)
        img2 = img.copy(); img2[:, :120] = (0, 255, 0)                                     # a green strip the fill must not borrow from
        src = np.ones(hole.shape, bool); src[:, :120] = False
        out, _ = SF.synth(img2, hole, src)
        self.assertEqual(int(((out[..., 1] > 200) & (out[..., 0] < 40) & hole).sum()), 0)

    def test_the_colour_at_the_edge_is_carried_in(self):
        img = np.full((200, 200, 3), 100, np.uint8); hole = hole_at(img.shape, 100, 100, 40)
        filled = img.copy(); filled[hole] = 150                                            # a patch from a sunlit place
        out = SF.carry_edge_colour(img, filled, hole)
        edge = hole & ~cv2.erode(hole.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        self.assertLess(float(out[edge].mean()), 112.0)                                    # at the edge it meets the picture beside it
        self.assertEqual(int((out[~hole] != filled[~hole]).any(-1).sum()), 0)


class TestLumpLoop(unittest.TestCase):
    """Others._refill: a person-sized lump left under the camera makes the capturer's hole again, by the first way that leaves none; no way that works leaves everything as it was."""
    def _others(self, d, answers):
        import privacy_inputs as PI
        fr = os.path.join(d, "frames")
        for sub in ("points", "clean", "capturer"):
            os.makedirs(os.path.join(fr, sub))
        json.dump([dict(id="001", filename="001.jpg")], open(os.path.join(fr, "points.json"), "w"))
        json.dump([[0, 1, 0]], open(os.path.join(fr, "up.json"), "w"))
        json.dump(dict(pairs=[]), open(os.path.join(fr, "hops.json"), "w"))
        pano = np.repeat(road(512, 1024)[..., :1], 3, 2)
        cv2.imwrite(os.path.join(fr, "points", "001.jpg"), pano); cv2.imwrite(os.path.join(fr, "clean", "001.jpg"), pano)
        hole = np.zeros((1024, 1024), np.uint8); hole[420:560, 440:580] = 255
        cv2.imwrite(os.path.join(fr, "capturer", "001_hole.png"), hole); cv2.imwrite(os.path.join(fr, "capturer", "001_mask.png"), hole)
        soft = np.zeros((1024, 1024), np.uint8); soft[430:550, 450:570] = 255
        cv2.imwrite(os.path.join(fr, "capturer", "001_soft.png"), soft)
        json.dump({}, open(os.path.join(fr, "capturer", "report.json"), "w"))
        pts = json.load(open(os.path.join(fr, "points.json"))); UP = json.load(open(os.path.join(fr, "up.json")))
        O = RO.Others(fr, pts, UP, PI.Inputs(fr, 1))
        calls = []

        def lump(k, p, models):
            calls.append(1)
            return answers[min(len(calls) - 1, len(answers) - 1)]
        O._lump = lump
        return fr, O, pano, calls

    def test_no_lump_nothing_is_done(self):
        with tempfile.TemporaryDirectory() as d:
            fr, O, pano, calls = self._others(d, [None])
            rec = {}
            out, buf, again = O._refill(0, pano, None, [0, 1, 0], 2.4, None, rec, lambda *a: None)
            self.assertFalse(again); self.assertIsNone(buf); self.assertTrue(np.array_equal(out, pano))
            self.assertEqual(rec["refill"]["first_check"], "ok"); self.assertEqual(len(calls), 1)

    def test_a_lump_is_made_again_by_the_first_way_that_leaves_none(self):
        with tempfile.TemporaryDirectory() as d:
            fr, O, pano, calls = self._others(d, ["a person-sized shape (1.7 m2) is still near the camera", None])
            rec = {}
            out, buf, again = O._refill(0, pano, None, [0, 1, 0], 2.4, None, rec, lambda *a: None)
            self.assertTrue(again); self.assertIsNotNone(buf)
            self.assertEqual(rec["refill"]["won"], "synth"); self.assertEqual(len(rec["refill"]["tries"]), 1)
            self.assertGreater(int((out != pano).any(-1).sum()), 1000)
            soft = cv2.imread(os.path.join(fr, "capturer", "001_soft.png"), 0)
            self.assertEqual(int((soft[430:550, 450:570] > 0).sum()), 0)                    # what was made again is no soft smudge

    def test_when_no_way_works_the_picture_and_its_record_stay_as_they_were(self):
        with tempfile.TemporaryDirectory() as d:
            fr, O, pano, calls = self._others(d, ["a person-sized shape (1.7 m2) is still near the camera"])
            soft0 = cv2.imread(os.path.join(fr, "capturer", "001_soft.png"), 0)
            rec = {}
            out, buf, again = O._refill(0, pano, None, [0, 1, 0], 2.4, None, rec, lambda *a: None)
            self.assertFalse(again); self.assertTrue(np.array_equal(out, pano)); self.assertIsNone(rec["refill"]["won"])
            self.assertEqual(len(rec["refill"]["tries"]), RO.REFILL_TRIES)                  # every way was tried, no more
            self.assertTrue(np.array_equal(cv2.imread(os.path.join(fr, "capturer", "001_soft.png"), 0), soft0))

    def test_a_stop_where_a_person_is_still_found_is_not_made_again(self):
        with tempfile.TemporaryDirectory() as d:
            fr, O, pano, calls = self._others(d, ["x"])
            rec = dict(left=[dict(kind="person")])
            out, buf, again = O._refill(0, pano, None, [0, 1, 0], 2.4, None, rec, lambda *a: None)
            self.assertFalse(again); self.assertEqual(calls, [])                           # held anyway: no time spent

    def test_a_stop_with_nobody_under_the_camera_has_nothing_to_make_again(self):
        with tempfile.TemporaryDirectory() as d:
            fr, O, pano, calls = self._others(d, ["x"])
            os.remove(os.path.join(fr, "capturer", "001_hole.png"))
            out, buf, again = O._refill(0, pano, None, [0, 1, 0], 2.4, None, {}, lambda *a: None)
            self.assertFalse(again); self.assertEqual(calls, [])

    def test_it_runs_only_behind_the_switch(self):
        src = open(os.path.join(HERE, "remove_others.py"), encoding="utf-8").read()
        chain = open(os.path.join(HERE, "privacy_chain.py"), encoding="utf-8").read()
        self.assertIn("others.paint_stop(", chain)                                          # the only caller is behind `if others is not None`
        self.assertIn("RO.Others(frames, pts, UP, P) if RO.enabled() else None", chain)
        self.assertNotIn("synth_fill", open(os.path.join(HERE, "privacy_gate.py"), encoding="utf-8").read())
        self.assertNotIn("synth_fill", open(os.path.join(HERE, "remove_capturer.py"), encoding="utf-8").read())
        self.assertIn("import synth_fill as SF", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
