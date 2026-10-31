"""Tests for SHOP_TOUR_PEOPLE (2026-10-27): people stay in the picture, smudged; nothing is held; the paint-out and the review are SHOP_TOUR_PEOPLE=remove.
    python qa/test_people_blur.py
No models, GPU or footage: a fake detector finds a magenta blob, a fake segmentation cuts it out. The old (remove) flow is tested by the other files, which set SHOP_TOUR_PEOPLE=remove.
"""
import json, os, sys, tempfile, unittest
from unittest import mock

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "qa"))
import people_mode          # noqa: E402
import people_blur as PB    # noqa: E402
import privacy_gate as PG   # noqa: E402
import privacy_margin as PMG  # noqa: E402
import remove_others as RO  # noqa: E402
import manifest             # noqa: E402

W, H = 2048, 1024


def mag(img):
    return (img[..., 0] > 200) & (img[..., 1] < 80) & (img[..., 2] > 200)


def textured():
    rng = np.random.RandomState(3)
    return np.clip(rng.normal(128, 50, (H, W, 3)), 0, 255).astype(np.uint8)


def put_person(img, az0, az1, el0, el1):
    c0, c1 = int((az0 + 180) / 360 * W), int((az1 + 180) / 360 * W)
    r0, r1 = int((90 - el1) / 180 * H), int((90 - el0) / 180 * H)
    img[r0:r1, c0:c1] = (255, 0, 255)
    return (r0, r1, c0, c1)


class FakeDet:
    def persons_at(self, view, thr):
        m = mag(view)
        if m.sum() < 20:
            return []
        ys, xs = np.nonzero(m)
        return [(0.9, [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])]

    def faces(self, view):
        return []


class FakeModels:
    def segment_box(self, view, box):
        return [(mag(view), 0.9)]


def env(**kw):
    return mock.patch.dict(os.environ, kw)


class TestMode(unittest.TestCase):
    def test_default_is_keep_and_unknown_means_keep(self):
        """2026-10-30: the default was blur (people smudged whole); the owner asked for faces only - keep"""
        with mock.patch.dict(os.environ):
            os.environ.pop("SHOP_TOUR_PEOPLE", None)
            self.assertEqual(people_mode.mode(), "keep"); self.assertFalse(people_mode.holds())
            for v, want in (("remove", "remove"), ("KEEP", "keep"), (" blur ", "blur"), ("banana", "keep")):
                os.environ["SHOP_TOUR_PEOPLE"] = v; self.assertEqual(people_mode.mode(), want)

    def test_the_paint_out_and_the_margin_run_only_in_remove_mode(self):
        with env(SHOP_TOUR_PEOPLE="blur", SHOP_TOUR_REMOVE_OTHERS="1", SHOP_TOUR_PRIVACY_MARGIN="1"):
            self.assertFalse(RO.enabled()); self.assertFalse(PMG.enabled()); self.assertEqual(PMG.mode_word(), "off")
        with env(SHOP_TOUR_PEOPLE="keep"):
            self.assertFalse(RO.enabled()); self.assertFalse(PMG.enabled())
        with env(SHOP_TOUR_PEOPLE="remove"):
            os.environ.pop("SHOP_TOUR_REMOVE_OTHERS", None); os.environ.pop("SHOP_TOUR_PRIVACY_MARGIN", None)
            self.assertTrue(RO.enabled()); self.assertTrue(PMG.enabled())
            os.environ["SHOP_TOUR_REMOVE_OTHERS"] = "0"; self.assertFalse(RO.enabled())          # (the older switch still works in the older flow)
            os.environ["SHOP_TOUR_PRIVACY_MARGIN"] = "0"; self.assertFalse(PMG.enabled())


class TestSmudge(unittest.TestCase):
    def test_smudge_blurs_inside_the_mask_and_leaves_the_rest_alone(self):
        img = textured(); mask = np.zeros((H, W), bool); mask[400:600, 800:900] = True
        out = PB.smudge(img, mask, sigma_px=30, dilate_px=10)
        inner = (slice(430, 570), slice(820, 880))
        self.assertLess(out[inner].std(), 0.25 * img[inner].std())                       # the texture is gone inside the mask
        far = np.ones((H, W), bool); far[300:700, 650:1050] = False                       # 200 px beyond the mask on every side
        self.assertTrue((out[far] == img[far]).all())                                      # untouched outside
        self.assertEqual(out.shape, img.shape); self.assertEqual(out.dtype, np.uint8)

    def test_no_mask_no_change_and_darken(self):
        img = textured()
        self.assertTrue((PB.smudge(img, np.zeros((H, W), bool), 30, 10) == img).all())
        m = np.zeros((H, W), bool); m[400:600, 800:900] = True
        a = PB.smudge(img, m, 30, 10); b = PB.smudge(img, m, 30, 10, darken=0.8)
        self.assertLess(b[430:570, 820:880].mean(), 0.9 * a[430:570, 820:880].mean())

    def test_a_mask_over_the_seam_is_smudged_on_both_sides(self):
        img = textured(); m = np.zeros((H, W), bool); m[400:600, :60] = True; m[400:600, -60:] = True
        out = PB.smudge(img, m, 20, 6)
        self.assertLess(out[430:570, :40].std(), 0.4 * img[430:570, :40].std()); self.assertLess(out[430:570, -40:].std(), 0.4 * img[430:570, -40:].std())
        self.assertTrue((out[:, 400:1600] == img[:, 400:1600]).all())

    def test_a_person_is_smudged_inside_his_mask_and_nothing_else_changes(self):
        img = textured(); box = put_person(img, 36, 44, -12, 3)
        person = dict(kind="person", score=0.9, az=40.0, el=-4.5, deg=15.0)
        out, rec = PB.smudge_person(img, person, FakeDet(), FakeModels())
        r0, r1, c0, c1 = box
        pure = lambda a: (a == np.array([255, 0, 255], np.uint8)).all(-1).sum()
        self.assertGreater(pure(img), 1000); self.assertLess(pure(out), 0.02 * pure(img))                   # his hard-edged shape is gone (a smudge of his colour is what is left)
        core = (slice(r0 + 10, r1 - 10), slice(c0 + 5, c1 - 5))
        self.assertLess(out[core].std(axis=(0, 1)).mean(), 12)                                                   # a smooth smudge (the noise round him is not carried in)
        az = np.linspace(-180, 180, W, endpoint=False) + 180 / W
        away = np.abs(((az - 40) + 180) % 360 - 180) > 40                                                         # more than 40 degrees from him
        self.assertTrue((out[:, away] == img[:, away]).all())
        self.assertEqual(rec["kind"], "person"); self.assertGreater(rec["sigma_px"], 3); self.assertGreater(rec["mask_px"], 0)

    def test_the_walker_is_not_counted_and_not_smudged_twice(self):
        img = textured(); put_person(img, -4, 4, -80, -50)
        person = dict(kind="person", score=0.9, az=0.0, el=-65.0, deg=30.0)
        walker = np.zeros((H // 2, W // 2), bool); walker[int((90 + 55) / 180 * H / 2):int((90 + 75) / 180 * H / 2), W // 4 - 20:W // 4 + 20] = True
        self.assertTrue(PB.is_walker(person, walker))
        out, recs = PB.smudge_people(img, [person], FakeDet(), FakeModels(), walker)
        self.assertEqual(recs, []); self.assertTrue((out == img).all())
        out2, recs2 = PB.smudge_people(img, [person], FakeDet(), FakeModels(), None)
        self.assertEqual(len(recs2), 1)

    def test_a_person_the_models_cannot_segment_is_still_smudged(self):
        class Boom(FakeModels):
            def segment_box(self, view, box):
                raise RuntimeError("no card")
        img = textured(); put_person(img, 36, 44, -12, 3)
        out, recs = PB.smudge_people(img, [dict(kind="person", score=0.9, az=40.0, el=-4.5, deg=15.0)], FakeDet(), Boom(), None, log=lambda *a: None)
        self.assertEqual(len(recs), 1); self.assertIn("error", recs[0]); self.assertFalse(recs[0].get("failed"))
        self.assertFalse((out == img).all())

    def test_the_walker_hole_is_smudged_and_darkened(self):
        import capturer as C
        img = np.full((H, W, 3), 200, np.uint8); img = np.clip(img.astype(int) + np.random.RandomState(5).normal(0, 30, img.shape), 0, 255).astype(np.uint8)
        hole = np.zeros((C.VIEW_N, C.VIEW_N), np.uint8); hole[420:620, 450:560] = 255
        out = PB.smudge_walker(img, hole, [0.0, 1.0, 0.0])
        self.assertFalse((out == img).all())
        diff = (out.astype(int) != img.astype(int)).any(-1)
        ys, xs = np.nonzero(diff)
        self.assertGreater(ys.min(), H * 0.5)                                                        # only below the horizon, round the nadir
        self.assertLess(out[diff].mean(), img[diff].mean())                                          # darker
        self.assertTrue((PB.smudge_walker(img, np.zeros_like(hole), [0.0, 1.0, 0.0]) == img).all())


class TestGateAndManifest(unittest.TestCase):
    def _frames(self, d):
        fr = os.path.join(d, "frames")
        for sub in ("clean", "capturer", "safe"):
            os.makedirs(os.path.join(fr, sub))
        img = textured(); put_person(img, 36, 44, -12, 3)
        pts = [{"id": "001", "filename": "001.jpg"}, {"id": "002", "filename": "002.jpg"}]
        for p in pts:
            cv2.imwrite(os.path.join(fr, "clean", p["filename"]), img)
            cv2.imwrite(os.path.join(fr, "capturer", p["id"] + "_mask.png"), np.zeros((1024, 1024), np.uint8))
        json.dump(pts, open(os.path.join(fr, "points.json"), "w"))
        return fr, pts, img

    def _gate(self, fr, pts, state):
        g = PG.Gate.__new__(PG.Gate)
        g.frames, g.pts, g.state, g.old = fr, pts, state, {}
        g.safe = os.path.join(fr, "safe"); g.UP = [[0.0, 1.0, 0.0]] * len(pts); g.det = FakeDet(); g.models = FakeModels(); g.people = people_mode.mode()
        return g

    def test_blur_mode_holds_nothing_and_the_manifest_says_all_clear(self):
        person = dict(kind="person", score=0.9, az=40.0, el=-4.5, deg=15.0)
        with env(SHOP_TOUR_PEOPLE="blur"), tempfile.TemporaryDirectory() as d:
            fr, pts, img = self._frames(d); state = {}
            g = self._gate(fr, pts, state)
            with mock.patch.object(PG, "scan", lambda det, pano: [person]), mock.patch.object(PG, "nadir_persons", lambda det, pano, up: []):
                for k in range(2):
                    g.stop(k)
            for sid in ("001", "002"):
                self.assertFalse(state[sid]["held"]); self.assertEqual(state[sid]["reasons"], []); self.assertEqual(state[sid]["people_blurred"], 1)
                out = cv2.imread(os.path.join(fr, "safe", sid + ".jpg"))
                self.assertLess(((out == np.array([255, 0, 255], np.uint8)).all(-1)).sum(), 50)       # the person is smudged in the picture that goes into the tour (no pure shape left)
            PG.finish(fr, pts, state)
            doc = json.load(open(os.path.join(fr, "privacy.json")))
            self.assertEqual(doc["people_mode"], "blur")
            pb = manifest.privacy_block(fr, ["001", "002"])
            self.assertEqual(pb["status"], "all_clear"); self.assertEqual(pb["held"], []); self.assertEqual(pb["people_blurred"], 2); self.assertEqual(pb["people_mode"], "blur")

    def test_blur_mode_even_a_stop_the_old_flow_would_hold_is_all_clear(self):
        with tempfile.TemporaryDirectory() as d:
            fr = os.path.join(d, "frames"); os.makedirs(fr)
            json.dump(dict(version=1, people_mode="blur", stops={"001": dict(held=True, reasons=["another person in the picture"], people_blurred=3)}), open(os.path.join(fr, "privacy.json"), "w"))
            pb = manifest.privacy_block(fr, ["001"])
            # (a stop that is in the tour and still held is the build's bug, and is still said)
            self.assertEqual(pb["status"], "held"); self.assertEqual([h["id"] for h in pb["held"]], ["001"])
            json.dump(dict(version=1, people_mode="blur", stops={"001": dict(held=False, reasons=[], people_blurred=3)}), open(os.path.join(fr, "privacy.json"), "w"))
            pb = manifest.privacy_block(fr, ["001"])
            self.assertEqual((pb["status"], pb["held"], pb["people_blurred"], pb["people_mode"]), ("all_clear", [], 3, "blur"))

    def test_the_older_flow_writes_the_older_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            fr = os.path.join(d, "frames"); os.makedirs(fr)
            json.dump(dict(version=1, stops={"001": dict(held=True, reasons=["another person"]), "002": dict(held=False, reasons=[])}), open(os.path.join(fr, "privacy.json"), "w"))
            pb = manifest.privacy_block(fr, ["002"])
            self.assertEqual(pb["status"], "held"); self.assertEqual([h["id"] for h in pb["held"]], ["001"]); self.assertNotIn("people_mode", pb); self.assertNotIn("people_blurred", pb)

    def test_keep_mode_leaves_the_person_and_holds_nothing(self):
        person = dict(kind="person", score=0.9, az=40.0, el=-4.5, deg=15.0)
        with env(SHOP_TOUR_PEOPLE="keep"), tempfile.TemporaryDirectory() as d:
            fr, pts, img = self._frames(d); state = {}
            g = self._gate(fr, pts, state)
            with mock.patch.object(PG, "scan", lambda det, pano: [person]), mock.patch.object(PG, "nadir_persons", lambda det, pano, up: []):
                g.stop(0)
            self.assertFalse(state["001"]["held"]); self.assertEqual(state["001"]["people_blurred"], 0)
            out = cv2.imread(os.path.join(fr, "safe", "001.jpg"))
            self.assertGreater(mag(out).sum(), 0.5 * mag(img).sum())                                # (the jpeg of the safe copy eats some of it; a smudge would leave next to none)


if __name__ == "__main__":
    unittest.main()
