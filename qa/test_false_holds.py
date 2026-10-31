"""Tests for the rules that stop the privacy gate holding stops that have nobody in them (and the one that holds a person it used to miss).
python qa/test_false_holds.py   (CPU only, no models)"""
import os, sys, unittest

os.environ.setdefault("SHOP_TOUR_PEOPLE", "remove")        # these tests are about the older flow: the paint-out and the holds (the default is now to smudge people and hold nothing: qa/test_people_blur.py)
import cv2
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import capturer as C
import privacy_gate as pg
import check_capturer as cc


def floor(seed=0, n=1024):
    rng = np.random.default_rng(seed)
    g = rng.normal(120, 18, (n, n)).astype(np.float32)                 # a textured concrete floor
    g = cv2.GaussianBlur(g, (0, 0), 1.2) + rng.normal(0, 6, (n, n)).astype(np.float32)
    return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def blob(n=1024, cx=500, cy=380, rx=60, ry=110):
    m = np.zeros((n, n), np.uint8); cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return m > 0


class FillOnly(unittest.TestCase):
    mpp = 2.4 / (1024 / 2 / np.tan(np.radians(65)))

    def test_a_smooth_patch_that_is_the_whole_shape_is_the_fill(self):
        v = floor(); soft = blob(); v[soft] = (115, 118, 120)
        v = np.where(soft[..., None], cv2.GaussianBlur(v, (0, 0), 15), v)
        only, share, left = cc.fill_only(v, soft, soft, self.mpp)
        self.assertTrue(only, (share, left))

    def test_the_same_place_in_a_textured_picture_is_not(self):
        v = floor(); soft = blob()                                         # the control: him standing there, textured pixels under the same mask
        only, share, left = cc.fill_only(v, soft, soft, self.mpp)
        self.assertFalse(only); self.assertLess(share, 0.1)

    def test_a_smooth_patch_with_a_real_thing_beside_it_is_not(self):
        v = floor(); soft = blob(); v = np.where(soft[..., None], cv2.GaussianBlur(v, (0, 0), 15), v)
        sam = soft.copy(); sam[300:480, 560:660] = True                    # his leg, left out of the fill: a textured piece of the shape
        only, share, left = cc.fill_only(v, sam, soft, self.mpp)
        self.assertFalse(only, (share, left)); self.assertGreater(left, 0.1)

    def test_no_soft_fill_is_never_the_fill(self):
        self.assertEqual(cc.fill_only(floor(), blob(), None, self.mpp)[0], False)
        self.assertEqual(cc.fill_only(floor(), blob(), np.zeros((1024, 1024), bool), self.mpp)[0], False)


class SmallFaces(unittest.TestCase):
    def face(self, score, deg, az=0.0, el=-14.0):
        return dict(kind="face", score=score, az=az, el=el, deg=deg)

    def test_a_spool_hub_is_small_and_alone(self):
        self.assertTrue(pg.face_is_small_and_alone(self.face(0.81, 8.6, -101.5, -73.9), []))
        self.assertTrue(pg.face_is_small_and_alone(self.face(0.84, 4.5, -66.3, -14.3), [dict(kind="person", score=0.9, az=100.0, el=-70.0)]))

    def test_a_wide_face_or_a_sure_one_holds_on_its_own(self):
        self.assertFalse(pg.face_is_small_and_alone(self.face(0.80, 12.9), []))     # the camera holder's own, 10-17 degrees
        self.assertFalse(pg.face_is_small_and_alone(self.face(0.91, 4.0), []))

    def test_a_person_beside_it_holds_it(self):
        self.assertFalse(pg.face_is_small_and_alone(self.face(0.82, 5.0, 33, -13), [dict(kind="person", score=0.7, az=40.0, el=-5.0)]))

    def test_the_person_must_be_near(self):
        self.assertTrue(pg.face_is_small_and_alone(self.face(0.82, 5.0, 33, -13), [dict(kind="person", score=0.9, az=-120.0, el=-5.0)]))


class NadirPersons(unittest.TestCase):
    up = [0.0, 1.0, 0.0]

    def test_a_box_at_the_middle_of_the_view_is_straight_down(self):
        out = pg.nadir_boxes([(0.39, [400, 400, 624, 624])], self.up)
        self.assertEqual(len(out), 1)
        self.assertLess(out[0]["el"], -85); self.assertEqual(out[0]["score"], 0.39); self.assertTrue(out[0]["nadir"])

    def test_a_box_far_from_straight_down_is_left_to_the_side_views(self):
        self.assertEqual(pg.nadir_boxes([(0.9, [0, 0, 100, 100])], self.up), [])

    def test_the_polar_limit_is_the_one_given(self):
        boxes = [(0.5, [480, 100, 560, 200])]
        self.assertEqual(len(pg.nadir_boxes(boxes, self.up, polar_max=70)), 1)
        self.assertEqual(len(pg.nadir_boxes(boxes, self.up, polar_max=20)), 0)

    def test_a_detector_without_the_search_adds_nothing(self):
        class Old: pass
        self.assertEqual(pg.nadir_persons(Old(), np.zeros((512, 1024, 3), np.uint8), self.up), [])

    def test_the_straight_down_search_asks_for_the_lower_bar(self):
        calls = []
        class D:
            def persons_at(self, view, thr):
                calls.append((view.shape[:2], thr)); return [(0.4, [450, 450, 570, 570])]
        out = pg.nadir_persons(D(), np.zeros((512, 1024, 3), np.uint8), self.up)
        self.assertEqual(calls, [((1024, 1024), pg.NADIR_SCORE)]); self.assertEqual(len(out), 1)
        self.assertLess(pg.NADIR_SCORE, pg.PERSON_SCORE)


class HorizonPeople(unittest.TestCase):
    def setUp(self):                                                              # the gate alone: the paint-out (on by default since 2026-10-22) adds the zoomed look, which these tests do not ask for
        self._keep = os.environ.get("SHOP_TOUR_REMOVE_OTHERS"); os.environ["SHOP_TOUR_REMOVE_OTHERS"] = "0"

    def tearDown(self):
        os.environ.pop("SHOP_TOUR_REMOVE_OTHERS", None)
        if self._keep is not None:
            os.environ["SHOP_TOUR_REMOVE_OTHERS"] = self._keep

    def d(self, score, az=-153.0, el=0.0, view=(180, 0.0)):
        return dict(score=score, az=az, el=el, deg=3.0, view=view)

    def test_two_views_one_of_them_good_hold_a_far_person(self):                  # the flagger: 0.49 in his best view, 0.32 in another
        out = pg.horizon_people([self.d(0.49), self.d(0.32, -154.0, 2.0, (225, 0.0))])
        self.assertEqual(len(out), 1); self.assertEqual(out[0]["score"], 0.49); self.assertTrue(out[0]["horizon"])

    def test_one_view_alone_is_not_enough(self):
        self.assertEqual(pg.horizon_people([self.d(0.48), self.d(0.33)]), [])      # two boxes in the SAME view

    def test_a_weak_best_view_is_not_enough(self):
        self.assertEqual(pg.horizon_people([self.d(0.36), self.d(0.33, view=(225, 0.0))]), [])

    def test_only_round_the_horizon(self):
        self.assertEqual(pg.horizon_people([self.d(0.47, el=-40.0), self.d(0.45, el=-40.0, view=(225, 0.0))]), [])

    def test_two_people_are_two_clusters(self):
        out = pg.horizon_people([self.d(0.45), self.d(0.4, view=(225, 0.0)), self.d(0.44, az=60.0, view=(45, 0.0)), self.d(0.35, az=60.0, view=(90, 0.0))])
        self.assertEqual(len(out), 2)

    def test_scan_adds_the_far_person_and_keeps_what_it_always_found(self):
        class Det:
            def persons_many_at(self, views, thr):
                assert thr == pg.HORIZON_FLOOR
                out = [[] for _ in views]
                out[0] = [(0.6, [500, 480, 520, 540]), (0.42, [100, 480, 120, 540])]      # one sure person, one not
                out[1] = [(0.33, [100, 480, 120, 540])]
                return out
            def faces(self, v): return []
        found = pg.scan(Det(), np.zeros((512, 1024, 3), np.uint8))
        sure = [f for f in found if not f.get("horizon")]
        self.assertEqual([f["score"] for f in sure], [0.6])                           # what scores 0.5 or more is as before

    def test_it_can_be_switched_off(self):
        os.environ["SHOP_TOUR_HORIZON_PERSONS"] = "0"
        try:
            class Det:
                def persons_many_at(self, views, thr): raise AssertionError("not asked")
                def persons_many(self, views): return [[] for _ in views]
                def faces(self, v): return []
            self.assertEqual(pg.scan(Det(), np.zeros((512, 1024, 3), np.uint8)), [])
        finally:
            del os.environ["SHOP_TOUR_HORIZON_PERSONS"]


if __name__ == "__main__":
    unittest.main()
