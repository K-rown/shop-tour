"""Tests for the faster privacy path (2026-10-21): the ring search, the box-local scoring, the cached pixel maps, the one-process chain.
    python qa/test_privacy_chain.py
No models, GPU or footage: fakes stand in for the detectors and the segmentation.
"""
import json, os, sys, tempfile, unittest

os.environ.setdefault("SHOP_TOUR_PEOPLE", "remove")        # these tests are about the older flow: the paint-out and the holds (the default is now to smudge people and hold nothing: qa/test_people_blur.py)

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "qa"))
import capturer as cap            # noqa: E402
import privacy_gate as pg         # noqa: E402
import privacy_margin as pm       # noqa: E402
from test_units import synth_pano as _synth  # noqa: E402


def synth_pano(h, C=(0.0, 0.0, 0.0), turn=0.0, W=1024):
    return _synth(0, h, C, turn, W)


class TestRingSearch(unittest.TestCase):
    def _scene(self):
        h, az, d, turn = 2.0, 70.0, 2.5, 25.0
        a = np.radians(az); C = (np.sin(a) * d, 0.0, np.cos(a) * d)
        pano_i = synth_pano(h); pano_j = synth_pano(h, C, turn)
        m = np.zeros((cap.VIEW_N, cap.VIEW_N), bool); m[470:560, 480:550] = True
        fill = cap.Fill(m, [0, 1, 0], h)
        vc = cap.level_view_crop(pano_i, fill)
        k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        rpx = lambda x: max(1, int(x * fill.f / fill.h))
        hole = fill.hole
        ring = (cv2.dilate(hole.astype(np.uint8), k(rpx(cap.HALO_M))) > 0) & ~(cv2.dilate(hole.astype(np.uint8), k(rpx(0.05))) > 0)
        return fill, vc, ring, pano_j, h, az, d, turn

    def test_the_ring_search_finds_the_neighbour_and_prefers_the_right_turn(self):
        fill, vc, ring, pano_j, h, az, d, turn = self._scene()
        rs = cap.RingSearch(fill, vc, ring); gs = cap.small_gray(pano_j)
        good = cap.align_search(rs, gs, [0, 1, 0], h, az, d, turn, None)
        bad = cap.align_search(rs, gs, [0, 1, 0], h, az, d, -turn, None)
        self.assertGreater(good["ncc"], 0.4)
        self.assertGreater(good["ncc"], bad["ncc"] + 0.15)                        # a wrong turn sign is clearly worse
        self.assertTrue(abs(good["dyaw"]) <= 5 and abs(good["dd"]) <= 0.4 and abs(good["dl"]) <= 0.2)

    def test_the_ring_search_agrees_with_the_search_it_replaces(self):
        fill, vc, ring, pano_j, h, az, d, turn = self._scene()
        new = cap.align_search(cap.RingSearch(fill, vc, ring), cap.small_gray(pano_j), [0, 1, 0], h, az, d, turn, None)
        old, _, _ = cap.align_old(fill, vc, ring, pano_j, [0, 1, 0], h, az, d, turn, None)
        self.assertLess(abs(new["ncc"] - old["ncc"]), 0.15)
        self.assertLessEqual(abs(new["dyaw"] - old["dyaw"]), 2.5); self.assertLessEqual(abs(new["dd"] - old["dd"]), 0.2)

    def test_a_neighbour_whose_own_mask_covers_the_ring_gives_no_correlation(self):
        fill, vc, ring, pano_j, h, az, d, turn = self._scene()
        everything = np.ones((cap.VIEW_N, cap.VIEW_N), bool)
        r = cap.align_search(cap.RingSearch(fill, vc, ring), cap.small_gray(pano_j), [0, 1, 0], h, az, d, turn, everything)
        self.assertEqual(r["ncc"], -1.0)                                          # too few points left: the same -1 the old search gave


def old_score_candidates(cands, disp, cam_h, mpp):
    """capturer.score_candidates as it was before the box-local version (the same maths over the whole picture)."""
    best, bs, passed, found = None, 0.0, 0, []
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.3 / mpp) + 1,) * 2)
    seen = []
    for k, m, iou in cands:
        area = m.sum() * mpp * mpp
        if not 0.12 <= area <= 1.8 or iou < 0.5:
            continue
        if any((m & o).sum() > 0.9 * (m | o).sum() for o in seen):
            continue
        seen.append(m)
        mu = m.astype(np.uint8)
        cnts, _ = cv2.findContours(mu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea))) if cnts else 0
        if hull == 0 or m.sum() / hull < 0.55:
            continue
        ring = (cv2.dilate(mu, ker) > 0) & ~(cv2.dilate(mu, np.ones((9, 9), np.uint8)) > 0)
        if ring.sum() < 200:
            continue
        inside = np.median(disp[m]); out = disp[ring]
        h = cam_h * (1 - np.median(out) / inside); share = (out < inside * 0.93).mean()
        if h < 0.35 or share < 0.7:
            continue
        passed += 1
        sc = h + 0.5 * iou + 0.3 * min(area, 1.0)
        found.append((sc, m))
        if sc > bs:
            best, bs = m, sc
    if best is not None:
        ys, xs = np.nonzero(best); c0 = np.array([xs.mean(), ys.mean()]); u = best.copy()
        for sc, m in sorted(found, key=lambda t: -t[0]):
            ys, xs = np.nonzero(m)
            if np.hypot(xs.mean() - c0[0], ys.mean() - c0[1]) * mpp < 0.7 and (u | m).sum() * mpp * mpp <= 2.0:
                u |= m
        best = u
    return best, bs, passed


class TestScoreCandidates(unittest.TestCase):
    def test_the_same_pick_as_before_on_a_busy_set_of_candidates(self):
        n = cap.VIEW_N
        d, f = cap.view_dirs(); cam_h = 2.4; mpp = cam_h / f
        rng = np.random.default_rng(3)
        disp = np.full((n, n), 1.0, np.float32)
        yy, xx = np.mgrid[0:n, 0:n]
        cands = []
        for k in range(40):                               # blobs of person size, some standing high above the floor, some overlapping each other, some too small or too big
            cx, cy = rng.integers(150, n - 150, 2); rx = rng.integers(15, 70); ry = rng.integers(25, 110)
            m = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1
            if k % 3 == 0:
                disp[m] = 1.6
            cands.append((k, m, float(rng.uniform(0.4, 0.95))))
            if k % 5 == 0:
                cands.append((k, m.copy(), 0.9))           # the same shape twice
        for shift in (0, 1):
            cs = cands[shift:]
            a = old_score_candidates(cs, disp, cam_h, mpp); b = cap.score_candidates(cs, disp, cam_h, mpp)
            self.assertEqual(a[2], b[2]); self.assertAlmostEqual(a[1], b[1], places=9)
            self.assertEqual(a[0] is None, b[0] is None)
            if a[0] is not None:
                self.assertTrue((a[0] == b[0]).all())
        self.assertGreater(old_score_candidates(cands, disp, cam_h, mpp)[2], 0)            # (the set has real picks in it)


class TestCachedMaps(unittest.TestCase):
    def test_level_view_and_cut_give_what_they_gave_before(self):
        rng = np.random.default_rng(5)
        pano = rng.integers(0, 255, (256, 512, 3)).astype(np.uint8)
        up = [0.03, 0.998, -0.05]
        d, _ = cap.view_dirs(128, 130.0)
        u, v = cap.pano_uv(d @ cap.rot_to_y(up), *pano.shape[:2])
        want = cv2.remap(pano, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        np.testing.assert_array_equal(cap.level_view(pano, up, 128), want)
        np.testing.assert_array_equal(cap.level_view(pano, up, 128), want)       # the second time is the cached maps
        from depth_panos import cut_view
        for yaw, pitch in ((0.0, 0.0), (45.0, 0.0), (90.0, -40.0)):
            np.testing.assert_array_equal(pg.cut(pano, yaw, pitch), cut_view(pano, dict(yaw=yaw, pitch=pitch, fov=pg.VIEW_FOV), pg.VIEW_N))

    def test_scan_asks_the_detector_once_for_all_views_and_gets_the_same_answer(self):
        class Det:
            def persons(self, v):
                return [(0.9, [10.0, 10.0, 60.0, 120.0])] if v.mean() > 100 else []

            def faces(self, v):
                return []

        class DetMany(Det):
            calls = 0

            def persons_many(self, views):
                DetMany.calls += 1
                return [self.persons(v) for v in views]
        pano = np.full((256, 512, 3), 140, np.uint8)
        a = pg.scan(DetMany(), pano)
        self.assertEqual(DetMany.calls, 1)
        b = pg.scan(Det(), pano)
        self.assertEqual(a, b); self.assertGreater(len(a), 0)


def old_paste_back(pano, crop_img, hole, fill, feather_px=6):
    """capturer.paste_back as it was before it worked on the part the crop can cover only."""
    H, W = pano.shape[:2]
    r0 = int(H * 0.5)
    az = ((np.arange(W) + 0.5) / W * 2 * np.pi - np.pi).astype(np.float32); el = (np.pi / 2 - (np.arange(r0, H) + 0.5) / H * np.pi).astype(np.float32)
    A, E = np.meshgrid(az, el)
    r = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1)
    L = r @ cap.rot_to_y(fill.up).T.astype(np.float32)
    dy = np.maximum(-L[..., 1], 1e-3)
    col = (fill.f * L[..., 0] / dy + cap.FILL_N / 2 - fill.x0 - 0.5).astype(np.float32)
    row = (cap.FILL_N / 2 - fill.y0 - 0.5 - fill.f * L[..., 2] / dy).astype(np.float32)
    inside = (L[..., 1] < -0.3) & (col > 1) & (col < fill.shape[1] - 2) & (row > 1) & (row < fill.shape[0] - 2)
    wmask = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), feather_px)
    w = cv2.remap(wmask, col, row, cv2.INTER_LINEAR, borderValue=0) * inside
    w = np.where(cv2.remap(hole.astype(np.float32), col, row, cv2.INTER_LINEAR, borderValue=0) > 0.5, 1.0, w) * inside
    new = cv2.remap(crop_img, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = pano.copy()
    sub = out[r0:H]
    sub[:] = (sub.astype(np.float32) * (1 - w[..., None]) + new.astype(np.float32) * w[..., None]).astype(np.uint8)
    return out


class TestPasteBack(unittest.TestCase):
    def test_pasting_the_floor_back_gives_the_same_picture_as_before(self):
        for up in ([0, 1, 0], [0.04, 0.997, -0.06]):
            for box in ((470, 560, 480, 550), (300, 700, 250, 800)):
                pano = synth_pano(2.0, W=2048)
                m = np.zeros((cap.VIEW_N, cap.VIEW_N), bool); m[box[0]:box[1], box[2]:box[3]] = True
                fill = cap.Fill(m, up, 2.0)
                crop = (cap.level_view_crop(pano, fill) * 0.7).astype(np.uint8)
                a = old_paste_back(pano, crop, fill.hole, fill); b = cap.paste_back(pano, crop, fill.hole, fill)
                np.testing.assert_array_equal(a, b)
                self.assertGreater(int((a != pano).any(-1).sum()), 100)                  # (something was pasted)


class TestFillPool(unittest.TestCase):
    def test_the_floors_are_made_in_the_order_given_and_each_stop_is_announced_when_done(self):
        import remove_capturer as rc
        real = rc.fill_one
        seen = []
        rc.fill_one = lambda frames, k: (k, dict(status="filled", real_pct=float(k)), f"stop {k} done")
        try:
            with tempfile.TemporaryDirectory() as d:
                rep = {}; rp = os.path.join(d, "report.json")
                pool = rc.FillPool(d, ["001", "002", "003"], rep, rp, on_done=seen.append, jobs=1)
                for k in (0, 2, 1):
                    pool.submit(k)
                pool.close()
                self.assertEqual(seen, [0, 2, 1])
                self.assertEqual(rep["003"]["real_pct"], 2.0); self.assertEqual(json.load(open(rp))["002"]["status"], "filled")
                rc.fill_one = lambda frames, k: (_ for _ in ()).throw(RuntimeError("boom"))
                bad = rc.FillPool(d, ["001"], {}, rp, jobs=1)
                with self.assertRaises(RuntimeError):
                    bad.submit(0)
        finally:
            rc.fill_one = real

    def test_the_mask_stage_says_when_each_mask_is_there(self):
        import remove_capturer as rc
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "capturer"))
            ids = ["001", "002", "003"]
            for i in ids:
                cv2.imwrite(os.path.join(d, "capturer", i + "_mask.png"), np.zeros((cap.VIEW_N, cap.VIEW_N), np.uint8))
            pts = [dict(id=i, filename=i + ".jpg") for i in ids]
            got = []
            rc.mask_stage(d, pts, [[0, 1, 0]] * 3, None, [0, 1, 2], {}, lambda: None, lambda: None, reuse_masks=True, on_mask=got.append)
            self.assertEqual(got, [0, 1, 2])                                         # kept masks are announced too, and no model was asked for


class TestChain(unittest.TestCase):
    def test_the_margin_sweeps_only_the_stops_not_already_swept(self):
        asked = []

        def fake_sweep(frames, scales, ids=None, **kw):
            asked.append(sorted(ids)); return {sid: [(1.0, 2.4, []), (0.93, 2.2, []), (1.07, 2.6, [])] for sid in ids}
        with tempfile.TemporaryDirectory() as d:
            fr = os.path.join(d, "frames")
            for sub in ("safe", "clean", "capturer"):
                os.makedirs(os.path.join(fr, sub))
            ids = ["001", "002", "003"]
            json.dump([dict(id=i, filename=i + ".jpg") for i in ids], open(os.path.join(fr, "points.json"), "w"))
            for i in ids:
                cv2.imwrite(os.path.join(fr, "clean", i + ".jpg"), np.full((32, 64, 3), 90, np.uint8))
            json.dump(dict(version=1, stops={i: dict(held=False, reasons=[], persons=[], faces=[]) for i in ids}), open(os.path.join(fr, "privacy.json"), "w"))
            old = pm.sweep; pm.sweep = fake_sweep
            try:
                pre = {"001": [(1.0, 2.4, []), (0.93, 2.2, []), (1.07, 2.6, [])], "002": [(1.0, 2.4, []), (0.93, 2.2, ["a person-sized shape (0.31 m2) is still near the camera"], "best")]}
                pm.main(argv=["--frames", fr], precomputed=pre)
            finally:
                pm.sweep = old
            self.assertEqual(asked, [["003"]])                                    # 001 and 002 were swept while their pictures were being made: only 003 is left to do
            st = json.load(open(os.path.join(fr, "privacy.json")))["stops"]
            self.assertTrue(st["002"]["held"]); self.assertFalse(st["001"]["held"]); self.assertFalse(st["003"]["held"])

    def test_the_chain_runs_the_gate_the_margin_and_the_steps_it_joins(self):
        src = open(os.path.join(HERE, "privacy_gate.py"), encoding="utf-8").read()
        self.assertIn("G = Gate(a.frames, pts, state, old)", src); self.assertIn("G.stop(k)", src)
        chain = open(os.path.join(HERE, "privacy_chain.py"), encoding="utf-8").read()
        for needle in ("PG.Gate(", "RC.mask_stage(", "RC.FillPool(", "on_mask=on_mask", "PM.sweep(", "PG.finish(", "PM.main("):
            self.assertIn(needle, chain)
        for name in ('"poses.json"', '"poses_map.json"', '"vio"', "poses.npz", '"colmap"'):
            self.assertNotIn(name, chain)                                          # the chain, like the steps it joins, reads nothing from vio or the map


if __name__ == "__main__":
    unittest.main(verbosity=2)
