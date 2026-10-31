"""Tests for the speed round of 2026-10-22: the same numbers, made sooner.
    python qa/test_speed14.py
Every faster path is checked against the plain one (cv2, numpy, one thread, every step finished) on the same input. The video card is used where there is one; without a card
those tests run the fallback.
"""
import contextlib, io, json, os, sys, tempfile, threading, time, unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "qa"))
import capturer as cap            # noqa: E402

try:
    import torch
    CUDA = torch.cuda.is_available()
except Exception:                 # pragma: no cover
    torch, CUDA = None, False


def with_card():
    """Start a card context in this process so that capturer's card paths are the ones used (they use the card only in a process that already has one)."""
    if CUDA:
        torch.zeros(1).cuda()


class TestMorphOnTheCard(unittest.TestCase):
    def setUp(self):
        with_card()
        self.rng = np.random.default_rng(3)

    def _masks(self):
        out = []
        for H, W in ((300, 340), (211, 157)):
            m = (cv2.GaussianBlur(self.rng.random((H, W), np.float32), (0, 0), 5) > 0.5).astype(np.uint8)
            e = np.zeros((H, W), np.uint8); e[40:90, 30:120] = 1
            e2 = np.zeros((H, W), np.uint8); e2[:12, :] = 1; e2[:, -5:] = 1                # a blob that reaches the picture's edge is the border case
            out += [m, e, e2, np.zeros((H, W), np.uint8)]
        return out

    def test_dilate_erode_and_close_are_cv2s_numbers(self):
        for m in self._masks():
            for r in (1, 5, 12, 13, 30, 71, 150):                                      # (under 12 it is cv2's own call; 150 is wider than the picture)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                np.testing.assert_array_equal(cap.dilate_ellipse(m, r), cv2.dilate(m, k), f"dilate r={r}")
                np.testing.assert_array_equal(cap.erode_ellipse(m, r), cv2.erode(m, k), f"erode r={r}")
                np.testing.assert_array_equal(cap.close_ellipse(m, r), cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), f"close r={r}")

    def test_the_float_picture_of_the_depth_gets_the_same_erosion(self):
        d = cv2.GaussianBlur(self.rng.random((260, 300), np.float32) * 5 + 1, (0, 0), 3)
        for r in (20, 72):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            np.testing.assert_array_equal(cap.erode_ellipse(d, r), cv2.erode(d, k))
            np.testing.assert_array_equal(cap.dilate_ellipse(d, r), cv2.dilate(d, k))

    def test_switched_off_it_is_cv2(self):
        m = self._masks()[0]; k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
        old = os.environ.get("SHOP_TOUR_GPU_MORPH")
        os.environ["SHOP_TOUR_GPU_MORPH"] = "0"
        try:
            self.assertFalse(cap._gpu_morph_wanted())
            np.testing.assert_array_equal(cap.dilate_ellipse(m, 30), cv2.dilate(m, k))
        finally:
            os.environ.pop("SHOP_TOUR_GPU_MORPH", None)
            if old is not None:
                os.environ["SHOP_TOUR_GPU_MORPH"] = old

    def test_the_card_is_used_when_the_process_has_one(self):
        self.assertEqual(cap._gpu_morph_wanted(), CUDA)

    def test_a_card_that_runs_out_of_memory_leaves_the_numbers_to_the_cpu(self):
        m = self._masks()[0]; k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
        real, broken = cap._morph_on_card, cap._GPU_BROKEN[0]
        calls = []

        def boom(*a, **kw):
            calls.append(1); raise RuntimeError("CUDA out of memory")
        cap._morph_on_card = boom; cap._GPU_BROKEN[0] = False
        try:
            if CUDA:
                np.testing.assert_array_equal(cap.dilate_ellipse(m, 30), cv2.dilate(m, k))        # the card failed: cv2 answers
                self.assertTrue(cap._GPU_BROKEN[0])
                np.testing.assert_array_equal(cap.dilate_ellipse(m, 30), cv2.dilate(m, k))
                self.assertEqual(len(calls), 1)                                                    # ... and the card is not asked again by this process
        finally:
            cap._morph_on_card, cap._GPU_BROKEN[0] = real, broken


class TestMorphForTheFillWorkers(unittest.TestCase):
    """The fill workers have no card of their own: the main process makes their big dilations (remove_capturer.MorphServer) and the numbers are cv2's."""
    def test_a_worker_process_gets_cv2s_numbers_from_the_main_process(self):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        import remove_capturer as rc
        with_card()
        ctx = multiprocessing.get_context("spawn")
        rng = np.random.default_rng(9)
        m = (cv2.GaussianBlur(rng.random((260, 300), np.float32), (0, 0), 5) > 0.5).astype(np.uint8)
        server = rc.MorphServer(ctx, 2)
        try:
            with ProcessPoolExecutor(max_workers=2, mp_context=ctx, initializer=rc._fill_worker, initargs=(1, server.init_args())) as ex:
                futs = [ex.submit(cap.dilate_ellipse, m, r) for r in (20, 40, 60, 5)]
                got = [f.result(timeout=240) for f in futs]
        finally:
            server.close()
        for g, r in zip(got, (20, 40, 60, 5)):
            np.testing.assert_array_equal(g, cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))))

    def test_a_server_that_never_answers_leaves_the_numbers_to_the_worker(self):
        import queue, remove_capturer as rc
        req, rep = queue.Queue(), [queue.Queue()]
        m = np.zeros((120, 120), np.uint8); m[50:70, 50:70] = 1
        keep, wait = cap.REMOTE_MORPH[0], rc.MorphServer.WAIT_S
        rc.MorphServer.WAIT_S = 0.2
        cap.REMOTE_MORPH[0] = rc._remote_morph(req, rep, 0)
        try:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
            np.testing.assert_array_equal(cap.dilate_ellipse(m, 20), cv2.dilate(m, k))
            self.assertIsNone(cap.REMOTE_MORPH[0])                                                     # it is not asked again
        finally:
            cap.REMOTE_MORPH[0], rc.MorphServer.WAIT_S = keep, wait
            cap._GPU_BROKEN[0] = False


class TestLevelMapsOnTheCard(unittest.TestCase):
    def test_the_maps_are_numpys_maps_bit_for_bit(self):
        with_card()
        rng = np.random.default_rng(5)
        for _ in range(3):
            up = tuple(float(x) for x in (lambda v: v / np.linalg.norm(v))(rng.normal(size=3) * 0.1 + np.array([0, 1.0, 0])))
            for fov in (130.0, 75.0):
                os.environ["SHOP_TOUR_GPU_MAPS"] = "0"; cap._level_maps.cache_clear()
                a = cap._level_maps(up, 256, fov, 400, 800)
                os.environ["SHOP_TOUR_GPU_MAPS"] = "1"; cap._level_maps.cache_clear()
                b = cap._level_maps(up, 256, fov, 400, 800)
                np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
        os.environ.pop("SHOP_TOUR_GPU_MAPS", None); cap._level_maps.cache_clear()


class TestMaskStageThreads(unittest.TestCase):
    """remove_capturer.mask_stage with several masks at once: the same files, the same report, one on_mask at a time."""
    def _run(self, threads, fail=None):
        import remove_capturer as rc
        real_find, real_hole = cap.find_mask, rc.final_hole
        seen, busy, overlap = [], [0], [False]

        def fake_find(models, pano, up, h, hint_fn=None, boxes_fn=None):
            time.sleep(0.02)
            k = int(pano[0, 0, 0])
            if fail is not None and k == fail:
                raise RuntimeError("boom")
            m = np.zeros((cap.VIEW_N, cap.VIEW_N), bool); m[100 + 5 * k:200 + 5 * k, 300:400] = True
            return m, dict(view=pano, height=None, mpp=0.01, passed=1, score=0.5, fallback=None)
        cap.find_mask = fake_find; rc.final_hole = lambda view, mask, height, h, mpp: mask
        try:
            with tempfile.TemporaryDirectory() as d:
                os.makedirs(os.path.join(d, "capturer")); os.makedirs(os.path.join(d, "points"))
                ids = [f"{i:03d}" for i in range(1, 8)]
                for i, sid in enumerate(ids):
                    cv2.imwrite(os.path.join(d, "points", sid + ".png"), np.full((8, 16, 3), i, np.uint8))
                cv2.imwrite(os.path.join(d, "capturer", "004_mask.png"), np.zeros((cap.VIEW_N, cap.VIEW_N), np.uint8))           # one mask already there (reuse)
                pts = [dict(id=i, filename=i + ".png") for i in ids]
                P = type("P", (), {"cam_h": lambda self, k: 2.4})()
                rep = {}

                def on_mask(k):
                    busy[0] += 1
                    overlap[0] |= busy[0] > 1
                    time.sleep(0.005); seen.append(k); busy[0] -= 1
                rc.mask_stage(d, pts, [[0, 1, 0]] * 7, P, list(range(7)), rep, lambda: object(), lambda: type("D", (), {"clone": lambda s: s})(),
                              reuse_masks=True, on_mask=on_mask, threads=threads)
                files = {i: cv2.imread(os.path.join(d, "capturer", i + "_mask.png"), 0).copy() for i in ids}
                return seen, overlap[0], rep, files
        finally:
            cap.find_mask, rc.final_hole = real_find, real_hole

    def test_the_masks_and_the_report_are_the_same_with_one_thread_or_three(self):
        s1, o1, r1, f1 = self._run(1)
        s3, o3, r3, f3 = self._run(3)
        self.assertEqual(sorted(s1), list(range(7))); self.assertEqual(sorted(s3), list(range(7)))            # every stop announced once
        self.assertFalse(o1 or o3)                                                                           # ... and never two announcements at once
        self.assertEqual(r1, r3)
        for i in f1:
            np.testing.assert_array_equal(f1[i], f3[i])
        self.assertTrue(f3["004"].sum() == 0 and f3["003"].sum() > 0)                                         # the kept mask was not redone

    def test_a_failure_in_a_thread_is_the_steps_failure(self):
        with self.assertRaises(RuntimeError):
            self._run(3, fail=2)

    def test_one_thread_announces_in_the_order_of_the_stops(self):
        seen, *_ = self._run(1)
        self.assertEqual(seen, list(range(7)))


class TestSideRays(unittest.TestCase):
    def test_the_cached_rays_are_the_rays(self):
        import remove_capturer as rc
        from depth_panos import view_rays
        np.testing.assert_array_equal(rc._side_rays(90.0, -40.0, 90.0, 64), view_rays(90.0, -40.0, 90.0, 64))
        self.assertIs(rc._side_rays(90.0, -40.0, 90.0, 64), rc._side_rays(90.0, -40.0, 90.0, 64))


class TestStepSync(unittest.TestCase):
    def setUp(self):
        self.keep = {k: os.environ.pop(k, None) for k in ("SHOP_TOUR_DONE_DIR", "SHOP_TOUR_PARENT_PID")}

    def tearDown(self):
        for k, v in self.keep.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_nothing_waits_when_nobody_started_the_step_early(self):
        import stepsync
        self.assertFalse(stepsync.enabled()); self.assertEqual(stepsync.wait_for("stitch"), "ok")

    def test_it_waits_for_the_marker_and_hears_failure(self):
        import stepsync
        with tempfile.TemporaryDirectory() as d:
            os.environ["SHOP_TOUR_DONE_DIR"] = d
            self.assertIsNone(stepsync.status("hops"))
            threading.Timer(0.4, lambda: stepsync.mark(d, "hops", "ok")).start()
            t = time.time(); self.assertEqual(stepsync.wait_for("hops", poll=0.05), "ok"); self.assertGreaterEqual(time.time() - t, 0.3)
            stepsync.mark(d, "vio", "failed"); self.assertEqual(stepsync.wait_for("vio"), "failed")

    def test_it_stops_waiting_when_tour_py_is_gone(self):
        import stepsync, subprocess
        p = subprocess.Popen([sys.executable, "-c", "pass"]); p.wait()
        with tempfile.TemporaryDirectory() as d:
            os.environ["SHOP_TOUR_DONE_DIR"] = d; os.environ["SHOP_TOUR_PARENT_PID"] = str(p.pid)
            self.assertEqual(stepsync.wait_for("floor", poll=0.05), "failed")


class TestEarlyStarts(unittest.TestCase):
    """tour.run_steps with early starts, with stand-in steps that wait through stepsync the way the real ones do."""
    def _run(self, d, spec, early=True, prior=(), **kw):
        """spec: [(name, seconds, exit, waits_for)]; a step 'waits' by calling stepsync.wait_for once it has slept half its time."""
        import tour
        os.makedirs(os.path.join(d, "frames"), exist_ok=True)
        trace = os.path.join(d, "trace.txt")
        S = []
        for name, secs, code, waits in spec:
            out = os.path.join(d, "frames", name + ".txt")
            prog = ("import os,sys,time\n"
                    f"sys.path.insert(0,{HERE!r})\n"
                    "import stepsync\n"
                    f"open({trace!r},'a').write({name!r}+' start %f\\n' % time.time())\n"
                    f"time.sleep({secs / 2})\n"
                    f"w = {list(waits)!r}\n"
                    "got = [stepsync.wait_for(x, poll=0.02) for x in w]\n"
                    f"open({trace!r},'a').write({name!r}+' waited %f\\n' % time.time())\n"
                    f"time.sleep({secs / 2})\n"
                    f"print({name!r}+' said hello')\n"
                    f"if {code} or 'failed' in got: print('boom from {name}'); sys.exit({code or 7})\n"
                    f"open({out!r},'w').write('x')\n"
                    f"open({trace!r},'a').write({name!r}+' end %f\\n' % time.time())\n")
            S.append((name, [], [f"frames/{name}.txt"], (lambda p=prog: [[sys.executable, "-c", p]])))
        for name in prior:                                                       # outputs of an earlier run
            open(os.path.join(d, "frames", name + ".txt"), "w").write("old")
        old = (tour.WORK, tour.F)
        tour.WORK, tour.F = d, os.path.join(d, "frames")
        buf, err, timings = io.StringIO(), None, None
        try:
            with contextlib.redirect_stdout(buf):
                timings = tour.run_steps(S, "pinhole", "x.insv", "x.insv", early=early, **kw)
        except SystemExit as e:
            err = str(e.code)
        finally:
            tour.WORK, tour.F = old
        ev = {}
        if os.path.exists(trace):
            with open(trace) as fh:
                for line in fh:
                    n, what, t = line.split(); ev[(n, what)] = float(t)
        return timings, buf.getvalue(), err, ev

    # a small walk: the front (calib -> stitch), vio, the two chains, the stops' map, the privacy chain and the rings, with the times scaled down
    SPEC = [("calib", 0.2, 0, []), ("stitch", 1.0, 0, []), ("vio", 5.0, 0, ["stitch"]), ("level", 0.2, 0, []), ("hops", 0.4, 0, []), ("map", 1.0, 0, ["vio"]),
            ("depth", 0.8, 0, []), ("floor", 0.4, 0, []), ("rings", 0.1, 0, []), ("capturer", 1.6, 0, ["hops"]), ("privacy", 0.1, 0, [])]

    def test_vio_begins_with_the_lens_fit_and_reads_the_stops_when_they_are_there(self):
        with tempfile.TemporaryDirectory() as d:
            timings, out, err, ev = self._run(d, self.SPEC)
            self.assertIsNone(err, err)
            self.assertLess(ev[("vio", "start")], ev[("stitch", "end")])                  # it began while the stops were being stitched ...
            self.assertGreaterEqual(ev[("vio", "waited")], ev[("stitch", "end")] - 0.01)   # ... and read them only once they were done
            self.assertLess(ev[("capturer", "start")], ev[("hops", "end")])                 # the privacy chain began before the hops were done ...
            self.assertGreaterEqual(ev[("capturer", "waited")], ev[("hops", "end")] - 0.01)  # ... and made the floors only when they were
            self.assertLess(ev[("map", "start")], ev[("vio", "end")])                       # the stops' map began while vio still worked ...
            self.assertGreaterEqual(ev[("map", "waited")], ev[("vio", "end")] - 0.01)       # ... and took its path when it was done
            self.assertGreaterEqual(ev[("rings", "start")], ev[("map", "end")] - 0.01)      # what reads the map's result still waits for the whole step
            self.assertGreaterEqual(ev[("privacy", "start")], ev[("capturer", "end")] - 0.01)
            self.assertFalse(os.environ.get("SHOP_TOUR_DONE_DIR"))                          # the environment is left as it was

    def test_without_early_starts_every_step_waits_for_what_it_reads(self):
        with tempfile.TemporaryDirectory() as d:
            timings, out, err, ev = self._run(d, self.SPEC, early=False)
            self.assertIsNone(err, err)
            self.assertGreaterEqual(ev[("vio", "start")], ev[("stitch", "end")] - 0.01)
            self.assertGreaterEqual(ev[("map", "start")], ev[("vio", "end")] - 0.01)
            self.assertGreaterEqual(ev[("capturer", "start")], ev[("hops", "end")] - 0.01)

    def test_a_step_that_has_output_from_an_earlier_run_waits_as_before(self):
        """Whether such a step is up to date is decided from the files of the steps it reads: it does not begin early."""
        with tempfile.TemporaryDirectory() as d:
            timings, out, err, ev = self._run(d, self.SPEC, prior=("vio",))
            self.assertIsNone(err, err)
            self.assertGreaterEqual(ev.get(("vio", "start"), 1e18), ev[("stitch", "end")] - 0.01)

    def test_a_vio_that_fails_does_not_hide_behind_the_map_that_began_early(self):
        spec = [s if s[0] != "vio" else ("vio", 1.0, 3, ["stitch"]) for s in self.SPEC]
        with tempfile.TemporaryDirectory() as d:
            timings, out, err, ev = self._run(d, spec)
            self.assertIsNotNone(err)
            self.assertIn("[vio] failed", err); self.assertNotIn("[map] failed", err)       # vio's message is the news; map's exit is the consequence
            self.assertIn(("capturer", "end"), ev)                                           # what does not need vio still finished
            self.assertNotIn(("rings", "end"), ev)

    def test_a_failure_of_the_hops_stops_a_chain_that_began_early(self):
        spec = [s if s[0] != "hops" else ("hops", 0.4, 4, []) for s in self.SPEC]
        with tempfile.TemporaryDirectory() as d:
            timings, out, err, ev = self._run(d, spec)
            self.assertIsNotNone(err); self.assertIn("[hops] failed", err)
            self.assertNotIn(("capturer", "end"), ev)


class TestChainGate(unittest.TestCase):
    def test_the_chain_waits_for_the_hops_before_the_floors_and_not_before_the_masks(self):
        src = open(os.path.join(HERE, "privacy_chain.py"), encoding="utf-8").read()
        for needle in ("stepsync.wait_for(step", '("hops",)', "opened.wait()", "if stepsync.enabled():", "needs.update("):
            self.assertIn(needle, src)
        self.assertLess(src.index("threading.Thread(target=watch"), src.index("RC.mask_stage("))      # the watcher is up before the masks begin
        pi = open(os.path.join(HERE, "privacy_inputs.py"), encoding="utf-8").read()
        self.assertIn("def _links(self)", pi)                                                           # hops.json is read when the neighbours are first asked for

    def test_the_inputs_give_the_stick_height_before_the_hops_exist(self):
        import privacy_inputs as PI
        with tempfile.TemporaryDirectory() as d:
            old = os.environ.pop("SHOP_TOUR_PRIVACY_INPUTS", None)
            try:
                P = PI.Inputs(d, 3)
                self.assertEqual(P.cam_h(1), PI.stick_height())
                with self.assertRaises(OSError):
                    P.neighbours(0)
                json.dump(dict(pairs=[dict(i=0, j=1, ok=True, az_ij=10.0, az_ji=-170.0, yaw=5.0, dist_m=1.85, inliers=30)]), open(os.path.join(d, "hops.json"), "w"))
                self.assertEqual([n["j"] for n in P.neighbours(0)], [1])
            finally:
                if old is not None:
                    os.environ["SHOP_TOUR_PRIVACY_INPUTS"] = old


class TestTsdfOnlyTheFrustum(unittest.TestCase):
    def test_the_volume_is_the_same_as_over_the_whole_reach_cube(self):
        import dollhouse as D
        rng = np.random.default_rng(7)
        vols = {}
        colors = [(rng.random((64, 64, 3)) * 255).astype(np.uint8) for _ in range(3)]
        for mode in ("0", "1"):
            os.environ["SHOP_TOUR_TSDF_TIGHT"] = mode
            vol = D.Tsdf([-6.0, -6.0, -0.3], [6.0, 6.0, 3.5], 0.1, 0.3, torch.device("cpu"))
            n = 64
            f = n / 2 / np.tan(np.radians(45)); K = (f, n / 2, n / 2)
            for k, (yaw, cam) in enumerate(((0.0, (0.0, 0.0, 1.5)), (1.3, (1.0, -1.0, 1.5)), (2.9, (-2.0, 0.5, 1.2)))):
                depth = np.zeros((n, n), np.float32); depth[8:56, 6:58] = (3.0 + 0.5 * np.sin(np.arange(52))[None, :] + k).astype(np.float32)   # a wall over most of the face, no depth round it
                c, s = np.cos(yaw), np.sin(yaw)
                R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]); w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = -R @ np.array(cam)
                vol.integrate(depth, colors[k], K, w2c, reach=14.0, cam_pos=np.array(cam, float))
            vols[mode] = [vol.tsdf.numpy().copy(), vol.w.numpy().copy(), vol.col.numpy().copy(), vol.cw.numpy().copy()]
        os.environ.pop("SHOP_TOUR_TSDF_TIGHT", None)
        self.assertGreater(int((vols["0"][1] > 0).sum()), 500)                                             # (the test volume is not empty)
        for a, b in zip(vols["0"], vols["1"]):
            np.testing.assert_array_equal(a, b)

    def test_a_face_without_depth_touches_nothing(self):
        import dollhouse as D
        vol = D.Tsdf([-2.0, -2.0, 0.0], [2.0, 2.0, 2.0], 0.1, 0.3, torch.device("cpu"))
        vol.integrate(np.zeros((16, 16), np.float32), np.zeros((16, 16, 3), np.uint8), (8.0, 8.0, 8.0), np.eye(4), cam_pos=np.zeros(3))
        self.assertEqual(int((vol.w > 0).sum()), 0)


class TestVioFeaturesWhileDecoding(unittest.TestCase):
    def test_the_database_is_the_one_a_single_pass_makes(self):
        try:
            import pycolmap
        except Exception:
            self.skipTest("no pycolmap")
        import sqlite3
        import colmap_det, vio_map
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as d:
            idir, mdir = os.path.join(d, "images"), os.path.join(d, "masks"); os.makedirs(idir); os.makedirs(mdir)
            names = [f"f_{i:04d}.jpg" for i in range(11)] + ["s_001.jpg", "s_002.jpg"]
            for n in names:
                im = cv2.GaussianBlur((rng.random((256, 512, 3)) * 255).astype(np.uint8), (0, 0), 2.0)
                im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite(os.path.join(idir, n), im)
                cv2.imwrite(os.path.join(mdir, n + ".png"), np.full((256, 512), 255, np.uint8))
            pycolmap.logging.minloglevel = 2
            ro = pycolmap.ImageReaderOptions(); ro.camera_model = "EQUIRECTANGULAR"; ro.mask_path = mdir
            one = os.path.join(d, "one"); os.makedirs(one)
            colmap_det.extract(pycolmap, os.path.join(one, "db.db"), idir, ro, 512, 0)
            two = os.path.join(d, "two"); os.makedirs(two)
            ov = vio_map.Overlap(two, idir, mdir, 512, 0, batch=4, idle=0.2)
            order = list(names); rng.shuffle(order)
            for n in order:
                ov.add(n)
            self.assertTrue(ov.finish(len(names)))

            def dump(db):
                c = sqlite3.connect(db)
                try:
                    return {t: c.execute(f"select * from {t} order by 1, 2").fetchall() for t in ("cameras", "images", "keypoints", "descriptors", "frames", "frame_data", "rigs")}
                finally:
                    c.close()
            a, b = dump(os.path.join(one, "db.db")), dump(os.path.join(two, "db.db"))
            for t in a:
                self.assertEqual(a[t], b[t], t)
            self.assertGreater(len(a["keypoints"]), 5)

    def test_a_picture_that_never_came_is_a_failure_and_the_database_goes(self):
        try:
            import pycolmap  # noqa: F401
        except Exception:
            self.skipTest("no pycolmap")
        import vio_map
        with tempfile.TemporaryDirectory() as d:
            idir, mdir = os.path.join(d, "images"), os.path.join(d, "masks"); os.makedirs(idir); os.makedirs(mdir)
            ov = vio_map.Overlap(d, idir, mdir, 256, 0, idle=0.1)
            ov.add("missing.jpg")
            self.assertFalse(ov.finish(1))
            self.assertFalse(os.path.exists(os.path.join(d, "db.db")))


class TestLensMapsAndFit(unittest.TestCase):
    def test_the_lens_maps_come_out_the_same_made_side_by_side(self):
        import stitch_lib as sl
        P = sl.default_params(512)
        out = {}
        for th in ("1", "4"):
            os.environ["SHOP_TOUR_MAP_THREADS"] = th
            out[th] = sl.build_maps(P, 512, rows=64)
        os.environ.pop("SHOP_TOUR_MAP_THREADS", None)
        for k in out["1"]:
            np.testing.assert_array_equal(out["1"][k], out["4"][k], k)

    def test_the_fit_residual_is_the_old_ones_with_the_other_lens_taken_from_the_cache(self):
        import calibrate_stitch as CS, stitch_lib as sl
        rng = np.random.default_rng(4)
        pairs = [(cv2.GaussianBlur(rng.random((240, 240), np.float32) * 255, (0, 0), 2.0), cv2.GaussianBlur(rng.random((240, 240), np.float32) * 255, (0, 0), 2.0)) for _ in range(2)]
        bands = CS.band_rays(step=1.0)
        P0 = sl.scaled(sl.default_params(480), 0.5)
        x0 = np.array([P0[n] for n in sl.PARAM_NAMES]); scale = np.array([20, 20, 6, 6, 6, 6, 0.02, 0.02, 0.01, 0.01, 0.01])

        def old(x):                                        # the residual as it was before the lens sides were kept: everything from the parameters, every time
            P = dict(P0)
            for name, val in zip(sl.PARAM_NAMES, x):
                P[name] = val
            geo = []
            for v in bands:
                uA, vA, thA = sl.project(sl.to_lens(v, "A", P), "A", P)
                uF, vF, thF = sl.project(sl.to_lens(v, "F", P), "F", P)
                geo.append((uA, vA, uF, vF, ((thA < np.radians(94)) & (thF < np.radians(94))).astype(np.float32)))
            out = []
            for A, F in pairs:
                for uA, vA, uF, vF, valid in geo:
                    a = cv2.remap(A, uA, vA, cv2.INTER_LINEAR); f = cv2.remap(F, uF, vF, cv2.INTER_LINEAR)
                    r = (CS.local_norm(a, valid) - CS.local_norm(f, valid)) * valid
                    out.append(r[::3, ::3].ravel()); out.append(np.array([(1.0 - valid.mean()) * 40.0]))
            return np.concatenate(out)
        CS._SIDE.clear()
        base = x0 + scale * rng.normal(size=11) * 0.3
        np.testing.assert_array_equal(CS.residuals(base, pairs, bands, P0), old(base))
        for j in range(11):                                # the steps of a Jacobian: one parameter moved, the other lens (or this one's twin) from the cache
            x = base.copy(); x[j] += 0.3 * scale[j]
            np.testing.assert_array_equal(CS.residuals(x, pairs, bands, P0), old(x), f"parameter {j}")
        self.assertLessEqual(len(CS._SIDE), CS.SIDE_KEEP)


class TestImuFitsBesideEachOther(unittest.TestCase):
    def test_the_result_is_the_same_with_one_thread_or_four(self):
        import imu_scale as S
        s_true, g = 1.5, 9.80665
        ts = np.arange(0.5, 40.0, 1.0)
        pos = lambda t: np.stack([2.0 * np.sin(2 * np.pi * t / 7.0), 1.5 * np.sin(2 * np.pi * t / 11.0 + 1.0), 0.0 * t], -1)
        acc = lambda t: np.stack([-2.0 * (2 * np.pi / 7.0) ** 2 * np.sin(2 * np.pi * t / 7.0), -1.5 * (2 * np.pi / 11.0) ** 2 * np.sin(2 * np.pi * t / 11.0 + 1.0), 0.0 * t], -1)
        C = pos(ts) / s_true
        Rcw = np.tile(np.eye(3), (len(ts), 1, 1))
        tt = np.arange(0.0, 42.0, 1.0 / S.RATE)
        rng = np.random.default_rng(1)
        f = acc(tt) + np.array([0, 0, g]) + rng.normal(0, 0.15, (len(tt), 3))
        out = {}
        for th in ("1", "4"):
            os.environ["SHOP_TOUR_IMU_THREADS"] = th
            out[th] = json.loads(json.dumps(S.solve_arrays(tt + 81.0, f, np.zeros_like(f), ts, C, Rcw, offsets=[80.75, 81.0, 81.25], log=lambda *a, **k: None)))
        os.environ.pop("SHOP_TOUR_IMU_THREADS", None)
        self.assertEqual(out["1"], out["4"])
        self.assertAlmostEqual(out["4"]["scale_m_per_unit"], s_true, delta=0.08 * s_true)


if __name__ == "__main__":
    unittest.main(verbosity=2)
