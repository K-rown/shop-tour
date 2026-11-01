"""Fast unit tests for the maths and logic of the pipeline. No footage, GPU or browser needed, so they run on a fresh clone:
    python qa/test_units.py        (or: python -m unittest qa.test_units -v)
They check conventions that are easy to break silently: lens model, coordinate frames, the floor-plane projection between stops,
the nadir mask mapping, the exposure solver and tour.py's staleness logic / step order.
"""
import json, math, os, sys, tempfile, time, unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import stitch_lib as sl          # noqa: E402
import nadir                     # noqa: E402
import exposure                  # noqa: E402
import floorlink                 # noqa: E402
import hop_geometry as hg        # noqa: E402
import tour                      # noqa: E402
import imutil                    # noqa: E402
import build_tour                # noqa: E402
import capturer as cap           # noqa: E402
import privacy_gate as pg        # noqa: E402
import remove_capturer as rc     # noqa: E402
import dollhouse as dh           # noqa: E402
import floorplan as fp           # noqa: E402
sys.path.insert(0, os.path.join(HERE, "qa"))
import persp                     # noqa: E402
import grade_measure as gm       # noqa: E402


class TestStitchLib(unittest.TestCase):
    def test_pano_ray_convention(self):
        np.testing.assert_allclose(sl.rays(0, 0), [0, 0, 1], atol=1e-9)        # pano centre looks along +z
        np.testing.assert_allclose(sl.rays(90, 0), [1, 0, 0], atol=1e-9)       # azimuth is clockwise: 90 deg is +x (right)
        np.testing.assert_allclose(sl.rays(0, 90), [0, 1, 0], atol=1e-9)       # elevation up is +y

    def test_lens_axes_project_to_the_image_centre(self):
        P = sl.default_params(3840)
        for which, az in (("A", 0.0), ("F", 180.0)):                           # lens A = pano centre, lens F = looks backwards
            v = sl.rays(az, 0.0)[None]
            u, w, th = sl.project(sl.to_lens(v, which, P), which, P)
            self.assertAlmostEqual(float(u[0]), P["cx" + which], places=2)
            self.assertAlmostEqual(float(w[0]), P["cy" + which], places=2)
            self.assertAlmostEqual(float(th[0]), 0.0, places=5)

    def test_equidistant_lens_radius(self):
        P = sl.default_params(3840)
        v = sl.rays(0.0, 0.0)
        v90 = sl.rays(90.0, 0.0)[None]                                           # 90 deg off the lens A axis
        u, w, th = sl.project(sl.to_lens(v90, "A", P), "A", P)
        r = float(np.hypot(u[0] - P["cxA"], w[0] - P["cyA"]))
        self.assertAlmostEqual(r, P["fA"] * (np.pi / 2 + P["kA"] * (np.pi / 2) ** 3), delta=0.01)


class TestHopGeometry(unittest.TestCase):
    def test_level_matrix_takes_up_to_y(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            u = np.array([rng.normal() * 0.15, 1.0, rng.normal() * 0.15]); u /= np.linalg.norm(u)
            np.testing.assert_allclose(hg.level_matrix(u) @ u, [0, 1, 0], atol=1e-9)
            R = hg.level_matrix(u)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)

    def test_bearings(self):
        b = hg.bearings(np.array([[1000.0, 500.0], [0.0, 0.0], [500.0, 500.0]]), 2000, 1000)
        np.testing.assert_allclose(b[0], [0, 0, 1], atol=1e-9)                  # image centre = pano centre
        np.testing.assert_allclose(np.linalg.norm(b, axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(b[2], [-1, 0, 0], atol=1e-9)                 # a quarter of the way across = 90 deg to the left


def _links(az=0.0, d=2.0, turn=0.0, h=1.8):
    fl = floorlink.FloorLinks.__new__(floorlink.FloorLinks)
    fl.n = 2
    fl.links = [{1: {"to": 1, "az": az, "d": d, "turn": turn}}, {0: {"to": 0, "az": az + 180, "d": d, "turn": -turn}}]
    fl.cam_h = [h, h]; fl.L = [np.eye(3), np.eye(3)]; fl.turn_sign = 1.0
    return fl


class TestFloorLinks(unittest.TestCase):
    def test_floor_point_under_the_other_stop_is_straight_down(self):
        fl = _links(az=40.0, d=2.0)
        el = -np.arctan2(1.8, 2.0)                                              # floor point exactly 2 m away, along az 40
        aj, ej, hz = fl.project(0, 1, np.array([np.radians(40.0)]), np.array([el]))
        self.assertAlmostEqual(float(ej[0]), -np.pi / 2, places=6)              # from stop 1 it is directly below
        self.assertAlmostEqual(float(hz[0]), 0.0, places=6)

    def test_turn_is_added_to_the_azimuth(self):
        fl = _links(az=0.0, d=3.0, turn=30.0)
        az = np.array([np.radians(10.0)]); el = np.array([np.radians(-35.0)])
        fl0 = _links(az=0.0, d=3.0, turn=0.0)
        a1, e1, _ = fl.project(0, 1, az, el); a0, e0, _ = fl0.project(0, 1, az, el)
        self.assertAlmostEqual(float(a1[0] - a0[0]), np.radians(30.0), places=9)
        self.assertAlmostEqual(float(e1[0]), float(e0[0]), places=9)            # turning does not change elevation

    def test_raw_uv_centre_and_levelling(self):
        fl = _links()
        u, v = fl.raw_uv(0, np.array([0.0]), np.array([0.0]), 2000, 1000)
        self.assertAlmostEqual(float(u[0]), 999.5, places=6); self.assertAlmostEqual(float(v[0]), 499.5, places=6)
        up = np.array([0.1, 1.0, 0.05]); up /= np.linalg.norm(up)
        fl.L[0] = hg.level_matrix(up)
        u, v = fl.raw_uv(0, np.array([0.0]), np.array([np.pi / 2 - 1e-9]), 2000, 1000)   # levelled zenith -> the raw "up" direction
        az_r = (u[0] + 0.5) / 2000 * 2 * np.pi - np.pi; el_r = np.pi / 2 - (v[0] + 0.5) / 1000 * np.pi
        raw = np.array([np.cos(el_r) * np.sin(az_r), np.sin(el_r), np.cos(el_r) * np.cos(az_r)])
        np.testing.assert_allclose(raw, up, atol=2e-3)


class TestNadir(unittest.TestCase):
    def test_straight_down_view_maps_to_the_bottom_of_the_pano(self):
        m = np.zeros((101, 101), np.uint8); m[48:53, 48:53] = 255               # a spot in the middle of the straight-down view
        eq = nadir.view_to_equirect(m, 256, 512)
        self.assertGreater(int((eq[-6:] > 0).mean()), 0)                       # bottom rows, whatever the azimuth
        self.assertEqual(int((eq[:128] > 0).sum()), 0)                         # nothing above the horizon

    def test_central_person_keeps_only_regions_near_the_middle(self):
        m = np.zeros((200, 200), np.uint8)
        m[95:110, 90:105] = 255                                                  # next to the centre: the person under the camera
        m[20:40, 150:190] = 255                                                  # far out: a cable spool the model calls "person"
        keep = nadir.central_person(m)
        self.assertGreater(int((keep[95:110, 90:105] > 0).sum()), 0)
        self.assertEqual(int((keep[20:40, 150:190] > 0).sum()), 0)

    def test_person_hole_none_without_a_person_and_has_pole_rows(self):
        self.assertIsNone(nadir.person_hole(np.zeros((100, 100), np.uint8), 256, 512))
        m = np.zeros((100, 100), np.uint8); m[45:55, 45:55] = 255
        hole = nadir.person_hole(m, 256, 512)
        self.assertTrue(hole[-1].all())                                          # the stretched pole rows are always part of the hole

    def test_nobody_found_leaves_the_floor_alone(self):
        rng = np.random.default_rng(2)
        img = (rng.integers(60, 180, (1024, 2048, 3))).astype(np.uint8)
        out = nadir.fill_nadir(img, person=np.zeros((100, 100), np.uint8), floor=np.full((100, 100), 255, np.uint8))
        self.assertEqual(nadir.fill_nadir.last_mask, "nobody")
        h = img.shape[0]
        changed = (out != img).any(-1)
        self.assertEqual(int(changed[: int(h * 0.93)].sum()), 0)                  # nothing above the pole rows is painted over
        out2 = nadir.fill_nadir(img)                                                # no mask file at all: the fixed outline is the fallback
        self.assertEqual(nadir.fill_nadir.last_mask, "fixed")
        self.assertGreater(int((out2 != img).any(-1).sum()), 1000)

    def test_fill_survives_a_mask_that_covers_everything(self):
        img = np.full((512, 1024, 3), 120, np.uint8)
        out = nadir.fill_nadir(img, person=np.full((100, 100), 255, np.uint8))
        self.assertEqual(out.shape, img.shape)
        self.assertTrue(np.isfinite(out.astype(np.float32)).all())

    def test_fill_only_changes_the_bottom_band(self):
        rng = np.random.default_rng(1)
        h, w = 1024, 2048
        img = (np.linspace(80, 160, h)[:, None, None] + rng.normal(0, 3, (h, w, 3))).clip(0, 255).astype(np.uint8)
        person = np.zeros((200, 200), np.uint8); person[85:115, 85:115] = 255
        out = nadir.fill_nadir(img, person=person, floor=np.full((200, 200), 255, np.uint8))
        self.assertEqual(out.shape, img.shape); self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(out[: int(h * 0.5)], img[: int(h * 0.5)])
        self.assertGreater(int((out != img).any(-1).sum()), 1000)


class TestExposure(unittest.TestCase):
    def test_solver_recovers_known_gains(self):
        rng = np.random.default_rng(3)
        n = 6
        true = np.exp(rng.uniform(-0.2, 0.2, (n, 3)))
        rows = [(i, j) for i in range(n) for j in range(i + 1, n)]
        offs = np.array([np.log(true[i]) - np.log(true[j]) for i, j in rows])     # offset = log c_j - log c_i with g_i c_i = g_j c_j
        g = exposure.gains_from_offsets(rows, offs, np.ones(len(rows)), n)
        want = true / np.exp(np.log(true).mean(0))                                # the solve fixes the mean log gain at 0
        np.testing.assert_allclose(g, want, atol=1e-6)

    def test_no_pairs_means_no_correction(self):
        g = exposure.gains_from_offsets([], [], [], 4)
        np.testing.assert_allclose(g, np.ones((4, 3)))

    def test_gains_are_clamped(self):
        rows = [(0, 1)]
        g = exposure.gains_from_offsets(rows, np.array([[2.0, 2.0, 2.0]]), np.ones(1), 2)
        self.assertGreaterEqual(float(g.min()), exposure.LO - 1e-9); self.assertLessEqual(float(g.max()), exposure.HI + 1e-9)


class TestImutil(unittest.TestCase):
    def test_missing_file_names_the_file(self):
        with self.assertRaises(FileNotFoundError) as cm:
            imutil.imread(os.path.join(tempfile.gettempdir(), "no_such_pano_xyz.jpg"), wait=0.01)
        self.assertIn("no_such_pano_xyz.jpg", str(cm.exception))

    def test_file_that_appears_late_is_read(self):
        import threading
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "late.png")
            threading.Timer(0.3, lambda: cv2.imwrite(path, np.full((8, 8, 3), 77, np.uint8))).start()
            img = imutil.imread(path, tries=20, wait=0.1)                 # a file that is not there yet (or locked) is waited for
            self.assertEqual(int(img[0, 0, 0]), 77)

    def test_corrupt_file_raises_ioerror(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.jpg")
            with open(path, "wb") as fh:
                fh.write(b"not a picture")
            with self.assertRaises(IOError):
                imutil.imread(path, tries=2, wait=0.01)


class TestMemoryDefaults(unittest.TestCase):
    def test_parallel_jobs_scale_with_ram(self):
        real = imutil.total_ram_gb
        try:
            for ram, want in ((8, 1), (12, 3), (16, 4), (32, 4), (64, 4)):
                imutil.total_ram_gb = lambda r=ram: r
                self.assertEqual(imutil.parallel_jobs(2.2, 4), want, f"{ram} GB")
        finally:
            imutil.total_ram_gb = real
        self.assertGreater(imutil.total_ram_gb(), 0.5)


class TestBuild(unittest.TestCase):
    def test_prune_removes_pictures_of_stops_that_no_longer_exist(self):
        with tempfile.TemporaryDirectory() as d:
            for sub, names in (("panos", ["001.jpg", "002.jpg", "003.jpg"]), ("depth", ["001.png", "003.png"]), ("thumbs", ["001.jpg"])):
                os.makedirs(os.path.join(d, sub))
                for n in names:
                    open(os.path.join(d, sub, n), "wb").close()
            n = build_tour.prune(d, {"panos": {"001.jpg", "002.jpg"}, "depth": {"001.png"}, "thumbs": {"001.jpg"}})
            self.assertEqual(n, 2)
            self.assertEqual(sorted(os.listdir(os.path.join(d, "panos"))), ["001.jpg", "002.jpg"])
            self.assertEqual(sorted(os.listdir(os.path.join(d, "depth"))), ["001.png"])


class TestTour(unittest.TestCase):
    def test_step_order_is_the_documented_pipeline_order(self):
        names = [s[0] for s in tour.steps("x.insv")]
        self.assertEqual(names, ["stops", "calib", "stitch", "level", "hops", "map", "depth", "floor", "rings", "capturer", "privacy", "build", "dollhouse", "plan"])

    def test_video_on_another_drive_does_not_crash(self):
        other = "D:/Videos/walk.insv" if os.name == "nt" else "/mnt/d/Videos/walk.insv"
        self.assertTrue(tour.video_path(other).endswith("walk.insv"))              # os.path.relpath raised ValueError across drives
        self.assertEqual(tour.video_path(os.path.join(HERE, "raw", "walk.insv")), os.path.join("raw", "walk.insv"))

    def test_a_different_video_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.insv"), os.path.join(d, "b.insv")
            for path, n in ((a, 10), (b, 20)):
                with open(path, "wb") as fh:
                    fh.write(b"x" * n)
            fr = os.path.join(d, "frames")
            self.assertFalse(tour.source_changed(a, fr))        # nothing remembered yet
            tour.remember_source(a, fr)
            self.assertFalse(tour.source_changed(a, fr))        # same video again
            self.assertTrue(tour.source_changed(b, fr))         # another walk: everything must be rebuilt
            self.assertTrue(tour.source_changed(b, fr))         # ...and merely looking does not change the record (a failed attempt must not count)
            tour.remember_source(b, fr)
            self.assertFalse(tour.source_changed(b, fr))
            self.assertTrue(tour.source_changed(a, fr))

    def test_tour_report_flags_gaps_and_missing_stops(self):
        nodes = [{"id": f"{k:03d}", "x": float(k * 2), "y": 0.0, "links": [1, 2]} for k in range(6)]
        self.assertEqual(tour.tour_report(nodes), [])
        nodes[3]["x"] = 40.0                                   # a 32 m jump: stops 003 -> 004 and 004 -> 005
        nodes.append({"id": "007", "x": None, "y": None, "links": []})
        rep = " | ".join(tour.tour_report(nodes))
        self.assertIn("007", rep); self.assertIn("m apart", rep)

    def test_video_problems_are_explained(self):
        ok = {"streams": [{"codec_type": "video", "width": 3840, "height": 3840, "duration": "75.0"}] * 2}
        self.assertIsNone(tour.video_problem(ok))
        self.assertIn("cannot read", tour.video_problem({}))                                  # damaged / not a video
        eq = {"streams": [{"codec_type": "video", "width": 3840, "height": 1920, "duration": "30.0"}]}
        self.assertIsNone(tour.video_problem(eq))                                             # a stitched 2:1 equirectangular export is accepted
        sq = {"streams": [{"codec_type": "video", "width": 1920, "height": 1080, "duration": "30.0"}]}
        self.assertIn("not 2:1", tour.video_problem(sq))                                       # an ordinary video is not
        odd = {"streams": [ok["streams"][0], dict(ok["streams"][0], width=2880)]}
        self.assertIn("differ in size", tour.video_problem(odd))
        short = {"streams": [dict(ok["streams"][0], duration="2.0")] * 2}
        self.assertIn("shorter", tour.video_problem(short))

    def test_video_kind(self):
        import imutil
        d = {"width": 3840, "height": 3840}
        self.assertEqual(imutil.video_kind([d, d]), "dual")
        self.assertEqual(imutil.video_kind([{"width": 3840, "height": 1920}]), "equirect")
        self.assertIsNone(imutil.video_kind([{"width": 1920, "height": 1080}]))
        self.assertIsNone(imutil.video_kind([d, {"width": 2880, "height": 2880}]))
        self.assertIsNone(imutil.video_kind([]))

    def test_interrupted_steps_are_noticed(self):
        with tempfile.TemporaryDirectory() as d:
            fr = os.path.join(d, "frames"); os.makedirs(os.path.join(fr, "points"))
            pts = [{"id": f"{k:03d}", "filename": f"{k:03d}.jpg"} for k in (1, 2, 3)]
            with open(os.path.join(fr, "points.json"), "w") as fh:
                json.dump(pts, fh)
            self.assertIn("panoramas missing", tour.incomplete("stitch", fr, d))
            for p in pts:
                open(os.path.join(fr, "points", p["filename"]), "wb").close()
            self.assertIsNone(tour.incomplete("stitch", fr, d))
            with open(os.path.join(fr, "up.json"), "w") as fh:
                json.dump([[0, 1, 0]] * 2, fh)                                # one stop short
            self.assertIn("does not cover", tour.incomplete("level", fr, d))
            self.assertIn("privacy.json missing", tour.incomplete("build", fr, d))            # no build without the privacy check
            with open(os.path.join(fr, "privacy.json"), "w") as fh:
                json.dump({"stops": {p["id"]: {"held": False} for p in pts}}, fh)
            os.makedirs(os.path.join(fr, "safe"))
            for p in pts:
                open(os.path.join(fr, "safe", p["filename"]), "wb").close()
            self.assertIn("tour.json missing", tour.incomplete("build", fr, d))
            with open(os.path.join(fr, "poses.json"), "w") as fh:
                fh.write('{"nodes": [{"id": "001"')                          # cut off in the middle of writing
            self.assertIn("unreadable", tour.incomplete("rings", fr, d))

    def test_imwrite_is_atomic_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.jpg")
            imutil.imwrite(path, np.full((16, 16, 3), 90, np.uint8))
            self.assertEqual(os.listdir(d), ["a.jpg"])
            self.assertIsNotNone(cv2.imread(path))

    def test_harmonic_patch_meets_the_floor_and_adds_no_detail(self):
        S = 256
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        disc = np.stack([100 + 0.15 * xx, 110 + 0.1 * yy, 120 + 0 * xx], -1).astype(np.uint8)      # a smooth floor with a gradient
        hole = (np.hypot(xx - S / 2, yy - S / 2) < 40).astype(np.uint8) * 255
        flat = disc.copy(); flat[hole > 0] = (95, 105, 118)                                           # the plain fill: right on average, wrong at the edge
        ok = np.ones((S, S), bool); ok[hole > 0] = False
        fixed = nadir.harmonic_correction(flat, disc, hole, ok, np.full_like(disc, (95, 105, 118)))
        before, after = nadir.seam_score(disc, flat, hole), nadir.seam_score(disc, fixed, hole)
        self.assertLess(after, before)
        np.testing.assert_array_equal(fixed[hole == 0], flat[hole == 0])                                # nothing outside the hole changes
        hf = lambda x: x.astype(np.float32) - cv2.GaussianBlur(x, (0, 0), 4).astype(np.float32)
        moved = float(np.abs(fixed.astype(np.float32) - flat.astype(np.float32))[hole > 0].mean())
        self.assertGreater(moved, 2.0)                                                                  # it did move the patch ...
        self.assertLess(float(np.abs(hf(fixed) - hf(flat))[hole > 0].mean()), 0.3 * moved)             # ... by a slowly varying amount: fine detail is a small part of it
        self.assertTrue(np.array_equal(nadir.harmonic_correction(flat, disc, hole, np.zeros((S, S), bool), flat), flat))   # no known floor around: unchanged

    def test_tiles_cover_the_picture_with_padding(self):
        import build_tour
        rng = np.random.default_rng(1)
        img = rng.integers(0, 255, (60, 120, 3), dtype=np.uint8)                 # 2:1, 6 x 3 tiles of 20 x 20
        T = build_tour.cut_tiles(img, cols=6, rows=3, pad=4)
        self.assertEqual(len(T), 18)
        self.assertEqual(T[(0, 0)].shape, (28, 28, 3))
        for (r, c), t in T.items():                                            # the inside of every tile is exactly that part of the picture
            np.testing.assert_array_equal(t[4:-4, 4:-4], img[r * 20:(r + 1) * 20, c * 20:(c + 1) * 20])
        np.testing.assert_array_equal(T[(1, 0)][4:-4, :4], img[20:40, -4:])        # left padding of the first column wraps round the back
        np.testing.assert_array_equal(T[(1, 5)][4:-4, -4:], img[20:40, :4])
        np.testing.assert_array_equal(T[(0, 2)][:4, 4:-4], np.repeat(img[:1, 40:60], 4, 0))   # the poles repeat the edge row

    def test_work_folder_paths(self):
        old = tour.WORK
        try:
            tour.WORK = os.path.join(tempfile.gettempdir(), "walkX")
            self.assertEqual(tour._abs("frames/points.json"), os.path.join(tour.WORK, "frames/points.json"))
            self.assertEqual(tour._abs("tour/tour.json"), os.path.join(tour.WORK, "tour/tour.json"))
            self.assertEqual(tour._abs("viewer/index.html"), os.path.join(tour.HERE, "viewer/index.html"))   # code stays beside tour.py
            self.assertEqual(tour._abs("build_tour.py"), os.path.join(tour.HERE, "build_tour.py"))
        finally:
            tour.WORK = old

    def test_glob_patterns_take_the_work_folder_literally(self):
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, "shop [day 1]")
            os.makedirs(os.path.join(w, "frames"))
            open(os.path.join(w, "frames", "points.json"), "w").close()
            old = tour.WORK
            try:
                tour.WORK = w
                self.assertEqual(len(tour.mtimes(["frames/points.json"])[1]), 1)       # with the brackets read as a glob class this finds nothing
                self.assertFalse(tour.stale(["frames/points.json"], ["frames/points.json"]))
            finally:
                tour.WORK = old

    def test_one_tour_per_folder(self):
        with tempfile.TemporaryDirectory() as d:
            lf = tour.acquire_lock(d)
            self.assertTrue(os.path.exists(lf))
            tour.acquire_lock(d)                                  # the same process again is fine
            tour.release_lock(lf)
            self.assertFalse(os.path.exists(lf))
            with open(os.path.join(d, ".tour.lock"), "w") as fh:
                fh.write("2147483000")                           # a process that does not exist: a leftover lock
            tour.acquire_lock(d); tour.release_lock(os.path.join(d, ".tour.lock"))
        self.assertTrue(tour._pid_alive(os.getpid()))
        self.assertFalse(tour._pid_alive(2147483000))

    def test_stale_logic(self):
        old = tour.HERE
        try:
            with tempfile.TemporaryDirectory() as d:
                tour.HERE = d
                a, b = os.path.join(d, "in.txt"), os.path.join(d, "out.txt")
                self.assertTrue(tour.stale(["in.txt"], ["out.txt"]))              # output missing
                for path, txt in ((a, "x"), (b, "y")):
                    with open(path, "w") as fh:
                        fh.write(txt)
                t = time.time(); os.utime(a, (t - 100, t - 100)); os.utime(b, (t, t))
                self.assertFalse(tour.stale(["in.txt"], ["out.txt"]))             # output newer than input
                os.utime(a, (t + 100, t + 100))
                self.assertTrue(tour.stale(["in.txt"], ["out.txt"]))              # input newer than output
        finally:
            tour.HERE = old


def synth_pano(w, h_cam, C=(0.0, 0.0, 0.0), turn=0.0, W=1024):
    """A pano of a checkered floor plane (y = -h_cam, cells of 0.4 m, smooth colours) seen from camera position C (frame of stop i), turned by `turn` degrees."""
    H = W // 2
    az = ((np.arange(W) + 0.5) / W * 2 * np.pi - np.pi); el = (np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi)
    A, E = np.meshgrid(az, el)
    r = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1)
    D = r @ cap.roty(np.radians(turn))                                  # pano_j ray -> world ray (inverse of the turn)
    C = np.array(C, float)
    t = (-h_cam - C[1]) / np.minimum(D[..., 1], -1e-3)
    X = C + D * t[..., None]
    chk = ((np.floor(X[..., 0] / 0.4) + np.floor(X[..., 2] / 0.4)) % 2)
    img = np.stack([60 + 150 * chk + 20 * np.sin(X[..., 0]), 90 + 100 * chk, 200 - 120 * chk + 20 * np.cos(X[..., 2])], -1)
    img[D[..., 1] > -0.05] = 128
    return np.clip(img, 0, 255).astype(np.uint8)


class TestCapturer(unittest.TestCase):
    def test_level_view_round_trip_through_the_pano(self):
        pano = synth_pano(2.0, 2.0, W=1024)
        up = [0.03, 0.998, -0.05]
        v = cap.level_view(pano, up, 512)
        back, ok = cap.view_to_pano(v, up, *pano.shape[:2], n=512)
        sel = ok & (np.arange(pano.shape[0])[:, None] > pano.shape[0] * 0.75)
        self.assertGreater(sel.sum(), 1000)
        self.assertLess(float(np.abs(back.astype(int) - pano.astype(int))[sel].mean()), 12.0)

    def test_the_floor_seen_from_a_neighbour_matches_the_floor_seen_from_here(self):
        h, az, d, turn = 2.0, 70.0, 2.5, 25.0
        a = np.radians(az); C = (np.sin(a) * d, 0.0, np.cos(a) * d)
        pano_i = synth_pano(0, h); pano_j = synth_pano(0, h, C, turn)
        m = np.zeros((cap.VIEW_N, cap.VIEW_N), bool); m[470:560, 480:550] = True
        fill = cap.Fill(m, [0, 1, 0], h)
        here = cap.level_view_crop(pano_i, fill)
        good, ok = cap.sample_from(fill, pano_j, [0, 1, 0], h, az, d, turn, None)
        bad, ok2 = cap.sample_from(fill, pano_j, [0, 1, 0], h, az, d, -turn, None)
        g = lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 3)
        e_good = float(np.abs(g(here) - g(good))[ok].mean()); e_bad = float(np.abs(g(here) - g(bad))[ok2].mean())
        self.assertLess(e_good, 8.0)
        self.assertGreater(e_bad, 2.5 * e_good)                          # a wrong turn sign is clearly worse

    def test_a_capturer_mask_inside_the_neighbours_own_mask_is_not_used(self):
        h = 2.0
        pano = synth_pano(0, h); m = np.zeros((cap.VIEW_N, cap.VIEW_N), bool); m[470:560, 480:550] = True
        fill = cap.Fill(m, [0, 1, 0], h)
        a = np.radians(80.0); d = 2.0
        allmask = np.ones((cap.VIEW_N, cap.VIEW_N), bool)                 # the neighbour's whole nadir cone is 'him'
        _, ok = cap.sample_from(fill, pano, [0, 1, 0], h, 80.0, d, 0.0, allmask)
        self.assertEqual(int(ok.sum()), 0)

    def test_clean_regions_keeps_blobs_and_drops_speckle(self):
        m = np.zeros((200, 200), bool); m[40:120, 40:120] = True; m[150, 150] = True; m[160:163, 20:23] = True
        out = cap.clean_regions(m, 5, 3, 500)
        self.assertTrue(out[80, 80]); self.assertFalse(out[150, 150]); self.assertFalse(out[161, 21])

    def test_neighbours_come_previous_next_then_nearest(self):
        P = {"nodes": [{"links": [{"to": 3, "az": 0, "d": 5.0, "turn": 0}, {"to": 5, "az": 0, "d": 1.0, "turn": 0}, {"to": 2, "az": 0, "d": 9.5, "turn": 0},
                                  {"to": 4, "az": 0, "d": 2.0, "turn": 0}, {"to": 1, "az": 0, "d": 3.0, "turn": 0}]}] * 8}
        got = [n["j"] for n in rc.neighbours(P, 4)]
        self.assertEqual(got[:2], [3, 5]); self.assertNotIn(2, got)         # 9.5 m is too far to see the same floor well

    def test_faces_are_blurred_where_they_are_and_nowhere_else(self):
        rng = np.random.default_rng(1)
        pano = rng.integers(0, 255, (256, 512, 3)).astype(np.uint8)
        out = pg.blur_faces(pano, [{"az": 0.0, "el": 0.0, "deg": 20.0}])
        c = out[128 - 6:128 + 6, 256 - 6:256 + 6].astype(float).std(); far = out[20:40, 20:60].astype(float).std()
        self.assertLess(c, 0.4 * far)                                       # centre blurred
        np.testing.assert_array_equal(out[:, :60], pano[:, :60])            # far side untouched

    def test_one_person_seen_in_two_views_is_one_detection(self):
        f = [dict(kind="person", score=0.9, az=10.0, el=0.0, deg=30), dict(kind="person", score=0.6, az=14.0, el=2.0, deg=30), dict(kind="person", score=0.5, az=200.0, el=0.0, deg=30)]
        self.assertEqual(len(pg.dedupe(f)), 2)

    def test_pose_geometry_of_the_view_dirs(self):
        d, f = cap.view_dirs(256, 130.0)
        np.testing.assert_allclose(d[128, 128], [0, -1, 0], atol=0.01)       # the middle looks straight down
        self.assertGreater(d[10, 128, 2], 0.5)                                # image up = forward


def _wj(path, obj, raw=None):
    with open(path, "wb" if raw is not None else "w") as fh:
        fh.write(raw) if raw is not None else json.dump(obj, fh)


class TestPrivacyBuild(unittest.TestCase):
    """build_tour.py fails closed: no privacy result, no tour; a held-back stop is not in it and nothing of it is written."""
    def _frames(self, d, held=(), with_privacy=True):
        import subprocess
        fr = os.path.join(d, "frames"); os.makedirs(os.path.join(fr, "safe")); os.makedirs(os.path.join(fr, "points"))
        ids = ["001", "002", "003"]
        pts = [{"id": i, "filename": i + ".jpg", "x": float(k), "y": 0.0, "timestamp": float(k)} for k, i in enumerate(ids)]
        _wj(os.path.join(fr, "points.json"), pts)
        _wj(os.path.join(fr, "hops.json"), {"hops": [{"ok": False}, {"ok": False}], "pairs": []})
        link = lambda j: {"to": j, "az": 90.0, "turn": 0.0, "d": 2.0, "el": 40.0, "near": True}
        json.dump({"scale": 1.0, "nodes": [{"id": i, "x": float(k), "y": 0.0, "H": 0.0, "cam_h": 2.0, "links": [link(j) for j in range(3) if j != k]} for k, i in enumerate(ids)]},
                  open(os.path.join(fr, "poses.json"), "w"))
        _wj(os.path.join(fr, "three.min.js"), None, b"x")
        img = (np.random.default_rng(0).integers(0, 255, (64, 128, 3))).astype(np.uint8)
        for i in ids:
            cv2.imwrite(os.path.join(fr, "points", i + ".jpg"), img); cv2.imwrite(os.path.join(fr, "safe", i + ".jpg"), img)
        if with_privacy:
            _wj(os.path.join(fr, "privacy.json"), {"version": 1, "stops": {i: {"held": i in held, "reasons": [], "persons": [], "faces": []} for i in ids}})
        for i in held:
            os.remove(os.path.join(fr, "safe", i + ".jpg"))
        return subprocess.run([sys.executable, os.path.join(HERE, "build_tour.py"), "--tiles", "off"], cwd=d, capture_output=True, text=True)

    def test_no_privacy_result_no_tour(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._frames(d, with_privacy=False)
            self.assertNotEqual(r.returncode, 0); self.assertIn("privacy", r.stdout + r.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, "tour", "tour.json")))

    def test_held_stop_is_left_out_and_links_are_renumbered(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._frames(d, held=("002",))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            with open(os.path.join(d, "tour", "tour.json")) as fh:
                tj = json.load(fh)
            self.assertEqual([n["id"] for n in tj["nodes"]], ["001", "003"])
            self.assertEqual([[L["to"] for L in n["links"]] for n in tj["nodes"]], [[1], [0]])         # the link to 002 is gone, 003 is now number 1
            for sub in ("panos", "panos_lo", "panos_xs", "thumbs"):
                self.assertNotIn("002.jpg", os.listdir(os.path.join(d, "tour", sub)))
            self.assertIn("HELD BACK", r.stdout)

    def test_everything_held_builds_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._frames(d, held=("001", "002", "003"))
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse(os.path.exists(os.path.join(d, "tour", "tour.json")))

    def test_kept_stop_goes_in(self):
        with tempfile.TemporaryDirectory() as d:
            self._frames(d, held=("002",))
            with open(os.path.join(d, "frames", "privacy.json")) as fh:
                pv = json.load(fh)
            pv["stops"]["002"]["kept"] = True
            _wj(os.path.join(d, "frames", "privacy.json"), pv)
            cv2.imwrite(os.path.join(d, "frames", "safe", "002.jpg"), np.zeros((64, 128, 3), np.uint8))
            import subprocess
            r = subprocess.run([sys.executable, os.path.join(HERE, "build_tour.py"), "--tiles", "off"], cwd=d, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            with open(os.path.join(d, "tour", "tour.json")) as fh:
                self.assertEqual(len(json.load(fh)["nodes"]), 3)

    def test_a_held_stop_never_has_a_safe_picture(self):
        """privacy_gate removes the 'safe' copy of a stop that is held now (a stale copy must not slip into the tour)."""
        with tempfile.TemporaryDirectory() as d:
            fr = os.path.join(d, "frames"); os.makedirs(os.path.join(fr, "safe"))
            p = os.path.join(fr, "safe", "002.jpg"); _wj(p, None, b"")
            pv = {"stops": {"002": {"held": True}}}
            keep = [i for i in ("002",) if os.path.exists(os.path.join(fr, "safe", i + ".jpg")) and not (pv["stops"][i]["held"] and not pv["stops"][i].get("kept"))]
            self.assertEqual(keep, [])                                  # build_tour's rule: held and not kept = not built, even if a picture is there


class TestWhereWalksLive(unittest.TestCase):
    def test_inside_project(self):
        self.assertTrue(tour.inside_project(os.path.join(HERE, "raw", "walk.insv")))
        self.assertTrue(tour.inside_project(HERE))
        self.assertTrue(tour.inside_project(os.path.join(HERE, "..", os.path.basename(HERE), "raw", "x.insv")))      # a roundabout path is still inside
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(tour.inside_project(os.path.join(d, "walk.insv")))
        self.assertFalse(tour.inside_project(HERE + "-public"))                                                       # a sibling folder with the same prefix is not inside

    def test_walks_root_follows_the_environment_variable(self):
        old = os.environ.get("SHOP_TOUR_WALKS")
        try:
            os.environ["SHOP_TOUR_WALKS"] = "X:/my walks"
            self.assertEqual(tour.walks_root(), "X:/my walks")
            del os.environ["SHOP_TOUR_WALKS"]
            self.assertTrue(tour.walks_root().replace("\\", "/").endswith("walks"))
        finally:
            if old is not None:
                os.environ["SHOP_TOUR_WALKS"] = old

    def test_a_video_inside_the_project_is_refused_with_the_way_out(self):
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(HERE, "tour.py"), os.path.join(HERE, "raw", "nothing.insv")], capture_output=True, text=True,
                           env=dict(os.environ, SHOP_TOUR_WALKS="Q:/walks"))
        self.assertNotEqual(r.returncode, 0)
        out = r.stdout + r.stderr
        self.assertIn("inside the project folder", out); self.assertIn("Q:/walks", out); self.assertIn("SHOP_TOUR_WALKS", out)
        self.assertFalse(os.path.exists(os.path.join(HERE, "raw", "nothing")))                                       # nothing was created next to it

    def test_the_default_work_folder_is_the_folder_of_the_video(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, os.path.join(HERE, "tour.py"), os.path.join(d, "walk.insv")], capture_output=True, text=True)
            self.assertNotIn("inside the project folder", r.stdout + r.stderr)                                       # outside the project it is accepted as far as location goes
            self.assertFalse(os.path.exists(os.path.join(d, "walk")))                                                # and no folder named after the video is made


class TestDollhouse(unittest.TestCase):
    def test_level_to_world_puts_the_forward_direction_on_the_heading(self):
        for H in (0.0, 90.0, -30.0, 200.0):
            fw = dh.level_to_world(H) @ np.array([0, 0, 1.0])
            np.testing.assert_allclose(fw, [np.sin(np.radians(H)), np.cos(np.radians(H)), 0], atol=1e-9)        # bearing H: x east, y north
            np.testing.assert_allclose(dh.level_to_world(H) @ np.array([0, 1.0, 0]), [0, 0, 1], atol=1e-9)       # up stays up

    def test_range_decoding_matches_the_encoding_of_place_rings(self):
        png = np.zeros((2, 2, 3), np.uint8); png[..., 2] = [[0, 255], [128, 64]]
        r = dh.decode_range(png)
        self.assertAlmostEqual(float(r[0, 0]), 0.2, places=5); self.assertAlmostEqual(float(r[0, 1]), 60.0, places=3)
        self.assertAlmostEqual(float(r[1, 0]), 0.2 * 300 ** (128 / 255), places=3)

    def test_every_face_of_a_sphere_of_constant_range_lands_on_that_sphere(self):
        node = {"x": 3.0, "y": -2.0, "H": 40.0, "cam_h": 2.1}
        rng = np.full((256, 512), 3.0, np.float32)
        pano = np.full((512, 1024, 3), 90, np.uint8)
        f = dh.FACE / 2
        for col, zd, c2w, name in dh.stop_faces("frames", node, [0.02, 0.999, -0.03], rng, pano, None):
            ys, xs = np.nonzero(zd > 0)
            self.assertGreater(len(ys), 1000, name)
            x = (xs[::50] + 0.5 - dh.FACE / 2) / f * zd[ys[::50], xs[::50]]; y = (ys[::50] + 0.5 - dh.FACE / 2) / f * zd[ys[::50], xs[::50]]
            Pw = np.stack([x, y, zd[ys[::50], xs[::50]]], -1) @ c2w[:3, :3].T + c2w[:3, 3]
            np.testing.assert_allclose(np.linalg.norm(Pw - np.array([3.0, -2.0, 2.1]), axis=1), 3.0, atol=0.06)      # edge pixels removed, the rest on the sphere
            self.assertAlmostEqual(float(np.linalg.det(c2w[:3, :3])), 1.0, places=6)                                  # a proper camera, not a mirror

    def test_tsdf_recovers_a_flat_floor_at_the_right_height(self):
        import torch
        node = {"x": 0.0, "y": 0.0, "H": 0.0, "cam_h": 2.0}
        az = ((np.arange(512) + 0.5) / 512 * 2 * np.pi - np.pi); el = (np.pi / 2 - (np.arange(256) + 0.5) / 256 * np.pi)
        A, E = np.meshgrid(az, el)
        rng = np.where(np.sin(E) < -0.1, 2.0 / np.maximum(-np.sin(E), 0.1), 0.0).astype(np.float32)             # a floor 2 m below the camera
        pano = np.full((256, 512, 3), 120, np.uint8)
        vol = dh.Tsdf([-5, -5, -0.5], [5, 5, 1.0], 0.05, 0.3, torch.device("cpu"))
        for col, zd, c2w, name in dh.stop_faces("frames", node, [0, 1, 0], rng, pano, None):
            vol.integrate(zd, col, (dh.FACE / 2, dh.FACE / 2, dh.FACE / 2), np.linalg.inv(c2w), cam_pos=c2w[:3, 3], reach=6.0)
        V, T, C = vol.mesh()
        self.assertGreater(len(V), 500)
        self.assertLess(abs(float(np.median(V[:, 2]))), 0.06)                                                    # the floor is at Z = 0

    def test_quantised_export_round_trips(self):
        import open3d as o3d, tempfile
        m = o3d.geometry.TriangleMesh.create_box(2.0, 3.0, 1.0); m.compute_vertex_normals()
        m.vertex_colors = o3d.utility.Vector3dVector(np.tile([[0.2, 0.5, 0.8]], (len(m.vertices), 1)))
        with tempfile.TemporaryDirectory() as d:
            meta = dh.export(m, d)
            raw = np.fromfile(os.path.join(d, "dollhouse.bin"), np.uint8)
            self.assertEqual(len(raw), meta["bytes"])
            q = np.frombuffer(raw[:meta["vertices"] * 6].tobytes(), np.int16).reshape(-1, 3)
            V = np.array(meta["lo"]) + (q.astype(np.float64) + 32768) / 65535 * np.array(meta["span"])
            np.testing.assert_allclose(np.sort(V.max(0) - V.min(0)), [1.0, 2.0, 3.0], atol=1e-3)


class TestFloorPlan(unittest.TestCase):
    def _room(self, w, l, h, theta):
        import math
        c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        corners = np.array([[0, 0], [w, 0], [w, l], [0, l]], float)
        V, T = [], []
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            base = len(V)
            V += [[a[0], a[1], 0], [b[0], b[1], 0], [b[0], b[1], h], [a[0], a[1], h]]
            T += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
        V = np.array(V); V[:, :2] = V[:, :2] @ np.array([[c, -s], [s, c]]).T
        return V, np.array(T)

    def test_a_rotated_rectangular_room_gets_its_size_and_angle_back(self):
        V, T = self._room(6.0, 4.0, 2.5, 20.0)
        plan = fp.build_plan(V, T, 1.0)
        out = fp.outer(plan)
        self.assertEqual(sorted(out), ["u", "v"])
        size = sorted([out["u"][1] - out["u"][0], out["v"][1] - out["v"][0]])
        self.assertAlmostEqual(size[0], 4.0, delta=0.12); self.assertAlmostEqual(size[1], 6.0, delta=0.12)
        self.assertAlmostEqual((plan["theta"] - 20.0 + 45) % 90 - 45, 0.0, delta=1.0)

    def test_feet_and_inches(self):
        self.assertEqual(fp.ft_in(3.048), "10' 0\"")
        self.assertEqual(fp.ft_in(0.0254 * 18), "1' 6\"")


class TestGradeMeasure(unittest.TestCase):
    def test_pixel_to_direction_round_trip(self):
        az, el = persp.pixel_to_azel(500, 500, 40.0, -10.0, 70.0)                  # the middle of the view is the view direction
        self.assertAlmostEqual(az, 40.0, delta=0.2); self.assertAlmostEqual(el, -10.0, delta=0.2)
        az2, el2 = persp.pixel_to_azel(900, 500, 0.0, 0.0, 90.0)                   # right of centre = clockwise = larger azimuth
        self.assertGreater(az2, 30.0)
        az3, el3 = persp.pixel_to_azel(500, 100, 0.0, 0.0, 90.0)                   # above centre = up
        self.assertGreater(el3, 30.0)

    def test_a_point_one_metre_ahead_lands_one_metre_ahead(self):
        class W(gm.Walk):
            def __init__(self):
                self.P = [{"x": 2.0, "y": 3.0, "H": 90.0, "cam_h": 2.0}]; self.UP = [[0, 1, 0]]; self.idx = {"001": 0}; self.mode = "depth"
            def range_map(self, sid):
                return np.full((256, 512), 1.0, np.float32)
        w = W()
        p, r = w.point("001", 0.0, 0.0)                                            # heading 90 = east: one metre east of (2, 3), at the camera height
        np.testing.assert_allclose(p, [3.0, 3.0, 2.0], atol=1e-6)
        p2, _ = w.point("001", 0.0, -90.0)                                         # straight down: one metre below the camera
        np.testing.assert_allclose(p2, [2.0, 3.0, 1.0], atol=1e-6)

    def test_measure_kinds(self):
        class W(gm.Walk):
            def __init__(self):
                self.P = [{"x": 0.0, "y": 0.0, "H": 0.0, "cam_h": 2.0}]; self.UP = [[0, 1, 0]]; self.idx = {"001": 0}; self.mode = "depth"
            def range_map(self, sid):
                return np.full((256, 512), 2.0, np.float32)
        w = W()
        pts = [{"stop": "001", "az": 0.0, "el": 0.0}, {"stop": "001", "az": 90.0, "el": 0.0}]
        inch, _ = gm.measure(w, {"kind": "dist", "points": pts})
        self.assertAlmostEqual(inch, 2 * math.sqrt(2) / 0.0254, delta=1.0)
        h, _ = gm.measure(w, {"kind": "height", "points": [{"stop": "001", "az": 0.0, "el": -90.0}, {"stop": "001", "az": 0.0, "el": 90.0}]})
        self.assertAlmostEqual(h, 4.0 / 0.0254, delta=1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
