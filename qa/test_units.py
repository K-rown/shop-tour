"""Fast unit tests for the maths and logic of the pipeline. No footage, GPU or browser needed, so they run on a fresh clone:
    python qa/test_units.py        (or: python -m unittest qa.test_units -v)
They check conventions that are easy to break silently: lens model, coordinate frames, the floor-plane projection between stops,
the nadir mask mapping, the exposure solver and tour.py's staleness logic / step order.
"""
import json, math, os, sys, tempfile, time, unittest

os.environ.setdefault("SHOP_TOUR_PEOPLE", "remove")        # these tests are about the older flow: the paint-out and the holds (the default is now to smudge people and hold nothing: qa/test_people_blur.py)

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
import privacy_inputs as pin     # noqa: E402
import dollhouse as dh           # noqa: E402
import floorplan as fp           # noqa: E402
import level_panos as lp         # noqa: E402
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
        self.assertEqual(names, ["stops", "calib", "stitch", "vio", "level", "hops", "map", "depth", "floor", "rings", "capturer", "privacy", "pii", "early", "sphere", "register", "mvs", "align", "build", "dollhouse", "plan"])

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

    def test_a_lock_whose_number_now_belongs_to_another_program_is_ignored(self):
        """Windows hands a dead process's number to the next program: a browser that took it must not block every retry."""
        keep = (tour._pid_alive, tour._pid_image)
        try:
            tour._pid_alive = lambda pid: True                    # the number is in use...
            with tempfile.TemporaryDirectory() as d:
                lf = os.path.join(d, ".tour.lock")
                def lock_of(n):
                    with open(lf, "w") as fh:
                        fh.write(str(n))
                lock_of(4242)
                tour._pid_image = lambda pid: "chrome.exe"       # ...by a browser: ignored, and the lock is ours now
                tour.acquire_lock(d)
                self.assertEqual(open(lf).read().strip(), str(os.getpid()))
                tour.release_lock(lf)
                lock_of(4242)
                tour._pid_image = lambda pid: "python.exe"       # by Python: a real tour.py at work
                with self.assertRaises(SystemExit) as cm:
                    tour.acquire_lock(d)
                self.assertIn("already working", str(cm.exception))
                tour._pid_image = lambda pid: "python3.12.exe"
                with self.assertRaises(SystemExit):
                    tour.acquire_lock(d)
                tour._pid_image = lambda pid: None               # cannot be told: the careful answer, it counts as Python
                with self.assertRaises(SystemExit):
                    tour.acquire_lock(d)
        finally:
            tour._pid_alive, tour._pid_image = keep
        self.assertTrue(tour._pid_is_python(os.getpid()))
        self.assertTrue((tour._pid_image(os.getpid()) or "python").startswith("python"))

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

    def test_a_box_round_the_whole_person_adds_the_body_the_point_prompts_missed(self):
        """2026-10-21: SAM given a point on his head returned the head; the detector's BOX round him gives the body (walk 1 stop 015: 0.6 m2 -> ~2 m2)."""
        n = cap.VIEW_N; mpp = 0.0100
        head = np.zeros((n, n), bool); head[400:440, 500:540] = True
        body = np.zeros((n, n), bool); body[400:600, 480:560] = True
        far = np.zeros((n, n), bool); far[100:300, 100:180] = True

        class FakeModels:
            def __init__(self, masks): self.masks = masks
            def segment_box(self, view, box): return [(self.masks[0], 0.9), (self.masks[1], 0.95), (self.masks[1], 0.8)]
        dbg = {}
        got = cap.add_person_boxes(FakeModels([head, body]), None, head, [(0.9, [470.0, 390.0, 570.0, 610.0])], mpp, dbg)
        self.assertGreater(int(got.sum()), 4 * int(head.sum())); self.assertTrue(got[550, 520]); self.assertIn("box_added_m2", dbg)
        # what the point prompts found is a crate across the room, and the box finds him elsewhere: the crate is dropped, he is the mask
        dbg = {}
        got = cap.add_person_boxes(FakeModels([body, body]), None, far, [(0.9, [470.0, 390.0, 570.0, 610.0])], mpp, dbg)
        np.testing.assert_array_equal(got, body); self.assertIn("box_replaced", dbg)
        # two boxes: the best score is him (a weak one nearer the nadir does not outrank it), the masks of boxes that touch his join
        dbg = {}
        got = cap.add_person_boxes(FakeModels([head, far]), None, None, [(0.9, [470.0, 390.0, 570.0, 610.0])], mpp, dbg)
        self.assertTrue(got[420, 520])
        # nothing found by the depth and point prompts: the box's mask is the mask, said so
        dbg = {}
        got = cap.add_person_boxes(FakeModels([body, body]), None, None, [(0.9, [470.0, 390.0, 570.0, 610.0])], mpp, dbg)
        self.assertTrue(got[550, 520]); self.assertEqual(dbg["fallback"], "person detector box")
        # a mask that spills far outside its box, or is the size of the floor, is refused
        huge = np.zeros((n, n), bool); huge[:, :] = True
        self.assertIsNone(cap.add_person_boxes(FakeModels([huge, huge]), None, None, [(0.9, [470.0, 390.0, 570.0, 610.0])], mpp, {}))

    def test_a_weak_detection_counts_only_near_straight_down(self):
        """The capturer stands right under the camera: a 0.3 detection there is him, the same score at 50 degrees is not taken."""
        n = cap.VIEW_N; calls = []

        class FakeDet:
            torch = None
            def faces(self, v): return []
        det = FakeDet()
        near = [0.3, [500.0, 470.0, 560.0, 560.0]]             # centre at about 14 degrees from the nadir (view pixels at 130 degrees)
        far = [0.3, [700.0, 570.0, 760.0, 630.0]]               # about 44 degrees out
        strong_far = [0.8, [800.0, 400.0, 860.0, 500.0]]             # about 54 degrees out, a strong score: kept
        orig = rc._persons_below
        rc._persons_below = lambda d, views, thr: [[(s, list(b)) for s, b in (near, far, strong_far)] if k == 0 else [] for k in range(len(views))]
        try:
            fn = rc.person_boxes_fn(lambda: det)
            pano = np.zeros((256, 512, 3), np.uint8)
            got = fn(np.zeros((n, n, 3), np.uint8), pano, [0, 1, 0])
        finally:
            rc._persons_below = orig
        scores = [round(s, 2) for s, _ in got]
        self.assertIn(0.3, scores); self.assertIn(0.8, scores); self.assertEqual(len(got), 2)       # the weak one far from the nadir is dropped

    def test_one_neighbour_supplies_the_floor_where_two_do_not_agree(self):
        """The soft fill was a dark smudge where he stood whenever two neighbours did not agree; one well aligned neighbour is real floor."""
        h = 2.0
        pano_i = synth_pano(0, h); az, d, turn = 80.0, 2.0, 0.0
        a = np.radians(az); Cn = (np.sin(a) * d, 0.0, np.cos(a) * d)
        pano_j = synth_pano(0, h, Cn, turn)
        m = np.zeros((cap.VIEW_N, cap.VIEW_N), bool); m[470:560, 480:550] = True
        pano_dirty = pano_i.copy()
        # his own picture: paint a dark blob over the hole (what he looks like from above)
        view = cap.level_view(pano_dirty, [0, 1, 0]); view[m] = (30, 30, 30)
        back, ok = cap.view_to_pano(view, [0, 1, 0], *pano_i.shape[:2])
        pano_dirty[ok] = back[ok]
        nb = [dict(j=1, az=az, d=d, turn=turn)]
        load = lambda j: (pano_j, [0, 1, 0], h, None, None)
        new, st, dbg = cap.fill_stop(0, pano_dirty, [0, 1, 0], h, m, nb, load)
        self.assertEqual(st["real_pct"], 0.0)                            # one neighbour cannot AGREE with another ...
        self.assertGreater(st["single_pct"], 60.0)                       # ... but it still gives real floor
        self.assertLess(st["blur_pct"], 40.0)
        vn = cap.level_view(new, [0, 1, 0]); vo = cap.level_view(pano_i, [0, 1, 0])
        g = lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 3)
        err = float(np.abs(g(vn) - g(vo))[m].mean())
        self.assertLess(err, 25.0)                                        # the hole looks like the floor (a dark blob would be ~100 off)

    def test_a_soft_fill_gets_the_grain_of_the_floor_round_it(self):
        """2026-10-21: a flat blur where he stood is what a person still sees; the floor's own grain (a displaced copy of the floor round the hole) goes over it."""
        rng = np.random.default_rng(3)
        H = W = 700
        floor = np.clip(rng.normal(120, 14, (H, W, 1)) + np.zeros((H, W, 3)), 0, 255).astype(np.float32)
        floor = cv2.GaussianBlur(floor, (0, 0), 1.2)
        soft = np.zeros((H, W), bool); soft[300:420, 300:420] = True
        flat = floor.copy(); flat[soft] = 120                                    # the soft fill: the flat colour
        vc = np.clip(flat, 0, 255).astype(np.uint8)
        out, share = cap.add_floor_detail(vc.copy(), vc, soft, np.ones((H, W), bool))
        g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = lambda x, m: float(np.abs(cv2.Laplacian(cv2.GaussianBlur(g(x), (0, 0), 1.0), cv2.CV_32F))[m].mean())
        ring = np.zeros((H, W), bool); ring[100:200, 100:600] = True
        self.assertGreater(share, 0.9)
        self.assertLess(lap(vc, soft), 0.3 * lap(vc, ring))                       # it was flat ...
        self.assertGreater(lap(out, soft), 0.6 * lap(vc, ring)); self.assertLess(lap(out, soft), 1.6 * lap(vc, ring))     # ... and now has the floor's grain
        np.testing.assert_array_equal(out[~soft], vc[~soft])                      # nothing outside the patch changes
        # no floor to take the grain from: the blur is left as it is
        out2, share2 = cap.add_floor_detail(vc.copy(), vc, soft, np.zeros((H, W), bool))
        np.testing.assert_array_equal(out2, vc); self.assertEqual(share2, 0.0)

    def test_a_very_large_hole_does_not_break_the_alignment_search(self):
        """2026-10-21: a capturer close under a low camera (outdoors, 7 m2 with his helmet) made a ring of more than 32,767 points and cv2.remap refused it."""
        h = 2.0
        pano = synth_pano(0, h)
        m = np.zeros((cap.VIEW_N, cap.VIEW_N), bool); m[300:700, 300:700] = True
        fill = cap.Fill(m, [0, 1, 0], h)
        vc = cap.level_view_crop(pano, fill)
        ring = np.ones(fill.hole.shape, bool); ring[fill.hole] = False
        rs = cap.RingSearch(fill, vc, ring)
        self.assertLessEqual(len(rs.A), cap.RING_MAX_POINTS); self.assertEqual(len(rs.A), len(rs.P))
        a = np.radians(80.0); Cn = (np.sin(a) * 2.0, 0.0, np.cos(a) * 2.0)
        pj = synth_pano(0, h, Cn, 0.0)
        mj = np.ones((cap.VIEW_N, cap.VIEW_N), bool)                       # the neighbour's mask exists, so the remap of its mask runs too
        par = cap.align_search(rs, cap.small_gray(pj), [0, 1, 0], h, 80.0, 2.0, 0.0, mj)
        self.assertIn("ncc", par)

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
            self.assertIn("every stop was held back by the privacy check", r.stdout + r.stderr)             # the line older apps look for is still printed
            with open(os.path.join(d, "tour", "tour.json")) as fh:
                tj = json.load(fh)
            self.assertEqual(tj["nodes"], []); self.assertTrue(tj["held_all"])                               # a placeholder, never a tour

    def test_everything_held_still_writes_a_manifest_the_app_refuses_as_held(self):
        """The app can only offer the review when it finds a manifest. With every stop held there is no tour, but the manifest says so: status held, every
        stop listed with its reasons, no stops in the tour. manifest.ts judge() refuses status 'held' (or any held stop) before it looks at the stop count."""
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            self._frames(d, held=("001", "002", "003"))
            with open(os.path.join(d, "tour", "manifest.json")) as fh:
                m = json.load(fh)
            self.assertEqual(m["manifest_version"], 1)
            self.assertEqual(m["stops"], {"count": 0, "ids": []})
            self.assertEqual(m["privacy"]["status"], "held"); self.assertEqual([h["id"] for h in m["privacy"]["held"]], ["001", "002", "003"])
            self.assertEqual(m["privacy"]["checked"], 3); self.assertEqual((m["privacy"]["kept"], m["privacy"]["left_out"]), ([], []))
            self.assertEqual(m["measure"]["status"], "hidden")
            for k in ("engine", "created", "size_bytes", "features"):                                            # the shape manifest.ts parseManifest reads
                self.assertIn(k, m)
            pv = json.load(open(os.path.join(d, "frames", "privacy.json")))                                      # two held, one answered (left out): no stops, the open ones listed
            pv["stops"]["003"]["left_out"] = True; _wj(os.path.join(d, "frames", "privacy.json"), pv)
            r = subprocess.run([sys.executable, os.path.join(HERE, "build_tour.py"), "--tiles", "off"], cwd=d, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            m = json.load(open(os.path.join(d, "tour", "manifest.json")))
            self.assertEqual(m["privacy"]["status"], "held"); self.assertEqual([h["id"] for h in m["privacy"]["held"]], ["001", "002"]); self.assertEqual(m["privacy"]["left_out"], ["003"])

    def test_the_all_held_placeholder_is_never_called_a_finished_build(self):
        with tempfile.TemporaryDirectory() as d:
            self._frames(d, held=("001", "002", "003"))
            self.assertIn("placeholder", tour.incomplete("build", os.path.join(d, "frames"), os.path.join(d, "tour")))

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
        for col, zd, c2w, name, _wm in dh.stop_faces("frames", node, [0.02, 0.999, -0.03], rng, pano, None):
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
        for col, zd, c2w, name, _wm in dh.stop_faces("frames", node, [0, 1, 0], rng, pano, None):
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


class TestImuAndScale(unittest.TestCase):
    """The accelerometer reader and the visual-inertial scale."""
    @staticmethod
    def make_insv(path, n=6000, g_lsb=1025):
        import struct
        rng = np.random.default_rng(1)
        junk = rng.integers(0, 256, 5000, dtype=np.uint8).tobytes()               # the rest of the trailer (other records)
        t0 = 81_109_567
        body = bytearray()
        for k in range(n):
            acc = np.array([-g_lsb, 0, 0], float) + rng.normal(0, 8, 3)
            gyr = rng.normal(0, 4, 3)
            body += struct.pack("<II", t0 + k * 1000 + int(rng.integers(-3, 4)), 0) + struct.pack("<6H", *[int(round(v)) + 32768 for v in (*acc, *gyr)])
        tail = junk + bytes(body) + junk[:777]
        trailer = tail + struct.pack("<II", len(tail) + 40, 3) + b"8db42d694ccc418790edff439fe026bf"
        with open(path, "wb") as fh:
            fh.write(b"\x00" * 1000 + trailer)

    def test_reader_finds_the_stream_and_units(self):
        import imu as IMU
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.insv"); self.make_insv(p)
            r = IMU.read_imu(p)
        self.assertEqual(len(r["t"]), 6000)
        self.assertAlmostEqual(r["rate"], 1000, delta=2)
        self.assertAlmostEqual(float(np.median(np.linalg.norm(r["acc"], axis=1))), 9.80665, delta=0.01)    # gravity fixes the unit
        self.assertLess(abs(float(np.degrees(r["gyro"]).mean())), 0.5)

    def test_reader_refuses_a_file_without_a_trailer(self):
        import imu as IMU
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.mp4"); open(p, "wb").write(b"\x00" * 5000)
            with self.assertRaises(ValueError):
                IMU.read_imu(p)

    def test_second_divided_difference_is_half_the_acceleration(self):
        import imu_scale as S

        class P:
            pass
        o = P(); o.t_img = np.array([0.0, 0.7, 2.0]); a = 0.8
        o.C = np.stack([0.5 * a * o.t_img ** 2, 0 * o.t_img, 0 * o.t_img], 1)          # p'' = a, unevenly spaced pictures
        np.testing.assert_allclose(S.Problem.dd(o, 0, 1, 2), [a / 2, 0, 0], atol=1e-9)    # p[t0,t1,t2] = integral of (hat of area 1/2) * p''

    def test_scale_is_recovered_from_a_synthetic_walk(self):
        import imu_scale as S
        s_true, g = 1.5, 9.80665
        ts = np.arange(0.5, 60.0, 1.0)
        pos = lambda t: np.stack([2.0 * np.sin(2 * np.pi * t / 7.0), 1.5 * np.sin(2 * np.pi * t / 11.0 + 1.0), 0.0 * t], -1)
        acc = lambda t: np.stack([-2.0 * (2 * np.pi / 7.0) ** 2 * np.sin(2 * np.pi * t / 7.0), -1.5 * (2 * np.pi / 11.0) ** 2 * np.sin(2 * np.pi * t / 11.0 + 1.0), 0.0 * t], -1)
        C = pos(ts) / s_true                                                              # the pictures' path in map units
        Rcw = np.tile(np.eye(3), (len(ts), 1, 1))                                        # camera frame = world frame, z up
        tt = np.arange(0.0, 62.0, 1.0 / S.RATE)                                          # sensor clock = video clock + 81 s
        f = acc(tt) + np.array([0, 0, g])                                                  # what an accelerometer feels
        r = S.solve_arrays(tt + 81.0, f, np.zeros_like(f), ts, C, Rcw, offsets=[80.75, 81.0, 81.25], log=lambda *a, **k: None)
        self.assertAlmostEqual(r["scale_m_per_unit"], s_true, delta=0.05 * s_true)
        self.assertGreater(r["r2"], 0.9)


class TestMvsAndAlign(unittest.TestCase):
    def test_ncc_cost_is_zero_for_the_same_picture_and_high_for_noise(self):
        import torch, mvs
        W, H = 128, 64
        rng = np.random.default_rng(0)
        stretch = lambda a: (a - a.min()) / (a.max() - a.min())                         # full contrast, like a textured wall
        img = stretch(cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 2.0))
        noise = stretch(cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 2.0))
        ref = dict(id="a", pos=np.zeros(3), W=np.eye(3)); nb = dict(id="b", pos=np.zeros(3), W=np.eye(3)); nc = dict(id="c", pos=np.zeros(3), W=np.eye(3))
        dev = torch.device("cpu")
        g = {"a": torch.from_numpy(img)[None, None], "b": torch.from_numpy(img)[None, None], "c": torch.from_numpy(noise)[None, None]}
        same = mvs.Level(ref, [nb], g, {}, dev).cost(3.0)
        other = mvs.Level(ref, [nc], g, {}, dev).cost(3.0)
        self.assertLess(float(same[10:-10].mean()), 0.02)
        self.assertGreater(float(other[10:-10].mean()), 0.6)

    def test_depth_align_puts_the_model_on_the_stereo_depth_and_keeps_the_stereo(self):
        import depth_align as da
        rng = np.random.default_rng(0)
        r_da = (3.0 + 2.0 * np.abs(np.sin(np.linspace(0, 6, 1024)))[None, :] * np.ones((512, 1))).astype(np.float32)
        eq = np.zeros((512, 1024, 3), np.uint8); eq[..., 1] = 200; eq[..., 2] = da.encode(r_da)
        r_da_dec = dh.decode_range(eq)
        mv = np.zeros_like(r_da_dec); ys, xs = rng.integers(0, 512, 4000), rng.integers(0, 1024, 4000)
        mv[ys, xs] = r_da_dec[ys, xs] * 0.8                                                  # the stereo depth is 20 % nearer, everywhere
        r, keep, st = da.align_one(eq, mv)
        self.assertEqual(st["mode"], "anchored")
        np.testing.assert_allclose(r[ys, xs], mv[ys, xs], rtol=1e-6)                         # stereo is never changed
        far = np.ones(r.shape, bool); far[ys, xs] = False
        self.assertAlmostEqual(float(np.median(r[far] / r_da_dec[far])), 0.8, delta=0.03)   # the rest follows it
        r2, k2, st2 = da.align_one(eq, np.zeros_like(mv))
        self.assertTrue(st2["mode"].startswith("model only")); np.testing.assert_allclose(r2, r_da_dec)


class TestTexture(unittest.TestCase):
    def test_block_layout_and_corners(self):
        import dollhouse_texture as dt
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32); T = np.array([[0, 1, 2], [1, 3, 2]])
        P, nrm = dt.block_points(V, T, 6, 341)
        np.testing.assert_allclose(P[0, 0, 0], V[0]); np.testing.assert_allclose(P[0, 0, 5], V[1]); np.testing.assert_allclose(P[0, 5, 0], V[2])   # corner texels sit on the vertices
        np.testing.assert_allclose(nrm[0], [0, 0, 1], atol=1e-6)
        self.assertTrue(np.all(P[0, :, :, 2] == 0))                                        # the mirrored half stays on the triangle's plane
        self.assertTrue(np.all(P[0, :, :, 0] + P[0, :, :, 1] <= 1 + 1e-5))                 # ...and inside the triangle (u + v <= 1): the half beyond the long edge is mirrored back
        per_row, H = dt.layout(1000, 6, 2046)
        self.assertEqual(per_row, 341); self.assertEqual(H, 6 * 3)

    def test_a_wall_takes_the_colour_of_the_picture_that_faces_it(self):
        import dollhouse_texture as dt
        with tempfile.TemporaryDirectory() as d:
            W, H = 512, 256                                                                 # one stop at the origin: the picture is red straight ahead, green elsewhere
            az = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
            img = np.zeros((H, W, 3), np.uint8); img[:] = (0, 160, 0)
            img[:, np.abs(az) < 0.6] = (0, 0, 220)                                         # BGR red in front
            fp = os.path.join(d, "s.jpg"); cv2.imwrite(fp, img)
            V = np.array([[-1, 3, -1], [1, 3, -1], [-1, 3, 1], [1, 3, 1]], np.float32); T = np.array([[0, 1, 2], [1, 3, 2]])    # a 2 x 2 m wall at y = 3 (map north)
            stops = [dict(id="001", file=fp, pos=np.array([0.0, 0.0, 0.0]), W=dh.level_to_world(0.0), up=[0, 1, 0])]
            atlas, meta, seen = dt.bake(d, V, T, stops, log=lambda *a, **k: None, size=512)
        blk = atlas[:6, :12].reshape(-1, 3).astype(float)
        self.assertGreater(float(blk[:, 0].mean()), 150)                                    # RGB: red high, green low: the red part of the picture
        self.assertLess(float(blk[:, 1].mean()), 80)
        self.assertGreater(float(seen.min()), 0.9)


class TestNotesServer(unittest.TestCase):
    def test_validation(self):
        import serve_tour as st
        good = [{"id": "n1", "x": 1, "y": 2, "z": 1.2, "text": "hi", "photo": None}, {"id": "n2", "x": 0, "y": 0, "z": 0, "photo": "data:image/jpeg;base64,AAAA"}]
        self.assertTrue(st.valid(good)); self.assertTrue(st.valid([]))
        for bad in ({"id": "n"}, [{"id": "n", "x": "a", "y": 0, "z": 0}], [{"id": 3, "x": 0, "y": 0, "z": 0}], [{"id": "n", "x": 0, "y": 0, "z": 0, "photo": "http://x/y.jpg"}],
                    [{"id": "n", "x": 0, "y": 0, "z": 0, "text": "x" * 5000}], [{"id": "n%d" % i, "x": 0, "y": 0, "z": 0} for i in range(2001)]):
            self.assertFalse(st.valid(bad))

    def test_put_writes_next_to_the_tour_and_in_the_walk_folder_and_nothing_else(self):
        import serve_tour as st, threading, http.client
        with tempfile.TemporaryDirectory() as d:
            tour_dir = os.path.join(d, "walk", "tour"); os.makedirs(tour_dir)
            open(os.path.join(tour_dir, "index.html"), "w").write("<html>ok</html>")
            srv = st.Srv(("127.0.0.1", 0), st.handler_for(tour_dir)); port = srv.server_address[1]
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                body = json.dumps([{"id": "n1", "x": 1, "y": 2, "z": 1, "text": "check the breaker"}])
                c.request("PUT", "/notes.json", body, {"Content-Type": "application/json"}); r = c.getresponse(); r.read()
                self.assertEqual(r.status, 204)
                with open(os.path.join(tour_dir, "notes.json")) as fh:
                    self.assertEqual(json.load(fh)[0]["text"], "check the breaker")
                with open(os.path.join(d, "walk", "notes.json")) as fh:
                    self.assertEqual(json.load(fh)[0]["id"], "n1")
                c.request("PUT", "/index.html", "<html>pwned</html>"); r = c.getresponse(); r.read()
                self.assertEqual(r.status, 405)
                with open(os.path.join(tour_dir, "index.html")) as fh:
                    self.assertEqual(fh.read(), "<html>ok</html>")
                c.request("PUT", "/notes.json", "not json"); r = c.getresponse(); r.read(); self.assertEqual(r.status, 400)
                c.request("GET", "/notes.json"); r = c.getresponse(); self.assertEqual(r.status, 200); r.read()
            finally:
                srv.shutdown()


class TestPipelineSteps(unittest.TestCase):
    def test_new_steps_and_their_order(self):
        names = [s[0] for s in tour.steps("x.insv")]
        self.assertLess(names.index("stitch"), names.index("vio")); self.assertLess(names.index("vio"), names.index("map"))
        self.assertLess(names.index("privacy"), names.index("mvs")); self.assertLess(names.index("mvs"), names.index("align")); self.assertLess(names.index("align"), names.index("build"))
        self.assertLess(names.index("build"), names.index("dollhouse")); self.assertLess(names.index("dollhouse"), names.index("plan"))
        cmd = [s for s in tour.steps("x.insv") if s[0] == "map"][0][3]()[0]
        self.assertIn("--video", cmd)


class TestPlanAndPrune(unittest.TestCase):
    @staticmethod
    def wall_mesh(x0, y0, x1, y1, z0=0.0, z1=3.0, n=12):
        """A vertical wall from (x0, y0) to (x1, y1), z0..z1, as triangles."""
        V, T = [], []
        for i in range(n + 1):
            for j in range(n + 1):
                V.append([x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, z0 + (z1 - z0) * j / n])
        idx = lambda i, j: i * (n + 1) + j
        for i in range(n):
            for j in range(n):
                T += [[idx(i, j), idx(i + 1, j), idx(i, j + 1)], [idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]]
        return np.array(V, float), np.array(T)

    def test_wall_support_tells_a_backed_wall_from_a_guess(self):
        V, T = self.wall_mesh(0.0, 0.0, 4.0, 0.0)                                        # a wall along u at v = 0, 4 m long
        walls = [dict(axis="v", pos=0.0, a=0.0, b=4.0, length=4.0),                       # on the wall
                 dict(axis="v", pos=2.0, a=0.0, b=4.0, length=4.0),                       # 2 m beside it: nothing there
                 dict(axis="v", pos=0.0, a=0.0, b=8.0, length=8.0)]                       # twice as long as the wall: half backed
        s = fp.wall_support(V, T, walls, 0.0)
        self.assertGreater(s[0], 0.95); self.assertLess(s[1], 0.05); self.assertAlmostEqual(s[2], 0.5, delta=0.08)

    def test_triangles_no_stop_can_see_are_dropped(self):
        import dollhouse_texture as dt
        # a wall 3 m north of a stop, and a second small wall hidden right behind it (4 m north) and a floater far to the side that faces away
        V1, T1 = self.wall_mesh(-3.0, 3.0, 3.0, 3.0)
        V2, T2 = self.wall_mesh(-0.5, 4.0, 0.5, 4.0, 1.0, 2.0, 2)
        V = np.concatenate([V1, V2]); T = np.concatenate([T1, T2 + len(V1)])
        stops = [dict(id="001", pos=np.array([0.0, 0.0, 1.5]), W=dh.level_to_world(0.0))]
        seen = dt.visible_triangles(V, T, stops, W=512, H=256)
        self.assertGreater(seen[:len(T1)].mean(), 0.2)                                    # part of the front wall is in the field of view of this one stop
        self.assertFalse(seen[len(T1):].any())                                            # the one behind it is seen by nobody


class TestCameraHeightAndStereoSwitch(unittest.TestCase):
    def test_sparse_floor_finds_the_edge_of_the_cloud_not_the_densest_layer(self):
        import place_rings as pr

        class Pt:
            def __init__(self, xyz):
                self.xyz = xyz; self.error = 0.5
                self.track = type("T", (), {"length": staticmethod(lambda: 4)})()

        class Rec:
            pass
        rng = np.random.default_rng(0)
        n = 6000
        z = rng.uniform(-0.9, 0.7, n)                                                    # points from the floor (-0.9) up to the camera height (0.7)
        dense = rng.uniform(-0.6, -0.5, 2500)                                            # a thick layer of clutter 0.35 m above the floor: the densest layer
        zz = np.concatenate([z, dense, rng.uniform(-1.3, -1.0, 40)])                      # and a few points below the floor (outliers)
        P = np.stack([rng.uniform(-5, 5, len(zz)), rng.uniform(-5, 5, len(zz)), zz], 1)
        rec = Rec(); rec.points3D = {i: Pt(p_) for i, p_ in enumerate(P)}
        cams = {0: np.array([0.0, 0.0, 0.7]), 1: np.array([1.0, 0.0, 0.7])}
        floor = pr.sparse_floor(rec, [0, 0, 1.0], cams)
        self.assertIsNotNone(floor)
        self.assertAlmostEqual(floor, -0.9, delta=0.08)                                   # not -0.55, the dense clutter layer
        self.assertIsNone(pr.sparse_floor(rec, [0, 0, 1.0], {}))                          # no cameras: no answer
        few = Rec(); few.points3D = {i: Pt(p_) for i, p_ in enumerate(P[:100])}
        self.assertIsNone(pr.sparse_floor(few, [0, 0, 1.0], cams))                       # too few points: no answer

    def test_stereo_is_off_by_default_and_needs_the_accelerometer_check(self):
        import mvs
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("no poses", mvs.poses_not_confirmed(d))
            with open(os.path.join(d, "poses.json"), "w") as fh:
                json.dump({"scale_info": {"source": "hops (camera-height prior)"}}, fh)
            self.assertIn("did not confirm", mvs.poses_not_confirmed(d))
            with open(os.path.join(d, "poses.json"), "w") as fh:
                json.dump({"scale_info": {"source": "accelerometer + pictures (imu_scale.py)"}}, fh)
            self.assertIsNone(mvs.poses_not_confirmed(d))


class TestScalePriors(unittest.TestCase):
    def test_priors_factor_and_combination(self):
        import scale_priors as sp
        nodes = [{"cam_h": 2.0, "roof_m": 3.5}, {"cam_h": 2.2, "roof_m": 3.7}, {"cam_h": 2.1}]
        est = sp.estimate(nodes)
        self.assertAlmostEqual(est["camera_height"]["measured_m"], 2.1, places=3)
        self.assertAlmostEqual(est["camera_height"]["factor_typical"], 2.3 / 2.1, places=3)       # alone it would multiply the lengths by 2.3 / 2.1
        self.assertAlmostEqual(est["roof_height"]["factor_lo"], 3.0 / 3.6, places=3)
        c = sp.combine(est)
        self.assertTrue(c["consistent"]); self.assertTrue(c["current_inside"])                      # 1.0 is inside [max(1.8/2.1, 3/3.6), min(2.8/2.1, 6/3.6)]
        too_low = sp.combine(sp.estimate([{"cam_h": 1.2, "roof_m": 4.0}]))
        self.assertFalse(too_low["current_inside"])                                                # a 1.2 m camera is outside what a person can hold
        self.assertIsNone(sp.combine(sp.estimate([{}])))


class TestReviewMarks(unittest.TestCase):
    def test_a_ring_is_drawn_where_the_person_was_found_and_nowhere_else(self):
        img = np.full((512, 1024, 3), 90, np.uint8)
        out = pg.mark_detections(img, [dict(az=22.0, el=-79.0, deg=30.0)])
        self.assertTrue((img == 90).all(), "the input picture is left alone")
        x, y = int(round((22 / 360 + 0.5) * 1024)), int(round((0.5 + 79 / 180) * 512))      # 575, 481
        r = max(28, int(30 / 360 * 1024 / 2 * 1.3))
        ring = out[y, x + r]                                                  # a point on the ring: white
        self.assertTrue((ring == 255).all())
        self.assertTrue((out[y, x] == 90).all(), "the middle of the ring is the picture, untouched")
        changed = (out != 90).any(axis=2)
        self.assertGreater(changed.sum(), 100)
        far = changed.copy(); far[max(0, y - r - 8):y + r + 8, max(0, x - r - 8):x + r + 8] = False
        self.assertEqual(int(far.sum()), 0, "nothing is drawn away from the ring")

    def test_a_ring_across_the_edge_comes_out_the_other_side(self):
        img = np.full((512, 1024, 3), 90, np.uint8)
        out = pg.mark_detections(img, [dict(az=179.0, el=0.0, deg=20.0)])
        self.assertTrue((out[:, :40] != 90).any(), "the part of the ring past the right edge wraps to the left edge")

    def test_no_detection_no_change(self):
        img = np.full((64, 128, 3), 7, np.uint8)
        self.assertTrue((pg.mark_detections(img, []) == img).all())
        self.assertTrue((pg.mark_detections(img, None) == img).all())


class TestManifest(unittest.TestCase):
    def _tour(self, d, held=None, pct=7.9, ids=("001", "003")):
        os.makedirs(os.path.join(d, "frames")); os.makedirs(os.path.join(d, "tour"))
        stops = {i: dict(held=False, reasons=[]) for i in ("001", "002", "003")}
        for sid, extra in (held or {}).items():
            stops[sid] = dict(held=True, reasons=["a person"], **extra)
        _wj(os.path.join(d, "frames", "privacy.json"), dict(version=1, stops=stops))
        tj = dict(title="T", nodes=[dict(id=i) for i in ids])
        if pct is not None:
            tj["measure_pct"] = pct
        _wj(os.path.join(d, "tour", "tour.json"), tj)
        with open(os.path.join(d, "tour", "x.bin"), "wb") as fh:
            fh.write(b"0" * 1000)

    def test_clear_walk_writes_the_contract(self):
        import manifest
        with tempfile.TemporaryDirectory() as d:
            self._tour(d)
            m = manifest.write(os.path.join(d, "frames"), os.path.join(d, "tour"))
            on_disk = json.load(open(os.path.join(d, "tour", "manifest.json")))
            self.assertEqual(m, on_disk)
            self.assertEqual(on_disk["manifest_version"], 1)
            self.assertEqual(on_disk["stops"], dict(count=2, ids=["001", "003"]))
            self.assertEqual(on_disk["privacy"]["status"], "all_clear")
            self.assertEqual(on_disk["measure"], dict(status="shown", error_pct=7.9, max_pct=10.0, source="graded against tape on this walk"))
            self.assertGreaterEqual(on_disk["size_bytes"], 1000)
            self.assertIsNone(on_disk["walked"])
            self.assertTrue(on_disk["created"].endswith("Z")); self.assertEqual(on_disk["engine"]["name"], "shop-tour")

    def test_a_held_stop_nobody_looked_at_is_held(self):
        import manifest
        with tempfile.TemporaryDirectory() as d:
            self._tour(d, held={"002": {}})
            p = manifest.build(os.path.join(d, "frames"), os.path.join(d, "tour"))["privacy"]
            self.assertEqual(p["status"], "held"); self.assertEqual([h["id"] for h in p["held"]], ["002"])

    def test_looking_at_a_held_stop_clears_the_review_either_way(self):
        import manifest
        with tempfile.TemporaryDirectory() as d:
            self._tour(d, held={"002": {"left_out": True}})
            p = manifest.build(os.path.join(d, "frames"), os.path.join(d, "tour"))["privacy"]
            self.assertEqual((p["status"], p["left_out"], p["held"]), ("all_clear", ["002"], []))
        with tempfile.TemporaryDirectory() as d:
            self._tour(d, held={"002": {"kept": True}}, ids=("001", "002", "003"))
            p = manifest.build(os.path.join(d, "frames"), os.path.join(d, "tour"))["privacy"]
            self.assertEqual((p["status"], p["kept"]), ("all_clear", ["002"]))

    def test_a_stop_in_the_tour_that_was_not_cleared_is_reported(self):
        import manifest
        with tempfile.TemporaryDirectory() as d:
            self._tour(d, held={"002": {}}, ids=("001", "002"))
            p = manifest.build(os.path.join(d, "frames"), os.path.join(d, "tour"))["privacy"]
            self.assertEqual(p["status"], "held")

    def test_no_privacy_result_is_held_and_measure_is_hidden_above_ten_percent(self):
        import manifest
        with tempfile.TemporaryDirectory() as d:
            self._tour(d, pct=12.0)
            os.remove(os.path.join(d, "frames", "privacy.json"))
            m = manifest.build(os.path.join(d, "frames"), os.path.join(d, "tour"))
            self.assertEqual(m["privacy"]["status"], "held"); self.assertEqual(m["measure"]["status"], "hidden")
        with tempfile.TemporaryDirectory() as d:
            self._tour(d, pct=None)
            self.assertEqual(manifest.build(os.path.join(d, "frames"), os.path.join(d, "tour"))["measure"], dict(status="hidden", error_pct=None, max_pct=10.0, source="not checked on this walk"))

    def test_walked_on_comes_from_the_camera_file_name(self):
        import manifest
        with tempfile.TemporaryDirectory() as d:
            _wj(os.path.join(d, "source.json"), dict(name="VID_" + "20260101_101500_00_001.insv", size=1))
            self.assertEqual(manifest.walked_on(d), "2026-01-01")
            _wj(os.path.join(d, "source.json"), dict(name="walk.insv"))
            self.assertIsNone(manifest.walked_on(d))
            self.assertIsNone(manifest.walked_on(os.path.join(d, "nope")))

    def test_leave_out_is_recorded_without_scanning(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            self._tour(d, held={"002": {}})
            _wj(os.path.join(d, "frames", "points.json"), [dict(id=i, filename=i + ".jpg") for i in ("001", "002", "003")])
            r = subprocess.run([sys.executable, os.path.join(HERE, "privacy_gate.py"), "--leave-out", "002"], cwd=d, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            st = json.load(open(os.path.join(d, "frames", "privacy.json")))["stops"]["002"]
            self.assertTrue(st["left_out"]); self.assertFalse(st.get("kept"))


class TestEarlyTour(unittest.TestCase):
    """The first tour (build_early.py -> tour_early/): only when asked for, its own folder, partial in tour.json and the manifest, never fails the run."""
    def test_the_step_waits_for_pii_and_rings_and_the_sphere_waits_for_it(self):
        S = tour.steps("x.insv"); names = [x[0] for x in S]
        self.assertEqual(set(tour.prereqs("early", names)), {"pii", "rings"})
        self.assertIn("early", tour.prereqs("sphere", names))                       # poses.json is not rewritten under the early build
        e = [x for x in S if x[0] == "early"][0]
        self.assertEqual(e[2], ["tour_early/manifest.json"])                        # never the final tour/
        self.assertTrue(tour._abs("tour_early/manifest.json").startswith(tour.WORK))

    def test_it_is_off_unless_asked_for_and_never_fails(self):
        import build_early
        old = os.environ.pop("SHOP_TOUR_EARLY_TOUR", None)
        try:
            self.assertFalse(build_early.wanted())
            os.environ["SHOP_TOUR_EARLY_TOUR"] = "1"
            self.assertTrue(build_early.wanted())
        finally:
            os.environ.pop("SHOP_TOUR_EARLY_TOUR", None)
            if old is not None:
                os.environ["SHOP_TOUR_EARLY_TOUR"] = old

    def test_a_rebuild_after_a_review_skips_it(self):
        import build_early
        here = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                os.makedirs(os.path.join("frames", "register"))
                _wj(os.path.join("frames", "pii.json"), {}); time.sleep(0.02)
                self.assertFalse(build_early.rebuild_after_review())                 # no registration yet: a first run
                _wj(os.path.join("frames", "register", "stats.json"), {})
                self.assertTrue(build_early.rebuild_after_review())
            finally:
                os.chdir(here)

    def test_partial_travels_in_the_manifest_and_a_final_has_none(self):
        import manifest
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "frames")); os.makedirs(os.path.join(d, "t1")); os.makedirs(os.path.join(d, "t2"))
            _wj(os.path.join(d, "frames", "privacy.json"), dict(version=1, stops={"001": dict(held=False, reasons=[])}))
            _wj(os.path.join(d, "t1", "tour.json"), dict(title="T", nodes=[dict(id="001")], partial=True))
            _wj(os.path.join(d, "t2", "tour.json"), dict(title="T", nodes=[dict(id="001")]))
            a = manifest.write(os.path.join(d, "frames"), os.path.join(d, "t1")); b = manifest.write(os.path.join(d, "frames"), os.path.join(d, "t2"))
            self.assertTrue(a["partial"]); self.assertNotIn("partial", b)
            self.assertEqual(a["privacy"]["status"], "all_clear"); self.assertFalse(a["features"]["dollhouse"] or a["features"]["plan"])


class TestRunSteps(unittest.TestCase):
    """tour.run_steps: vio runs beside level/hops/depth/floor (CPU beside GPU) and is joined before map; SHOP_TOUR_SERIAL / serial=True is the old order."""
    def _run(self, d, spec, **kw):
        """spec: [(name, seconds, exit)]; every step writes frames/<name>.txt and a start/end line to trace.txt. Returns (timings, printed, error)."""
        import contextlib, io
        os.makedirs(os.path.join(d, "frames"), exist_ok=True)
        trace = os.path.join(d, "trace.txt")
        S = []
        for name, secs, code in spec:
            out = os.path.join(d, "frames", name + ".txt")
            prog = ("import os,sys,time\n"
                    f"open({trace!r},'a').write({name!r}+' start %f\\n' % time.time())\n"
                    f"time.sleep({secs})\n"
                    f"print({name!r}+' said hello')\n"
                    f"if {code}: print('boom from {name}'); sys.exit({code})\n"
                    f"open({out!r},'w').write('x')\n"
                    f"open({trace!r},'a').write({name!r}+' end %f\\n' % time.time())\n")
            S.append((name, [], [f"frames/{name}.txt"], (lambda p=prog: [[sys.executable, "-c", p]])))
        old = (tour.WORK, tour.F)
        tour.WORK, tour.F = d, os.path.join(d, "frames")
        buf, err, timings = io.StringIO(), None, None
        try:
            with contextlib.redirect_stdout(buf):
                timings = tour.run_steps(S, "pinhole", "x.insv", "x.insv", **kw)
        except SystemExit as e:
            err = str(e.code)
        finally:
            tour.WORK, tour.F = old
        ev = {}
        if os.path.exists(trace):
            with open(trace) as fh:
                for line in fh:
                    n, what, t = line.split()
                    ev[(n, what)] = float(t)
        return timings, buf.getvalue(), err, ev

    ORDER = [("vio", 4.0, 0), ("level", 0.1, 0), ("hops", 0.1, 0), ("map", 0.1, 0), ("depth", 0.4, 0), ("floor", 0.4, 0), ("rings", 0.1, 0)]

    def test_prereqs_are_the_data_dependencies_of_the_real_steps(self):
        names = [s[0] for s in tour.steps("x.insv")]
        keep = os.environ.pop("SHOP_TOUR_PRIVACY_INPUTS", None)
        try:
            P = {n: set(tour.prereqs(n, names)) for n in names}
            self.assertEqual(P["stops"], set()); self.assertEqual(P["calib"], {"stops"}); self.assertEqual(P["stitch"], {"calib"})
            self.assertEqual(P["level"], {"stitch"}); self.assertEqual(P["depth"], {"stitch"})              # level -> hops and depth -> floor are two chains
            self.assertEqual(P["hops"], {"level"}); self.assertEqual(P["floor"], {"depth"})
            self.assertEqual(P["capturer"], {"hops"}); self.assertEqual(P["privacy"], {"capturer"})   # neither reads the map or vio by default; the fill never used the floor step's masks
            self.assertEqual(P["map"], {"hops", "vio"}); self.assertIn("floor", P["rings"])
            self.assertEqual(P["build"], P["dollhouse"]); self.assertNotIn("dollhouse", P["build"]); self.assertEqual(P["plan"], {"dollhouse"})
            os.environ["SHOP_TOUR_PRIVACY_INPUTS"] = "map"              # opt-in: the privacy path reads the map's poses.json, so its steps stay behind map
            self.assertIn("map", tour.prereqs("capturer", names))
        finally:
            os.environ.pop("SHOP_TOUR_PRIVACY_INPUTS", None)
            if keep is not None:
                os.environ["SHOP_TOUR_PRIVACY_INPUTS"] = keep
        # every prerequisite is a step that exists, and the graph has no cycle: the pipeline order is a valid order
        for n in names:
            for p in tour.PREREQ.get(n, []):
                self.assertIn(p, names); self.assertLess(names.index(p), names.index(n), f"{p} before {n}")

    def test_the_graph_matches_what_each_step_reads(self):
        """A step whose input pattern is written by another step must wait (directly or through others) for that step."""
        S = tour.steps("x.insv"); names = [s[0] for s in S]
        writer = {}
        for n, ins, outs, _ in S:
            for o in outs:
                writer[o] = n
        def reach(n, seen=None):
            seen = set() if seen is None else seen
            for p in tour.PREREQ.get(n, []):
                if p not in seen:
                    seen.add(p); reach(p, seen)
            return seen
        for n, ins, outs, _ in S:
            for i in ins:
                w = writer.get(i)
                if w and w != n and i.startswith("frames/"):
                    self.assertIn(w, reach(n), f"{n} reads {i} written by {w}")

    def test_vio_runs_beside_depth_and_is_joined_before_map(self):
        with tempfile.TemporaryDirectory() as d:
            timings, out, err, ev = self._run(d, self.ORDER)
            self.assertIsNone(err, err)
            self.assertLess(ev[("depth", "start")], ev[("vio", "end")])                 # depth and floor ran while vio was still working
            self.assertLess(ev[("floor", "end")], ev[("vio", "end")])
            self.assertGreaterEqual(ev[("map", "start")], ev[("vio", "end")] - 0.01)      # map waited for vio
            lines = out.splitlines()
            at = lambda s: next(i for i, l in enumerate(lines) if l.startswith(s))
            self.assertLess(at("[vio] running"), at("[level] running"))                  # the app's progress parser sees every step start...
            self.assertLess(at("[vio] done in"), at("[map] running"))                    # ...and vio end before map starts
            self.assertIn("vio said hello", out)                                         # vio's own output is printed when it is joined
            for k in ("vio", "level", "hops", "map", "depth", "floor", "rings"):
                self.assertIn(k, timings)
            self.assertGreater(timings["vio_hidden"], 0.5)
            self.assertLess(timings["vio_hidden"], timings["vio"])

    def test_two_chains_run_beside_each_other_and_the_join_waits_for_both(self):
        with tempfile.TemporaryDirectory() as d:
            spec = [("level", 0.6, 0), ("hops", 0.6, 0), ("depth", 0.6, 0), ("floor", 0.6, 0), ("capturer", 0.1, 0)]
            timings, out, err, ev = self._run(d, spec)
            self.assertIsNone(err, err)
            self.assertLess(ev[("depth", "start")], ev[("hops", "end")])                  # depth ran while level -> hops was still working
            self.assertGreaterEqual(ev[("capturer", "start")], ev[("hops", "end")] - 0.01)   # capturer needs the hops (it used to wait for the floor too, whose masks it never used)
            self.assertGreater(timings["parallel_hidden"], 0.8)
            self.assertIn("front", timings)
            lines = out.splitlines()
            for n in ("level", "depth", "hops", "floor", "capturer"):                     # the app's reader sees every start and end, whole
                self.assertEqual(sum(1 for l in lines if l.startswith(f"[{n}] running...")), 1)
                self.assertEqual(sum(1 for l in lines if l.startswith(f"[{n}] done in")), 1)
            self.assertIn("level said hello", out)

    def test_build_runs_beside_dollhouse_and_plan_waits_for_dollhouse(self):
        with tempfile.TemporaryDirectory() as d:
            spec = [("align", 0.1, 0), ("build", 0.5, 0), ("dollhouse", 0.8, 0), ("plan", 0.1, 0)]
            timings, out, err, ev = self._run(d, spec)
            self.assertIsNone(err, err)
            self.assertLess(ev[("dollhouse", "start")], ev[("build", "end")])
            self.assertGreaterEqual(ev[("plan", "start")], ev[("dollhouse", "end")] - 0.01)

    def test_a_step_that_does_not_need_vio_still_finishes_when_vio_fails(self):
        with tempfile.TemporaryDirectory() as d:
            spec = [("vio", 0.2, 4), ("level", 0.5, 0), ("hops", 0.1, 0), ("map", 0.1, 0), ("depth", 0.1, 0), ("floor", 0.1, 0)]
            timings, out, err, ev = self._run(d, spec)
            self.assertIn("[vio] failed (exit 4)", err)
            for n in ("level", "hops", "depth", "floor"):                                 # nothing that does not need vio is lost
                self.assertTrue(os.path.exists(os.path.join(d, "frames", n + ".txt")), n)
            self.assertFalse(os.path.exists(os.path.join(d, "frames", "map.txt")))

    def test_vio_failing_while_depth_runs_fails_the_run_with_its_exit_and_log(self):
        with tempfile.TemporaryDirectory() as d:
            spec = [("vio", 0.6, 3)] + self.ORDER[1:]
            timings, out, err, ev = self._run(d, spec)
            self.assertIsNotNone(err)
            self.assertIn("[vio] failed (exit 3)", err)
            self.assertIn("boom from vio", err)                                          # the end of its log, like a step run in line
            self.assertTrue(os.path.exists(os.path.join(d, "frames", "depth.txt")))     # depth finished beside it and is kept
            self.assertFalse(os.path.exists(os.path.join(d, "frames", "map.txt")))      # map never ran

    def test_a_step_in_line_failing_stops_vio(self):
        with tempfile.TemporaryDirectory() as d:
            spec = [("vio", 30.0, 0), ("level", 0.1, 0), ("hops", 0.1, 0), ("depth", 0.2, 5), ("floor", 0.1, 0), ("map", 0.1, 0)]
            t0 = time.time()
            timings, out, err, ev = self._run(d, spec)
            self.assertIn("[depth] failed (exit 5)", err)
            self.assertLess(time.time() - t0, 15)                                        # it did not wait for vio's 30 seconds
            time.sleep(0.3)
            self.assertNotIn(("vio", "end"), ev)

    def test_serial_runs_in_pipeline_order(self):
        with tempfile.TemporaryDirectory() as d:
            spec = [("vio", 0.3, 0), ("level", 0.1, 0), ("hops", 0.1, 0), ("map", 0.1, 0), ("depth", 0.1, 0), ("floor", 0.1, 0), ("rings", 0.1, 0)]
            timings, out, err, ev = self._run(d, spec, serial=True)
            self.assertIsNone(err, err)
            starts = sorted((t, n) for (n, w), t in ev.items() if w == "start")
            self.assertEqual([n for _, n in starts], [s[0] for s in spec])
            self.assertLessEqual(ev[("vio", "end")], ev[("level", "start")])
            self.assertEqual(timings["vio_hidden"], 0.0)
            os.environ["SHOP_TOUR_SERIAL"] = "1"
            try:
                self.assertTrue(tour.serial_run())
            finally:
                del os.environ["SHOP_TOUR_SERIAL"]
            self.assertFalse(tour.serial_run())

    def test_from_a_step_forces_the_steps_after_it_in_pipeline_order(self):
        with tempfile.TemporaryDirectory() as d:
            spec = [("vio", 0.1, 0), ("level", 0.1, 0), ("hops", 0.1, 0), ("map", 0.1, 0), ("depth", 0.1, 0), ("floor", 0.1, 0), ("rings", 0.1, 0)]
            self._run(d, spec)                                                           # everything built once
            timings, out, err, ev = self._run(d, spec, start="depth")                    # depth is moved up in the run order, but "from depth" still means depth, floor, rings
            self.assertIsNone(err, err)
            self.assertEqual(sorted(k for k in timings if k not in ("wall", "parallel_hidden", "front")), ["depth", "floor", "rings"])
            self.assertIn("[level] up to date", out); self.assertIn("[map] up to date", out)

    def test_only_a_step_runs_it_in_line(self):
        with tempfile.TemporaryDirectory() as d:
            timings, out, err, ev = self._run(d, [("vio", 0.1, 0), ("level", 0.1, 0), ("map", 0.1, 0)], only="vio")
            self.assertEqual(sorted(timings), ["vio", "vio_hidden"]); self.assertEqual(timings["vio_hidden"], 0.0)

    def test_total_counts_vio_once(self):
        with tempfile.TemporaryDirectory() as d:
            old = tour.write_timings({"stops": 90.0, "calib": 130.0, "stitch": 40.0, "vio": 480.0, "level": 15.0, "depth": 60.0, "floor": 200.0, "vio_hidden": 275.0}, d)
            self.assertEqual(old["total"], 40.0 + 480.0 + 15.0 + 60.0 + 200.0 - 275.0)
            again = tour.write_timings({"map": 100.0}, d)                                # a later run keeps the earlier times and adds its own
            self.assertEqual(again["total"], 40.0 + 480.0 + 15.0 + 60.0 + 200.0 + 100.0 - 275.0)
            self.assertEqual(again["vio"], 480.0)


class TestVioExtract(unittest.TestCase):
    """vio_map.decode_lens: one sequential pass picks the same frames as a seek per time (what vio extraction did before)."""
    FPS, SECS, SZ = 24, 10, 64

    def _video(self, d):
        """A two-stream video (like an .insv: two lenses) whose frame n shows n as 8 black/white stripes (robust to compression)."""
        import shutil, subprocess
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not installed")
        frames = np.zeros((self.FPS * self.SECS, self.SZ, self.SZ, 3), np.uint8)
        for n in range(len(frames)):
            for bit in range(8):
                if (n >> bit) & 1:
                    frames[n, :, bit * 8:(bit + 1) * 8] = 255
        raw = os.path.join(d, "raw.rgb"); frames.tofile(raw)
        v = os.path.join(d, "two.mp4")
        inp = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.SZ}x{self.SZ}", "-r", str(self.FPS), "-i", raw]
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", *inp, *inp, "-map", "0:v", "-map", "1:v", "-c:v", "libx264", "-crf", "12", "-g", "48", "-pix_fmt", "yuv420p", v],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return v

    def _n(self, img):
        row = img[self.SZ // 2]
        return sum(1 << bit for bit in range(8) if row[bit * 8 + 4].mean() > 128)

    def test_frame_times_are_the_documented_ones(self):
        import vio_map
        t = vio_map.frame_times(10.0, 1.0)
        np.testing.assert_allclose(t, np.arange(0.8, 9.5, 1.0))
        self.assertEqual(len(vio_map.frame_times(0.5, 1.0)), 0)

    def test_one_pass_picks_the_frames_a_seek_per_time_picks(self):
        import queue, vio_map
        from stitch_points import grab_both
        with tempfile.TemporaryDirectory() as d:
            v = self._video(d)
            dur = self.SECS - 0.0
            times = vio_map.frame_times(dur, 1.0)
            for hw in (False, True):
                q = queue.Queue()
                got, err = vio_map.decode_lens(v, 1, times, (self.SZ, self.SZ), q, hw)
                self.assertEqual(got, len(times), err)
                picked = []
                while not q.empty():
                    k, img = q.get()
                    picked.append(self._n(img))
                for k, n in enumerate(picked):
                    self.assertLess(abs(n / self.FPS - times[k]), 1.0 / self.FPS + 1e-6, (hw, k, n, times[k]))      # within one frame of the time asked for
                    self.assertGreaterEqual(n / self.FPS, times[k] - 1e-6)                                      # the first frame at or after it
                if not hw:
                    want = []
                    for t in times[[0, 3, 6]]:
                        f, b = os.path.join(d, "f.bmp"), os.path.join(d, "b.bmp")
                        grab_both(v, float(t), f, b)
                        want.append(self._n(cv2.imread(b)))
                    for k, w in zip([0, 3, 6], want):
                        self.assertLessEqual(abs(picked[k] - w), 1, (k, picked[k], w))                          # the old way's frame, to within one frame

    def test_a_video_shorter_than_asked_delivers_what_it_has(self):
        import queue, vio_map
        with tempfile.TemporaryDirectory() as d:
            v = self._video(d)
            times = np.arange(7.8, 20.0, 1.0)                      # past the end: only 7.8, 8.8 and 9.8 exist
            q = queue.Queue()
            got, err = vio_map.decode_lens(v, 0, times, (self.SZ, self.SZ), q, False)
            self.assertEqual(got, 3)


class TestColmapDet(unittest.TestCase):
    """Same recording in, same map out (colmap_det.py): the pictures are numbered by name whatever order the threads finished in."""

    def _db(self, path, order):
        import sqlite3
        c = sqlite3.connect(path)
        c.executescript("""
            create table images (image_id integer primary key autoincrement not null, name text not null unique, camera_id integer not null);
            create table keypoints (image_id integer primary key not null, rows integer, cols integer, data blob);
            create table descriptors (image_id integer primary key not null, type integer, rows integer, cols integer, data blob);
            create table frames (frame_id integer primary key autoincrement not null, rig_id integer not null);
            create table frame_data (frame_id integer not null, data_id integer not null, sensor_id integer not null, sensor_type integer not null);
            create table matches (pair_id integer primary key not null, rows integer, cols integer, data blob);
            create table two_view_geometries (pair_id integer primary key not null, rows integer, cols integer, data blob, config integer);""")
        for k, name in enumerate(order, 1):                                  # finished in this order: ids are handed out in it
            c.execute("insert into images values (?, ?, 1)", (k, name)); c.execute("insert into keypoints values (?, 1, 2, ?)", (k, name.encode()))
            c.execute("insert into descriptors values (?, 0, 1, 2, ?)", (k, name.encode())); c.execute("insert into frames values (?, 1)", (k,))
            c.execute("insert into frame_data values (?, ?, 1, 0)", (k, k))
        c.commit(); c.close()

    def _read(self, path):
        import sqlite3
        c = sqlite3.connect(path)
        r = dict(images=c.execute("select image_id, name from images order by image_id").fetchall(),
                 kp=c.execute("select image_id, data from keypoints order by image_id").fetchall(),
                 ds=c.execute("select image_id, data from descriptors order by image_id").fetchall(),
                 fd=c.execute("select frame_id, data_id from frame_data order by frame_id").fetchall(),
                 fr=c.execute("select frame_id from frames order by frame_id").fetchall())
        c.close()
        return r

    def test_pictures_are_numbered_by_name_whatever_order_they_finished_in(self):
        import colmap_det
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.db"), os.path.join(d, "b.db")
            self._db(a, ["003.jpg", "001.jpg", "004.jpg", "002.jpg"]); self._db(b, ["002.jpg", "004.jpg", "003.jpg", "001.jpg"])
            self.assertTrue(colmap_det.canonical_ids(a)); self.assertTrue(colmap_det.canonical_ids(b))
            ra, rb = self._read(a), self._read(b)
            self.assertEqual(ra, rb)
            self.assertEqual(ra["images"], [(1, "001.jpg"), (2, "002.jpg"), (3, "003.jpg"), (4, "004.jpg")])
            self.assertEqual(ra["kp"][2], (3, b"003.jpg")); self.assertEqual(ra["ds"][0], (1, b"001.jpg"))      # the features went with their pictures
            self.assertEqual(ra["fd"], [(1, 1), (2, 2), (3, 3), (4, 4)]); self.assertEqual(ra["fr"], [(1,), (2,), (3,), (4,)])
            self.assertFalse(colmap_det.canonical_ids(a))                                                       # already in order: nothing to do

    def test_a_database_that_is_not_what_was_expected_is_left_alone(self):
        import colmap_det, sqlite3
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.db"); self._db(a, ["002.jpg", "001.jpg"])
            c = sqlite3.connect(a); c.execute("insert into matches values (1, 0, 2, x'')"); c.commit(); c.close()
            before = self._read(a)
            with self.assertRaises(RuntimeError):
                colmap_det.canonical_ids(a)                                                                     # matches exist: ids are in use
            self.assertEqual(self._read(a), before)

    def test_seeds_and_threads(self):
        import colmap_det
        keep = {k: os.environ.get(k) for k in ("SHOP_TOUR_COLMAP_DET", "SHOP_TOUR_COLMAP_THREADS")}
        try:
            os.environ.pop("SHOP_TOUR_COLMAP_THREADS", None)
            os.environ["SHOP_TOUR_COLMAP_DET"] = "1"
            self.assertTrue(colmap_det.deterministic()); self.assertEqual(colmap_det.threads(), 1)
            os.environ["SHOP_TOUR_COLMAP_THREADS"] = "4"
            self.assertEqual(colmap_det.threads(), 4)
            os.environ["SHOP_TOUR_COLMAP_DET"] = "0"; os.environ.pop("SHOP_TOUR_COLMAP_THREADS")
            self.assertFalse(colmap_det.deterministic()); self.assertEqual(colmap_det.threads(), -1)
            seeded = []
            class Fake:
                @staticmethod
                def set_random_seed(n):
                    seeded.append(n)
            colmap_det.start(Fake); self.assertEqual(seeded, [])                                                 # the old behaviour does not touch the generator
            os.environ["SHOP_TOUR_COLMAP_DET"] = "1"
            colmap_det.start(Fake); self.assertEqual(seeded, [colmap_det.SEED])
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_every_colmap_call_of_the_engine_goes_through_it(self):
        for f in ("map_stops.py", "vio_map.py"):
            src = open(os.path.join(HERE, f), encoding="utf-8").read()
            self.assertIn("colmap_det.start(pycolmap)", src); self.assertIn("colmap_det.mapper_options(pycolmap)", src)
            self.assertIn("colmap_det.extract(", src); self.assertIn("colmap_det.match_", src)
            self.assertNotIn("pycolmap.IncrementalPipelineOptions()", src); self.assertNotIn("pycolmap.extract_features(", src)
            self.assertNotIn("pycolmap.match_image_pairs(", src); self.assertNotIn("pycolmap.match_exhaustive(", src)

    def _matching_db(self, d, name):
        import sqlite3
        c = sqlite3.connect(os.path.join(d, name))
        c.execute("create table images(image_id integer primary key, name text)")
        c.execute("create table matches(pair_id integer primary key not null, rows integer not null, cols integer not null, data blob)")
        c.execute("create table two_view_geometries(pair_id integer primary key not null, rows integer not null, cols integer not null, data blob, config integer not null)")
        c.executemany("insert into images values (?, ?)", [(i + 1, f"f_{i:03d}.jpg") for i in range(40)])
        c.commit(); c.close()
        return os.path.join(d, name)

    def test_matching_is_dealt_to_fixed_processes_and_merged_by_pair(self):
        """Matching runs in a FIXED number of single-thread processes over a fixed split of the pair list, so what a pair gets never depends on
        the scheduler or the number of cores; the rows are merged into the main database in the order of the processes."""
        import colmap_det, sqlite3
        names = [f"f_{i:03d}.jpg" for i in range(40)]
        pairs = [(names[i], names[j]) for i in range(40) for j in range(i + 1, min(40, i + 4))]            # 111 pairs
        with tempfile.TemporaryDirectory() as d:
            lst = os.path.join(d, "pairs.txt")
            with open(lst, "w") as fh:
                for a, b in reversed(pairs):
                    fh.write(f"{b} {a}" + chr(10))                                                          # written the other way round, and one of them twice
                fh.write(f"{pairs[0][0]} {pairs[0][1]}" + chr(10))
            self.assertEqual(colmap_det.read_pairs(lst), [(b, a) for a, b in reversed(pairs)])               # file order and direction kept, each pair once
            db = self._matching_db(d, "main.db")
            started = []

            class FakeProc:
                returncode = 0
                def __init__(self, cmd, **kw):
                    cdb, cl = cmd[-2], cmd[-1]
                    todo = colmap_det.read_pairs(cl)
                    started.append((cmd[-3], len(todo)))
                    c = sqlite3.connect(cdb)
                    ids = {n: i for i, n in c.execute("select image_id, name from images").fetchall()}
                    c.executemany("insert into matches values (?, 1, 2, ?)", [(min(ids[a], ids[b]) * 100000 + max(ids[a], ids[b]), b"m") for a, b in todo])
                    c.executemany("insert into two_view_geometries values (?, 1, 2, ?, 2)", [(min(ids[a], ids[b]) * 100000 + max(ids[a], ids[b]), b"g") for a, b in todo])
                    c.commit(); c.close()
                def communicate(self): return (b"", None)
                def poll(self): return 0

            keep = {k: os.environ.get(k) for k in ("SHOP_TOUR_COLMAP_DET", "SHOP_TOUR_COLMAP_MATCH_PROCS", "SHOP_TOUR_COLMAP_MATCH")}
            real = colmap_det.subprocess.Popen
            try:
                os.environ.pop("SHOP_TOUR_COLMAP_DET", None); os.environ["SHOP_TOUR_COLMAP_MATCH_PROCS"] = "4"; os.environ["SHOP_TOUR_COLMAP_MATCH"] = "colmap"        # fast matching is the default now; this is the colmap path
                colmap_det.subprocess.Popen = FakeProc
                colmap_det.match_pairs(None, db, lst, d)
            finally:
                colmap_det.subprocess.Popen = real
                for k, v in keep.items():
                    os.environ.pop(k, None)
                    if v is not None:
                        os.environ[k] = v
            self.assertEqual(len(started), 4); self.assertEqual(sum(n for _, n in started), len(pairs)); self.assertLessEqual(max(n for _, n in started) - min(n for _, n in started), 1)
            c = sqlite3.connect(db)
            self.assertEqual(c.execute("select count(*) from matches").fetchone()[0], len(pairs)); self.assertEqual(c.execute("select count(*) from two_view_geometries").fetchone()[0], len(pairs))
            c.close()
            self.assertEqual([f for f in os.listdir(d) if f.startswith("match_chunk")], [])                    # the copies are cleaned up

    def test_few_pairs_or_the_old_way_are_matched_in_one_process(self):
        import colmap_det
        calls = []

        class Fake:
            class ImportedPairingOptions:
                def __init__(self, match_list_path): self.p = match_list_path
            class FeatureMatchingOptions:
                num_threads = 0
            class TwoViewGeometryOptions:
                class ransac: random_seed = 0
            @staticmethod
            def match_image_pairs(db, **kw): calls.append(("pairs", db))
            @staticmethod
            def match_exhaustive(db, **kw): calls.append(("all", db))

        keep = {k: os.environ.get(k) for k in ("SHOP_TOUR_COLMAP_DET", "SHOP_TOUR_COLMAP_MATCH_PROCS", "SHOP_TOUR_COLMAP_MATCH")}
        try:
            with tempfile.TemporaryDirectory() as d:
                os.environ["SHOP_TOUR_COLMAP_MATCH"] = "colmap"                                              # fast matching is the default now; this is the colmap path
                lst = os.path.join(d, "p.txt")
                with open(lst, "w") as fh:
                    fh.write("a b" + chr(10) + "a c" + chr(10))
                os.environ.pop("SHOP_TOUR_COLMAP_DET", None); os.environ.pop("SHOP_TOUR_COLMAP_MATCH_PROCS", None)
                colmap_det.match_pairs(Fake, "x.db", lst, d)
                self.assertEqual(calls, [("pairs", "x.db")])                                                  # two pairs: not worth eight pythons
                self.assertEqual(colmap_det.match_procs(), colmap_det.PROCS)
                os.environ["SHOP_TOUR_COLMAP_MATCH_PROCS"] = "1"
                big = os.path.join(d, "big.txt")
                with open(big, "w") as fh:
                    for i in range(200):
                        fh.write(f"a{i} b{i}" + chr(10))
                colmap_det.match_pairs(Fake, "y.db", big, d)
                self.assertEqual(calls[-1], ("pairs", "y.db"))
                os.environ["SHOP_TOUR_COLMAP_DET"] = "0"; os.environ["SHOP_TOUR_COLMAP_MATCH_PROCS"] = "8"
                colmap_det.match_exhaustive(Fake, "z.db", d)
                self.assertEqual(calls[-1], ("all", "z.db"))                                                  # the old way keeps COLMAP's own exhaustive matcher
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


class TestSpeedRound1(unittest.TestCase):
    """2026-10-21 (speed): the walk-wide picture map decodes on the card and maps fast, the accelerometer fit runs beside the stop map."""

    def _env(self, **kw):
        keep = {k: os.environ.get(k) for k in kw}
        self.addCleanup(lambda: [os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v) for k, v in keep.items()])
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_decode_modes_order_and_override(self):
        import vio_map
        self._env(SHOP_TOUR_VIO_DECODE=None)
        self.assertEqual(vio_map.decode_modes("nonexistent.insv")[-2:], ["auto", "sw"])           # no card path for a file that cannot be probed
        for want, order in (("cuda", ["cuda", "sw"]), ("auto", ["auto", "sw"]), ("sw", ["sw"])):
            self._env(SHOP_TOUR_VIO_DECODE=want)
            self.assertEqual(vio_map.decode_modes("x.insv"), order)

    def test_card_decode_command_keeps_frames_on_the_card_and_goes_back_to_the_videos_own_format(self):
        import queue, vio_map
        seen = []

        class FakeProc:
            def __init__(self, cmd, **kw):
                seen.append(cmd)
                self.stdout = type("S", (), {"readinto": lambda s, m: 0, "close": lambda s: None})()
                self.stderr = type("S", (), {"read": lambda s: b""})()

            def poll(self): return 0

            def wait(self): return 0

            def kill(self): pass
        real = vio_map.subprocess.Popen
        vio_map.subprocess.Popen = FakeProc
        try:
            times = np.arange(0.8, 5.0, 1.0)
            for mode in ("cuda", "auto", "sw", True, False):
                vio_map.decode_lens("v.insv", 0, times, (4, 4), queue.Queue(), mode)
        finally:
            vio_map.subprocess.Popen = real
        cuda, auto, sw = seen[0], seen[1], seen[2]
        self.assertIn("cuda", cuda); self.assertIn("-hwaccel_output_format", cuda)
        vf = cuda[cuda.index("-vf") + 1]
        self.assertTrue(vf.startswith("select=")); self.assertTrue(vf.endswith("hwdownload,format=nv12,format=yuvj420p"))   # select on the card, download only what is kept
        self.assertIn("-hwaccel", auto); self.assertNotIn("-hwaccel_output_format", auto)
        self.assertNotIn("-hwaccel", sw)
        self.assertEqual(seen[3], auto); self.assertEqual(seen[4], sw)                                                          # the old True/False still mean auto/software

    def test_fast_settings_follow_the_privacy_mode_and_the_switch(self):
        import colmap_det
        self._env(SHOP_TOUR_VIO_FAST=None, SHOP_TOUR_PRIVACY_INPUTS=None)
        self.assertTrue(colmap_det.vio_fast())                                                  # the privacy path reads nothing from the map by default
        self._env(SHOP_TOUR_VIO_FAST="0"); self.assertFalse(colmap_det.vio_fast())
        self._env(SHOP_TOUR_VIO_FAST=None, SHOP_TOUR_PRIVACY_INPUTS="map")
        self.assertFalse(colmap_det.vio_fast())                                                 # where the gate reads the map, the map must not move
        self._env(SHOP_TOUR_VIO_FAST="1"); self.assertTrue(colmap_det.vio_fast())

    def test_global_options_take_threads_and_track_cap_and_extraction_the_first_octave(self):
        import colmap_det
        from types import SimpleNamespace as NS
        self._env(SHOP_TOUR_COLMAP_DET="1", SHOP_TOUR_COLMAP_THREADS=None)

        def opts():
            m = NS(random_seed=0, num_threads=0, keep_max_num_tracks=2 ** 31 - 1, rotation_averaging=NS(random_seed=0), global_positioning=NS(random_seed=0),
                   retriangulation=NS(random_seed=0), bundle_adjustment=NS(ceres=NS(solver_options=NS(num_threads=0))))
            return NS(random_seed=0, num_threads=0, mapper=m)
        fake = NS(GlobalPipelineOptions=opts)
        o = colmap_det.global_options(fake)
        self.assertEqual((o.num_threads, o.mapper.num_threads, o.mapper.keep_max_num_tracks), (1, 1, 2 ** 31 - 1))        # unchanged by default
        o = colmap_det.global_options(fake, 8, 15000)
        self.assertEqual((o.num_threads, o.mapper.num_threads, o.mapper.bundle_adjustment.ceres.solver_options.num_threads, o.mapper.keep_max_num_tracks), (8, 8, 8, 15000))
        self.assertEqual((o.random_seed, o.mapper.random_seed), (colmap_det.SEED, colmap_det.SEED))                          # still seeded
        fe = NS(FeatureExtractionOptions=lambda: NS(sift=NS(max_num_features=0, first_octave=-1), num_threads=0))
        self.assertEqual(colmap_det.extraction_options(fe, 4096).sift.first_octave, -1)
        self.assertEqual(colmap_det.extraction_options(fe, 4096, 0).sift.first_octave, 0)

    def test_imu_fit_beside_the_map_starts_only_when_it_is_needed(self):
        import map_stops, subprocess
        with tempfile.TemporaryDirectory() as d:
            fr = os.path.join(d, "frames"); os.makedirs(os.path.join(fr, "vio"))
            self.assertIsNone(map_stops.start_imu_scale(None, fr))                               # no video: no fit
            self.assertIsNone(map_stops.start_imu_scale("v.insv", fr))                           # no vio/poses.npz yet
            vio = os.path.join(fr, "vio", "poses.npz"); open(vio, "wb").write(b"x")
            started = []
            real = subprocess.Popen
            subprocess.Popen = lambda cmd, **kw: started.append(cmd) or "proc"
            try:
                self.assertEqual(map_stops.start_imu_scale("v.insv", fr), "proc")                # stale or missing scale_imu.json: start
                self.assertEqual(started[0][-3:], ["v.insv", fr, os.path.join(fr, "scale_imu.json")])
                sj = os.path.join(fr, "scale_imu.json"); open(sj, "w").write("{}")
                os.utime(sj, (os.path.getmtime(vio) + 5, os.path.getmtime(vio) + 5))
                self.assertIsNone(map_stops.start_imu_scale("v.insv", fr))                       # a fresh one is already there
                os.utime(sj, (os.path.getmtime(vio) - 5, os.path.getmtime(vio) - 5))
                self._env(SHOP_TOUR_IMU_BESIDE="0")
                self.assertIsNone(map_stops.start_imu_scale("v.insv", fr))                       # switched off
            finally:
                subprocess.Popen = real

    def test_imu_orientation_and_residual_agree_with_the_einsum_form(self):
        import imu_scale as S
        from scipy.spatial.transform import Rotation as Rot
        rng = np.random.default_rng(3)
        ts = np.arange(0.5, 40.0, 1.0)
        Rcw = Rot.from_rotvec(rng.normal(0, 0.3, (len(ts), 3))).as_matrix()
        C = np.cumsum(rng.normal(0, 0.5, (len(ts), 3)), 0)
        tt = np.arange(0.0, 42.0, 1.0 / S.RATE)
        f = rng.normal(0, 1.0, (len(tt), 3)) + np.array([0, 0, 9.8]); gy = rng.normal(0, 0.2, (len(tt), 3))
        P = S.Problem(ts, C, Rcw, tt + 81.0, f, gy)
        x = np.r_[1.4, 0.1, -0.2, 0.3, 1.2, 0.4, 0.05, -0.02, 0.01]
        r = P.residuals(x, 81.0)
        # the same arithmetic written with einsum, as it was before the matmul form
        Rcs = Rot.from_rotvec(x[1:4]).as_matrix(); k, frac, Q, Qend = P.chain
        Rpic = P.Rwc_img @ Rcs
        M = np.einsum("kji,kjl->kil", Rpic[:-1] @ Qend, Rpic[1:])
        corr = Rot.from_rotvec(frac[:, None] * Rot.from_matrix(M).as_rotvec()[k]).as_matrix()
        Rws = np.einsum("nij,njk,nkl->nil", Rpic[k], Q, corr)
        np.testing.assert_allclose(P.sensor_orientation(Rcs), Rws, atol=1e-12)
        th, ph = x[4], x[5]; up = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
        ci = P._cat_idx
        aw = np.einsum("nij,nj->ni", Rws[ci], P.f[ci] - x[6:9]) - S.G * up
        want = (np.add.reduceat(aw * P._cat_w[:, None], P._starts, axis=0) - x[0] * P.pm).ravel()
        np.testing.assert_allclose(r, want, atol=1e-10)


class TestPrivacyMargin(unittest.TestCase):
    """privacy_margin.py: a stop that passed but would be held with its camera height measured a little differently is held too. It only ever ADDS
    holds, on stops the gate passed, and leaves what a person answered alone."""
    def _state(self):
        return {"001": dict(held=False, reasons=[], persons=[], faces=[]),
                "002": dict(held=True, reasons=["a face (looking +33 deg sideways, -13 deg up); it is blurred in any copy that is kept"], persons=[], faces=[{"az": 1.0, "el": 0.0, "deg": 10.0}]),
                "003": dict(held=True, reasons=["a person-sized shape (0.4 m2) is still near the camera"], persons=[], faces=[], kept=True, kept_on="2026-10-20"),
                "004": dict(held=True, reasons=["another person in the picture"], persons=[], faces=[], left_out=True, left_out_on="2026-10-20"),
                "005": dict(held=False, reasons=[], persons=[], faces=[])}

    def test_only_adds_holds_and_leaves_answers_alone(self):
        import copy, privacy_margin as pm
        S = self._state(); before = copy.deepcopy(S)
        hit = [(1.0, 2.4, []), (0.93, 2.23, ["a person-sized shape (0.31 m2) is still near the camera"], "best")]
        found = {sid: list(hit) for sid in S}                                                    # every stop "would be held at x0.93"
        added = pm.apply_margin(S, found)
        self.assertEqual(added, ["001", "005"])                                                  # only the two the gate had passed
        for sid in ("002", "003", "004"):
            self.assertEqual(S[sid], before[sid])                                                # held, kept and left-out stops are byte for byte as they were
        for sid in ("001", "005"):
            self.assertTrue(S[sid]["held"]); self.assertFalse(S[sid].get("kept")); self.assertFalse(S[sid].get("left_out"))
            self.assertEqual(len(S[sid]["reasons"]), 1)
            r = S[sid]["reasons"][0]
            self.assertTrue(r.startswith("held to be safe: with the camera height measured a little differently (7% lower)"), r)
            self.assertIn("a person-sized shape (0.31 m2) is still near the camera", r)
            self.assertEqual(S[sid]["margin"]["scale"], 0.93); self.assertEqual(S[sid]["persons"], []); self.assertEqual(S[sid]["faces"], [])

    def test_never_removes_a_hold(self):
        import copy, itertools, privacy_margin as pm
        base = self._state()
        hits_options = [[], [(1.0, 2.4, [])], [(0.93, 2.2, ["x"])], [(1.07, 2.6, ["y"], "other"), (0.93, 2.2, [])], [(1.0, 2.4, ["z"])]]
        for combo in itertools.product(range(len(hits_options)), repeat=len(base)):
            S = copy.deepcopy(base); held0 = {k for k, v in S.items() if v["held"]}
            reasons0 = {k: list(v["reasons"]) for k, v in S.items()}
            pm.apply_margin(S, {sid: hits_options[c] for sid, c in zip(sorted(S), combo)})
            self.assertTrue(held0 <= {k for k, v in S.items() if v["held"]})                   # every earlier hold is still a hold
            for k in S:
                self.assertEqual(S[k]["reasons"][:len(reasons0[k])], reasons0[k])             # and keeps its own reasons, first
                self.assertEqual(bool(S[k].get("kept")), bool(base[k].get("kept"))); self.assertEqual(bool(S[k].get("left_out")), bool(base[k].get("left_out")))

    def test_nothing_found_changes_nothing_and_unknown_stops_are_not_invented(self):
        import copy, privacy_margin as pm
        S = self._state(); before = copy.deepcopy(S)
        self.assertEqual(pm.apply_margin(S, {"001": [(0.93, 2.2, []), (1.07, 2.6, [])], "999": [(1.0, 2.0, ["a person-sized shape"])]}), [])
        self.assertEqual(S, before)

    def test_the_smallest_change_in_height_is_the_one_reported(self):
        import privacy_margin as pm
        S = self._state()
        pm.apply_margin(S, {"001": [(1.07, 2.6, ["b"]), (0.93, 2.2, ["a"]), (1.0, 2.4, ["c"], "other")]})
        self.assertEqual(S["001"]["margin"]["scale"], 1.0); self.assertEqual(S["001"]["margin"]["which"], "other")
        self.assertIn("more than one person-sized shape", S["001"]["reasons"][-1])
        S = self._state(); pm.apply_margin(S, {"001": [(1.07, 2.6, ["b"]), (0.93, 2.2, ["a"])]})
        self.assertEqual(S["001"]["margin"]["scale"], 0.93)                                      # a tie goes to the lower height (smaller metres per pixel is the unsafe side)
        self.assertIn("7% higher", pm.margin_reason(1.07, ["b"])); self.assertIn("7% lower", pm.margin_reason(0.93, ["b"]))

    def test_the_manifest_and_the_tour_read_the_added_hold_like_any_other(self):
        import manifest, privacy_margin as pm
        S = self._state(); pm.apply_margin(S, {"001": [(0.93, 2.2, ["a person-sized shape (0.3 m2) is still near the camera"])]})
        with tempfile.TemporaryDirectory() as d:
            _wj(os.path.join(d, "privacy.json"), dict(version=1, stops=S))
            b = manifest.privacy_block(d, [])
        self.assertEqual(b["status"], "held")
        ids = {h["id"]: h["reasons"] for h in b["held"]}
        self.assertIn("001", ids); self.assertIn("held to be safe", ids["001"][0]); self.assertEqual(b["kept"], ["003"]); self.assertEqual(b["left_out"], ["004"])

    def test_memo_runs_the_picture_models_once_per_picture(self):
        import privacy_margin as pm
        class Fake:
            n_disp = n_seg = 0
            def disparity(self, v):
                Fake.n_disp += 1; return np.full(v.shape[:2], 3.0, np.float32)
            def segment_many(self, v, pts, chunk=8):
                Fake.n_seg += 1; return [(0, np.ones((2, 2), bool), 0.9), (1, np.zeros((2, 2), bool), 0.8)]
            other = "passed through"
        m = pm.Memo(Fake()); v = np.zeros((4, 4, 3), np.uint8); w = np.ones((4, 4, 3), np.uint8)
        for _ in range(3):                                                                         # the same stop at three camera heights
            m.disparity(v); m.segment_many(v, [(1.0, 2.0)])
        self.assertEqual((Fake.n_disp, Fake.n_seg), (1, 1))
        d = m.disparity(v); d[:] = 0; self.assertEqual(float(m.disparity(v)[0, 0]), 3.0)         # a caller changing its copy cannot change the remembered one
        m.disparity(w); m.segment_many(w, [(1.0, 2.0)]); self.assertEqual((Fake.n_disp, Fake.n_seg), (2, 2))
        self.assertEqual(m.other, "passed through")
        m.only = ["one"]; self.assertEqual(m.segment_many(w, [(1.0, 2.0)]), ["one"]); self.assertEqual(Fake.n_seg, 2)    # judging one candidate on its own costs no model run
        m.only = None; self.assertEqual(len(m.segment_many(w, [(1.0, 2.0)])), 2)
        a, b, c = (0, np.eye(4, dtype=bool), 0.9), (1, np.eye(4, dtype=bool), 0.9), (2, np.ones((4, 4), bool), 0.9)
        self.assertEqual([x[0] for x in pm.distinct([a, b, c])], [0, 2])                           # a copy of a mask already taken is not judged twice

    def test_switch_and_scales(self):
        import privacy_margin as pm
        keep = {k: os.environ.get(k) for k in ("SHOP_TOUR_PRIVACY_MARGIN", "SHOP_TOUR_PRIVACY_MARGIN_SCALES")}
        try:
            os.environ.pop("SHOP_TOUR_PRIVACY_MARGIN", None); os.environ.pop("SHOP_TOUR_PRIVACY_MARGIN_SCALES", None)
            self.assertTrue(pm.enabled()); self.assertTrue(tour.privacy_margin_on()); self.assertEqual(pm.scales_from_env(), pm.SCALES)
            names = [os.path.basename(c[1]) for s in tour.steps("x.insv") if s[0] == "privacy" for c in s[3]()]
            self.assertEqual(names, ["privacy_margin.py", "privacy_gate.py", "privacy_margin.py"])           # snapshot of his answers, the gate, the margin (which restores them)
            for off in ("0", "off", "false", "No"):
                os.environ["SHOP_TOUR_PRIVACY_MARGIN"] = off
                self.assertFalse(pm.enabled()); self.assertFalse(tour.privacy_margin_on())
                names = [os.path.basename(c[1]) for s in tour.steps("x.insv") if s[0] == "privacy" for c in s[3]()]
                self.assertEqual(names, ["privacy_margin.py", "privacy_gate.py", "privacy_margin.py"])      # off still restores his answers and writes the marker (which says "off")
            os.environ.pop("SHOP_TOUR_PRIVACY_MARGIN")
            os.environ["SHOP_TOUR_PRIVACY_MARGIN_SCALES"] = "0.9, 1.1"; self.assertEqual(pm.scales_from_env(), (0.9, 1.1))
            os.environ["SHOP_TOUR_PRIVACY_MARGIN_SCALES"] = "banana"; self.assertEqual(pm.scales_from_env(), pm.SCALES)
            os.environ["SHOP_TOUR_PRIVACY_MARGIN_SCALES"] = "0.1,3"; self.assertEqual(pm.scales_from_env(), pm.SCALES)       # not a camera height
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_main_writes_the_added_hold_removes_its_picture_and_a_second_run_changes_nothing(self):
        import privacy_margin as pm
        with tempfile.TemporaryDirectory() as d:
            fr = os.path.join(d, "frames")
            for sub in ("safe", "clean", "capturer"):
                os.makedirs(os.path.join(fr, sub))
            ids = ["001", "002", "003", "004", "005"]
            _wj(os.path.join(fr, "points.json"), [dict(id=i, filename=i + ".jpg") for i in ids])
            _wj(os.path.join(fr, "up.json"), [[0, 1, 0]] * 5)
            for i in ids:
                cv2.imwrite(os.path.join(fr, "clean", i + ".jpg"), np.full((32, 64, 3), 90, np.uint8))
                if i in ("001", "005"):
                    cv2.imwrite(os.path.join(fr, "safe", i + ".jpg"), np.full((32, 64, 3), 90, np.uint8))
            _wj(os.path.join(fr, "privacy.json"), dict(version=1, stops=self._state()))
            calls = []
            def fake_sweep(frames, scales, ids=None, **kw):
                calls.append((sorted(ids), tuple(scales)))
                return {"001": [(1.0, 2.4, []), (0.93, 2.2, ["a person-sized shape (0.31 m2) is still near the camera"], "best")],
                        "005": [(1.0, 2.4, []), (0.93, 2.2, []), (1.07, 2.6, [])]}
            old_sweep, old_argv = pm.sweep, sys.argv
            pm.sweep = fake_sweep
            try:
                sys.argv = ["privacy_margin.py", "--frames", fr]
                pm.main()
                self.assertEqual(calls, [(["001", "005"], (1.0, 0.93, 1.07))])                     # only the two the gate had passed; 1.0 first, then the smallest change
                pv = json.load(open(os.path.join(fr, "privacy.json")))
                st = pv["stops"]
                self.assertTrue(st["001"]["held"]); self.assertFalse(st["005"]["held"])
                self.assertFalse(os.path.exists(os.path.join(fr, "safe", "001.jpg"))); self.assertTrue(os.path.exists(os.path.join(fr, "safe", "005.jpg")))
                self.assertEqual(st["003"], self._state()["003"]); self.assertEqual(st["004"], self._state()["004"]); self.assertEqual(st["002"], self._state()["002"])
                self.assertEqual(pv["margin"]["added"], ["001"])
                self.assertIn("001", open(os.path.join(fr, "review", "review.html"), encoding="utf-8").read())      # the review page lists the new hold
                first = open(os.path.join(fr, "privacy.json")).read()
                sys.argv = ["privacy_margin.py", "--frames", fr]
                pm.main()                                                                          # 001 is held now, so it is not tried again; nothing else was found
                self.assertEqual(calls[-1][0], ["005"])
                self.assertEqual(open(os.path.join(fr, "privacy.json")).read(), first)
                os.environ["SHOP_TOUR_PRIVACY_MARGIN"] = "0"
                n = len(calls); pm.main(); self.assertEqual(len(calls), n)                         # off: no work, nothing touched
                self.assertEqual(open(os.path.join(fr, "privacy.json")).read(), first)
            finally:
                pm.sweep, sys.argv = old_sweep, old_argv
                os.environ.pop("SHOP_TOUR_PRIVACY_MARGIN", None)

    def test_a_margin_that_cannot_finish_leaves_no_marker_and_the_step_is_not_done(self):
        """If the margin crashes, the gate's privacy.json is fresh but the step is NOT done: no marker, so tour.py runs the margin again."""
        import privacy_margin as pm
        with tempfile.TemporaryDirectory() as d:
            fr = os.path.join(d, "frames"); os.makedirs(os.path.join(fr, "clean"))
            _wj(os.path.join(fr, "points.json"), [dict(id="001", filename="001.jpg")])
            _wj(os.path.join(fr, "privacy.json"), dict(version=1, stops={"001": dict(held=False, reasons=[], persons=[], faces=[])}))
            cv2.imwrite(os.path.join(fr, "clean", "001.jpg"), np.full((32, 64, 3), 90, np.uint8))
            before = open(os.path.join(fr, "privacy.json")).read()
            def boom(*a, **k):
                raise RuntimeError("out of memory")
            old_sweep, old_argv = pm.sweep, sys.argv
            pm.sweep = boom; sys.argv = ["privacy_margin.py", "--frames", fr]
            try:
                with self.assertRaises(RuntimeError):
                    pm.main()
            finally:
                pm.sweep, sys.argv = old_sweep, old_argv
            self.assertEqual(open(os.path.join(fr, "privacy.json")).read(), before)                 # the gate's answer is not touched...
            self.assertFalse(os.path.exists(os.path.join(fr, "privacy_margin.json")))               # ...and nothing says the step finished

    def test_an_answer_after_the_margin_works_as_on_any_held_stop(self):
        """--leave-out on a stop the margin held works exactly as on any held stop (privacy_gate.py is not changed), and the margin never undoes it."""
        import subprocess, privacy_margin as pm
        with tempfile.TemporaryDirectory() as d:
            fr = os.path.join(d, "frames"); os.makedirs(os.path.join(fr, "clean"))
            _wj(os.path.join(fr, "points.json"), [dict(id=i, filename=i + ".jpg") for i in ("001", "002")]); _wj(os.path.join(fr, "up.json"), [[0, 1, 0]] * 2)
            cv2.imwrite(os.path.join(fr, "clean", "001.jpg"), np.full((32, 64, 3), 90, np.uint8)); cv2.imwrite(os.path.join(fr, "clean", "002.jpg"), np.full((32, 64, 3), 90, np.uint8))
            S = {"001": dict(held=False, reasons=[], persons=[], faces=[]), "002": dict(held=False, reasons=[], persons=[], faces=[])}
            pm.apply_margin(S, {"001": [(0.93, 2.2, ["x"])], "002": [(0.93, 2.2, ["y"])]})
            _wj(os.path.join(fr, "privacy.json"), dict(version=1, stops=S))
            r = subprocess.run([sys.executable, os.path.join(HERE, "privacy_gate.py"), "--frames", fr, "--leave-out", "002"], cwd=d, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            st = json.load(open(os.path.join(fr, "privacy.json")))["stops"]
            self.assertTrue(st["002"]["left_out"]); self.assertTrue(st["002"]["held"]); self.assertIn("margin", st["002"])
            pm.apply_margin(st, {"002": [(0.93, 2.2, ["y"])]})
            self.assertTrue(st["002"]["left_out"]); self.assertEqual(len(st["002"]["reasons"]), 1)


class TestPrivacyDone(unittest.TestCase):
    """The privacy step is done only when privacy_margin.py says so (frames/privacy_margin.json, its last act), and his answers (left out, kept) and the margin's
    holds survive a re-scan by privacy_gate.py. privacy_gate.py is not changed: tour.py keeps a snapshot of the earlier privacy.json, privacy_margin.py restores."""
    IDS = ["001", "002", "003", "004"]
    GATE_REASON = "a person-sized shape (0.4 m2) is still near the camera"

    def _walk(self, d, stops, snap=None, marker=False):
        """frames/ with points.json, a cleaned and a 'safe' picture for every stop that is not held, privacy.json (the gate's new result), an optional snapshot."""
        fr = os.path.join(d, "frames")
        for sub in ("clean", "safe", "capturer"):
            os.makedirs(os.path.join(fr, sub), exist_ok=True)
        ids = sorted(stops)
        _wj(os.path.join(fr, "points.json"), [dict(id=i, filename=i + ".jpg") for i in ids])
        _wj(os.path.join(fr, "up.json"), [[0, 1, 0]] * len(ids))
        for i in ids:
            cv2.imwrite(os.path.join(fr, "clean", i + ".jpg"), np.full((32, 64, 3), 90, np.uint8))
            if not stops[i].get("held") or stops[i].get("kept"):
                cv2.imwrite(os.path.join(fr, "safe", i + ".jpg"), np.full((32, 64, 3), 90, np.uint8))
        _wj(os.path.join(fr, "privacy.json"), dict(version=1, stops=stops))
        if snap is not None:
            _wj(os.path.join(fr, "privacy_prev.json"), dict(version=1, stops=snap))
        if marker:
            import privacy_answers as PA, privacy_margin as pm
            PA.write_marker(fr, "on", pm.SCRIPT_VERSION)
        return fr

    def _clear(self, **extra):
        return dict(held=False, reasons=[], persons=[], faces=[], **extra)

    def _margin(self, fr, found=None, sweep=None):
        """privacy_margin.main() on fr with the heavy sweep replaced."""
        import privacy_margin as pm
        old_sweep, old_argv = pm.sweep, sys.argv
        pm.sweep = sweep or (lambda frames, scales, ids=None, **kw: {k: v for k, v in (found or {}).items() if ids is None or k in ids})
        sys.argv = ["privacy_margin.py", "--frames", fr]
        try:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                pm.main()
        finally:
            pm.sweep, sys.argv = old_sweep, old_argv

    def _st(self, fr):
        return json.load(open(os.path.join(fr, "privacy.json")))["stops"]

    # ---- 1. a margin killed hard is run again -------------------------------------------------------------------------

    def test_a_margin_cut_off_before_its_marker_is_done_again_and_only_the_margin(self):
        import privacy_answers as PA, privacy_margin as pm
        with tempfile.TemporaryDirectory() as d:
            fr = self._walk(d, {i: self._clear() for i in self.IDS})              # the gate has just written this; the margin has not finished
            old = (tour.WORK, tour.F); tour.WORK, tour.F = d, fr
            try:
                ins = ["frames/clean/*.jpg", "privacy_gate.py", "capturer.py"]
                t = time.time()
                os.utime(os.path.join(fr, "privacy.json"), (t + 1000, t + 1000))  # newer than every input: the gate's result is current
                self.assertIn("privacy_margin", tour.incomplete("privacy"))        # no marker: not done, however fresh privacy.json looks
                self.assertTrue(tour.privacy_margin_only(False, None, ins))        # ...and only the margin is owed
                self.assertFalse(tour.privacy_margin_only(True, None, ins)); self.assertFalse(tour.privacy_margin_only(False, "privacy", ins))     # --from / --only mean the gate too
                tour._PRIVACY_MARGIN_ONLY[0] = True
                self.assertEqual([os.path.basename(c[1]) for c in tour.privacy_cmds()], ["privacy_margin.py"])                                  # never the gate again
                tour._PRIVACY_MARGIN_ONLY[0] = False
                def killed(*a, **k):
                    raise KeyboardInterrupt("killed in the margin")
                with self.assertRaises(KeyboardInterrupt):
                    self._margin(fr, sweep=killed)
                self.assertFalse(os.path.exists(os.path.join(fr, PA.MARKER)))
                self.assertIsNotNone(tour.incomplete("privacy"))                   # still not done
                self._margin(fr, found={"002": [(0.93, 2.2, ["a person-sized shape"], "best")]})
                self.assertIsNone(tour.incomplete("privacy"))                      # marker written: done
                self.assertTrue(self._st(fr)["002"]["held"])
                self.assertFalse(tour.privacy_margin_only(False, None, ins))
            finally:
                tour.WORK, tour.F = old

    def test_a_kill_between_the_verdict_and_the_marker_is_redone_to_the_same_answer(self):
        import privacy_answers as PA
        with tempfile.TemporaryDirectory() as d:
            snap = {"001": self._clear(), "002": dict(held=True, reasons=["a face"], persons=[], faces=[], left_out=True, left_out_on="2026-10-18"), "003": self._clear(), "004": self._clear()}
            fr = self._walk(d, {i: self._clear() for i in self.IDS}, snap=snap)
            real = PA.write_marker
            def die(*a, **k):
                raise KeyboardInterrupt("killed before the marker")
            PA.write_marker = die
            try:
                with self.assertRaises(KeyboardInterrupt):
                    self._margin(fr, found={"003": [(1.0, 2.0, ["a person-sized shape"], "best")]})
            finally:
                PA.write_marker = real
            first = open(os.path.join(fr, "privacy.json")).read()                  # the restored verdict was written...
            self.assertTrue(self._st(fr)["002"]["left_out"]); self.assertTrue(self._st(fr)["003"]["held"])
            self.assertTrue(os.path.exists(os.path.join(fr, PA.PREV)))             # ...but the snapshot is still there and no marker: not done
            self.assertFalse(os.path.exists(os.path.join(fr, PA.MARKER)))
            self._margin(fr, found={"003": [(1.0, 2.0, ["a person-sized shape"], "best")]})
            self.assertEqual(open(os.path.join(fr, "privacy.json")).read(), first)  # redone: the same answer, nothing doubled
            self.assertFalse(os.path.exists(os.path.join(fr, PA.PREV)))
            self.assertIsNone(PA.marker_problem(fr, "on", __import__("privacy_margin").SCRIPT_VERSION))

    def test_run_steps_runs_just_the_margin_after_a_kill_and_then_calls_the_step_done(self):
        """tour.run_steps with the real privacy wiring and stand-ins for the gate and the sweep: a hard kill (os._exit) in the margin, then a carry-on."""
        import contextlib, io, subprocess
        with tempfile.TemporaryDirectory() as d:
            prev = {"001": self._clear(), "002": dict(held=True, reasons=["a face"], persons=[], faces=[], left_out=True, left_out_on="2026-10-18"), "003": self._clear(), "004": self._clear()}
            fr = self._walk(d, prev, marker=True)                                  # a finished walk: privacy.json with his answer, and the marker for it
            t = time.time()
            for n in ("privacy.json", "privacy_margin.json"):
                os.utime(os.path.join(fr, n), (t - 500, t - 500))                 # older than the pictures: the gate is owed
            stub = os.path.join(d, "stub.py")
            with open(stub, "w") as fh:
                fh.write("import json, os, sys\n"
                         f"HERE = {HERE!r}; FR = {fr!r}\n"
                         "sys.path.insert(0, HERE)\n"
                         "which = sys.argv[1]\n"
                         "open(os.path.join(os.path.dirname(FR), 'trace.txt'), 'a').write(which + ' ' + ' '.join(sys.argv[2:]) + '\\n')\n"
                         "if which == 'privacy_gate.py':\n"
                         "    pts = json.load(open(os.path.join(FR, 'points.json')))\n"
                         "    json.dump(dict(version=1, stops={p['id']: dict(held=False, reasons=[], persons=[], faces=[]) for p in pts}), open(os.path.join(FR, 'privacy.json'), 'w'))\n"
                         "else:\n"
                         "    import privacy_margin as pm\n"
                         "    def sweep(frames, scales, ids=None, **kw):\n"
                         "        if os.environ.get('KILL_MARGIN') == '1':\n"
                         "            os._exit(9)\n"
                         "        return {}\n"
                         "    pm.sweep = sweep\n"
                         "    sys.argv = ['privacy_margin.py', '--frames', FR] + sys.argv[2:]\n"
                         "    pm.main()\n")
            ins = ["frames/clean/*.jpg", "privacy_gate.py", "capturer.py"]
            outs = ["frames/privacy.json", "frames/privacy_margin.json"]
            S = [("privacy", ins, outs, lambda: tour.privacy_cmds())]
            old = (tour.WORK, tour.F, tour.py); tour.WORK, tour.F = d, fr
            tour.py = lambda script, *a: [sys.executable, stub, script, *a]
            def go(kill=False):
                buf = io.StringIO(); err = None
                os.environ["KILL_MARGIN"] = "1" if kill else "0"
                try:
                    with contextlib.redirect_stdout(buf):
                        tour.run_steps(S, "pinhole", "x.insv", "x.insv")
                except SystemExit as e:
                    err = str(e.code)
                return buf.getvalue(), err
            trace = os.path.join(d, "trace.txt")
            try:
                out, err = go(kill=True)                                          # run 1: snapshot, gate, then the margin is killed hard
                self.assertIsNotNone(err); self.assertIn("[privacy] failed", err)
                self.assertEqual([l.split()[0] for l in open(trace).read().splitlines()], ["privacy_margin.py", "privacy_gate.py", "privacy_margin.py"])
                self.assertFalse(self._st(fr)["002"].get("left_out"))             # privacy.json is the gate's fresh result: his answer is not in it any more...
                self.assertTrue(os.path.exists(os.path.join(fr, "privacy_prev.json")))     # ...but it is in the snapshot
                open(trace, "w").close()
                out, err = go()                                                    # run 2 (carry-on): just the margin, on top of the gate's result
                self.assertIsNone(err, out)
                self.assertIn("only the margin is owed", out)
                self.assertEqual([l.split()[0] for l in open(trace).read().splitlines()], ["privacy_margin.py"])      # the gate did not run again
                st = self._st(fr)
                self.assertTrue(st["002"]["held"] and st["002"]["left_out"])      # his answer came back
                self.assertFalse(os.path.exists(os.path.join(fr, "privacy_prev.json")))
                out, err = go()                                                    # run 3: done, nothing runs
                self.assertIn("[privacy] up to date", out)
            finally:
                tour.WORK, tour.F, tour.py = old
                os.environ.pop("KILL_MARGIN", None)

    def test_an_answer_does_not_make_the_step_look_unfinished_but_a_new_verdict_does(self):
        import privacy_answers as PA, privacy_margin as pm
        with tempfile.TemporaryDirectory() as d:
            stops = {"001": self._clear(), "002": dict(held=True, reasons=[self.GATE_REASON], persons=[], faces=[]), "003": self._clear(), "004": self._clear()}
            fr = self._walk(d, stops, marker=True)
            self.assertIsNone(PA.pending(fr, "on", pm.SCRIPT_VERSION))
            st = self._st(fr); st["002"]["kept"] = True; st["002"]["kept_on"] = "2026-10-20"      # privacy_gate.py --keep 002
            _wj(os.path.join(fr, "privacy.json"), dict(version=1, stops=st))
            self.assertIsNone(PA.pending(fr, "on", pm.SCRIPT_VERSION))             # he answered: the margin is not owed again
            st["003"]["left_out"] = True; _wj(os.path.join(fr, "privacy.json"), dict(version=1, stops=st))
            self.assertIsNone(PA.pending(fr, "on", pm.SCRIPT_VERSION))
            st["001"]["held"] = True; st["001"]["reasons"] = ["a face"]; _wj(os.path.join(fr, "privacy.json"), dict(version=1, stops=st))      # a new verdict
            self.assertIn("not the one", PA.pending(fr, "on", pm.SCRIPT_VERSION))
            self.assertIn("older version", PA.marker_problem(fr, "on", pm.SCRIPT_VERSION + 1) or "")

    def test_the_marker_knows_whether_the_margin_was_on_and_which_version_made_it(self):
        import privacy_answers as PA, privacy_margin as pm
        with tempfile.TemporaryDirectory() as d:
            fr = self._walk(d, {i: self._clear() for i in self.IDS})
            self.assertIn("no privacy_margin.json", PA.pending(fr, "on", pm.SCRIPT_VERSION))
            os.environ["SHOP_TOUR_PRIVACY_MARGIN"] = "0"
            try:
                self._margin(fr)                                                   # off: no sweep, but the marker is written and says "off"
                self.assertEqual(json.load(open(os.path.join(fr, PA.MARKER)))["margin"], "off")
                self.assertIsNone(PA.pending(fr, "off", pm.SCRIPT_VERSION))
                self.assertIn("off when this was made", PA.pending(fr, "on", pm.SCRIPT_VERSION))     # switched on later: the margin is owed
            finally:
                os.environ.pop("SHOP_TOUR_PRIVACY_MARGIN", None)
            self._margin(fr)                                                       # on, nothing to hold: still writes the marker, privacy.json untouched
            self.assertIsNone(PA.pending(fr, "on", pm.SCRIPT_VERSION))

    def test_the_snapshot_is_kept_while_the_step_is_unfinished(self):
        import privacy_answers as PA, privacy_margin as pm
        with tempfile.TemporaryDirectory() as d:
            fr = self._walk(d, {"001": self._clear(), "002": dict(held=True, reasons=["a face"], persons=[], faces=[], left_out=True)}, marker=True)
            self.assertEqual(PA.snapshot(fr, "on", pm.SCRIPT_VERSION), "snapshot made")           # a finished step: its privacy.json is the one to protect
            self.assertTrue(PA.load_snapshot(fr)["002"]["left_out"])
            _wj(os.path.join(fr, "privacy.json"), dict(version=1, stops={"001": self._clear(), "002": self._clear()}))     # the gate wrote its new result; the margin was killed
            self.assertIn("kept the snapshot", PA.snapshot(fr, "on", pm.SCRIPT_VERSION))          # a second go must not replace his answers with that
            self.assertTrue(PA.load_snapshot(fr)["002"]["left_out"])
            with tempfile.TemporaryDirectory() as e:
                self.assertEqual(PA.snapshot(os.path.join(e, "nothing"), "on", pm.SCRIPT_VERSION), "no previous privacy.json")

    # ---- 2. his answers and the margin's holds survive a re-scan ------------------------------------------------------

    def test_a_left_out_stop_and_a_margin_hold_survive_a_rescan(self):
        import manifest, privacy_answers as PA
        margin_reason = "held to be safe: with the camera height measured a little differently (7% lower) this stop would be held: a person-sized shape"
        prev = {"001": self._clear(),
                "002": dict(held=True, reasons=["a face (looking +33 deg sideways, -13 deg up)"], persons=[], faces=[], left_out=True, left_out_on="2026-10-18"),
                "003": dict(held=True, reasons=[margin_reason], persons=[], faces=[], margin=dict(scale=0.93, cam_h=2.2, tried=[1.0, 0.93, 1.07], problems=["x"], which="best")),
                "004": dict(held=True, reasons=[margin_reason], persons=[], faces=[], margin=dict(scale=0.93, cam_h=2.2, tried=[1.0], problems=["x"], which="best"), left_out=True, left_out_on="2026-10-19")}
        new = {i: self._clear() for i in self.IDS}                                # the gate scanned again: every stop looks clear
        rep = PA.restore(prev, new)
        self.assertTrue(new["002"]["held"] and new["002"]["left_out"] and new["002"]["left_out_on"] == "2026-10-18")
        self.assertEqual(new["002"]["reasons"], prev["002"]["reasons"])           # what he saw is what the review shows
        self.assertTrue(new["003"]["held"] and not new["003"].get("left_out") and not new["003"].get("kept") and new["003"]["margin"]["scale"] == 0.93)
        self.assertEqual(new["003"]["reasons"], [margin_reason])
        self.assertTrue(new["004"]["left_out"])                                   # answered: the answer wins over the margin hold
        self.assertFalse(new["001"]["held"])
        self.assertEqual((rep["left_out"], rep["margin"], rep["kept"], rep["reheld"]), (["002", "004"], ["003"], [], []))
        again = json.loads(json.dumps(new)); PA.restore(prev, again)
        self.assertEqual(again, new)                                              # restoring twice changes nothing
        with tempfile.TemporaryDirectory() as d:                                  # what manifest.py and build_tour.py read
            fr = os.path.join(d, "frames"); os.makedirs(fr)
            _wj(os.path.join(fr, "privacy.json"), dict(version=1, stops=new))
            b = manifest.privacy_block(fr, ["001"])
            self.assertEqual(b["left_out"], ["002", "004"]); self.assertEqual([h["id"] for h in b["held"]], ["003"]); self.assertEqual(b["status"], "held")
            self.assertTrue(all(new[i]["held"] and not new[i].get("kept") for i in ("002", "003", "004")))      # build_tour.py's rule: held and not kept = not in the tour

    def test_a_kept_stop_stays_kept_only_for_what_he_saw(self):
        import privacy_answers as PA
        A, B, C = "a face (looking +33 deg sideways, -13 deg up)", "another person in the picture (looking +10 deg sideways)", "a person-sized shape (0.4 m2) is still near the camera"
        prev = {"001": dict(held=True, reasons=[A, B], persons=[], faces=[], kept=True, kept_on="2026-10-18")}
        same = {"001": dict(held=True, reasons=[A], persons=[], faces=[], kept="2026-10-18", kept_on="2026-10-18")}                   # the gate kept it itself
        PA.restore(prev, same)
        self.assertTrue(same["001"]["kept"] and same["001"]["held"] and same["001"]["kept_on"] == "2026-10-18")
        more = {"001": dict(held=True, reasons=[A, C], persons=[], faces=[], kept="2026-10-18", kept_on="2026-10-18")}                # the gate kept it, but C is new
        rep = PA.restore(prev, more)
        self.assertFalse(more["001"].get("kept")); self.assertNotIn("kept_on", more["001"]); self.assertTrue(more["001"]["held"])
        self.assertEqual(rep["reheld"], ["001"])
        self.assertIn(C, more["001"]["reasons"][-1]); self.assertTrue(more["001"]["reasons"][-1].startswith(PA.AGAIN_PREFIX)); self.assertIn(A, more["001"]["reasons"])
        once = json.loads(json.dumps(more)); PA.restore(prev, more)
        self.assertEqual(more, once)                                              # twice = once: one note, not two
        gone = {"001": dict(held=False, reasons=[], persons=[], faces=[])}        # the check finds nothing now: clear on its own
        PA.restore(prev, gone)
        self.assertFalse(gone["001"]["held"]); self.assertNotIn("kept", gone["001"])
        # he kept it for the margin's reason; the re-scan finds nothing and the margin finds the same thing again: still kept
        R = "held to be safe: with the camera height measured a little differently (7% lower) this stop would be held: x"
        prev2 = {"001": dict(held=True, reasons=[R], persons=[], faces=[], kept=True, kept_on="2026-10-19", margin=dict(scale=0.93))}
        again = {"001": dict(held=True, reasons=[R], persons=[], faces=[], margin=dict(scale=0.93))}
        rep = PA.restore(prev2, again)
        self.assertTrue(again["001"]["kept"]); self.assertEqual(rep["kept"], ["001"])
        other = {"001": dict(held=True, reasons=[R.replace("7% lower", "7% higher")], persons=[], faces=[], margin=dict(scale=1.07))}      # a different thing this time
        self.assertEqual(PA.restore(prev2, other)["reheld"], ["001"]); self.assertFalse(other["001"].get("kept"))

    def test_margin_main_puts_his_answers_back_and_keeps_the_pictures_in_step(self):
        import manifest, privacy_answers as PA, privacy_margin as pm
        R = pm.margin_reason(0.93, ["a person-sized shape"], "best")
        with tempfile.TemporaryDirectory() as d:
            prev = {"001": dict(held=True, reasons=["a face"], persons=[], faces=[], left_out=True, left_out_on="2026-10-18"),
                    "002": dict(held=True, reasons=[R], persons=[], faces=[], kept=True, kept_on="2026-10-19", margin=dict(scale=0.93)),
                    "003": dict(held=True, reasons=[R], persons=[], faces=[], margin=dict(scale=0.93, cam_h=2.2, tried=[1.0, 0.93], problems=["x"], which="best")),
                    "004": self._clear()}
            new = {i: self._clear() for i in self.IDS}                             # the gate's new result: all clear, all with a safe picture
            fr = self._walk(d, new, snap=prev)
            self._margin(fr, found={"002": [(0.93, 2.2, ["a person-sized shape"], "best")]})       # the margin finds 002 again, with the very reason he saw
            st = self._st(fr)
            self.assertTrue(st["001"]["held"] and st["001"]["left_out"])
            self.assertTrue(st["002"]["held"] and st["002"]["kept"] and st["002"]["kept_on"] == "2026-10-19")       # kept for what he saw
            self.assertTrue(st["003"]["held"] and "left_out" not in st["003"] and "kept" not in st["003"])         # the margin hold stays, unanswered
            self.assertFalse(st["004"]["held"])
            safe = lambda i: os.path.exists(os.path.join(fr, "safe", i + ".jpg"))
            self.assertEqual([safe(i) for i in self.IDS], [False, True, False, True])        # left out and held: no picture; kept: its picture is back; clear: untouched
            b = manifest.privacy_block(fr, ["002", "004"])
            self.assertEqual((b["kept"], b["left_out"], [h["id"] for h in b["held"]]), (["002"], ["001"], ["003"]))
            self.assertFalse(os.path.exists(os.path.join(fr, PA.PREV)))
            self.assertIsNone(PA.pending(fr, "on", pm.SCRIPT_VERSION))
            self.assertIn("003", open(os.path.join(fr, "review", "review.html"), encoding="utf-8").read())
            # the same walk, but 002 is now held for something he did not see: he must look again
            new2 = {i: self._clear() for i in self.IDS}
            os.makedirs(os.path.join(d, "two")); fr2 = self._walk(os.path.join(d, "two"), new2, snap=prev)
            self._margin(fr2, found={"002": [(1.07, 2.6, ["a different shape"], "best")]})
            st2 = self._st(fr2)
            self.assertTrue(st2["002"]["held"]); self.assertFalse(st2["002"].get("kept")); self.assertFalse(os.path.exists(os.path.join(fr2, "safe", "002.jpg")))
            self.assertTrue(any(r.startswith(PA.AGAIN_PREFIX) for r in st2["002"]["reasons"]))

    def test_with_no_snapshot_nothing_is_restored_and_only_the_margin_may_add(self):
        """No snapshot (a walk finished before this check existed): nothing is restored and nothing is changed except the margin's own holds."""
        with tempfile.TemporaryDirectory() as d:
            stops = {"001": self._clear(), "002": dict(held=True, reasons=["a face"], persons=[], faces=[], left_out=True), "003": self._clear(), "004": self._clear()}
            fr = self._walk(d, stops)
            before = self._st(fr)
            self._margin(fr)
            self.assertEqual(self._st(fr), before)
            self.assertTrue(os.path.exists(os.path.join(fr, "privacy_margin.json")))


class TestColmapIdsAreNotLeftBehind(unittest.TestCase):
    def test_a_table_that_holds_picture_or_frame_ids_blocks_the_renumbering(self):
        import colmap_det, sqlite3
        helper = TestColmapDet()
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.db"); helper._db(a, ["002.jpg", "001.jpg"])
            c = sqlite3.connect(a)
            c.execute("create table pose_priors (image_id integer primary key not null, position blob, coordinate_system integer)")
            c.commit(); c.close()
            self.assertTrue(colmap_det.canonical_ids(a))                           # an EMPTY pose_priors points at nothing: the renumbering goes ahead
            b = os.path.join(d, "b.db"); helper._db(b, ["002.jpg", "001.jpg"])
            c = sqlite3.connect(b)
            c.execute("create table pose_priors (image_id integer primary key not null, position blob, coordinate_system integer)")
            c.execute("insert into pose_priors values (1, x'00', 1)"); c.commit(); c.close()
            before = helper._read(b)
            with self.assertRaises(RuntimeError) as cm:
                colmap_det.canonical_ids(b)                                        # rows would be left pointing at the old numbers
            self.assertIn("pose_priors", str(cm.exception)); self.assertIn("image_id", str(cm.exception))
            self.assertEqual(helper._read(b), before)                              # and the database is untouched
            e = os.path.join(d, "e.db"); helper._db(e, ["002.jpg", "001.jpg"])
            c = sqlite3.connect(e)
            c.execute("create table rig_things (rig_id integer, frame_id integer, note text)"); c.execute("insert into rig_things values (1, 2, 'x')")   # another table, found from the schema
            c.execute("create table unrelated (k integer, v text)"); c.execute("insert into unrelated values (1, 'x')")
            c.commit(); c.close()
            with self.assertRaises(RuntimeError) as cm:
                colmap_det.canonical_ids(e)
            self.assertIn("rig_things", str(cm.exception)); self.assertIn("frame_id", str(cm.exception))
            f = os.path.join(d, "f.db"); helper._db(f, ["001.jpg", "002.jpg"])      # already in order: nothing to do, nothing to complain about
            c = sqlite3.connect(f); c.execute("create table pose_priors (image_id integer primary key not null)"); c.execute("insert into pose_priors values (1)"); c.commit(); c.close()
            self.assertFalse(colmap_det.canonical_ids(f))


class TestPrivacyInputs(unittest.TestCase):
    """The privacy path reads no vio/map output: one stick height and the neighbours from hops.json (privacy_inputs.py)."""
    PAIRS = {"pairs": [
        {"i": 3, "j": 4, "ok": True, "consistent": True, "az_ij": 158.7, "az_ji": -14.2, "yaw": -6.9, "dist_m": 1.85},
        {"i": 2, "j": 3, "ok": True, "consistent": True, "az_ij": 139.5, "az_ji": -15.0, "yaw": -25.6, "dist_m": 4.0},
        {"i": 3, "j": 6, "ok": True, "consistent": True, "az_ij": 30.2, "az_ji": 1.0, "yaw": -84.8, "dist_m": None},       # no distance: unusable
        {"i": 3, "j": 7, "ok": False, "consistent": True, "az_ij": 14.5, "az_ji": 1.0, "yaw": -144.4, "dist_m": 2.0},     # not a usable pair
        {"i": 3, "j": 8, "ok": True, "consistent": False, "az_ij": 14.5, "az_ji": 1.0, "yaw": -97.1, "dist_m": 2.0},     # disagrees with the chain
        {"i": 3, "j": 5, "ok": True, "consistent": True, "az_ij": 9.2, "az_ji": 2.0, "yaw": -175.5, "dist_m": 0.5}]}

    def test_stick_height_is_one_value_for_every_stop_and_the_environment_only_moves_it_inside_a_sane_range(self):
        keep = os.environ.pop("SHOP_TOUR_PRIVACY_CAM_H", None)
        try:
            self.assertEqual(pin.stick_height(), pin.PRIVACY_CAM_H)
            os.environ["SHOP_TOUR_PRIVACY_CAM_H"] = "2.2"; self.assertEqual(pin.stick_height(), 2.2)
            for bad in ("9", "0.2", "x"):
                os.environ["SHOP_TOUR_PRIVACY_CAM_H"] = bad; self.assertEqual(pin.stick_height(), pin.PRIVACY_CAM_H)
        finally:
            os.environ.pop("SHOP_TOUR_PRIVACY_CAM_H", None)
            if keep is not None:
                os.environ["SHOP_TOUR_PRIVACY_CAM_H"] = keep

    def test_hop_links_use_the_poses_conventions_both_ways_and_rescale_the_distance(self):
        L = pin.hop_links(self.PAIRS, 9, 2.4)
        k = 2.4 / pin.HOP_CAM_H
        self.assertEqual(L[3][4]["az"], 158.7); self.assertEqual(L[3][4]["turn"], 6.9); self.assertAlmostEqual(L[3][4]["d"], 1.85 * k, 2)
        self.assertEqual(L[4][3]["az"], -14.2); self.assertEqual(L[4][3]["turn"], -6.9)
        self.assertEqual(L[3][2]["az"], -15.0); self.assertEqual(L[3][2]["turn"], -25.6)          # a pair measured 2 -> 3, seen from 3
        self.assertIsNone(L[3][6]["d"])                                                           # no distance on the floor: kept, offered at a ladder of distances
        self.assertNotIn(7, L[3]); self.assertNotIn(8, L[3])                                      # not ok / not consistent with the chain: unusable

    def test_neighbours_previous_and_next_first_then_the_nearest_within_reach(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump(self.PAIRS, open(os.path.join(d, "hops.json"), "w"))
            keep = os.environ.get("SHOP_TOUR_PRIVACY_INPUTS")
            os.environ["SHOP_TOUR_PRIVACY_INPUTS"] = "own"
            try:
                I = pin.Inputs(d, 9)
                self.assertEqual(I.mode, "own"); self.assertEqual(I.cam_h(0), pin.PRIVACY_CAM_H); self.assertEqual(I.cam_h(8), pin.PRIVACY_CAM_H)
                nb = I.neighbours(3)
                self.assertEqual([n["j"] for n in nb], [2, 4, 5, 6])
                self.assertEqual(len(nb[0]["ds"]), 3); self.assertEqual(nb[3]["ds"], [x for x in pin.LADDER])        # a known distance is tried at 80/100/125%, an unknown one at the ladder
                self.assertEqual(I.neighbours(0), [])                                           # a stop with no usable pair has no neighbour: its fill is soft and the check says so
            finally:
                os.environ.pop("SHOP_TOUR_PRIVACY_INPUTS", None)
                if keep is not None:
                    os.environ["SHOP_TOUR_PRIVACY_INPUTS"] = keep

    def test_the_map_way_is_opt_in(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"nodes": [{"cam_h": 2.31, "links": [{"to": 1, "az": 5, "d": 1.5, "turn": 2}]}, {"cam_h": None, "links": []}]}, open(os.path.join(d, "poses.json"), "w"))
            keep = os.environ.pop("SHOP_TOUR_PRIVACY_INPUTS", None)
            try:
                self.assertEqual(pin.mode(), "own")                                                         # the default reads nothing from the map
                os.environ["SHOP_TOUR_PRIVACY_INPUTS"] = "map"
                I = pin.Inputs(d, 2)
                self.assertEqual((I.mode, I.cam_h(0), I.cam_h(1)), ("map", 2.31, 2.0)); self.assertEqual(I.neighbours(0), [dict(j=1, az=5, d=1.5, turn=2)])
            finally:
                os.environ.pop("SHOP_TOUR_PRIVACY_INPUTS", None)
                if keep is not None:
                    os.environ["SHOP_TOUR_PRIVACY_INPUTS"] = keep

    def test_the_speed_paths_are_on_by_default_only_where_the_privacy_path_reads_nothing_from_the_map(self):
        import colmap_det
        keep = {k: os.environ.pop(k, None) for k in ("SHOP_TOUR_PRIVACY_INPUTS", "SHOP_TOUR_COLMAP_MATCH", "SHOP_TOUR_MAPPER")}
        try:
            self.assertTrue(colmap_det.fast_match()); self.assertTrue(colmap_det.use_global())                  # own is the default: the map is free to change
            os.environ["SHOP_TOUR_PRIVACY_INPUTS"] = "map"
            self.assertFalse(colmap_det.fast_match()); self.assertFalse(colmap_det.use_global())
            os.environ["SHOP_TOUR_PRIVACY_INPUTS"] = "own"
            os.environ["SHOP_TOUR_COLMAP_MATCH"] = "colmap"; os.environ["SHOP_TOUR_MAPPER"] = "incremental"      # an explicit choice always wins
            self.assertFalse(colmap_det.fast_match()); self.assertFalse(colmap_det.use_global())
        finally:
            for k in keep:
                os.environ.pop(k, None)
                if keep[k] is not None:
                    os.environ[k] = keep[k]

    def test_the_privacy_path_reads_nothing_from_vio_or_the_map(self):
        for f in ("remove_capturer.py", "capturer.py", "privacy_gate.py", "privacy_margin.py"):
            src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", f), encoding="utf-8").read()
            for name in ('"poses.json"', '"poses_map.json"', '"vio"', "poses.npz", '"colmap"'):
                self.assertNotIn(name, src, f + " reads " + name)


class TestLevelChoice(unittest.TestCase):
    """The verticals come first (2026-10-30): a street's skyline tilted the owner's stops 20-41 degrees while the vertical edges had them within a few."""
    def up(self, tx, tz):
        return lp.mk_up(tx, tz)

    def test_a_believable_edge_vote_beats_a_skyline_of_roofs(self):
        u, c, way = lp.choose_up(self.up(3, 0), 3.2, (self.up(40, 0), 0.4, 0.33))
        self.assertAlmostEqual(lp.tilt_of(u), 3, places=5); self.assertIn("roofs and trees", way)

    def test_the_skyline_agreeing_is_still_the_edges(self):
        u, c, way = lp.choose_up(self.up(4, 0), 2.0, (self.up(5, 0), 0.3, 0.38))
        self.assertAlmostEqual(lp.tilt_of(u), 4, places=5); self.assertIn("agrees", way)

    def test_a_beach_has_no_vertical_so_the_horizon_levels_it(self):
        u, c, way = lp.choose_up(self.up(30, 0), 1.8, (self.up(31, 2), 0.9, 0.5))
        self.assertAlmostEqual(lp.tilt_of(u), math.degrees(math.acos(math.cos(math.radians(31)) * math.cos(math.radians(2)))), places=4)
        self.assertTrue(way.startswith("horizon"))

    def test_nothing_to_read_is_level(self):
        u, c, way = lp.choose_up(self.up(32, 0), 1.8, None)
        self.assertEqual(list(u), [0.0, 1.0, 0.0]); self.assertTrue(way.startswith("level"))

    def test_the_sensor_levels_a_stop_through_the_mount(self):
        """gravity read in the sensor frame, turned by the mount, is the pano's up; a stop the stream does not cover is None"""
        from scipy.spatial.transform import Rotation as Rot
        M = Rot.from_rotvec(lp.MOUNT_ROTVEC).as_matrix()
        t = np.arange(0, 10, 0.001)
        for want in (np.array([0.0, 1.0, 0.0]), lp.mk_up(25, -10)):
            f = np.tile(M.T @ (9.80665 * want), (len(t), 1)) + np.random.default_rng(1).normal(0, 0.8, (len(t), 3))   # a walker's jitter, zero-mean
            d = dict(t=t, acc=f, gyro=np.zeros((len(t), 3)), rate=1000.0)
            u = lp.up_from_imu(d, 5.0)
            self.assertLess(np.degrees(np.arccos(np.clip(float(u @ want), -1, 1))), 0.6)
        self.assertIsNone(lp.up_from_imu(dict(t=t, acc=f, gyro=np.zeros((len(t), 3)), rate=1000.0), 40.0))

    def test_a_turn_during_the_window_is_taken_out_by_the_gyroscope(self):
        """the sensor spins half a turn about the lens axis across the window: every sample still votes for the same up at the stop's instant"""
        from scipy.spatial.transform import Rotation as Rot
        M = Rot.from_rotvec(lp.MOUNT_ROTVEC).as_matrix()
        t = np.arange(0, 6, 0.001); w = np.pi / 3                            # rad/s about the sensor's z
        ang = w * (t - 3.0)
        g_s0 = M.T @ np.array([0.0, 1.0, 0.0])                                 # gravity in the sensor frame at the stop (t = 3)
        Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        f = np.array([Rz(-a) @ g_s0 * 9.80665 for a in ang])                   # the body turned by +a: the fixed gravity reads turned by -a
        d = dict(t=t, acc=f, gyro=np.tile([0, 0, w], (len(t), 1)), rate=1000.0)
        u = lp.up_from_imu(d, 3.0)
        self.assertLess(np.degrees(np.arccos(np.clip(float(u @ [0, 1, 0]), -1, 1))), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
