"""Tests for the blur of plates, faces and printed names (2026-10-22: pii_blur.py).
    python qa/test_pii_blur.py
No models, GPU or footage: a fake finder reports a vehicle box, a face box, the text detector's boxes and the plate model's boxes on synthetic pictures (the plate and the name tag are drawn white rectangles with letter-like strokes;
the fake "text detector" boxes every bright blob, the fake "plate model" every bright blob in a view cut round a vehicle).
"""
import json, os, sys, tempfile, types, unittest

os.environ.setdefault("SHOP_TOUR_PEOPLE", "remove")        # these tests are about the older flow: the paint-out and the holds (the default is now to smudge people and hold nothing: qa/test_people_blur.py)

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "qa"))
import pii_blur as PB       # noqa: E402
import manifest             # noqa: E402
import privacy_gate as PG   # noqa: E402

W = 4096
VEH_AZ, VEH_EL = 60.0, -2.0                      # the vehicle: 20 degrees wide, 12 high
FACE_AZ, FACE_EL = -80.0, 5.0


def window(img, kind, az, el, hw, hh):
    """(index, mask) of the tangent-plane rectangle (half extents hw, hh degrees) centred on (az, el) of an equirect picture, with no growth."""
    r = dict(kind=kind, az=az, el=el, hw=hw, hh=hh)
    rows, cols, m, _ = PB._mask_window(r, *img.shape[:2])
    return np.ix_(rows, cols), m


def fill(img, az, el, hw, hh, color):
    g = PB.GROW["plate"]
    ix, m = window(img, "plate", az, el, (hw - g[1]) / g[0], (hh - g[1]) / g[0])
    sub = img[ix]; sub[m > 0.5] = color; img[ix] = sub


def plate(img, az, el, hw=1.8, hh=0.55, chars=7):
    """A white plate with dark vertical strokes (letters) on it."""
    g = PB.GROW["plate"]
    ix, m = window(img, "plate", az, el, (hw - g[1]) / g[0], (hh - g[1]) / g[0])
    sub = img[ix]; on = m > 0.5; sub[on] = (235, 235, 235)
    ys, xs = np.nonzero(on); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); w, h = x1 - x0, y1 - y0
    for k in range(chars):
        a = int(x0 + w * (0.08 + 0.84 * k / chars)); b = int(x0 + w * (0.08 + 0.84 * (k + 0.55) / chars))
        sub[int(y0 + 0.2 * h):int(y1 - 0.2 * h), a:b] = (25, 25, 25)
    img[ix] = sub


def scene(with_plate=True, with_tag=False):
    img = np.full((W // 2, W, 3), 110, np.uint8)
    fill(img, VEH_AZ, VEH_EL, 10.0, 6.0, (70, 70, 80))                          # the vehicle's body
    if with_plate:
        plate(img, VEH_AZ, VEH_EL - 3.2)                                        # low on it
    if with_tag:
        fill(img, FACE_AZ, FACE_EL - 4, 3.0, 8.0, (60, 60, 160))                # a "person" in a red shirt ...
        plate(img, FACE_AZ, FACE_EL - 3.6, hw=1.6, hh=0.8, chars=5)            # ... with a tag on his chest
    ix, m = window(img, "face", FACE_AZ, FACE_EL, 0.7, 0.9)
    sub = img[ix]; sub[m > 0.5] = (150, 170, 200)                               # a skin-coloured disc with dark eyes and mouth
    ys, xs = np.nonzero(m > 0.5); cy, cx = int(ys.mean()), int(xs.mean()); r = max(2, int((xs.max() - xs.min()) * 0.1))
    for dx, dy in ((-3, -2), (3, -2), (0, 3)):
        cv2.circle(sub, (cx + dx * r, cy + dy * r), r, (20, 20, 20), -1)
    img[ix] = sub
    return img


class FakeFinder:
    """Reports the vehicle (and with person=True the man in the red shirt) where the scene put them, in whatever view sees them, and the one skin-coloured disc as a face."""
    def __init__(self, person=False, boom=False, text=True, plate_model=False, plate_score=0.9, face_score=0.9, text_score=0.9, skip_texty=True):
        self.person, self.boom, self.use_text, self.use_plates = person, boom, text, plate_model
        self.plate_score, self.face_score, self.text_score = plate_score, face_score, text_score
        self.missing = []
        self.skip_texty = skip_texty                                             # (the synthetic strokes are a few pixels wide; texty() has its own test below)

    @staticmethod
    def _box(yaw, pitch, az, el, hw, hh):
        from depth_panos import view_rays
        d = view_rays(yaw, pitch, PG.VIEW_FOV, PG.VIEW_N)
        a = np.degrees(np.arctan2(d[..., 0], d[..., 2])); e = np.degrees(np.arcsin(np.clip(d[..., 1], -1, 1)))
        inside = (np.abs((a - az + 180) % 360 - 180) <= hw) & (np.abs(e - el) <= hh)
        if inside.sum() < 200:
            return None
        ys, xs = np.nonzero(inside)
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    def objects(self, views):
        if self.boom:
            raise RuntimeError("detector exploded")
        out = []
        for yaw, pitch in PG.VIEWS:
            lst = []
            b = self._box(yaw, pitch, VEH_AZ, VEH_EL, 10.0, 6.0)
            if b:
                lst.append(("vehicle", 0.9, b))
            if self.person:
                b = self._box(yaw, pitch, FACE_AZ, FACE_EL - 4, 3.0, 8.0)
                if b:
                    lst.append(("person", 0.9, b))
            out.append(lst)
        return out

    @staticmethod
    def _blobs(img, score=0.9):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m = cv2.dilate((g > 225).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= 150:
                out.append((score, [float(x), float(y), float(x + w), float(y + h)]))
        return out

    def text(self, img):
        return self._blobs(img, self.text_score) if self.use_text else []

    def plates(self, imgs):
        return [self._blobs(i, self.plate_score) if self.use_plates else [] for i in imgs]

    def faces(self, view):
        m = cv2.inRange(view, (135, 155, 185), (165, 185, 215))
        if int(m.sum() / 255) < 150:
            return []
        ys, xs = np.nonzero(m)
        return [(self.face_score, [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])]


class TestText(unittest.TestCase):
    def test_a_text_box_on_a_low_vehicle_is_a_plate_on_a_person_a_name_and_elsewhere_text(self):
        img = scene(with_plate=True, with_tag=True)
        plate(img, -20.0, 25.0, hw=2.4, hh=0.9, chars=8)                                  # a sign on bare ground: no vehicle, no person
        regions = PB.detect(FakeFinder(person=True), img)
        kinds = sorted(r["kind"] for r in regions)
        self.assertEqual(kinds, ["face", "name", "plate", "text"], regions)
        near = lambda k, az, el: min(PG._ang((r["az"], r["el"]), (az, el)) for r in regions if r["kind"] == k)
        self.assertLess(near("plate", VEH_AZ, VEH_EL - 3.2), 1.0); self.assertLess(near("name", FACE_AZ, FACE_EL - 3.6), 1.5); self.assertLess(near("text", -20.0, 25.0), 1.5)

    def test_the_plate_model_finds_a_plate_the_text_detector_did_not_and_one_plate_is_one_blur(self):
        img = scene()
        only_model = PB.detect(FakeFinder(text=False, plate_model=True), img)
        self.assertEqual([r["kind"] for r in only_model].count("plate"), 1, only_model)
        both = PB.detect(FakeFinder(text=True, plate_model=True), img)
        self.assertEqual([r["kind"] for r in both].count("plate"), 1, both)               # the same plate seen twice is one region
        self.assertEqual([r["kind"] for r in both].count("name") + [r["kind"] for r in both].count("text"), 0, both)

    def test_a_plain_scene_has_no_text_and_a_tall_box_is_not_a_plate(self):
        img = np.full((W // 2, W, 3), 110, np.uint8)
        self.assertEqual(PB.detect(FakeFinder(), img), [])
        fill(img, VEH_AZ, VEH_EL, 10.0, 6.0, (70, 70, 80))
        g = PB.GROW["plate"]
        ix, m = window(img, "plate", VEH_AZ, VEH_EL - 3.0, 0.4 / g[0], 2.0 / g[0])     # a bright tall bar on a vehicle: a pole, a poster
        sub = img[ix]; sub[m > 0.5] = (240, 240, 240); img[ix] = sub
        self.assertEqual([r for r in PB.detect(FakeFinder(text=False, plate_model=True), img) if r["kind"] == "plate"], [])


class TestPrecision(unittest.TestCase):
    """The measured rules of 2026-10-22 (pii3): a plate needs the plate model sure, or the text detector boxing the same place; a plate is plate-sized; a face is kept in a head zoom at any score, in a view
    on a person or a driver at 0.6, anywhere at 0.88; a text box the detector is sure of skips the glyph check."""
    def plates_of(self, img, **kw):
        return [r for r in PB.detect(FakeFinder(**kw), img) if r["kind"] == "plate"]

    def test_a_plate_the_model_is_unsure_of_needs_the_text_detector_to_box_the_same_place(self):
        img = scene(with_plate=False)
        plate(img, VEH_AZ, VEH_EL + 1.5)                                                                        # a plate-shaped thing high on the vehicle: a headlight, a grille
        self.assertEqual(self.plates_of(img, text=False, plate_model=True, plate_score=0.5), [])               # the model alone at 0.5 is not enough ...
        self.assertEqual(len(self.plates_of(img, text=True, plate_model=True, plate_score=0.5)), 1)          # ... with the text detector on the same place it is
        self.assertEqual(len(self.plates_of(img, text=False, plate_model=True, plate_score=0.75)), 1)        # ... and at 0.7 and over it is on its own
        low = scene()                                                                                           # the same thing LOW on the vehicle, where a plate is, narrow beside it: kept at 0.5 on the model alone
        self.assertEqual(len(self.plates_of(low, text=False, plate_model=True, plate_score=0.5)), 1)

    def test_a_plate_is_plate_sized_beside_its_vehicle(self):
        img = scene(with_plate=False)
        ix, m = window(img, "plate", VEH_AZ, VEH_EL, 8.0, 4.5)                                                  # a bright box over most of the vehicle: the vehicle, not its plate
        sub = img[ix]; sub[m > 0.5] = (240, 240, 240); img[ix] = sub
        self.assertEqual(self.plates_of(img, text=False, plate_model=True), [])
        self.assertEqual(len(self.plates_of(scene(), text=False, plate_model=True)), 1)                        # a real plate on it is kept

    def test_faces_are_kept_in_a_head_zoom_at_any_score_and_in_a_view_only_with_a_person_or_a_high_score(self):
        img = scene(with_tag=True)
        faces = lambda **kw: [r for r in PB.detect(FakeFinder(**kw), img) if r["kind"] == "face"]
        self.assertEqual(len(faces(person=True, face_score=0.4)), 1)             # the head zoom of the person finds it
        self.assertEqual(len(faces(person=False, face_score=0.7)), 0)            # on nothing, at 0.7: a shelf or a fan
        self.assertEqual(len(faces(person=False, face_score=0.9)), 1)            # a poster's face
        self.assertEqual(len(faces(person=True, face_score=0.7)), 1)

    def test_a_wide_box_the_detector_is_sure_of_gets_a_second_chance_that_foliage_and_wheels_do_not(self):
        sign = np.full((60, 220, 3), 40, np.uint8)                                                              # white on red, two lines of unequal letters: texty() wants one row of equal glyphs
        sign[:] = (40, 40, 190)
        cv2.rectangle(sign, (20, 8), (60, 50), (240, 240, 240), -1); cv2.rectangle(sign, (90, 14), (200, 52), (240, 240, 240), -1)
        box = [0, 0, 220, 60]
        self.assertFalse(PB.texty(sign, box)); self.assertTrue(PB.texty_lite(sign, box))
        leaves = np.full((60, 220, 3), (30, 90, 30), np.uint8)
        for cx, cy, r in ((20, 15, 9), (150, 50, 8), (200, 30, 12)):
            cv2.circle(leaves, (cx, cy), r, (120, 230, 150), -1)
        self.assertFalse(PB.texty_lite(leaves, box))                                                           # green: leaves
        wheel = np.full((60, 60, 3), 60, np.uint8); cv2.circle(wheel, (30, 30), 22, (200, 200, 200), -1)
        self.assertFalse(PB.texty_lite(wheel, [0, 0, 60, 60]))                                                 # square: a wheel, not a line of text
        one = np.full((60, 220, 3), 40, np.uint8); cv2.rectangle(one, (60, 10), (150, 50), (240, 240, 240), -1)
        self.assertFalse(PB.texty_lite(one, box))                                                              # one piece: a bar


class TestTexty(unittest.TestCase):
    def test_lettering_passes_and_foliage_wires_and_smooth_surfaces_do_not(self):
        rng = np.random.RandomState(3)
        word = np.full((60, 220, 3), 225, np.uint8)
        cv2.putText(word, "ZGV9747", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 30), 3, cv2.LINE_AA)
        box = [0, 0, 220, 60]
        self.assertTrue(PB.texty(word, box))
        inverse = 255 - word
        self.assertTrue(PB.texty(inverse, box))                                          # light letters on a dark door
        leaves = np.full((60, 220, 3), 90, np.uint8)
        for cx, cy, r in ((20, 15, 9), (70, 40, 22), (120, 12, 6), (150, 50, 8), (200, 30, 28)):
            cv2.circle(leaves, (cx, cy), r, (200, 230, 190), -1)
        self.assertFalse(PB.texty(leaves, box))                                          # a few blobs of different sizes and heights: no row of equal glyphs
        self.assertFalse(PB.texty(np.full((60, 220, 3), 120, np.uint8), box))           # smooth
        wire = np.full((60, 220, 3), 200, np.uint8); cv2.line(wire, (0, 10), (220, 50), (20, 20, 20), 3)
        self.assertFalse(PB.texty(wire, box))                                            # one stroke


class TestDetectAndBlur(unittest.TestCase):
    def test_the_plate_and_the_face_are_found_and_blurred_beyond_reading(self):
        img = scene()
        regions = PB.detect(FakeFinder(), img)
        kinds = [r["kind"] for r in regions]
        self.assertEqual(kinds.count("plate"), 1, regions); self.assertEqual(kinds.count("face"), 1, regions)
        pl = next(r for r in regions if r["kind"] == "plate")
        self.assertLess(PG._ang((pl["az"], pl["el"]), (VEH_AZ, VEH_EL - 3.2)), 1.0)
        out, counts = PB.blur_regions(img, regions)
        self.assertEqual(counts, dict(plate=1, face=1, name=0, text=0))
        ix, m = window(img, "plate", VEH_AZ, VEH_EL - 3.2, 1.2, 0.35)                   # the middle of the plate
        self.assertGreater(float(np.abs(img[ix][m > 0.5].astype(float) - out[ix][m > 0.5].astype(float)).mean()), 15.0)
        g = cv2.cvtColor(out[ix], cv2.COLOR_BGR2GRAY).astype(float); o = cv2.cvtColor(img[ix], cv2.COLOR_BGR2GRAY).astype(float)
        self.assertLess(float(g[m > 0.5].std()), 0.5 * float(o[m > 0.5].std()))        # the dark/bright contrast the letters needed is gone
        ys, xs = np.nonzero(m > 0.5)
        self.assertLess(float(np.abs(np.diff(g[ys.min():ys.max(), xs.min():xs.max()], axis=1)).mean()), 3.0)     # smooth across, no strokes
        changed = (out != img).any(-1)
        self.assertLess(int(changed.sum()), 0.03 * changed.size)                         # only the two regions moved
        self.assertGreaterEqual(PB.CELL_MIN, 8)

    def test_the_same_picture_gives_the_same_regions_and_pixels(self):
        img = scene()
        a, b = PB.detect(FakeFinder(), img), PB.detect(FakeFinder(), img)
        self.assertEqual(a, b)
        self.assertTrue(np.array_equal(PB.blur_regions(img, a)[0], PB.blur_regions(img, b)[0]))

    def test_a_name_tag_on_a_person_is_a_name_and_a_vehicle_without_a_plate_gives_none(self):
        img = scene(with_plate=False, with_tag=True)
        regions = PB.detect(FakeFinder(person=True), img)
        self.assertEqual([r for r in regions if r["kind"] == "plate"], [])
        names = [r for r in regions if r["kind"] == "name"]
        self.assertGreaterEqual(len(names), 1, regions)
        self.assertLess(min(PG._ang((r["az"], r["el"]), (FACE_AZ, FACE_EL - 2.2)) for r in names), 1.5)

    def test_a_region_across_the_seam_wraps(self):
        img = np.full((W // 2, W, 3), 120, np.uint8)
        img[:, :40] = 200; img[:, -40:] = 40                                              # a hard edge at the seam
        out, c = PB.blur_regions(img, [dict(kind="face", az=179.7, el=0.0, hw=1.0, hh=1.0)])
        self.assertEqual(c["face"], 1)
        self.assertTrue((out[:, :40] != img[:, :40]).any() and (out[:, -40:] != img[:, -40:]).any())     # both sides of the seam were blurred


class TestStep(unittest.TestCase):
    def _frames(self, d, n=2):
        fr = os.path.join(d, "frames"); os.makedirs(os.path.join(fr, "safe"))
        pts = [dict(id=f"{i:03d}", filename=f"{i:03d}.jpg") for i in range(1, n + 1)]
        json.dump(pts, open(os.path.join(fr, "points.json"), "w"))
        json.dump(dict(version=1, stops={p["id"]: dict(held=False, reasons=[]) for p in pts}), open(os.path.join(fr, "privacy.json"), "w"))
        cv2.imwrite(os.path.join(fr, "safe", "001.jpg"), scene(), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(fr, "safe", "002.jpg"), np.full((W // 2, W, 3), 100, np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
        return fr, pts

    def test_the_step_blurs_records_and_does_not_blur_twice(self):
        with tempfile.TemporaryDirectory() as d:
            fr, pts = self._frames(d)
            p1 = os.path.join(fr, "safe", "001.jpg"); before = open(p1, "rb").read()
            rec = PB.run(fr, pts, fd_factory=FakeFinder, log=lambda *a: None)
            body = PB.write_record(fr, rec)
            self.assertEqual(body["totals"], dict(plate=1, face=1, name=0, text=0)); self.assertEqual(body["stops_blurred"], 1); self.assertEqual(body["errors"], [])
            after = open(p1, "rb").read(); self.assertNotEqual(before, after)
            pv = json.load(open(os.path.join(fr, "privacy.json")))["stops"]
            self.assertEqual(pv["001"]["pii"]["counts"], dict(plate=1, face=1, name=0, text=0)); self.assertEqual(pv["002"]["pii"]["counts"], dict(plate=0, face=0, name=0, text=0))

            def no_load():
                raise AssertionError("must not load the detectors again")
            rec2 = PB.run(fr, pts, fd_factory=no_load, log=lambda *a: None)
            self.assertEqual(open(p1, "rb").read(), after)                                # a second run leaves the file alone: no second blur, no recompression
            self.assertEqual(rec2["001"]["counts"], rec["001"]["counts"])

    def test_a_failure_is_never_a_gate_and_the_stop_ships_as_it_was(self):
        with tempfile.TemporaryDirectory() as d:
            fr, pts = self._frames(d)
            p1 = os.path.join(fr, "safe", "001.jpg"); before = open(p1, "rb").read()
            rec = PB.run(fr, pts, fd_factory=lambda: FakeFinder(boom=True), log=lambda *a: None)
            self.assertEqual(open(p1, "rb").read(), before)
            self.assertIn("exploded", rec["001"]["error"]); self.assertEqual(sum(rec["001"]["counts"].values()), 0)
            body = PB.write_record(fr, rec); self.assertEqual(body["errors"], ["001", "002"])
            m = manifest.pii_block(fr, ["001", "002"])
            self.assertEqual(m["status"], "partial"); self.assertEqual(m["errors"], ["001", "002"])

            def no_weights():
                raise RuntimeError("no weights")
            rec = PB.run(fr, pts, fd_factory=no_weights, log=lambda *a: None)             # the detectors do not even load
            self.assertIn("did not load", rec["001"]["error"]); self.assertEqual(open(p1, "rb").read(), before)

    def test_the_manifest_counts_by_kind_and_knows_off_and_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            fr, pts = self._frames(d)
            self.assertEqual(manifest.pii_block(fr, ["001", "002"])["status"], "not_run")
            PB.write_record(fr, PB.run(fr, pts, fd_factory=FakeFinder, log=lambda *a: None))
            m = manifest.pii_block(fr, ["001", "002"])
            self.assertEqual((m["status"], m["blurred"], m["stops_blurred"]), ("ok", dict(plate=1, face=1, name=0, text=0), 1))
            self.assertEqual(manifest.pii_block(fr, ["002"])["blurred"], dict(plate=0, face=0, name=0, text=0))     # only the stops that are in the tour count
            json.dump(dict(version=1, enabled=False, stops={}), open(os.path.join(fr, "pii.json"), "w"))
            self.assertEqual(manifest.pii_block(fr, ["001"])["status"], "off")

    def test_another_copy_of_the_stops_is_blurred_with_the_same_regions(self):
        with tempfile.TemporaryDirectory() as d:
            fr, pts = self._frames(d)
            hd = os.path.join(d, "hd"); os.makedirs(hd)
            cv2.imwrite(os.path.join(hd, "001.jpg"), cv2.resize(scene(), (W * 3 // 2, W * 3 // 4)), [cv2.IMWRITE_JPEG_QUALITY, 95])     # (a copy of the same moment at another size)
            PB.write_record(fr, PB.run(fr, pts, fd_factory=FakeFinder, log=lambda *a: None))
            done = PB.apply_dir(fr, hd, os.path.join(d, "hd_out"))
            self.assertEqual(done["001"], dict(plate=1, face=1, name=0, text=0))
            a = cv2.imread(os.path.join(hd, "001.jpg")); b = cv2.imread(os.path.join(d, "hd_out", "001.jpg"))
            self.assertEqual(a.shape, b.shape); self.assertGreater(int((a != b).any(-1).sum()), 500)

    def test_the_step_is_in_the_pipeline_and_on_by_default(self):
        import tour
        names = [s[0] for s in tour.steps("x.insv")]
        self.assertEqual(names.index("pii"), names.index("privacy") + 1)
        self.assertEqual(set(tour.prereqs("pii", names)), {"privacy"})
        for n in ("mvs", "build", "dollhouse"):
            self.assertIn("pii", tour.prereqs(n, names))
        keep = os.environ.pop("SHOP_TOUR_PII_BLUR", None)
        try:
            self.assertTrue(PB.enabled())
            os.environ["SHOP_TOUR_PII_BLUR"] = "0"; self.assertFalse(PB.enabled())
        finally:
            os.environ.pop("SHOP_TOUR_PII_BLUR", None)
            if keep is not None:
                os.environ["SHOP_TOUR_PII_BLUR"] = keep


class TestStream(unittest.TestCase):
    """THE BLUR BESIDE THE PRIVACY CHAIN (2026-10-30): pii_blur.Stream, fed stop by stop by privacy_chain.py, must leave exactly the pictures and the record the step leaves."""
    _frames = TestStep._frames

    def test_the_stream_leaves_what_the_step_leaves(self):
        with tempfile.TemporaryDirectory() as d:
            fa, pts = self._frames(os.path.join(d, "a"))
            fb, _ = self._frames(os.path.join(d, "b"))
            ra = PB.run(fa, pts, fd_factory=FakeFinder, log=lambda *a: None); ba = PB.write_record(fa, ra)
            s = PB.Stream(fb, pts, fd_factory=FakeFinder, log=lambda *a: None)
            for k in (1, 0):                                                    # (in whatever order the checks finish)
                s.submit(k)
            rb = s.close(); bb = PB.write_record(fb, rb, streamed=True)
            self.assertEqual(list(rb), [p["id"] for p in pts])                  # the record in the stop order, as run() writes it
            self.assertEqual(rb, ra)
            for p in pts:
                self.assertEqual(open(os.path.join(fa, "safe", p["filename"]), "rb").read(), open(os.path.join(fb, "safe", p["filename"]), "rb").read())
            self.assertTrue(bb.pop("streamed")); self.assertNotIn("streamed", ba); self.assertEqual(ba, bb)
            self.assertEqual(json.load(open(os.path.join(fa, "privacy.json"))), json.load(open(os.path.join(fb, "privacy.json"))))

    def test_a_stop_never_submitted_or_written_again_is_looked_at_in_close(self):
        with tempfile.TemporaryDirectory() as d:
            fr, pts = self._frames(d)
            seen = []

            class Counting(FakeFinder):
                def objects(self, views):
                    seen.append(1)
                    return super().objects(views)
            s = PB.Stream(fr, pts, fd_factory=Counting, log=lambda *a: None)
            s.submit(1)                                                         # 001 is never submitted
            import time
            t = time.time()
            while "002" not in s.got and time.time() - t < 60:
                time.sleep(0.05)
            cv2.imwrite(os.path.join(fr, "safe", "002.jpg"), scene(), [cv2.IMWRITE_JPEG_QUALITY, 95])     # 002 written again after it was looked at
            rec = s.close()
            self.assertEqual(sorted(rec), ["001", "002"])
            self.assertEqual(rec["001"]["counts"], dict(plate=1, face=1, name=0, text=0))
            self.assertEqual(rec["002"]["counts"], dict(plate=1, face=1, name=0, text=0))           # the new picture's, not the first one's
            self.assertEqual(len(seen), 3)

    def test_a_stop_with_no_safe_picture_is_left_out_and_a_failure_is_never_a_gate(self):
        with tempfile.TemporaryDirectory() as d:
            fr, pts = self._frames(d)
            os.remove(os.path.join(fr, "safe", "002.jpg"))
            before = open(os.path.join(fr, "safe", "001.jpg"), "rb").read()
            s = PB.Stream(fr, pts, fd_factory=lambda: FakeFinder(boom=True), log=lambda *a: None)
            s.submit(0); s.submit(1)
            rec = s.close()
            self.assertEqual(list(rec), ["001"]); self.assertIn("exploded", rec["001"]["error"])
            self.assertEqual(open(os.path.join(fr, "safe", "001.jpg"), "rb").read(), before)
            PB.write_record(fr, rec, streamed=True)
            import tour
            self.assertIn("could not look", tour.incomplete("pii", fr))       # tour.py's pii step looks again at what the thread could not
            PB.write_record(fr, PB.run(fr, pts, fd_factory=lambda: FakeFinder(boom=True), log=lambda *a: None))
            self.assertIsNone(tour.incomplete("pii", fr))                       # ... once: the step's own record is final, as it always was

    def test_only_where_a_safe_picture_is_never_taken_back(self):
        keep = {k: os.environ.get(k) for k in ("SHOP_TOUR_PEOPLE", "SHOP_TOUR_PII_STREAM", "SHOP_TOUR_PII_BLUR")}
        try:
            for k in keep:
                os.environ.pop(k, None)
            self.assertTrue(PB.stream_wanted())                                 # keep (the default)
            os.environ["SHOP_TOUR_PEOPLE"] = "blur"; self.assertTrue(PB.stream_wanted())
            os.environ["SHOP_TOUR_PEOPLE"] = "remove"; self.assertFalse(PB.stream_wanted())         # the margin may still hold a stop after its check
            os.environ["SHOP_TOUR_PEOPLE"] = "keep"; os.environ["SHOP_TOUR_PII_STREAM"] = "0"; self.assertFalse(PB.stream_wanted())
            os.environ.pop("SHOP_TOUR_PII_STREAM"); os.environ["SHOP_TOUR_PII_BLUR"] = "0"; self.assertFalse(PB.stream_wanted())
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_the_chain_feeds_it_after_each_check_and_writes_the_record_last(self):
        src = open(os.path.join(HERE, "privacy_chain.py"), encoding="utf-8").read()
        self.assertIn("PB.stream_wanted()", src)
        self.assertLess(src.index("mine.gate.stop(k)"), src.index("stream.submit(k)"))
        self.assertLess(src.index("PG.finish(frames, pts, state)"), src.index("PB.write_record(frames, rec, streamed=True)"))
        self.assertLess(src.index("PM.main(argv="), src.index("stream.close()"))


if __name__ == "__main__":
    unittest.main()
