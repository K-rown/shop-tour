"""A detail try that dies in its own process is no map — the step goes on with the map it already made (map_stops.run_sfm_apart)."""
import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import map_stops


class MapApart(unittest.TestCase):
    def test_a_try_that_dies_is_no_map(self):
        with tempfile.TemporaryDirectory() as d:
            frames = os.path.join(d, "frames"); os.makedirs(os.path.join(frames, "points"))
            json.dump([{"filename": "missing.jpg"}], open(os.path.join(frames, "points.json"), "w"))
            pts, rec = map_stops.run_sfm_apart(frames, os.path.join(d, "work_try1"), 256, 512, 0)
            self.assertIsNone(rec, "a try whose process failed hands back no map")
            self.assertTrue(getattr(map_stops.run_sfm_apart, "died", False), "and it says so, so the loop stops")
            self.assertEqual([p["filename"] for p in pts], ["missing.jpg"], "the stops are still the walk's own")

    def test_the_retries_run_apart_and_the_first_try_in_place(self):
        src = open(map_stops.__file__, encoding="utf-8").read()
        self.assertIn("(run_sfm if t == 0 else run_sfm_apart)(a.frames, wdir, width, nfeat, win)", src)
        self.assertIn('if getattr(run_sfm_apart, "died", False):', src, "after a try dies, no heavier try is started")


if __name__ == "__main__":
    unittest.main()
