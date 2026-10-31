"""The sphere depth, the registration, the UV atlas and the top-down plan: the pure parts, on synthetic rooms (no model, no GPU needed).

  python -m unittest qa.test_engine_v2 -v
"""
import math, os, sys, tempfile, unittest
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import depth_sphere as DS        # noqa: E402
import dollhouse_uv as DU        # noqa: E402
import floorplan as FP           # noqa: E402
import register_stops as RS      # noqa: E402
import level_panos as LP         # noqa: E402


def room_range(q, half_w=4.0, half_d=6.0, cam_h=2.0, roof=4.5):
    """Range along levelled unit rays q (..., 3) to the walls, floor and roof of a box room; the camera at the origin, cam_h above the floor."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(q[..., 0] != 0, np.where(q[..., 0] > 0, half_w, -half_w) / q[..., 0], np.inf)
        tz = np.where(q[..., 2] != 0, np.where(q[..., 2] > 0, half_d, -half_d) / q[..., 2], np.inf)
        ty = np.where(q[..., 1] < 0, -cam_h / q[..., 1], np.where(q[..., 1] > 0, (roof - cam_h) / q[..., 1], np.inf))
    return np.minimum(np.minimum(tx, tz), ty).astype(np.float32)


class SphereDepth(unittest.TestCase):
    def test_view_scales_come_back_from_the_overlaps(self):
        """14 views of one room, each given its own wrong scale: the solve puts them back together to a few percent, mean 0 (MoGe's metre)."""
        n = 160
        Bs = [DS.basis(v) for v in DS.VIEWS]
        d, f = DS.view_rays(n)
        rs = np.random.default_rng(3)
        true = rs.normal(0, 0.3, len(Bs)); true -= true.mean()
        rngs, masks = [], []
        for B, t in zip(Bs, true):
            q = d @ B                                                         # view rays -> levelled
            r = room_range(q) * math.exp(-t)                                  # a view that thinks the room is exp(-t) times its size
            rngs.append(r.astype(np.float32)); masks.append(np.ones((n, n), bool))
        x, rep = DS.solve_scales(rngs, masks, Bs, None, n, f)
        np.testing.assert_allclose(x, true, atol=0.03)
        self.assertEqual(rep["anchors"], 0); self.assertGreater(rep["pairs"], 20)

    def test_sparse_points_say_what_the_map_thinks_of_the_metre(self):
        """With anchors 1.5x the views' ranges, map_over_moge reports 1.5 and the view scales stay MoGe's (mean 0)."""
        n = 128
        Bs = [DS.basis(v) for v in DS.VIEWS]
        d, f = DS.view_rays(n)
        rngs = [room_range(d @ B).astype(np.float32) for B in Bs]; masks = [np.ones((n, n), bool)] * len(Bs)
        rs = np.random.default_rng(1)
        q = rs.normal(size=(3000, 3)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        anchors = (q, room_range(q) * 1.5)
        x, rep = DS.solve_scales(rngs, masks, Bs, anchors, n, f)
        self.assertAlmostEqual(float(x.mean()), 0.0, places=6)
        self.assertAlmostEqual(rep["map_over_moge"], 1.5, places=2)
        self.assertGreater(rep["anchors"], 10)

    def test_composite_covers_the_sphere_and_keeps_the_room(self):
        import torch
        n = 128; W, H = 512, 256
        Bs = [DS.basis(v) for v in DS.VIEWS]
        d, f = DS.view_rays(n)
        rngs = [room_range(d @ B).astype(np.float32) for B in Bs]; masks = [np.ones((n, n), bool)] * len(Bs)
        rng, spread, have = DS.composite(rngs, masks, np.zeros(len(Bs)), Bs, n, f, W, H, torch.device("cpu"))
        self.assertGreater(have.mean(), 0.999)
        q = DS.eq_rays(W, H)
        truth = room_range(q)
        err = np.abs(rng - truth) / truth
        self.assertLess(float(np.median(err)), 0.02)
        self.assertLess(float(np.percentile(err, 95)), 0.06)                  # the corners of the room, sampled across view borders

    def test_floor_height_is_the_camera_height(self):
        W, H = 512, 256
        q = DS.eq_rays(W, H)
        rng = room_range(q, cam_h=2.3)
        h, sure = DS.floor_height(rng, q, None)
        self.assertAlmostEqual(h, 2.3, delta=0.03)
        self.assertGreater(sure, 0.3)

    def test_the_ground_under_him_not_the_beach_below(self):
        """A deck 2.3 m over its own ground with a beach 9 m below beyond it: the stick height is the deck's, even when the ground mask calls both ground."""
        W, H = 512, 256
        q = DS.eq_rays(W, H)
        rng = room_range(q, half_w=40, half_d=40, cam_h=2.3, roof=50)                      # the deck's ground, a big open box
        far = (q[..., 2] > 0.3) & (q[..., 1] < 0)                                           # ahead and below: the ground drops to a beach 9 m under the camera
        with np.errstate(divide="ignore"):
            beach = np.where(q[..., 1] < 0, -9.0 / q[..., 1], np.inf).astype(np.float32)
        rng = np.where(far, np.minimum(beach, 60.0), rng).astype(np.float32)
        ground = q[..., 1] < -0.05                                                          # everything below the horizon is "ground" to the segmenter
        h, sure = DS.floor_height(rng, q, ground)
        self.assertAlmostEqual(h, 2.3, delta=0.1)


class Registration(unittest.TestCase):
    def test_flatten_keeps_only_the_turn_and_the_slide(self):
        c, s = math.cos(0.3), math.sin(0.3)
        T = np.array([[c, -s, 0, 1.0], [s, c, 0, -2.0], [0, 0, 1, 0.7], [0, 0, 0, 1]])
        T2 = T.copy(); T2[:3, :3] = T[:3, :3] @ np.array([[1, 0, 0], [0, math.cos(0.1), -math.sin(0.1)], [0, math.sin(0.1), math.cos(0.1)]])   # a tilt that must go
        F = RS.flatten(T2)
        np.testing.assert_allclose(F[:3, :3], T[:3, :3], atol=1e-9)
        np.testing.assert_allclose(F[:3, 3], [1.0, -2.0, 0.7])

    def test_two_stops_of_one_room_meet(self):
        """A box room seen from two stops; the second stop's pose is 0.4 m and 6 degrees off: ICP brings it back."""
        import open3d as o3d
        W, H = 256, 128
        q = DS.eq_rays(W, H)

        def cloud(pos, heading):
            # the room in world coordinates, seen from `pos`: rays in the levelled frame turned by the heading
            import dollhouse as D
            M = D.level_to_world(heading)
            qw = q @ M.T                                                      # world rays
            # the box room in world: walls at x = +-4, y = +-6, floor z = 0, roof z = 4.5; the camera at pos
            with np.errstate(divide="ignore", invalid="ignore"):
                tx = np.where(qw[..., 0] != 0, (np.where(qw[..., 0] > 0, 4.0, -4.0) - pos[0]) / qw[..., 0], np.inf)
                ty = np.where(qw[..., 1] != 0, (np.where(qw[..., 1] > 0, 6.0, -6.0) - pos[1]) / qw[..., 1], np.inf)
                tz = np.where(qw[..., 2] < 0, -pos[2] / qw[..., 2], np.where(qw[..., 2] > 0, (4.5 - pos[2]) / qw[..., 2], np.inf))
            r = np.minimum(np.minimum(tx, ty), tz)
            P = (qw * r[..., None]).reshape(-1, 3) + pos
            # a little texture on the walls so ICP has something to grip: a bump every metre
            P[:, 2] += 0.03 * np.sin(P[:, 0] * 6) * np.sin(P[:, 1] * 6)
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(0.07)
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
            return pc
        a = cloud(np.array([0.0, 0.0, 2.0]), 0.0)
        true_b = np.array([1.5, 2.0, 2.0])
        b_true = cloud(true_b, 20.0)                                          # where stop b really is
        # the map thinks b is 0.4 m away and turned 6 degrees: move b's cloud by that error, then register it onto a
        err = np.eye(4); th = math.radians(6); err[:2, :2] = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]; err[0, 3], err[1, 3] = 0.3, -0.25
        b_wrong = o3d.geometry.PointCloud(b_true); b_wrong.transform(err)
        T, fit, rmse, info = RS.pair_icp(b_wrong, a)
        self.assertGreater(fit, 0.5)
        corr = RS.flatten(T) @ err
        self.assertLess(abs(math.degrees(math.atan2(corr[1, 0], corr[0, 0]))), 0.8)
        self.assertLess(float(np.hypot(corr[0, 3], corr[1, 3])), 0.06)


class WalkStaysOnePiece(unittest.TestCase):
    def test_a_pose_graph_with_a_break_in_it_moves_nothing_and_a_weak_edge_mends_it(self):
        """What register_stops.py guards against: Open3D leaves every node of a disconnected graph where it was."""
        import open3d as o3d
        T = np.eye(4); T[0, 3] = 1.0
        def solve(edges):
            pg = o3d.pipelines.registration.PoseGraph()
            for _ in range(4):
                pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
            for s_, t_, M, w in edges:
                pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(s_, t_, M, np.eye(6) * w, uncertain=False))
            opt = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=0.1, edge_prune_threshold=0.25, preference_loop_closure=0.4, reference_node=0)
            o3d.pipelines.registration.global_optimization(pg, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(), o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), opt)
            return [round(float(np.asarray(n.pose)[0, 3]), 2) for n in pg.nodes]
        self.assertEqual(solve([(1, 0, T, 1.0), (3, 2, T, 1.0)]), [0.0, 0.0, 0.0, 0.0])                  # node 2 has no path to node 0: nothing moves at all
        self.assertEqual(solve([(1, 0, T, 1.0), (2, 1, np.eye(4), 0.01), (3, 2, T, 1.0)]), [0.0, 1.0, 1.0, 2.0])   # a weak identity edge mends the walk


class Atlas(unittest.TestCase):
    def _box(self):
        import open3d as o3d
        m = o3d.geometry.TriangleMesh.create_box(3.0, 2.0, 1.0).subdivide_midpoint(3)
        return np.asarray(m.vertices, np.float32), np.asarray(m.triangles, np.int64)

    def test_box_unwrap_puts_every_triangle_inside_the_square_with_no_overlap(self):
        V, T = self._box()
        V2, T2, UV2, _ = DU.box_unwrap(V, T, size=1024, log=lambda *a: None)
        self.assertEqual(len(T2), len(T))
        self.assertTrue((UV2 >= 0).all() and (UV2 <= 1).all())
        np.testing.assert_allclose(np.sort(V2.view([("", V2.dtype)] * 3), axis=0).view(V2.dtype).reshape(-1, 3)[0], V.min(0), atol=1e-6)
        ti, P, tri = DU.texel_points(V2, T2, UV2, 1024)
        self.assertGreater(len(np.unique(ti)), 0.98 * len(ti), "texels claimed twice (neighbours may share the texels on their common edge, nothing more)")
        # a texel's 3D point lies on the box's surface
        on = (np.isclose(P[:, 0], 0, atol=0.02) | np.isclose(P[:, 0], 3, atol=0.02) | np.isclose(P[:, 1], 0, atol=0.02) | np.isclose(P[:, 1], 2, atol=0.02)
              | np.isclose(P[:, 2], 0, atol=0.02) | np.isclose(P[:, 2], 1, atol=0.02))
        self.assertGreater(on.mean(), 0.999)

    def test_six_faces_make_about_six_charts(self):
        V, T = self._box()
        DU_log = []
        DU.box_unwrap(V, T, size=1024, log=DU_log.append)
        n = int(DU_log[-1].split("->")[1].split("charts")[0])
        self.assertLessEqual(n, 8)

    def test_two_stops_a_gain_apart_come_out_within_3_percent_after_equalisation(self):
        """Three stops see the same wall; stop 1's picture is 0.8 of stop 0's and stop 2's is 1.15 of it: after the solve the same texels agree within 3 %."""
        rng = np.random.default_rng(3)
        n = 30000
        base = rng.uniform(40, 200, (n, 3)).astype(np.float32)
        true = np.array([[1.0] * 3, [0.8] * 3, [1.15] * 3], np.float32)
        def sampler(si, rows):
            return np.clip(base[rows] * true[si], 0, 255), np.ones(len(rows), bool)
        vis = np.ones((n, 3), bool); ti = np.tile(np.arange(3), (n, 1))
        pairs = DU.pair_rows(ti, vis, 3, cap=5000)
        self.assertEqual(sorted(pairs), [(0, 1), (0, 2), (1, 2)])
        self.assertTrue(all(len(r) <= 5000 for r in pairs.values()))
        logged = []
        g = DU.exposure_gains(3, pairs, sampler, log=logged.append)
        self.assertEqual(g.shape, (3, 3))
        seen = [(sampler(s, np.arange(n))[0] * g[s]).mean() for s in range(3)]
        self.assertLess(max(seen) / min(seen), 1.03, seen)
        self.assertIn("gains", logged[-1])
        self.assertAlmostEqual(float(np.median(g[:, 0] * true[:, 0])) / float(np.median(g[:, 0])), float(np.median(true[:, 0])), places=1)   # the median stop stays where it was

    def test_a_gain_is_held_to_its_limits(self):
        n = 5000
        base = np.full((n, 3), 100, np.float32)
        true = np.array([[1.0] * 3, [1.0] * 3, [0.3] * 3], np.float32)                  # a stop three times too dark: the fix stops at 1/0.7
        g = DU.exposure_gains(3, {(0, 2): np.arange(n), (1, 2): np.arange(n)}, lambda s, r: (base[r] * true[s], np.ones(len(r), bool)), log=lambda *a: None)
        self.assertTrue(((g >= DU.GAIN_LO - 1e-6) & (g <= DU.GAIN_HI + 1e-6)).all())
        self.assertAlmostEqual(float(g[2, 0]), DU.GAIN_HI, places=3)

    def test_a_boundary_texel_is_a_blend_of_the_two_stops(self):
        """Two stops side by side: the texels beside the boundary are half one, half the other, falling to the plain stop three texels out; a third stop touching
        a corner never makes it a three-way blend."""
        S = 32
        pick = np.zeros((S, S), np.int16); pick[:, 16:] = 1
        pick[:8, 28:] = 2                                                               # a third stop in a corner, far from the 0 | 1 boundary
        ti = np.arange(S * S)
        rows, other, w = DU.feather_plan(pick, ti, np.ones(S * S, bool))
        col = {0: 100.0, 1: 200.0, 2: 50.0}
        mine = pick.ravel()[ti[rows]]
        got = (1 - w) * np.vectorize(col.get)(mine) + w * np.vectorize(col.get)(other)
        at = lambda y, x: float(got[list(zip(*np.divmod(ti[rows], S))).index((y, x))])
        self.assertAlmostEqual(at(10, 15), 150.0, places=3)                             # either side of the boundary: 50 / 50
        self.assertAlmostEqual(at(10, 16), 150.0, places=3)
        self.assertLess(at(10, 13), at(10, 14)); self.assertLess(at(10, 14), at(10, 15))   # the share of the other stop rises toward the boundary
        self.assertNotIn((10, 12), set(zip(*np.divmod(ti[rows], S))))                   # four texels out: the stop's own picture alone
        self.assertEqual(len(rows), len(set(rows.tolist())))                            # one other stop per texel: never more than two blended
        self.assertTrue((w <= 0.5 + 1e-6).all())
        # nothing eligible, nothing blended
        self.assertEqual(len(DU.feather_plan(pick, ti, np.zeros(S * S, bool))[0]), 0)

    def test_the_roof_comes_off_at_the_plans_cut(self):
        import dollhouse as DH
        self.assertEqual(DH.roof_cut(3.4), 2.6)                                         # a high shop ceiling: the plan's 2.6 m
        self.assertEqual(DH.roof_cut(2.5), 2.3)                                         # a low room: just under its ceiling
        self.assertEqual(DH.roof_cut(None), 2.6)
        self.assertEqual(DH.ROOF_CUT_MAX, 2.6)


class Plan(unittest.TestCase):
    def test_top_down_sees_the_floor_and_cuts_the_roof(self):
        import open3d as o3d
        # a room: floor 6 x 4 at z = 0, a roof at 3, a crate 1 x 1 x 1 at (1, 1)
        floor = o3d.geometry.TriangleMesh.create_box(6, 4, 0.02)
        roof = o3d.geometry.TriangleMesh.create_box(6, 4, 0.02).translate((0, 0, 3.0))
        crate = o3d.geometry.TriangleMesh.create_box(1, 1, 1).translate((1, 1, 0.02))
        m = floor + roof + crate
        V, T = np.asarray(m.vertices), np.asarray(m.triangles)
        img, (lo, hi), got = FP.top_down(V, T, None, None, np.eye(2), cut=2.0, px_per_m=50)
        self.assertIsNotNone(img)
        inside = got[int((hi[1] - 3.9) * 50):int((hi[1] - 0.1) * 50), int((0.1 - lo[0]) * 50):int((5.9 - lo[0]) * 50)]   # the room itself (the frame has a paper margin)
        self.assertGreater(inside.mean(), 0.99)
        # the crate stands out from the floor in grey level (its top is at 1 m, the floor at 0), the roof is gone (nothing at 3 m shows)
        H, W = got.shape
        cx = int((1.5 - lo[0]) * 50); cy = int((hi[1] - 1.5) * 50); fx = int((4.5 - lo[0]) * 50); fy = int((hi[1] - 3.0) * 50)
        self.assertGreater(int(img[cy, cx, 0]), int(img[fy, fx, 0]) + 30)

    def test_ft_in(self):
        self.assertEqual(FP.ft_in(0.3048), "1' 0\"")
        self.assertEqual(FP.ft_in(17.68), "58' 0\"")


if __name__ == "__main__":
    unittest.main()


class Levelling(unittest.TestCase):
    """level_panos: the skyline levels a pano where there is sky (a beach has no vertical edge - the edge vote hit its clamp on every shore stop)."""
    @staticmethod
    def tilted_labels(tx, tz, W=1024, H=512, hill=None):
        """ADE20K-like labels: sky (2) above a horizon tilted by (tx, tz) degrees, sea (26) below it; an optional hill (14) rising above the horizon
        over a band of azimuths."""
        u = LP.mk_up(tx, tz)
        v, x = np.mgrid[0:H, 0:W].astype(np.float32)
        th = (x + 0.5) / W * 2 * np.pi - np.pi; ph = np.pi / 2 - (v + 0.5) / H * np.pi
        p = np.stack([np.cos(ph) * np.sin(th), np.sin(ph), np.cos(ph) * np.cos(th)], -1)
        e = p @ u
        lab = np.where(e > 0, 2, 26).astype(np.int16)
        if hill:
            a0, a1, h = hill
            band = (th > np.radians(a0)) & (th < np.radians(a1)) & (e > 0) & (e < np.sin(np.radians(h)))
            lab[band] = 14
        return lab

    def test_the_skyline_finds_a_tilted_horizon_to_a_degree(self):
        for tx, tz in [(0, 0), (12, -7), (-25, 18), (31, 0)]:
            u, score, share = LP.horizon_up(self.tilted_labels(tx, tz))
            want = LP.mk_up(tx, tz)
            err = math.degrees(math.acos(float(np.clip(np.dot(u, want), -1, 1))))
            self.assertLess(err, 1.0, (tx, tz, err))
            self.assertGreater(score, LP.HORIZON_MIN)

    def test_a_hill_above_the_horizon_does_not_pull_it(self):
        u, score, share = LP.horizon_up(self.tilted_labels(8, 5, hill=(-60, 60, 20)))
        err = math.degrees(math.acos(float(np.clip(np.dot(u, LP.mk_up(8, 5)), -1, 1))))
        self.assertLess(err, 1.0, err)

class Fusion(unittest.TestCase):
    """dollhouse.agreement: a stop's depth checked by its neighbours' points - confirmed keeps its weight, a floating sheet nobody else sees stands alone,
    and a neighbour seeing PAST it is not a denial (parallax sees between joists and through shelving)."""
    def test_a_floor_two_stops_see_is_confirmed_and_a_floating_sheet_stands_alone(self):
        import torch, dollhouse as D
        dev = torch.device("cpu"); h = 2.0
        W, H = 512, 256
        az = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi; el = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
        A_, E_ = np.meshgrid(az, el); dy = -np.sin(E_)
        rng = np.where(dy > 0.05, h / np.maximum(dy, 0.05), 0).astype(np.float32)             # a flat floor h below the camera, all round
        nodeB = dict(x=3.0, y=0.0, H=0.0, cam_h=h, z=0.0)
        X = torch.from_numpy(D.stop_cloud(rng, nodeB, [0, 1, 0], True, step=1))
        n = 256; d, f = D.face_rays(n=n)
        nodeA = dict(x=0.0, y=0.0, H=0.0, cam_h=h, z=0.0)
        fw, rt, upv = (np.array(v, float) for v in D.FACES["down"])
        dl = d[..., 0:1] * rt + d[..., 1:2] * upv + d[..., 2:3] * fw
        down = -dl[..., 1]; r = np.where(down > 0.05, h / np.maximum(down, 0.05), 0); zd = (r * d[..., 2]).astype(np.float32)
        M = D.level_to_world(0.0); pos = np.array([0.0, 0.0, D.cam_z(nodeA)])
        c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = M @ rt, -(M @ upv), M @ fw, pos
        fac, share = D.agreement(zd, c2w, f, [X], dev)
        self.assertGreater(share, 0.9, share)
        zd2 = zd.copy(); zd2[100:140, 100:140] *= 0.5                                           # a sheet floating halfway between camera and floor
        fac2, _ = D.agreement(zd2, c2w, f, [X], dev)
        self.assertAlmostEqual(float(np.median(fac2[104:136, 104:136])), D.AGREE_ALONE, places=2)
        self.assertGreater(float(np.median(fac2[20:60, 20:60])), 0.99)
