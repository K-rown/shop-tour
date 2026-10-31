"""Same recording in, same map out.

COLMAP's defaults are not repeatable: the mapper picks its starting pair and its RANSAC samples from a clock-seeded random number generator
(pycolmap random_seed = -1) and, with several threads, adds up its sums in whatever order the threads finish. Measured on one walk (2026-10-20):
three runs from pixel-identical pictures put the metric scale at 1.78 / 1.71 / 1.75 and the camera height of a stop at 2.34 - 2.81 m, and
the privacy gate - which sizes the person-shaped leftover under the camera in metres from that height - held 13 / 12 / 11 stops of the same
recording. Everything after the maps (capturer, privacy gate, with the same poses.json) is repeatable to the last digit.

So every COLMAP call of this engine takes its seed and its thread count from here:
  SHOP_TOUR_COLMAP_DET=0        old behaviour (clock seed, all cores), for an experiment
  SHOP_TOUR_COLMAP_THREADS=N    threads for the mapper (default 1 when deterministic: threads change the order of floating point sums)
  SHOP_TOUR_COLMAP_MATCH_PROCS=N  matching runs in N independent single-thread processes (default 8; 1 = one process). Measured on one walk (1,372 pairs):
                                N = 8 and N = 12 give database rows byte-identical to one single-thread process - a pair's RANSAC samples depend on the seed
                                and the pair, not on which process or in what order it is verified - but only while the file keeps each pair's direction.
"""
import os
import numpy as np
import shutil
import subprocess
import sys

SEED = 7


def deterministic():
    return os.environ.get("SHOP_TOUR_COLMAP_DET", "1").strip().lower() not in ("0", "false", "no", "off")


def threads():
    try:
        n = int(os.environ.get("SHOP_TOUR_COLMAP_THREADS", "0"))
    except ValueError:
        n = 0
    return n if n > 0 else (1 if deterministic() else -1)


def _env_int(name):
    try:
        return int(os.environ.get(name, "0"))
    except ValueError:
        return 0


def io_threads():
    """Threads for feature extraction (SHOP_TOUR_COLMAP_IO_THREADS; default: the cores, up to 8). The features of a picture do not depend on the
    thread count; only the numbering of the pictures does, and extract() puts that right afterwards."""
    n = _env_int("SHOP_TOUR_COLMAP_IO_THREADS")
    return n if n > 0 else (min(os.cpu_count() or 1, 8) if deterministic() else -1)


def match_threads():
    """Threads for matching (SHOP_TOUR_COLMAP_MATCH_THREADS; default 1). The geometric verification of a pair draws its RANSAC samples from a per-thread
    generator that the seed does not reach, so with several threads 5 of 1,372 pairs of one walk got different inlier counts from run to run
    (measured), and that was enough to move the map."""
    n = _env_int("SHOP_TOUR_COLMAP_MATCH_THREADS")
    return n if n > 0 else threads()


PROCS = 8
MIN_PAIRS = 64          # fewer pairs than this are matched in the one process: starting eight pythons costs more than the matching


def match_procs():
    n = _env_int("SHOP_TOUR_COLMAP_MATCH_PROCS")
    return n if n > 0 else PROCS


def start(pycolmap):
    """Call once after importing pycolmap: seeds COLMAP's own generator."""
    if deterministic():
        pycolmap.set_random_seed(SEED)


def extraction_options(pycolmap, features, first_octave=None):
    eo = pycolmap.FeatureExtractionOptions(); eo.sift.max_num_features = features
    if first_octave is not None:
        eo.sift.first_octave = first_octave          # COLMAP's default -1 doubles the picture first: ~4x the time for the finest features
    if deterministic():
        eo.num_threads = io_threads()
    return eo


def matching_options(pycolmap):
    mo = pycolmap.FeatureMatchingOptions()
    if deterministic():
        mo.num_threads = match_threads()
    return mo


def verification_options(pycolmap):
    vo = pycolmap.TwoViewGeometryOptions()
    if deterministic():
        vo.ransac.random_seed = SEED
    return vo


def mapper_options(pycolmap):
    o = pycolmap.IncrementalPipelineOptions()
    if deterministic():
        o.random_seed = SEED; o.num_threads = threads()
        o.mapper.random_seed = SEED; o.mapper.num_threads = threads()
    return o


def global_options(pycolmap, n_threads=None, max_tracks=None):
    """The global mapper (rotation averaging, then positions, then bundle adjustment: no growing one picture at a time, so ~3x faster on a walk's 176
    pictures). Seeded and single-thread like the incremental one unless the caller asks for more (n_threads) - the walk-wide map of vio_map.py does
    (see vio_fast). max_tracks: the positioning and the first bundle adjustments work on at most this many of the picture-to-picture tracks (the final
    retriangulation still makes the points from all the matches)."""
    o = pycolmap.GlobalPipelineOptions()
    if max_tracks:
        o.mapper.keep_max_num_tracks = int(max_tracks)
    if deterministic():
        n = n_threads or threads()
        o.random_seed = SEED; o.num_threads = n
        m = o.mapper
        m.random_seed = SEED; m.num_threads = n
        m.rotation_averaging.random_seed = SEED
        m.global_positioning.random_seed = SEED
        m.retriangulation.random_seed = SEED
        m.bundle_adjustment.ceres.solver_options.num_threads = n
    return o


def vio_fast():
    """The walk-wide picture map (vio_map.py) with its fast settings, SHOP_TOUR_VIO_FAST=1 (default where the privacy path reads nothing from the map - see _speed_default):
    SIFT from the picture's own size (first octave 0, not COLMAP's -1 that doubles it first: 29 s instead of 122 s for 176 pictures, 3,400 instead of 4,900 features each),
    and the global mapper on 8 threads and on at most 30,000 tracks (49,000 on one thread: 178 s; the positioning alone was 70 s and its bundle adjustments 66 s).
    Measured on walk 1 (unloaded): features + matching + map 315 s -> ~100 s; against the same pictures mapped the old way the stops moved 3.6 cm (median, max 7 cm) and the
    metric scale 0.7 %, 25 of 25 stops and every link still placed. That is the size of the map's own noise (any change of the pictures moves the scale by 0.4 - 1.4 %).
    Not bit for bit repeatable with 8 threads (two runs: camera centres 0.2 mm apart); SHOP_TOUR_VIO_THREADS=1 makes it so, SHOP_TOUR_VIO_FAST=0 is the old behaviour
    (SHOP_TOUR_VIO_FIRST_OCTAVE, _MAX_TRACKS and _THREADS set each piece by itself)."""
    return os.environ.get("SHOP_TOUR_VIO_FAST", _speed_default("0", "1")).strip().lower() in ("1", "true", "yes", "on")


def use_global():
    """SHOP_TOUR_MAPPER=incremental is COLMAP's incremental mapper (the default); SHOP_TOUR_MAPPER=global tries the global one first, with the incremental one as the fallback when it
    places too few pictures."""
    return os.environ.get("SHOP_TOUR_MAPPER", _speed_default("incremental", "global")).strip().lower() == "global"


def canonical_ids(db_path):
    """With several threads COLMAP numbers the pictures (and their frames) in the order the threads happen to finish, and the mapper's choices depend on that
    numbering. The features themselves do not depend on the thread count (checked: identical keypoints and descriptors per picture), so after the extraction
    the pictures are renumbered 1..N in the order of their NAMES. Only the extraction has run at this point, so only these tables hold ids. Raises
    (and leaves the database untouched) if it does not look the way COLMAP 4 writes it, or if any other table that holds picture or frame ids (pose_priors...) has
    rows: those would be left pointing at the old numbers."""
    import sqlite3
    c = sqlite3.connect(db_path)
    try:
        imgs = c.execute("select image_id, name from images order by name").fetchall()
        if c.execute("select count(*) from matches").fetchone()[0] or c.execute("select count(*) from two_view_geometries").fetchone()[0]:
            raise RuntimeError("matches already in the database")
        fd = c.execute("select frame_id, data_id from frame_data").fetchall()
        if len(fd) != len(imgs) or len({f for f, _ in fd}) != len(fd) or {d for _, d in fd} != {i for i, _ in imgs} or c.execute("select count(*) from frames").fetchone()[0] != len(fd):
            raise RuntimeError("frames and pictures are not one to one")
        o2n = {old: new for new, (old, _) in enumerate(imgs, 1)}
        f2n = {f: o2n[d] for f, d in fd}
        if all(o == n for o, n in o2n.items()) and all(o == n for o, n in f2n.items()):
            return False
        ours = {"images", "keypoints", "descriptors", "frames", "frame_data", "matches", "two_view_geometries"}
        for (t,) in c.execute("select name from sqlite_master where type = 'table' and name not like 'sqlite_%'").fetchall():
            if t in ours:
                continue
            cols = [r[1] for r in c.execute(f'pragma table_info("{t}")').fetchall()]
            ref = [x for x in ("image_id", "frame_id") if x in cols]
            if ref and c.execute(f'select count(*) from "{t}"').fetchone()[0]:
                raise RuntimeError(f"table {t} holds {', '.join(ref)} values and would be left pointing at the old numbers")
        off = 1 << 30
        c.execute("pragma foreign_keys = off")
        c.execute("begin")
        c.execute("create temp table m_img(old integer primary key, new integer)"); c.execute("create temp table m_frame(old integer primary key, new integer)")
        c.executemany("insert into m_img values (?, ?)", list(o2n.items())); c.executemany("insert into m_frame values (?, ?)", list(f2n.items()))
        for t, col in (("images", "image_id"), ("keypoints", "image_id"), ("descriptors", "image_id"), ("frame_data", "data_id")):
            c.execute(f"update {t} set {col} = {col} + {off}")
            c.execute(f"update {t} set {col} = (select new from m_img where old = {col} - {off})")
        for t in ("frames", "frame_data"):
            c.execute(f"update {t} set frame_id = frame_id + {off}")
            c.execute(f"update {t} set frame_id = (select new from m_frame where old = frame_id - {off})")
        c.execute("commit")
        return True
    except Exception:
        try:
            c.execute("rollback")
        except Exception:
            pass
        raise
    finally:
        c.close()


def extract(pycolmap, db, img_dir, ro, features, first_octave=None):
    """pycolmap.extract_features with the options of this module; with several threads the pictures are renumbered by name afterwards (canonical_ids)."""
    for k in [k for k in _DESC if k[0] == db]:            # a database written again in this process: its cached descriptors are the old picture's
        del _DESC[k]
    eo = extraction_options(pycolmap, features, first_octave)
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro, extraction_options=eo)
    if deterministic() and io_threads() != 1:
        canonical_ids(db)


def extract_names(pycolmap, db, img_dir, names, ro, eo, same_camera=False):
    """The features of the named pictures of img_dir, added to the database (a walk's pictures made in batches while they are still being written: vio_map.Overlap).
    same_camera: the pictures use the camera the first batch made (id 1), as one pass over the folder gives them one. The caller renumbers the pictures by name at the end
    (canonical_ids)."""
    if same_camera:
        ro.existing_camera_id = 1
    pycolmap.extract_features(db, img_dir, image_names=sorted(names), camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro, extraction_options=eo)


def _db_names(db_path):
    import sqlite3
    c = sqlite3.connect(db_path)
    try:
        return [r[0] for r in c.execute("select name from images order by name")]
    finally:
        c.close()


def read_pairs(path):
    """The pair list file as a list of (a, b), in file order, each pair once. The way round a pair is written is kept (COLMAP matches a -> b), so the
    split below depends on nothing but the file, and a one-process run of the same file gives the same rows."""
    out, seen = [], set()
    with open(path) as fh:
        for line in fh:
            w = line.split()
            if len(w) >= 2 and w[0] != w[1] and frozenset(w[:2]) not in seen:
                seen.add(frozenset(w[:2])); out.append((w[0], w[1]))
    return out


def _write_pairs(path, pairs):
    with open(path, "w") as fh:
        for a, b in pairs:
            fh.write(f"{a} {b}\n")


def _match_one(pycolmap, db, pair_file):
    pycolmap.match_image_pairs(db, matching_options=matching_options(pycolmap), pairing_options=pycolmap.ImportedPairingOptions(match_list_path=pair_file),
                               verification_options=verification_options(pycolmap))


def _merge_chunk(main, chunk):
    import sqlite3
    c = sqlite3.connect(main)
    try:
        c.execute("attach database ? as ch", (chunk,))
        c.execute("begin")
        for t in ("matches", "two_view_geometries"):
            cols = [r[1] for r in c.execute(f"pragma main.table_info({t})")]
            cl = ", ".join(cols)
            c.execute(f"insert into main.{t} ({cl}) select {cl} from ch.{t} order by pair_id")
        c.execute("commit")
    finally:
        c.close()


# ---- fast matching (the descriptor step done here, the geometric verification by COLMAP) ----
# COLMAP's CPU matcher (a FLANN index per picture, single thread) costs ~1.4 s a pair of 4096 features. The same decisions - the best two matches by angle,
# the ratio test, max_distance, the mutual check - are a float32 matrix product here: ~0.2 s a pair. There is no random number in it, so the matches depend
# on the pair alone (not on the process, the order, or the number of cores); COLMAP then verifies them (pycolmap.verify_matches) with the seed as before.
# OFF by default unless the privacy path is vio-free (see _speed_default): its matches differ a little from COLMAP's (the map moves ~2%, and the privacy held list of the test walk changed: 4 stops it holds
# today were not held), so the gate's answers change with it. SHOP_TOUR_COLMAP_MATCH=fast turns it on.

def fast_match():
    return os.environ.get("SHOP_TOUR_COLMAP_MATCH", _speed_default("colmap", "fast")).strip().lower() == "fast"


def _speed_default(slow, fast):
    """The speed paths (fast matching, the global mapper) move the map, and with it everything the privacy check used to read from the map. They are therefore ON
    by default only where the privacy path reads nothing from the map (SHOP_TOUR_PRIVACY_INPUTS=own, privacy_inputs.py), where they cannot change which stops are held
    (measured: the same held list with them off and on). Otherwise they stay off, and an environment variable still turns each on."""
    import privacy_inputs
    return fast if privacy_inputs.mode() == "own" else slow


_DESC = {}


def _load_desc(db_path, ids):
    import sqlite3
    c = sqlite3.connect(db_path)
    try:
        for i in ids:
            if (db_path, i) in _DESC:
                continue
            # keyed by the DATABASE too (2026-10-28): one process maps a walk several times (map_stops' detail retry) into fresh databases whose
            # image ids repeat, and a cache keyed by id alone handed the retry the first run's descriptors — its matches named keypoints that
            # were not there, almost none verified, and a retry with more features placed 0 stops where the first pass placed 20
            r = c.execute("select rows, cols, data from descriptors where image_id=?", (i,)).fetchone()
            _DESC[(db_path, i)] = np.frombuffer(r[2], np.uint8).reshape(r[0], r[1]).astype(np.float32) if r and r[0] else np.zeros((0, 128), np.float32)
    finally:
        c.close()


BLOCK = 256


def _top2_rows(a, b):
    """For each row of a: the index of the most similar row of b, its similarity and the second best (rows of a, in blocks small enough to stay in cache)."""
    n = len(a)
    best = np.empty(n, np.int64); bv = np.empty(n, np.float32); sv = np.empty(n, np.float32)
    scale = np.float32(1.0 / (512.0 * 512.0))
    bt = np.ascontiguousarray(b.T)
    for i in range(0, n, BLOCK):
        blk = (a[i:i + BLOCK] @ bt) * scale
        j = blk.argmax(axis=1); r = np.arange(len(blk))
        bv[i:i + BLOCK] = blk[r, j]; best[i:i + BLOCK] = j
        blk[r, j] = -1.0
        sv[i:i + BLOCK] = blk.max(axis=1)
    return best, bv, sv


def match_descriptors(d1, d2, max_ratio=0.8, max_distance=0.7, cross_check=True):
    """COLMAP's SIFT matching of two descriptor sets (uint8, L1-root, scaled to 512), as an (n, 2) array of indices. Distances are angles: acos(dot / 512^2)."""
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), np.uint32)

    def ok(bv, sv):
        best = np.arccos(np.minimum(bv, 1.0)); second = np.arccos(np.minimum(sv, 1.0))
        return (best <= max_distance) & (best <= max_ratio * second)
    b1, v1, s1 = _top2_rows(d1, d2)
    keep = ok(v1, s1)
    if cross_check:
        b2, v2, s2 = _top2_rows(d2, d1)
        keep &= ok(v2, s2)[b1] & (b2[b1] == np.arange(len(d1)))
    i = np.nonzero(keep)[0]
    return np.stack([i, b1[i]], 1).astype(np.uint32)


def _gpu_match(db_path, jobs, mo):
    """The same decisions on the graphics card (torch), or None when there is none (or it is too full): one pair is two 4096 x 4096 products, a few
    milliseconds. Full float32 (no TF32), so the products are the same every run on one machine; topk breaks no ties differently from run to run."""
    if os.environ.get("SHOP_TOUR_COLMAP_MATCH_DEVICE", "").strip().lower() == "cpu":
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        torch.backends.cuda.matmul.allow_tf32 = False
        dev = torch.device("cuda")
        need = sorted({i for a, b, _ in jobs for i in (a, b)})
        _load_desc(db_path, need)
        mx, mdist, cc = mo
        scale = 1.0 / (512.0 * 512.0)
        gd = {i: torch.from_numpy(_DESC[(db_path, i)]).to(dev) for i in need}

        def ok(bv, sv):
            best = torch.acos(torch.clamp(bv * scale, max=1.0)); second = torch.acos(torch.clamp(sv * scale, max=1.0))
            return (best <= mdist) & (best <= mx * second)
        out = []
        with torch.no_grad():
            for a, b, k in jobs:
                d1, d2 = gd[a], gd[b]
                if len(d1) < 2 or len(d2) < 2:
                    out.append((k, a, b, np.zeros((0, 2), np.uint32))); continue
                sim = d1 @ d2.T
                v1, b1 = sim.topk(2, dim=1)
                keep = ok(v1[:, 0], v1[:, 1])
                if cc:
                    v2, b2 = sim.topk(2, dim=0)
                    keep &= ok(v2[0], v2[1])[b1[:, 0]] & (b2[0][b1[:, 0]] == torch.arange(len(d1), device=dev))
                i = torch.nonzero(keep).squeeze(1)
                m = torch.stack([i, b1[i, 0]], 1).cpu().numpy().astype(np.uint32)
                out.append((k, a, b, m))
        del gd
        torch.cuda.empty_cache()
        return out
    except Exception as e:                                  # no card, not enough memory: the processor does it
        print(f"  (matching on the graphics card not possible: {type(e).__name__}: {str(e)[:120]}; using the processor)", flush=True)
        return None


def _fast_chunk(args):
    db_path, items, mo = args
    out = []
    _load_desc(db_path, {i for a, b, _ in items for i in (a, b)})
    for a, b, k in items:
        out.append((k, a, b, match_descriptors(_DESC[(db_path, a)], _DESC[(db_path, b)], *mo)))
    return out


def fast_match_pairs(pycolmap, db, pair_file, work=None):
    """Match the pairs of pair_file into db (see above). Pairs go to match_procs() single-thread processes; each pair's matches depend on it alone."""
    import sqlite3
    from concurrent.futures import ProcessPoolExecutor
    pairs = read_pairs(pair_file)
    c = sqlite3.connect(db)
    try:
        ids = {n: i for i, n in c.execute("select image_id, name from images")}
    finally:
        c.close()
    mo0 = pycolmap.FeatureMatchingOptions().sift
    mo = (mo0.max_ratio, mo0.max_distance, mo0.cross_check)
    jobs = []
    for k, (a, b) in enumerate(pairs):
        ia, ib = ids[a], ids[b]
        jobs.append((min(ia, ib), max(ia, ib), k))
    jobs.sort(key=lambda t: (t[0], t[1]))                  # neighbours together: a process loads a picture's descriptors once
    res = _gpu_match(db, jobs, mo)
    n = max(1, min(match_procs(), len(jobs)))
    per = max(1, -(-len(jobs) // (n * 4)))
    chunks = [(db, jobs[i:i + per], mo) for i in range(0, len(jobs), per)]
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"                                # inherited by the worker processes: one thread each, so the product is the same on any machine
    if res is not None:
        pass
    elif n == 1:
        res = []
        for ch in chunks:
            res.extend(_fast_chunk(ch))
    else:
        res = []
        with ProcessPoolExecutor(max_workers=n) as ex:
            for r in ex.map(_fast_chunk, chunks):
                res.extend(r)
    res.sort(key=lambda t: t[0])                           # written in the pair file's order
    d = pycolmap.Database.open(db)
    try:
        for k, a, b, m in res:
            if len(m):
                d.write_matches(a, b, m)
    finally:
        d.close()
    pycolmap.verify_matches(db, pair_file, verification_options(pycolmap))


def match_pairs(pycolmap, db, pair_file, work=None):
    """Match the pairs of pair_file into db. Deterministic and fast: the pair list, in file order, is dealt out to match_procs() single-thread processes
    (pair k goes to process k % N), each with the seed, each into its own copy of the database; the rows are then put into db in the order of the
    processes. What a pair gets depends only on its process and its place in that process's list - never on scheduling or on the number of cores."""
    n = match_procs()
    if fast_match():
        return fast_match_pairs(pycolmap, db, pair_file, work)
    if not deterministic() or n <= 1:
        return _match_one(pycolmap, db, pair_file)
    pairs = read_pairs(pair_file)
    if len(pairs) < MIN_PAIRS:
        return _match_one(pycolmap, db, pair_file)
    work = work or os.path.dirname(os.path.abspath(db))
    n = min(n, len(pairs))
    chunks, procs = [], []
    try:
        for k in range(n):
            cdb = os.path.join(work, f"match_chunk{k}.db"); cl = os.path.join(work, f"match_chunk{k}.txt")
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(cdb + suffix):
                    os.remove(cdb + suffix)
            shutil.copyfile(db, cdb)
            _write_pairs(cl, pairs[k::n])
            chunks.append((cdb, cl))
        for cdb, cl in chunks:
            procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "match-chunk", cdb, cl], stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
        outs = [p.communicate()[0] for p in procs]
        for k, p in enumerate(procs):
            if p.returncode != 0:
                raise RuntimeError(f"matching process {k} failed ({p.returncode}): " + outs[k].decode("utf-8", "replace")[-400:])
        for cdb, _ in chunks:
            _merge_chunk(db, cdb)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        for cdb, cl in chunks:
            for f in (cdb, cdb + "-wal", cdb + "-shm", cl):
                try:
                    os.remove(f)
                except OSError:
                    pass


def match_exhaustive(pycolmap, db, work=None):
    """All pairs of the pictures in db, the same way (match_pairs)."""
    if not deterministic() or match_procs() <= 1:
        return pycolmap.match_exhaustive(db, matching_options=matching_options(pycolmap), verification_options=verification_options(pycolmap))
    names = _db_names(db)
    if len(names) * (len(names) - 1) // 2 < MIN_PAIRS:
        return pycolmap.match_exhaustive(db, matching_options=matching_options(pycolmap), verification_options=verification_options(pycolmap))
    work = work or os.path.dirname(os.path.abspath(db))
    lst = os.path.join(work, "pairs_all.txt")
    _write_pairs(lst, [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))])
    return match_pairs(pycolmap, db, lst, work)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "match-chunk":
        import pycolmap
        start(pycolmap)
        if not os.environ.get("SHOP_TOUR_COLMAP_LOG"):
            pycolmap.logging.minloglevel = 2
        _match_one(pycolmap, sys.argv[2], sys.argv[3])
    else:
        sys.exit("colmap_det.py is a module; the only command is: match-chunk <database> <pair list>")
