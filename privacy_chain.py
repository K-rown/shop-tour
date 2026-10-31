"""The capturer step and the privacy step in ONE process: the person holding the stick is painted out of every stop, and every stop is checked for people.

Usage:  python privacy_chain.py [--frames frames]
Does exactly what  remove_capturer.py,  privacy_gate.py  and  privacy_margin.py  do one after the other (same files, same checks, same thresholds, same
words; those three still work on their own), with these differences in HOW:
  - the models (Depth Anything, SAM 2, the person detector) are loaded ONCE, not three times, and the person detector loads while the masks are made;
  - the floor of each stop is rebuilt in separate processes (CPU) as soon as the masks it reads are there, and as soon as a stop's picture is written two threads of
    this process run the gate and the margin on it (GPU): the checks of the first stops are done while the masks and floors of the last are still being made;
  - the gate and the margin share the depth and SAM work of a picture (the margin used to work them out again);
  - begun early by tour.py (tour.EARLY: right after the levelling), the masks are made while the hops are still being worked out; a stop's floor is begun when the hops
    are done (stepsync: frames/.done/hops) and the stop's masks and its neighbours' are there. Run by hand, everything is there at once and nothing waits.
SHOP_TOUR_PEOPLE (people_mode.py) decides what happens to people. blur (the default): the person holding the stick is smudged where he stood (people_blur.py), every other person the gate finds is
smudged in the stop's picture, nothing is held, and the paint-out and the margin do not run. keep: as blur, but the other people are left in the picture as photographed.
remove: the older flow, unchanged. Unless SHOP_TOUR_REMOVE_OTHERS=0 (on by default since 2026-10-22) every OTHER person the gate's scan finds is also painted out (remove_others.py) before the gate
looks at the stop, and the gate then looks at the painted picture: a person still found holds the stop.
Writes: frames/capturer/*, frames/clean/*.jpg (as remove_capturer.py), frames/privacy.json, frames/safe/*, frames/review/review.html, frames/privacy_margin.json (as the
gate and the margin). tour.py's privacy step then finds nothing left to do.
THE BLUR RIDES ALONG (2026-10-30, speed; pii_blur.stream_wanted: SHOP_TOUR_PEOPLE keep or blur, SHOP_TOUR_PII_STREAM not 0): as soon as a stop's picture in frames/safe/
is final the pii step's own code blurs its plates, faces and printed names on a thread of this process (pii_blur.Stream, the gate's person detector shared), while
the masks of the later stops are still being made; frames/pii.json is written last, so tour.py's pii step finds itself up to date (a stop the thread could not look at
is looked at again by that step). In remove mode a stop can still be held after its check, so the pii step runs after the chain as before.
"""
import argparse, json, os, sys, threading, time

import stepsync
import privacy_answers as PA
import privacy_gate as PG
import privacy_inputs as PI
import privacy_margin as PM
import pii_blur as PB
import remove_capturer as RC
import remove_others as RO


CHECK_THREADS = max(1, int(os.environ.get("SHOP_TOUR_CHECK_THREADS", "") or 2))        # stops checked at once (the GPU work and the picture work of two stops overlap)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames")
    a = ap.parse_args()
    t_all = time.time()
    frames = a.frames
    pts = json.load(open(os.path.join(frames, "points.json")))
    UP = json.load(open(os.path.join(frames, "up.json"))); P = PI.Inputs(frames, len(pts))
    ids = [p["id"] for p in pts]; todo = list(range(len(ids)))
    cdir = os.path.join(frames, "capturer"); os.makedirs(cdir, exist_ok=True)
    os.makedirs(os.path.join(frames, "clean"), exist_ok=True)
    rep_path = os.path.join(cdir, "report.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    print("privacy snapshot:", PA.snapshot(frames, PM.mode_word(), PM.SCRIPT_VERSION), flush=True)         # what privacy_margin.py --snapshot does before the gate
    pf = os.path.join(frames, "privacy.json")
    old = json.load(open(pf)) if os.path.exists(pf) else {}
    state = old.get("stops", {})

    import capturer as C
    import torch, transformers                                     # (imported here, once, so that the two loads below do not race for them)
    box = {}

    def load_det():
        try:
            box["det"] = PG.Detectors()
        except BaseException as e:                                  # reported where it is needed
            box["det_err"] = e
    th = threading.Thread(target=load_det, daemon=True); th.start()
    t = time.time(); box["models"] = PM.Memo(C.Models()); print(f"[chain] models loaded ({time.time() - t:.1f}s)", flush=True)

    def get_det():
        th.join()
        if "det_err" in box:
            raise box["det_err"]
        return box["det"]

    others = RO.Others(frames, pts, UP, P) if RO.enabled() else None
    stream = PB.Stream(frames, pts, fd_factory=lambda: PB.Finder(det=get_det().clone()), log=lambda m: print(f"  pii {m}", flush=True)) if PB.stream_wanted() else None
    scales = PM.scales_to_try(None); found = {}
    spent = dict(gate=0.0, margin=0.0)
    mine = threading.local()

    def checked(k):
        """The gate and the margin for stop k, in one of the checking threads. Each thread has its own copy of what is not safe to share (the face detector, the
        remembered depth and SAM of the picture in hand); the models themselves are shared."""
        if not hasattr(mine, "gate"):
            import copy
            mine.models = PM.Memo(copy.copy(box["models"]._m))
            mine.gate = PG.Gate(frames, pts, state, old, det=get_det().clone(), models=mine.models, rep_cap=rep)
        t0 = time.time()
        if others is not None:                                      # unless SHOP_TOUR_REMOVE_OTHERS=0: every other person painted out of the picture before the gate looks (and looks again)
            others.paint_stop(k, mine.gate.det, mine.models)
        mine.gate.stop(k)
        t1 = time.time()
        sid = ids[k]; s = state[sid]
        if PM.enabled() and not s.get("held") and not s.get("kept"):
            found.update(PM.sweep(frames, scales, ids={sid}, models=mine.models, rep_cap=rep, panos={mine.gate.last[0]: mine.gate.last[1]}))
        spent["gate"] += t1 - t0; spent["margin"] += time.time() - t1
        if stream is not None:                                      # (keep / blur: stop k's safe picture is written once, here, and never taken back)
            stream.submit(k)
    # The floor of a stop is begun as soon as the masks it reads (its own and its neighbours') are there, so the floors are made while the masks of the later
    # stops still are; a stop's check begins when its floor is written.
    needs = {}
    ready, queued = set(), set()
    t = time.time()
    from concurrent.futures import ThreadPoolExecutor
    futs = []
    gate_lock, opened, gate_err = threading.Lock(), threading.Event(), []
    with ThreadPoolExecutor(max_workers=CHECK_THREADS) as tp:
        pool = RC.FillPool(frames, ids, rep, rep_path, on_done=lambda k: futs.append(tp.submit(checked, k)))

        def flush():                                                # (the gate's lock held)
            if gate_err:
                return
            for q in todo:
                if q not in queued and needs[q] <= ready:
                    queued.add(q); pool.submit(q)

        def on_mask(k):
            with gate_lock:
                ready.add(k)
                if opened.is_set():
                    flush()

        def open_gate():
            """The hops are there: who sees the floor under whom is known, and the floors can be made."""
            needs.update({k: {k} | {n["j"] for n in P.neighbours(k)} for k in todo})
            with gate_lock:
                opened.set(); flush()

        def watch():
            for step in ("hops",):
                if stepsync.wait_for(step, log=lambda m: print(f"[chain]{m}", flush=True)) != "ok":
                    gate_err.append(f"the {step} step did not finish"); opened.set(); return
            open_gate()
        if stepsync.enabled():                                      # begun early (tour.EARLY): the masks do not need the hops, the floors do
            threading.Thread(target=watch, daemon=True).start()
        else:
            open_gate()
        try:
            RC.mask_stage(frames, pts, UP, P, todo, rep, lambda: box["models"], get_det, on_mask=on_mask)
            json.dump(dict(rep), open(rep_path, "w"), indent=1)
            print(f"[chain] masks ({time.time() - t:.1f}s)", flush=True)
            opened.wait()
            if gate_err:
                raise SystemExit(gate_err[0])
            with gate_lock:
                for q in todo:
                    if q not in queued:
                        queued.add(q); pool.submit(q)
        finally:
            pool.close()
        for fu in list(futs):
            fu.result()                                                    # (a stop that failed to check fails the step, as it did when the gate was a step of its own)
    if others is not None:
        others.write_report()
    print(f"[chain] masks, floors and checks ({time.time() - t:.1f}s; thread-seconds in the gate {spent['gate']:.1f}, in the margin {spent['margin']:.1f})", flush=True)
    t = time.time()
    PG.finish(frames, pts, state)
    print(f"[chain] verdicts written ({time.time() - t:.1f}s)", flush=True)
    t = time.time()
    PM.main(argv=["--frames", frames], precomputed=found, models=box["models"], rep_cap=rep)
    print(f"[chain] margin applied ({time.time() - t:.1f}s)", flush=True)
    if stream is not None:
        t = time.time()
        rec = stream.close()
        body = PB.write_record(frames, rec, streamed=True)          # (after the verdicts: it adds each stop's pii entry to privacy.json)
        tt = body["totals"]
        print(f"[chain] blur beside the chain: {len(rec)} pictures, {tt['plate']} plates, {tt['face']} faces, {tt['name']} printed names, {tt['text']} other text regions "
              f"({stream.busy:.1f}s on its thread, {time.time() - t:.1f}s after the checks)" + (f"; {len(body['errors'])} not looked at: {', '.join(body['errors'])}" if body["errors"] else ""), flush=True)
    try:
        print(f"[chain] card memory: this process held at most {torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GB", flush=True)
    except Exception:
        pass
    print(f"[chain] done ({time.time() - t_all:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
