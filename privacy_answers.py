"""What survives a privacy re-scan, and how tour.py knows the privacy step finished. Pure Python, no pictures; privacy_margin.py and tour.py use it.

privacy_gate.py is not changed. Its full scan rebuilds the state of every stop: it keeps a person's "kept" answer only for a stop that still has reasons,
and it never keeps "left_out". A stop held ONLY by privacy_margin.py has no gate reasons, so after a re-scan it would look clear and go into the tour
unreviewed. So tour.py copies the previous frames/privacy.json (frames/privacy_prev.json) before the gate runs, and privacy_margin.py - the last command
of the step - restores from it:

  - left out: a stop he left out stays left out (held, not in the tour, recorded as reviewed again);
  - kept: a stop he kept stays kept only if everything the check now says about it is something he already saw when he kept it; otherwise it is held
    again for a fresh look, with a reason that says so (his old answer is not used for something he never saw);
  - margin hold: a stop the margin held stays held unless he answered it.
Nothing here ever removes a hold the gate or the margin made now, and restoring twice gives the same result.

frames/privacy_margin.json is the step's "done" marker, written as the very last act: {version, margin: "on"|"off", signature, at}. The signature covers
what the step decides (held, reasons, margin per stop) and not his answers (kept, left_out), so answering a stop with privacy_gate.py does not make the
step look unfinished. A marker that is missing or does not match, or a snapshot still lying there, means the step was cut off (a crash, a kill, a reboot).
"""
import datetime, hashlib, json, os, shutil

PREV = "privacy_prev.json"
MARKER = "privacy_margin.json"
MARGIN_PREFIX = "held to be safe:"
AGAIN_PREFIX = "held again for a fresh look: you kept this stop after looking at it, but the check now also reports something you have not seen: "
LEFT_OUT_REASON = "left out of the tour by your earlier decision (kept out when the privacy check ran again)"
MARGIN_REASON = MARGIN_PREFIX + " an earlier check held this stop (a person-sized shape that depends on a centimetre of camera height); it stays held until you have looked at it"


def _read(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def stops_of_file(path):
    d = _read(path)
    s = d.get("stops") if isinstance(d, dict) else None
    return s if isinstance(s, dict) else None


def stops_of(frames):
    """frames/privacy.json's stops, or None when it is missing, damaged or has no stops table."""
    return stops_of_file(os.path.join(frames, "privacy.json"))


def signature(stops):
    """What the privacy step decided, without his answers: held, reasons and the margin record of every stop."""
    body = {sid: [bool(s.get("held")), [str(r) for r in (s.get("reasons") or [])], s.get("margin")] for sid, s in sorted(stops.items()) if isinstance(s, dict)}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def marker_problem(frames, mode, version):
    """Why the marker does not vouch for the current privacy.json, or None. mode: "on"/"off" (the margin); version: privacy_margin.SCRIPT_VERSION."""
    stops = stops_of(frames)
    if stops is None:
        return "privacy.json missing"
    m = _read(os.path.join(frames, MARKER))
    if not isinstance(m, dict):
        return "the privacy margin has not finished (no privacy_margin.json)"
    if m.get("version") != version:
        return "the privacy margin was made by an older version of the check"
    if m.get("margin") != mode:
        return f"the privacy margin was {m.get('margin')} when this was made, and is {mode} now"
    if m.get("signature") != signature(stops):
        return "privacy.json is not the one the privacy margin finished with"
    return None


def pending(frames, mode, version):
    """Why the privacy step must (partly) run again, or None: a snapshot still lying there, or a marker that does not vouch for privacy.json."""
    if os.path.exists(os.path.join(frames, PREV)):
        return "an earlier privacy run did not finish (its snapshot of your answers is still there)"
    return marker_problem(frames, mode, version)


def write_marker(frames, mode, version, now=None):
    stops = stops_of(frames)
    if stops is None:
        raise RuntimeError("privacy.json missing: no marker to write")
    path = os.path.join(frames, MARKER)
    with open(path + ".tmp", "w") as fh:
        json.dump(dict(version=version, margin=mode, signature=signature(stops), at=(now or datetime.datetime.now()).strftime("%Y-%m-%dT%H:%M:%S")), fh, indent=1)
    os.replace(path + ".tmp", path)


def snapshot(frames, mode, version):
    """Before the gate runs: keep the previous privacy.json (his answers, the margin holds). A snapshot already lying there is KEPT while the
    step is unfinished: privacy.json may already be the gate's new result, which has none of his answers in it. Returns what was done."""
    pf, sp = os.path.join(frames, "privacy.json"), os.path.join(frames, PREV)
    if stops_of(frames) is None:
        return "no previous privacy.json"
    if os.path.exists(sp) and marker_problem(frames, mode, version):
        return "kept the snapshot of an unfinished run"
    shutil.copyfile(pf, sp + ".tmp")
    os.replace(sp + ".tmp", sp)
    return "snapshot made"


def load_snapshot(frames):
    return stops_of_file(os.path.join(frames, PREV))


def drop_snapshot(frames):
    try:
        os.remove(os.path.join(frames, PREV))
    except OSError:
        pass


def restore(prev, state):
    """Put his answers and the margin's holds from `prev` (the earlier privacy.json's stops) back on `state` (the new result), IN PLACE.
    Returns {"left_out": [...], "kept": [...], "reheld": [...], "margin": [...]}: the stops each rule touched (kept: restored as kept)."""
    rep = dict(left_out=[], kept=[], reheld=[], margin=[])
    for sid, o in (prev or {}).items():
        n = state.get(sid)
        if not isinstance(o, dict) or not isinstance(n, dict):
            continue
        if o.get("kept"):
            seen = {str(r) for r in (o.get("reasons") or [])}
            now = [str(r) for r in (n.get("reasons") or []) if not str(r).startswith(AGAIN_PREFIX)]
            if not now:
                n.pop("kept", None); n.pop("kept_on", None)          # nothing to look at now: the stop is clear on its own (a held stop always has a reason)
                continue
            new_only = [r for r in now if r not in seen]
            n["held"] = True
            if not new_only:                                         # everything the check says now is what he saw when he kept it
                n["kept"] = True
                n["kept_on"] = o.get("kept_on") or (o["kept"] if isinstance(o["kept"], str) else "earlier")
                rep["kept"].append(sid)
            else:
                n.pop("kept", None); n.pop("kept_on", None)
                n["reasons"] = now + [AGAIN_PREFIX + "; ".join(new_only)]
                rep["reheld"].append(sid)
        elif o.get("left_out"):
            n.pop("kept", None); n.pop("kept_on", None)
            n["held"] = True
            n["left_out"] = True
            n["left_out_on"] = o.get("left_out_on") or "earlier"
            if not n.get("reasons"):
                n["reasons"] = [str(r) for r in (o.get("reasons") or [])] or [LEFT_OUT_REASON]
            rep["left_out"].append(sid)
        elif o.get("held") and o.get("margin") and not n.get("held"):
            n["held"] = True
            have = [str(r) for r in (n.get("reasons") or [])]
            mine = [str(r) for r in (o.get("reasons") or []) if str(r).startswith(MARGIN_PREFIX)] or [MARGIN_REASON]
            n["reasons"] = have + [r for r in mine if r not in have]
            n["margin"] = o["margin"]
            rep["margin"].append(sid)
    return rep
