"""Measure the privacy gate against hand labels.  python qa/judge_labels.py --frames <frames> --labels <labels.json> --walk walk1

labels.json: {"labels": {"walk1/009": {"a": "me"|"none"|"other"|"unsure"}, ...}} (a person looked at the cleaned pictures without seeing the verdicts;
on walk 2 "other" is also him: his own head at the nadir). Reads <frames>/privacy.json and prints the held list against the labels:
  --must-hold DIR:ID,ID   (repeatable) stops that must be HELD in another run's frames DIR (real strangers, a camera holder left in the picture): printed every run,
                           exit 1 with --fail-on-leak when one passes. --labels/--walk may be left out when only --must-hold is wanted.
  --known DIR:ID,ID=word  (repeatable) stops with a known cause listed on their own line, counted neither as a miss nor as a win - e.g. a bus-shelter
                           POSTER of a man's face (a photograph of a person: held, one press, not chased).
  held&none    a false hold (it costs him a press)           pass&me / pass&other   a person LEFT IN a stop that goes into the tour (must be listed, never loosened)
  held&me      a legitimate hold (he is left in the picture)  held&other / held&unsure
Exit code 0 always (it is a measurement, not a test); --fail-on-leak exits 1 when a stop labelled me/other passes.
"""
import argparse, json, os, sys


def load(frames, labels, walk):
    st = json.load(open(os.path.join(frames, "privacy.json")))["stops"]
    lab = json.load(open(labels)); lab = lab.get("labels", lab)
    return st, {k.split("/", 1)[1]: v["a"] for k, v in lab.items() if k.startswith(walk + "/")}


def judge(st, lab):
    rows = {}
    for sid, a in sorted(lab.items()):
        s = st.get(sid)
        if s is None:
            continue
        held = bool(s.get("held")) and not s.get("kept")
        rows.setdefault(("held" if held else "pass") + "&" + a, []).append(sid)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames"); ap.add_argument("--labels"); ap.add_argument("--walk")
    ap.add_argument("--known", action="append", default=[], metavar="DIR:ID,ID=word")
    ap.add_argument("--must-hold", action="append", default=[], dest="must", metavar="DIR:ID,ID")
    ap.add_argument("--fail-on-leak", action="store_true"); ap.add_argument("--reasons", action="store_true")
    a = ap.parse_args(argv)
    leaked = []
    for spec in a.known:
        left, word = spec.rsplit("=", 1); d, ids = left.rsplit(":", 1)
        stk = json.load(open(os.path.join(d, "privacy.json")))["stops"]
        for sid in ids.split(","):
            s = stk.get(sid)
            print(f"  known {word} {d}/{sid}: " + ("held" if s and s.get("held") and not s.get("kept") else "passes") + "  (not counted)")
    for spec in a.must:
        d, ids = spec.rsplit(":", 1)
        stm = json.load(open(os.path.join(d, "privacy.json")))["stops"]
        for sid in ids.split(","):
            s = stm.get(sid)
            ok = bool(s) and bool(s.get("held")) and not s.get("kept")
            print(f"  must hold {d}/{sid}: " + ("held" if ok else "PASSES - A PERSON GOES INTO THE TOUR"))
            if not ok:
                leaked.append(f"{d}:{sid}")
    if not (a.frames and a.labels and a.walk):
        if a.fail_on_leak and leaked:
            sys.exit(1)
        return
    st, lab = load(a.frames, a.labels, a.walk)
    rows = judge(st, lab)
    order = ["held&none", "held&me", "held&other", "held&unsure", "pass&me", "pass&other", "pass&none", "pass&unsure"]
    print(f"{a.walk}: {len(lab)} labelled stops")
    for k in order:
        ids = rows.get(k, [])
        print(f"  {k:13s} {len(ids):2d}  {' '.join(ids)}")
        if a.reasons and k.startswith("held"):
            for sid in ids:
                print(f"       {sid}: " + " | ".join(r[:110] for r in st[sid]["reasons"]))
    leak = rows.get("pass&me", []) + rows.get("pass&other", [])
    if a.fail_on_leak and (leak or leaked):
        sys.exit(1)


if __name__ == "__main__":
    main()
