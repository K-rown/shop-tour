# Contributing

Thanks for looking. shop-tour turns one 360 walk into a walkable tour and holds back, paints out or blurs people before anything is published. Read README.md
(what it does, how to run it), docs/GUIDE.md (every step, switch and limit) and CAPTURE.md (how to record a walk) first.

## Run the tests

No footage, GPU, browser or model download needed for these (the detectors are replaced by fakes) (run from the repository root, with the packages from `requirements.txt`):

```bash
python qa/test_app.py              # 13: the upload page and the setup scripts' choices (standard library only, a stand-in engine)
python qa/test_units.py            # 161: lens model, frames, floor projection, exposure, step order, staleness, RAM rule, run lock
python qa/test_privacy_chain.py    # 11
python qa/test_false_holds.py      # 20
python qa/test_neighbour_check.py  # 11
python qa/test_remove_others.py    # 14  (about 18 minutes on a CPU alone)
python qa/test_synth_fill.py       # 13  (patch synthesis: about 42 minutes on a CPU alone)
python qa/test_pii_blur.py         # 22
python qa/test_speed14.py          # 29
python qa/test_engine_v2.py        # 16: sphere depth, registration, levelling, the UV atlas
python qa/test_heading_check.py    # 6
python qa/test_people_blur.py      # 13: the three people modes
python qa/test_walked_path.py      # 3
```

All must pass before a pull request. CI (`.github/workflows/tests.yml`) runs `qa/test_units.py` on Linux and Windows, Python 3.10 to 3.13.
`qa/qa_tour.py` (a built tour in a headless browser; `pip install -r requirements-dev.txt && python -m playwright install chromium`) and `qa/check_capturer.py`
need a walk you have built yourself.

## The privacy law

The privacy gate (`privacy_gate.py`, `privacy_margin.py`, `remove_capturer.py`, `remove_others.py`, `pii_blur.py`, `neighbour_check.py`) fails closed, and it stays that way:

1. **A change that makes the gate hold fewer stops, lowers a detector bar, or removes a hold may never go in without a labelled reason.** Label every stop of at least one
   real walk by hand without looking at the verdicts, run `python qa/judge_labels.py --frames <frames> --labels <labels.json> --walk <name>` before and after, and put both
   outputs in the pull request. Add `--must-hold <dir>:<ids>` for stops that have a person in them: those must stay held. A hold is only dropped when the labels say nobody
   is there, and the pull request says which stops and why.
2. A change that makes it hold more is fine; say what it costs (extra holds per 60 stops).
3. The blur step is never a gate and never fetches anything during a run; models are loaded by path and checked against a pinned sha256.
4. No new model unless its licence allows commercial use (Apache-2.0, MIT, BSD or similar), checked on its model card and added to `THIRD_PARTY_NOTICES.md`. No AGPL, no research-only weights.
5. Never commit footage, panoramas, thumbnails, floor plans or screenshots of a real place, or anything with a face, a plate, a screen or an address in it. `raw/`, `frames/`,
   `tour/`, `qa/shots*/` and `qa/work*/` are git-ignored for that reason; keep it that way.
6. A verdict must be repeatable: the same walk twice gives the same held list. Anything that changes randomness or thread order needs a determinism check.

## Style

Python, no formatter enforced; keep a step's script runnable on its own (`python <step>.py`), and add a unit test with any change to the maths. Speed changes need a switch
that turns them off and an output identical to the plain way.

## Reporting a bug

Use the issue template. Attach `frames/timings.json` and `frames/privacy.json` (and `tour/manifest.json` for the privacy status), never footage or pictures.
