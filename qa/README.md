# qa/

Everything here checks the pipeline or the built tour. Run from the repository root.

| script | what it does | needs |
|---|---|---|
| `test_units.py` | fast unit tests (lens model, frames, floor projection, nadir mask, exposure solver, step order/staleness, video checks, RAM rule, pruning) | numpy, opencv |
| `qa_tour.py [--stops 1,5,10] [--browser chromium\|webkit\|firefox]` | opens the built `tour/` in a headless browser at desktop and phone size and checks: load without errors and without internet, depth, floor cursor, tap-to-go and where it lands, keeps facing, glide time, view straight down not black, no person under the camera, exposure gains, only this tour's pictures, idle viewer does not redraw, a failed picture, rapid input. Writes `qa/report.json` and screenshots (git-ignored) | playwright (+ torch for the person check) |
| `check_nadir_person.py [--folder ...]` | segments the finished panoramas looking straight down and reports how much is still a person | torch, transformers |
| `glide_holes.py` | slows the glide and measures black holes in the frames | playwright |
| `exposure_check.py A B` | compares two builds on the brightness of the same physical points in neighbouring stops | opencv |
| `flash_test.py A B` | brightness step between consecutive stops in two builds (a weak test: content dominates) | playwright |
| `make_demo_gif.py` | makes `docs/demo.gif` from a built tour (only from a walk you may publish) | playwright, pillow |
| `make_demo_video.py` | records a ~30 s draft demo video (`docs/demo.mp4`, git-ignored) of a scripted walk through a built tour | playwright, ffmpeg |

`qa/shots*/`, `qa/report.json` and `qa/work/` hold footage-derived pictures and scratch files and are git-ignored.
