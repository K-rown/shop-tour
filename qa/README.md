# qa/

Everything here checks the pipeline or the built tour. Run from the repository root.

| script | what it does | needs |
|---|---|---|
| `test_units.py` | fast unit tests (lens model, frames, floor projection, nadir mask, exposure solver, step order/staleness, video checks, RAM rule, pruning) | numpy, opencv |
| `test_heading_check.py` | each stop's place and heading against the directions its own pictures measure to its neighbours (heading_check.py), a misturned stop taken out alone, the motion track placing a stop with its own turn when the map misplaced some of the stops it is fitted on, stops the map put on one point | numpy |
| `qa_tour.py [--stops 1,5,10] [--browser chromium\|webkit\|firefox]` | opens the built `tour/` in a headless browser at desktop and phone size and checks: load without errors and without internet, depth, floor cursor, tap-to-go and where it lands, keeps facing, glide time, view straight down not black, no person under the camera, exposure gains, only this tour's pictures, idle viewer does not redraw, a failed picture, rapid input. Writes `qa/report.json` and screenshots (git-ignored) | playwright (+ torch for the person check) |
| `test_gyro.py --tour DIR` | the viewer's Motion key on a phone-sized browser, fed synthetic `deviceorientation` events: off moves nothing, on the yaw follows the phone's turn and the pitch its tilt (no jump on the first reading, a drag still turns), no sensor in 2.5 s says so, iOS asks inside the press; the reel and the thumb cluster clear the 52 px foot strip | playwright, a built tour |
| `test_rings.py --tour DIR` | the rings on the floor: every ring on its stop's floor (camera height plus the ground's climb), none behind a wall, the walk's next/prev always, near the measured surface; desktop hover lights a ring and a click glides there; a phone press shows it pressed with a 44 px target at least; the big map's numbers never overlap | playwright, a built tour |
| `test_context_loss.py --tour DIR` | the black screen: the 3D context lost (WEBGL_lose_context) and given back, not given back (the viewer asks for it), or refused (the page says so with Tap to reload, and the key brings the picture back) | playwright, a built tour |
| `check_nadir_person.py [--folder ...]` | segments the finished panoramas looking straight down and reports how much is still a person | torch, transformers |
| `glide_probe.py --tour DIR` | glide frame times, near-black pixels mid-glide and the time to the first picture of one built tour | playwright |
| `glide_holes.py` | slows the glide and measures black holes in the frames | playwright |
| `exposure_check.py A B` | compares two builds on the brightness of the same physical points in neighbouring stops | opencv |
| `flash_test.py A B` | brightness step between consecutive stops in two builds (a weak test: content dominates) | playwright |
| `judge_labels.py --frames F --labels L.json --walk NAME [--must-hold DIR:IDS]` | the privacy gate's verdicts against labels you made by hand (see CONTRIBUTING.md) | a built `frames/` |
| `test_privacy_chain.py`, `test_false_holds.py`, `test_neighbour_check.py`, `test_remove_others.py`, `test_synth_fill.py`, `test_pii_blur.py`, `test_speed14.py` | unit tests of the privacy chain, the paint-out, the blur and the speed-ups (no footage) | torch, the models |
| `make_demo_gif.py` | makes `docs/demo.gif` from a built tour (only from a walk you may publish) | playwright, pillow |
| `make_demo_video.py` | records a ~30 s draft demo video (`docs/demo.mp4`, git-ignored) of a scripted walk through a built tour | playwright, ffmpeg |

`qa/shots*/`, `qa/report.json` and `qa/work/` hold footage-derived pictures and scratch files and are git-ignored.
