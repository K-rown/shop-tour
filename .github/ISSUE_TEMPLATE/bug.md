---
name: Bug report
about: Something failed or gave a wrong result
labels: bug
---

**Please do not attach video, panoramas, screenshots of a tour, or anything from `frames/` other than the two JSON files below. They can show people and places.**

**What happened / what did you expect**

**Command you ran** (e.g. `python tour.py D:/walks/place/2026-01-01/walk.insv --serve`)

**The step it stopped at, and the last lines of its output**

**Your machine:** OS, Python version, GPU and VRAM (or CPU only), RAM, `torch`/`torchvision` versions (`python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"`), camera model

**Attach (text files only):**
- `frames/timings.json`
- `frames/privacy.json` (what the privacy gate decided for each stop)
- optionally `tour/manifest.json`

For a *privacy* problem (a person, face, plate or sign that should have been held or blurred and was not), describe it in words (which stop number, what it was, how far away); do not attach the picture. If we need it we will ask for a way to see it privately.
