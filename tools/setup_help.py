"""The decisions setup.bat and setup.sh share, in one place (standard library only, so it runs before anything is installed).
  python tools/setup_help.py pyok           exit 0 when this Python is 3.11 to 3.13
  python tools/setup_help.py torch-tag      prints cu130 (an NVIDIA card with driver 580 or newer), cpu, or pypi (macOS); says why on stderr
  python tools/setup_help.py torch-ok TAG   exit 0 when torch and torchvision are installed at requirements.txt's pins, built for TAG
  python tools/setup_help.py torch-pins     prints "torch==X torchvision==Y" from requirements.txt
  python tools/setup_help.py reqs-current   exit 0 when requirements.txt has not changed since the last install here
  python tools/setup_help.py reqs-stamp     remember that requirements.txt is installed
"""
import hashlib, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQ = os.path.join(HERE, "requirements.txt")
STAMP = os.path.join(HERE, ".venv", "shop-tour-requirements.sha256")
MIN_DRIVER = 580                      # the pinned torch is the CUDA 13.0 build, which needs NVIDIA driver R580 or newer


def pins():
    out = {}
    with open(REQ, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    for line in lines:
        m = re.match(r"^(torch|torchvision)==([0-9.]+)", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def driver_major(smi_output):
    m = re.search(r"(\d+)\.", smi_output or "")
    return int(m.group(1)) if m else None


def torch_tag(platform=sys.platform, smi=None):
    if platform == "darwin":
        print("macOS: no NVIDIA graphics card, so the CPU build (slow: a tour takes well over an hour).", file=sys.stderr)
        return "pypi"
    if smi is None and shutil.which("nvidia-smi"):
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            smi = ""
    if not smi:
        print("No NVIDIA graphics card found: using the CPU build. It works, but a tour takes about 1.5 hours instead of about 10 minutes.", file=sys.stderr)
        return "cpu"
    d = driver_major(smi)
    if d is None or d < MIN_DRIVER:
        print(f"An NVIDIA graphics card was found, but its driver ({d or 'unknown'}) is older than {MIN_DRIVER}, which the fast GPU build needs. "
              "Update it at https://www.nvidia.com/drivers and run setup again. Using the slow CPU build for now.", file=sys.stderr)
        return "cpu"
    print(f"NVIDIA graphics card found (driver {d}): using the fast GPU build.", file=sys.stderr)
    return "cu130"


def torch_ok(tag):
    from importlib import metadata
    p = pins()
    try:
        tv, vv = metadata.version("torch"), metadata.version("torchvision")
    except metadata.PackageNotFoundError:
        return False
    if tag == "pypi":
        return tv.split("+")[0] == p["torch"] and vv.split("+")[0] == p["torchvision"]
    return tv == f"{p['torch']}+{tag}" and vv == f"{p['torchvision']}+{tag}"


def req_sum():
    with open(REQ, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "pyok":
        return 0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1
    if cmd == "torch-tag":
        print(torch_tag())
        return 0
    if cmd == "torch-ok":
        return 0 if torch_ok(argv[1]) else 1
    if cmd == "torch-pins":
        p = pins()
        print(f"torch=={p['torch']} torchvision=={p['torchvision']}")
        return 0
    if cmd == "reqs-current":
        try:
            with open(STAMP, encoding="utf-8") as fh:
                return 0 if fh.read().strip() == req_sum() else 1
        except OSError:
            return 1
    if cmd == "reqs-stamp":
        os.makedirs(os.path.dirname(STAMP), exist_ok=True)
        with open(STAMP, "w", encoding="utf-8") as fh:
            fh.write(req_sum())
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
