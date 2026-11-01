"""Read the camera's own motion sensor (accelerometer + gyroscope) out of an Insta360 .insv file.

Usage:  python imu.py <video.insv> [--save frames/imu.npz]
No third-party parser: the format was worked out from this project's own files (clean-room, nothing downloaded or decompiled). What was found in an X5 file:
  * The file ends with a trailer (32-byte magic, a length and a version in front of it) made of records.
  * One record is a stream of 20-byte samples at about 1 kHz: a 32-bit microsecond timestamp, 4 zero bytes, then six unsigned 16-bit numbers that are
    the sensor readings with an offset of 32768 (first three: acceleration, last three: rotation rate).
  * The stream is found by looking for a 20-byte stride at which the 32-bit values count up by about 1000 each step (no table of contents needed).
Units are fixed by physics, not by the vendor: acceleration scale = the one that makes the median length of the reading equal gravity (9.80665 m/s^2); rotation
scale is checked against the video (imu_scale.py) and defaults to the common +-2000 deg/s range (16.4 LSB per deg/s).
"""
import argparse, os, sys
import numpy as np

MAGIC = b"8db42d694ccc418790edff439fe026bf"
GYRO_LSB_PER_DPS = 16.4


def _read_trailer(path, max_bytes=40 * 1024 * 1024):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, size - 40))
        tail = f.read(40)
        if not tail.endswith(MAGIC):
            raise ValueError("no Insta360 trailer at the end of the file")
        ln = int.from_bytes(tail[:4], "little")
        n = min(ln, max_bytes)
        f.seek(size - n)
        return f.read(n)


def find_stream(raw):
    """(offset of the first sample's timestamp, number of samples) of the longest 20-byte-stride stream whose first u32 counts up by ~1000 per step."""
    a = np.frombuffer(raw, np.uint8)
    n = len(a) - 3
    v = a[:n].astype(np.uint32) | (a[1:n + 1].astype(np.uint32) << 8) | (a[2:n + 2].astype(np.uint32) << 16) | (a[3:n + 3].astype(np.uint32) << 24)
    best = (0, 0, 0)
    for ph in range(20):
        s = v[ph::20].astype(np.int64)
        d = np.diff(s)
        good = (d > 600) & (d < 1500)
        if good.sum() < 1000:
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, good.astype(np.int8), 0]))
        st, en = edges[0::2], edges[1::2]
        k = int(np.argmax(en - st))
        if en[k] - st[k] > best[2]:
            best = (ph + 20 * int(st[k]), int(st[k]), int(en[k] - st[k] + 1))
    off, _, cnt = best
    if cnt < 5000:
        raise ValueError("no motion-sensor stream found in the trailer")
    return off, cnt


def read_imu(path):
    """dict: t (s, the camera's clock, strictly increasing), acc (N, 3) m/s^2, gyro (N, 3) rad/s, rate (Hz)."""
    raw = _read_trailer(path)
    off, cnt = find_stream(raw)
    a = np.frombuffer(raw, np.uint8)
    rec = np.frombuffer(a[off:off + 20 * cnt].tobytes(), np.uint8).reshape(cnt, 20)
    ts = rec[:, 0:4].copy().view("<u4").reshape(-1).astype(np.float64) * 1e-6
    vals = np.concatenate([rec[:, 8:20].copy().view("<u2").reshape(-1, 6)], axis=1).astype(np.float64) - 32768.0
    acc_raw, gyr_raw = vals[:, :3], vals[:, 3:]
    g_lsb = float(np.median(np.linalg.norm(acc_raw, axis=1)))
    acc = acc_raw * (9.80665 / g_lsb)
    gyro = np.deg2rad(gyr_raw / GYRO_LSB_PER_DPS)
    ok = np.r_[True, np.diff(ts) > 0]
    return dict(t=ts[ok], acc=acc[ok], gyro=gyro[ok], rate=float(1.0 / np.median(np.diff(ts[ok]))), acc_lsb_per_g=g_lsb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video"); ap.add_argument("--save", default=None)
    a = ap.parse_args()
    d = read_imu(a.video)
    print(f"{len(d['t'])} samples, {d['t'][-1] - d['t'][0]:.1f} s, {d['rate']:.0f} Hz, 1 g = {d['acc_lsb_per_g']:.0f} LSB")
    if a.save:
        np.savez_compressed(a.save, **d)


if __name__ == "__main__":
    main()
