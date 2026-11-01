"""Look at a stop the way a person would, and read off directions: a flat (rectilinear) view of a panorama with a pixel grid, and the conversion
from a pixel of that view to the azimuth/elevation of the panorama that qa/grade_measure.py picks points with.

  python qa/persp.py <walk folder> <stop id> <yaw> <pitch> [fov=70] [out.png]      # writes the view (1000 px, grid every 100 px)
"""
import json, math, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from depth_panos import view_rays, cut_view          # noqa: E402
from imutil import imread                            # noqa: E402


def pixel_to_azel(px, py, yaw, pitch, fov, n=1000):
    """(azimuth, elevation) in degrees of the panorama (azimuth 0 = picture centre, clockwise) for pixel (px, py) of the view made by view()."""
    r = view_rays(yaw, pitch, fov, n)[int(py), int(px)]
    return math.degrees(math.atan2(r[0], r[2])), math.degrees(math.asin(max(-1, min(1, r[1]))))


def view(walk, sid, yaw, pitch, fov=70.0, n=1000, grid=True, src="points"):
    pano = imread(os.path.join(walk, "frames", src, sid + ".jpg"))
    v = cut_view(pano, dict(yaw=float(yaw), pitch=float(pitch), fov=float(fov)), n)
    if grid:
        for k in range(0, n, 100):
            cv2.line(v, (k, 0), (k, n), (0, 255, 255), 1); cv2.line(v, (0, k), (n, k), (0, 255, 255), 1)
            cv2.putText(v, str(k), (k + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1); cv2.putText(v, str(k), (2, k + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return v


if __name__ == "__main__":
    walk, sid, yaw, pitch = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    fov = float(sys.argv[5]) if len(sys.argv) > 5 else 70.0
    out = sys.argv[6] if len(sys.argv) > 6 else "view.png"
    cv2.imwrite(out, view(walk, sid, yaw, pitch, fov))
    print(out)
