"""Camera movement and object movement, from the picture. Needs OpenCV (optional: uv sync --extra camera).

Points of the picture are followed from one frame to the next (optical flow, every 0.1 s).
  camera  how far the whole picture moved, in pixels: the median of the movement of the points.
          The median is not moved by a few points that move on their own.
  object  how much the points moved differently from the picture, in pixels (the average of
          the difference, after a small margin for noise).
"""
import subprocess

import numpy as np

from .analysis import STEP

W, H = 320, 180      # frames used here: enough detail to follow the picture, fast to compute
MARGIN = 0.5         # pixels: smaller differences are noise


def available():
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


def camera_pass(path, duration):
    """One decode of the video. Returns (camera values, object values), in pixels per 0.1 s."""
    import cv2

    cmd = ["ffmpeg", "-v", "error", "-i", path, "-an",
           "-vf", f"fps=10,scale={W}:{H},format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = W * H
    n = int(duration / STEP) + 1
    camera, obj = [0.0], [0.0]
    prev = None
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        cur = np.frombuffer(buf, np.uint8).reshape(H, W).copy()
        if prev is not None:
            cam, ob = _frame_motion(cv2, prev, cur)
            camera.append(round(cam, 3))
            obj.append(round(ob, 3))
        prev = cur
    p.stdout.close()
    p.wait()
    pad = lambda v: v[:n] + [0.0] * (n - len(v))   # noqa: E731 - exactly n values
    return pad(camera), pad(obj)


def _frame_motion(cv2, prev, cur):
    """Camera and object movement between two frames (pixels)."""
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=300, qualityLevel=0.01, minDistance=5)
    if pts is None or len(pts) < 8:
        return 0.0, 0.0
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(15, 15), maxLevel=2)
    ok = status.reshape(-1) == 1
    if ok.sum() < 8:
        return 0.0, 0.0
    flow = (nxt - pts).reshape(-1, 2)[ok]
    background = np.median(flow, axis=0)               # the picture: most points agree on it
    camera = float(np.hypot(*background))
    residual = np.linalg.norm(flow - background, axis=1)
    obj = float(np.maximum(residual - MARGIN, 0).mean())
    return camera, obj
