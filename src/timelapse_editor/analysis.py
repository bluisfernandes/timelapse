"""Per-video analysis: values every 0.1 s (movement, and later brightness, sharpness...) and a cache.

The video is read once for everything that comes out of the same pass (thumbnails and the
movement series). The results are kept in a cache folder per video, in the user's cache, so
the next time the same video is opened nothing is read again.
"""
import hashlib
import json
import os
import subprocess
import tempfile

STEP = 0.1              # seconds per value of the graphs (the same step as the audio levels)
THUMB_FPS = 4           # thumbnails per second of video
MOTION_W, MOTION_H = 160, 90   # small frames: enough to see movement, fast to read

# bump a version when the way a series is computed changes: old cached values are then ignored
SERIES_VERSION = {"motion": 1}


def cache_root():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "timelapse-editor")


def fingerprint(path):
    """Identifies the video file: path, size and modification time."""
    st = os.stat(path)
    key = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def entry_dir(path):
    """Cache folder of the video. If the cache cannot be written, a temporary folder is used."""
    try:
        d = os.path.join(cache_root(), fingerprint(path))
        os.makedirs(d, exist_ok=True)
        return d
    except OSError:
        return tempfile.mkdtemp(prefix="tle_")


def read_store(d):
    try:
        with open(os.path.join(d, "series.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_store(d, store):
    tmp = os.path.join(d, "series.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f)
    os.replace(tmp, os.path.join(d, "series.json"))


def run_pass(path, duration, thumbs_dir=None):
    """One decode of the video. Returns (movement values, ok).

    Movement: average luma difference between frames 0.1 s apart (0 to 1); still parts are 0.
    The frames are taken at 10 fps, so the index matches the audio levels. With thumbs_dir, the
    thumbnails are written in the same pass.
    """
    motion = f"fps=10,scale={MOTION_W}:{MOTION_H},tblend=all_mode=difference,setpts=PTS-STARTPTS"
    if thumbs_dir:
        fc = f"[0:v]split=2[mo][th];[th]fps={THUMB_FPS},scale=320:-2[tout];[mo]{motion}[mout]"
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-an", "-filter_complex", fc,
               "-map", "[tout]", "-q:v", "4", os.path.join(thumbs_dir, "t%06d.jpg"),
               "-map", "[mout]", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    else:
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-an", "-vf", motion,
               "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    size = MOTION_W * MOTION_H
    # the gray output keeps the limited luma range (16 to 235) stretched to 0-255: 219/255 undoes it
    scale = (219 / 255) / 255 / size
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = int(duration / STEP) + 1
    levels = [0.0]   # the first output frame is the difference for 0.1 s, so frame i goes to step i + 1
    while len(levels) < n:
        frame = p.stdout.read(size)
        if len(frame) < size:
            break
        levels.append(round(sum(frame) * scale, 4))
    p.stdout.close()
    ok = p.wait() == 0
    return levels + [0.0] * (n - len(levels)), ok


def analyze(path, duration, d, thumbs=True):
    """Series of the video, from the cache when it is there. d is the cache folder of the video.
    Returns the movement values. The thumbnails are made in the same pass when they are missing."""
    store = read_store(d)
    need_motion = store.get("motion", {}).get("version") != SERIES_VERSION["motion"]
    thumbs_done = os.path.join(d, "thumbs.done")
    need_thumbs = thumbs and not os.path.exists(thumbs_done)
    if need_motion or need_thumbs:
        levels, ok = run_pass(path, duration, d if need_thumbs else None)
        if need_motion:
            store["motion"] = {"version": SERIES_VERSION["motion"], "values": levels}
            write_store(d, store)
        if need_thumbs and ok:
            open(thumbs_done, "w").close()
    return store["motion"]["values"]
