"""Per-video analysis: values every 0.1 s and a cache.

Series (one value per 0.1 s, index i = time i * STEP):
  motion      average luma difference between frames 0.1 s apart (0 to 1); still parts are 0
  brightness  average luma of the frame (0 to 1, black to white)
  sharpness   average detail of the frame (0 to 1); a blurred or flat frame has little detail
Events:
  cuts        times (in seconds) where the picture changes all at once: scene cuts

The video is read once, in one decode, for everything together and for the thumbnails. The results
are kept in a cache folder per video, in the user's cache, so the next time the same video is
opened nothing is read again.
"""
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import threading

STEP = 0.1              # seconds per value of the graphs (the same step as the audio levels)
THUMB_FPS = 4           # thumbnails per second of video
W, H = 160, 90          # size of the frames used for the series: enough to see changes, fast to read

# bump a version when the way a series is computed changes: old cached values are then ignored
SERIES_VERSION = {"motion": 1, "brightness": 1, "sharpness": 1, "cuts": 1, "camera": 1, "object": 1}
SERIES = ("motion", "brightness", "sharpness")


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


STORE_LOCK = threading.Lock()


def write_store(d, store):
    tmp = os.path.join(d, "series.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f)
    os.replace(tmp, os.path.join(d, "series.json"))


def update_store(d, updates):
    """Adds or replaces entries of the store. Two jobs may write at the same time: they take turns."""
    with STORE_LOCK:
        store = read_store(d)
        store.update(updates)
        write_store(d, store)
        return store


def _filter_graph(thumbs):
    """One decode, several outputs. The series are stacked in one small gray frame (W x 3H):
    movement on top, then brightness, then sharpness. The frames are made small before they are
    split, so the big frames are copied as little as possible. The first frame has no previous one,
    so the brightness and sharpness branches skip it and stay in step with the movement."""
    parts = []
    if thumbs:
        parts.append("[0:v]split=2[full][th]")
        parts.append(f"[th]fps={THUMB_FPS},scale=320:-2[tout]")
        src = "[full]"
    else:
        src = "[0:v]"
    parts.append(f"{src}fps=10,scale={W}:{H},format=gray,split=3[mo][br][sh]")
    parts.append("[mo]tblend=all_mode=difference,setpts=PTS-STARTPTS[m]")
    parts.append("[br]select=gte(n\\,1),setpts=PTS-STARTPTS[b]")
    parts.append("[sh]split=2[s1][s2];[s2]boxblur=1:1[s3];"
                 "[s1][s3]blend=all_mode=difference,select=gte(n\\,1),setpts=PTS-STARTPTS[d]")
    parts.append("[m][b][d]vstack=inputs=3[v]")
    return ";".join(parts)


def run_pass(path, duration, thumbs_dir=None):
    """One decode of the video. Returns (dict of series, ok). With thumbs_dir, the thumbnails
    are written in the same pass."""
    thumbs = bool(thumbs_dir)
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-an", "-filter_complex", _filter_graph(thumbs),
           "-map", "[v]", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    if thumbs:
        cmd += ["-map", "[tout]", "-q:v", "4", os.path.join(thumbs_dir, "t%06d.jpg")]
    size = W * H
    frame_size = 3 * size
    # the gray output keeps the limited luma range (16 to 235) stretched to 0-255: 219/255 undoes it
    # for the differences; the brightness is already (Y - 16) / 219 in those units
    diff_scale = (219 / 255) / 255 / size
    bright_scale = 1 / 255 / size
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = int(duration / STEP) + 1
    motion, brightness, sharpness = [0.0], [], []
    while True:
        frame = p.stdout.read(frame_size)
        if len(frame) < frame_size:
            break
        mv = memoryview(frame)
        motion.append(round(sum(mv[:size]) * diff_scale, 4))
        brightness.append(round(sum(mv[size:2 * size]) * bright_scale, 4))
        sharpness.append(round(sum(mv[2 * size:]) * diff_scale, 4))
    p.stdout.close()
    ok = p.wait() == 0
    # brightness and sharpness start one step later: the first value repeats the second one
    brightness = ([brightness[0]] if brightness else [0.0]) + brightness
    sharpness = ([sharpness[0]] if sharpness else [0.0]) + sharpness
    series = {"motion": motion, "brightness": brightness, "sharpness": sharpness}
    return {k: _fit(v, n) for k, v in series.items()}, ok


def _fit(values, n):
    """Exactly n values: cut or padded with zeros at the end."""
    return values[:n] + [0.0] * (n - len(values))


def detect_cuts(values, window=20, ratio=3.0, min_value=0.06):
    """Times of the picture changes all at once. A cut is a peak that is much bigger than the
    movement around it (ratio), and is not too small (min_value)."""
    cuts = []
    last = None   # index of the last cut found
    for i, v in enumerate(values):
        if v < min_value:
            continue
        around = values[max(0, i - window):i + window + 1]
        if v < max(statistics.median(around), 0.01) * ratio:
            continue
        if v < max(values[max(0, i - 2):i + 3]):   # only the top of a peak
            continue
        if last is not None and i - last <= 2:     # one cut gives one peak
            continue
        cuts.append(round(i * STEP, 1))
        last = i
    return cuts


def analyze(path, duration, d, thumbs=True):
    """Series and cuts of the video, from the cache when they are there. d is the cache folder.
    Returns the store: {"motion": {...}, "brightness": {...}, "sharpness": {...}, "cuts": {...}}.
    All series are computed in the same pass, and the thumbnails too when they are missing."""
    store = read_store(d)
    current = all(store.get(k, {}).get("version") == SERIES_VERSION[k] for k in SERIES)
    thumbs_done = os.path.join(d, "thumbs.done")
    need_thumbs = thumbs and not os.path.exists(thumbs_done)
    if not current or need_thumbs:
        series, ok = run_pass(path, duration, d if need_thumbs else None)
        updates = {k: {"version": SERIES_VERSION[k], "values": series[k]} for k in SERIES}
        updates["cuts"] = {"version": SERIES_VERSION["cuts"], "values": detect_cuts(series["motion"])}
        store = update_store(d, updates)
        if need_thumbs and ok:
            open(thumbs_done, "w").close()
    return store


def values_of(store, name):
    return store.get(name, {}).get("values", [])
