"""
Time-lapse editor with per-segment speed for Osmo Action 4 footage.

Requirements: Python 3.14+ and ffmpeg/ffprobe on PATH. Standard library only.
The interface lives in web/editor_ui.html, next to this package's modules.

WEB INTERFACE (browser):
    uv run timelapse-editor DJI_0001.MP4

COMMAND LINE (no interface):
    uv run timelapse-editor DJI_0001.MP4 --segments "0-10 4x mute; 10-20 1x; 20:00-end 8x corrected"
    uv run timelapse-editor DJI_0001.MP4 --segments-file segments.txt --output summary.mp4

Format of each segment:  START-END SPEED [mute|original|corrected]
  - START and END: seconds (10 or 10.5), m:ss (1:05.5), frame (f300) or "end".
  - SPEED: a number, with or without "x" (4, 4x, 0.5x).
  - Audio: default is "original" for 1x and "mute" for other speeds.
  - Segments are separated by ";" or line breaks. "#" starts a comment.
  - Parts without a segment are exported at 1x with the original audio.

Performance:
  - Thumbnails: a single ffmpeg pass in the background, written to disk.
  - 720p proxy for preview: uses the first hardware encoder that works on this
    machine (Intel QSV, NVIDIA NVENC, Apple VideoToolbox, AMD AMF) or libx264.
    The test is real, so an encoder is only used if it works.
  - Export always uses libx264 (CPU), to keep quality controlled by CRF.
"""

import argparse
import array
import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import analysis

HERE = os.path.dirname(os.path.abspath(__file__))
UI_FILE = os.path.join(HERE, "web", "editor_ui.html")

SPEED_MIN, SPEED_MAX = 0.05, 16.0
THUMB_FPS = analysis.THUMB_FPS  # thumbnails per second of video
CRF_H264 = {"high": 18, "medium": 23, "compact": 28, "lossless": 0}
CRF_H265 = {"high": 22, "medium": 26, "compact": 30, "lossless": 0}  # CRF 0: practically lossless (4:2:0), large file
HEIGHT = {"original": None, "1080p": 1080, "720p": 720}
FORMATS = ("mp4_h264", "mp4_h265", "mov_h264")
QUALITIES = ("high", "medium", "compact", "lossless")

# Encoders for the preview proxy, in order of preference.
# Each one is only used if the encoding test works on this machine.
ENCODERS = [
    ("h264_qsv", ["-c:v", "h264_qsv", "-b:v", "6M"]),          # Intel (Quick Sync)
    ("h264_nvenc", ["-c:v", "h264_nvenc", "-b:v", "6M"]),      # NVIDIA
    ("h264_videotoolbox", ["-c:v", "h264_videotoolbox", "-b:v", "6M"]),  # Apple
    ("h264_amf", ["-c:v", "h264_amf", "-b:v", "6M"]),          # AMD
    ("libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]),  # CPU, always
]

STEP_MOTION = analysis.STEP   # seconds per value of the audio and motion graphs
LOCK = threading.Lock()
THUMB_LOCK = threading.Lock()
STATE = {
    "info": None, "cache": None, "tmp": None, "audio": None, "audio_ready": threading.Event(),
    "results": None, "results_ready": threading.Event(), "camera_jobs": set(),
    "thumbs_ready": False, "proxy": None, "proxy_encoder": None,
}


# ---------------------------------------------------------------- ffmpeg

def check_tools():
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"ERROR: '{tool}' not found. Install ffmpeg and try again.")


def probe(path):
    # one ffprobe call (each call takes a while here): the first video and the first audio stream
    r = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "stream=codec_type,width,height,r_frame_rate,nb_frames,sample_rate:format=duration",
         "-of", "json", path],
        capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    streams = d.get("streams") or []
    st = next(x for x in streams if x.get("codec_type") == "video")
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    dur = float(d["format"]["duration"])
    nb = st.get("nb_frames", "")
    frames = int(nb) if str(nb).isdigit() else round(dur * fps)

    a_st = next((x for x in streams if x.get("codec_type") == "audio"), None)
    has_audio = a_st is not None
    sample_rate = int(a_st.get("sample_rate", 48000)) if has_audio else 48000

    return {
        "path": os.path.abspath(path),
        "name": os.path.basename(path),
        "duration": dur,
        "fps": fps,
        "frames": frames,
        "width": st["width"],
        "height": st["height"],
        "audio": has_audio,
        "sample_rate": sample_rate,
    }


def compute_audio(path):
    """Loudness (RMS) every 0.1 s, mono at 8 kHz. Keeps only the values."""
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "8000",
         "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    win = 800 * 2
    buf, levels = b"", []
    while True:
        chunk = p.stdout.read(1 << 16)
        if not chunk:
            break
        buf += chunk
        cut = (len(buf) // win) * win
        for i in range(0, cut, win):
            s = array.array("h")
            s.frombytes(buf[i:i + win])
            levels.append(round(((sum(x * x for x in s) / len(s)) ** 0.5) / 32768, 3))
        buf = buf[cut:]
    p.stdout.close()
    p.wait()
    return levels


def compute_motion(path, duration):
    """Movement values only (used by the tests)."""
    return analysis.run_pass(path, duration)[0]["motion"]


def _job_media(path, duration, cache):
    """Thumbnails and series, from the cache when they are there, else from one decode."""
    store = analysis.analyze(path, duration, cache)
    results = {"series": {k: analysis.values_of(store, k) for k in analysis.SERIES},
               "cuts": analysis.values_of(store, "cuts")}
    with LOCK:
        if STATE["info"] and STATE["info"]["path"] == path:
            STATE["results"] = results
            STATE["thumbs_ready"] = os.path.exists(os.path.join(cache, "thumbs.done"))
    STATE["results_ready"].set()


def camera_status(start):
    """Camera and object movement: needs OpenCV. Computed on request, in the background (one
    more pass over the video) and kept in the cache, so the next time it is ready at once."""
    from . import camera
    with LOCK:
        info, d = STATE["info"], STATE["cache"]
    if not info:
        return {"state": "none"}
    if not camera.available():
        return {"state": "unavailable"}
    store = analysis.read_store(d)
    if all(store.get(k, {}).get("version") == analysis.SERIES_VERSION[k] for k in ("camera", "object")):
        return {"state": "ready", "series": {k: analysis.values_of(store, k) for k in ("camera", "object")}}
    with LOCK:
        if start and info["path"] not in STATE["camera_jobs"]:
            STATE["camera_jobs"].add(info["path"])
            threading.Thread(target=_job_camera, args=(info["path"], info["duration"], d), daemon=True).start()
        running = info["path"] in STATE["camera_jobs"]
    return {"state": "computing" if running else "idle"}


def _job_camera(path, duration, d):
    from . import camera
    try:
        cam_values, obj_values = camera.camera_pass(path, duration)
        analysis.update_store(d, {
            "camera": {"version": analysis.SERIES_VERSION["camera"], "values": cam_values},
            "object": {"version": analysis.SERIES_VERSION["object"], "values": obj_values}})
    finally:
        with LOCK:
            STATE["camera_jobs"].discard(path)


def _job_audio(path, has_audio):
    levels = compute_audio(path) if has_audio else []
    with LOCK:
        if STATE["info"] and STATE["info"]["path"] == path:
            STATE["audio"] = levels
    STATE["audio_ready"].set()


def detect_encoder():
    """Really tests each encoder. Returns (name, arguments)."""
    for name, args in ENCODERS:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30",
             "-t", "0.5", "-pix_fmt", "yuv420p", *args, "-f", "null", "-"],
            capture_output=True)
        if r.returncode == 0:
            return name, args
    return ENCODERS[-1]


def _job_proxy(path, tmp):
    """720p proxy for preview (the original is kept for export)."""
    def proxy_cmd(enc_args, out):
        return ["ffmpeg", "-y", "-v", "error", "-i", path, "-vf", "scale=-2:720",
                "-pix_fmt", "yuv420p", *enc_args, "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart", out]

    name, args = detect_encoder()
    out = os.path.join(tmp, "proxy.mp4")
    r = subprocess.run(proxy_cmd(args, out), capture_output=True)
    if r.returncode != 0 and name != "libx264":
        # the encoder passed the test but failed on the video: fall back to the CPU
        name, args = ENCODERS[-1]
        r = subprocess.run(proxy_cmd(args, out), capture_output=True)
    with LOCK:
        if STATE["info"] and STATE["info"]["path"] == path and r.returncode == 0:
            STATE["proxy"] = out
            STATE["proxy_encoder"] = name


def load_video(raw):
    path = os.path.abspath(os.path.expanduser(str(raw).strip().strip('"').strip("'")))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found: {raw}")
    info = probe(path)
    with LOCK:
        old_tmp = STATE["tmp"]
        STATE.update(info=info, cache=analysis.entry_dir(info["path"]), tmp=tempfile.mkdtemp(prefix="tle_"),
                     audio=None, results=None, thumbs_ready=False, proxy=None, proxy_encoder=None)
        STATE["audio_ready"].clear()
        STATE["results_ready"].clear()
    if old_tmp:
        shutil.rmtree(old_tmp, ignore_errors=True)
    cache = STATE["cache"]
    threading.Thread(target=_job_audio, args=(info["path"], info["audio"]), daemon=True).start()
    threading.Thread(target=_job_media, args=(info["path"], info["duration"], cache), daemon=True).start()
    threading.Thread(target=_job_proxy, args=(info["path"], STATE["tmp"]), daemon=True).start()
    return info


def _extract_frame(frame, info, out, vf="", qargs=()):
    """Writes the exact frame number `frame` to `out`.
    Fast path: seek by time (-ss before -i). On some videos it does not return the last
    frame; in that case it decodes from the start and selects by frame number, which is
    exact (slower, but only in that rare case)."""
    base = ["ffmpeg", "-v", "error"]
    t = (frame + 0.5) / info["fps"]
    fast = base + ["-ss", f"{t:.4f}", "-i", info["path"]]
    if vf:
        fast += ["-vf", vf]
    fast += ["-frames:v", "1", *qargs, "-y", out]
    r = subprocess.run(fast, capture_output=True, text=True)
    if r.returncode == 0 and os.path.exists(out):
        return r
    sel = f"select=eq(n\\,{frame})" + (f",{vf}" if vf else "")
    slow = base + ["-i", info["path"], "-vf", sel, "-fps_mode", "passthrough",
                   "-frames:v", "1", *qargs, "-y", out]
    return subprocess.run(slow, capture_output=True, text=True)


def _seek_thumb(frame, info):
    """Fallback: generates a specific thumbnail if the thumbnail pass has not finished yet."""
    fn = os.path.join(STATE["cache"], f"f{frame}.jpg")
    with THUMB_LOCK:
        if not os.path.exists(fn):
            _extract_frame(frame, info, fn, vf="scale=320:-2", qargs=["-q:v", "4"])
    return fn if os.path.exists(fn) else None


def get_thumb(frame):
    info = STATE["info"]
    if not info:
        return None
    frame = max(0, min(info["frames"] - 1, frame))
    # thumbnail of the exact frame (sampling 4 per second repeated frames at the cuts)
    fn = _seek_thumb(frame, info)
    if not fn:
        return None
    with open(fn, "rb") as f:
        return f.read()


def atempo_chain(speed):
    parts, s = [], speed
    while s > 2.0 + 1e-9:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5 - 1e-9:
        parts.append("atempo=0.5")
        s /= 0.5
    parts.append(f"atempo={s:.6f}")
    return ",".join(parts)


def audio_filter(mode, sp, sr, vol=1.0):
    """Audio filter for a segment (None = no filter). vol: level volume (0 to 1)."""
    if mode == "mute":
        parts = [atempo_chain(sp)] if abs(sp - 1.0) >= 1e-6 else []
        return ",".join(parts + ["volume=0"])
    if mode == "original":
        base = None if abs(sp - 1.0) < 1e-6 else f"asetrate={sr}*{sp:.6f},aresample={sr}"
    else:
        base = None if abs(sp - 1.0) < 1e-6 else atempo_chain(sp)
    parts = [base] if base else []
    if abs(vol - 1.0) > 1e-3:
        parts.append(f"volume={vol:.4f}")
    return ",".join(parts) or None


def codec_args(fmt, quality):
    if fmt == "mp4_h265":
        return (["-c:v", "libx265", "-tag:v", "hvc1", "-crf", str(CRF_H265.get(quality, 26)),
                 "-preset", "medium", "-pix_fmt", "yuv420p"], ".mp4")
    ext = ".mov" if fmt == "mov_h264" else ".mp4"
    return (["-c:v", "libx264", "-crf", str(CRF_H264.get(quality, 23)),
             "-preset", "medium", "-pix_fmt", "yuv420p"], ext)


def run_ffmpeg(cmd, dur=None, on_progress=None):
    """Runs ffmpeg. With on_progress, reports the progress (0 to 1) by reading ffmpeg's own progress output."""
    if on_progress is None or not dur:
        return subprocess.run(cmd, capture_output=True, text=True)
    cmd = cmd[:-1] + ["-progress", "pipe:1", "-nostats"] + cmd[-1:]
    with tempfile.TemporaryFile("w+") as errf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, text=True)
        for line in p.stdout:
            if line.startswith("out_time_us="):
                try:
                    on_progress(min(1.0, int(line.split("=", 1)[1]) / 1e6 / dur))
                except ValueError:
                    pass
        p.stdout.close()
        rc = p.wait()
        errf.seek(0)
        err = errf.read()
    return subprocess.CompletedProcess(cmd, rc, "", err)


def export_pieces(info, pieces, fmt="mp4_h264", res="original", quality="medium", output=None,
                  progress=None):
    """
    pieces: (first_frame, last_frame, speed, audio_mode), in frames.

    Each segment is encoded to a temporary file, one at a time. This way memory
    does not grow with the number of segments or the resolution. Then the files
    are joined without re-encoding (concat, stream copy).
    Fractional speed: the output frame falls between two frames and ffmpeg
    rounds it to the nearest one, as it does by default.
    """
    if not pieces:
        return {"ok": False, "msg": "No segments to export."}
    pieces = [(p[0], p[1], p[2], p[3], p[4] if len(p) > 4 else 1.0) for p in pieces]
    for p in pieces:
        sp = p[2]
        if not (SPEED_MIN <= sp <= SPEED_MAX):
            return {"ok": False, "msg": f"Speed out of range: {sp}"}

    fps, sr, has_audio = info["fps"], info["sample_rate"], info["audio"]
    vargs, ext = codec_args(fmt, quality)
    h = HEIGHT.get(res)

    out = (output or "").strip()
    if not out:
        out = os.path.splitext(info["path"])[0] + "_timelapse_" + datetime.now().strftime("%Y%m%d-%H%M") + ext
    else:
        out = os.path.expanduser(out)
        if not os.path.splitext(out)[1]:
            out += ext
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    tmp = tempfile.mkdtemp(prefix="tle_exp_")
    try:
        parts = []
        for i, (sf, ef, sp, au, vol) in enumerate(pieces):
            part = os.path.join(tmp, f"p{i:05d}.mp4")
            # expected output frames: the segment duration divided by the speed.
            # Without a limit, ffmpeg produced 1 to 2 extra frames per accelerated segment (item 55).
            n_out = max(1, round((ef - sf) / sp))
            # the output is a constant frame rate: the stretched timestamps of a slow segment are
            # filled with repeated frames (fps), so the frame count is the expected one
            vf = [f"setpts=(PTS-STARTPTS)/{sp:.6f}", f"fps={fps:.6f}"]
            if h:
                vf.append(f"scale=-2:{h}")
            cmd = ["ffmpeg", "-y", "-v", "error",
                   "-ss", f"{sf / fps:.6f}", "-t", f"{(ef - sf) / fps:.6f}", "-i", info["path"],
                   "-map", "0:v:0", "-vf", ",".join(vf), *vargs]
            if has_audio:
                af = audio_filter(au, sp, sr, vol)
                cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k", "-ar", str(sr)]
                if af:
                    cmd += ["-af", af]
            else:
                cmd += ["-an"]
            # video limited to the exact number of frames; the -t with half a frame of margin trims the audio
            # without losing the last frame (an exact -t dropped the last frame at the edge)
            cmd += ["-frames:v", str(n_out), "-t", f"{(n_out + 0.5) / fps:.6f}", part]
            dur = (ef - sf) / fps / sp
            cb = (lambda f, i=i: progress((i + f) / len(pieces))) if progress else None
            r = run_ffmpeg(cmd, dur, cb)
            if r.returncode != 0:
                reason = (r.stderr or "").strip()[-400:]
                if r.returncode < 0:
                    reason = (reason + " — the process was killed by the system (low memory?)").strip()
                return {"ok": False, "msg": f"Failed on segment {i + 1}: {reason or 'unknown error'}"}
            parts.append(part)

        lista = os.path.join(tmp, "list.txt")
        with open(lista, "w", encoding="utf-8") as f:
            for p in parts:
                f.write("file '" + p.replace("'", "'\\''") + "'\n")
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lista,
             "-c", "copy", "-movflags", "+faststart", out],
            capture_output=True, text=True)
        if r.returncode != 0:
            return {"ok": False, "msg": "Failed to join the segments: " + (r.stderr or "")[-400:]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"ok": True, "msg": f"Done: {out}", "output": out}


def pieces_from_body(body, info):
    """Converts the segments sent by the interface (seconds) into frames, aligned to the frame."""
    fps, nframes = info["fps"], info["frames"]
    pieces = []
    for p in body.get("pieces", []):
        sf = round(float(p["start"]) * fps)
        ef = min(nframes, round(float(p["end"]) * fps))
        au = p.get("audio", "original")
        if au not in ("mute", "original", "corrected"):
            au = "original"
        vol = min(1.0, max(0.0, float(p.get("volume", 1.0))))
        if ef > sf:
            pieces.append((sf, ef, float(p["speed"]), au, vol))
    return pieces


def export_from_json(body, progress=None):
    """Export requested by the interface (times in seconds, already aligned to the frame)."""
    info = STATE["info"]
    if not info:
        return {"ok": False, "msg": "No video loaded."}
    pieces = pieces_from_body(body, info)
    return export_pieces(info, pieces, body.get("format", "mp4_h264"),
                         body.get("resolution", "original"), body.get("quality", "medium"),
                         body.get("output"), progress)


def export_frame(body):
    """Saves a frame of the original video as PNG or JPG, in the chosen folder."""
    info = STATE["info"]
    if not info:
        return {"ok": False, "msg": "No video loaded."}
    fr = max(0, min(info["frames"] - 1, int(body.get("frame", 0))))
    fmt = body.get("format", "png")
    if fmt not in ("png", "jpg"):
        return {"ok": False, "msg": "Invalid format: use PNG or JPG."}
    quality = max(1, min(100, int(body.get("quality", 95))))
    folder = os.path.abspath(os.path.expanduser(body.get("folder") or os.path.dirname(info["path"])))
    if not os.path.isdir(folder):
        return {"ok": False, "msg": "Folder not found."}
    base = os.path.splitext(os.path.basename(info["path"]))[0]
    ext = ".png" if fmt == "png" else ".jpg"
    out = os.path.join(folder, f"{base}_frame{fr}{ext}")
    n = 2
    while os.path.exists(out):   # does not overwrite: adds _2, _3...
        out = os.path.join(folder, f"{base}_frame{fr}_{n}{ext}")
        n += 1
    # centre of the frame, as in the thumbnails (see _extract_frame)
    qargs = []
    if fmt == "jpg":   # JPG quality scale: 100 = best (qscale 2), 1 = worst (qscale 31)
        qargs = ["-q:v", str(round(2 + (100 - quality) * 29 / 99))]
    r = _extract_frame(fr, info, out, qargs=qargs)
    if r.returncode != 0 or not os.path.exists(out):
        return {"ok": False, "msg": "Failed to save the frame: " + (r.stderr or "")[-300:]}
    return {"ok": True, "msg": "Frame saved: " + out, "output": out}


# background export: the interface checks progress at /api/export/progress
EXPORT_JOB = {"running": False, "frac": 0.0, "msg": "", "ok": None}
EXPORT_LOCK = threading.Lock()


def export_async(body):
    with EXPORT_LOCK:
        if EXPORT_JOB["running"]:
            return {"ok": False, "msg": "An export is already running."}
        EXPORT_JOB.update(running=True, frac=0.0, msg="Exporting…", ok=None)

    def task():
        try:
            r = export_from_json(body, progress=lambda f: EXPORT_JOB.update(frac=f))
        except Exception as exc:
            r = {"ok": False, "msg": f"Unexpected error: {exc}"}
        EXPORT_JOB.update(running=False, frac=1.0 if r["ok"] else EXPORT_JOB["frac"],
                          msg=r["msg"], ok=r["ok"])

    threading.Thread(target=task, daemon=True).start()
    return {"ok": True, "msg": "Export started."}


def output_sample(pieces, fps, max_s=4.0):
    """First max_s seconds of output, as frame ranges, to estimate the size."""
    out, total = [], 0.0
    for sf, ef, sp, au, vol in pieces:
        if total >= max_s:
            break
        frames = min(ef - sf, max(1, round((max_s - total) * sp * fps)))
        out.append((sf, sf + frames, sp, au, vol))
        total += frames / fps / sp
    return out, total


def estimate_from_json(body):
    """
    Estimated size of each format: encodes only the first seconds of output
    with the same format and quality, measures the size and scales by the total time.
    """
    info = STATE["info"]
    if not info:
        return {"ok": False, "msg": "No video loaded."}
    fps = info["fps"]
    pieces = pieces_from_body(body, info)
    if not pieces:
        return {"ok": False, "msg": "Nothing to export."}
    fmt = body.get("format", "mp4_h264")
    res = body.get("resolution", "original")
    quality = body.get("quality", "medium")
    total_s = sum((ef - sf) / fps / sp for sf, ef, sp, _au, _vol in pieces)
    sample, sample_s = output_sample(pieces, fps)
    _vargs, ext = codec_args(fmt, quality)
    tmp = tempfile.mkdtemp(prefix="tle_est_")
    try:
        r = export_pieces(info, sample, fmt, res, quality, os.path.join(tmp, "sample" + ext))
        if not r["ok"]:
            return r
        size = os.path.getsize(r["output"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"ok": True, "bytes": int(size / sample_s * total_s),
            "sample_s": round(sample_s, 2), "total_s": round(total_s, 2)}


# ---------------------------------------------------------------- text segments (CLI)

SEG_RE = re.compile(
    r"^\s*(\S+?)\s*-\s*(\S+?)\s+(\d+(?:[.,]\d+)?)\s*x?\s*(mute|original|corrected)?\s*$", re.I)


def parse_time_frames(tok, info):
    t = tok.strip().lower()
    if t == "end":
        return info["frames"]
    if t.startswith("f"):
        return int(t[1:])
    if ":" in t:
        m, s = t.split(":", 1)
        sec = int(m) * 60 + float(s.replace(",", "."))
    else:
        sec = float(t.replace(",", "."))
    return round(sec * info["fps"])


def parse_segments(text, info):
    """Converts the segment text into a list of segments in frames, with 1x in the gaps."""
    fr = info["frames"]
    segs = []
    for n, line in enumerate(re.split(r"[;\n]", text), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = SEG_RE.match(line)
        if not m:
            raise ValueError(f"segment {n} is invalid: '{line}'. "
                             "Use: START-END SPEED [mute|original|corrected]")
        sf = parse_time_frames(m.group(1), info)
        ef = min(fr, parse_time_frames(m.group(2), info))
        sp = float(m.group(3).replace(",", "."))
        au = (m.group(4) or ("original" if sp == 1 else "mute")).lower()
        if not (SPEED_MIN <= sp <= SPEED_MAX):
            raise ValueError(f"segment {n}: speed {sp} out of range ({SPEED_MIN} to {SPEED_MAX})")
        if sf >= fr:
            raise ValueError(f"segment {n}: the start ({sf} frames) is past the end of the video ({fr} frames)")
        if sf < 0 or ef <= sf:
            raise ValueError(f"segment {n}: the end must be greater than the start")
        segs.append((sf, ef, sp, au))

    segs.sort()
    for a, b in zip(segs, segs[1:]):
        if b[0] < a[1]:
            raise ValueError(f"overlapping segments: frame {b[0]} starts before the previous one ends ({a[1]})")

    pieces, cur = [], 0
    for sf, ef, sp, au in segs:
        if sf > cur:
            pieces.append((cur, sf, 1.0, "original"))
        pieces.append((sf, ef, sp, au))
        cur = ef
    if cur < fr:
        pieces.append((cur, fr, 1.0, "original"))
    return pieces


# ---------------------------------------------------------------- import window

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".mts", ".m2ts", ".avi", ".mxf")


def browse_dir(path):
    """Lists folders and videos in a folder on the computer where the program runs (for the Import window)."""
    path = (path or "").strip()
    if not path:
        path = os.path.dirname(STATE["info"]["path"]) if STATE["info"] else "~"
    base = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(base):
        return {"ok": False, "msg": "Folder not found."}
    try:
        names = sorted(os.listdir(base), key=str.lower)
    except OSError as exc:
        return {"ok": False, "msg": f"Could not open the folder: {exc}"}
    folders, videos = [], []
    for n in names:
        if n.startswith("."):
            continue
        full = os.path.join(base, n)
        try:
            if os.path.isdir(full):
                folders.append({"name": n, "path": full})
            elif n.lower().endswith(VIDEO_EXT):
                videos.append({"name": n, "path": full, "size": os.path.getsize(full)})
        except OSError:
            continue
    parent = os.path.dirname(base)
    return {"ok": True, "path": base, "parent": parent if parent != base else None,
            "dirs": folders, "videos": videos}


# ---------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj))

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/api/browse":
            return self._json(browse_dir(q.get("path", [""])[0]))
        if u.path == "/api/export/progress":
            return self._json(dict(EXPORT_JOB))
        if u.path == "/":
            with open(UI_FILE, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/api/info":
            info = STATE["info"]
            if info:
                info = dict(info, proxy_ready=bool(STATE["proxy"]),
                            proxy_encoder=STATE["proxy_encoder"],
                            thumbs_ready=STATE["thumbs_ready"])
            return self._json({"info": info})
        if u.path == "/api/audio":
            STATE["audio_ready"].wait(timeout=600)
            return self._json({"levels": STATE["audio"] or []})
        if u.path == "/api/camera":
            return self._json(camera_status(q.get("start", ["0"])[0] == "1"))
        if u.path == "/api/analysis":   # does not wait: the page asks again until it is ready
            res = STATE["results"] if STATE["results_ready"].is_set() else None
            return self._json({"ready": res is not None, **(res or {"series": {}, "cuts": []})})
        if u.path == "/api/thumb":
            data = get_thumb(int(q.get("f", ["0"])[0]))
            if data:
                return self._send(200, data, "image/jpeg")
            return self._send(404, b"", "image/jpeg")
        if u.path == "/video":
            return self._serve_video()
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"ok": False, "msg": "Invalid JSON."}, 400)
        if self.path == "/api/export":
            return self._json(export_async(body))
        if self.path == "/api/estimate":
            return self._json(estimate_from_json(body))
        if self.path == "/api/frame":
            return self._json(export_frame(body))
        if self.path == "/api/load":
            try:
                info = load_video(body.get("path", ""))
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                return self._json({"ok": False, "msg": f"Could not open the video: {exc}"})
            return self._json({"ok": True, "info": dict(info, proxy_ready=False,
                               proxy_encoder=None, thumbs_ready=False)})
        self._send(404, "not found", "text/plain")

    def _serve_video(self):
        info = STATE["info"]
        if not info:
            return self._send(404, "no video", "text/plain")
        # preview uses the proxy when it is ready; the original is used for export
        path = STATE["proxy"] or info["path"]
        size = os.path.getsize(path)
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                status = 206
        end = min(end, size - 1)
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        # the browser routinely drops range requests (seek, source change);
        # that is not an error worth a traceback
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True


def cleanup():
    # the cache folder of the video is kept for the next time; only the temporary files go
    if STATE.get("tmp"):
        shutil.rmtree(STATE["tmp"], ignore_errors=True)


def run_cli(args):
    info = probe(args.video)
    text = args.segments or ""
    if args.segments_file:
        with open(args.segments_file, encoding="utf-8") as f:
            text += "\n" + f.read()
    try:
        pieces = parse_segments(text, info)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"{len(pieces)} segments (including 1x gaps) · video: {info['frames']} frames")
    r = export_pieces(info, pieces, args.format, args.resolution, args.quality, args.output)
    print(r["msg"])
    sys.exit(0 if r["ok"] else 1)


def main():
    ap = argparse.ArgumentParser(description="Time-lapse editor with per-segment speed.")
    ap.add_argument("video", nargs="?", help="path to the video (e.g. DJI_0001.MP4)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--segments", help='segments, e.g. "0-10 4x mute; 10-20 1x; 20:00-end 8x corrected"')
    ap.add_argument("--segments-file", help="text file with one segment per line")
    ap.add_argument("--output", help="output file (default: next to the video, with _timelapse)")
    ap.add_argument("--format", default="mp4_h264", choices=FORMATS)
    ap.add_argument("--resolution", default="original", choices=list(HEIGHT))
    ap.add_argument("--quality", default="medium", choices=QUALITIES)
    args = ap.parse_args()

    check_tools()
    if args.segments or args.segments_file:
        if not args.video:
            ap.error("the --segments mode needs the video path")
        if not os.path.isfile(args.video):
            sys.exit(f"ERROR: file not found: {args.video}")
        return run_cli(args)

    if not os.path.isfile(UI_FILE):
        sys.exit(f"ERROR: web/editor_ui.html not found in {HERE}")
    atexit.register(cleanup)
    if args.video:
        if not os.path.isfile(args.video):
            sys.exit(f"ERROR: file not found: {args.video}")
        info = load_video(args.video)
        print(f"Video: {info['name']} · {info['width']}x{info['height']} · "
              f"{info['fps']:.3f} fps · {info['frames']} frames")
    else:
        STATE["audio_ready"].set()
        STATE["results_ready"].set()
        print("No video: the interface opens in simulation mode. Use the Import button on screen to choose a video.")

    # SIGTERM (kill) ends the program the normal way, so the temporary files are removed too
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Editor at {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
