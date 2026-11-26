"""Unit tests of the analysis cache, the cut detection and the camera movement (no interface)."""
import os
import subprocess
import sys
import time

import pytest

import timelapse_editor as te
import timelapse_editor.analysis as an
from conftest import requires_ffmpeg


# ---------------------------------------------------------------- cache

def test_fingerprint_is_stable_and_depends_on_the_file(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    g = tmp_path / "b.mp4"
    g.write_bytes(b"x")
    assert an.fingerprint(str(f)) == an.fingerprint(str(f))
    assert an.fingerprint(str(f)) != an.fingerprint(str(g))


def test_entry_dir_lives_in_the_cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    d = an.entry_dir(str(f))
    assert d.startswith(str(tmp_path / "cache" / "timelapse-editor"))
    assert os.path.isdir(d)


def test_entry_dir_falls_back_to_a_temporary_folder(tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")                 # a file where the cache folder should be
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocker))
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    d = an.entry_dir(str(f))
    assert os.path.isdir(d) and not d.startswith(str(blocker))


def test_a_damaged_store_is_read_as_empty(tmp_path):
    (tmp_path / "series.json").write_text("{not json", encoding="utf-8")
    assert an.read_store(str(tmp_path)) == {}
    assert an.read_store(str(tmp_path / "missing")) == {}


def test_update_store_merges_the_entries(tmp_path):
    an.update_store(str(tmp_path), {"a": {"version": 1, "values": [1]}})
    store = an.update_store(str(tmp_path), {"b": {"version": 1, "values": [2]}})
    assert store == an.read_store(str(tmp_path))
    assert set(store) == {"a", "b"}


def test_values_of_a_missing_series_is_empty():
    assert an.values_of({}, "motion") == []


@requires_ffmpeg
def test_analyze_without_thumbnails_writes_no_thumbnails(tmp_path):
    clip = str(tmp_path / "clip.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", clip], check=True)
    d = str(tmp_path / "entry")
    os.makedirs(d)
    store = an.analyze(clip, 2.0, d, thumbs=False)
    assert set(an.SERIES) <= set(store) and "cuts" in store
    assert not os.path.exists(os.path.join(d, "thumbs.done"))
    assert not [f for f in os.listdir(d) if f.endswith(".jpg")]


# ---------------------------------------------------------------- scene cuts

def test_detect_cuts_finds_one_spike():
    values = [0.02] * 100
    values[40] = 0.3
    assert an.detect_cuts(values) == [4.0]


def test_detect_cuts_ignores_steady_movement_and_tiny_spikes():
    steady = [0.1 + 0.005 * (i % 5) for i in range(100)]   # no peak much bigger than its neighbours
    assert an.detect_cuts(steady) == []
    tiny = [0.0] * 100
    tiny[50] = 0.03                                         # below the minimum of a cut
    assert an.detect_cuts(tiny) == []


def test_detect_cuts_gives_one_cut_per_peak():
    values = [0.01] * 100
    values[30], values[31] = 0.3, 0.25
    assert an.detect_cuts(values) == [3.0]


# ---------------------------------------------------------------- camera

def test_camera_is_unavailable_without_opencv(monkeypatch):
    import timelapse_editor.camera as cam
    monkeypatch.setitem(sys.modules, "cv2", None)          # importing it raises ImportError
    assert cam.available() is False


def test_a_flat_picture_has_no_camera_or_object_movement():
    pytest.importorskip("cv2")
    import cv2
    import numpy as np
    import timelapse_editor.camera as cam
    flat = np.full((cam.H, cam.W), 128, np.uint8)
    assert cam._frame_motion(cv2, flat, flat.copy()) == (0.0, 0.0)


def test_camera_status_follows_the_state_of_the_video(monkeypatch, clean_state, tmp_path):
    import timelapse_editor.camera as cam
    monkeypatch.setitem(te.STATE, "camera_jobs", set())
    assert te.camera_status(False) == {"state": "none"}
    clean_state["info"] = {"path": str(tmp_path / "v.mp4"), "duration": 1.0}
    monkeypatch.setattr(cam, "available", lambda: False)
    assert te.camera_status(True) == {"state": "unavailable"}
    monkeypatch.setattr(cam, "available", lambda: True)
    started = []
    monkeypatch.setattr(te, "_job_camera", lambda *a: started.append(a))
    assert te.camera_status(False) == {"state": "idle"} and started == []
    assert te.camera_status(True) == {"state": "computing"}
    assert te.camera_status(True) == {"state": "computing"} and len(started) == 1   # no second job
    te.STATE["camera_jobs"].clear()


@requires_ffmpeg
def test_camera_job_writes_the_series_to_the_cache(monkeypatch, clean_state, tmp_path):
    pytest.importorskip("cv2")
    clip = str(tmp_path / "clip.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", clip], check=True)
    info = te.probe(clip)
    monkeypatch.setitem(te.STATE, "camera_jobs", set())
    clean_state["info"] = info
    assert te.camera_status(True) == {"state": "computing"}
    deadline = time.time() + 120
    status = te.camera_status(False)
    while status["state"] == "computing" and time.time() < deadline:
        time.sleep(0.5)
        status = te.camera_status(False)
    assert status["state"] == "ready"
    assert len(status["series"]["camera"]) == len(status["series"]["object"]) > 20
    assert te.STATE["camera_jobs"] == set()           # the job is done and forgotten


def test_a_result_for_another_video_is_dropped(monkeypatch, clean_state, tmp_path):
    monkeypatch.setitem(te.STATE, "results", None)
    clean_state["info"] = {"path": "/the/current/video.mp4"}
    monkeypatch.setattr(te.analysis, "analyze",
                        lambda p, d, c: {"motion": {"version": 1, "values": [0.1]}})
    te._job_media("/an/older/video.mp4", 1.0, str(tmp_path))
    assert te.STATE["results"] is None


def test_a_result_for_the_current_video_is_kept(monkeypatch, clean_state, tmp_path):
    monkeypatch.setitem(te.STATE, "results", None)
    clean_state["info"] = {"path": "/the/current/video.mp4"}
    store = {"motion": {"version": 1, "values": [0.0, 0.2]},
             "brightness": {"version": 1, "values": [0.5]},
             "sharpness": {"version": 1, "values": [0.0]},
             "cuts": {"version": 1, "values": [1.5]}}
    monkeypatch.setattr(te.analysis, "analyze", lambda p, d, c: store)
    te._job_media("/the/current/video.mp4", 1.0, str(tmp_path))
    assert te.STATE["results"]["series"]["motion"] == [0.0, 0.2]
    assert te.STATE["results"]["cuts"] == [1.5]


def test_two_equal_peaks_next_to_each_other_give_one_cut():
    values = [0.01] * 100
    values[30] = values[31] = 0.3
    assert an.detect_cuts(values) == [3.0]


def test_camera_frame_with_too_few_tracked_points_gives_zero(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    import timelapse_editor.camera as cam
    rng = np.random.default_rng(1)
    prev = rng.integers(0, 255, (cam.H, cam.W), dtype=np.uint8)
    # the tracker loses every point: nothing to measure
    real = cv2.calcOpticalFlowPyrLK
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK",
                        lambda a, b, p, *args, **kw: (p.copy(), np.zeros((len(p), 1), np.uint8), None))
    assert cam._frame_motion(cv2, prev, prev.copy()) == (0.0, 0.0)
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", real)
