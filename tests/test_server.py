"""
Server tests (timelapse_editor), with pytest.

Run from the project folder:
    uv run pytest -v

Tests that use video need ffmpeg and ffprobe on PATH; without them, they are skipped.
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import pytest

import timelapse_editor as te
from conftest import requires_ffmpeg


# ---------------------------------------------------------------- helpers

def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    return float(r.stdout.strip())


def has_audio(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=codec_name", "-of", "json", path],
                       capture_output=True, text=True)
    return bool(json.loads(r.stdout or "{}").get("streams"))


def format_and_codec(path):
    """(video codec, container format name), read by ffprobe."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name:format=format_name",
                        "-of", "json", path], capture_output=True, text=True)
    j = json.loads(r.stdout)
    return j["streams"][0]["codec_name"], j["format"]["format_name"]


def max_volume(path):
    """Audio peak in dB (volumedetect). None if there is no audio."""
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-vn", "-af", "volumedetect",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"max_volume: (-?[\d.]+) dB", r.stderr)
    return float(m.group(1)) if m else None


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "timelapse_editor", *args],
                          capture_output=True, text=True)


def frames_of_video(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return int(r.stdout.strip())


# ---------------------------------------------------------------- pure functions

def test_atempo_speed_1_changes_nothing():
    assert te.atempo_chain(1.0) == "atempo=1.000000"


@pytest.mark.parametrize("speed", [0.25, 0.5, 0.7, 1.5, 2.0, 3.0, 8.0, 16.0])
def test_atempo_chain_reproduces_the_speed(speed):
    factors = [float(p.split("=")[1]) for p in te.atempo_chain(speed).split(",")]
    assert math.prod(factors) == pytest.approx(speed, rel=1e-5)
    for f in factors:
        assert 0.5 - 1e-6 <= f <= 2.0 + 1e-6, f"factor {f} out of 0.5 to 2.0"


@pytest.mark.parametrize("mode, speed, volume, expected", [
    ("original", 1.0, 1.0, None),
    ("original", 2.0, 1.0, "asetrate=48000*2.000000,aresample=48000"),
    ("original", 1.0, 0.5, "volume=0.5000"),
    ("mute", 1.0, 1.0, "volume=0"),
])
def test_audio_filter(mode, speed, volume, expected):
    f = te.audio_filter(mode, speed, 48000, volume)
    if expected is None:
        assert f is None
    else:
        assert expected in f


def test_accelerated_mute_stays_muted():
    f = te.audio_filter("mute", 2.0, 48000)
    assert f.startswith("atempo=2.0") and f.endswith("volume=0")


def test_corrected_audio_chains_atempo():
    assert te.audio_filter("corrected", 4.0, 48000).startswith("atempo=2.0,atempo=2.0")


@pytest.mark.parametrize("fmt, quality, ext, codec, crf", [
    ("mp4_h264", "medium", ".mp4", "libx264", "23"),
    ("mov_h264", "high", ".mov", "libx264", "18"),
    ("mp4_h265", "high", ".mp4", "libx265", "22"),
    ("mp4_h264", "lossless", ".mp4", "libx264", "0"),
    ("mp4_h265", "lossless", ".mp4", "libx265", "0"),
    ("mp4_h264", "nonexistent", ".mp4", "libx264", "23"),   # unknown quality falls back to the default
])
def test_codec_and_crf(fmt, quality, ext, codec, crf):
    args, e = te.codec_args(fmt, quality)
    assert e == ext
    assert codec in args
    assert args[args.index("-crf") + 1] == crf


def test_h265_uses_hvc1_tag():
    args, _ = te.codec_args("mp4_h265", "high")
    assert "hvc1" in args


def test_pieces_converts_seconds_to_frames():
    info = {"fps": 30.0, "frames": 120}
    body = {"pieces": [{"start": 1.0, "end": 2.5, "speed": 2, "audio": "original", "volume": 1}]}
    assert te.pieces_from_body(body, info) == [(30, 75, 2.0, "original", 1.0)]


@pytest.mark.parametrize("volume, expected", [(3, 1.0), (-1, 0.0)])
def test_pieces_clamps_volume_between_0_and_1(volume, expected):
    body = {"pieces": [{"start": 0, "end": 1, "speed": 1, "volume": volume}]}
    assert te.pieces_from_body(body, {"fps": 30.0, "frames": 120})[0][4] == expected


def test_pieces_invalid_audio_becomes_original():
    body = {"pieces": [{"start": 0, "end": 1, "speed": 1, "audio": "xyz"}]}
    assert te.pieces_from_body(body, {"fps": 30.0, "frames": 120})[0][3] == "original"


def test_pieces_end_past_the_video_is_clamped():
    body = {"pieces": [{"start": 0, "end": 99, "speed": 1}]}
    assert te.pieces_from_body(body, {"fps": 30.0, "frames": 120})[0][1] == 120


def test_pieces_empty_segment_is_dropped():
    body = {"pieces": [{"start": 2, "end": 2, "speed": 1}]}
    assert te.pieces_from_body(body, {"fps": 30.0, "frames": 120}) == []


@pytest.mark.parametrize("pieces, fps, expected_total, end_frame", [
    ([(0, 100, 1.0, "original", 1.0), (100, 200, 2.0, "mute", 1.0)], 10.0, 4.0, 40),
    ([(0, 400, 4.0, "mute", 1.0)], 10.0, 4.0, 160),
    ([(0, 20, 1.0, "original", 1.0)], 10.0, 2.0, 20),
])
def test_output_sample_is_at_most_4_seconds(pieces, fps, expected_total, end_frame):
    sample, total = te.output_sample(pieces, fps)
    assert total == pytest.approx(expected_total)
    assert sample[0][1] == end_frame


@pytest.mark.parametrize("text, expected", [
    ("1:05.5", 1965), ("f120", 120), ("end", 120),
    ("2.5", 75), ("3,5", 105), ("0:00", 0),
])
def test_time_in_several_formats(text, expected):
    assert te.parse_time_frames(text, {"fps": 30.0, "frames": 120}) == expected


INFO_TEXT = {"fps": 30.0, "frames": 120}


def test_gaps_in_segments_become_original_1x():
    assert te.parse_segments("0-1 4x mute; 2-3 1x", INFO_TEXT) == [
        (0, 30, 4.0, "mute"), (30, 60, 1.0, "original"),
        (60, 90, 1.0, "original"), (90, 120, 1.0, "original")]


def test_default_audio_depends_on_speed():
    assert te.parse_segments("0-1 4x", INFO_TEXT)[0][3] == "mute"
    assert te.parse_segments("0-1 1x", INFO_TEXT)[0][3] == "original"


def test_comments_and_end_keyword():
    assert te.parse_segments("# test\n1-end 2x original  # fast", INFO_TEXT)[-1] == (30, 120, 2.0, "original")


@pytest.mark.parametrize("text, reason", [
    ("abc", "meaningless line"),
    ("0-1 999x", "speed out of range"),
    ("0-2 4x; 1-3 2x", "overlapping segments"),
    ("200-300 2x", "start after the end of the video"),
    ("2-1 2x", "end before the start"),
])
def test_invalid_segments_raise_an_error(text, reason):
    with pytest.raises(ValueError):
        te.parse_segments(text, INFO_TEXT)


@pytest.fixture
def folder_with_files(tmp_path):
    for folder in ["b_folder", "a_folder", ".hidden"]:
        (tmp_path / folder).mkdir()
    for name in ["Zeta.mp4", "alfa.MOV", "clip.mts", "notes.txt", ".hidden.mp4"]:
        (tmp_path / name).write_bytes(b"")
    return tmp_path


def test_browser_lists_folders_and_videos_in_order(folder_with_files):
    r = te.browse_dir(str(folder_with_files))
    assert r["ok"]
    assert [p["name"] for p in r["dirs"]] == ["a_folder", "b_folder"]
    assert [v["name"] for v in r["videos"]] == ["alfa.MOV", "clip.mts", "Zeta.mp4"]


def test_browser_parent_folder(folder_with_files):
    assert te.browse_dir(str(folder_with_files))["parent"] == str(folder_with_files.parent)
    assert te.browse_dir("/")["parent"] is None


def test_browser_folder_without_videos(folder_with_files):
    r = te.browse_dir(str(folder_with_files / "a_folder"))
    assert r["ok"] and r["dirs"] == [] and r["videos"] == []


def test_browser_missing_folder():
    assert not te.browse_dir("/does/not/exist")["ok"]


# ---------------------------------------------------------------- tools and errors

def test_without_ffmpeg_the_program_stops_with_a_message(monkeypatch):
    monkeypatch.setattr(te.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit) as exc:
        te.check_tools()
    assert "ffmpeg" in str(exc.value)


def test_missing_file_raises_an_error():
    with pytest.raises(FileNotFoundError):
        te.load_video("/does/not/exist/video.mp4")


def test_encoder_that_fails_falls_back_to_libx264(monkeypatch):
    monkeypatch.setattr(te, "ENCODERS", [("fake", ["-c:v", "does_not_exist"]),
                                         ("libx264", ["-c:v", "libx264"])])
    assert te.detect_encoder()[0] == "libx264"


def test_encoder_list_ends_with_libx264():
    assert te.ENCODERS[-1][0] == "libx264"


# ---------------------------------------------------------------- video (ffmpeg)

@requires_ffmpeg
def test_probe_reads_fps_frames_and_audio(clips):
    info = te.probe(clips["av"])
    assert info["fps"] == pytest.approx(30.0, abs=0.01)
    assert info["frames"] == 360
    assert (info["width"], info["height"]) == (640, 360)
    assert info["audio"] is True


@requires_ffmpeg
def test_probe_without_audio(clips):
    assert te.probe(clips["mute"])["audio"] is False


@requires_ffmpeg
def test_load_video_returns_info(clips):
    info = te.load_video(clips["av"])
    assert info["name"] == "test_clip.mp4"
    assert info["frames"] == 360


@requires_ffmpeg
def test_audio_shows_silence_and_the_tone(clips):
    levels = te.compute_audio(clips["av"])        # one value every 0.1 s
    assert len(levels) == pytest.approx(120, abs=2)
    silence = levels[0:30]                        # 0 s to 3 s
    tone = levels[45:55]                          # 4.5 s to 5.5 s
    assert max(silence) < 0.01
    assert min(tone) > 0.1


# ---------------------------------------------------------------- export (ffmpeg)

@pytest.fixture
def output_path(tmp_path):
    return lambda name: str(tmp_path / name)


@requires_ffmpeg
@pytest.mark.parametrize("fmt, codec, container", [
    ("mp4_h264", "h264", "mp4"),
    ("mp4_h265", "hevc", "mp4"),
    ("mov_h264", "h264", "mov"),
])
def test_output_formats(loaded, output_path, fmt, codec, container):
    out = output_path("f." + ("mov" if container == "mov" else "mp4"))
    r = te.export_pieces(loaded, [(0, 360, 1.0, "original", 1.0)], fmt=fmt, output=out)
    assert r["ok"], r.get("msg")
    codec_read, fmt_name = format_and_codec(out)
    assert codec_read == codec
    assert container in fmt_name
    assert duration(out) == pytest.approx(12.0, abs=0.15)


@requires_ffmpeg
def test_one_1x_segment_keeps_the_duration(loaded, output_path):
    out = output_path("1x.mp4")
    r = te.export_pieces(loaded, [(0, 360, 1.0, "original", 1.0)], output=out)
    assert r["ok"]
    assert duration(out) == pytest.approx(12.0, abs=0.15)
    assert has_audio(out)


@requires_ffmpeg
def test_regression_segments_with_speeds_keep_the_total(loaded, output_path):
    # old bug: the export came out with only part of the video. The total must match.
    pieces = [(0, 90, 1.0, "original", 1.0), (90, 180, 4.0, "mute", 1.0),
              (180, 360, 1.0, "original", 1.0)]
    out = output_path("full.mp4")
    r = te.export_pieces(loaded, pieces, output=out)
    assert r["ok"], r.get("msg")
    assert duration(out) == pytest.approx(3.0 + 0.75 + 6.0, abs=0.15)


@requires_ffmpeg
def test_speed_1x_keeps_every_frame(loaded, output_path):
    out = output_path("v1.mp4")
    assert te.export_pieces(loaded, [(0, 360, 1.0, "mute", 1.0)], output=out)["ok"]
    assert frames_of_video(out) == 360


@requires_ffmpeg
@pytest.mark.parametrize("speed", [0.5, 0.7, 1.5, 2.0, 3.0, 4.0, 8.0])
def test_speed_delivers_exact_frame_count(loaded, output_path, speed):
    # item 55: before, each accelerated segment came out with 1 to 2 extra frames
    out = output_path(f"v{speed}.mp4")
    assert te.export_pieces(loaded, [(0, 360, speed, "mute", 1.0)], output=out)["ok"]
    assert frames_of_video(out) == round(360 / speed)


@requires_ffmpeg
@pytest.mark.parametrize("speed", [2.0, 3.0, 0.5])
def test_speed_approximate_duration(loaded, output_path, speed):
    # tolerance of 0.15 s (about 5 frames)
    out = output_path(f"d{speed}.mp4")
    assert te.export_pieces(loaded, [(0, 360, speed, "mute", 1.0)], output=out)["ok"]
    assert duration(out) == pytest.approx(12.0 / speed, abs=0.15)


@requires_ffmpeg
def test_mute_level_has_no_sound(loaded, output_path):
    out = output_path("mute.mp4")
    te.export_pieces(loaded, [(0, 360, 1.0, "mute", 1.0)], output=out)
    assert max_volume(out) < -60


@requires_ffmpeg
def test_original_level_has_sound(loaded, output_path):
    out = output_path("sound.mp4")
    te.export_pieces(loaded, [(0, 360, 1.0, "original", 1.0)], output=out)
    assert max_volume(out) > -20


@requires_ffmpeg
def test_level_volume_lowers_the_sound(loaded, output_path):
    full, half = output_path("full_vol.mp4"), output_path("half_vol.mp4")
    te.export_pieces(loaded, [(0, 360, 1.0, "original", 1.0)], output=full)
    te.export_pieces(loaded, [(0, 360, 1.0, "original", 0.5)], output=half)
    # volume 0.5 is about -6 dB
    assert max_volume(full) - max_volume(half) == pytest.approx(6.0, abs=1.0)


@requires_ffmpeg
def test_video_without_audio_exports_without_audio(clips, output_path):
    info = te.probe(clips["mute"])
    out = output_path("noaudio.mp4")
    r = te.export_pieces(info, [(0, 120, 1.0, "original", 1.0)], output=out)
    assert r["ok"], r.get("msg")
    assert not has_audio(out)


@requires_ffmpeg
def test_progress_gets_close_to_1(loaded, output_path):
    values = []
    te.export_pieces(loaded, [(0, 180, 2.0, "original", 1.0), (180, 360, 1.0, "original", 1.0)],
                     output=output_path("prog.mp4"), progress=values.append)
    assert values
    assert all(0.0 <= v <= 1.0 for v in values)
    assert max(values) >= 0.9


@requires_ffmpeg
def test_default_name_with_date_and_time(loaded, clips, tmp_path):
    # without --output, the file goes next to the original video: a copy of the clip in the temporary folder
    file = tmp_path / "test_clip.mp4"
    shutil.copy(clips["av"], file)
    info = dict(loaded, path=str(file))
    r = te.export_pieces(info, [(0, 120, 1.0, "original", 1.0)])
    assert r["ok"], r.get("msg")
    assert re.match(r"^test_clip_timelapse_\d{8}-\d{4}\.mp4$", os.path.basename(r["output"]))


@requires_ffmpeg
def test_lossless_produces_a_larger_file_than_high(loaded, output_path):
    high, lossless = output_path("high.mp4"), output_path("lossless.mp4")
    te.export_pieces(loaded, [(0, 120, 1.0, "original", 1.0)], quality="high", output=high)
    te.export_pieces(loaded, [(0, 120, 1.0, "original", 1.0)], quality="lossless", output=lossless)
    assert os.path.getsize(lossless) > os.path.getsize(high)


@requires_ffmpeg
def test_refuses_speed_out_of_range(loaded, output_path):
    assert not te.export_pieces(loaded, [(0, 10, 999.0, "mute", 1.0)], output=output_path("x.mp4"))["ok"]


@requires_ffmpeg
def test_refuses_empty_list(loaded, output_path):
    assert not te.export_pieces(loaded, [], output=output_path("y.mp4"))["ok"]


@requires_ffmpeg
def test_creates_the_output_folder_when_missing(loaded, tmp_path):
    # project behavior: the output folder is created if missing (export_pieces uses makedirs)
    out = tmp_path / "new" / "x.mp4"
    r = te.export_pieces(loaded, [(0, 120, 1.0, "original", 1.0)], output=str(out))
    assert r["ok"], r.get("msg")
    assert out.exists()


@requires_ffmpeg
def test_size_estimate(loaded):
    body = {"pieces": [{"start": 0, "end": 4, "speed": 1, "audio": "original"}],
            "format": "mp4_h264", "quality": "high"}
    r = te.estimate_from_json(body)
    assert r["ok"], r.get("msg")
    assert r["bytes"] > 0
    assert r["total_s"] == pytest.approx(4.0, abs=0.02)


@requires_ffmpeg
def test_lossless_estimate_is_larger_than_high(loaded):
    base = {"pieces": [{"start": 0, "end": 4, "speed": 1, "audio": "original"}], "format": "mp4_h264"}
    high = te.estimate_from_json(dict(base, quality="high"))["bytes"]
    lossless = te.estimate_from_json(dict(base, quality="lossless"))["bytes"]
    assert lossless > high


def test_estimate_refuses_empty_list():
    assert not te.estimate_from_json({"pieces": []})["ok"]


@requires_ffmpeg
def test_background_export(loaded, tmp_path):
    out = str(tmp_path / "bg.mp4")
    r = te.export_async({"pieces": [{"start": 0, "end": 4, "speed": 1, "audio": "original"}],
                         "output": out})
    assert r["ok"]
    end = time.time() + 120
    while te.EXPORT_JOB["running"] and time.time() < end:
        time.sleep(0.1)
    assert not te.EXPORT_JOB["running"], "export did not finish in time"
    assert te.EXPORT_JOB["ok"], te.EXPORT_JOB["msg"]
    assert te.EXPORT_JOB["frac"] == 1.0
    assert os.path.exists(out)


# ---------------------------------------------------------------- frames and thumbnails (ffmpeg)

@requires_ffmpeg
def test_png_frame_name_and_signature(loaded, tmp_path):
    r = te.export_frame({"frame": 45, "format": "png", "folder": str(tmp_path)})
    assert r["ok"], r.get("msg")
    assert os.path.basename(r["output"]) == "test_clip_frame45.png"
    with open(r["output"], "rb") as f:
        assert f.read(4) == b"\x89PNG"


@requires_ffmpeg
def test_different_frames_make_different_images(loaded, tmp_path):
    a = te.export_frame({"frame": 0, "format": "png", "folder": str(tmp_path)})["output"]
    b = te.export_frame({"frame": 90, "format": "png", "folder": str(tmp_path)})["output"]
    with open(a, "rb") as fa, open(b, "rb") as fb:
        assert fa.read() != fb.read()


@requires_ffmpeg
def test_frame_does_not_overwrite(loaded, tmp_path):
    first = te.export_frame({"frame": 10, "format": "png", "folder": str(tmp_path)})["output"]
    second = te.export_frame({"frame": 10, "format": "png", "folder": str(tmp_path)})["output"]
    assert first != second
    assert second.endswith("_frame10_2.png")


@requires_ffmpeg
def test_jpg_with_higher_quality_takes_more_space(loaded, tmp_path):
    high = te.export_frame({"frame": 30, "format": "jpg", "quality": 95, "folder": str(tmp_path)})
    low = te.export_frame({"frame": 31, "format": "jpg", "quality": 10, "folder": str(tmp_path)})
    with open(high["output"], "rb") as f:
        assert f.read(2) == b"\xff\xd8"
    assert os.path.getsize(high["output"]) > os.path.getsize(low["output"])


@requires_ffmpeg
def test_frame_refuses_invalid_format_and_folder(loaded, tmp_path):
    assert not te.export_frame({"frame": 1, "format": "bmp", "folder": str(tmp_path)})["ok"]
    assert not te.export_frame({"frame": 1, "format": "png", "folder": "/does/not/exist"})["ok"]


@requires_ffmpeg
def test_frame_out_of_range_goes_to_the_last_one(loaded, tmp_path):
    r = te.export_frame({"frame": 9999, "format": "png", "folder": str(tmp_path)})
    assert r["ok"]
    assert r["output"].endswith("_frame359.png")


@requires_ffmpeg
def test_regression_last_frame_is_exported_as_image(loaded, tmp_path):
    # bug found by the tests: time-based seek did not return the last frame.
    # There is now a fallback that decodes from the start and selects by frame number.
    last = loaded["frames"] - 1
    assert te.get_thumb(last) is not None
    r = te.export_frame({"frame": last, "format": "png", "folder": str(tmp_path)})
    assert r["ok"], r.get("msg")
    assert os.path.getsize(r["output"]) > 1000


@requires_ffmpeg
def test_thumbnails_of_different_frames_are_different(loaded):
    t0, t60 = te.get_thumb(0), te.get_thumb(60)
    assert t0.startswith(b"\xff\xd8")
    assert t0 != t60


# ---------------------------------------------------------------- command line (ffmpeg)

@requires_ffmpeg
def test_cli_exports_from_the_segment_text(clips, tmp_path):
    out = str(tmp_path / "cli.mp4")
    r = run_cli(clips["av"], "--segments", "0-2 2x mute", "--output", out)
    assert r.returncode == 0, r.stderr
    assert "segments" in r.stdout
    assert os.path.exists(out)


@requires_ffmpeg
def test_cli_reads_segments_from_a_file(clips, tmp_path):
    segments_file = tmp_path / "segments.txt"
    segments_file.write_text("# first segment\n0-2 2x mute\n2-end 1x\n", encoding="utf-8")
    out = str(tmp_path / "file.mp4")
    r = run_cli(clips["av"], "--segments-file", str(segments_file), "--output", out)
    assert r.returncode == 0, r.stderr
    assert duration(out) == pytest.approx(1.0 + 10.0, abs=0.15)


@requires_ffmpeg
def test_cli_mov_format(clips, tmp_path):
    out = str(tmp_path / "cli.mov")
    r = run_cli(clips["av"], "--segments", "0-end 1x", "--format", "mov_h264", "--output", out)
    assert r.returncode == 0, r.stderr
    assert "mov" in format_and_codec(out)[1]


@requires_ffmpeg
def test_cli_accepts_lossless_quality(clips, tmp_path):
    out = str(tmp_path / "lossless.mp4")
    r = run_cli(clips["av"], "--segments", "0-end 1x", "--quality", "lossless", "--output", out)
    assert r.returncode == 0, r.stderr


def test_cli_refuses_invalid_quality(clips):
    r = run_cli(clips["av"], "--segments", "0-end 1x", "--quality", "exaggerated")
    assert r.returncode != 0


@requires_ffmpeg
def test_cli_invalid_text_exits_with_error(clips):
    r = run_cli(clips["av"], "--segments", "nonsense")
    assert r.returncode == 1
    assert "ERROR" in r.stderr


def test_cli_missing_video_file(tmp_path):
    r = run_cli(str(tmp_path / "does_not_exist.mp4"), "--segments", "0-1 1x")
    assert r.returncode == 1
    assert "not found" in r.stderr


@requires_ffmpeg
def test_export_with_corrected_audio_keeps_the_sound(loaded, output_path):
    out = output_path("corrected.mp4")
    r = te.export_pieces(loaded, [(0, 360, 2.0, "corrected", 1.0)], output=out)
    assert r["ok"], r.get("msg")
    assert has_audio(out)
    assert max_volume(out) > -30


@requires_ffmpeg
def test_cli_accepts_resolution(clips, tmp_path):
    out = str(tmp_path / "res.mp4")
    r = run_cli(clips["av"], "--segments", "0-end 1x", "--resolution", "720p", "--output", out)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(out)


@requires_ffmpeg
def test_motion_is_zero_while_still(tmp_path):
    # 2 s moving, 2 s of a flat colour (still), 2 s moving again. Values are every 0.1 s.
    out = str(tmp_path / "motion.mp4")
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
        "-f", "lavfi", "-i", "color=c=gray:size=320x240:rate=30:duration=2",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
        "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", out]
    subprocess.run(cmd, check=True)
    info = te.probe(out)
    levels = te.compute_motion(info["path"], info["duration"])
    assert len(levels) >= int(info["duration"] / te.STEP_MOTION)
    assert max(levels[:15]) > 0.01          # first moving part
    assert all(v == 0 for v in levels[25:35])   # still part, away from the cuts
    assert max(levels[45:]) > 0.01          # moving again


@requires_ffmpeg
def test_analysis_cache_is_used_the_second_time(tmp_path, monkeypatch):
    import timelapse_editor.analysis as an
    clip = str(tmp_path / "clip.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", clip], check=True)
    d = str(tmp_path / "entry")
    os.makedirs(d)
    first = an.analyze(clip, 2.0, d)
    assert os.path.exists(os.path.join(d, "thumbs.done"))
    # with the cache in place, no decode may run again
    monkeypatch.setattr(an, "run_pass", lambda *a, **k: pytest.fail("the video was read again"))
    assert an.analyze(clip, 2.0, d) == first


@requires_ffmpeg
def test_analysis_recomputes_when_the_series_version_changes(tmp_path, monkeypatch):
    import timelapse_editor.analysis as an
    clip = str(tmp_path / "clip.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", clip], check=True)
    d = str(tmp_path / "entry")
    os.makedirs(d)
    an.write_store(d, {"motion": {"version": 0, "values": [9.9]}})
    store = an.analyze(clip, 2.0, d, thumbs=False)
    values = an.values_of(store, "motion")
    assert values[0] == 0.0 and max(values) < 1
    assert store["motion"]["version"] == an.SERIES_VERSION["motion"]
    assert store["brightness"]["version"] == an.SERIES_VERSION["brightness"]


def test_fingerprint_changes_with_the_file(tmp_path):
    import timelapse_editor.analysis as an
    f = tmp_path / "v.mp4"
    f.write_bytes(b"abc")
    before = an.fingerprint(str(f))
    f.write_bytes(b"abcdef")
    assert an.fingerprint(str(f)) != before


def _make_clip(path, *parts):
    """Concatenates the given lavfi sources, 2 s each, into one video."""
    inputs, chains = [], []
    for i, src in enumerate(parts):
        inputs += ["-f", "lavfi", "-i", src]
        chains.append(f"[{i}:v]trim=duration=2,setpts=PTS-STARTPTS[p{i}]")
    fc = ";".join(chains) + ";" + "".join(f"[p{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[v]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", fc, "-map", "[v]",
                    "-r", "30", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", path],
                   check=True)
    return path


@requires_ffmpeg
def test_brightness_and_sharpness_follow_the_picture(tmp_path):
    clip = _make_clip(str(tmp_path / "scenes.mp4"),
                      "color=c=gray:size=320x240:rate=30",          # flat mid grey: no detail
                      "testsrc2=size=320x240:rate=30")               # full of detail
    s, ok = te.analysis.run_pass(clip, 4.0)
    assert ok
    b, sh = s["brightness"], s["sharpness"]
    assert abs(b[5] - 0.5) < 0.02                     # grey is in the middle of black and white
    assert sh[5] < 0.001                              # flat: no detail
    assert max(sh[25:35]) > 0.005                     # the detailed part has detail


@requires_ffmpeg
def test_scene_cuts_are_found_and_still_video_has_none(tmp_path):
    cuts = _make_clip(str(tmp_path / "cuts.mp4"),
                      "testsrc2=size=320x240:rate=30",
                      "mandelbrot=size=320x240:rate=30",
                      "color=c=gray:size=320x240:rate=30")
    s, _ = te.analysis.run_pass(cuts, 6.0)
    assert te.analysis.detect_cuts(s["motion"]) == [2.0, 4.0]
    still = _make_clip(str(tmp_path / "still.mp4"), "testsrc2=size=320x240:rate=30",
                       "testsrc2=size=320x240:rate=30")
    s, _ = te.analysis.run_pass(still, 4.0)
    assert te.analysis.detect_cuts(s["motion"]) == []
