"""
Interface tests (pytest + Playwright, Chromium).

Starts servers on their own ports, with --no-browser:
  - one with the 12 s clip loaded (tone only between 4 s and 6 s);
  - one without a video (import and project without a video);
  - one with the 8 min clip, for the waveform at high zoom.

Run from the project folder:
    uv run pytest -v

Needs ffmpeg on PATH, Playwright for Python and an installed Chromium.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import pytest
from playwright.sync_api import sync_playwright

import timelapse_editor as te
from conftest import requires_ffmpeg

PORT_VIDEO, PORT_EMPTY, PORT_LONG = 8791, 8792, 8793
READY = "() => typeof S !== 'undefined' && S.real === true && typeof AUD !== 'undefined' && AUD.length > 0"


# ---------------------------------------------------------------- servers and browser

def wait_for_server(port, limit=60):
    end = time.time() + limit
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
                r.read()
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"server did not start on port {port}")


def start_server(port, log_dir, video=None):
    args = [sys.executable, "-m", "timelapse_editor"]
    if video:
        args.append(video)
    args += ["--port", str(port), "--no-browser"]
    log = open(os.path.join(log_dir, f"server_{port}.log"), "w")
    # the analysis cache goes to a folder of the test run, not to the user's cache
    env = dict(os.environ, XDG_CACHE_HOME=os.path.join(log_dir, "cache"))
    proc = subprocess.Popen(args, stdout=log, stderr=log, env=env)
    wait_for_server(port)
    return proc


@pytest.fixture(scope="session")
def servers(clips, tmp_path_factory):
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("needs ffmpeg and ffprobe on PATH")
    log_dir = str(tmp_path_factory.mktemp("logs"))
    procs = [start_server(PORT_VIDEO, log_dir, clips["av"]),
             start_server(PORT_EMPTY, log_dir),
             start_server(PORT_LONG, log_dir, clips["long"])]
    yield
    for p in procs:
        p.terminate()
        p.wait(timeout=20)


@pytest.fixture(scope="session")
def browser(servers):
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def open_page(browser):
    contexts = []

    def _open(port):
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        contexts.append(ctx)
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append("JS: " + str(e)))
        # any HTTP error is a failure (including /api/thumb without a video)
        page.on("response", lambda r: errors.append(f"HTTP {r.status}: {r.url}") if r.status >= 400 else None)
        page.goto(f"http://127.0.0.1:{port}/")
        return page, errors

    yield _open
    for ctx in contexts:
        ctx.close()


def wait_ready(page):
    page.wait_for_function(READY, timeout=60000)


def state(page, expr):
    return page.evaluate(f"() => ({expr})")


def segments(page):
    return state(page, "timeline().map(g => ({id: g.id, start: g.start, end: g.end, level: g.level}))")


def block(page, i):
    sid = segments(page)[i]["id"]
    return page.locator(f'#lanes .blk[data-key="{sid}"]')


def drag(page, x0, y0, x1, y1, steps=12):
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x1, y1, steps=steps)
    page.mouse.up()


def center(loc):
    b = loc.bounding_box()
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


def no_errors(errors):
    assert errors == [], errors


# ---------------------------------------------------------------- opening and initial state

@requires_ffmpeg
def test_page_loads_with_version_and_video(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    assert page.text_content("#ver").startswith("v")
    info = state(page, "({fps: FPS, d: D, name: VIDEO_NAME})")
    assert info["fps"] == pytest.approx(30.0, abs=0.01)
    assert info["d"] == pytest.approx(12.0, abs=0.1)
    assert info["name"] == "test_clip.mp4"
    no_errors(errors)


@requires_ffmpeg
def test_initial_state_three_1x_segments_with_cuts_at_1_3_and_2_3(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    pieces = state(page, "timeline().map(p => ({start: p.start, end: p.end, level: p.level}))")
    d = state(page, "D")
    fps = state(page, "FPS")
    q = lambda v: round(v * fps) / fps
    assert len(pieces) == 3
    assert pieces[0]["start"] == 0
    assert pieces[0]["end"] == pytest.approx(q(d / 3), abs=1e-6)
    assert pieces[1]["end"] == pytest.approx(q(2 * d / 3), abs=1e-6)
    assert pieces[2]["end"] == pytest.approx(d, abs=1e-6)
    assert all(p["level"] == "l1x" for p in pieces)
    no_errors(errors)


@requires_ffmpeg
def test_automatic_gain_on_load(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    r = state(page, "({gain: S.gain, peak: Math.max(...AUD)})")
    expected = min(20, 1 / r["peak"]) if r["peak"] > 0 else 1
    assert r["gain"] == pytest.approx(expected, rel=1e-9)
    no_errors(errors)


@requires_ffmpeg
def test_project_with_video_applies_the_segments(open_page, clips, tmp_path):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    d = state(page, "D")
    # project saved in the old format (Portuguese keys): it must still open
    project = {"duration": d,
               "segs": [{"id": "a", "start": 0, "end": 4, "level": "l4"},
                        {"id": "b", "start": 4, "end": d, "level": "l1x"}],
               "lanes": [{"id": "l1x", "speed": 1, "audio": "original"},
                         {"id": "l4", "speed": 4, "audio": "mudo"},
                         {"id": "ign", "ignored": True}],
               "marks": [], "th": 0.4,
               "video": {"name": "test_clip.mp4", "path": clips["av"]}}
    file = tmp_path / "project_ok.json"
    file.write_text(json.dumps(project), encoding="utf-8")
    page.set_input_files("#pfile", str(file))
    page.wait_for_timeout(1500)
    pieces = state(page, "timeline().map(p => [p.start, p.end, p.level])")
    assert len(pieces) == 2, pieces
    assert pieces[0][2] == "l4"
    assert pieces[0][1] == pytest.approx(4.0, abs=1e-6)
    assert state(page, "S.lanes.find(l => l.id === 'l4').audio") == "mute"
    no_errors(errors)


@requires_ffmpeg
def test_saved_project_uses_version_2_and_english_keys(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    # headless Chromium would wait forever in the native save window: use the download fallback
    page.evaluate("() => { window.showSaveFilePicker = undefined; }")
    with page.expect_download() as info:
        page.click("#saveBtn")
    saved = json.loads(open(info.value.path(), encoding="utf-8").read())
    assert saved["version"] == 2
    for key in ("segments", "lanes", "marks", "threshold", "video"):
        assert key in saved, key
    assert "segs" not in saved and "th" not in saved
    no_errors(errors)


# ---------------------------------------------------------------- graphs menu

def test_graphs_menu_shows_and_hides_the_movement_row(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    assert page.locator("#tiles-motion").count() == 1
    page.click("#graphsBtn")
    page.wait_for_selector("#graphMenu:not([hidden])")
    page.uncheck('#graphMenu input[data-graph="motion"]')
    assert page.locator("#tiles-motion").count() == 0
    assert state(page, "S.graphs.motion") is False
    page.check('#graphMenu input[data-graph="motion"]')
    assert page.locator("#tiles-motion").count() == 1
    no_errors(errors)


def test_saved_project_keeps_the_graphs_choice(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    page.evaluate("() => { window.showSaveFilePicker = undefined; S.graphs.motion = false; }")
    with page.expect_download() as info:
        page.click("#saveBtn")
    saved = json.loads(open(info.value.path(), encoding="utf-8").read())
    assert saved["graphs"] == {"motion": False}
    no_errors(errors)


def test_scene_cut_marks_are_added_from_the_graphs_menu(open_page):
    # the simulated video has cuts at 30 s and 60 s
    page, errors = open_page(PORT_EMPTY)
    wait_ready_sim = "() => typeof S !== 'undefined' && S.real === false && typeof SER !== 'undefined'"
    page.wait_for_function(wait_ready_sim, timeout=60000)
    page.click("#graphsBtn")
    page.check('#graphMenu input[data-graph="brightness"]')
    page.check('#graphMenu input[data-graph="cuts"]')
    page.click('#graphrows button[data-act="cutmarks"]')
    marks = state(page, "S.marks.map(m => m.t)")
    assert 30 in marks and 60 in marks, marks
    no_errors(errors)


@requires_ffmpeg
def test_camera_graph_is_computed_when_turned_on(open_page):
    pytest.importorskip("cv2")
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    page.click("#graphsBtn")
    page.check('#graphMenu input[data-graph="camera"]')
    page.wait_for_function("() => SER.camera && SER.camera.length > 0", timeout=240000)
    assert page.locator("#tiles-camera").count() == 1
    assert max(state(page, "SER.camera")) > 0
    no_errors(errors)


# ---------------------------------------------------------------- keyboard

@requires_ffmpeg
def test_ctrl_arrows_move_the_cursor_10_frames_and_shift_1(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    page.evaluate("() => document.activeElement && document.activeElement.blur()")
    page.keyboard.press("Control+ArrowRight")
    k1 = state(page, "S.kbT")
    page.keyboard.press("Control+ArrowRight")
    k2 = state(page, "S.kbT")
    fps = state(page, "FPS")
    assert k1 is not None and k2 is not None
    assert k2 - k1 == pytest.approx(10 / fps, abs=1e-6)
    page.keyboard.press("Control+Shift+ArrowRight")
    k3 = state(page, "S.kbT")
    assert k3 - k2 == pytest.approx(1 / fps, abs=1e-6)
    no_errors(errors)


@requires_ffmpeg
def test_enter_with_provisional_cursor_seeks_and_does_not_start_play(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    page.focus("#play")
    page.keyboard.press("Control+ArrowRight")
    page.keyboard.press("Enter")
    page.wait_for_timeout(400)
    assert state(page, "vid.paused") is True, "Enter started Play"
    assert state(page, "S.kbT") is None
    no_errors(errors)


@requires_ffmpeg
def test_p_and_space_toggle_play_and_pause(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    playing = lambda: state(page, "!vid.paused")   # with a real video, play belongs to the <video>
    page.evaluate("() => document.activeElement && document.activeElement.blur()")
    page.keyboard.press("p")
    page.wait_for_timeout(600)
    assert playing() is True, "P did not start"
    page.keyboard.press("p")
    page.wait_for_timeout(300)
    assert playing() is False, "P did not pause"
    page.evaluate("() => document.activeElement && document.activeElement.blur()")
    page.keyboard.press(" ")
    page.wait_for_timeout(600)
    assert playing() is True, "space did not start"
    page.keyboard.press(" ")
    page.wait_for_timeout(300)
    assert playing() is False, "space did not pause"
    no_errors(errors)


@requires_ffmpeg
def test_time_field_applies_with_enter_and_cancels_with_esc(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    page.fill("#tin", "0:02.5")
    page.press("#tin", "Enter")
    assert state(page, "S.T") == pytest.approx(2.5, abs=0.04)
    page.fill("#tin", "0:09.0")
    page.press("#tin", "Escape")
    assert page.input_value("#tin") == state(page, "fmt(S.T)")
    assert state(page, "S.T") == pytest.approx(2.5, abs=0.04), "Esc applied the time"
    no_errors(errors)


# ---------------------------------------------------------------- mouse: blocks, cuts, selection

@requires_ffmpeg
def test_dragging_the_right_edge_changes_the_segment_end(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    before = segments(page)[0]
    pxs = state(page, "S.pxPerSec")
    b = block(page, 0).bounding_box()
    drag(page, b["x"] + b["width"] - 3, b["y"] + b["height"] / 2,
         b["x"] + b["width"] - 3 + 60, b["y"] + b["height"] / 2)
    after = segments(page)[0]
    assert after["end"] - before["end"] == pytest.approx(60 / pxs, abs=0.15)
    no_errors(errors)


@requires_ffmpeg
def test_dragging_the_left_edge_changes_the_segment_start(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    before = segments(page)[1]
    pxs = state(page, "S.pxPerSec")
    b = block(page, 1).bounding_box()
    drag(page, b["x"] + 3, b["y"] + b["height"] / 2, b["x"] + 3 - 40, b["y"] + b["height"] / 2)
    after = segments(page)[1]
    assert after["start"] == pytest.approx(before["start"] - 40 / pxs, abs=0.15)
    no_errors(errors)


@requires_ffmpeg
def test_esc_cancels_an_edge_drag(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    before = segments(page)
    b = block(page, 0).bounding_box()
    x, y = b["x"] + b["width"] - 3, b["y"] + b["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 80, y, steps=10)
    page.keyboard.press("Escape")
    page.mouse.move(x + 80, y, steps=2)
    page.mouse.up()
    assert segments(page) == before
    no_errors(errors)


@requires_ffmpeg
def test_dragging_a_block_to_another_level(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    b = block(page, 0)
    x, y = center(b)
    target = page.locator('#lanes .row[data-lane="l4"] .track')
    _, ya = center(target)
    drag(page, x, y, x, ya, steps=15)
    assert segments(page)[0]["level"] == "l4"
    no_errors(errors)


@requires_ffmpeg
def test_dragging_a_block_to_ignored_removes_it_from_the_export(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    before = state(page, "exportPieces().length")
    x, y = center(block(page, 0))
    _, yi = center(page.locator('#lanes .row[data-lane="ign"] .track'))
    drag(page, x, y, x, yi, steps=15)
    assert segments(page)[0]["level"] == "ign"
    assert state(page, "exportPieces().length") == before - 1
    no_errors(errors)


@requires_ffmpeg
def test_selection_with_ctrl_and_shift(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    block(page, 0).click()
    assert len(state(page, "S.msel")) == 1
    block(page, 2).click(modifiers=["Shift"])
    assert len(state(page, "S.msel")) == 3
    no_errors(errors)


@requires_ffmpeg
def test_cut_at_cursor(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    block(page, 1).click()
    page.evaluate("() => seek(6.0)")
    page.click("#cut")
    assert len(segments(page)) == 4
    assert any(abs(s["start"] - 6.0) < 0.05 for s in segments(page))
    no_errors(errors)


@requires_ffmpeg
def test_merge_joins_adjacent_segments_and_undo_restores(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    block(page, 0).click()
    block(page, 1).click(modifiers=["Control"])
    assert page.is_visible("#confirm")
    page.click("#confirm")
    assert len(segments(page)) == 2
    page.click("#undo")
    assert len(segments(page)) == 3
    no_errors(errors)


@requires_ffmpeg
def test_merge_refuses_different_levels(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    x, y = center(block(page, 1))
    _, y4 = center(page.locator('#lanes .row[data-lane="l4"] .track'))
    drag(page, x, y, x, y4, steps=15)
    block(page, 0).click()
    block(page, 1).click(modifiers=["Control"])
    assert not page.is_visible("#confirm")
    no_errors(errors)


# ---------------------------------------------------------------- levels, volume, deletion

@requires_ffmpeg
def test_editing_a_level_changes_speed_and_audio(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    page.click('#lanes .row[data-lane="l4"] button[data-act="edit"]')
    page.select_option('#lanes .row[data-lane="l4"] .pop select[data-act="speed"]', "2")
    page.select_option('#lanes .row[data-lane="l4"] .pop select[data-act="audio"]', "corrected")
    lane = state(page, "S.lanes.find(l => l.id === 'l4')")
    assert lane["speed"] == 2
    assert lane["audio"] == "corrected"
    no_errors(errors)


@requires_ffmpeg
def test_level_volume_goes_to_the_export(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    x, y = center(block(page, 0))
    _, y4 = center(page.locator('#lanes .row[data-lane="l4"] .track'))
    drag(page, x, y, x, y4, steps=15)
    page.click('#lanes .row[data-lane="l4"] button[data-act="edit"]')
    vol = page.locator('#lanes .row[data-lane="l4"] .pop input[data-act="vol"]')
    vol.evaluate("el => { el.value = '0.5'; el.dispatchEvent(new Event('input', {bubbles: true}));"
                 " el.dispatchEvent(new Event('change', {bubbles: true})); }")
    volumes = state(page, "exportPieces().map(p => p.volume)")
    assert 0.5 in volumes
    no_errors(errors)


@requires_ffmpeg
def test_delete_level_only_for_extra_levels(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    page.click('#lanes .row[data-lane="l1x"] button[data-act="edit"]')
    assert page.locator('#lanes .row[data-lane="l1x"] .pop button[data-act="del"]').count() == 0
    assert page.locator('#lanes .row[data-lane="ign"] button[data-act="edit"]').count() == 0
    page.click('#lanes .row[data-lane="l8"] button[data-act="edit"]')
    page.click('#lanes .row[data-lane="l8"] .pop button[data-act="del"]')
    assert "l8" not in state(page, "S.lanes.map(l => l.id)")
    no_errors(errors)


# ---------------------------------------------------------------- marker, help, audio

@requires_ffmpeg
def test_marker_at_cursor(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    page.evaluate("() => seek(3.0)")
    # seek is asynchronous, and S.T can go back to 0 for a frame before the video is positioned.
    # Waits for the cursor to stay at 3 s for 10 consecutive readings (about 0.5 s), not just for an instant.
    # Checked from Python in a loop: the same check inside wait_for_function did not settle in full runs.
    readings = []
    stable, deadline = 0, time.time() + 30
    while time.time() < deadline and stable < 10:
        v = state(page, "({video: vid.currentTime, cursor: S.T})")
        readings.append(v)
        ok = abs(v["video"] - 3.0) < 0.05 and abs(v["cursor"] - 3.0) < 0.05
        stable = stable + 1 if ok else 0
        page.wait_for_timeout(50)
    assert stable >= 10, f"cursor did not settle at 3 s; last readings: {readings[-5:]}"
    page.click("#addMark")
    marks = state(page, "S.marks.map(m => m.t)")
    assert len(marks) == 1 and marks[0] == pytest.approx(3.0, abs=0.05)
    no_errors(errors)


@requires_ffmpeg
def test_help_opens_switches_tab_and_closes_with_esc(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    page.click("#helpBtn")
    assert page.is_visible("#modal")
    page.click('.mtabs button[data-tab="tutorial"]')
    assert "on" in page.get_attribute('.mtabs button[data-tab="tutorial"]', "class")
    page.keyboard.press("Escape")
    assert not page.is_visible("#modal")
    no_errors(errors)


@requires_ffmpeg
def test_mark_speech_starts_detects_the_tone(open_page):
    # the clip has a tone between 4 s and 6 s and silence elsewhere: speech must start at 4 s
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    page.click('button[data-act="audio"]')
    page.click('button[data-act="speechMark"]')
    marks = state(page, "S.marks.map(m => m.t)")
    assert len(marks) == 1
    assert marks[0] == pytest.approx(4.0, abs=0.3)
    no_errors(errors)


@requires_ffmpeg
def test_speech_regions_with_default_threshold(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    regs = state(page, "regions(S.threshold)")
    assert len(regs) == 1
    assert regs[0][0] == pytest.approx(4.0, abs=0.3)
    assert regs[0][1] == pytest.approx(6.0, abs=0.3)
    no_errors(errors)


# ---------------------------------------------------------------- high zoom waveform

@requires_ffmpeg
def test_waveform_at_high_zoom_reaches_7_min_30(open_page):
    page, errors = open_page(PORT_LONG)
    wait_ready(page)
    page.evaluate("() => { S.pxPerSec = 200; render(); }")
    page.wait_for_timeout(500)
    r = state(page, """(() => {
        const cs = [...document.querySelectorAll('#wavetiles canvas')];
        const W = Math.round(D * S.pxPerSec);
        const x = Math.round(450 * S.pxPerSec);           // 7 min 30 s
        const tile = Math.floor(x / 16000);
        const c = cs[tile];
        const d = c.getContext('2d').getImageData(x - tile * 16000, 0, 1, 44).data;
        let painted = 0;
        for (let i = 3; i < d.length; i += 4) if (d[i] > 0) painted++;
        return {parts: cs.length, expected: Math.ceil(W / 16000), painted};
    })()""")
    assert r["parts"] == r["expected"]
    assert r["painted"] > 0, "waveform does not appear at 7 min 30 s"
    no_errors(errors)


@requires_ffmpeg
def test_levels_have_one_track_each_at_high_zoom(open_page):
    page, errors = open_page(PORT_LONG)
    wait_ready(page)
    page.evaluate("() => { S.pxPerSec = 200; render(); }")
    page.wait_for_timeout(300)
    n_levels = state(page, "S.lanes.length")
    assert page.locator("#lanes .row").count() == n_levels
    for i in range(n_levels):
        assert page.locator("#lanes .row").nth(i).locator(".track").count() == 1
    no_errors(errors)


# ---------------------------------------------------------------- frame and export

@requires_ffmpeg
def test_jpg_frame_with_quality_and_no_overwrite(open_page, clips):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    page.evaluate("() => seek(4)")
    page.click("#frameBtn")
    page.wait_for_selector("#browser", state="visible")
    page.select_option("#frameFmt", "jpg")
    page.locator("#frameQ").evaluate("el => { el.value = '50'; el.dispatchEvent(new Event('input', {bubbles: true})); }")
    page.click("#brSave")
    page.wait_for_timeout(800)
    # the panel closes after saving: reopens it for the second, identical frame
    page.click("#frameBtn")
    page.wait_for_selector("#browser", state="visible")
    page.click("#brSave")
    page.wait_for_timeout(800)
    folder = os.path.dirname(clips["av"])
    files = sorted(f for f in os.listdir(folder) if "_frame" in f and f.endswith(".jpg"))
    assert len(files) >= 2, files
    with open(os.path.join(folder, files[0]), "rb") as f:
        assert f.read(2) == b"\xff\xd8"
    no_errors(errors)


@requires_ffmpeg
def test_export_from_the_interface(open_page, clips):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    estimates = []
    page.on("request", lambda r: estimates.append(r.url) if r.url.endswith("/api/estimate") else None)
    page.click("#exportBtn")
    page.wait_for_selector("#expbox", state="visible")
    page.wait_for_function("() => [...document.querySelectorAll('[data-size]')]"
                           ".every(e => e.textContent.startsWith('≈'))", timeout=60000)
    assert estimates, "size preview was not requested"
    page.click("#expGo")
    page.wait_for_selector("#expClose", state="visible", timeout=180000)
    assert not page.is_visible("#expGo"), "Export button is still visible"
    assert not page.is_visible("#expCancel"), "Cancel button is still visible"
    folder = os.path.dirname(clips["av"])
    outputs = [f for f in os.listdir(folder) if f.startswith("test_clip_timelapse_")]
    assert outputs, "file without _YYYYMMDD-hhmm"
    assert te.probe(os.path.join(folder, outputs[-1]))["duration"] == pytest.approx(12.0, abs=0.3)
    estimates.clear()
    page.wait_for_timeout(2500)
    assert estimates == [], "a new estimate was requested after the export"
    page.click("#expClose")
    assert not page.is_visible("#expbox")
    no_errors(errors)


# ---------------------------------------------------------------- without video (import and project)

def test_import_opens_the_file_window(open_page):
    page, errors = open_page(PORT_EMPTY)
    page.click("#loadBtn")
    page.wait_for_selector("#browser", state="visible")
    assert page.text_content("#brTitle").strip()
    no_errors(errors)


def test_project_without_video_asks_for_import_with_the_saved_name(open_page, tmp_path):
    page, errors = open_page(PORT_EMPTY)
    file = tmp_path / "missing_project.json"
    file.write_text(json.dumps({
        "version": 2, "duration": 12, "segments": [], "lanes": [], "marks": [], "threshold": 0.4,
        "video": {"name": "video_that_does_not_exist.mp4",
                  "path": str(tmp_path / "does_not_exist" / "video_that_does_not_exist.mp4")}}), encoding="utf-8")
    page.set_input_files("#pfile", str(file))
    page.wait_for_selector("#browser", state="visible", timeout=15000)
    assert "Import the video: video_that_does_not_exist.mp4" in page.text_content("#brTitle")
    no_errors(errors)


@requires_ffmpeg
def test_esc_cancels_the_provisional_cursor(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    page.evaluate("() => document.activeElement && document.activeElement.blur()")
    page.keyboard.press("Control+ArrowRight")
    assert state(page, "S.kbT") is not None
    page.keyboard.press("Escape")
    assert state(page, "S.kbT") is None
    no_errors(errors)


@requires_ffmpeg
def test_edited_level_speed_applies_to_the_export(open_page):
    page, errors = open_page(PORT_VIDEO)
    wait_ready(page)
    x, y = center(block(page, 0))
    _, y4 = center(page.locator('#lanes .row[data-lane="l4"] .track'))
    drag(page, x, y, x, y4, steps=15)
    page.click('#lanes .row[data-lane="l4"] button[data-act="edit"]')
    page.select_option('#lanes .row[data-lane="l4"] .pop select[data-act="speed"]', "8")
    pieces = state(page, "exportPieces().map(p => p.speed)")
    assert 8 in pieces
    no_errors(errors)
