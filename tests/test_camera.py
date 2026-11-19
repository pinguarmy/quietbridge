"""Camera thread lifecycle with fake captures: no device, no MediaPipe model."""
import sys
import time

import numpy as np

from lipflow import camera


class FakeCap:
    def __init__(self, frames=0, opened=True, fps=0.0):
        self.frames, self.opened, self.fps = frames, opened, fps
        self.released = False
        self.reads = 0

    def isOpened(self):
        return self.opened

    def read(self):
        self.reads += 1
        if self.reads <= self.frames:
            return True, np.zeros((48, 64, 3), np.uint8)
        return False, None

    def set(self, *a):
        return True

    def get(self, prop):
        return self.fps

    def release(self):
        self.released = True


class FakeTracker:
    def detect(self, frame, ts):
        return None


def _wait_stopped(cam, timeout=5.0):
    end = time.time() + timeout
    while cam.is_open and time.time() < end:
        time.sleep(0.02)
    return not cam.is_open


def test_a_camera_that_stops_sending_frames_ends_with_an_error(monkeypatch):
    cap = FakeCap(frames=5)
    monkeypatch.setattr(camera, "NO_FRAME_TIMEOUT", 0.2)
    monkeypatch.setattr(camera, "FaceTracker", FakeTracker)
    cam = camera.Camera(0, idle_close=60.0)
    monkeypatch.setattr(cam, "_open", lambda: cap)
    cam.ensure_open()
    cam.start_recording()  # an active recording must not keep a dead camera spinning forever
    assert _wait_stopped(cam), "capture thread kept spinning on failed reads"
    assert cap.released and "stopped sending frames" in cam.error


def test_failed_reads_still_close_an_idle_camera(monkeypatch):
    cap = FakeCap(frames=0)
    monkeypatch.setattr(camera, "NO_FRAME_TIMEOUT", 60.0)
    monkeypatch.setattr(camera, "FaceTracker", FakeTracker)
    cam = camera.Camera(0, idle_close=0.1)
    monkeypatch.setattr(cam, "_open", lambda: cap)
    cam.ensure_open()
    assert _wait_stopped(cam) and cap.released and cam.error is None


def test_tracker_failure_releases_the_camera(monkeypatch):
    cap = FakeCap(frames=100)

    def broken_tracker():
        raise RuntimeError("face_landmarker.task not found")

    monkeypatch.setattr(camera, "FaceTracker", broken_tracker)
    cam = camera.Camera(0)
    monkeypatch.setattr(cam, "_open", lambda: cap)
    cam.ensure_open()
    assert _wait_stopped(cam)
    assert cap.released and "face_landmarker" in cam.error


def test_video_file_stand_in_stops_at_its_end(monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    cap = FakeCap(frames=4, fps=200.0)
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *a: cap)
    monkeypatch.setattr(camera, "FaceTracker", FakeTracker)
    cam = camera.Camera(str(clip), idle_close=60.0)
    cam.ensure_open()
    assert _wait_stopped(cam) and cap.released and cam.error is None


def test_unopenable_camera_is_released_and_reported(monkeypatch):
    cap = FakeCap(opened=False)
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *a: cap)
    monkeypatch.setattr(camera, "list_cameras", lambda: [])
    cam = camera.Camera(0)
    cam.ensure_open()
    assert _wait_stopped(cam)
    assert cap.released and "Could not open the camera" in cam.error


def test_linux_uses_opencvs_default_backend_and_lists_no_names(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(camera, "WINDOWS", False)
    assert camera._backend() == camera.cv2.CAP_ANY
    assert camera.list_cameras() == []  # must not import macOS AVFoundation
    assert camera.resolve_camera("auto") == 0
