import http.client
import json
import threading
import time
from types import SimpleNamespace

import numpy as np

from hackathon.contracts import LipResult
from hackathon.lips_adapter import LipsAdapter
from hackathon.runtime import Runtime, clock_ms
from hackathon.server import make_server
from hackathon.signs.recognizer import SignRecognizer


def wait_for(condition, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached")


def test_sign_tracking_uses_backend_result_and_handles_no_hands():
    landmark = SimpleNamespace(x=0.1, y=0.2, z=-0.03)
    result = SimpleNamespace(
        hand_landmarks=[[landmark]],
        handedness=[[SimpleNamespace(category_name="Left")]],
        gestures=[[SimpleNamespace(category_name="Thumb_Up", score=0.9)]],
    )

    class Backend:
        def recognize_for_video(self, image, timestamp_ms):
            return result

    class Stabilizer:
        def update(self, label, score, timestamp_ms):
            return []

    recognizer = SignRecognizer.__new__(SignRecognizer)
    recognizer._backend = Backend()
    recognizer._personal = None
    recognizer._motion = None
    recognizer._motion_frames = None
    recognizer._motion_start_ms = None
    recognizer._motion_last_boundary_ms = None
    recognizer._stabilizer = Stabilizer()
    recognizer._last_timestamp_ms = None
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    assert recognizer.process(frame, 1) == []
    assert recognizer.last_tracking == [{
        "handedness": "Left", "landmarks": [[0.1, 0.2, -0.03]],
        "label": "THUMB_UP", "score": 0.9,
    }]

    # Some lightweight fakes omit hand_landmarks entirely when no hand is found.
    result = SimpleNamespace(gestures=[], handedness=[])
    assert recognizer.process(frame, 2) == []
    assert recognizer.last_tracking == []


def test_lips_track_idle_normalize_and_append_only_while_recording(monkeypatch):
    outer = np.array([[40, 20], [80, 40], [60, 60]], dtype=np.float32)
    inner = np.array([[50, 35], [70, 45]], dtype=np.float32)
    obs = SimpleNamespace(outer_lips=outer, inner_lips=inner,
                          anchors="anchors", mouth_open=0.25)

    class Tracker:
        def __init__(self):
            self.obs = obs
            self.calls = []

        def detect(self, frame, timestamp_ms):
            self.calls.append(timestamp_ms)
            return self.obs

    adapter = LipsAdapter.__new__(LipsAdapter)
    adapter.tracker = Tracker()
    adapter.recording = None
    adapter.last_tracking = None
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    adapter.process(frame, 10)
    assert adapter.tracker.calls == [10]
    assert adapter.last_tracking == {
        "mouth_box": [0.2, 0.2, 0.2, 0.4],
        "outer_lips": [[0.2, 0.2], [0.4, 0.4], [0.3, 0.6]],
        "inner_lips": [[0.25, 0.35], [0.35, 0.45]],
    }

    rec = SimpleNamespace(ts=[], grays=[], anchors=[], mouth_open=[], _box=None)
    monkeypatch.setattr("lipflow.camera.face_crop", lambda gray, obs, rec: "crop")
    adapter.recording = rec
    adapter.process(frame, 20)
    assert rec.ts == [0.02]
    assert rec.grays == ["crop"]
    assert rec.anchors == ["anchors"]
    assert rec.mouth_open == [0.25]

    adapter.tracker.obs = None
    adapter.process(frame, 30)
    assert adapter.last_tracking is None
    assert rec.anchors[-1] is None


def test_lip_worker_processes_idle_and_start_timestamp_once(monkeypatch):
    instances = []

    class Adapter:
        def __init__(self):
            self.recording = None
            self.last_tracking = None
            self.calls = []
            self.starts = 0
            instances.append(self)

        def start(self):
            self.starts += 1
            self.recording = object()

        def process(self, frame, timestamp_ms):
            self.calls.append(timestamp_ms)
            self.last_tracking = {"mouth_box": [0, 0, 1, 1],
                                  "outer_lips": [], "inner_lips": []}

        def finish(self):
            return LipResult([], 0)

        def close(self):
            pass

    monkeypatch.setattr("hackathon.lips_adapter.LipsAdapter", Adapter)
    runtime = Runtime(vocabulary={})
    runtime.state["camera"] = "ready"
    thread = threading.Thread(target=runtime._lips, daemon=True)
    runtime.threads.append(thread)
    thread.start()
    try:
        wait_for(lambda: runtime.state["lips"] == "ready")
        adapter = instances[0]
        idle_ts = clock_ms() - 1000
        runtime.lip_frames.put((np.zeros((2, 2, 3), dtype=np.uint8), idle_ts))
        wait_for(lambda: adapter.calls == [idle_ts])
        assert adapter.recording is None

        runtime.action("start", {})
        start_ts = runtime.record_start
        packet = (np.zeros((2, 2, 3), dtype=np.uint8), start_ts)
        runtime.lip_frames.put(packet)
        wait_for(lambda: adapter.calls == [idle_ts, start_ts])
        assert adapter.starts == 1
        assert runtime.tracking_snapshot()["mouth_timestamp_ms"] == start_ts

        runtime.lip_frames.put(packet)
        time.sleep(0.1)
        assert adapter.calls == [idle_ts, start_ts]
    finally:
        runtime.close()
        thread.join(1)


def test_hand_worker_failure_clears_tracking(monkeypatch):
    class Recognizer:
        def __init__(self):
            self.last_tracking = []
            self.calls = 0

        def process(self, frame, timestamp_ms):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("fake failure")
            self.last_tracking = [{"handedness": "Left", "landmarks": [],
                                   "label": None, "score": 0.0}]
            return []

        def close(self):
            pass

    monkeypatch.setattr("hackathon.runtime.importlib.import_module",
                        lambda name: SimpleNamespace(SignRecognizer=Recognizer))
    runtime = Runtime(vocabulary={})
    thread = threading.Thread(target=runtime._hands, daemon=True)
    runtime.threads.append(thread)
    thread.start()
    try:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        runtime.hand_frames.put((frame, 100))
        wait_for(lambda: bool(runtime.tracking_snapshot()["hands"]))
        runtime.hand_frames.put((frame, 200))
        wait_for(lambda: runtime.state["signs"] == "unavailable")
        assert runtime.tracking_snapshot() == {"hands": [], "mouth": None}
    finally:
        runtime.close()
        thread.join(1)


def test_tracking_endpoint_deepcopy_and_record_redirect():
    runtime = Runtime(rehearsal=True, vocabulary={})
    runtime.tracking = {"hands": [{"handedness": "Right", "landmarks": [],
                                   "label": "THUMB_UP", "score": 0.9}],
                        "mouth": None, "hands_timestamp_ms": 123}
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("GET", "/api/tracking")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == runtime.tracking
        connection.close()

        copied = runtime.tracking_snapshot()
        copied["hands"].clear()
        assert len(runtime.tracking["hands"]) == 1

        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("GET", "/record")
        response = connection.getresponse()
        assert response.status == 302
        from urllib.parse import urlsplit, parse_qs
        location = urlsplit(response.getheader("Location"))
        assert location.netloc == "127.0.0.1:8766"
        assert parse_qs(location.query)["return_to"] == [f"http://127.0.0.1:{server.server_port}/"]
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)
        runtime.close()


def test_lip_finish_preserves_scores_and_real_capture_measurements(monkeypatch):
    adapter = LipsAdapter.__new__(LipsAdapter)
    adapter.recording = SimpleNamespace(ts=[1.0, 1.1, 1.3], grays=[1, 2, 3],
                                       anchors=[1, 2, 3], mouth_open=[0.1, 0.2, 0.1])

    class Reader:
        def encode(self, rois):
            return "encoded"

        def beam_search_details(self, encoded, nbest):
            assert encoded == "encoded" and nbest == 3
            return [{"text": "FIRST", "score": -1.25}, {"text": "SECOND", "score": -2.0}]

    adapter.reader = Reader()
    monkeypatch.setattr("lipflow.dictation.clip_problem", lambda rec: None)
    monkeypatch.setattr("lipflow.dictation.rois_for", lambda rec: np.zeros((12, 96, 96), dtype=np.uint8))
    result = adapter.finish()
    assert result.candidates == ["FIRST", "SECOND"]
    assert result.candidate_scores == [-1.25, -2.0]
    assert (result.start_ms, result.end_ms, result.raw_frames) == (1000, 1300, 3)
    assert result.capture_fps == 6.67
    assert result.max_frame_gap_ms == 200
