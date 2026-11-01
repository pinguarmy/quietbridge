"""MouthCapture tests with a fake tracker — no camera, no real model, no network."""
import numpy as np
import pytest

from hackathon.signs.mouth_capture import MouthCapture


class FakeObs:
    """Outer-lip points spanning x 40..80, y 30..60 on any frame."""
    def __init__(self):
        xs = np.linspace(40, 80, 20)
        ys = np.linspace(30, 60, 20)
        self.pts = np.stack([xs, ys], axis=1).astype(np.float32)

    @property
    def anchors(self):
        return np.vstack([self.pts.mean(0)] * 4)

    @property
    def mouth_open(self):
        return 0.2

    @property
    def outer_lips(self):
        return self.pts

    @property
    def inner_lips(self):
        return self.pts * np.array([0.8, 0.8])


class FakeTracker:
    def __init__(self):
        self.see_face = True
        self.calls = 0
        self.closed = False

    def detect(self, frame, ts_ms):
        self.calls += 1
        return FakeObs() if self.see_face else None

    def close(self):
        self.closed = True


def frame():
    return np.zeros((100, 200, 3), dtype=np.uint8)  # 200 wide, 100 tall


@pytest.fixture
def cap(tmp_path, monkeypatch):
    tracker = FakeTracker()
    c = MouthCapture(data_root=str(tmp_path), tracker_factory=lambda: tracker)
    monkeypatch.setattr("lipflow.dictation.clip_problem", lambda rec: None)
    monkeypatch.setattr("lipflow.dictation.rois_for",
                        lambda rec: np.zeros((20, 96, 96), dtype=np.uint8))
    c.fake_tracker = tracker
    return c


def test_save_contents(cap, tmp_path):
    cap.start("hello world", "train", "s1")
    for ts in (1000, 1033, 1066):
        cap.feed(frame(), ts)
    snap = cap.stop()
    assert snap["saved_count"] == 1 and snap["recording"] is False
    assert snap["mouth_box"] is None and snap["quality"] == 1.0
    path = snap["last_saved"]
    assert path.startswith(str(tmp_path / "train")) and path.split("/")[-1].startswith("s1_")
    with np.load(path, allow_pickle=False) as z:
        assert z["text"].shape == () and str(z["text"]) == "hello world"
        assert z["rois"].shape == (20, 96, 96) and z["rois"].dtype == np.uint8
        assert z["timestamps_ms"].dtype == np.int64
        assert list(z["timestamps_ms"]) == [1000, 1033, 1066]
        assert str(z["session_id"]) == "s1" and str(z["split"]) == "train"
        assert len(str(z["trial_id"])) == 32


def test_session_cross_split_rejected(cap):
    cap.start("a", "train", "s1")
    cap.feed(frame(), 0)
    cap.stop()
    with pytest.raises(ValueError, match="another split"):
        cap.start("b", "test", "s1")
    cap.start("c", "train", "s1")  # same split is fine
    cap.discard()


def test_two_trials_never_overwrite(cap, tmp_path):
    for text in ("one", "two"):
        cap.start(text, "train", "s1")
        cap.feed(frame(), 0)
        cap.stop()
    assert len(list((tmp_path / "train").glob("s1_*.npz"))) == 2


def test_no_frames_cannot_save_and_discard(tmp_path, monkeypatch):
    # Real clip_problem: an empty recording is genuinely too short to save.
    monkeypatch.setattr("lipflow.dictation.rois_for",
                        lambda rec: np.zeros((20, 96, 96), dtype=np.uint8))
    cap = MouthCapture(data_root=str(tmp_path), tracker_factory=FakeTracker)
    cap.start("nothing said", "train", "s2")
    with pytest.raises(ValueError, match="Too short"):
        cap.stop()
    assert cap.snapshot()["error"] and cap._rec is not None  # kept for discard
    assert cap.snapshot()["recording"] is True  # can continue/retry instead of a dead UI state
    snap = cap.discard()
    assert snap["frames"] == 0 and snap["error"] is None
    cap.start("again", "train", "s2")  # can start fresh after discard


def test_mouth_box_normalized(cap):
    cap.start("box", "train", "s3")
    box = cap.feed(frame(), 0)
    assert box == [0.2, 0.3, 0.2, 0.3]  # x0/200, y0/100, 40/200, 30/100
    assert cap.snapshot()["mouth_box"] == box
    assert len(cap.snapshot()["outer_lips"]) == 20
    assert len(cap.snapshot()["inner_lips"]) == 20
    np.testing.assert_allclose(cap.snapshot()["outer_lips"][0], [.2, .3])


def test_inactive_feed_does_not_detect(cap):
    assert cap.feed(frame(), 0) is None  # no detection yet: never pretend a live box
    assert cap.fake_tracker.calls == 0
    cap.start("x", "train", "s4")
    cap.feed(frame(), 0)
    cap.discard()
    assert cap.feed(frame(), 50) is None  # discarded clips must not leave a stale live box


def test_frame_quality_and_face_flag(cap):
    cap.start("q", "train", "s5")
    for ts in (0, 40, 80):
        cap.feed(frame(), ts)
    cap.fake_tracker.see_face = False
    cap.feed(frame(), 120)
    cap.feed(frame(), 160)
    snap = cap.snapshot()
    assert snap["frames"] == 5
    assert snap["mouth_box"] is None  # lost detection clears the guide
    assert snap["face"] is False          # latest frame had no face
    assert snap["quality"] == 0.6         # 3 of 5 frames had a face
    assert cap._rec.mouth_open[-1] == 0.0


def test_limit_stops_appending(cap):
    cap.start("long", "train", "s6")
    cap.feed(frame(), 0)
    cap.feed(frame(), 1000)
    box = cap.feed(frame(), 16_100)  # beyond 15 s: not appended
    assert cap.snapshot()["frames"] == 2
    assert cap.snapshot()["limit_reached"] is True
    assert box == [0.2, 0.3, 0.2, 0.3]
    cap.stop()  # saving what we have still works
    assert cap.saved_count == 1


def test_explicit_preview_tracks_without_recording_or_saving(cap, tmp_path):
    cap.preview()
    cap.feed(frame(), 1000)
    state = cap.snapshot()
    assert state["face"] is True
    assert state["mouth_box"] == [0.2, 0.3, 0.2, 0.3]
    assert state["recording"] is False and state["frames"] == 0
    assert list(tmp_path.rglob("*.npz")) == []
    cap.fake_tracker.see_face = False
    cap.feed(frame(), 1100)
    assert cap.snapshot()["mouth_box"] is None
    assert cap.snapshot()["face"] is False


def test_start_validation(cap):
    with pytest.raises(ValueError):
        cap.start("", "train", "s")
    with pytest.raises(ValueError):
        cap.start("x" * 301, "train", "s")
    with pytest.raises(ValueError):
        cap.start("hi", "dev", "s")
    with pytest.raises(ValueError):
        cap.start("hi", "train", "../evil")
    with pytest.raises(ValueError):
        cap.start("hi", "train", "has space")
    cap.start("ok", "train", "s7")
    with pytest.raises(ValueError, match="still open"):
        cap.start("again", "train", "s8")
    cap.discard()


def test_tracker_built_lazily_and_closed(tmp_path, monkeypatch):
    made = []
    monkeypatch.setattr("lipflow.dictation.clip_problem", lambda rec: None)
    monkeypatch.setattr("lipflow.dictation.rois_for",
                        lambda rec: np.zeros((12, 96, 96), dtype=np.uint8))
    c = MouthCapture(data_root=str(tmp_path),
                     tracker_factory=lambda: made.append(FakeTracker()) or made[0])
    assert made == []  # no tracker before the first start
    c.start("lazy", "train", "s9")
    assert len(made) == 1
    c.feed(frame(), 0)
    c.stop()
    c.start("lazy2", "train", "s9")
    assert len(made) == 1  # reused, not rebuilt
    c.close()
    assert made[0].closed and c._tracker is None and c._rec is None
