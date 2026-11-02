import numpy as np
import pytest

from hackathon.signs.capture_analysis import analyze_capture, analyze_npz


def test_real_timing_gaps_and_face_edges_are_measured():
    result = analyze_capture(
        [0, 40, 80, 200, 240],
        mouth_open=[0.1, 0.1, 0.3, 0.1, 0.1],
        face_present=[False, True, True, True, False],
    )
    assert result["raw_frames"] == 5
    assert result["duration_ms"] == 240
    assert result["capture_fps"] == 16.667
    assert result["max_frame_gap_ms"] == 120
    assert result["face_ratio"] == 0.6
    assert result["leading_no_face_ms"] == 40
    assert result["trailing_no_face_ms"] == 40
    assert result["mouth_activity_range"] == pytest.approx(0.04)
    assert result["leading_low_activity_ms"] == 80
    assert result["trailing_low_activity_ms"] == 160


def test_old_npz_without_activity_fields_remains_diagnostic(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, timestamps_ms=np.array([0, 66, 132]), text=np.array("hello"))
    result = analyze_npz(path)
    assert result["capture_fps"] == pytest.approx(15.152)
    assert result["face_ratio"] is None
    assert result["leading_low_activity_ms"] is None


def test_misaligned_or_nonmonotonic_metadata_is_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze_capture([0, 40, 40])
    with pytest.raises(ValueError, match="align"):
        analyze_capture([0, 40], face_present=[True])
    with pytest.raises(ValueError, match="align"):
        analyze_capture([0, 40], mouth_open=[0.1])
