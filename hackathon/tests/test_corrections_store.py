import json
from pathlib import Path

import numpy as np
import pytest

from hackathon.corrections import CorrectionStore, learning_texts, summarize_corrections
from hackathon.profiles import Profile


def test_correction_requires_explicit_learning_and_persists(tmp_path):
    profile = Profile("alice", "Alice", tmp_path)
    profile.ensure_dirs()
    source = profile.mouths_dir / "train" / "clip.npz"
    np.savez(source, text=np.array("wrong"), session_id=np.array("s1"), trial_id=np.array("t1"))
    store = CorrectionStore(profile)
    with pytest.raises(ValueError, match="Choose"):
        store.record(original="wrong", corrected="right")
    entry = store.record(original="wrong", corrected="right", candidates=["wrong", "write"],
                         source_paths=[source], learn=True)
    assert entry["sources"] == ["mouths/train/clip.npz"]
    assert entry["learn"] is True
    assert store.list() == [entry]


def test_corrections_reject_empty_same_or_external_sources(tmp_path):
    profile = Profile("alice", "Alice", tmp_path)
    store = CorrectionStore(profile)
    with pytest.raises(ValueError, match="does not change"):
        store.record(original="same", corrected="same", learn=True)
    with pytest.raises(ValueError, match="1–2000"):
        store.record(original="wrong", corrected="", learn=True)
    external = tmp_path / "external.npz"
    np.savez(external, text=np.array("outside"))
    with pytest.raises(ValueError, match="inside this profile"):
        store.record(original="wrong", corrected="right", source_paths=[external], learn=True)
    assert store.list() == []


def test_record_rejects_heldout_validation_and_test_sources(tmp_path):
    profile = Profile("alice", "Alice", tmp_path)
    profile.ensure_dirs()
    val_npz = profile.mouths_dir / "validation" / "clip.npz"
    np.savez(val_npz, text=np.array("val"))
    test_npz = profile.mouths_dir / "test" / "clip.npz"
    np.savez(test_npz, text=np.array("test"))

    store = CorrectionStore(profile)
    with pytest.raises(ValueError, match="held-out.*validation"):
        store.record(original="wrong", corrected="right", source_paths=[val_npz], learn=True)
    with pytest.raises(ValueError, match="held-out.*test"):
        store.record(original="wrong", corrected="right", source_paths=[test_npz], learn=True)
    assert store.list() == []


def test_record_rejects_npz_metadata_contradictions_and_non_scalar(tmp_path):
    profile = Profile("alice", "Alice", tmp_path)
    profile.ensure_dirs()
    store = CorrectionStore(profile)

    # NPZ placed under train/ but with internal split="validation"
    bad_train_npz = profile.mouths_dir / "train" / "bad_val.npz"
    np.savez(bad_train_npz, text=np.array("text"), split=np.array("validation"))
    with pytest.raises(ValueError, match="held-out"):
        store.record(original="wrong", corrected="right", source_paths=[bad_train_npz], learn=True)

    # NPZ placed under train/ but with internal split="other"
    mismatch_npz = profile.mouths_dir / "train" / "mismatch.npz"
    np.savez(mismatch_npz, text=np.array("text"), split=np.array("other"))
    with pytest.raises(ValueError, match="split contradiction"):
        store.record(original="wrong", corrected="right", source_paths=[mismatch_npz], learn=True)

    # Non-scalar split must fail closed
    non_scalar_npz = profile.mouths_dir / "train" / "non_scalar.npz"
    np.savez(non_scalar_npz, text=np.array("text"), split=np.array(["train", "val"]))
    with pytest.raises(ValueError, match="scalar"):
        store.record(original="wrong", corrected="right", source_paths=[non_scalar_npz], learn=True)

    # NPZ placed under train/ with matching split="train" succeeds
    good_npz = profile.mouths_dir / "train" / "good.npz"
    np.savez(good_npz, text=np.array("text"), split=np.array("train"))
    entry = store.record(original="wrong", corrected="right", source_paths=[good_npz], learn=True)
    assert entry["sources"] == ["mouths/train/good.npz"]


def test_record_accepts_daily_explicit_consent_without_sources(tmp_path):
    profile = Profile("alice", "Alice", tmp_path)
    store = CorrectionStore(profile)
    entry = store.record(original="misheard", corrected="clear speech", learn=True)
    assert entry["sources"] == []
    assert entry["learn"] is True
    assert entry["corrected"] == "clear speech"


def test_summarize_filters_legacy_heldout_and_flags_unknown(tmp_path):
    profile = Profile("alice", "Alice", tmp_path)
    profile.ensure_dirs()

    # Train clip with session/trial
    train_clip = profile.mouths_dir / "train" / "c1.npz"
    np.savez(train_clip, text=np.array("HELLO"), session_id=np.array("s_train_1"), trial_id=np.array("t1"), split=np.array("train"))

    # Validation clip
    val_clip = profile.mouths_dir / "validation" / "c2.npz"
    np.savez(val_clip, text=np.array("VAL"), session_id=np.array("s_val_1"), trial_id=np.array("t2"), split=np.array("validation"))

    # Manually populate corrections.jsonl (simulating legacy entries)
    jsonl_path = profile.profile_dir / "corrections.jsonl"
    entries = [
        # 1. Valid train source
        {"id": "e1", "profile_id": "alice", "original": "HELLO", "corrected": "HELLO THERE", "learn": True, "sources": ["mouths/train/c1.npz"]},
        # 2. Legacy held-out validation source (must be filtered out)
        {"id": "e2", "profile_id": "alice", "original": "VAL", "corrected": "VAL PROMPT", "learn": True, "sources": ["mouths/validation/c2.npz"]},
        # 3. Daily correction with explicit consent, no sources
        {"id": "e3", "profile_id": "alice", "original": "BYE", "corrected": "GOODBYE", "learn": True, "sources": []},
        # 4. Old record without 'sources' key
        {"id": "e4", "profile_id": "alice", "original": "THANKS", "corrected": "THANK YOU", "learn": True},
        # 5. Record from a different profile (must be ignored)
        {"id": "e5", "profile_id": "bob", "original": "NO", "corrected": "NOPE", "learn": True, "sources": []},
    ]
    with jsonl_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    summary = summarize_corrections(profile.profile_dir)
    assert summary["total_records"] == 4  # e1, e2, e3, e4 (e5 is from bob)
    assert summary["valid_count"] == 3    # e1 (train), e3 (unknown), e4 (unknown)
    assert summary["skipped_count"] == 1  # e2 (heldout)
    assert summary["source_coverage"]["verified_train"] == 1
    assert summary["source_coverage"]["unknown_provenance"] == 2
    assert summary["source_coverage"]["heldout_rejected"] == 1
    assert summary["source_coverage"]["invalid_sources"] == 0
    assert summary["sessions"] == ["s_train_1"]
    assert summary["trials"] == ["t1"]
    assert summary["provenance_unknown"] is True
    assert summary["correction_provenance_complete"] is False
    assert "HELLO THERE" in summary["texts"]
    assert "GOODBYE" in summary["texts"]
    assert "THANK YOU" in summary["texts"]
    assert "VAL PROMPT" not in summary["texts"]
    assert "NOPE" not in summary["texts"]

    # learning_texts API returns exactly the filtered texts
    texts = learning_texts(profile.profile_dir)
    assert texts == ["HELLO THERE", "GOODBYE", "THANK YOU"]


def test_summarize_pure_verified_train_yields_complete_provenance(tmp_path):
    profile = Profile("alice", "Alice", tmp_path)
    profile.ensure_dirs()
    c1 = profile.mouths_dir / "train" / "c1.npz"
    np.savez(c1, text=np.array("TRAIN"), session_id=np.array("s1"), trial_id=np.array("t1"), split=np.array("train"))

    store = CorrectionStore(profile)
    store.record(original="TRAIN", corrected="TRAIN PHRASE", source_paths=[c1], learn=True)

    summary = summarize_corrections(profile.profile_dir)
    assert summary["total_records"] == 1
    assert summary["valid_count"] == 1
    assert summary["skipped_count"] == 0
    assert summary["source_coverage"]["verified_train"] == 1
    assert summary["source_coverage"]["unknown_provenance"] == 0
    assert summary["provenance_unknown"] is False
    assert summary["correction_provenance_complete"] is True
    assert summary["sessions"] == ["s1"]
    assert summary["trials"] == ["t1"]
