""""What to record next" guidance: pure rules, inventory, and lifecycle wiring."""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest

from quietbridge import model_lifecycle
from quietbridge.profiles import Profile
from quietbridge.signs import capture_quality as cq
from quietbridge.signs.guidance import THRESHOLDS, guidance, inventory
from quietbridge.signs.tests.test_capture_quality import gesture_try

GESTURES = [{"label": "HELLO", "display_name": "Hello"}, {"label": "WATER", "display_name": "Water"},
            {"label": "YES", "display_name": "Yes"}, {"label": "REST", "display_name": "Resting"}]
PROMPTS = ["Hello, how are you?", "Please write it down.", "I need some help.", "Can you repeat that?"]


def row(tries, sessions=1, flagged=0):
    return {"tries": tries, "sessions": sessions, "flagged": flagged}


def full_inventory():
    return {
        "gestures": {
            "train": {"HELLO": row(6, 2), "WATER": row(6, 2), "YES": row(6, 2), "REST": row(6, 2)},
            "validation": {"HELLO": row(5), "WATER": row(5), "YES": row(5), "REST": row(6)},
            "test": {"HELLO": row(3), "WATER": row(3), "YES": row(3), "REST": row(3)},
        },
        "phrases": {
            "train": {p: row(3) for p in PROMPTS},
            "validation": {p: row(2) for p in PROMPTS},
            "test": {p: row(2) for p in PROMPTS},
        },
    }


def motion_report():
    per_label = {"HELLO": {"trials": 5, "correct": 2, "miss": 3, "wrong_label": 0},
                 "WATER": {"trials": 5, "correct": 3, "miss": 0, "wrong_label": 2},
                 "YES": {"trials": 5, "correct": 5, "miss": 0, "wrong_label": 0},
                 "REST": {"trials": 6, "correct": 5, "miss": 0, "wrong_label": 1}}
    return {
        "train_labels": {"HELLO": 6, "WATER": 6, "YES": 6, "REST": 6},
        "validation_crossfit": {"per_label": per_label,
                                "confusion": {"WATER -> YES": 2, "REST -> HELLO": 1, "HELLO -> <none>": 3}},
        "validation_ungated": {"per_label": per_label, "confusion": {}},
        # Final-check numbers deliberately different: guidance must never read them.
        "test_ungated": {"per_label": {"YES": {"trials": 3, "correct": 0, "miss": 3, "wrong_label": 0}},
                         "confusion": {"YES -> HELLO": 3}},
        "test_diagnostics": [{"trial_id": "x", "true": "YES"}],
        "test_score_gated_0.75": {"per_label": {}},
    }


def lips_report():
    rows = [{"text": "Please write it down.", "hypothesis": "please right"},
            {"text": "Please write it down.", "hypothesis": "please write it"},
            {"text": "Hello, how are you?", "hypothesis": "hello how are you"}]
    return {"selection": {"selected": "stock"}, "baseline": {"val": {"transcripts": rows},
                                                             "test": {"transcripts": [{"text": "I need some help.",
                                                                                       "hypothesis": "x"}]}}}


def by_id(result):
    return {item["id"]: item for item in result["items"]}


def test_weakest_gestures_confusions_and_rest_come_with_denominators():
    result = guidance(full_inventory(), GESTURES, PROMPTS, {"motion": ("abcdef123", motion_report())})
    items = by_id(result)
    weak = items["gesture-weak-HELLO"]
    assert weak["finding"] == "Hello was recognised in 2 of 5 check tries (candidate abcdef)."
    assert weak["next_step"].startswith("Record 5 more practice tries of Hello on a new occasion")
    assert weak["numbers"] == {"found": 2, "tries": 5, "candidate": "candidate abcdef", "split": "validation",
                               "basis": "crossfit"}
    confused = items["gesture-confused-WATER-YES"]
    assert "Water was taken for Yes 2 times in 5 check tries of Water" in confused["finding"]
    assert "look alike" in confused["next_step"] and "changing one" in confused["next_step"]
    rest = items["gesture-rest-fired"]
    assert rest["finding"] == ("Hands at rest triggered a gesture 1 time in 6 resting check tries, "
                               "taken for Hello (candidate abcdef).")
    assert "resting moments" in rest["next_step"]
    assert "gesture-weak-YES" not in items


def test_final_check_results_never_change_guidance():
    report = motion_report()
    before = guidance(full_inventory(), GESTURES, PROMPTS, {"motion": ("c1", report), "lips": ("c2", lips_report())})
    mutated = copy.deepcopy(report)
    mutated["test_ungated"] = {"per_label": {"HELLO": {"trials": 9, "correct": 0}}, "confusion": {"HELLO -> YES": 9}}
    mutated["test_diagnostics"] = []
    lips = lips_report()
    lips["baseline"]["test"]["transcripts"] = [{"text": "Hello, how are you?", "hypothesis": "nothing at all"}] * 5
    after = guidance(full_inventory(), GESTURES, PROMPTS, {"motion": ("c1", mutated), "lips": ("c2", lips)})
    assert after == before
    # Without validation results, test results alone produce no result items.
    only_test = {k: v for k, v in report.items() if not k.startswith("validation")}
    items = by_id(guidance(full_inventory(), GESTURES, PROMPTS, {"motion": ("c1", only_test)}))
    assert not any(i.startswith(("gesture-weak", "gesture-confused", "gesture-rest")) for i in items)


def test_in_sample_validation_is_labelled_optimistic_when_crossfit_is_missing():
    report = motion_report()
    del report["validation_crossfit"]
    item = by_id(guidance(full_inventory(), GESTURES, PROMPTS, {"motion": ("c1", report)}))["gesture-weak-HELLO"]
    assert "optimistic" in item["finding"] and item["numbers"]["basis"] == "in_sample"


def test_data_sufficiency_items_and_ordering():
    inv = full_inventory()
    inv["gestures"]["train"]["HELLO"] = row(3, 1)
    inv["gestures"]["train"]["WATER"] = row(6, 1)
    del inv["gestures"]["train"]["YES"]
    inv["gestures"]["validation"]["REST"] = row(2)
    del inv["gestures"]["test"]["WATER"]
    result = guidance(inv, GESTURES, PROMPTS, {})
    items = by_id(result)
    assert items["gesture-no-practice-YES"]["severity"] == "action"
    assert items["gesture-no-practice-YES"]["next_step"] == "Record 5 practice tries of Yes."
    few = items["gesture-few-practice-HELLO"]
    assert few["finding"] == "Hello has 3 practice tries (5 or more recommended)."
    assert few["next_step"] == "Record 2 more practice tries of Hello."
    assert items["gesture-one-occasion-WATER"]["severity"] == "info"
    check = items["gesture-few-check-REST"]
    assert "Resting has 2 check tries" in check["finding"] and "4 per gesture" in check["finding"]
    assert check["next_step"].startswith("Record 2 more check tries of resting moments")
    inv["gestures"]["train"]["REST"] = row(4, 1)
    assert by_id(guidance(inv, GESTURES, PROMPTS, {}))["gesture-few-practice-REST"]["next_step"] == \
        "Record 1 more practice try of resting moments."
    assert "gesture-no-final-WATER" in items
    severities = [i["severity"] for i in result["items"]]
    assert severities == sorted(severities, key={"action": 0, "warn": 1, "info": 2}.get)
    # Counts are never phrased as results.
    for item in result["items"]:
        if item["id"].startswith(("gesture-few", "gesture-no-practice", "gesture-one-occasion")):
            assert "recognised" not in item["finding"] and "accura" not in item["finding"]


def test_lip_guidance_never_recorded_hard_phrases_and_language_overlap():
    inv = full_inventory()
    del inv["phrases"]["train"]["I need some help."]
    del inv["phrases"]["train"]["Can you repeat that?"]
    report = {"text_overlap": {"validation_phrases": 4, "also_in_training": 4}}
    items = by_id(guidance(inv, GESTURES, PROMPTS, {"lips": ("lipcand", lips_report()),
                                                    "language": ("lmcand", report)}))
    never = items["lips-never-recorded"]
    assert never["finding"].startswith("2 of 4 phrases have no lip practice tries: “I need some help.”")
    hard = items["lips-hard-phrase-Please write it down."]
    assert hard["finding"].startswith("“Please write it down.”: 4 of 8 words wrong across 2 check tries")
    assert "standard model" in hard["finding"]
    assert "lips-hard-phrase-Hello, how are you?" not in items
    overlap = items["language-check-repeats-practice"]
    assert overlap["numbers"] == {"repeated": 4, "phrases": 4, "candidate": "candidate lmcand"}


def test_too_little_lip_data_is_an_action_with_the_trainer_limits():
    inv = full_inventory()
    inv["phrases"]["train"] = {PROMPTS[0]: row(4)}
    inv["phrases"]["validation"] = {PROMPTS[0]: row(1)}
    items = by_id(guidance(inv, GESTURES, PROMPTS, {}))
    assert items["lips-too-few-practice"]["severity"] == "action"
    assert items["lips-too-few-practice"]["numbers"]["required"] == THRESHOLDS["lip_practice_clips"]
    assert items["lips-too-few-check"]["next_step"].startswith("Record 4 more lip check tries")


def test_nothing_recorded_yet_does_not_nag():
    empty = {"gestures": {"train": {}, "validation": {}, "test": {}},
             "phrases": {"train": {}, "validation": {}, "test": {}}}
    result = guidance(empty, GESTURES, PROMPTS, {})
    assert [i["id"] for i in result["items"]] == ["gestures-not-started", "lips-not-started"]
    assert all(i["severity"] == "info" for i in result["items"])


def test_flagged_tries_are_reported_not_hidden():
    inv = full_inventory()
    inv["gestures"]["train"]["HELLO"] = row(6, 2, flagged=2)
    item = by_id(guidance(inv, GESTURES, PROMPTS, {}))["gesture-flagged-train-HELLO"]
    assert item["finding"] == "2 of 6 practice tries of Hello were flagged for capture quality when saved."
    inv["gestures"]["train"]["HELLO"] = row(6, 2, flagged=1)
    item = by_id(guidance(inv, GESTURES, PROMPTS, {}))["gesture-flagged-train-HELLO"]
    assert item["finding"] == "1 of 6 practice tries of Hello was flagged for capture quality when saved."
    assert "still used" in item["next_step"]


def _save_gesture(folder, name, label, session, quality=None, sign=False, move=0.3):
    lm, hp, ts = gesture_try(move=move)
    fields = dict(label=np.array(label), landmarks=lm, hand_present=hp, timestamps_ms=ts,
                  handedness=np.full((len(ts), 2), "Right"), session_id=np.array(session),
                  trial_id=np.array(name))
    if quality is not None:
        fields["quality_json"] = cq.to_npz_field(quality)
    if sign:
        fields["sign_language"] = np.array("bsl")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{name}.npz", **fields)


def test_inventory_counts_tries_occasions_and_flags_without_reading_results(tmp_path):
    seq, mouths = tmp_path / "seq", tmp_path / "mouths"
    _save_gesture(seq / "train", "a", "HELLO", "s1")
    _save_gesture(seq / "train", "b", "HELLO", "s2", move=0.0)  # older file, measured now: no movement
    stored = cq.with_decision(cq.gesture_quality(*gesture_try(n=4), "HELLO"), "saved_anyway")
    _save_gesture(seq / "train", "c", "HELLO", "s2", quality=stored)
    _save_gesture(seq / "validation", "d", "HELLO", "v1", sign=True)  # standard sign: not a personal gesture
    (mouths / "train").mkdir(parents=True)
    np.savez(mouths / "train" / "m.npz", text=np.array(" Hello there "), session_id=np.array("s1"),
             timestamps_ms=np.arange(30) * 40.0, face_present=np.zeros(30, dtype=bool),
             rois=np.zeros((2, 96, 96), dtype=np.uint8))
    (seq / "test").mkdir()
    (seq / "test" / "broken.npz").write_bytes(b"not a zip")
    inv = inventory(seq, mouths)
    assert inv["gestures"]["train"]["HELLO"] == {"tries": 3, "sessions": 2, "flagged": 2}
    assert inv["gestures"]["validation"] == {} and inv["sign_video_tries"] == 1
    assert inv["phrases"]["train"]["Hello there"] == {"tries": 1, "sessions": 1, "flagged": 1}
    assert inv["unreadable_files"] == 1


# -- lifecycle wiring --------------------------------------------------------------------------

def _candidate(profile, cid, kind, report, model=b"weights", status="completed", mtime=None):
    directory = profile.profile_dir / "models" / "candidates" / cid
    directory.mkdir(parents=True)
    name = {"motion": "motion.npz", "lips": "lip.pth", "language": "lm.pth"}[kind]
    (directory / name).write_bytes(model)
    report_path = directory / (name.rsplit(".", 1)[0] + ".report.json")
    report_path.write_text(json.dumps(report))
    if mtime is not None:
        os.utime(report_path, (mtime, mtime))
    (directory / "candidate.json").write_text(json.dumps({
        "id": cid, "kind": kind, "status": status, "error": None,
        "model_path": str(directory / name), "report_path": str(report_path)}))
    return directory / name


def test_lifecycle_guidance_uses_the_newest_completed_report(tmp_path):
    profile = Profile(id="alice", name="Alice", root_dir=tmp_path)
    lifecycle = model_lifecycle.ModelLifecycle()
    old = motion_report()
    new = motion_report()
    new["validation_crossfit"]["per_label"]["HELLO"] = {"trials": 5, "correct": 5, "miss": 0, "wrong_label": 0}
    _candidate(profile, "a" * 32, "motion", old, mtime=1_000)
    _candidate(profile, "b" * 32, "motion", new, mtime=2_000)
    _candidate(profile, "c" * 32, "motion", old, status="failed", mtime=3_000)
    result = lifecycle.guidance(profile)
    assert result["sources"]["motion"] == "candidate bbbbbb"
    assert "gesture-weak-HELLO" not in by_id(result)


def test_status_reports_restore_points_itself(tmp_path):
    profile = Profile(id="alice", name="Alice", root_dir=tmp_path)
    lifecycle = model_lifecycle.ModelLifecycle()
    status = lifecycle.status(profile)
    assert status["restore"]["motion"] == {"available": False, "restores": None}
    entry = profile.profile_dir / "models" / "history" / "lips" / "1-x"
    entry.mkdir(parents=True)
    (entry / "lip.pth").write_bytes(b"old")
    assert lifecycle.status(profile)["restore"]["lips"] == {"available": True, "restores": "previous_model"}


def test_report_hash_binds_a_candidate_to_its_exact_weights(tmp_path):
    import hashlib
    profile = Profile(id="alice", name="Alice", root_dir=tmp_path)
    lifecycle = model_lifecycle.ModelLifecycle()
    base = {"validation_trials": 2, "validation_ungated": {"trials": 2}}
    good = _candidate(profile, "a" * 32, "motion", {**base, "model_sha256": hashlib.sha256(b"weights").hexdigest()})
    _candidate(profile, "b" * 32, "motion", {**base, "model_sha256": "0" * 64})
    _candidate(profile, "c" * 32, "motion", base)
    described = {c["id"][0]: c for c in lifecycle.status(profile)["candidates"]}
    assert described["a"]["report_binding"] == "bound" and described["a"]["promotable"]
    assert described["b"]["report_binding"] == "mismatch" and not described["b"]["promotable"]
    assert "changed after its report" in described["b"]["report_warning"]
    assert described["c"]["report_binding"] == "legacy_unbound" and described["c"]["promotable"]
    good.write_bytes(b"tampered")
    assert not lifecycle.status(profile)["candidates"][0]["promotable"]


def test_experiment_only_lip_reports_are_never_promotable(tmp_path):
    import hashlib
    profile = Profile(id="alice", name="Alice", root_dir=tmp_path)
    report = {"counts": {"validation": 5}, "selection": {"selected": "candidate", "experiment_only": True},
              "candidate_sha256": hashlib.sha256(b"weights").hexdigest()}
    _candidate(profile, "a" * 32, "lips", report)
    (candidate,) = model_lifecycle.ModelLifecycle().status(profile)["candidates"]
    assert candidate["report_binding"] == "bound" and not candidate["promotable"]
    report["selection"].pop("experiment_only")
    (profile.profile_dir / "models" / "candidates" / ("a" * 32) / "lip.report.json").write_text(json.dumps(report))
    assert model_lifecycle.ModelLifecycle().status(profile)["candidates"][0]["promotable"]
