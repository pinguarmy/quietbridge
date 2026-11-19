"""`lipflow doctor` reports what it checked, and never a pass for something it didn't check."""
import io
import sys
import types

from lipflow import doctor


def _report():
    return doctor.Report(out=io.StringIO())


def _models(tmp_path, monkeypatch, sizes):
    root = tmp_path / "models"
    for rel, size in sizes.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.truncate(int(size))  # sparse: no real disk use
    monkeypatch.setattr(doctor, "MODELS", str(root))


def test_models_present_and_full_size_pass(tmp_path, monkeypatch):
    _models(tmp_path, monkeypatch, {"vsr/model.json": 10, "vsr/model.pth": 1e9, "lm/model.json": 10,
                                    "lm/model.pth": 3e8, "face_landmarker.task": 4e6,
                                    "lm/unigram5000.model": 3e5})
    rep = _report()
    doctor.check_models(rep)
    assert rep.counts["PASS"] == 6 and not rep.failures


def test_truncated_and_missing_models_fail_and_optional_tokenizer_warns(tmp_path, monkeypatch):
    _models(tmp_path, monkeypatch, {"vsr/model.json": 10, "vsr/model.pth": 5e6, "lm/model.json": 10,
                                    "lm/model.pth": 3e8})
    rep = _report()
    doctor.check_models(rep)
    out = rep.out.getvalue()
    assert "FAIL  models/vsr/model.pth: only 5.0 MB" in out
    assert "FAIL  models/face_landmarker.task: missing" in out
    assert "WARN  models/lm/unigram5000.model (optional" in out
    assert len(rep.failures) == 2 and rep.counts["WARN"] == 1


def test_a_crashing_check_is_a_failure_line_not_a_traceback():
    rep = _report()

    def boom(_rep):
        raise OSError("disk on fire")

    rep.run("models", boom)
    assert rep.failures and "crashed (OSError: disk on fire)" in rep.out.getvalue()


def test_camera_skip_is_not_a_pass():
    rep = _report()
    doctor.check_camera_device(rep, enabled=False)
    assert rep.counts == {**rep.counts, "SKIP": 1, "PASS": 0}


def test_linux_without_a_video_device_warns(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(doctor.glob, "glob", lambda pattern: [])
    rep = _report()
    doctor.check_camera_device(rep, enabled=True)
    assert rep.counts["WARN"] == 1 and rep.counts["PASS"] == 0 and not rep.failures


def test_dictation_app_is_not_applicable_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    rep = _report()
    doctor.check_dictation_app(rep)
    assert rep.counts["N/A"] == 1 and rep.counts["PASS"] == 0


def _fake_quietbridge(monkeypatch, tmp_path, *, model_bytes=None, backend=None):
    path = tmp_path / "gesture_recognizer.task"
    if model_bytes is not None:
        path.write_bytes(model_bytes)
    assets = types.ModuleType("hackathon.signs.assets")
    assets.MODEL_SHA256 = "0" * 64
    assets.default_model_path = lambda: path
    monkeypatch.setitem(sys.modules, "hackathon.signs.assets", assets)

    class Speaker:
        def __init__(self):
            self.backend = backend

        def _resolve(self):
            return backend, "/usr/bin/espeak-ng" if backend else None

        def _unavailable_message(self):
            return "install espeak-ng."

    speech = types.ModuleType("hackathon.speech")
    speech.Speaker = Speaker
    monkeypatch.setitem(sys.modules, "hackathon.speech", speech)


def test_quietbridge_missing_gesture_model_and_speech_warn(monkeypatch, tmp_path):
    _fake_quietbridge(monkeypatch, tmp_path)
    rep = _report()
    doctor.check_quietbridge(rep)
    out = rep.out.getvalue()
    assert "WARN  stock gesture model: missing" in out and "hackathon.signs.assets --download" in out
    assert "WARN  local speech: no backend found" in out and "install espeak-ng." in out
    assert not rep.failures


def test_quietbridge_wrong_gesture_model_fails_and_found_speech_passes(monkeypatch, tmp_path):
    _fake_quietbridge(monkeypatch, tmp_path, model_bytes=b"not the pinned model", backend="espeak-ng")
    rep = _report()
    doctor.check_quietbridge(rep)
    out = rep.out.getvalue()
    assert "SHA-256 differs" in out and len(rep.failures) == 1
    assert "PASS  local speech backend: espeak-ng (/usr/bin/espeak-ng); found on PATH, no sound was played" in out


def test_exit_code_follows_required_failures(monkeypatch, capsys):
    for name in ("check_models", "check_runtime", "check_camera_device", "check_dictation_app",
                 "check_quietbridge"):
        monkeypatch.setattr(doctor, name, lambda rep, *a: rep.add("WARN", "optional"))
    assert doctor.doctor(check_camera=False) == 0
    monkeypatch.setattr(doctor, "check_models", lambda rep: rep.add("FAIL", "models/vsr/model.pth: missing"))
    assert doctor.doctor(check_camera=False) == 1
    assert "Required checks failed: models/vsr/model.pth: missing" in capsys.readouterr().out

