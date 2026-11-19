# quietbridge/signs

Hand and sign input for QuietBridge, plus the Personalize recording lab.

| Module | Purpose |
| --- | --- |
| `recognizer.py`, `stabilizer.py` | Stock hand poses (MediaPipe GestureRecognizer) and the stable-event gate |
| `motion.py`, `motion_auto.py`, `train_motion.py` | Personal dynamic-gesture model, optional automatic segmentation, candidate training |
| `isolated.py`, `export_bsl1k.py` | Explicit isolated-sign video bundle (`--sign-model`) and the BSL-1K exporter |
| `lab.py`, `lab.html` | Personalize page: profiles, phrases, separate practice / validation / final-test sessions, model candidates |
| `mouth_capture.py`, `lip_lab.py`, `train_lips.py`, `train_language.py` | Labelled mouth-crop capture and personal lip / language-model candidates |
| `capture_analysis.py`, `assets.py`, `labels.json` | Capture timing report, explicit stock-model download, stock pose descriptions |

Personal gestures are always motion models recorded on the Personalize page. The older
single-frame personal pose model and its `QUIETBRIDGE_PERSONAL_GESTURES` setting were removed;
setting that variable now stops with an error instead of silently replacing the stock poses.

A stock pose or a named personal gesture is not a validated standard sign, and none of this is
continuous BSL/ASL translation. Tests in `tests/` use fake engines and synthetic data: they check
software contracts, not camera accuracy.

See [using QuietBridge](../../docs/using-quietbridge.md),
[training and evaluation](../../docs/training-and-evaluation.md) and
[standard signs](../../docs/standard-signs.md).
