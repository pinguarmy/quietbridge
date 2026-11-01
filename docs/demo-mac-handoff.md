# QuietBridge demo Mac handoff

The final live demo runs on **Peter's MacBook Pro**. Sam's Mac runs integration
development and the synthetic rehearsal UI. Its model/dependency downloads have
been stopped. No real recognition accuracy or vision latency has been verified
by the integration tests or rehearsal.

## Get the integration without disturbing personal work

On Peter's MacBook Pro, clone into a new directory. Keep the original Lipflow
checkout, uncommitted personalizations and installed app intact:

```sh
git clone --branch main https://github.com/pinguarmy/quietbridge.git ~/code/quietbridge-demo
cd ~/code/quietbridge-demo
```

If that directory already exists, inspect its `git status` before updating it; do
not reset it. The shared contract is `hackathon/contracts.py`. Peter's recognizer
imports those types and stays within `hackathon/signs/`.

## Reuse models and Python already on this Mac

Use an existing compatible Python 3.11/3.12 environment **on the MacBook Pro** if
it already runs Lipflow. From the new checkout, invoke that environment's Python
by its absolute path. The model files must resolve under this checkout's ignored
`models/` directory; copy them or link that directory to the existing local model
directory. Required files:

```text
models/vsr/model.json
models/vsr/model.pth
models/lm/model.json
models/lm/model.pth
models/face_landmarker.task
```

If the files exist only on the MacBook Air, transfer these model assets locally
to the Pro to avoid the slow venue network. Do not transfer personal clips,
settings or credentials. Do not copy a virtual environment between Macs.
For a fresh Python environment, use the installation commands in
`hackathon/README.md`. The asset downloader can also fetch missing files; no
reinstallation of the native Lipflow application is required.

## First live check

From the demo checkout, using the compatible environment's Python:

```sh
python -m unittest discover -s hackathon/tests -v
python -m hackathon.assets
python -m hackathon.server --camera 0
```

Close the original Lipflow app and any other camera app. Open
<http://127.0.0.1:8765>. Allow Camera access to the launching terminal/Python if
macOS asks, then wait for the camera and lip reader to show **ready**. If the
wrong camera appears, stop the server and select another `--camera` index.

Choose **Lips only** in Settings. Hold the record button or Space, mouth a clear
2–4 second English sentence and release. Review the displayed candidate and
press Confirm & speak. Record
the actual transcript, intended transcript and measured release-to-text time.
Do not use rehearsal numbers as vision performance measurements.

If lips fail, copy the visible error text from Settings → Diagnostics. The sign-only
path is integrated; `--sign-only` starts without loading VSR.

## Connect Peter's signs

Peter's recognizer is integrated and the root vocabulary matches its four labels.
Download the gesture model once and run his checker separately from the server:

```sh
python -m hackathon.signs.assets --download
python -m hackathon.signs.demo --camera 0 --seconds 60
```

Use these live trials to agree the fixed phrases. Entries remain `verified: false`
until tested and are ignored by live fusion. Enable only validated mappings, close
the checker, and restart the integrated server. For combined input, start recording
before gesturing, since a held pose does not emit repeatedly.

## Demo gate

Run each chosen gesture five times and check neutral hands for false triggers.
Try five new lip sentences, combined agreement, combined conflict, and one
missing modality. Turn Wi-Fi off and verify display and explicit Speak still
work. Require three consecutive successful live runs. Keep a consented offline
recording as fallback; do not commit it.

Sam's focused suite has passed 36 tests plus 14 subtests: 24 integration tests and
12 gesture stabilization/asset tests. This does not load real vision models.
Real recognition, audible speech onset and repeatability on the Pro remain live
acceptance checks. Peter owns the separate "hello dev day" phrase work.
