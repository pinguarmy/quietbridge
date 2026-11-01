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

Select **Lips**. Hold the record button or Space, mouth a clear 2–4 second English
sentence and release. Confirm the displayed candidate, then press Speak. Record
the actual transcript, intended transcript and measured release-to-text time.
Do not use rehearsal numbers as vision performance measurements.

If lips fail, copy the visible error text. The sign-only path remains available
once Peter's recognizer is integrated. `--sign-only` starts without loading VSR.

## Connect Peter's signs

Peter provides `hackathon/signs/recognizer.py` with `SignRecognizer.process` and
`close` as defined in the integration guide. Sam updates the configurable
`hackathon/vocabulary.json` to Peter's exact labels and agreed fixed phrases.
Placeholder entries remain `verified: false` and are ignored by live fusion.
Restart after integrating the recognizer or changing the mapping.

## Demo gate

Run each chosen gesture five times and check neutral hands for false triggers.
Try five new lip sentences, combined agreement, combined conflict, and one
missing modality. Turn Wi-Fi off and verify display and explicit Speak still
work. Require three consecutive successful live runs. Keep a consented offline
recording as fallback; do not commit it.

The integration has passed 20 tests using synthetic events and mocked model
workers. Real recognition, audible speech onset and repeatability on the Pro
remain live acceptance checks.
