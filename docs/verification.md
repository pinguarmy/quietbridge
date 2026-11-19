# Verification

The checks in this repository, and what each does and does not prove. Run focused checks while developing and the full set before handing off.

## Commands

```sh
uv run lipflow doctor
uv run pytest
node --check quietbridge/ui/app.js
node --test quietbridge/tests/test_ui_flow.cjs quietbridge/tests/test_cleanup_settings_ui.cjs "quietbridge/signs/tests/test_lab_*.cjs"
# Needs Node 22 or newer (CI uses 24) and an installed Chrome/Chromium; no new npm package.
# As root (some containers), point CHROME_BIN at a wrapper that adds --no-sandbox.
node scripts/check_web.mjs
node scripts/check_web.mjs --calibrate
node scripts/check_web.mjs --keyboard-profile
```

`lipflow doctor` checks the legacy Lipflow setup on macOS and Windows. On Linux it currently stops with `ModuleNotFoundError: No module named 'Quartz'`, because the `lipflow` CLI imports its macOS hotkey module outside Windows; the QuietBridge server does not depend on it.

The `test_lab_*.cjs` glob is quoted so Node expands it itself; this works the same in bash and PowerShell.

## What they check

**`uv run pytest`** runs the legacy Lipflow suite in `tests/` and the QuietBridge suites in `quietbridge/`. Tests that need something the host lacks skip with a reason naming it, rather than failing or passing silently:

- macOS-only modules (`Quartz`, `AppKit`) for the native Lipflow app, hotkey, paste and onboarding checks, including the `lipflow run`/`file` CLI wiring, which imports the macOS hotkey module outside Windows;
- a keyboard backend for `pynput` (on Linux this needs an X display) for the push-to-talk tests;
- downloaded model files, such as `models/lm/unigram5000.model` (run `./setup.sh` or `setup.ps1`), and the sample clips (`./setup.sh --samples`);
- Windows for the Windows-only checks.

Read the skip list (`-rs`) on each platform; a skipped test is unknown evidence, not a pass.

**The browser checks** launch their own headless Chrome, temporary profile and loopback rehearsal server. They use real DOM, native dialogs, selected keyboard events and real HTTP, with a silent speech recorder instead of audio. No physical camera, model workers or provider calls start. `CHROME_BIN` and `PYTHON_BIN` can select installed executables; on Windows use `.venv/Scripts/python.exe`. Each run logs source/served-script hashes and verifies the browser-loaded script. Calibration reruns the same assertions against known behavioral faults and must reject each at its designated check. This tests the suite's ability to catch those faults, not blind-reviewer skill or human usability. `--keyboard-profile` checks typed apply, explicit silent dispatch and draft navigation with CDP keyboard/text input and no DOM focus/click, then separately checks profile draft isolation with DOM-dispatched controls. It is not a full keyboard/capture, screen-reader or real-zoom test. See [the review protocol](review-protocol.md).

## What they do not prove

Synthetic checks, fake backends and rehearsal establish integration behavior, not live recognition accuracy. Real use still requires new-session camera trials, paired lips/gesture evaluation, REST/unknown false-trigger measurements and audible speech checks with real cameras, real models and real speakers, tested separately. Do not tune on the final-test session. Review reports distinguish software contracts from real-user and recognition effectiveness; follow [the review protocol](review-protocol.md).

## Continuous integration

| Workflow | Runs | Does not run |
| --- | --- | --- |
| [`windows.yml`](../.github/workflows/windows.yml) | `setup.ps1 -Samples` (dependencies, models, sample clips, Start-menu shortcut); `uv run pytest`; `node --check`; the Node UI contract tests; the three Chrome browser checks; `lipflow doctor` (print only, no camera); a real sample-clip lip read on the CPU | Camera, audible speech, real users |
| [`linux.yml`](../.github/workflows/linux.yml) | `node --check` and the Node UI contract tests (before the Python install, so they fail early); `uv sync --frozen`; `uv run quietbridge --help`; the three Chrome browser checks; `uv run pytest` without models | Models, sample clips, macOS/Windows-only tests, camera, audible speech |

The Linux workflow, and the `node --check`/Node UI steps in the Windows workflow, were added on 2026-11-18. They were validated locally (YAML parse, actionlint, and the Linux commands on a headless Linux host) but have not yet run on GitHub; treat them as unproven until a GitHub run passes.

Both workflows have read-only repository permissions and cancel superseded runs on the same ref. The Linux job's Python dependencies include the CUDA build of PyTorch that the lockfile resolves for Linux (about 3 GB); it is downloaded on each run rather than cached.
