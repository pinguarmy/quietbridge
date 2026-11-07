"""Isolated rehearsal server for check_web.mjs; never starts hardware or speech.

One JSON object per stdin/stdout line. The parent owns this process, browser and
all temporary data. Fault variants calibrate the test, not product capability.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hackathon.profiles import ProfileManager
from hackathon.runtime import Runtime
from hackathon import server as server_module


class SilentSpeaker:
    available = True

    def __init__(self):
        self.calls = []

    def speak(self, text):
        self.calls.append(text)

    def close(self):
        pass


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Calibration anchor changed; update the behavior mutation explicitly')
    return source.replace(old, new, 1)


def prepare_ui(destination, variant):
    shutil.copytree(ROOT / 'hackathon/ui', destination)
    app = destination / 'app.js'
    source = app.read_text()
    if variant == 'conflict-stale':
        source = replace_once(source,
            "          if (editorDirty && !confirmDiscard('Use these selected words instead of your unfinished edit? The unfinished edit will be replaced.')) return;\n", '')
        source = replace_once(source,
            "          if ($('corrected-text')) $('corrected-text').value = candidate;\n          editorDirty = false; persistEditor();",
            "          persistEditor();")
    elif variant == 'draft-loss':
        source = replace_once(source,
            "      } else if (!editorDirty) $('corrected-text').value = d.text || '';",
            "      } else { $('corrected-text').value = d.text || ''; editorDirty = false; persistEditor(); }")
    elif variant == 'choice-label-mismatch':
        source = replace_once(source,
            "        button.textContent = candidate === s.cleanup?.suggestion && s.cleanup?.status === 'ready' ? `AI suggestion: ${candidate}` : candidate;",
            "        button.textContent = candidate === 'No.' ? 'I am comfortable' : 'No.';")
    elif variant == 'confirm-edit-stale':
        source = replace_once(source,
            "    if (editorDirty || editVersion !== version || editorProfile !== profile || chosenText !== message\n        || state.revision !== revision || state.decision.status !== 'accepted' || state.decision.text !== message) {",
            "    if (state.revision !== revision || state.decision.status !== 'accepted' || state.decision.text !== message) {")
    app.write_text(source)


def identity(ui):
    source_paths = [
        'hackathon/server.py', 'hackathon/runtime.py', 'hackathon/speech.py',
        'hackathon/signs/lab.py', 'hackathon/signs/lab.html',
        'scripts/check_web.mjs', 'scripts/web_check_fixture.py',
    ]
    return {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ui.iterdir()) if p.is_file()},
        'source_files': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                         for name in source_paths if (ROOT / name).is_file()},
    }


class Fixture:
    def __init__(self, root):
        self.manager = ProfileManager(root / 'profiles')
        self.manager.ensure_default_profile()
        self.manager.create_profile('other', 'Other test profile')
        self.current = None

    def start(self, port=0):
        runtime = Runtime(rehearsal=True, profile_manager=self.manager)
        runtime.speaker.close()
        runtime.speaker = SilentSpeaker()
        server = server_module.make_server(runtime, port)
        thread = threading.Thread(target=server.serve_forever,
                                  kwargs={'poll_interval': 0.05}, daemon=True)
        thread.start()
        self.current = runtime, server, thread
        return server.server_port

    def snapshot(self):
        runtime, _, _ = self.current
        with runtime.lock:
            return {'state': runtime.snapshot(), 'speech_calls': list(runtime.speaker.calls)}

    def close(self):
        if self.current:
            runtime, server, thread = self.current
            self.current = None
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            runtime.close()

    def restart(self):
        port = self.current[1].server_port
        self.close()
        self.start(port)
        return self.snapshot()


def emit(payload):
    print(json.dumps(payload), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['control', 'conflict-stale', 'draft-loss', 'confirm-edit-stale', 'choice-label-mismatch'], default='control')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='quietbridge-web-check-') as temporary:
        root = Path(temporary)
        ui = root / 'ui'
        prepare_ui(ui, args.variant)
        server_module.UI = ui
        fixture = Fixture(root)
        try:
            port = fixture.start()
            emit({'ready': True, 'url': f'http://127.0.0.1:{port}/',
                  'identity': identity(ui), 'variant': args.variant,
                  'limits': 'Rehearsal + silent speaker only. No camera, workers, model training or provider calls.'})
            for line in sys.stdin:
                command = json.loads(line)
                request_id, operation = command['id'], command['op']
                if operation == 'snapshot':
                    emit({'id': request_id, 'ok': True, **fixture.snapshot()})
                elif operation == 'restart':
                    emit({'id': request_id, 'ok': True, **fixture.restart()})
                elif operation == 'diagnostics':
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                    emit({'id': request_id, 'ok': True})
                elif operation == 'stop':
                    emit({'id': request_id, 'ok': True})
                    break
                else:
                    emit({'id': request_id, 'ok': False, 'error': 'Unknown fixture command'})
        finally:
            fixture.close()


if __name__ == '__main__':
    main()
