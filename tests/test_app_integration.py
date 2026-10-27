import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from contextlib import nullcontext
import numpy as np
import psutil
import soundfile as sf
from app_service import SessionJobs, VoiceApplication
from audio_io import write_wav
from conversion import SvcConverter
from speech.contracts import SpeechOptions, SpeechResult, WorkerTimeout
from speech.supervisor import IsolatedSpeechGenerator
from voice_autopilot import run_cmd

ROOT = Path(__file__).resolve().parents[1]


class ApplicationTests(unittest.TestCase):
    def test_sessions_cancel_independently_and_release_after_failure(self):
        jobs = SessionJobs()
        with self.assertRaisesRegex(RuntimeError, 'operation failed'):
            with jobs.start('alice') as alice, jobs.start('bob') as bob:
                self.assertTrue(jobs.cancel('alice'))
                self.assertTrue(alice.is_set())
                self.assertFalse(bob.is_set())
                with self.assertRaisesRegex(RuntimeError, 'active job'):
                    with jobs.start('alice'):
                        pass
                raise RuntimeError('operation failed')
        self.assertFalse(jobs.cancel('alice'))
        with jobs.start('alice') as fresh:
            self.assertFalse(fresh.is_set())

    def test_converter_rejects_wrong_contract_before_loading_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'input.wav'
            sf.write(source, np.ones(24000)*.1, 24000, subtype='PCM_16')
            with patch('voice_autopilot.infer_once') as infer:
                with self.assertRaisesRegex(ValueError, 'PCM24'):
                    SvcConverter(Path('model'), Path('config'), 'speaker').convert(
                        SpeechResult(source, 'test', []), root/'output.wav')
            infer.assert_not_called()

    def test_conversion_failure_preserves_input_and_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root/'source.wav', root/'output.wav'
            write_wav(source, np.ones(48000)*.1, 48000)
            original = source.read_bytes()
            output.write_bytes(b'previous')
            with patch('voice_autopilot.infer_once', side_effect=RuntimeError('failed')):
                with self.assertRaises(RuntimeError):
                    SvcConverter(Path('model'), Path('config'), 'speaker').convert(
                        SpeechResult(source, 'test', []), output)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(output.read_bytes(), b'previous')
            app = VoiceApplication(root/'app')
            with self.assertRaises(FileNotFoundError):
                app.convert('session', source, root/'missing.pth', root/'missing.json', 'speaker')
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(list((root/'app').rglob('source.wav')), [])
            self.assertFalse(app.jobs.cancel('session'))

    def test_timeout_leaves_time_for_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            deadlines = []
            def run(command, work, options, deadline, cancel, backend):
                deadlines.append(deadline)
                if backend == 'bark':
                    raise WorkerTimeout('Primary deadline reached')
                write_wav(work/'result.wav', np.ones(48000)*.1, 48000)
                (work/'result.json').write_text(json.dumps({'chunks': []}))
            with patch('speech.supervisor.run_worker', side_effect=run), \
                 patch('speech.supervisor.gpu_job', side_effect=nullcontext):
                result = IsolatedSpeechGenerator().generate('Hello.', Path(temporary)/'out.wav',
                                                            SpeechOptions(request_timeout=10))
            self.assertEqual(result.backend, 'sapi')
            self.assertAlmostEqual(deadlines[1]-deadlines[0], 3)

    def test_publication_error_does_not_regenerate(self):
        with tempfile.TemporaryDirectory() as temporary:
            def run(command, work, options, deadline, cancel, backend):
                write_wav(work/'result.wav', np.ones(48000)*.1, 48000)
                (work/'result.json').write_text(json.dumps({'chunks': []}))
            with patch('speech.supervisor.run_worker', side_effect=run) as worker, \
                 patch('speech.supervisor.write_wav', side_effect=PermissionError('read only')), \
                 patch('speech.supervisor.gpu_job', side_effect=nullcontext):
                with self.assertRaises(PermissionError):
                    IsolatedSpeechGenerator().generate('Hello.', Path(temporary)/'out.wav')
            worker.assert_called_once()

    @unittest.skipUnless(os.name == 'nt', 'Windows process containment')
    def test_conversion_command_timeout_and_cancel_kill_descendants(self):
        for mode in ('timeout', 'cancel'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory/'go').touch()
                event = threading.Event()
                timer = threading.Timer(1, event.set)
                if mode == 'cancel':
                    timer.start()
                try:
                    with self.assertRaises((subprocess.TimeoutExpired, RuntimeError)):
                        run_cmd([sys.executable, '-B', str(ROOT/'tests'/'worker_fixture.py'),
                                 str(directory), 'tree'], timeout=1 if mode == 'timeout' else 20, cancel=event)
                finally:
                    timer.cancel()
                child = int((directory/'child.pid').read_text())
                deadline = time.monotonic()+5
                while psutil.pid_exists(child) and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertFalse(psutil.pid_exists(child))

    def test_ui_preserves_24_bit_audio_bytes(self):
        import gradio as gr
        from gradio_app import build_ui
        app = build_ui()
        self.assertTrue(app.enable_queue)
        self.assertTrue(any(c['type'] == 'file' for c in app.config['components']))
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/'sample.wav'
            write_wav(source, np.ones(4800)*.123456, 48000)
            uploaded = gr.File(type='file').preprocess([{'name': str(source), 'data': None, 'is_file': True}])
            try:
                self.assertEqual(Path(uploaded.name).read_bytes(), source.read_bytes())
                playback = gr.Audio(type='filepath').postprocess(str(source))
                self.assertEqual(Path(playback['name']).read_bytes(), source.read_bytes())
            finally:
                uploaded.close()

    def test_ui_page_and_config_render_over_http(self):
        from starlette.testclient import TestClient
        from gradio_app import build_ui
        app = build_ui()
        with TestClient(app.app) as client:
            page = client.get('/')
            self.assertEqual(page.status_code, 200)
            self.assertIn('text/html', page.headers['content-type'])
            config = client.get('/config')
            self.assertEqual(config.status_code, 200)
            self.assertTrue(config.json()['enable_queue'])

    def test_ui_state_identity_is_stable_and_separate_per_session(self):
        from gradio.state_holder import SessionState
        from gradio_app import build_ui, _session_id
        app = build_ui()
        state_id = next(c['id'] for c in app.config['components']
                        if c['type'] == 'state' and c['props'].get('value') == {})
        alice, bob = SessionState(app), SessionState(app)
        first = _session_id(alice[state_id])
        self.assertEqual(_session_id(alice[state_id]), first)
        self.assertNotEqual(_session_id(bob[state_id]), first)


if __name__ == '__main__':
    unittest.main()
