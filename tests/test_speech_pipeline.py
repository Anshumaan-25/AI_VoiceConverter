import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from unittest.mock import patch
import numpy as np
import psutil
import soundfile as sf
from audio_io import write_wav
from speech.contracts import SpeechError, SpeechOptions, WorkerTimeout
from speech.supervisor import IsolatedSpeechGenerator, run_worker
from speech.text import split_text
from speech.validation import SpeechValidator

ROOT = Path(__file__).resolve().parents[1]


class ValidationTests(unittest.TestCase):
    def test_float_headroom_is_attenuated_without_false_clipping(self):
        from speech.validation import finish_chunk
        signal = (1.2*np.sin(2*np.pi*333*np.arange(48000)/48000)).astype(np.float32)
        with patch('webrtcvad.Vad') as detector:
            detector.return_value.is_speech.return_value = True
            report = SpeechValidator().validate(signal, 48000, 'Hello.', 6)
        self.assertGreater(report['native_peak'], 1)
        self.assertLessEqual(float(np.max(np.abs(finish_chunk(signal,48000)))), .951)

    def test_controller_import_does_not_load_torch(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            "import sys; from speech.supervisor import IsolatedSpeechGenerator; assert 'torch' not in sys.modules"],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_silence_and_nonfinite_and_runaway_are_rejected(self):
        validator = SpeechValidator()
        for audio in (np.zeros(48000), np.full(48000, np.nan), np.ones(48000*13)*.1):
            with self.subTest(shape=audio.shape), self.assertRaises((SpeechError, ValueError)):
                validator.validate(audio, 48000, 'Hello.', 6)

    def test_carrier_rejected_before_vad(self):
        audio = .2*np.sin(2*np.pi*625*np.arange(48000*4)/48000)
        with self.assertRaisesRegex(SpeechError, 'carrier'):
            SpeechValidator().validate(audio, 48000, 'Hello.', 6)

    def test_text_bounds_preserve_all_words(self):
        text = 'Hello, world! '+ ' '.join(['speech']*60)+'.'
        self.assertEqual(' '.join(split_text(text)), text)
        self.assertGreater(len(split_text(text)), 3)
        for value in ('', '...', 'bad\x00text', 'a'*181):
            with self.assertRaises(SpeechError):
                split_text(value)

    def test_invalid_settings_fail_before_worker(self):
        for options in (SpeechOptions(chunk_timeout=float('nan')), SpeechOptions(max_chunk_seconds=90), SpeechOptions(seed=-1)):
            with self.assertRaises(ValueError):
                options.validate()


class SupervisorTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows kill-on-close job test')
    def test_controller_death_kills_worker_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            controller = subprocess.Popen([sys.executable, '-B', str(ROOT/'tests'/'controller_fixture.py'), str(directory)],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                          creationflags=subprocess.CREATE_NO_WINDOW)
            controller_pid = None
            try:
                deadline = time.monotonic()+15
                while not (directory/'child.pid').exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue((directory/'child.pid').exists())
                controller_pid = int((directory/'controller.pid').read_text())
                owned = [psutil.Process(int((directory/name).read_text())) for name in ('worker.pid','child.pid')]
                psutil.Process(controller_pid).kill()
                controller.wait(timeout=10)
                _, alive = psutil.wait_procs(owned, timeout=10)
                self.assertEqual(alive, [])
            finally:
                if controller_pid is None and (directory/'controller.pid').exists():
                    controller_pid = int((directory/'controller.pid').read_text())
                if controller_pid and psutil.pid_exists(controller_pid):
                    psutil.Process(controller_pid).kill()
                if controller.poll() is None:
                    controller.kill()
                controller.wait(timeout=10)

    def fixture(self, directory, mode, options, cancel=None):
        return run_worker([sys.executable, '-B', str(ROOT/'tests'/'worker_fixture.py'), str(directory), mode],
                          directory, options, time.monotonic()+30, cancel)

    def assert_dead(self, directory):
        process = json.loads((directory/'process.json').read_text())
        self.assertTrue(process['exited'])
        self.assertFalse(psutil.pid_exists(process['pid']))
        child = directory/'child.pid'
        if child.exists():
            pid = int(child.read_text())
            for _ in range(20):
                if not psutil.pid_exists(pid):
                    break
                time.sleep(.05)
            self.assertFalse(psutil.pid_exists(pid))

    def test_load_timeout_kills_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaises(WorkerTimeout):
                self.fixture(directory, 'load', SpeechOptions(load_timeout=.5))
            self.assert_dead(directory)

    def test_chunk_timeout_kills_descendants(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaises(WorkerTimeout):
                self.fixture(directory, 'tree', SpeechOptions(chunk_timeout=.5))
            self.assertTrue((directory/'child.pid').exists())
            self.assert_dead(directory)

    def test_crash_kills_descendants(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaisesRegex(SpeechError, 'code 7'):
                self.fixture(directory, 'crash', SpeechOptions())
            self.assert_dead(directory)

    def test_cancellation_kills_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            cancel = threading.Event()
            timer = threading.Timer(.5, cancel.set)
            timer.start()
            try:
                with self.assertRaisesRegex(SpeechError, 'cancelled'):
                    self.fixture(directory, 'load', SpeechOptions(), cancel)
            finally:
                timer.cancel()
            self.assert_dead(directory)

    def test_whole_request_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaisesRegex(WorkerTimeout, 'Whole-request'):
                run_worker([sys.executable, '-B', str(ROOT/'tests'/'worker_fixture.py'), str(directory), 'load'],
                           directory, SpeechOptions(), time.monotonic()+.5)
            self.assert_dead(directory)

    def test_busy_gpu_can_fall_back_without_starting_gpu_worker(self):
        from gpu_lock import GpuBusyError
        with tempfile.TemporaryDirectory() as temp:
            def fake_run(command, work, options, deadline, cancel, backend):
                self.assertEqual(backend, 'sapi')
                write_wav(work/'result.wav', np.ones(48000)*.1, 48000)
                (work/'result.json').write_text(json.dumps(dict(chunks=[{'text':'Hello.'}])))
            with patch('speech.supervisor.gpu_job', side_effect=GpuBusyError('busy')), \
                 patch('speech.supervisor.run_worker', side_effect=fake_run) as worker, \
                 patch('speech.supervisor.automatic_backends', return_value=['bark','sapi']):
                result = IsolatedSpeechGenerator().generate('Hello.', Path(temp)/'out.wav')
            self.assertEqual(result.backend, 'sapi')
            worker.assert_called_once()

    def test_fallback_restarts_entire_utterance_and_preserves_old_output_on_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            output = directory/'result.wav'
            output.write_bytes(b'previous-result')
            calls = []
            def fake_run(command, work, options, deadline, cancel, backend):
                calls.append(backend)
                if backend == 'bark':
                    raise SpeechError('rejected chunk')
                write_wav(work/'result.wav', np.ones(48000)*.1, 48000)
                (work/'result.json').write_text(json.dumps(dict(chunks=[{'text':'Hello.'}])))
            with patch('speech.supervisor.run_worker', side_effect=fake_run), \
                 patch('speech.supervisor.gpu_job', side_effect=nullcontext), \
                 patch('speech.supervisor.automatic_backends', return_value=['bark','sapi']):
                result = IsolatedSpeechGenerator().generate('Hello.', output)
            self.assertEqual(calls, ['bark','sapi'])
            self.assertEqual(result.backend, 'sapi')
            self.assertEqual(sf.info(str(output)).subtype, 'PCM_24')
            previous = output.read_bytes()
            with patch('speech.supervisor.run_worker', side_effect=SpeechError('all failed')), \
                 patch('speech.supervisor.gpu_job', side_effect=nullcontext):
                with self.assertRaises(SpeechError):
                    IsolatedSpeechGenerator().generate('Hello.', output)
            self.assertEqual(output.read_bytes(), previous)

    @unittest.skipUnless(os.environ.get('VOICE_TEST_GPU') == '1', 'Explicit GPU lifecycle test')
    def test_gpu_allocation_is_released_after_forced_timeout(self):
        from benchmark_speech import gpu_usage
        before = gpu_usage()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaises(WorkerTimeout):
                run_worker([sys.executable, '-B', str(ROOT/'tests'/'worker_fixture.py'), str(directory), 'gpu'],
                           directory, SpeechOptions(chunk_timeout=.5), time.monotonic()+30, backend='bark')
            self.assertGreater(json.loads((directory/'allocated.json').read_text())['bytes'], 100*1024**2)
            self.assert_dead(directory)
        after = gpu_usage()
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertLessEqual(after[0], before[0]+32)


if __name__ == '__main__':
    unittest.main()
