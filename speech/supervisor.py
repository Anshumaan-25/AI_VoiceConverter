from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

from audio_io import read_audio, write_wav
from gpu_lock import gpu_job, GpuBusyError
from .contracts import SpeechError, SpeechOptions, SpeechResult, WorkerTimeout
from .diagnostics import atomic_json
from .processes import ProcessTree
from .text import split_text

ROOT = Path(__file__).resolve().parents[1]


def automatic_backends():
    qualification = json.loads(Path(__file__).with_name('qualification.json').read_text(encoding='utf-8'))
    backends = qualification['automatic_order']
    if not backends or any(name not in ('bark', 'coqui', 'sapi') for name in backends):
        raise SpeechError('Invalid automatic backend qualification configuration')
    if 'bark' in backends:
        raise SpeechError('Bark is parked indefinitely and cannot be an automatic backend')
    return backends


def run_worker(command, directory, options, deadline, cancel=None, backend='sapi'):
    """Bound startup, each chunk, finalization and the entire request separately."""
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    if backend != 'bark' or options.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    flags = (subprocess.CREATE_NO_WINDOW | 0x00000004) if os.name == 'nt' else 0  # CREATE_SUSPENDED
    tree = None
    process = None
    started = time.monotonic()
    phase_key = ('load', -1)
    phase_started = started
    peak_rss = 0
    try:
        with (directory/'worker.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=flags, start_new_session=os.name != 'nt')
            try:
                tree = ProcessTree(process)
                tree.resume()
            except BaseException:
                # Real worker is waiting on its gate, so it has no descendants or GPU allocations yet.
                process.kill()
                process.wait(timeout=10)
                raise
            (directory/'go').touch()
            while True:
                if cancel is not None and cancel.is_set():
                    raise SpeechError('Speech request cancelled')
                now = time.monotonic()
                if now >= deadline:
                    raise WorkerTimeout('Whole-request deadline exceeded')
                try:
                    state = json.loads((directory/'state.json').read_text(encoding='utf-8'))
                    key = (state['phase'], state['chunk'])
                    if key != phase_key:
                        phase_key, phase_started = key, now
                except (FileNotFoundError, json.JSONDecodeError, PermissionError):
                    pass
                budget = options.load_timeout if phase_key[0] == 'load' else options.chunk_timeout
                if now-phase_started > budget:
                    raise WorkerTimeout(f'{phase_key[0]} timeout at chunk {phase_key[1]} after {budget:g}s')
                if os.fstat(log.fileno()).st_size > 8*1024*1024:
                    raise SpeechError('Worker exceeded diagnostic log limit')
                try:
                    import psutil
                    parent = psutil.Process(process.pid)
                    members = [parent]+parent.children(recursive=True)
                    tree.members.update({p.pid: p for p in members})
                    peak_rss = max(peak_rss, sum(p.memory_info().rss for p in members))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                code = process.poll()
                if code is not None:
                    if code:
                        error_file = directory/'error.json'
                        error = json.loads(error_file.read_text(encoding='utf-8')) if error_file.exists() else {}
                        raise SpeechError(error.get('message', f'Worker exited with code {code}; see worker.log'))
                    return
                time.sleep(.05)
    finally:
        if tree is not None:
            tree.close()  # No next backend until worker termination is confirmed.
        atomic_json(directory/'process.json', dict(pid=process.pid if process else None,
                    elapsed_seconds=time.monotonic()-started, peak_tree_rss_bytes=peak_rss,
                    exited=process is not None and process.poll() is not None,
                    returncode=process.returncode if process else None))


class IsolatedSpeechGenerator:
    def generate(self, text, output, options=None, cancel=None):
        options = options or SpeechOptions()
        options.validate()
        output = Path(output).expanduser().resolve()
        if output.suffix.lower() != '.wav':
            raise ValueError('Speech output must be WAV')
        chunks = split_text(text)
        deadline = time.monotonic()+options.request_timeout
        diagnostics = None
        if options.diagnostics:
            diagnostics = Path(options.diagnostics).expanduser().resolve()/uuid.uuid4().hex
            diagnostics.mkdir(parents=True)
        attempts = []
        # Explicit model selection is available for diagnostics, including unqualified models.
        if options.backend == 'auto':
            backends = automatic_backends()
        else:
            backends = [options.backend]
        with tempfile.TemporaryDirectory(prefix='voice-worker-') as temporary:
            workspace = diagnostics or Path(temporary)
            for backend_index, backend in enumerate(backends):
                if time.monotonic() >= deadline or (cancel is not None and cancel.is_set()):
                    break
                directory = workspace/backend
                directory.mkdir()
                request = dict(backend=backend, chunks=chunks, options=asdict(options))
                atomic_json(directory/'request.json', request)
                print(f'[tts] starting isolated {backend} worker', flush=True)
                # A hung primary must not consume the fallback's entire budget.
                # Reserve bounded time for a fresh utterance, not just its last chunk.
                reserve = min(180., max(30., len(chunks)*5.), options.request_timeout*.3)
                attempt_deadline = deadline-reserve if backend_index < len(backends)-1 else deadline
                try:
                    lock = gpu_job() if backend == 'bark' and options.device != 'cpu' else nullcontext()
                    with lock:
                        run_worker([sys.executable, '-X', 'utf8', '-B', '-m', 'speech.worker', str(directory)],
                                   directory, options, attempt_deadline, cancel, backend)
                    result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
                    # Verify the persisted contract before exposing an artifact to conversion.
                    import soundfile as sf
                    info = sf.info(str(directory/'result.wav'))
                    if (info.samplerate, info.subtype, info.channels) != (48000, 'PCM_24', 1):
                        raise SpeechError('Worker output violates the audio contract')
                    audio, sr = read_audio(directory/'result.wav')
                except (SpeechError, GpuBusyError, OSError, ValueError) as exc:
                    attempts.append(dict(backend=backend, status='failed', error=str(exc)))
                    print(f'[tts] {backend} rejected: {exc}', flush=True)
                    if diagnostics:
                        atomic_json(diagnostics/'summary.json', dict(attempts=attempts, status='failed'))
                    continue
                if cancel is not None and cancel.is_set():
                    raise SpeechError('Speech request cancelled before publication')
                if time.monotonic() >= deadline:
                    raise WorkerTimeout('Request deadline exceeded before publication')
                # Publishing failures are I/O errors, not reasons to regenerate with another voice.
                write_wav(output, audio, sr)
                attempts.append(dict(backend=backend, status='passed'))
                report = dict(backend=backend, chunks=result['chunks'], attempts=attempts,
                              output=str(output), diagnostics=str(diagnostics) if diagnostics else None)
                if diagnostics:
                    atomic_json(diagnostics/'summary.json', report)
                return SpeechResult(output, backend, result['chunks'], attempts, diagnostics)
            if cancel is not None and cancel.is_set():
                raise SpeechError('Speech request cancelled')
            raise SpeechError('No speech backend completed within the request budget: '+
                              '; '.join(f"{a['backend']}: {a.get('error', '')}" for a in attempts))
