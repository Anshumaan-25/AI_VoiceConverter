"""Local application operations; no model imports or GPU state in the controller."""
from contextlib import contextmanager
from pathlib import Path
import threading
import uuid
from audio_io import read_audio, write_wav
from conversion import SvcConverter
from speech.contracts import SpeechOptions, SpeechResult
from speech.supervisor import IsolatedSpeechGenerator

ROOT = Path(__file__).resolve().parent


class SessionJobs:
    """One running operation per browser session; cancellation cannot cross sessions."""
    def __init__(self):
        self._lock = threading.Lock()
        self._active = {}

    @contextmanager
    def start(self, session):
        if not session:
            raise ValueError('A browser session is required')
        with self._lock:
            if session in self._active:
                raise RuntimeError('This session already has an active job')
            cancel = threading.Event()
            self._active[session] = cancel
        try:
            yield cancel
        finally:
            with self._lock:
                self._active.pop(session, None)

    def cancel(self, session):
        with self._lock:
            event = self._active.get(session)
            if event is None:
                return False
            event.set()
            return True


class VoiceApplication:
    def __init__(self, output_root=None):
        self.output_root = Path(output_root) if output_root else ROOT/'logs'/'app'
        self.jobs = SessionJobs()

    def generate(self, session, text, backend='sapi', diagnostics=False):
        with self.jobs.start(session) as cancel:
            run = self.output_root/uuid.uuid4().hex
            options = SpeechOptions(backend=backend,
                                    diagnostics=str(run/'diagnostics') if diagnostics else None)
            return IsolatedSpeechGenerator().generate(text, run/'speech.wav', options, cancel)

    def convert(self, session, source, model, config, speaker, transpose=0, f0_predictor='rmvpe',
                backend='so-vits-svc', index=None, index_rate=0, speaker_id=0):
        with self.jobs.start(session) as cancel:
            if backend == 'rvc':
                from rvc_adapter import RvcConverter,RvcOptions
                if isinstance(transpose,bool) or not float(transpose).is_integer() or isinstance(speaker_id,bool) or not float(speaker_id).is_integer():
                    raise ValueError('Pitch shift and speaker ID must be integers')
                options = RvcOptions(str(model),str(index) if index else None,index_rate,
                                     int(transpose),f0_predictor,int(speaker_id))
                options.validate()
                adapter = RvcConverter(options,cancel=cancel)
                import soundfile as sf
                if not .15 <= sf.info(str(source)).duration <= options.max_input_seconds:
                    raise ValueError('RVC source duration is outside the request limit')
            elif backend == 'so-vits-svc':
                adapter = SvcConverter(Path(model),Path(config),speaker,int(transpose),f0_predictor,cancel=cancel)
            else:
                raise ValueError('Unknown voice conversion backend')
            run = self.output_root/uuid.uuid4().hex
            audio, sr = read_audio(source)
            normalized = run/'source.wav'
            write_wav(normalized, audio, sr)
            speech = SpeechResult(normalized, 'uploaded', [])
            try:
                return adapter.convert(speech, run/'converted.wav')
            finally:
                normalized.unlink(missing_ok=True)
