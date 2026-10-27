"""Controller-side RVC interface. Model imports occur only in venv-rvc workers."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid
import soundfile as sf
from audio_io import read_audio, write_wav
from gpu_lock import gpu_job
from speech.contracts import SpeechOptions, SpeechResult, SpeechError
from speech.diagnostics import atomic_json
from speech.supervisor import run_worker

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RvcOptions:
    model: str
    index: str | None = None
    index_rate: float = 0
    transpose: int = 0
    f0_predictor: str = 'rmvpe'
    speaker_id: int = 0
    device: str = 'cuda'
    load_timeout: float = 180
    chunk_timeout: float = 120
    request_timeout: float = 900
    max_input_seconds: float = 600

    def validate(self):
        if self.device not in ('cpu', 'cuda') or self.f0_predictor not in ('rmvpe','harvest','pm'):
            raise ValueError('Unsupported RVC device or pitch detector')
        for value in (self.index_rate, self.load_timeout, self.chunk_timeout,
                      self.request_timeout, self.max_input_seconds):
            if isinstance(value, bool) or not isinstance(value, (float,int)) or not math.isfinite(value):
                raise ValueError('RVC numeric options must be finite numbers')
        if not 0 <= self.index_rate <= 1:
            raise ValueError('Index strength must be between zero and one')
        if any(v <= 0 for v in (self.load_timeout,self.chunk_timeout,self.request_timeout,self.max_input_seconds)):
            raise ValueError('RVC deadlines/input limit must be positive')
        if self.max_input_seconds > 3600:
            raise ValueError('RVC request input limit cannot exceed one hour')
        for name, low, high in (('transpose',-24,24),('speaker_id',0,1023)):
            value = getattr(self,name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'Invalid RVC {name}')
        if not self.model or not Path(self.model).is_file() or Path(self.model).suffix.lower() != '.pth':
            raise FileNotFoundError('Select an existing trusted exported RVC .pth checkpoint')
        if Path(self.model).stat().st_size > 1024**3:
            raise ValueError('RVC checkpoint exceeds the 1 GiB inference limit')
        if self.index_rate:
            if not self.index or not Path(self.index).is_file():
                raise FileNotFoundError('Nonzero retrieval strength requires an existing index')
            if Path(self.index).suffix.lower() != '.index' or Path(self.index).stat().st_size > 512*1024**2:
                raise ValueError('Unsupported retrieval index or index exceeds 512 MiB')


class RvcConverter:
    def __init__(self, options: RvcOptions, cancel=None, diagnostics=None):
        self.options, self.cancel = options, cancel
        self.diagnostics = Path(diagnostics) if diagnostics else None

    def convert(self, speech: SpeechResult, output: Path) -> Path:
        self.options.validate()
        from rvc_runtime import python_path, verify_source, verify_assets
        interpreter = python_path()
        if not interpreter.is_file():
            raise FileNotFoundError('RVC runtime missing; run setup_rvc.ps1')
        source, output = speech.path.resolve(), Path(output).resolve()
        if source == output or (output.exists() and os.path.samefile(source,output)):
            raise ValueError('RVC output must not alias its input')
        if output.suffix.lower() != '.wav':
            raise ValueError('RVC output must be WAV')
        # Also protect explicitly selected model/index assets against overwrite.
        for protected in (self.options.model, self.options.index):
            if protected and (output == Path(protected).resolve() or
                              (output.exists() and Path(protected).exists() and os.path.samefile(output,protected))):
                raise ValueError('RVC destination aliases a selected asset')
        info = sf.info(str(source))
        if (info.samplerate,info.subtype,info.channels) != (48000,'PCM_24',1):
            raise ValueError('RVC input must be 48 kHz PCM24 mono WAV')
        if not .15 <= info.duration <= self.options.max_input_seconds:
            raise ValueError('RVC input duration is outside the configured request limit')
        if self.cancel is not None and self.cancel.is_set():
            raise SpeechError('RVC request cancelled')
        verify_source()
        verify_assets()
        deadline = time.monotonic()+self.options.request_timeout
        with tempfile.TemporaryDirectory(prefix='rvc-worker-') as temporary:
            directory = self.diagnostics/uuid.uuid4().hex if self.diagnostics else Path(temporary)
            directory.mkdir(parents=True,exist_ok=True)
            options = asdict(self.options)
            options['model'] = str(Path(options['model']).resolve())
            options['index'] = str(Path(options['index']).resolve()) if options['index'] else None
            atomic_json(directory/'request.json',dict(input=str(source),options=options))
            deadlines = SpeechOptions(load_timeout=self.options.load_timeout,
                                      chunk_timeout=self.options.chunk_timeout,
                                      request_timeout=self.options.request_timeout,
                                      device=self.options.device)
            lock = gpu_job() if self.options.device == 'cuda' else nullcontext()
            with lock:
                # 'bark' means retain CUDA visibility in the shared process runner.
                run_worker([str(interpreter),'-X','utf8','-B',str(ROOT/'rvc_worker.py'),str(directory)],
                           directory,deadlines,deadline,self.cancel,
                           backend='bark' if self.options.device == 'cuda' else 'rvc-cpu')
            if self.cancel is not None and self.cancel.is_set():
                raise SpeechError('RVC cancelled before publication')
            if time.monotonic() >= deadline:
                raise SpeechError('RVC deadline exceeded before publication')
            result = directory/'result.wav'
            metadata = sf.info(str(result))
            if (metadata.samplerate,metadata.subtype,metadata.channels) != (48000,'PCM_24',1):
                raise SpeechError('RVC worker violated the audio contract')
            audio, sr = read_audio(result)
            from rvc_audio import validate_conversion
            validate_conversion(audio,sr,info.duration)
            if (self.cancel is not None and self.cancel.is_set()) or time.monotonic() >= deadline:
                raise SpeechError('RVC cancelled or deadline exceeded before publication')
            write_wav(output,audio,sr)
        return output
