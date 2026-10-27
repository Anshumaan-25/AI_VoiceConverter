from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import math


class SpeechError(RuntimeError):
    pass


class WorkerTimeout(SpeechError):
    pass


@dataclass(frozen=True)
class SpeechOptions:
    backend: str = "auto"
    seed: int = 42
    device: str = "auto"
    bark_voice: str = "v2/en_speaker_6"
    bark_text_temp: float = 0.7
    bark_waveform_temp: float = 0.7
    bark_min_eos_p: float = 0.2
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    coqui_speaker: str | None = None
    coqui_language: str | None = None
    sapi_voice: str | None = None
    load_timeout: float = 240
    chunk_timeout: float = 180
    request_timeout: float = 900
    max_chunk_seconds: float = 12
    diagnostics: str | None = None

    def validate(self):
        if self.backend == 'bark':
            raise ValueError('Bark is parked indefinitely. Use SAPI for speech or the Suno vocal-to-RVC workflow for songs.')
        if self.backend not in ("auto", "bark", "coqui", "sapi"):
            raise ValueError("Unknown speech backend")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu or cuda")
        for name in ("load_timeout", "chunk_timeout", "request_timeout", "max_chunk_seconds",
                     "bark_text_temp", "bark_waveform_temp"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.bark_min_eos_p) or not 0 <= self.bark_min_eos_p <= 1:
            raise ValueError("EOS probability must be between zero and one")
        if not 0 < self.max_chunk_seconds <= 12:
            raise ValueError("Chunk duration must be at most 12 seconds")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be in [0, 2**32)")


@dataclass
class SpeechResult:
    path: Path
    backend: str
    chunks: list[dict]
    attempts: list[dict] = field(default_factory=list)
    diagnostics: Path | None = None


class SpeechGenerator(Protocol):
    def generate(self, text: str, output: Path, options: SpeechOptions) -> SpeechResult: ...


class AudioValidator(Protocol):
    def validate(self, audio, sample_rate: int, text: str, max_seconds: float) -> dict: ...


class VoiceConverter(Protocol):
    """Future RVC adapter consumes a validated 48 kHz PCM24 WAV artifact."""
    def convert(self, speech: SpeechResult, output: Path) -> Path: ...
