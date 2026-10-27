"""Adapters consume the same persisted speech contract, independently of TTS."""
from dataclasses import dataclass
from pathlib import Path
import soundfile as sf
from audio_io import read_audio, write_wav
from speech.contracts import SpeechResult


@dataclass
class SvcConverter:
    model: Path
    config: Path
    speaker: str
    transpose: int = 0
    f0_predictor: str = 'rmvpe'
    timeout: float = 900
    cancel: object = None

    def convert(self, speech: SpeechResult, output: Path) -> Path:
        from voice_autopilot import infer_once
        source, output = speech.path.resolve(), Path(output).resolve()
        if source == output:
            raise ValueError('Conversion must use a new output path')
        if output.suffix.lower() != '.wav':
            raise ValueError('Conversion output must be WAV')
        info = sf.info(str(source))
        if (info.samplerate, info.subtype, info.channels) != (48000, 'PCM_24', 1):
            raise ValueError('Converter input must be 48 kHz PCM24 mono WAV')
        read_audio(source)
        generated = infer_once(source, self.model, self.config, self.speaker,
                               self.transpose, self.f0_predictor,
                               timeout=self.timeout, cancel=self.cancel)
        if self.cancel is not None and self.cancel.is_set():
            raise RuntimeError('Conversion cancelled before publication')
        audio, sr = read_audio(generated)
        write_wav(output, audio, sr)
        return output
