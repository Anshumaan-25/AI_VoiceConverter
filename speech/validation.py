"""Speech-only checks. VAD is not a transcript or an intelligibility score."""
import numpy as np
from audio_io import resample, validate_audio
from .contracts import SpeechError


class SpeechValidator:
    def validate(self, audio, sample_rate: int, text: str, max_seconds: float) -> dict:
        audio = validate_audio(audio, sample_rate)
        if audio.ndim != 1:
            raise SpeechError("Speech backends must return mono audio")
        duration = len(audio) / sample_rate
        if duration > max_seconds + .08:
            raise SpeechError(f"Chunk duration {duration:.2f}s exceeds {max_seconds:.2f}s")
        rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
        if duration < .15 or rms < 1e-4:
            raise SpeechError("Chunk is empty, silent or too short for speech")
        # Neural decoders return floating PCM, which may exceed unity without
        # clipping. Attenuate that signal; reject actual plateaus at full scale.
        peak_amplitude = float(np.max(np.abs(audio)))
        clipped = float(np.mean(np.isclose(np.abs(audio), 1.0, atol=1e-7, rtol=0)))
        if clipped > .005 or peak_amplitude > 8:
            raise SpeechError("Generated speech is saturated or has implausible amplitude")
        # Detect persistent carrier output before normalization or trimming.
        window = sample_rate // 2
        run = 0
        prior = None
        maximum_run = 0
        for start in range(0, len(audio)-window+1, window):
            segment = audio[start:start+window]
            power = np.abs(np.fft.rfft(segment*np.hanning(window)))**2
            peak = int(np.argmax(power))
            frequency = peak*sample_rate/window
            ratio = power[max(0, peak-1):peak+2].sum()/max(power.sum(), 1e-20)
            tonal = frequency > 80 and ratio > .8 and np.sqrt(np.mean(segment**2)) > .003
            run = (run+1 if prior is not None and abs(frequency-prior) <= 10 else 1) if tonal else 0
            prior = frequency if tonal else None
            maximum_run = max(maximum_run, run)
        if maximum_run >= 6:
            raise SpeechError("Chunk contains a sustained carrier tone")
        import webrtcvad
        vad = webrtcvad.Vad(2)
        down = resample(audio / max(1.0, peak_amplitude/.95), sample_rate, 16000)
        pcm = (np.clip(down, -1, 1)*32767).astype('<i2')
        flags = [vad.is_speech(pcm[i:i+320].tobytes(), 16000)
                 for i in range(0, len(pcm)-319, 320)]
        active = sum(flags)*.02
        fraction = sum(flags)/max(len(flags), 1)
        if active < .12 or fraction < .12:
            raise SpeechError(f"Insufficient detected speech: {active:.2f}s ({fraction:.0%})")
        return dict(duration_seconds=duration, rms=rms, clipped_fraction=clipped, native_peak=peak_amplitude,
                    speech_seconds=active, speech_fraction=fraction,
                    carrier_seconds=maximum_run*.5)


def finish_chunk(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Keep 120 ms padding around audible content; never boost quiet noise."""
    peak = float(np.max(np.abs(audio)))
    active = np.flatnonzero(np.abs(audio) >= max(1e-5, peak*.001))
    if not len(active):
        raise SpeechError("No audible content")
    pad = round(.12*sample_rate)
    result = audio[max(0, active[0]-pad):min(len(audio), active[-1]+pad+1)].copy()
    # Only attenuate; avoid turning low-level artifacts into loud output.
    if peak > .95:
        result *= .95/peak
    return result
