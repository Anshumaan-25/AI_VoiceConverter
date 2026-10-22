from __future__ import annotations

import argparse
import math
import random
import re
import traceback
import wave
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np


DEFAULT_TARGET_SR = 48000
DEFAULT_BARK_VOICE = "v2/en_speaker_6"
DEFAULT_COQUI_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
DEFAULT_BARK_TEXT_TEMP = 0.3
DEFAULT_BARK_WAVEFORM_TEMP = 0.35
DEFAULT_BARK_MIN_EOS_P = 0.2
DEFAULT_BARK_TRIM_TOP_DB = 38
DEFAULT_BARK_CHUNK_WORDS = 25
DEFAULT_BARK_CHUNK_PAUSE_MS = 150


class TTSError(RuntimeError):
    """Raised when TTS generation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1B standalone TTS engine (Bark first, Coqui fallback)")
    parser.add_argument("--text", type=str, default=None, help="Stylized performance text")
    parser.add_argument("--text-file", type=Path, default=None, help="Path to a text file containing stylized text")
    parser.add_argument("--output", type=Path, required=True, help="Output WAV file path")
    parser.add_argument("--model", choices=["auto", "bark", "coqui"], default="auto", help="TTS model selection")
    parser.add_argument("--target-sr", type=int, default=DEFAULT_TARGET_SR, help="Target output sample rate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")

    # Bark options
    parser.add_argument("--bark-voice", type=str, default=DEFAULT_BARK_VOICE, help="Bark history prompt voice preset")
    parser.add_argument("--bark-text-temp", type=float, default=DEFAULT_BARK_TEXT_TEMP, help="Bark text temperature")
    parser.add_argument(
        "--bark-waveform-temp",
        type=float,
        default=DEFAULT_BARK_WAVEFORM_TEMP,
        help="Bark waveform temperature",
    )
    parser.add_argument(
        "--bark-min-eos-p",
        type=float,
        default=DEFAULT_BARK_MIN_EOS_P,
        help="Bark minimum EOS probability for early stopping",
    )

    # Coqui options
    parser.add_argument("--coqui-model", type=str, default=DEFAULT_COQUI_MODEL, help="Coqui model name")
    parser.add_argument("--coqui-speaker", type=str, default=None, help="Optional speaker id for multi-speaker models")
    parser.add_argument("--coqui-language", type=str, default=None, help="Optional language for multilingual models")

    args = parser.parse_args()
    if bool(args.text) == bool(args.text_file):
        parser.error("Provide exactly one of --text or --text-file")
    if args.target_sr != 48000:
        parser.error("Persisted TTS WAV must use 48000 Hz")
    if args.output.suffix.lower() != ".wav":
        parser.error("Output must have a .wav extension")
    if not all(math.isfinite(v) for v in (args.bark_text_temp, args.bark_waveform_temp, args.bark_min_eos_p)) or args.bark_text_temp <= 0 or args.bark_waveform_temp <= 0 or not 0 <= args.bark_min_eos_p <= 1:
        parser.error("Invalid Bark temperature/EOS probability")
    return args


def load_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return args.text_file.read_text(encoding="utf-8")
    return str(args.text)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import os
        import torch
        
        # Bounded default; configurable for CPU fallback benchmarks.
        threads = max(1, int(os.environ.get("VOICE_CPU_THREADS", min(4, os.cpu_count() or 1))))
        torch.set_num_threads(threads)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def split_stylized_lines(text: str) -> List[str]:
    return [part.strip() for line in text.splitlines()
            for part in re.split(r"(?<=[.!?])\s+", line) if part.strip()]


def trailing_pause_seconds(line: str, is_last: bool) -> float:
    if is_last:
        return 0.0

    stripped = line.rstrip()
    if stripped.endswith("..."):
        return 0.35
    if stripped.endswith(("!", "?")):
        return 0.28
    if stripped.endswith((",", ";", ":")):
        return 0.18
    return 0.22


def clean_text_for_bark(text: str) -> str:
    # Preserve Unicode and punctuation needed for natural phrasing.
    return " ".join(text.replace("\x00", " ").split())


def chunk_text_by_words(text: str, max_words: int = DEFAULT_BARK_CHUNK_WORDS) -> List[str]:
    if max_words < 1:
        raise ValueError("max_words must be positive")
    chunks = []
    for sentence in split_stylized_lines(text):
        words = clean_text_for_bark(sentence).split()
        current = []
        for word in words:
            if len(word) > 180:
                raise TTSError("A word exceeds the supported Bark text length")
            if current and (len(current) >= max_words or len(" ".join(current + [word])) > 180):
                chunks.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            chunks.append(" ".join(current))
    return chunks


def normalize_peak_db(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Normalize audio so the peak amplitude hits *target_db*."""
    if not audio.size or not np.isfinite(audio).all():
        raise TTSError("Generated audio is empty or non-finite")
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    target_linear = 10.0 ** (target_db / 20.0)
    return (audio * (target_linear / peak)).astype(np.float32)


def validate_speech_duration(audio: np.ndarray, sample_rate: int, text: str) -> None:
    """Reject obvious decoder runaways; this does not verify spoken content."""
    word_count = len(text.split())
    maximum_seconds = max(8.0, word_count * 2.5 + 3.0)
    if sample_rate <= 0 or len(audio) / sample_rate > maximum_seconds:
        raise TTSError(f"Generated speech exceeded the {maximum_seconds:.1f}s limit for {word_count} words; decoder output is unqualified")


def validate_generated_speech(audio: np.ndarray, sample_rate: int) -> None:
    """Reject sustained carrier-like output instead of exporting it as speech.

    This conservative speech-only check is not an intelligibility score. Six
    consecutive half-second windows must concentrate over 80% of their energy
    into the same narrow frequency band. Long sustained musical notes may also
    be rejected; this TTS path is intended for spoken input to conversion.
    """
    if not audio.size or not np.isfinite(audio).all():
        raise TTSError("Generated speech is empty or non-finite")
    window = sample_rate // 2
    if window < 1:
        raise TTSError("Invalid generated sample rate")
    consecutive = 0
    previous_frequency = None
    for start in range(0, len(audio) - window + 1, window):
        segment = audio[start:start + window]
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(window))) ** 2
        peak = int(np.argmax(spectrum))
        frequency = peak * sample_rate / window
        concentration = float(spectrum[max(0, peak-1):peak+2].sum() / max(float(spectrum.sum()), 1e-20))
        tonal = frequency > 80 and concentration > 0.8 and float(np.sqrt(np.mean(segment ** 2))) > .003
        if tonal:
            consecutive = consecutive + 1 if previous_frequency is not None and abs(frequency - previous_frequency) <= 10 else 1
            previous_frequency = frequency
        else:
            consecutive = 0
            previous_frequency = None
        if consecutive >= 6:
            raise TTSError("Generated audio contains a sustained narrow-band tone and is not qualified speech")


def trim_audio_edges(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    wav = ensure_mono_float32(audio)
    if wav.size == 0:
        return wav

    amplitude = np.abs(wav)
    if not np.any(amplitude > 0):
        return np.empty(0, dtype=np.float32)

    threshold = max(float(10.0 ** (-DEFAULT_BARK_TRIM_TOP_DB / 20.0)), 0.010)
    active = np.where(amplitude >= threshold)[0]
    if active.size == 0:
        return np.empty(0, dtype=np.float32)

    pad_start = int(round(sample_rate * 0.005))
    pad_end = int(round(sample_rate * 0.010))

    start = max(0, int(active[0]) - pad_start)
    end = min(wav.size, int(active[-1]) + pad_end + 1)

    if end <= start:
        return wav
    return wav[start:end]


def hard_noise_gate(audio: np.ndarray, sample_rate: int,
                    threshold_db: float = -40.0,
                    min_silence_ms: float = 100.0) -> np.ndarray:
    """Sterile-Room Protocol: Hard Noise Gate.

    Scans from the END of the audio backwards. If a sustained region
    (>= *min_silence_ms*) drops below *threshold_db* RMS, the audio is
    cropped at the START of that silent region, killing trailing
    hallucinations in one cut.
    """
    if audio.size == 0:
        return audio

    threshold_linear = 10.0 ** (threshold_db / 20.0)
    window_samples = int(round(sample_rate * min_silence_ms / 1000.0))
    if window_samples < 1 or audio.size < window_samples:
        return audio

    # Scan from the end backwards in non-overlapping windows
    crop_point = audio.size  # default: keep everything
    pos = audio.size - window_samples
    while pos >= 0:
        window = audio[pos:pos + window_samples]
        w_rms = float(np.sqrt(np.mean(window ** 2)))
        if w_rms < threshold_linear:
            # This window is silent â€” mark as candidate crop point
            crop_point = pos
            pos -= window_samples
        else:
            # Hit active audio â€” stop scanning
            break

    if crop_point < audio.size:
        # Add a tiny 10ms fade-out to avoid click at the cut
        fade_len = min(int(round(sample_rate * 0.010)), crop_point)
        if fade_len > 0:
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            audio[crop_point - fade_len:crop_point] *= fade
        cropped = audio[:crop_point]
        print(f"[tts] noise-gate: cropped trailing silence at "
              f"{crop_point / sample_rate:.2f}s "
              f"(removed {(audio.size - crop_point) / sample_rate:.2f}s)")
        return cropped

    return audio


def resolve_bark_history_prompt(history_prompt: str) -> str:
    if re.fullmatch(r"v2/en_speaker_\d+", history_prompt):
        return history_prompt

    print(
        f"[tts] Bark history_prompt '{history_prompt}' is not a v2 preset; "
        f"using '{DEFAULT_BARK_VOICE}' for cleaner output"
    )
    return DEFAULT_BARK_VOICE


def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    wav = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(wav).all():
        raise TTSError("Audio contains NaN or infinity")
    if wav.ndim == 1:
        return wav
    if wav.ndim == 2:
        return wav.mean(axis=1).astype(np.float32)
    raise TTSError(f"Unsupported audio shape: {wav.shape}")


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    from audio_io import resample
    return resample(audio, src_sr, dst_sr)


def write_wav_mono(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    from audio_io import write_wav
    write_wav(path, audio, sample_rate)


def concat_with_pauses(segments: Sequence[np.ndarray], pauses_sec: Sequence[float], sample_rate: int) -> np.ndarray:
    parts: List[np.ndarray] = []
    for idx, segment in enumerate(segments):
        parts.append(segment.astype(np.float32))
        pause = pauses_sec[idx]
        if pause > 0:
            parts.append(np.zeros(int(round(sample_rate * pause)), dtype=np.float32))

    if not parts:
        raise TTSError("No audio segments were produced.")
    return np.concatenate(parts)


def synthesize_with_bark(
    lines: Sequence[str],
    seed: int,
    bark_voice: str,
    bark_text_temp: float,
    bark_waveform_temp: float,
    bark_min_eos_p: float,
) -> Tuple[np.ndarray, int]:
    """Chunk-and-Stitch Bark synthesis (Semantic Horizon Limitation).

    1. Flatten all input lines into one string.
    2. Split into tiny N-word chunks (default 25 words).
    3. Generate each chunk independently with stabilized params.
    4. Stitch with 150ms silence buffers.
    5. Apply hard noise gate + peak normalization on the merged result.
    """
    import os
    # Must be set before Bark imports; CPU offloading is essential on 4 GB GPUs.
    os.environ["SUNO_USE_SMALL_MODELS"] = "True"
    os.environ["SUNO_OFFLOAD_CPU"] = "True"
    try:
        from bark import SAMPLE_RATE as bark_sample_rate  # type: ignore
        from bark import preload_models  # type: ignore
        from bark.api import generate_text_semantic, semantic_to_waveform  # type: ignore
    except Exception as exc:
        raise TTSError(
            "Bark is not installed. Install with: pip install -r requirements-runtime.txt"
        ) from exc

    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)
    print("[tts] loading small Bark models with CPU offloading (first run downloads weights)", flush=True)
    preload_models()

    bark_voice = resolve_bark_history_prompt(bark_voice)

    # --- Semantic Horizon Limitation: flatten & chunk by words ---
    full_text = "\n".join(lines)
    bark_chunks = chunk_text_by_words(full_text, max_words=DEFAULT_BARK_CHUNK_WORDS)
    if not bark_chunks:
        raise TTSError("No usable Bark text chunks after normalization.")

    print(f"[tts] chunk-and-stitch: {len(bark_chunks)} chunks "
          f"(max {DEFAULT_BARK_CHUNK_WORDS} words each)")
    for i, c in enumerate(bark_chunks):
        print(f"  [{i}] \"{c}\"")

    set_seed(seed)

    sr = int(bark_sample_rate)
    pause_samples = int(round(sr * DEFAULT_BARK_CHUNK_PAUSE_MS / 1000.0))
    silence_buffer = np.zeros(pause_samples, dtype=np.float32)

    parts: List[np.ndarray] = []
    for idx, chunk_text in enumerate(bark_chunks):
        print(f"[tts] generating chunk {idx + 1}/{len(bark_chunks)}: \"{chunk_text}\"")

        semantic_tokens = generate_text_semantic(
            chunk_text,
            history_prompt=bark_voice,
            temp=bark_text_temp,
            min_eos_p=bark_min_eos_p,
        )
        audio = semantic_to_waveform(
            semantic_tokens,
            history_prompt=bark_voice,
            temp=bark_waveform_temp,
        )
        mono = ensure_mono_float32(np.asarray(audio))
        mono = hard_noise_gate(mono, sample_rate=sr)
        mono = trim_audio_edges(mono, sample_rate=sr)
        if mono.size == 0:
            raise TTSError(f"Bark chunk {idx + 1} produced no usable audio; refusing to omit words")
        parts.append(mono)

        # Insert 150ms silence buffer between chunks (not after the last)
        if idx < len(bark_chunks) - 1:
            parts.append(silence_buffer.copy())

    if not parts:
        raise TTSError("All Bark chunks produced empty audio.")

    merged = np.concatenate(parts)

    # --- Post-processing on the full merged signal ---
    merged = hard_noise_gate(merged, sample_rate=sr)
    merged = normalize_peak_db(merged, target_db=-1.0)
    validate_generated_speech(merged, sr)
    validate_speech_duration(merged, sr, full_text)

    duration = merged.size / sr
    print(f"[tts] chunk-and-stitch complete: {duration:.2f}s, "
          f"{merged.size} samples @ {sr} Hz")

    return merged, sr


def synthesize_with_coqui(
    lines: Sequence[str],
    seed: int,
    coqui_model: str,
    coqui_speaker: str | None,
    coqui_language: str | None,
) -> Tuple[np.ndarray, int]:
    try:
        from TTS.api import TTS  # type: ignore
    except Exception as exc:
        raise TTSError(
            "Coqui TTS is not installed. Install with: pip install TTS"
        ) from exc

    set_seed(seed)
    tts = TTS(model_name=coqui_model, progress_bar=False, gpu=False)
    out_sr = int(getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 22050))

    segments: List[np.ndarray] = []
    pauses: List[float] = []
    for idx, line in enumerate(lines):
        kwargs = {}
        if coqui_speaker:
            kwargs["speaker"] = coqui_speaker
        if coqui_language:
            kwargs["language"] = coqui_language

        wav = tts.tts(text=line, **kwargs)
        mono = ensure_mono_float32(np.asarray(wav))
        if mono.size == 0:
            raise TTSError("Coqui returned empty audio.")
        segments.append(mono)
        pauses.append(trailing_pause_seconds(line, is_last=(idx == len(lines) - 1)))

    merged = concat_with_pauses(segments, pauses, sample_rate=out_sr)
    validate_generated_speech(merged, out_sr)
    validate_speech_duration(merged, out_sr, " ".join(lines))
    return merged, out_sr


def run_tts(args: argparse.Namespace) -> Path:
    text = load_text(args)
    lines = split_stylized_lines(text)
    if not lines:
        raise TTSError("Input text is empty after normalization.")

    model_choice = args.model.lower()

    def bark_generate() -> Tuple[np.ndarray, int]:
        print("[tts] model=bark")
        return synthesize_with_bark(
            lines=lines,
            seed=args.seed,
            bark_voice=args.bark_voice,
            bark_text_temp=args.bark_text_temp,
            bark_waveform_temp=args.bark_waveform_temp,
            bark_min_eos_p=args.bark_min_eos_p,
        )

    def bark_call() -> Tuple[np.ndarray, int]:
        from gpu_lock import gpu_job
        import gc
        import sys
        with gpu_job():
            try:
                return bark_generate()
            finally:
                generation = sys.modules.get("bark.generation")
                if generation is not None:
                    generation.clean_models()
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def coqui_call() -> Tuple[np.ndarray, int]:
        print("[tts] model=coqui")
        return synthesize_with_coqui(
            lines=lines,
            seed=args.seed,
            coqui_model=args.coqui_model,
            coqui_speaker=args.coqui_speaker,
            coqui_language=args.coqui_language,
        )

    audio: np.ndarray
    sample_rate: int

    if model_choice == "bark":
        audio, sample_rate = bark_call()
    elif model_choice == "coqui":
        audio, sample_rate = coqui_call()
    else:
        # auto: Bark primary, Coqui fallback
        try:
            audio, sample_rate = bark_call()
        except Exception as bark_exc:
            print("[tts] Bark failed, falling back to Coqui")
            print(f"[tts] Bark error: {bark_exc}")
            audio, sample_rate = coqui_call()

    mono = ensure_mono_float32(audio)
    normalized = resample_audio(mono, src_sr=sample_rate, dst_sr=args.target_sr)
    output_path = args.output.expanduser().resolve()
    try:
        write_wav_mono(output_path, normalized, sample_rate=args.target_sr)
    except Exception as exc:
        raise TTSError(f"Failed to write output WAV: {output_path} ({exc})") from exc

    if not output_path.exists() or output_path.stat().st_size <= 44:
        raise TTSError(f"Output file was not written correctly: {output_path}")

    print(f"[tts] saved={output_path} sr={args.target_sr} channels=1")
    return output_path


def main() -> None:
    args = parse_args()
    try:
        run_tts(args)
    except Exception as exc:
        print(f"[tts] error: {exc}")
        print(traceback.format_exc())
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
