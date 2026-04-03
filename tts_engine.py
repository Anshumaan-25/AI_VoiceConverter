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


DEFAULT_TARGET_SR = 44100
DEFAULT_BARK_VOICE = "v2/en_speaker_9"
DEFAULT_COQUI_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
DEFAULT_BARK_TEXT_TEMP = 0.2
DEFAULT_BARK_WAVEFORM_TEMP = 0.35
DEFAULT_BARK_MIN_EOS_P = 0.5
DEFAULT_BARK_TRIM_TOP_DB = 38


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
    if not args.text and not args.text_file:
        parser.error("Provide either --text or --text-file")
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
        
        # Safety Net: Limit to 2 CPU threads so fallbacks don't crash the system
        threads = 2
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
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines

    # Keep sentence-level rhythm if no explicit line breaks are present.
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if part.strip()]
    return parts if parts else [" ".join(text.split())]


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
    """Sterile-Room Protocol: strip ALL non-alphanumeric/whitespace characters."""
    line = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    line = re.sub(r'\s+', ' ', line).strip()
    return line


def build_bark_chunks(lines: Sequence[str], max_chars: int = 140) -> List[str]:
    normalized_lines = [clean_text_for_bark(line) for line in lines]
    normalized_lines = [line for line in normalized_lines if line]
    if not normalized_lines:
        return []

    chunks: List[str] = []
    current = normalized_lines[0]
    for line in normalized_lines[1:]:
        candidate = f"{current} {line}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = line
    chunks.append(current)
    return chunks


def trim_audio_edges(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    wav = ensure_mono_float32(audio)
    if wav.size == 0:
        return wav

    try:
        import librosa  # type: ignore

        trimmed, _ = librosa.effects.trim(
            wav,
            top_db=DEFAULT_BARK_TRIM_TOP_DB,
            ref=1.0,
            frame_length=1024,
            hop_length=256,
        )
        if isinstance(trimmed, np.ndarray) and trimmed.size > 0:
            wav = trimmed.astype(np.float32)
    except Exception:
        pass

    amplitude = np.abs(wav)
    if not np.any(amplitude > 0):
        return wav

    threshold = max(float(10.0 ** (-DEFAULT_BARK_TRIM_TOP_DB / 20.0)), 0.010)
    active = np.where(amplitude >= threshold)[0]
    if active.size == 0:
        return wav

    max_trim = int(round(sample_rate * 0.30))
    pad_start = int(round(sample_rate * 0.005))
    pad_end = int(round(sample_rate * 0.010))

    start = max(0, int(active[0]) - pad_start)
    end = min(wav.size, int(active[-1]) + pad_end + 1)

    start = min(start, max_trim)
    end = max(end, wav.size - max_trim)
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
            # This window is silent — mark as candidate crop point
            crop_point = pos
            pos -= window_samples
        else:
            # Hit active audio — stop scanning
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
    if wav.ndim == 1:
        return wav
    if wav.ndim == 2:
        return wav.mean(axis=1).astype(np.float32)
    raise TTSError(f"Unsupported audio shape: {wav.shape}")


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)

    gcd = math.gcd(src_sr, dst_sr)
    up = dst_sr // gcd
    down = src_sr // gcd

    try:
        from scipy.signal import resample_poly  # type: ignore

        return resample_poly(audio, up, down).astype(np.float32)
    except Exception:
        # Deterministic fallback without scipy.
        src_idx = np.arange(audio.shape[0], dtype=np.float32)
        dst_len = max(1, int(round(audio.shape[0] * (dst_sr / src_sr))))
        dst_idx = np.linspace(0, audio.shape[0] - 1, num=dst_len, dtype=np.float32)
        return np.interp(dst_idx, src_idx, audio).astype(np.float32)


def write_wav_mono(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


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
    try:
        from bark import SAMPLE_RATE as bark_sample_rate  # type: ignore
        from bark import preload_models  # type: ignore
        from bark.api import generate_text_semantic, semantic_to_waveform  # type: ignore
    except Exception as exc:
        raise TTSError(
            "Bark is not installed. Install with: pip install bark"
        ) from exc

    preload_models()

    bark_voice = resolve_bark_history_prompt(bark_voice)
    bark_chunks = build_bark_chunks(lines)
    if not bark_chunks:
        raise TTSError("No usable Bark text chunks after normalization.")

    set_seed(seed)

    segments: List[np.ndarray] = []
    pauses: List[float] = []
    for idx, line in enumerate(bark_chunks):
        semantic_tokens = generate_text_semantic(
            line,
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
        mono = trim_audio_edges(mono, sample_rate=int(bark_sample_rate))
        mono = hard_noise_gate(mono, sample_rate=int(bark_sample_rate))
        if mono.size == 0:
            raise TTSError("Bark returned empty audio.")
        segments.append(mono)
        pauses.append(trailing_pause_seconds(line, is_last=(idx == len(bark_chunks) - 1)))

    merged = concat_with_pauses(segments, pauses, sample_rate=int(bark_sample_rate))
    return merged, int(bark_sample_rate)


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
    return merged, out_sr


def run_tts(args: argparse.Namespace) -> Path:
    text = load_text(args)
    lines = split_stylized_lines(text)
    if not lines:
        raise TTSError("Input text is empty after normalization.")

    model_choice = args.model.lower()

    def bark_call() -> Tuple[np.ndarray, int]:
        print("[tts] model=bark")
        return synthesize_with_bark(
            lines=lines,
            seed=args.seed,
            bark_voice=args.bark_voice,
            bark_text_temp=args.bark_text_temp,
            bark_waveform_temp=args.bark_waveform_temp,
            bark_min_eos_p=args.bark_min_eos_p,
        )

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
