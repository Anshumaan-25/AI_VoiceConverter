"""CLI for isolated, validated speech generation. Models run only in child processes."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from speech.contracts import SpeechError as TTSError

DEFAULT_TARGET_SR = 48000
DEFAULT_BARK_VOICE = "v2/en_speaker_6"
DEFAULT_COQUI_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
DEFAULT_BARK_TEXT_TEMP = 0.7
DEFAULT_BARK_WAVEFORM_TEMP = 0.7
DEFAULT_BARK_MIN_EOS_P = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated speech generation with bounded workers and validated audio")
    parser.add_argument("--text", type=str, default=None, help="Stylized performance text")
    parser.add_argument("--text-file", type=Path, default=None, help="Path to a text file containing stylized text")
    parser.add_argument("--output", type=Path, required=True, help="Output WAV file path")
    parser.add_argument("--model", choices=["auto", "coqui", "sapi"], default="auto", help="Speech backend; auto uses Windows SAPI. Bark is parked indefinitely.")
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

    parser.add_argument("--diagnostics", type=Path, help="Save stage traces, tokens and audio in a unique run directory")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--load-timeout", type=float, default=240)
    parser.add_argument("--chunk-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--max-chunk-seconds", type=float, default=12)
    parser.add_argument("--sapi-voice", help="Installed Windows voice, e.g. Microsoft David Desktop")
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
        with args.text_file.open(encoding="utf-8") as stream:
            text = stream.read(10001)
        if len(text) > 10000:
            raise TTSError("Text file exceeds 10000 characters")
        return text
    return str(args.text)


def run_tts(args: argparse.Namespace) -> Path:
    from speech.contracts import SpeechOptions
    from speech.supervisor import IsolatedSpeechGenerator
    options = SpeechOptions(
        backend=args.model, seed=args.seed, device=args.device,
        bark_voice=args.bark_voice, bark_text_temp=args.bark_text_temp,
        bark_waveform_temp=args.bark_waveform_temp, bark_min_eos_p=args.bark_min_eos_p,
        coqui_model=args.coqui_model, coqui_speaker=args.coqui_speaker,
        coqui_language=args.coqui_language, sapi_voice=args.sapi_voice,
        load_timeout=args.load_timeout, chunk_timeout=args.chunk_timeout,
        request_timeout=args.request_timeout, max_chunk_seconds=args.max_chunk_seconds,
        diagnostics=str(args.diagnostics) if args.diagnostics else None)
    result = IsolatedSpeechGenerator().generate(load_text(args), args.output, options)
    print(f"[tts] saved={result.path} backend={result.backend} sr=48000 channels=1")
    if result.diagnostics:
        print(f"[tts] diagnostics={result.diagnostics}")
    return result.path


def main() -> None:
    args = parse_args()
    try:
        run_tts(args)
    except Exception as exc:
        print(f"[tts] error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
