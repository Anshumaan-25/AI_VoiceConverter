from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from audio_io import safe_child, read_audio, write_wav
from gpu_lock import gpu_job
from pathlib import Path
from typing import Iterable, List


ROOT_DIR = Path(__file__).resolve().parent
SVC_DIR = ROOT_DIR / "so-vits-svc"
DEFAULT_SOURCE_DIR = ROOT_DIR / "so-vits-dataset" / "raw"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
WORD_RE = re.compile(r"[A-Za-z0-9']+")

FILLER_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "them",
    "they",
    "to",
    "we",
    "with",
    "you",
    "your",
}

WEAK_CONNECTORS = {
    "and",
    "through",
    "with",
    "my",
    "the",
    "a",
    "an",
    "in",
    "of",
    "to",
    "for",
    "on",
}

AUXILIARY_WORDS = {
    "am",
    "are",
    "be",
    "been",
    "being",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "was",
    "were",
    "will",
    "would",
}

ACTION_WORDS = {
    "break",
    "build",
    "fight",
    "grind",
    "grow",
    "lead",
    "move",
    "push",
    "rise",
    "run",
    "shift",
    "stand",
    "work",
}

EMOTION_WORDS = {
    "fire",
    "focus",
    "heart",
    "pain",
    "power",
    "pressure",
    "strength",
    "storm",
    "truth",
}

NATURAL_PAUSE_STARTERS = {
    "but",
    "then",
    "when",
    "while",
    "now",
    "still",
    "because",
}

PROFANITY_WORDS = {
    "ass",
    "bastard",
    "bitch",
    "damn",
    "fuck",
    "fucking",
    "hell",
    "shit",
}

IMPACT_WORDS = {
    "break",
    "build",
    "burn",
    "change",
    "create",
    "dream",
    "fight",
    "fire",
    "focus",
    "grind",
    "grow",
    "heart",
    "hustle",
    "lead",
    "level",
    "move",
    "power",
    "push",
    "rise",
    "run",
    "shift",
    "shine",
    "stand",
    "storm",
    "strength",
    "truth",
    "win",
    "work",
}

INTENSITY_PROFILES = {
    "low": {"split_threshold": 12, "terminal": ".", "emphasis": 1, "allow_strong_pause": False},
    "medium": {"split_threshold": 10, "terminal": ".", "emphasis": 2, "allow_strong_pause": False},
    "high": {"split_threshold": 9, "terminal": "!", "emphasis": 2, "allow_strong_pause": True},
}


def _python_bin() -> str:
    return sys.executable


def _ensure_paths() -> None:
    if not SVC_DIR.exists():
        raise FileNotFoundError(f"Missing so-vits repository at: {SVC_DIR}")


def run_cmd(args: List[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("[cmd]", " ".join(args))
    merged_env = None
    if env:
        merged_env = dict(os.environ)
        merged_env.update(env)
    process = subprocess.Popen(args, cwd=cwd, env=merged_env)
    try:
        code = process.wait(timeout=86400)
    except BaseException:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
        else:
            process.kill()
        process.wait()
        raise
    result = subprocess.CompletedProcess(args, code)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(args)}")


def resolve_ffmpeg_bin() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pylint: disable=broad-except
        return None


def ffmpeg_exists() -> bool:
    return resolve_ffmpeg_bin() is not None


def _split_sentences(text: str) -> List[str]:
    parts = [part.strip() for part in re.split(r"[.!?;:]+", text) if part.strip()]
    return parts


def _word_score(word: str, next_word: str | None = None, reinforced_words: set[str] | None = None) -> int:
    w = word.lower()
    if not w or w in FILLER_WORDS or w in AUXILIARY_WORDS:
        return -100

    score = 0
    if w in ACTION_WORDS:
        score += 5
    if w in EMOTION_WORDS:
        score += 5
    if w in IMPACT_WORDS:
        score += 2
    if w in PROFANITY_WORDS:
        score += 4
    if len(w) >= 5:
        score += 1
    if reinforced_words and w in reinforced_words:
        score += 2

    if next_word and w in ACTION_WORDS and next_word.lower() in WEAK_CONNECTORS:
        score -= 4

    if w.endswith(("ing", "ed", "ize", "ise", "ify", "tion", "ment", "ness", "ity", "ship")):
        score += 2
    elif w.endswith(("er", "or", "s")):
        score += 1
    return score


def _collect_reinforced_words(sentences: List[str]) -> set[str]:
    counts: dict[str, int] = {}
    for sentence in sentences:
        for word in WORD_RE.findall(sentence):
            w = word.lower()
            if w in ACTION_WORDS or w in EMOTION_WORDS or w in IMPACT_WORDS:
                counts[w] = counts.get(w, 0) + 1
    return {w for w, count in counts.items() if count > 1}


def _split_index_score(words: List[str], split_idx: int) -> tuple[int, int]:
    left = words[split_idx - 1].lower()
    right = words[split_idx].lower()
    left_len = split_idx
    right_len = len(words) - split_idx

    score = -abs(left_len - right_len)
    if left in WEAK_CONNECTORS or left in AUXILIARY_WORDS:
        score -= 6
    if right in WEAK_CONNECTORS or right in AUXILIARY_WORDS:
        score -= 6
    if left in ACTION_WORDS or left in EMOTION_WORDS or left in IMPACT_WORDS:
        score += 3
    if right in {"but", "then", "still", "now"}:
        score += 1

    return (score, -abs(split_idx - (len(words) // 2)))


def _group_sentence_bars(words: List[str], split_threshold: int) -> List[List[str]]:
    if len(words) < split_threshold:
        return [words]

    candidate_indices = [idx for idx in range(4, len(words) - 3)]
    if not candidate_indices:
        mid = len(words) // 2
        return [words[:mid], words[mid:]]

    best_idx = max(candidate_indices, key=lambda idx: _split_index_score(words, idx))
    left = words[best_idx - 1].lower()
    right = words[best_idx].lower()
    if left in (WEAK_CONNECTORS | AUXILIARY_WORDS) or right in (WEAK_CONNECTORS | AUXILIARY_WORDS):
        return [words]

    return [words[:best_idx], words[best_idx:]]


def _pause_mark(intensity: str, last_word: str, next_first_word: str) -> str:
    profile = INTENSITY_PROFILES[intensity]
    lw = last_word.lower()
    nw = next_first_word.lower()
    if profile["allow_strong_pause"] and lw in (ACTION_WORDS | EMOTION_WORDS) and nw in NATURAL_PAUSE_STARTERS:
        return "..."
    if nw in NATURAL_PAUSE_STARTERS:
        return ","
    return ""


def _detect_user_structure(text: str) -> dict[str, bool]:
    has_line_breaks = "\n" in text
    has_punctuation = bool(re.search(r"[,;:!?]", text))
    words = WORD_RE.findall(text)
    has_caps = any(word.isupper() and len(word) > 1 for word in words)
    return {
        "has_line_breaks": has_line_breaks,
        "has_punctuation": has_punctuation,
        "has_caps": has_caps,
        "structured": has_line_breaks or has_punctuation or has_caps,
    }


def _safe_cut(words: List[str], preferred_cut: int) -> int:
    n_words = len(words)
    candidates = [preferred_cut, preferred_cut + 1, preferred_cut - 1, preferred_cut + 2, preferred_cut - 2]
    for cut in candidates:
        if not (2 <= cut <= n_words - 2):
            continue
        left = words[cut - 1].lower()
        right = words[cut].lower()
        if left in (WEAK_CONNECTORS | AUXILIARY_WORDS):
            continue
        if right in (WEAK_CONNECTORS | AUXILIARY_WORDS):
            continue
        return cut

    return max(2, min(n_words - 2, preferred_cut))


def _cadence_segments(words: List[str], cadence_variant: int) -> List[List[str]]:
    n_words = len(words)
    if n_words <= 5:
        return [words]

    if n_words >= 9:
        first_pref = 2 if cadence_variant == 0 else 3
        second_pref = 5 if cadence_variant == 0 else 6
        cut_1 = _safe_cut(words, first_pref)
        cut_2 = _safe_cut(words, second_pref)
        if cut_2 <= cut_1 + 1:
            cut_2 = min(n_words - 2, cut_1 + 2)
        if cut_2 <= cut_1:
            return [words[:cut_1], words[cut_1:]]
        return [words[:cut_1], words[cut_1:cut_2], words[cut_2:]]

    cut_1 = _safe_cut(words, 2 if cadence_variant == 0 else 3)
    return [words[:cut_1], words[cut_1:]]


def _format_bar_with_cadence(words: List[str], cadence_variant: int) -> str:
    segments = _cadence_segments(words, cadence_variant)
    if len(segments) <= 1:
        return " ".join(words)

    # Keep commas rare: only one light pause inside long bars when boundary sounds natural.
    if len(words) < 8:
        return " ".join(words)

    split_positions: List[int] = []
    cursor = 0
    for segment in segments[:-1]:
        cursor += len(segment)
        split_positions.append(cursor)

    selected_split: int | None = None
    for split in split_positions:
        left = words[split - 1].lower()
        right = words[split].lower()
        if left in (WEAK_CONNECTORS | AUXILIARY_WORDS):
            continue
        if right in (WEAK_CONNECTORS | AUXILIARY_WORDS):
            continue
        if right in NATURAL_PAUSE_STARTERS:
            selected_split = split
            break

    if selected_split is None:
        return " ".join(words)

    left_text = " ".join(words[:selected_split])
    right_text = " ".join(words[selected_split:])
    return f"{left_text}, {right_text}"


def _pick_emphasis_indices(
    words: List[str],
    intensity: str,
    reinforced_words: set[str],
    light: bool = False,
) -> set[int]:
    if not words:
        return set()

    emphasis_count = int(INTENSITY_PROFILES[intensity]["emphasis"])
    if light:
        emphasis_count = 1

    scored: List[tuple[int, int]] = []
    for idx, word in enumerate(words):
        next_word = words[idx + 1] if idx + 1 < len(words) else None
        score = _word_score(word, next_word=next_word, reinforced_words=reinforced_words)
        if score <= 0:
            continue
        scored.append((score, idx))

    if not scored:
        return set()

    n_words = len(words)
    selected: List[int] = []
    profanity_indices = {idx for _, idx in scored if words[idx].lower() in PROFANITY_WORDS}
    max_profanity = 2 if intensity == "high" and not light else 1

    def _can_pick(idx: int) -> bool:
        if idx not in profanity_indices:
            return True
        current_profanity = sum(1 for item in selected if item in profanity_indices)
        return current_profanity < max_profanity

    def _pick_best(candidates: List[tuple[int, int]]) -> int | None:
        for _, idx in sorted(candidates, reverse=True):
            if _can_pick(idx) and idx not in selected:
                return idx
        return None

    if emphasis_count >= 2 and n_words >= 4:
        mid_start = max(1, n_words // 3)
        mid_end = min(n_words - 1, (2 * n_words) // 3 + 1)
        mid_candidates = [
            (score - abs(idx - (n_words // 2)), idx)
            for score, idx in scored
            if mid_start <= idx < mid_end
        ]

        final_score = _word_score(words[-1], reinforced_words=reinforced_words)
        final_is_weak = final_score <= 0 or words[-1].lower() in (WEAK_CONNECTORS | AUXILIARY_WORDS)
        end_candidates = [
            (score + (3 if idx == n_words - 1 else 2 if idx >= n_words - 2 else 1), idx)
            for score, idx in scored
            if idx >= max(0, n_words - 3) and (not final_is_weak or idx < n_words - 1)
        ]

        mid_idx = _pick_best(mid_candidates)
        if mid_idx is not None:
            selected.append(mid_idx)
        end_idx = _pick_best(end_candidates)
        if end_idx is not None and end_idx not in selected:
            selected.append(end_idx)

    ranked = sorted(
        ((score + (1 if idx >= n_words - 2 else 0), idx) for score, idx in scored),
        reverse=True,
    )
    for _, idx in ranked:
        if idx not in selected and _can_pick(idx):
            selected.append(idx)
        if len(selected) >= emphasis_count:
            break

    return set(selected)


def _apply_indices_to_words(words: List[str], selected_set: set[int]) -> List[str]:
    output: List[str] = []
    for idx, word in enumerate(words):
        output.append(word.upper() if idx in selected_set else word)
    return output


def _apply_indices_to_text(line: str, selected_set: set[int]) -> str:
    if not selected_set:
        return line

    pieces: List[str] = []
    last_end = 0
    word_idx = 0
    for match in WORD_RE.finditer(line):
        pieces.append(line[last_end : match.start()])
        word = match.group(0)
        pieces.append(word.upper() if word_idx in selected_set else word)
        last_end = match.end()
        word_idx += 1
    pieces.append(line[last_end:])
    return "".join(pieces)


def _enhance_preserving_structure(text: str, intensity: str, reinforced_words: set[str], light: bool) -> str:
    lines = text.splitlines(keepends=True)
    if not lines:
        return text

    output_lines: List[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        suffix = raw_line[len(line) :]
        if not line.strip():
            output_lines.append(raw_line)
            continue

        words = WORD_RE.findall(line)
        selected = _pick_emphasis_indices(words, intensity=intensity, reinforced_words=reinforced_words, light=light)
        output_lines.append(_apply_indices_to_text(line, selected) + suffix)

    return "".join(output_lines)


def stylize_performance_text(text: str, intensity: str = "medium", mode: str = "FULL") -> str:
    if not text.strip():
        return ""

    mode_upper = mode.upper()
    if mode_upper == "RAW":
        return text

    structure = _detect_user_structure(text)
    clean = " ".join(text.strip().split())

    profile = INTENSITY_PROFILES[intensity]
    split_threshold = int(profile["split_threshold"])
    terminal_mark = str(profile["terminal"])

    lines: List[str] = []
    sentences = _split_sentences(clean)
    reinforced_words = _collect_reinforced_words(sentences)

    if mode_upper == "ASSIST":
        return _enhance_preserving_structure(text, intensity=intensity, reinforced_words=reinforced_words, light=True)

    if mode_upper == "FULL" and structure["structured"]:
        return _enhance_preserving_structure(text, intensity=intensity, reinforced_words=reinforced_words, light=False)

    for sentence_idx, sentence in enumerate(sentences):
        words = WORD_RE.findall(sentence)
        if not words:
            continue

        bars = _group_sentence_bars(words, split_threshold=split_threshold)
        for bar_idx, bar_words in enumerate(bars):
            selected = _pick_emphasis_indices(bar_words, intensity=intensity, reinforced_words=reinforced_words)
            emphasized = _apply_indices_to_words(bar_words, selected)
            cadence_variant = (sentence_idx + bar_idx) % 2
            line = _format_bar_with_cadence(emphasized, cadence_variant=cadence_variant)
            if bar_idx < len(bars) - 1:
                next_first_word = bars[bar_idx + 1][0]
                line = f"{line}{_pause_mark(intensity, emphasized[-1], next_first_word)}"
            else:
                line = f"{line}{terminal_mark}"
            lines.append(line)

    return "\n".join(lines)


def list_audio_files(source_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            files.append(path)
    return sorted(files)


def convert_to_wav_44k_mono(src: Path, dst: Path) -> None:
    audio, sr = read_audio(src, target_sr=48000)
    write_wav(dst, audio, sr)


def list_speakers_from_config(config_path: Path) -> List[str]:
    if not config_path.is_file():
        return []
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spk = config.get("spk", {})
    if isinstance(spk, dict):
        return sorted(str(k) for k in spk.keys())
    return []


def prepare_dataset(source_dir: Path, speaker: str, clean_target: bool = False) -> Path:
    _ensure_paths()
    if not ffmpeg_exists():
        raise EnvironmentError("ffmpeg is not available in PATH. Install ffmpeg first.")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset folder not found: {source_dir}")

    files = list_audio_files(source_dir)
    if not files:
        raise RuntimeError(f"No audio files found in: {source_dir}")

    speaker_dir = safe_child(SVC_DIR / "dataset_raw", speaker)
    if source_dir.resolve() == speaker_dir or speaker_dir in source_dir.resolve().parents or source_dir.resolve() in speaker_dir.parents:
        raise ValueError("Source cannot be inside the target directory")
    if speaker_dir.exists() and any(speaker_dir.iterdir()) and not clean_target:
        raise ValueError("Target contains existing data; use a new speaker or --clean-target")
    # Keep old data recoverable, including if preparation fails.
    if speaker_dir.exists() and clean_target:
        backup_root = SVC_DIR / "dataset_backups"
        backup_root.mkdir(exist_ok=True)
        speaker_dir.rename(backup_root / (speaker_dir.name + "-" + uuid.uuid4().hex))
    from dataset_tools import prepare_clips
    records = prepare_clips(files, speaker_dir)
    print(f"Prepared {len(records)} clips in {speaker_dir}")
    return speaker_dir


def run_preprocess(speech_encoder: str, f0_predictor: str, skip_loudnorm: bool) -> None:
    _ensure_paths()
    py = _python_bin()

    build_dir = SVC_DIR / "dataset" / ("prepared-" + uuid.uuid4().hex)
    resample_cmd = [py, "resample.py", "--out_dir2", str(build_dir)]
    if skip_loudnorm:
        resample_cmd.append("--skip_loudnorm")

    run_cmd(resample_cmd, cwd=SVC_DIR)
    target = SVC_DIR / "dataset" / "44k"
    if target.exists():
        backup_root = SVC_DIR / "dataset_backups"
        backup_root.mkdir(exist_ok=True)
        target.rename(backup_root / ("prepared-" + uuid.uuid4().hex))
    build_dir.rename(target)
    run_cmd([py, "preprocess_flist_config.py", "--speech_encoder", speech_encoder], cwd=SVC_DIR)

    from dataset_tools import validate_training_data
    validate_training_data(SVC_DIR)
    with gpu_job():
        run_cmd([py, "preprocess_hubert_f0.py", "--f0_predictor", f0_predictor, "--num_processes", "1"], cwd=SVC_DIR)


def run_train(model_name: str = "44k") -> None:
    safe_child(SVC_DIR / "logs", model_name)
    _ensure_paths()
    py = _python_bin()
    from dataset_tools import validate_training_data
    validate_training_data(SVC_DIR)
    # Windows + recent torch builds can fail with libuv-enabled store initialization.
    with gpu_job():
        run_cmd([py, "train.py", "-c", "configs/config.json", "-m", model_name],
                cwd=SVC_DIR, env={"USE_LIBUV": "0"})



def infer_once(
    input_audio: Path,
    model_path: Path,
    config_path: Path,
    speaker: str,
    transpose: int = 0,
    f0_predictor: str = "rmvpe",
    wav_format: str = "wav",
) -> Path:
    _ensure_paths()
    if not ffmpeg_exists():
        raise EnvironmentError("ffmpeg is not available in PATH. Install ffmpeg first.")

    input_audio = input_audio.resolve()
    model_path = model_path.resolve()
    config_path = config_path.resolve()

    if not input_audio.is_file():
        raise FileNotFoundError(f"Input audio not found: {input_audio}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_dir = SVC_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if speaker not in list_speakers_from_config(config_path):
        raise ValueError("Speaker is not present in the selected model configuration")
    if model_path.stem == "G_0":
        raise ValueError("G_0 is an unqualified initial checkpoint; select a validated trained model")
    if wav_format != "wav":
        raise ValueError("Inference output must be WAV; use export_mix.py for final MP3")
    raw_name = f"input_{uuid.uuid4().hex}.wav"
    raw_file = raw_dir / raw_name
    try:
        convert_to_wav_44k_mono(input_audio, raw_file)
        with gpu_job():
            run_cmd([_python_bin(), "inference_main.py", "-m", str(model_path), "-c", str(config_path),
                     "-n", raw_name, "-t", str(transpose), "-s", speaker, "-f0p", f0_predictor,
                     "-wf", "wav", "-cl", "5"], cwd=SVC_DIR)
        candidates = list((SVC_DIR / "results").glob(f"{raw_name}_*"))
        if len(candidates) != 1:
            raise RuntimeError("Inference did not produce exactly one output")
        read_audio(candidates[0])
        return candidates[0]
    finally:
        raw_file.unlink(missing_ok=True)


def check_env() -> None:
    from health_check import check
    check()


def run_performance_engine(text: str, intensity: str, mode: str) -> None:
    print(stylize_performance_text(text=text, intensity=intensity, mode=mode))


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autopilot wrappers for so-vits-svc workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Check runtime environment")
    p_check.set_defaults(func=lambda args: check_env())

    p_prep = sub.add_parser("prepare-data", help="Convert source audio into dataset_raw/<speaker>")
    p_prep.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DIR)
    p_prep.add_argument("--speaker", type=str, required=True)
    p_prep.add_argument("--clean-target", action="store_true")
    p_prep.set_defaults(
        func=lambda args: prepare_dataset(args.source, args.speaker, clean_target=args.clean_target)
    )

    p_pre = sub.add_parser("preprocess", help="Run so-vits preprocessing pipeline")
    p_pre.add_argument("--speech-encoder", type=str, default="vec768l12")
    p_pre.add_argument("--f0-predictor", type=str, default="rmvpe")
    p_pre.add_argument("--skip-loudnorm", action="store_true")
    p_pre.set_defaults(
        func=lambda args: run_preprocess(args.speech_encoder, args.f0_predictor, args.skip_loudnorm)
    )

    p_train = sub.add_parser("train", help="Start model training")
    p_train.add_argument("--model-name", type=str, default="44k")
    p_train.set_defaults(func=lambda args: run_train(args.model_name))

    p_infer = sub.add_parser("infer", help="Run one inference conversion")
    p_infer.add_argument("--input", type=Path, required=True)
    p_infer.add_argument("--model", type=Path, required=True)
    p_infer.add_argument("--config", type=Path, required=True)
    p_infer.add_argument("--speaker", type=str, required=True)
    p_infer.add_argument("--transpose", type=int, default=0)
    p_infer.add_argument("--f0-predictor", type=str, default="rmvpe")
    p_infer.add_argument("--wav-format", type=str, default="wav")
    p_infer.set_defaults(
        func=lambda args: print(
            infer_once(
                input_audio=args.input,
                model_path=args.model,
                config_path=args.config,
                speaker=args.speaker,
                transpose=args.transpose,
                f0_predictor=args.f0_predictor,
                wav_format=args.wav_format,
            )
        )
    )

    p_perf = sub.add_parser("performance", help="Phase 1A: stylize text into performance-ready phrasing")
    p_perf.add_argument("--text", type=str, required=True)
    p_perf.add_argument("--intensity", type=str, choices=["low", "medium", "high"], default="medium")
    p_perf.add_argument("--mode", type=str, choices=["FULL", "ASSIST", "RAW"], default="FULL")
    p_perf.set_defaults(func=lambda args: run_performance_engine(args.text, args.intensity, args.mode))

    return parser


def main() -> None:
    parser = _arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
