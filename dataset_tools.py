"""Prepare bounded clips while preserving recording-level split provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np
from audio_io import read_audio, write_wav, STORAGE_SR

def prepare_clips(files, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    for source in files:
        audio, sr = read_audio(source, STORAGE_SR)
        peak = float(np.max(np.abs(audio)))
        if peak < 1e-5:
            raise ValueError(f"Recording is silent: {source}")
        digest = hashlib.sha256((audio / peak * 32767).astype('<i2').tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        # Split near quiet points; fixed upper bound also handles continuous speech.
        start = 0
        while len(audio) - start >= 2 * sr:
            end = min(start + 8 * sr, len(audio))
            if end < len(audio):
                candidates = range(start + 5 * sr, end, sr // 20)
                end = min(candidates, key=lambda i: float(np.mean(audio[i:i+sr//50] ** 2)))
            clip = audio[start:end]
            accepted = None
            if float(np.sqrt(np.mean(clip ** 2))) >= 0.003:
                name = f"{digest[:16]}_{start:010d}.wav"
                write_wav(destination / name, clip, sr)
                accepted = {"file": name, "recording": digest, "source": Path(source).name,
                            "start_sample": start, "end_sample": end}
                records.append(accepted)
            # Preserve a short final tail by extending the previous clip up to 10 seconds.
            if 0 < len(audio) - end < 2 * sr and accepted is not None:
                tail_clip = audio[start:]
                if len(tail_clip) <= 10 * sr:
                    write_wav(destination / accepted["file"], tail_clip, sr)
                    accepted["end_sample"] = len(audio)
                    end = len(audio)
            start = end
    if not records:
        raise ValueError("No usable clips of at least two seconds")
    (destination / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records

def validate_training_data(svc_dir):
    import soundfile as sf
    svc_dir = Path(svc_dir)
    splits = []
    for split in ("train", "val"):
        listing = svc_dir / "filelists" / (split + ".txt")
        files = [svc_dir / line.strip() for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not files:
            raise ValueError(f"{split} split is empty")
        for path in files:
            info = sf.info(str(path))
            if not 2 <= info.duration <= 10:
                raise ValueError(f"Training clip must be 2–10 seconds: {path} ({info.duration:.2f}s)")
        splits.append(files)
    if len(splits[0]) < 20:
        raise ValueError("Need at least 20 training clips; this is a safety floor, not a quality guarantee")
    # Generated clip names carry their source recording hash.
    groups = [{p.stem.split('_')[0] for p in files} for files in splits]
    if groups[0] & groups[1]:
        raise ValueError("Training and validation share a source recording")
    return splits
