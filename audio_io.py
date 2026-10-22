"""Validated audio boundaries. Persist intermediates as 48 kHz PCM24 WAV."""
from pathlib import Path
import os
import tempfile
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

STORAGE_SR = 48000

def validate_audio(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate <= 0 or audio.ndim not in (1, 2) or not audio.size:
        raise ValueError("Audio must be nonempty mono/stereo with a positive sample rate")
    if audio.ndim == 2 and audio.shape[1] not in (1, 2):
        raise ValueError("Only mono and stereo recordings are supported")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinite samples")
    return audio

def resample(audio, source_sr, target_sr):
    audio = validate_audio(audio, source_sr)
    if target_sr <= 0:
        raise ValueError("Target sample rate must be positive")
    if source_sr == target_sr:
        return audio.copy()
    divisor = math.gcd(int(source_sr), int(target_sr))
    return resample_poly(audio, target_sr // divisor, source_sr // divisor, axis=0).astype(np.float32)

def read_audio(path, target_sr=None, mono=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError):
        import shutil, subprocess
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        channels = 1 if mono else 2
        decoded = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
                                  "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(channels),
                                  "-ar", "48000", "pipe:1"], capture_output=True, timeout=600)
        if decoded.returncode:
            raise ValueError(f"Cannot decode {path}: {decoded.stderr.decode(errors='replace')[-1000:]}")
        audio = np.frombuffer(decoded.stdout, dtype='<f4').reshape(-1, channels)
        sr = 48000
    validate_audio(audio, sr)
    audio = audio.mean(axis=1) if mono else audio
    if target_sr is not None:
        audio = resample(audio, sr, target_sr)
        sr = target_sr
    return audio, sr

def write_wav(path, audio, source_sr):
    path = Path(path)
    if path.suffix.lower() != ".wav":
        raise ValueError("Intermediate output must have a .wav extension")
    audio = resample(audio, source_sr, STORAGE_SR)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.99:
        audio *= 0.99 / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix=".wav", dir=str(path.parent))
    os.close(fd)
    try:
        sf.write(temporary, audio, STORAGE_SR, subtype="PCM_24")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

def safe_child(root, name):
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Speaker name must contain only letters, digits, underscore or hyphen")
    if name.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1,10)], *[f"LPT{i}" for i in range(1,10)]}:
        raise ValueError("Reserved Windows filename")
    root = Path(root).resolve()
    child = (root / name).resolve()
    if child.parent != root:
        raise ValueError("Target escapes dataset directory")
    return child
