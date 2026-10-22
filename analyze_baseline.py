"""Describe a supplied recording independently of its PCM bit depth."""
import argparse
import json
import numpy as np
from audio_io import read_audio

def analyze_wav(path):
    audio, sr = read_audio(path)
    size = max(1, int(sr * .5))
    presence = total = 0.0
    tonal_windows = []
    for start in range(0, len(audio) - size + 1, size):
        spectrum = np.abs(np.fft.rfft(audio[start:start+size] * np.hanning(size))) ** 2
        frequency = np.fft.rfftfreq(size, 1/sr)
        total += float(spectrum.sum())
        presence += float(spectrum[(frequency >= 4000) & (frequency <= 8000)].sum())
        concentration = float(spectrum.max() / max(spectrum.sum(), 1e-20))
        if concentration > .3:
            tonal_windows.append({"seconds": start/sr, "concentration": concentration})
    return {"presence_4_8khz_fraction": presence/max(total, 1e-20), "tonal_windows": tonal_windows, "path": str(path), "sample_rate": sr, "duration": len(audio) / sr,
            "peak": float(np.max(np.abs(audio))), "rms": float(np.sqrt(np.mean(audio ** 2))),
            "clipped_samples": int(np.count_nonzero(np.abs(audio) >= 0.9999))}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    args = parser.parse_args()
    try:
        print(json.dumps(analyze_wav(args.audio), indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Audio analysis failed: {exc}")
