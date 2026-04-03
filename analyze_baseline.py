"""Analyze baseline test.wav for beeping/hallucination artifacts."""
import wave
import struct
import sys
import numpy as np

WAV_PATH = sys.argv[1] if len(sys.argv) > 1 else "test.wav"

# --- Read WAV ---
with wave.open(WAV_PATH, "rb") as wf:
    n_channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    sample_rate = wf.getframerate()
    n_frames = wf.getnframes()
    raw = wf.readframes(n_frames)

duration = n_frames / sample_rate
print(f"=== BASELINE WAV ANALYSIS ===")
print(f"File: {WAV_PATH}")
print(f"Channels: {n_channels}, Sample width: {sample_width} bytes, Sample rate: {sample_rate} Hz")
print(f"Frames: {n_frames}, Duration: {duration:.3f}s")
print(f"File size: {len(raw) + 44} bytes")

# Convert to float32 [-1, 1]
if sample_width == 2:
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
elif sample_width == 4:
    samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
else:
    raise ValueError(f"Unsupported sample width: {sample_width}")

if n_channels > 1:
    samples = samples.reshape(-1, n_channels).mean(axis=1)

# --- Global stats ---
peak = np.max(np.abs(samples))
rms = np.sqrt(np.mean(samples ** 2))
rms_db = 20 * np.log10(rms + 1e-10)
peak_db = 20 * np.log10(peak + 1e-10)
print(f"\nPeak amplitude: {peak:.4f} ({peak_db:.1f} dB)")
print(f"RMS amplitude:  {rms:.4f} ({rms_db:.1f} dB)")

# --- Segment analysis (1-second windows) ---
print(f"\n--- Per-second RMS and Peak ---")
window_sec = 1.0
window_samples = int(sample_rate * window_sec)
for i in range(0, len(samples), window_samples):
    chunk = samples[i:i + window_samples]
    if len(chunk) < window_samples // 4:
        break
    t_start = i / sample_rate
    t_end = (i + len(chunk)) / sample_rate
    c_rms = np.sqrt(np.mean(chunk ** 2))
    c_peak = np.max(np.abs(chunk))
    c_rms_db = 20 * np.log10(c_rms + 1e-10)
    c_peak_db = 20 * np.log10(c_peak + 1e-10)
    print(f"  [{t_start:5.1f}s - {t_end:5.1f}s]  RMS={c_rms:.4f} ({c_rms_db:+.1f} dB)  Peak={c_peak:.4f} ({c_peak_db:+.1f} dB)")

# --- Detect tonal/beeping artifacts via spectral analysis ---
print(f"\n--- Tonal Artifact Detection (per 0.5s window) ---")
from numpy.fft import rfft, rfftfreq

window_sec_fft = 0.5
window_fft = int(sample_rate * window_sec_fft)
found_tonal = False

for i in range(0, len(samples) - window_fft, window_fft):
    chunk = samples[i:i + window_fft]
    t_start = i / sample_rate
    t_end = (i + window_fft) / sample_rate

    # Apply Hann window
    windowed = chunk * np.hanning(len(chunk))
    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(len(chunk), 1.0 / sample_rate)

    # Normalize spectrum
    spectrum_db = 20 * np.log10(spectrum / (np.max(spectrum) + 1e-10) + 1e-10)

    # Find dominant peaks (above -20dB relative to max, excluding DC)
    mask = (freqs > 50) & (spectrum_db > -20)
    if np.any(mask):
        peak_freqs = freqs[mask]
        peak_mags = spectrum_db[mask]
        # Sort by magnitude
        sort_idx = np.argsort(peak_mags)[::-1]
        top_freqs = peak_freqs[sort_idx[:5]]
        top_mags = peak_mags[sort_idx[:5]]

        # Check for narrow tonal peaks (potential beeping)
        # A beep would show as a very narrow, dominant spectral peak
        max_mag_idx = np.argmax(spectrum[freqs > 50]) + np.searchsorted(freqs, 50)
        dominant_freq = freqs[max_mag_idx]
        dominant_mag = spectrum[max_mag_idx]
        total_energy = np.sum(spectrum[freqs > 50] ** 2)
        peak_energy = np.sum(spectrum[max(0, max_mag_idx - 3):max_mag_idx + 4] ** 2)
        concentration = peak_energy / (total_energy + 1e-10)

        if concentration > 0.3:  # More than 30% energy in a narrow band = likely tonal
            found_tonal = True
            print(f"  ⚠ [{t_start:5.1f}s - {t_end:5.1f}s]  TONAL PEAK @ {dominant_freq:.0f} Hz  "
                  f"(energy concentration: {concentration:.1%})")

if not found_tonal:
    print("  No strong tonal artifacts detected in spectral analysis.")

# --- Detect trailing silence / noise floor ---
print(f"\n--- Trailing Region Analysis (last 2 seconds) ---")
tail_samples = min(int(2.0 * sample_rate), len(samples))
tail = samples[-tail_samples:]
tail_rms = np.sqrt(np.mean(tail ** 2))
tail_rms_db = 20 * np.log10(tail_rms + 1e-10)
tail_peak = np.max(np.abs(tail))

# Check for sustained silence below -40dB
silence_threshold = 10.0 ** (-40.0 / 20.0)  # -40 dB
silence_window = int(0.1 * sample_rate)  # 100ms windows
silent_regions = []
for j in range(0, len(tail) - silence_window, silence_window // 2):
    window = tail[j:j + silence_window]
    w_rms = np.sqrt(np.mean(window ** 2))
    if w_rms < silence_threshold:
        t = (len(samples) - tail_samples + j) / sample_rate
        silent_regions.append(t)

print(f"  Tail RMS: {tail_rms:.4f} ({tail_rms_db:+.1f} dB)")
print(f"  Tail Peak: {tail_peak:.4f}")
if silent_regions:
    print(f"  Silent regions (<-40dB, 100ms windows): {len(silent_regions)} found")
    print(f"  First at: {silent_regions[0]:.2f}s, Last at: {silent_regions[-1]:.2f}s")
else:
    print(f"  No sustained silence regions detected in tail.")

# --- Zero-crossing rate anomaly detection ---
print(f"\n--- Zero-Crossing Rate Analysis ---")
zcr_window = int(0.25 * sample_rate)
zcrs = []
for i in range(0, len(samples) - zcr_window, zcr_window):
    chunk = samples[i:i + zcr_window]
    zc = np.sum(np.abs(np.diff(np.sign(chunk))) > 0) / len(chunk)
    zcrs.append((i / sample_rate, zc))

if zcrs:
    zcr_values = [z[1] for z in zcrs]
    mean_zcr = np.mean(zcr_values)
    std_zcr = np.std(zcr_values)
    print(f"  Mean ZCR: {mean_zcr:.4f}, Std: {std_zcr:.4f}")
    anomalies = [(t, z) for t, z in zcrs if abs(z - mean_zcr) > 2 * std_zcr]
    if anomalies:
        print(f"  ZCR anomalies (>2σ from mean):")
        for t, z in anomalies:
            print(f"    @ {t:.2f}s: ZCR={z:.4f} ({'HIGH' if z > mean_zcr else 'LOW'})")
    else:
        print(f"  No ZCR anomalies detected.")

print(f"\n=== ANALYSIS COMPLETE ===")
