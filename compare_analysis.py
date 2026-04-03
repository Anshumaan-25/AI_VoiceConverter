"""Compare baseline vs Sterile-Room test.wav waveforms."""
import wave
import numpy as np
from numpy.fft import rfft, rfftfreq

def analyze_wav(path, label):
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    duration = n_frames / sample_rate
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    peak = np.max(np.abs(samples))
    rms = np.sqrt(np.mean(samples ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)

    # Tonal detection
    tonal_peaks = []
    window_fft = int(sample_rate * 0.5)
    for i in range(0, len(samples) - window_fft, window_fft):
        chunk = samples[i:i + window_fft]
        t_start = i / sample_rate
        windowed = chunk * np.hanning(len(chunk))
        spectrum = np.abs(rfft(windowed))
        freqs = rfftfreq(len(chunk), 1.0 / sample_rate)
        max_mag_idx = np.argmax(spectrum[freqs > 50]) + np.searchsorted(freqs, 50)
        dominant_freq = freqs[max_mag_idx]
        total_energy = np.sum(spectrum[freqs > 50] ** 2)
        peak_energy = np.sum(spectrum[max(0, max_mag_idx - 3):max_mag_idx + 4] ** 2)
        concentration = peak_energy / (total_energy + 1e-10)
        if concentration > 0.3:
            tonal_peaks.append((t_start, dominant_freq, concentration))

    # Trailing silence
    silence_threshold = 10.0 ** (-40.0 / 20.0)
    tail_samples = min(int(2.0 * sample_rate), len(samples))
    tail = samples[-tail_samples:]
    silence_window = int(0.1 * sample_rate)
    silent_count = 0
    for j in range(0, len(tail) - silence_window, silence_window // 2):
        w = tail[j:j + silence_window]
        if np.sqrt(np.mean(w ** 2)) < silence_threshold:
            silent_count += 1

    # ZCR anomalies
    zcr_window = int(0.25 * sample_rate)
    zcrs = []
    for i in range(0, len(samples) - zcr_window, zcr_window):
        chunk = samples[i:i + zcr_window]
        zc = np.sum(np.abs(np.diff(np.sign(chunk))) > 0) / len(chunk)
        zcrs.append(zc)
    mean_zcr = np.mean(zcrs) if zcrs else 0
    std_zcr = np.std(zcrs) if zcrs else 0
    zcr_anomaly_count = sum(1 for z in zcrs if abs(z - mean_zcr) > 2 * std_zcr)

    return {
        "label": label,
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "rms_db": rms_db,
        "tonal_peaks": tonal_peaks,
        "silent_regions": silent_count,
        "zcr_anomalies": zcr_anomaly_count,
        "file_size": len(raw) + 44,
    }

# Run analysis
baseline = analyze_wav("test_baseline_pre_sterile.wav", "BASELINE (pre-fix)")
sterile = analyze_wav("test.wav", "STERILE-ROOM (post-fix)")

print("=" * 72)
print("STERILE-ROOM PROTOCOL: BEFORE vs AFTER COMPARISON")
print("=" * 72)

fmt = "{:<30} {:>18} {:>18}"
print(fmt.format("Metric", "BASELINE", "STERILE-ROOM"))
print("-" * 72)
print(fmt.format("Duration (s)", f"{baseline['duration']:.3f}", f"{sterile['duration']:.3f}"))
print(fmt.format("File size (bytes)", str(baseline['file_size']), str(sterile['file_size'])))
print(fmt.format("Peak amplitude", f"{baseline['peak']:.4f}", f"{sterile['peak']:.4f}"))
print(fmt.format("RMS (dB)", f"{baseline['rms_db']:.1f}", f"{sterile['rms_db']:.1f}"))
print(fmt.format("Tonal peaks (>30%)", str(len(baseline['tonal_peaks'])), str(len(sterile['tonal_peaks']))))
print(fmt.format("Silent regions (<-40dB)", str(baseline['silent_regions']), str(sterile['silent_regions'])))
print(fmt.format("ZCR anomalies", str(baseline['zcr_anomalies']), str(sterile['zcr_anomalies'])))

print("\n--- BASELINE Tonal Peaks ---")
if baseline['tonal_peaks']:
    for t, f, c in baseline['tonal_peaks']:
        print(f"  @ {t:.1f}s: {f:.0f} Hz (concentration: {c:.1%})")
else:
    print("  None detected")

print("\n--- STERILE-ROOM Tonal Peaks ---")
if sterile['tonal_peaks']:
    for t, f, c in sterile['tonal_peaks']:
        print(f"  @ {t:.1f}s: {f:.0f} Hz (concentration: {c:.1%})")
else:
    print("  None detected")

# Verdict
print("\n" + "=" * 72)
dur_improved = sterile['duration'] < baseline['duration'] * 0.7
tonal_improved = len(sterile['tonal_peaks']) < len(baseline['tonal_peaks'])
silence_improved = sterile['silent_regions'] < baseline['silent_regions']

if dur_improved and tonal_improved:
    print("VERDICT: SIGNIFICANT IMPROVEMENT")
elif tonal_improved or dur_improved:
    print("VERDICT: PARTIAL IMPROVEMENT")
else:
    print("VERDICT: NO IMPROVEMENT (further investigation needed)")

print(f"  Duration reduction: {baseline['duration']:.2f}s -> {sterile['duration']:.2f}s " +
      f"({(1 - sterile['duration']/baseline['duration'])*100:.0f}% shorter)")
print(f"  Tonal artifacts:    {len(baseline['tonal_peaks'])} -> {len(sterile['tonal_peaks'])}")
print(f"  Trailing silence:   {baseline['silent_regions']} -> {sterile['silent_regions']} regions")
print("=" * 72)
