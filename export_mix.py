"""CPU-only final stem mix; input stems must already exist."""
import argparse
from pathlib import Path
import subprocess
from audio_io import read_audio
from voice_autopilot import resolve_ffmpeg_bin

def export_mix(vocals, instrumental, output, presence_db=8.0, vocal_gain_db=0):
    from tempfile import TemporaryDirectory
    from audio_io import write_wav
    vocals, instrumental, output = map(Path, (vocals, instrumental, output))
    import math,os
    if output.suffix.lower() not in ('.mp3','.wav') or not math.isfinite(presence_db) or not 0 <= presence_db <= 12:
        raise ValueError("Final export must be MP3 or PCM24 WAV; presence gain must be 0–12 dB")
    if not math.isfinite(vocal_gain_db) or not -12 <= vocal_gain_db <= 12:
        raise ValueError('Vocal mix gain must be finite and within -12 to +12 dB')
    if any(output.resolve() == p.resolve() or (output.exists() and p.exists() and os.path.samefile(output,p)) for p in (vocals,instrumental)):
        raise ValueError("Output cannot overwrite an input")
    ffmpeg = resolve_ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("FFmpeg is unavailable")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="voice-mix-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        for source, name in ((vocals, "vocals.wav"), (instrumental, "instrumental.wav")):
            audio, sr = read_audio(source, mono=False)
            write_wav(temporary / name, audio, sr)
        filters = (f"[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                   f"volume={vocal_gain_db}dB,equalizer=f=5656.854:t=o:w=1:g={presence_db}[v];"
                   "[1:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[i];"
                   "[v][i]amix=inputs=2:duration=longest:normalize=0,"
                   "alimiter=limit=0.89:level=0:latency=1[out]")
        rendered = temporary / ('final'+output.suffix.lower())
        codec=['-c:a','libmp3lame','-b:a','320k'] if output.suffix.lower() == '.mp3' else ['-c:a','pcm_s24le']
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(temporary / "vocals.wav"),
                        "-i", str(temporary / "instrumental.wav"), "-filter_complex", filters,
                        "-map", "[out]", "-ar", "48000", *codec,
                        str(rendered)], check=True, timeout=1800)
        rendered.replace(output)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocals", required=True)
    parser.add_argument("--instrumental", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--presence-db", type=float, default=8.0)
    args = parser.parse_args()
    print(export_mix(args.vocals, args.instrumental, args.output, args.presence_db))
