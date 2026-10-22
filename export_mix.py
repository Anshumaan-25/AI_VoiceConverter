"""CPU-only final stem mix; input stems must already exist."""
import argparse
from pathlib import Path
import subprocess
from audio_io import read_audio
from voice_autopilot import resolve_ffmpeg_bin

def export_mix(vocals, instrumental, output, presence_db=8.0):
    from tempfile import TemporaryDirectory
    from audio_io import write_wav
    vocals, instrumental, output = map(Path, (vocals, instrumental, output))
    if output.suffix.lower() != ".mp3" or not 0 <= presence_db <= 12:
        raise ValueError("Final export must be MP3; presence gain must be 0–12 dB")
    if output.resolve() in (vocals.resolve(), instrumental.resolve()):
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
        filters = (f"[0:a]equalizer=f=5656.854:t=o:w=1:g={presence_db}[v];"
                   "[v][1:a]amix=inputs=2:duration=longest:normalize=0,"
                   "alimiter=limit=0.89:level=0:latency=1[out]")
        rendered = temporary / "final.mp3"
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(temporary / "vocals.wav"),
                        "-i", str(temporary / "instrumental.wav"), "-filter_complex", filters,
                        "-map", "[out]", "-ar", "48000", "-c:a", "libmp3lame", "-b:a", "320k",
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
