"""Fail-fast environment validation without loading model weights."""
import importlib
import subprocess
import sys

def check():
    failures = []
    if sys.version_info[:2] != (3, 10):
        failures.append("Python 3.10 is required")
    for module in ("torch", "torchaudio", "numpy", "scipy", "soundfile", "gradio", "bark", "fairseq", "pyworld", "torchcrepe", "librosa", "faiss"):
        try:
            imported = importlib.import_module(module)
            print(f"{module}: {getattr(imported, '__version__', 'import OK')}")
        except Exception as exc:
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    try:
        import torch, torchaudio
        if torch.__version__ != "2.1.0+cu121" or torchaudio.__version__ != "2.1.0+cu121":
            failures.append("Require torch and torchaudio 2.1.0+cu121")
        if not torch.cuda.is_available():
            failures.append("CUDA is unavailable")
        else:
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as exc:
        failures.append(str(exc))
    from voice_autopilot import resolve_ffmpeg_bin
    ffmpeg = resolve_ffmpeg_bin()
    if not ffmpeg:
        failures.append("FFmpeg is unavailable")
    else:
        subprocess.run([ffmpeg, "-version"], check=True, stdout=subprocess.DEVNULL)
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if result.returncode:
        failures.append(result.stdout.strip())
    if failures:
        raise RuntimeError("Environment is not ready: " + "; ".join(failures))
    print("Runtime checks passed. Dataset and model quality need separate validation.")

if __name__ == "__main__":
    check()
