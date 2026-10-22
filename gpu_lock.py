"""Cross-process serialization of GPU jobs on this Windows workstation."""
from contextlib import contextmanager
from pathlib import Path
import tempfile

@contextmanager
def gpu_job():
    import msvcrt
    path = Path(tempfile.gettempdir()) / "ai_voiceconverter_gpu.lock"
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another voice-processing job is using the GPU") from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
